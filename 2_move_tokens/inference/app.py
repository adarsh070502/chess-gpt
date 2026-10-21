import atexit
import os
import random
import shutil
import threading
from pathlib import Path

import chess
import chess.engine
import chess.svg
import gradio as gr
import torch
import torch.nn.functional as F

from model import CKPT_PATH, Transormer, block_size, ckpt, device

HERE = Path(__file__).parent
MODES = [
    "Raw sample (may be illegal)",
    "Sample from legal moves",
    "Top legal move",
    "Stockfish picks from model's top moves",
]
RESULTS = ["Model wins", "Draw", "Model loses"]
ELOS = [str(e) for e in range(800, 2401, 200)]   # rating buckets used in training: <elo_800> ... <elo_2400>
TEMPERATURE = 1.0     # 1.0 = the model's own probabilities; lower = more greedy
MAX_RETRIES = 10      # raw mode: illegal samples allowed before a random legal move is played
TOP_K = 5             # moves shown in the log, and candidates handed to Stockfish
STOCKFISH_TIME = 0.1  # seconds Stockfish spends ranking the candidates

# ---------------------------------------------------------------- model loading
stoi, itos = ckpt["stoi"], ckpt["itos"]

model = Transormer(ckpt["config"]["n_emb"])
model.load_state_dict(ckpt["model_state_dict"])
model.to(device).eval()
print(f"Loaded {CKPT_PATH} (step {ckpt.get('step', '?')})", flush=True)


# ---------------------------------------------------------------- inference
def prompt_tokens(st):
    """The game so far, written exactly like a line of games_uci.txt (without <end>).

    <start> <result> <white elo> <black elo> e2e4 e7e5 ...
    The result and ratings are chosen by the user; "Model wins" means the side the model plays.
    """
    model_white = st["human"] == chess.BLACK
    model_wins, model_loses = ("<white_wins>", "<black_wins>") if model_white else ("<black_wins>", "<white_wins>")
    result = {"Model wins": model_wins, "Draw": "<draw>", "Model loses": model_loses}[st["result"]]
    model_elo, human_elo = f"<elo_{st['model_elo']}>", f"<elo_{st['human_elo']}>"
    white_elo, black_elo = (model_elo, human_elo) if model_white else (human_elo, model_elo)
    return ["<start>", result, white_elo, black_elo] + [m.uci() for m in st["board"].move_stack]


@torch.no_grad()
def next_token_probs(tokens):
    """Probability of every vocab token being the next one."""
    idx = torch.tensor([[stoi[t] for t in tokens]], dtype=torch.long, device=device)
    logits, _ = model(idx)
    return F.softmax(logits[0, -1] / TEMPERATURE, dim=-1).cpu()


# ---------------------------------------------------------------- stockfish
def find_stockfish():
    candidates = [
        os.environ.get("STOCKFISH_PATH"),
        HERE.parent.parent / "stockfish" / "stockfish",       # shared chess_gpt/stockfish/ (Linux build)
        HERE.parent.parent / "stockfish" / "stockfish.exe",   # shared chess_gpt/stockfish/ (Windows build)
        shutil.which("stockfish"),
        "/usr/games/stockfish",                 # Hugging Face Space: packages.txt -> stockfish
    ]
    for c in candidates:
        if c and Path(c).is_file():
            return str(Path(c).resolve())
    return None


_engine = None
_engine_lock = threading.Lock()   # one engine shared by all users; analyse one position at a time


def stockfish_rank(board: chess.Board, moves):
    """Returns [(move, score_in_centipawns_for_side_to_move), ...] best first, or None if no engine."""
    global _engine
    with _engine_lock:
        if _engine is None:
            path = find_stockfish()
            if path is None:
                return None
            _engine = chess.engine.SimpleEngine.popen_uci(path)
            atexit.register(_engine.quit)
        infos = _engine.analyse(
            board, chess.engine.Limit(time=STOCKFISH_TIME), root_moves=moves, multipv=len(moves)
        )
    return [(i["pv"][0], i["score"].pov(board.turn).score(mate_score=100_000)) for i in infos]


# ---------------------------------------------------------------- choosing a move
def model_move(st):
    """Returns (move, source, info).

    source: "MODEL" (model's choice), "MODEL+SF" (Stockfish chose among the model's top
            legal moves) or "RANDOM" (model couldn't be used / gave no legal move).
    info:   dict with what the log shows: prob of the chosen move, probability the model
            put on illegal moves, its top moves, raw-mode attempts, Stockfish ranking.
    """
    board, mode = st["board"], st["mode"]
    legal = {m.uci(): m for m in board.legal_moves}
    info = {}

    tokens = prompt_tokens(st)
    unknown = [t for t in tokens if t not in stoi]
    if unknown or len(tokens) > block_size:
        info["why"] = f"move {unknown[0]} not in vocab" if unknown else f"game longer than {block_size} tokens"
        return random.choice(list(legal.values())), "RANDOM", info

    probs = next_token_probs(tokens)
    legal_ids = torch.tensor([stoi[u] for u in legal if u in stoi], dtype=torch.long)
    legal_probs = probs[legal_ids]
    info["illegal_mass"] = 1 - legal_probs.sum().item()
    top_p, top_i = probs.topk(TOP_K)
    info["top"] = [(itos[i], p, itos[i] in legal) for p, i in zip(top_p.tolist(), top_i.tolist())]

    def pick(token_id):
        info["p"] = probs[token_id].item()
        return legal[itos[token_id]]

    if mode == MODES[0]:   # raw: sample from everything, retry if illegal
        info["attempts"] = []
        for _ in range(MAX_RETRIES):
            tok = itos[torch.multinomial(probs, 1).item()]
            info["attempts"].append(tok)
            if tok in legal:
                return pick(stoi[tok]), "MODEL", info
        return random.choice(list(legal.values())), "RANDOM", info

    if len(legal_ids) == 0:    # no legal move is in the vocab (practically never)
        return random.choice(list(legal.values())), "RANDOM", info

    if mode == MODES[1]:   # sample, but only among legal moves
        return pick(legal_ids[torch.multinomial(legal_probs, 1)].item()), "MODEL", info

    order = legal_ids[legal_probs.argsort(descending=True)]
    if mode == MODES[3]:   # Stockfish chooses among the model's top legal moves
        cands = [legal[itos[i]] for i in order[:TOP_K].tolist()]
        ranked = stockfish_rank(board, cands)
        if ranked is not None:
            info["sf"] = [(m.uci(), cp) for m, cp in ranked]
            return pick(stoi[ranked[0][0].uci()]), "MODEL+SF", info
        info["why"] = "Stockfish not found, used top legal move"
    return pick(order[0].item()), "MODEL", info   # top legal move


# ---------------------------------------------------------------- game logic
def fresh_state(human_color, mode=MODES[1], result=RESULTS[0], model_elo="2000", human_elo="2000"):
    return {
        "board": chess.Board(),
        "human": human_color,
        "mode": mode,
        "result": result,
        "model_elo": model_elo,
        "human_elo": human_elo,
        "model_moves": 0,
        "fallbacks": 0,
        "retries": 0,           # raw mode: illegal samples this game
        "top1_legal": 0,        # model's single most likely token was a legal move
        "illegal_mass": [],     # per move: probability the model put on illegal moves
        "selected": None,       # square picked by the first click on the board
        "log": [],
        "note": "",
    }


def play_model_turn(st):
    board = st["board"]
    move_label = f"{board.fullmove_number}{'.' if board.turn == chess.WHITE else '...'}"
    move, source, info = model_move(st)
    san = board.san(move)
    board.push(move)

    parts = [f"{move_label} {san:<7} {source:<8}", st["mode"]]
    if "p" in info:
        parts.append(f"p={info['p']:.1%}")
    if "illegal_mass" in info:
        st["illegal_mass"].append(info["illegal_mass"])
        st["top1_legal"] += info["top"][0][2]
        parts.append(f"on illegal moves {info['illegal_mass']:.1%}")
        parts.append("top: " + ", ".join(f"{t} {p:.0%}{'' if ok else ' (illegal)'}" for t, p, ok in info["top"]))
    if "attempts" in info:
        failed = len(info["attempts"]) - (source == "MODEL")   # the last one succeeded unless RANDOM
        st["retries"] += failed
        parts.append(f"samples {len(info['attempts'])}, failed {failed}: " + " ".join(info["attempts"]))
    if "sf" in info:
        parts.append("Stockfish: " + ", ".join(f"{u} {cp:+d}" for u, cp in info["sf"]))
    if "why" in info:
        parts.append(info["why"])

    if source == "RANDOM":
        st["fallbacks"] += 1
        st["note"] = f"Played a random legal move **{san}** ({info.get('why', 'no legal sample')})."
    else:
        st["model_moves"] += 1
        who = "Stockfish picked" if source == "MODEL+SF" else "Model played"
        st["note"] = f"{who} **{san}** (model gave it {info['p']:.1%})."

    line = " | ".join(parts)
    st["log"].append(line)
    print(line, flush=True)


def square_click(evt: gr.EventData, st):
    """Click once on your piece to select it, then on a highlighted square to move."""
    board = st["board"]
    if board.is_game_over() or board.turn != st["human"]:
        return render(st)
    sq = chess.parse_square(evt.square)
    sel = st["selected"]

    if sel is not None and sq != sel:
        # Promotions from a click always queen; use the dropdown to under-promote.
        candidates = [m for m in board.legal_moves if m.from_square == sel and m.to_square == sq]
        if candidates:
            move = next((m for m in candidates if m.promotion in (None, chess.QUEEN)), candidates[0])
            st["selected"] = None
            return play(board.san(move), st)

    piece = board.piece_at(sq)
    if sq != sel and piece and piece.color == st["human"]:
        st["selected"] = sq
    else:
        st["selected"] = None
    return render(st)


def board_html_for(st):
    """The SVG board with a transparent 8x8 grid on top; each cell carries its square name."""
    board, sel = st["board"], st["selected"]
    fill = {}
    if sel is not None:
        fill[sel] = "#f6c90e99"
        for m in board.legal_moves:
            if m.from_square == sel:
                fill[m.to_square] = "#2e8b5766"
    svg = chess.svg.board(
        board,
        orientation=st["human"],
        lastmove=board.peek() if board.move_stack else None,
        fill=fill,
        size=480,
    )
    # Rows top to bottom, as seen from the human's side of the board.
    ranks = range(7, -1, -1) if st["human"] == chess.WHITE else range(8)
    files = range(8) if st["human"] == chess.WHITE else range(7, -1, -1)
    cells = "".join(
        f'<div class="sq" data-sq="{chess.square_name(chess.square(f, r))}"></div>'
        for r in ranks for f in files
    )
    return f'<div class="board-wrap">{svg}<div class="grid">{cells}</div></div>'


def render(st):
    board = st["board"]
    board_html = board_html_for(st)

    your_turn = not board.is_game_over() and board.turn == st["human"]
    legal = sorted(board.san(m) for m in board.legal_moves) if your_turn else []

    pgn = chess.Board().variation_san(board.move_stack) if board.move_stack else ""

    if board.is_game_over():
        outcome = board.outcome()
        reason = outcome.termination.name.replace("_", " ").lower()
        status = f"Game over: **{board.result()}** ({reason})."
    else:
        status = "Your move." if your_turn else "Model is thinking..."
    if st["note"]:
        status = st["note"] + "\n\n" + status

    n = len(st["illegal_mass"])
    if n:
        avg = sum(st["illegal_mass"]) / n
        status += (
            f"\n\nThis game: model's most likely move was legal {st['top1_legal']}/{n} times, "
            f"average probability on illegal moves {avg:.1%}, random fallbacks {st['fallbacks']}"
            + (f", failed raw samples {st['retries']}" if st["retries"] else "")
        )

    log = "\n".join(reversed(st["log"]))   # newest first
    return st, board_html, gr.update(choices=legal, value=None), pgn, status, log


def new_game(color_choice, mode, result, model_elo, human_elo):
    human = chess.WHITE if color_choice == "White" else chess.BLACK
    st = fresh_state(human, mode, result, model_elo, human_elo)
    if human == chess.BLACK:
        play_model_turn(st)
    return render(st)


def play(move_san, st):
    board = st["board"]
    if not move_san or board.is_game_over() or board.turn != st["human"]:
        return render(st)
    board.push_san(move_san)
    st["selected"] = None
    st["note"] = ""
    if not board.is_game_over():
        play_model_turn(st)
    return render(st)


# ---------------------------------------------------------------- UI
BOARD_CSS = """
.board-wrap { position: relative; max-width: 480px; margin: auto; }
.board-wrap svg { display: block; width: 100%; height: auto; }
/* chess.svg draws a 15px coordinate margin around a 360px board (390 total) */
.grid { position: absolute; inset: 3.846%; display: grid;
        grid-template: repeat(8, 1fr) / repeat(8, 1fr); }
.sq { cursor: pointer; }
.sq:hover { box-shadow: inset 0 0 0 3px #1e90ffaa; }
"""

# One listener on the container survives every re-render of the board inside it.
BOARD_JS = """
element.addEventListener('click', (e) => {
    const cell = e.target.closest('[data-sq]');
    if (cell) trigger('click', {square: cell.dataset.sq});
});
"""

with gr.Blocks(title="Chess GPT (move tokens)") as demo:
    gr.Markdown(
        "# Chess GPT (move tokens)\n"
        "A small transformer trained from scratch on Lichess games, one token per move (UCI). "
        "It was never told the rules of chess.\n\n"
        "Click a piece, then click a highlighted square to move (or pick from the dropdown).\n\n"
        "**Model mode**: *Raw sample* lets the model pick from all moves it knows, legal or not "
        f"(after {MAX_RETRIES} illegal tries a random legal move is played). The other modes only "
        "consider legal moves: *Sample* picks by the model's probabilities, *Top* takes its favourite, "
        f"and *Stockfish* picks the best of the model's top {TOP_K}.\n\n"
        "**Asked result / ratings** are put at the start of the game, like in training: the model "
        "imitates games with that result between players of those ratings.\n\n"
        f"**Context**: the model reads at most {block_size} tokens: 4 for start/result/ratings, then "
        f"one per half-move, so it can follow a game for {block_size - 4} half-moves "
        f"(**{(block_size - 4) // 2} full moves**). A typical game is 35-40 moves, so this covers "
        "virtually every game; past the limit a random legal move is played. Expect weaker play well "
        "before that, since very few training games went past move 70-80."
    )
    state = gr.State()
    with gr.Row():
        board_html = gr.HTML(css_template=BOARD_CSS, js_on_load=BOARD_JS)
        with gr.Column():
            color = gr.Radio(["White", "Black"], value="White", label="Play as")
            mode = gr.Dropdown(MODES, value=MODES[1], label="Model mode (can change mid-game)")
            with gr.Row():
                result = gr.Dropdown(RESULTS, value=RESULTS[0], label="Asked result")
                model_elo = gr.Dropdown(ELOS, value="2000", label="Model rating")
                human_elo = gr.Dropdown(ELOS, value="2000", label="Your rating")
            new_btn = gr.Button("New game")
            move_dd = gr.Dropdown(label="Your move (type to search)", choices=[], interactive=True)
            play_btn = gr.Button("Play move", variant="primary")
            status = gr.Markdown()
            pgn = gr.Textbox(label="Game so far", lines=4, interactive=False)
    log = gr.Textbox(
        label="Model log (newest first): p = model's probability for the played move, "
        "top = its 5 most likely tokens",
        lines=8, interactive=False,
    )

    def setter(key):
        def set_value(value, st):
            if st:
                st[key] = value
            return st
        return set_value

    outputs = [state, board_html, move_dd, pgn, status, log]
    settings = [color, mode, result, model_elo, human_elo]
    new_btn.click(new_game, inputs=settings, outputs=outputs)
    play_btn.click(play, inputs=[move_dd, state], outputs=outputs)
    board_html.click(square_click, inputs=state, outputs=outputs)
    for key, comp in [("mode", mode), ("result", result), ("model_elo", model_elo), ("human_elo", human_elo)]:
        comp.change(setter(key), inputs=[comp, state], outputs=state)
    demo.load(new_game, inputs=settings, outputs=outputs)

if __name__ == "__main__":
    demo.launch()
