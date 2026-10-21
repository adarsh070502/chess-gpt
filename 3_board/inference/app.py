"""
Play against the board-input Chess GPT (trained by chess_gpt/3_board/chess_gpt_board_embeddings.py).

Same app as 2_move_tokens/inference/app.py. The difference is what the model is given each turn:
    move model:  the whole game so far as move tokens  <start> <result> <elo> <elo> e2e4 e7e5 ...
    board model: only the current board, 73 tokens     <side> <castling x4> <ep> <result> <elo> <elo> + 64 squares
python-chess keeps the board here, so the model never has to work it out from the moves.
"""
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

from model import CKPT_PATH, Transormer, ckpt, device, encode_board, move_to_id, moves

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
# model.py has already picked and loaded the checkpoint (ckpt); build the network and put the weights in.
model = Transormer(ckpt["config"]["n_emb"])
model.load_state_dict(ckpt["model_state_dict"])
model.to(device).eval()     # eval(): no dropout at inference
print(f"Loaded {CKPT_PATH} (step {ckpt.get('step', '?')})", flush=True)


# ---------------------------------------------------------------- inference
def board_tokens(st):
    """The current position as the 73 token ids the model was trained on.

    The result and the ratings are chosen by the user; "Model wins" means the side the model plays.
    They go into the board tokens exactly like in training, so the model imitates that kind of game.
    """
    model_white = st["human"] == chess.BLACK
    # Training tokens are written from White's point of view (<white_wins> / <black_wins>),
    # so "Model wins" has to be turned into the right colour.
    model_wins, model_loses = ("<white_wins>", "<black_wins>") if model_white else ("<black_wins>", "<white_wins>")
    result = {"Model wins": model_wins, "Draw": "<draw>", "Model loses": model_loses}[st["result"]]
    # In training the first rating token is always White's, the second Black's.
    model_elo, human_elo = f"<elo_{st['model_elo']}>", f"<elo_{st['human_elo']}>"
    white_elo, black_elo = (model_elo, human_elo) if model_white else (human_elo, model_elo)
    return encode_board(st["board"], result, white_elo, black_elo)


@torch.no_grad()
def next_move_probs(st):
    """Probability of each of the 1,968 moves being the move to play now: one forward pass, (1, 73) -> (1968,)."""
    idx = torch.tensor([board_tokens(st)], dtype=torch.long, device=device)   # (1, 73): a batch of one board
    logits, _ = model(idx)                                                    # (1, 1968)
    # Temperature: dividing the logits by T < 1 sharpens the distribution (more greedy), T > 1 flattens it.
    return F.softmax(logits[0] / TEMPERATURE, dim=-1).cpu()


# ---------------------------------------------------------------- stockfish
def find_stockfish():
    candidates = [
        os.environ.get("STOCKFISH_PATH"),
        HERE.parent.parent / "stockfish" / "stockfish",       # shared chess_gpt/stockfish/ (Linux build)
        HERE.parent.parent / "stockfish" / "stockfish.exe",   # shared chess_gpt/stockfish/ (Windows build)
        shutil.which("stockfish"),
        "/usr/games/stockfish",                 # Debian/Ubuntu: apt install stockfish
    ]
    for c in candidates:
        if c and Path(c).is_file():
            return str(Path(c).resolve())
    return None


_engine = None
_engine_lock = threading.Lock()   # one engine shared by all users; analyse one position at a time


def stockfish_rank(board: chess.Board, candidate_moves):
    """Returns [(move, score_in_centipawns_for_side_to_move), ...] best first, or None if no engine."""
    global _engine
    with _engine_lock:
        if _engine is None:
            path = find_stockfish()
            if path is None:
                return None
            _engine = chess.engine.SimpleEngine.popen_uci(path)
            atexit.register(_engine.quit)
        # root_moves: only analyse these moves; multipv: return a score for each of them
        infos = _engine.analyse(
            board, chess.engine.Limit(time=STOCKFISH_TIME), root_moves=candidate_moves, multipv=len(candidate_moves)
        )
    return [(i["pv"][0], i["score"].pov(board.turn).score(mate_score=100_000)) for i in infos]


# ---------------------------------------------------------------- choosing a move
def model_move(st):
    """Returns (move, source, info).

    source: "MODEL" (model's choice), "MODEL+SF" (Stockfish chose among the model's top
            legal moves) or "RANDOM" (raw mode gave no legal move in MAX_RETRIES tries).
    info:   dict with what the log shows: prob of the chosen move, probability the model
            put on illegal moves, its top moves, raw-mode attempts, Stockfish ranking.
    """
    board, mode = st["board"], st["mode"]
    legal = {m.uci(): m for m in board.legal_moves}   # "e2e4" -> chess.Move, for the moves allowed right now
    info = {}

    probs = next_move_probs(st)                                     # (1968,), sums to 1
    # Every legal move is in the output vocab (it holds every possible UCI move), so no fallback is needed here.
    legal_ids = torch.tensor([move_to_id[u] for u in legal], dtype=torch.long)
    legal_probs = probs[legal_ids]                                  # the model's probability for each legal move
    # Everything not on a legal move: a measure of how well the model "sees" the position.
    info["illegal_mass"] = 1 - legal_probs.sum().item()
    top_p, top_i = probs.topk(TOP_K)                                # the model's favourites, legal or not
    info["top"] = [(moves[i], p, moves[i] in legal) for p, i in zip(top_p.tolist(), top_i.tolist())]

    def pick(move_id):
        info["p"] = probs[move_id].item()
        return legal[moves[move_id]]

    if mode == MODES[0]:   # raw: sample from all 1,968 moves, retry if illegal
        info["attempts"] = []
        for _ in range(MAX_RETRIES):
            mv = moves[torch.multinomial(probs, 1).item()]
            info["attempts"].append(mv)
            if mv in legal:
                return pick(move_to_id[mv]), "MODEL", info
        return random.choice(list(legal.values())), "RANDOM", info

    if mode == MODES[1]:   # sample, but only among legal moves (multinomial renormalises legal_probs itself)
        return pick(legal_ids[torch.multinomial(legal_probs, 1)].item()), "MODEL", info

    order = legal_ids[legal_probs.argsort(descending=True)]       # legal move ids, most likely first
    if mode == MODES[3]:   # Stockfish chooses among the model's top legal moves
        cands = [legal[moves[i]] for i in order[:TOP_K].tolist()]
        ranked = stockfish_rank(board, cands)
        if ranked is not None:
            info["sf"] = [(m.uci(), cp) for m, cp in ranked]
            return pick(move_to_id[ranked[0][0].uci()]), "MODEL+SF", info
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
        "top1_legal": 0,        # model's single most likely move was legal
        "illegal_mass": [],     # per move: probability the model put on illegal moves
        "selected": None,       # square picked by the first click on the board
        "log": [],
        "note": "",
    }


def play_model_turn(st):
    board = st["board"]
    move_label = f"{board.fullmove_number}{'.' if board.turn == chess.WHITE else '...'}"
    move, source, info = model_move(st)
    san = board.san(move)    # SAN ("Nf3") must be computed before the move is played
    board.push(move)

    # One log line per model move, same format as the move-token app so the two can be compared
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
        st["note"] = f"Played a random legal move **{san}** (no legal move in {MAX_RETRIES} raw samples)."
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

with gr.Blocks(title="Chess GPT (board input)") as demo:
    gr.Markdown(
        "# Chess GPT (board input)\n"
        "A small transformer trained from scratch on Lichess games. Each turn it sees only the "
        "**current board** (64 squares + whose turn, castling rights, en passant) and picks a move. "
        "It was never told the rules of chess.\n\n"
        "Click a piece, then click a highlighted square to move (or pick from the dropdown).\n\n"
        "**Model mode**: *Raw sample* lets the model pick from all 1,968 moves it knows, legal or not "
        f"(after {MAX_RETRIES} illegal tries a random legal move is played). The other modes only "
        "consider legal moves: *Sample* picks by the model's probabilities, *Top* takes its favourite, "
        f"and *Stockfish* picks the best of the model's top {TOP_K}.\n\n"
        "**Asked result / ratings** are given to the model with every board, like in training: it "
        "imitates games with that result between players of those ratings.\n\n"
        "**No move history**: the model doesn't know how the position arose, so it can't see "
        "repetitions, but there is also no limit on how long a game can be."
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
        "top = its 5 most likely moves",
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
    demo.launch(server_port=7861)   # 7861, so it can run next to the move-token app on 7860
