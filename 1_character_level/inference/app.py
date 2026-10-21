import atexit
import os
import random
import shutil
import threading
from collections import Counter
from pathlib import Path

import chess
import chess.engine
import chess.svg
import gradio as gr
import torch
import torch.nn.functional as F

from model import Transormer, block_size, ckpt, device

HERE = Path(__file__).parent
MODES = ["Single sample", "Best of N: most common", "Best of N: Stockfish picks"]
N_CANDIDATES = 10     # samples per move in the "Best of N" modes
STOCKFISH_TIME = 0.1  # seconds Stockfish spends ranking the candidates
MAX_RETRIES = 10    # model attempts before falling back to a random legal move
TEMPERATURE = 0.1    # lower = more confident/greedy, higher = more varied
MAX_MOVE_CHARS = 10   # longest SAN move is ~7 chars, e.g. "exd8=Q+"

# ---------------------------------------------------------------- model loading
stoi, itos = ckpt["stoi"], ckpt["itos"]

model = Transormer(ckpt["config"]["n_emb"])
model.load_state_dict(ckpt["model_state_dict"])
model.to(device).eval()


# ---------------------------------------------------------------- inference
def game_prefix(board: chess.Board) -> str:
    """Write the game so far exactly like the training data, ready for the next move.

    Start:              ";1. "
    After 1. e4:        ";1. e4 "
    After 1. e4 e5:     ";1. e4 e5 2. "
    """
    text = ";"
    if board.move_stack:
        text += chess.Board().variation_san(board.move_stack)
    if board.turn == chess.WHITE:
        text += (" " if board.move_stack else "") + f"{board.fullmove_number}. "
    else:
        text += " "
    return text


@torch.no_grad()
def sample_move_text(prefix: str) -> str:
    """Generate characters until a space: that's one move in SAN."""
    idx = torch.tensor([[stoi[c] for c in prefix]], dtype=torch.long, device=device)
    out = ""
    for _ in range(MAX_MOVE_CHARS):
        logits, _ = model(idx[:, -block_size:])
        probs = F.softmax(logits[:, -1, :] , dim=-1) #/ TEMPERATURE
        nxt = torch.multinomial(probs, num_samples=1)
        ch = itos[nxt.item()]
        if ch in (" ", "\n", ";"):
            break
        out += ch
        idx = torch.cat([idx, nxt], dim=1)
    return out


def try_sample(board: chess.Board, prefix: str):
    """One model sample, checked against the rules. Returns (text, verdict, move or None)."""
    text = sample_move_text(prefix)
    try:
        return text, "legal", board.parse_san(text)   # raises ValueError if illegal/garbage
    except chess.IllegalMoveError:
        return text, "illegal", None      # well-formed SAN, not legal here
    except chess.AmbiguousMoveError:
        return text, "ambiguous", None    # e.g. "Nd2" when two knights can go
    except ValueError:
        return text, "invalid", None      # not SAN at all


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
def model_move(board: chess.Board, mode: str):
    """Returns (move, source, attempts, picked_note).

    source:   "MODEL" (model's own choice), "MODEL+SF" (Stockfish chose among the model's
              legal candidates) or "RANDOM" (model produced no legal move).
    attempts: list of (generated_text, verdict) in sampling order, e.g.
              [("Nf9", "invalid"), ("Qxe7", "illegal"), ("Nf6", "legal")]
    """
    prefix = game_prefix(board)
    attempts = []

    if mode == MODES[0]:   # single sample: first legal move wins
        for _ in range(MAX_RETRIES):
            text, verdict, move = try_sample(board, prefix)
            attempts.append((text, verdict))
            if move:
                return move, "MODEL", attempts, f"attempt {len(attempts)}/{MAX_RETRIES}"
        return random.choice(list(board.legal_moves)), "RANDOM", attempts, ""

    # best of N: sample N times, then pick among the distinct legal moves
    counts = Counter()
    for _ in range(N_CANDIDATES):
        text, verdict, move = try_sample(board, prefix)
        attempts.append((text, verdict))
        if move:
            counts[move] += 1
    if not counts:
        return random.choice(list(board.legal_moves)), "RANDOM", attempts, ""

    summary = lambda m: f"{board.san(m)} x{counts[m]}"
    if mode == MODES[2]:
        ranked = stockfish_rank(board, list(counts))
        if ranked is not None:
            cands = ", ".join(f"{summary(m)} ({cp:+d})" for m, cp in ranked)
            return ranked[0][0], "MODEL+SF", attempts, f"Stockfish ranked: {cands}"
        note = "Stockfish not found, used most common instead. "
    else:
        note = ""
    best = counts.most_common(1)[0][0]
    cands = ", ".join(summary(m) for m, _ in counts.most_common())
    return best, "MODEL", attempts, note + f"most common: {cands}"


# ---------------------------------------------------------------- game logic
def fresh_state(human_color, mode=MODES[0]):
    return {
        "board": chess.Board(),
        "human": human_color,
        "mode": mode,
        "model_moves": 0,
        "fallbacks": 0,
        "retries": 0,       # failed model attempts this game (includes the ones before a fallback)
        "selected": None,   # square picked by the first click on the board
        "log": [],
        "note": "",
    }


def play_model_turn(st):
    board = st["board"]
    move_label = f"{board.fullmove_number}{'.' if board.turn == chess.WHITE else '...'}"
    move, source, attempts, picked = model_move(board, st["mode"])
    san = board.san(move)
    board.push(move)

    failed = sum(1 for _, verdict in attempts if verdict != "legal")
    st["retries"] += failed
    # identical samples are grouped: 'Kxh7' -> illegal x3
    tries = ", ".join(
        f"{text or '<empty>'!r} -> {verdict}" + (f" x{n}" if n > 1 else "")
        for (text, verdict), n in Counter(attempts).items()
    )
    if source == "RANDOM":
        st["fallbacks"] += 1
        st["note"] = f"Model gave no legal move in {len(attempts)} tries, played random legal move **{san}**."
    else:
        st["model_moves"] += 1
        who = "Stockfish picked" if source == "MODEL+SF" else "Model played"
        st["note"] = f"{who} **{san}** ({picked})."

    line = (
        f"{move_label} {san:<7} {source:<8} | {st['mode']} | samples {len(attempts)}, failed {failed}"
        + (f" | {picked}" if picked else "")
        + f" | samples: {tries}"
    )
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

    total = st["model_moves"] + st["fallbacks"]
    if total:
        rate = 100 * st["model_moves"] / total
        status += (
            f"\n\nModel's own legal moves this game: {st['model_moves']}/{total} ({rate:.0f}%), "
            f"random fallbacks: {st['fallbacks']}, failed samples (retries): {st['retries']}"
        )

    log = "\n".join(reversed(st["log"]))   # newest first
    return st, board_html, gr.update(choices=legal, value=None), pgn, status, log


def new_game(color_choice, mode):
    human = chess.WHITE if color_choice == "White" else chess.BLACK
    st = fresh_state(human, mode)
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

with gr.Blocks(title="Chess GPT") as demo:
    gr.Markdown(
        "# Chess GPT\n"
        "A small transformer trained from scratch on Lichess games, using only the move text. "
        "It was never told the rules of chess. If it suggests an illegal move "
        f"{MAX_RETRIES} times in a row, a random legal move is played instead.\n\n"
        "Click a piece, then click a highlighted square to move (or pick from the dropdown).\n\n"
        f"**Model mode**: *Single sample* plays the model's first legal move. *Best of N* samples "
        f"{N_CANDIDATES} moves and plays the most common legal one, or lets Stockfish pick the best "
        "of them (Stockfish only chooses among the model's ideas, it never adds its own)."
    )
    state = gr.State()
    with gr.Row():
        board_html = gr.HTML(css_template=BOARD_CSS, js_on_load=BOARD_JS)
        with gr.Column():
            color = gr.Radio(["White", "Black"], value="White", label="Play as")
            mode = gr.Dropdown(MODES, value=MODES[0], label="Model mode (can change mid-game)")
            new_btn = gr.Button("New game")
            move_dd = gr.Dropdown(label="Your move (type to search)", choices=[], interactive=True)
            play_btn = gr.Button("Play move", variant="primary")
            status = gr.Markdown()
            pgn = gr.Textbox(label="Game so far", lines=4, interactive=False)
    log = gr.Textbox(
        label="Model log (newest first): MODEL = model's own choice, "
        "MODEL+SF = Stockfish chose among model's moves, RANDOM = fallback",
        lines=8, interactive=False,
    )

    def set_mode(m, st):
        if st:
            st["mode"] = m
        return st

    outputs = [state, board_html, move_dd, pgn, status, log]
    new_btn.click(new_game, inputs=[color, mode], outputs=outputs)
    play_btn.click(play, inputs=[move_dd, state], outputs=outputs)
    board_html.click(square_click, inputs=state, outputs=outputs)
    mode.change(set_mode, inputs=[mode, state], outputs=state)
    demo.load(new_game, inputs=[color, mode], outputs=outputs)

if __name__ == "__main__":
    demo.launch()
