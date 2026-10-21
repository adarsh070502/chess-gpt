"""
Play against the flip + 8-move-history board model trained with the legal-move target
(chess_gpt/5_board_flip_hist8_legal/chess_gpt_board_fast_allpos_flip_hist8_legal_16M.py). Same app as 4_board_flip_hist8/inference/app.py.

Same app as 3_board/inference/app.py (old 73-token board model). The difference is what the model is given each turn:
    old board model: the current board, 73 tokens, from White's side
    this model:      the current board seen from the side to move + the last 8 moves, 81 tokens
model.move_probs() does the encoding and maps the answer back to real moves when the board was mirrored,
so everything below works with normal (unmirrored) moves.
"""
import random
from pathlib import Path

import chess
import chess.svg
import gradio as gr
import torch

from model import CKPT_PATH, ckpt, load_model, move_probs, move_to_id, moves

HERE = Path(__file__).parent
MODES = [
    "Raw sample (may be illegal)",
    "Sample from legal moves",
    "Top legal move",
]
RESULTS = ["Model wins", "Draw", "Model loses"]
ELOS = [str(e) for e in range(800, 2401, 200)]   # rating buckets used in training: <elo_800> ... <elo_2400>
TEMPERATURE = 1.0     # 1.0 = the model's own probabilities; lower = more greedy
MAX_RETRIES = 10      # raw mode: illegal samples allowed before a random legal move is played
TOP_K = 5             # moves shown in the log

# ---------------------------------------------------------------- model loading
# model.py has already picked and loaded the checkpoint (ckpt); build the network and put the weights in.
model = load_model()
print(f"Loaded {CKPT_PATH} (step {ckpt.get('step', '?')})", flush=True)


# ---------------------------------------------------------------- inference
def game_tokens(st):
    """Result and rating tokens for the model, written like the training data (from White's side).

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
    return result, white_elo, black_elo


@torch.no_grad()
def next_move_probs(st):
    """Probability of each of the 1,968 moves (real orientation) being the move to play now: one forward pass."""
    # Temperature: dividing the logits by T < 1 sharpens the distribution (more greedy), T > 1 flattens it.
    return move_probs(model, st["board"], *game_tokens(st), temperature=TEMPERATURE)


# ---------------------------------------------------------------- choosing a move
def model_move(st):
    """Returns (move, source, info).

    source: "MODEL" (model's choice) or "RANDOM" (raw mode gave no legal move in MAX_RETRIES tries).
    info:   dict with what the log shows: prob of the chosen move, probability the model
            put on illegal moves, its top moves, raw-mode attempts.
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

    return pick(legal_ids[legal_probs.argmax()].item()), "MODEL", info   # top legal move


# ---------------------------------------------------------------- game logic
def fresh_state(human_color, mode=MODES[2], result=RESULTS[0], model_elo="2000", human_elo="2000"):
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

    if source == "RANDOM":
        st["fallbacks"] += 1
        st["note"] = f"Played a random legal move **{san}** (no legal move in {MAX_RETRIES} raw samples)."
    else:
        st["model_moves"] += 1
        st["note"] = f"Model played **{san}** (model gave it {info['p']:.1%})."

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

with gr.Blocks(title="Chess GPT (board + history + legal)") as demo:
    gr.Markdown(
        "# Chess GPT (board + history + legal-move training)\n"
        "A small transformer trained from scratch on Lichess games. Each turn it sees the "
        "**current board from the side to move** (64 squares, castling rights, en passant) plus the "
        "**last 8 moves**, and picks a move. "
        "It was never told the rules of chess.\n\n"
        "Click a piece, then click a highlighted square to move (or pick from the dropdown).\n\n"
        "**Model mode**: *Raw sample* lets the model pick from all 1,968 moves it knows, legal or not "
        f"(after {MAX_RETRIES} illegal tries a random legal move is played). The other modes only "
        "consider legal moves: *Sample* picks by the model's probabilities, *Top* takes its favourite.\n\n"
        "**Asked result / ratings** are given to the model with every board, like in training: it "
        "imitates games with that result between players of those ratings.\n\n"
        "**Short memory**: the model sees only the last 8 moves, not the whole game, so it can't plan "
        "around threefold repetition; there is no limit on how long a game can be."
    )
    state = gr.State()
    with gr.Row():
        board_html = gr.HTML(css_template=BOARD_CSS, js_on_load=BOARD_JS)
        with gr.Column():
            color = gr.Radio(["White", "Black"], value="White", label="Play as")
            mode = gr.Dropdown(MODES, value=MODES[2], label="Model mode (can change mid-game)")   # default: top legal move = the model's real preference
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
    demo.launch(server_port=7863)   # 7863: 7860 move tokens, 7861 old board, 7862 flip + history
