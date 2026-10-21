"""
Hugging Face Space: play against Chess GPT, a transformer trained from scratch on Lichess games.

The model (30M or 16M) sees the current board from the side to move + the last 8 moves and gives a probability
for each of the 1,968 possible moves. By default it plays its most likely legal move ("Strongest"); "Varied" picks a
legal move at random, weighted by those probabilities, so games differ. It is always told that it wins; the ratings
are the user's choice, defaulting to 2400 / 2400 as in the Elo test (the strength reported on the model card).
model.py downloads the weights from the model repo and does the encoding; everything below works with normal moves.
"""
import chess
import chess.svg
import gradio as gr
import torch

from model import MODEL_SIZES, REPO_ID, ChessModel

try:
    import spaces  # available on Hugging Face ZeroGPU Spaces
    gpu = spaces.GPU
except ImportError:  # running locally
    gpu = lambda fn: fn


# ZeroGPU Spaces require at least one @spaces.GPU function to start. The model runs on CPU (one fast forward pass
# per move), so this one is never called: moves don't use the Space's GPU quota or wait for a GPU.
@gpu
def _zerogpu_placeholder():
    pass

TOP_K = 5             # moves shown in the log
ELOS = [str(e) for e in range(800, 2401, 200)]   # rating buckets used in training: <elo_800> ... <elo_2400>
DEFAULT_ELO = "2400"  # both ratings default to 2400, as in the Elo test
STRONGEST = "Strongest (always its top move)"
VARIED = "Varied (random, weighted by its probabilities)"

# ---------------------------------------------------------------- model loading
MODELS = {size: ChessModel(size) for size in MODEL_SIZES}
for m in MODELS.values():
    print(f"Loaded {m.size}: {m.n_params:,} params, step {m.step}", flush=True)


# ---------------------------------------------------------------- inference
def model_move(st):
    """The model's move, plus what the log shows: its probability, the probability the model put on illegal moves,
    and its top moves (legal or not).
    STRONGEST: the most likely legal move (same position -> same move, so the same opening every game).
    VARIED: a legal move drawn at random with the model's probabilities as weights, so a move it gives 40% is
            played about 40% of the time: different games, but still moves the model believes in."""
    board, net = st["board"], MODELS[st["model"]]
    legal = {m.uci(): m for m in board.legal_moves}

    # Extra tokens, written like the training data (from White's side, White's rating first):
    # result is always "model wins"; the ratings are the user's choice (default 2400 / 2400, as in the Elo test).
    model_white = st["human"] == chess.BLACK
    result = "<white_wins>" if model_white else "<black_wins>"
    model_elo, human_elo = f"<elo_{st['model_elo']}>", f"<elo_{st['human_elo']}>"
    white_elo, black_elo = (model_elo, human_elo) if model_white else (human_elo, model_elo)
    probs = net.move_probs(board, result, white_elo, black_elo)   # (1968,), sums to 1

    legal_ids = torch.tensor([net.move_to_id[u] for u in legal], dtype=torch.long)
    legal_probs = probs[legal_ids]
    if st["choice"] == VARIED:
        best = legal_ids[torch.multinomial(legal_probs, 1)].item()   # multinomial renormalises the legal probs itself
    else:
        best = legal_ids[legal_probs.argmax()].item()                # top legal move: the model's strongest setting

    top_p, top_i = probs.topk(TOP_K)
    info = {
        "p": probs[best].item(),
        "illegal_mass": 1 - legal_probs.sum().item(),             # how much the model believed in impossible moves
        "top": [(net.moves[i], p, net.moves[i] in legal) for p, i in zip(top_p.tolist(), top_i.tolist())],
    }
    return legal[net.moves[best]], info


# ---------------------------------------------------------------- game logic
def fresh_state(human_color, model_size, choice, model_elo, human_elo):
    return {
        "board": chess.Board(),
        "human": human_color,
        "model": model_size,
        "choice": choice,
        "model_elo": model_elo,
        "human_elo": human_elo,
        "top1_legal": 0,        # model's single most likely move was legal
        "illegal_mass": [],     # per move: probability the model put on illegal moves
        "selected": None,       # square picked by the first click on the board
        "log": [],
        "note": "",
    }


def play_model_turn(st):
    board = st["board"]
    move_label = f"{board.fullmove_number}{'.' if board.turn == chess.WHITE else '...'}"
    move, info = model_move(st)
    san = board.san(move)    # SAN must be computed before the move is played
    board.push(move)

    st["illegal_mass"].append(info["illegal_mass"])
    st["top1_legal"] += info["top"][0][2]
    st["note"] = f"Model played **{san}** (it gave it {info['p']:.1%})."
    st["log"].append(" | ".join([
        f"{move_label} {san:<7} {st['model']}, {'varied' if st['choice'] == VARIED else 'strongest'}",
        f"p={info['p']:.1%}",
        f"on illegal moves {info['illegal_mass']:.1%}",
        "top: " + ", ".join(f"{t} {p:.0%}{'' if ok else ' (illegal)'}" for t, p, ok in info["top"]),
    ]))


def square_click(evt: gr.EventData, st):
    """Click once on your piece to select it, then on a highlighted square to move."""
    board = st["board"]
    if board.is_game_over() or board.turn != st["human"]:
        return render(st)
    sq = chess.parse_square(evt.square)
    sel = st["selected"]

    if sel is not None and sq != sel:
        # A pawn reaching the last rank always becomes a queen.
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
    )   # no size: only a viewBox, so the board scales to its container
    # Sizing styles are inline (not only in css_template) so the board fills the column on every screen, phones too.
    svg = svg.replace("<svg ", '<svg style="display:block;width:100%;height:auto" ', 1)
    ranks = range(7, -1, -1) if st["human"] == chess.WHITE else range(8)
    files = range(8) if st["human"] == chess.WHITE else range(7, -1, -1)
    cells = "".join(
        f'<div class="sq" data-sq="{chess.square_name(chess.square(f, r))}" style="cursor:pointer"></div>'
        for r in ranks for f in files
    )
    # chess.svg draws a 15px coordinate margin around a 360px board (390 total): the grid is inset by 15/390 = 3.846%
    grid_style = ("position:absolute;inset:3.846%;display:grid;"
                  "grid-template-columns:repeat(8,1fr);grid-template-rows:repeat(8,1fr)")
    wrap_style = "position:relative;width:100%;max-width:480px;margin:0 auto"
    return (f'<div class="board-wrap" style="{wrap_style}">{svg}'
            f'<div class="grid" style="{grid_style}">{cells}</div></div>')


def render(st):
    board = st["board"]
    your_turn = not board.is_game_over() and board.turn == st["human"]
    pgn =chess.Board().variation_san(board.move_stack) if board.move_stack else ""

    if board.is_game_over():
        reason = board.outcome().termination.name.replace("_", " ").lower()
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
            f"average probability on illegal moves {avg:.1%}"
        )

    log = "\n".join(reversed(st["log"]))   # newest first
    return st, board_html_for(st), pgn, status, log


def new_game(color_choice, model_size, choice, model_elo, human_elo):
    human = chess.WHITE if color_choice == "White" else chess.BLACK
    st = fresh_state(human, model_size, choice, model_elo, human_elo)
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
# Layout is inline in board_html_for(); only the hover effect needs CSS (hover can't be written inline).
BOARD_CSS = """
.sq:hover { box-shadow: inset 0 0 0 3px #1e90ffaa; }
"""

# One listener on the container survives every re-render of the board inside it.
BOARD_JS = """
element.addEventListener('click', (e) => {
    const cell = e.target.closest('[data-sq]');
    if (cell) trigger('click', {square: cell.dataset.sq});
});
"""

DESCRIPTION = f"""
# ♟️ Chess GPT

A transformer trained **from scratch** on 3.0M human games from Lichess to predict the move a human would play.
It was never told the rules of chess and does **no search**: every move is one forward pass.

**What the model sees each turn** (81 tokens): the current board from the side to move (64 squares, castling
rights, en passant) and the **last 8 moves**. It doesn't see the rest of the game, so it can't plan around
threefold repetition. It is always told that it wins the game.

**Move choice**: *Strongest* always plays the model's most likely legal move (its strongest setting, used to
measure its Elo), so it opens the same way every game. *Varied* picks a legal move at random, weighted by the
model's probabilities: a move it rates 40% is played about 40% of the time, so every game is different.

**Ratings** are given to the model like in training: it imitates a game between players of those ratings.
The default (both 2400) is its strongest setting and the one used to measure its Elo; lower ratings make it play
more like a weaker human. Changes apply from the model's next move.

**Results** (200K held-out positions, and 600 games against Stockfish 18 with no search):

| Model | Human move-match top-1 / top-5 | ≈ Elo vs Stockfish |
|---|---|---|
| 30M | 50.8% / 87.7% | 1558 |
| 16M | 49.9% / 86.9% | 1492 |

Elo is on Stockfish's `UCI_Elo` scale (0.1 s/move), not a Lichess or FIDE rating.
Weights and details: [{REPO_ID}](https://huggingface.co/{REPO_ID}).

**How to play**: click a piece, then a highlighted square.
"""

with gr.Blocks(title="Chess GPT") as demo:
    gr.Markdown(DESCRIPTION)
    state = gr.State()
    with gr.Row():
        # Own column with a minimum width: on a phone the controls wrap below instead of squeezing the board.
        with gr.Column(scale=1, min_width=300):
            board_html = gr.HTML(css_template=BOARD_CSS, js_on_load=BOARD_JS)
        with gr.Column(scale=1, min_width=300):
            with gr.Row():
                color = gr.Radio(["White", "Black"], value="White", label="Play as")
                model_size = gr.Radio(MODEL_SIZES, value=MODEL_SIZES[0], label="Model")
            choice = gr.Radio([STRONGEST, VARIED], value=STRONGEST, label="Move choice (can change mid-game)")
            with gr.Row():
                model_elo = gr.Dropdown(ELOS, value=DEFAULT_ELO, label="Model rating")
                human_elo = gr.Dropdown(ELOS, value=DEFAULT_ELO, label="Your rating")
            new_btn = gr.Button("New game", variant="primary")
            status = gr.Markdown()
            pgn = gr.Textbox(label="Game so far", lines=4, interactive=False)
    log = gr.Textbox(
        label="Model log (newest first): p = model's probability for the played move, top = its 5 most likely moves",
        lines=8, interactive=False,
    )

    def setter(key):
        def set_value(value, st):
            if st:
                st[key] = value
            return st
        return set_value

    outputs = [state, board_html, pgn, status, log]
    settings = [color, model_size, choice, model_elo, human_elo]
    new_btn.click(new_game, inputs=settings, outputs=outputs)
    board_html.click(square_click, inputs=state, outputs=outputs)
    for key, comp in [("model", model_size), ("choice", choice), ("model_elo", model_elo), ("human_elo", human_elo)]:
        comp.change(setter(key), inputs=[comp, state], outputs=state)
    demo.load(new_game, inputs=settings, outputs=outputs)

if __name__ == "__main__":
    demo.launch()
