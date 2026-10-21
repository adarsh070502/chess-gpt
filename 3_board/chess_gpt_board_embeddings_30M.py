"""
30M version of chess_gpt_board_embeddings.py: identical except the model size, checkpoint names and
the learning-rate schedule.

Schedule (warmup -> constant -> decay, "WSD"), so the run can be stopped at any time:
    1. train_from_scratch(): warmup, then a constant lr. Stop with Ctrl+C whenever the GPU is needed;
       chess_board_30M_v2_latest.pt is saved every eval_interval steps.
    2. train_from_checkpoint("chess_board_30M_v2_latest.pt"): continue at the constant lr (repeat as needed).
    3. train_from_checkpoint("chess_board_30M_v2_latest.pt", decay=True): once, at the end. Lowers the lr
       to min_lr over ~15% of the steps done so far; this is where the loss drops, like the end of cosine.

Chess GPT with board input: the model sees the current board, not the move history.

x = the board before a move (73 tokens), y = the move a human played on it (one of 1,968 UCI moves).
python-chess plays the moves and keeps the board, so the model never has to track it.
Plan and reasoning: BOARD_INPUT_PLAN.md. The move-history model is chess_gpt_move_tokens.py.

One board = 73 tokens, always in this order:
    position 0       <white> / <black>                  side to move (the move is read from here)
    positions 1-4    <K>/<noK> <Q>/<noQ> <k>/<nok> <q>/<noq>   castling rights
    position 5       <ep_none> / <ep_a> ... <ep_h>       en passant file
    position 6       <white_wins> / <black_wins> / <draw>    result, as in the move model
    positions 7-8    <elo_...> <elo_...>                 White's, then Black's rating bucket
    positions 9-72   64 squares a8 b8 ... h8 a7 ... h1   '.' empty, PNBRQK White, pnbrqk Black

Differences from the move-history GPT:
    - no causal mask (every square sees every other square)
    - one prediction per board: logits (B, 1968) read from position 0, targets (B,)

Run from the repo root:  python chess_gpt/3_board/chess_gpt_board_embeddings_30M.py
The first run builds the dataset from games.bin, the move model's data (boards_*.bin / moves_*.bin);
later runs reuse it.
"""
import json
import math
import multiprocessing as mp
import os
import time
from pathlib import Path

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyper parameters
max_iters = 211000    # upper limit: one full pass over the 27.0M train positions at 128 per step (~32 h)
batch_size = 128    # boards per batch; 128 x 73 tokens ≈ the tokens per step of the move model
block_size = 73     # tokens per board, always the same (see the layout above)
eval_interval = 500
learning_rate = 2e-4   # constant lr after warmup (the first 30M run used a 3e-4 peak)
min_lr = 2e-5          # end of the final decay (10% of learning_rate)
warmup_iters = 1000    # linear ramp 0 -> learning_rate
decay_start = None     # set by train_from_checkpoint(..., decay=True): step where the final decay begins
decay_iters = None     # length of the final decay
n_emb = 512    # ~29M params (the 16M version: n_emb 396, 6 heads, 8 layers)
num_heads = 8
n_layer = 9 # no of multi attention and feedforward blocks
dropout = 0.0
eval_iters = 100 # used to decide on how many batches should the loss be calculatd on

device = 'cuda' if torch.cuda.is_available() else 'cpu' # Don't run on CPU if possible (it's slow. really.)

# Dataset settings
HERE = Path(__file__).parent         # folder of this script; its own data files (.bin) live here
DATA = HERE.parent / "data"          # shared data: games.bin, vocab.json (made by data/data_with_player_rating_start_token.ipynb)
CKPT_DIR = HERE / "inference"       # checkpoints are saved here, next to the inference code that loads them
POSITIONS_PER_GAME = 10     # random positions taken from each game (all of them would be ~27 GB)
MOVE_BLOCK_SIZE = 256       # the move model drops games longer than 257 tokens; drop the same ones
N_WORKERS = os.cpu_count() - 1       # 1 core reads games.bin and writes the output, the rest convert games


# ---------------------------------------------------------------- vocab
# Input vocab: every token that can appear in a board sequence (44 in total).
# Its id is just its index in this list, e.g. "." = 0, "P" = 1, ..., "<white>" = 13.
#
# 13 square tokens: what stands on a square.
#   "."          empty square
#   P N B R Q K  White pawn, knight, bishop, rook, queen, king (uppercase = White, as in FEN)
#   p n b r q k  the same for Black (lowercase = Black)
SQUARE_TOKENS = [".", "P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]
# Whose turn it is. The same board needs a completely different move depending on this.
SIDE_TOKENS = ["<white>", "<black>"]
# Castling rights, one yes/no pair per castling option:
#   K = White kingside (O-O), Q = White queenside (O-O-O), k / q = the same for Black.
# Rights can't be seen from the pieces: a king that moved and came back looks the same but can't castle.
CASTLE_TOKENS = ["<K>", "<noK>", "<Q>", "<noQ>", "<k>", "<nok>", "<q>", "<noq>"]
# En passant: the file (column) on which a pawn can be captured en passant right now, or none.
# Only the file is needed; the rank follows from whose turn it is.
EP_TOKENS = ["<ep_none>"] + [f"<ep_{f}>" for f in "abcdefgh"]
# Game result and rating buckets: the same conditioning tokens as the move model, so at play time
# we can ask for "play like a 2000 who wins". The dict keys are PGN results; only the values are tokens.
RESULT_TOKENS = {"1-0": "<white_wins>", "0-1": "<black_wins>", "1/2-1/2": "<draw>"}
ELO_TOKENS = [f"<elo_{b}>" for b in range(800, 2401, 200)]   # <elo_800>, <elo_1000>, ..., <elo_2400>

input_vocab = SQUARE_TOKENS + SIDE_TOKENS + CASTLE_TOKENS + EP_TOKENS + list(RESULT_TOKENS.values()) + ELO_TOKENS
stoi = {t: i for i, t in enumerate(input_vocab)}   # token -> id, e.g. "N" -> 2
itos = {i: t for t, i in stoi.items()}             # id -> token, e.g. 2 -> "N"
vocab_size = len(input_vocab)    # 44

# Output vocab: the 1,968 UCI moves, in the same order as the move model's vocab.json.
# vocab.json = 15 special tokens (<pad>, <start>, <elo_...> ...) + 1,968 moves; the moves are the tokens
# without "<". The model's last layer has one output per move, and the target y is the move's index here.
with open(DATA / "vocab.json") as f:
    moves = [t for t in json.load(f) if not t.startswith("<")]
move_to_id = {mv: i for i, mv in enumerate(moves)}  # "e2e4" -> its index in moves
n_moves = len(moves)             # 1968


def encode_board(board, result, white_elo_tok, black_elo_tok):
    """python-chess board -> 73 token ids (layout in the docstring at the top)."""
    # board.ep_square is set after every double pawn push, even when no enemy pawn can actually
    # capture en passant. Only mark it when the capture is really possible, so the token means something.
    ep = board.ep_square if board.has_legal_en_passant() else None

    # The first 9 tokens: game state that the 64 squares don't show.
    tokens = [
        "<white>" if board.turn == chess.WHITE else "<black>",                     # 0: side to move
        "<K>" if board.has_kingside_castling_rights(chess.WHITE) else "<noK>",     # 1: White O-O allowed?
        "<Q>" if board.has_queenside_castling_rights(chess.WHITE) else "<noQ>",    # 2: White O-O-O allowed?
        "<k>" if board.has_kingside_castling_rights(chess.BLACK) else "<nok>",     # 3: Black O-O allowed?
        "<q>" if board.has_queenside_castling_rights(chess.BLACK) else "<noq>",    # 4: Black O-O-O allowed?
        # 5: en passant file. square_file gives 0-7, FILE_NAMES turns it into "a".."h".
        "<ep_none>" if ep is None else f"<ep_{chess.FILE_NAMES[chess.square_file(ep)]}>",
        result, white_elo_tok, black_elo_tok,                                      # 6, 7, 8: same for every position of a game
    ]

    # Tokens 9-72: the 64 squares, read like a printed board: rank 8 left to right (a8 ... h8),
    # then rank 7, ..., down to rank 1 (a1 ... h1). Square a8 is always token 9, h1 always token 72,
    # so positional embedding 9 always means "square a8".
    # python-chess numbers ranks and files 0-7 (rank 0 = rank 1, file 0 = file a), hence range(7, -1, -1).
    for rank in range(7, -1, -1):            # rank 8 first, like FEN
        for file in range(8):
            piece = board.piece_at(chess.square(file, rank))   # a Piece, or None if the square is empty
            tokens.append(piece.symbol() if piece else ".")    # symbol(): "P" for a White pawn, "n" for a Black knight, ...

    # Strings -> ids. Always 9 + 64 = 73 ids.
    return [stoi[t] for t in tokens]


# ---------------------------------------------------------------- dataset (built once from games.bin)
# games.bin holds the move model's games, already filtered (no bullet, no timeouts, legal moves):
#     <start> <result> <elo_W> <elo_B> e2e4 e7e5 ... <end>
# All games are stored back to back as uint16 token ids, with ids from vocab.json.
# Here each game is replayed on a python-chess board, and at a few random moments we save
# (board before the move, move played) as one training example.
with open(DATA / "vocab.json") as f:
    game_vocab = json.load(f)       # token id -> token, for games.bin


def game_index():
    """games.bin plus (starts, lengths) of its games, exactly as chess_gpt_move_tokens.py selects them."""
    # memmap: use the 449 MB file like an array without loading it into RAM.
    games_data = np.memmap(DATA / "games.bin", dtype=np.uint16, mode="r")
    # Every game begins with <start>, so the positions of the <start> token are the game starts.
    starts = np.flatnonzero(games_data == game_vocab.index("<start>"))
    # A game's length = distance to the next game's start (the last game runs to the end of the file).
    lengths = np.diff(starts, append=len(games_data))
    # The move model drops games longer than block_size + 1 tokens before its 90/10 split.
    # Dropping the same 207 games keeps the split boundary at the same game, so both models get the same val games.
    keep = lengths <= MOVE_BLOCK_SIZE + 1
    return games_data, starts[keep], lengths[keep]


def convert_game(item):
    """(game_number, token ids of one game) -> x (k, 73) uint16 boards, y (k,) uint16 moves."""
    # Runs in the worker processes, one game per call.
    game_number, ids = item
    tokens = [game_vocab[i] for i in ids]   # ids -> strings, e.g. ["<start>", "<white_wins>", "<elo_1400>", ..., "e2e4", ..., "<end>"]
    # Positions 1-3 are the result and the two rating buckets; the moves sit between them and the final <end>.
    result, elo_w, elo_b, ucis = tokens[1], tokens[2], tokens[3], tokens[4:-1]   # without <start> and <end>

    # Pick POSITIONS_PER_GAME different move numbers (plies) at random, without repeats.
    # Seeded by game number: forked worker processes would otherwise share the same random state.
    # It also makes the choice reproducible: the same game always gives the same positions,
    # so they can be found again later (e.g. to test the move model on exactly the same positions).
    rng = np.random.default_rng(game_number)
    keep = set(rng.choice(len(ucis), size=min(POSITIONS_PER_GAME, len(ucis)), replace=False).tolist())

    # Replay the whole game from the starting position. Every move has to be played, because a later
    # board depends on all earlier moves, but only the chosen plies are saved.
    board = chess.Board()
    xs, ys = [], []
    for ply, uci in enumerate(ucis):
        if ply in keep:
            # Save BEFORE playing the move: the input is the position the player saw,
            # the target is what they played in it.
            xs.append(encode_board(board, result, elo_w, elo_b))   # x = board before the move
            ys.append(move_to_id[uci])                              # y = the move played
        board.push(chess.Move.from_uci(uci))                       # no legality check: games.bin was checked when it was built
    # uint16 = 2 bytes per number (ids are below 65,536). x: (10, 73), y: (10,)
    return np.array(xs, dtype=np.uint16), np.array(ys, dtype=np.uint16)


def build_dataset():
    """games.bin -> boards_{train,val}.bin (N, 73) and moves_{train,val}.bin (N,), uint16.
    Same split as the move model: first 90% of games train, last 10% val, so both models are tested on the same games."""
    start = time.time()
    games_data, starts, lengths = game_index()
    # Split by GAME, not by position: positions from one game are nearly identical,
    # so having some in train and some in val would make val look better than it is.
    n_train_games = int(0.9 * len(starts))
    n_pos = {"train": 0, "val": 0}   # positions written so far, for the progress print
    # Four output files, opened once and appended to: boards and moves for train and for val.
    # Row i of boards_*.bin and entry i of moves_*.bin belong together.
    files = {s: (open(HERE / f"boards_{s}.bin", "wb"), open(HERE / f"moves_{s}.bin", "wb")) for s in ("train", "val")}

    # A generator of (game_number, token ids of the game), one game at a time, so the
    # 3M games are never all in memory as Python lists.
    games = ((i, games_data[s:s+L].tolist()) for i, (s, L) in enumerate(zip(starts.tolist(), lengths.tolist())))
    # Pool: N_WORKERS processes run convert_game in parallel. "fork" = each worker starts as a copy of
    # this process, so it already has the vocabs and functions without re-importing anything.
    with mp.get_context("fork").Pool(N_WORKERS) as pool:
        # imap: send games to the workers in chunks of 500 and return the results IN GAME ORDER.
        # The order matters: game i must go to the same split as in the move model.
        for i, (x, y) in enumerate(pool.imap(convert_game, games, chunksize=500)):
            split = "train" if i < n_train_games else "val"
            # tofile appends the raw bytes: x adds 10 rows x 73 ids, y adds 10 ids.
            x.tofile(files[split][0])
            y.tofile(files[split][1])
            n_pos[split] += len(y)
            if (i + 1) % 500_000 == 0:
                print(f"games {i + 1:,} / {len(starts):,}, positions {sum(n_pos.values()):,} ({time.time() - start:.0f}s)", flush=True)

    for fb, fm in files.values():
        fb.close()
        fm.close()
    # Save both vocabs next to the data, so the meaning of every id in the .bin files is recorded.
    with open(HERE / "board_vocab.json", "w") as f:
        json.dump({"input_vocab": input_vocab, "moves": moves}, f)

    print(f"\nDone in {time.time() - start:.0f}s: games train {n_train_games:,} / val {len(starts) - n_train_games:,}, "
          f"positions train {n_pos['train']:,} / val {n_pos['val']:,}")


def load_split(split):
    """Memmapped (boards (N, 73), moves (N,)): read from disk only where batches touch them."""
    # The .bin files are flat lists of numbers. reshape(-1, 73) turns the boards file back into
    # one row per board; -1 means "work out the number of rows" (27,014,690 for train).
    boards = np.memmap(HERE / f"boards_{split}.bin", dtype=np.uint16, mode="r").reshape(-1, block_size)
    moves_ = np.memmap(HERE / f"moves_{split}.bin", dtype=np.uint16, mode="r")
    return boards, moves_


# ---------------------------------------------------------------- batches + eval
def get_batch(split, batch_size):
    """batch_size random boards: x (B, 73) token ids, y (B,) move ids. Every board has the same length, so no padding."""
    boards, moves_ = data[split]
    # batch_size random row numbers. Sorting them means the rows are read from the file in order,
    # which is faster than jumping back and forth.
    ix = np.sort(np.random.randint(0, len(moves_), batch_size))   # sorted = fewer random jumps in the file
    # Only these rows are read from disk. uint16 -> int64 because nn.Embedding and cross_entropy need int64 (long).
    x = torch.from_numpy(boards[ix].astype(np.int64))
    y = torch.from_numpy(moves_[ix].astype(np.int64))
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad
def evaluate_loss():

    out = {}
    m.eval()

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        correct = 0

        for k in range(eval_iters):
            x, y = get_batch(split, batch_size)
            logits, loss = m(x, y)

            losses[k] = loss
            correct += (logits.argmax(dim=-1) == y).sum().item()

        out[split] = losses.mean()
        out[f"{split}_acc"] = correct / (eval_iters * batch_size)   # top move = the human's move
    m.train()
    return out


# ---------------------------------------------------------------- model
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()

        self.query = nn.Linear(n_emb, head_size)
        self.keys = nn.Linear(n_emb, head_size)
        self.values = nn.Linear(n_emb, head_size)
        self.head_size = head_size
        # no tril buffer: a board has no "future", every square may look at every other square

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape

        q = self.query(x) # x = (B,T,n_emb), self.query = (n_emb, head_size), q = (B, T, head_size)
        k = self.keys(x) # x = (B,T,n_emb), self.keys = (n_emb, head_size), k = (B, T, head_size)

        wei = q @ k.transpose(-2,-1) * self.head_size**-0.5 # q = (B, T, head_size), k.T = (B, head_size, T), wei = (B, T, T)
        wei = F.softmax(wei, dim=-1)   # no masked_fill: encoder attention
        wei = self.dropout(wei)

        v = self.values(x) # x = (B,T,n_emb), self.values = (n_emb, head_size), v = (B, T, head_size)

        out = wei @ v # wei = (B, T, T), v = (B, T, head_size), out = (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()

        self.heads = nn.ModuleList([Head(head_size) for i in range(num_heads)])
        self.proj = nn.Linear(n_emb, n_emb)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_emb):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_emb, 4*n_emb),
            nn.ReLU(),
            nn.Linear(4*n_emb, n_emb),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        out = self.net(x)
        return out


class Block(nn.Module):
    def __init__(self, n_emb, num_heads):
        super().__init__()

        head_size = n_emb // num_heads
        self.multi_attension = MultiHeadAttention(num_heads, head_size)
        self.feed_forward = FeedForward(n_emb)
        self.ln1 = nn.LayerNorm(n_emb)                 # Layer normalization (normalizes the output of the self-attention layer)
        self.ln2 = nn.LayerNorm(n_emb)                 # Layer normalization (normalizes the output of the feed-forward layer)

    def forward(self, x):
        x = x + self.multi_attension(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))

        return x


class Transormer(nn.Module):
    def __init__(self, n_emb):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, n_emb)  #self.embedding_layer = (44, n_emb): board tokens
        self.positional_embeddings = nn.Embedding(block_size, n_emb) #self.positional_embeddings = (73, n_emb): one row per position/square

        self.blocks = nn.Sequential(*[Block(n_emb, num_heads) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_emb)
        self.linear = nn.Linear(n_emb, n_moves) # self.linear = (n_emb, 1968): output vocab is moves, not board tokens

    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_emb = self.embedding_layer(idx) # token_emb = (B, T, n_emb)
        positional_emb = self.positional_embeddings(torch.arange(T, device=device)) # positional_emb = (T, n_emb)

        x = token_emb + positional_emb # x = (B, T, n_emb)

        attention = self.blocks(x)  # (B, T, n_emb): every position has now seen the whole board

        x = self.ln_f(attention[:, 0, :])  # (B, n_emb): read the answer at position 0 (side to move), like BERT's [CLS]
        logits = self.linear(x) # logits = (B, n_moves)

        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits, targets)   # targets = (B,)
        return logits, loss

    @torch.no_grad
    def generate(self, board, result="<white_wins>", white_elo="<elo_2000>", black_elo="<elo_2000>", max_moves=200):
        """Plays a whole game from `board` with python-chess keeping the board.
        Each turn: encode the board, one forward pass, illegal moves set to -inf, sample."""
        board = board.copy()
        for _ in range(max_moves):
            if board.is_game_over():
                break
            idx = torch.tensor([encode_board(board, result, white_elo, black_elo)], dtype=torch.long, device=device)
            logits, _ = self(idx)
            legal = [move_to_id[mv.uci()] for mv in board.legal_moves]
            mask = torch.full_like(logits, float("-inf"))
            mask[0, legal] = 0
            probs = F.softmax(logits + mask, dim=-1)
            ix = torch.multinomial(probs, num_samples=1).item()
            board.push(chess.Move.from_uci(moves[ix]))
        return board


# ---------------------------------------------------------------- training the model
def get_lr(step):
    """Linear warmup to learning_rate, then constant; during the final decay, cosine from learning_rate to min_lr."""
    if step < warmup_iters:
        return learning_rate * (step + 1) / warmup_iters
    if decay_start is None or step < decay_start:
        return learning_rate
    progress = min(1.0, (step - decay_start) / decay_iters)   # 0 -> 1 over the decay
    return min_lr + 0.5 * (learning_rate - min_lr) * (1 + math.cos(math.pi * progress))


def save_checkpoint(m, opt, step, path):
    # Save everything needed for inference, plus optimizer + step so training can resume
    torch.save({
        "model_state_dict": m.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "step": step,
        "input_vocab": input_vocab,
        "moves": moves,
        "config": {
            "n_emb": n_emb, "num_heads": num_heads, "n_layer": n_layer,
            "block_size": block_size, "vocab_size": vocab_size, "n_moves": n_moves, "dropout": dropout,
        },
    }, CKPT_DIR / path)
    print(f"Model saved to {CKPT_DIR / path}")


def train(m, opt, start_step, num_iters, name):
    # Total parameters
    n_params = sum(p.numel() for p in m.parameters())
    print(f"Total parameters: {n_params/1e6:.2f}M")

    m.train()
    best_val = float("inf")
    for step in range(start_step, start_step + num_iters):
        x, y = get_batch("train", batch_size)

        lr = get_lr(step)   # step counts from 0 of the whole run, so a resumed run continues the same schedule
        for group in opt.param_groups:
            group["lr"] = lr

        logits, loss = m(x, y)

        loss.backward()
        opt.step()
        opt.zero_grad()   # gradients add up across backward() calls; clear them every step

        if step % eval_interval == 0:
            out = evaluate_loss()
            print(f"Loss at step {step}: train {out['train']:.4f}, val {out['val']:.4f}, "
                  f"val acc {out['val_acc']:.1%}, lr {lr:.2e}")
            # Only latest + best are kept: a 16M checkpoint with optimizer state is ~200 MB
            save_checkpoint(m, opt, step, f"{name}_latest.pt")
            if out["val"] < best_val:
                best_val = out["val"]
                save_checkpoint(m, opt, step, f"{name}_best.pt")

    # Play one game against itself, python-chess keeps the board
    m.eval()
    board = m.generate(chess.Board())
    print(chess.Board().variation_san(board.move_stack))
    print("Result:", board.result(claim_draw=True), f"after {len(board.move_stack)} plies")
    m.train()

    save_checkpoint(m, opt, start_step + num_iters, f"{name}.pt")
    return m


def train_from_scratch(num_iters=max_iters):
    global m   # evaluate_loss() uses the global m
    m = Transormer(n_emb).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=learning_rate)
    return train(m, opt, 0, num_iters, "chess_board_30M_v2")


def train_from_checkpoint(path, num_iters=None, decay=False):
    """decay=False: continue at the constant lr. decay=True: the final decay, ~15% of the steps done so far."""
    global m, decay_start, decay_iters   # evaluate_loss() uses the global m
    ckpt = torch.load(CKPT_DIR / path, map_location=device, weights_only=True)   # names are relative to inference/

    # The checkpoint must have the same architecture and vocabulary as this script
    config = {"n_emb": n_emb, "num_heads": num_heads, "n_layer": n_layer,
              "block_size": block_size, "vocab_size": vocab_size, "n_moves": n_moves, "dropout": dropout}
    if ckpt["config"] != config:
        raise ValueError(f"Checkpoint config {ckpt['config']} does not match script config {config}")
    if ckpt["input_vocab"] != input_vocab or ckpt["moves"] != moves:
        raise ValueError("Checkpoint vocabulary does not match this script")

    m = Transormer(n_emb).to(device)
    m.load_state_dict(ckpt["model_state_dict"])
    opt = torch.optim.AdamW(m.parameters(), lr=learning_rate)
    opt.load_state_dict(ckpt["optimizer_state_dict"])
    start_step = ckpt["step"] + 1

    if decay:
        decay_start = start_step
        decay_iters = num_iters or max(5000, int(0.15 * start_step))
        print(f"Final decay from {path}: steps {start_step} -> {start_step + decay_iters}, lr {learning_rate} -> {min_lr}")
        return train(m, opt, start_step, decay_iters, "chess_board_30M_v2_decayed")

    if num_iters is None:              # by default, continue up to max_iters (or until stopped)
        num_iters = max_iters - start_step
    print(f"Resuming from {path} at step {start_step}, constant lr {learning_rate}")
    return train(m, opt, start_step, num_iters, "chess_board_30M_v2")


if __name__ == "__main__":
    if not (HERE / "boards_train.bin").exists():
        build_dataset()

    data = {split: load_split(split) for split in ("train", "val")}
    print(f"positions: train {len(data['train'][1]):,}, val {len(data['val'][1]):,}")

    train_from_scratch()
    # train_from_checkpoint("chess_board_30M_v2_latest.pt")               # continue after a stop (constant lr)
    # train_from_checkpoint("chess_board_30M_v2_latest.pt", decay=True)   # once, at the end: final lr decay
