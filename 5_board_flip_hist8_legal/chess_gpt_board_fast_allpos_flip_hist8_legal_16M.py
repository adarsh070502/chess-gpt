"""
Board model with a second training target: which moves are legal. ~17M params.
Copy of 4_board_flip_hist8/chess_gpt_board_fast_allpos_flip_hist8_16M.py (same input, same data, same schedule)
plus ONE change, the legal-move target:

    5. Next to the move head (self.linear: "which move did the human play?", softmax over 1,968 moves) the model has
       a second output layer, self.legal_head: "is each of the 1,968 moves legal here?", 1,968 separate yes/no
       answers (sigmoid). Labels come free from python-chess (board.legal_moves). Both heads read the same vector at
       position 0, so the shared Blocks must learn attacked squares and blocked paths, which the move head also uses.
       loss = cross_entropy(move) + LEGAL_WEIGHT * binary_cross_entropy(legal, summed over the 1,968 moves)
       At inference only the move head is used; the legal head only shapes training.

The four changes of the folder-4 model, for reference (board in, move out, as in 3_board/):

    1. Faster training: one fused attention call for all heads (F.scaled_dot_product_attention) instead of
       a Python loop over Head modules, and bf16 autocast (matmuls in bf16, weights and optimizer stay fp32).
    2. All positions: training boards are made on the fly by DataLoader workers that replay games from
       games.bin and pick random plies, so every one of the ~190M train positions can be used (no 27 GB file).
    3. Board seen from the side to move: when Black is to move, the board is mirrored (rank 8 <-> rank 1,
       colours swapped), so the model always plays "upwards" with the uppercase pieces. The target move,
       the history moves, the result and the ratings are flipped the same way.
    4. History: the last 8 moves are added as 8 extra tokens (as Leela / Maia-2 do), so 81 tokens per position.

One position = 81 tokens, always in this order (all from the side to move's point of view, "me"):
    position 0       <cls>                                   constant; the move is read from here (like BERT's [CLS])
    positions 1-4    <K>/<noK> <Q>/<noQ> <k>/<nok> <q>/<noq>  castling rights: uppercase = mine, lowercase = opponent's
    position 5       <ep_none> / <ep_a> ... <ep_h>            en passant file
    position 6       <me_wins> / <me_loses> / <draw>          result of the game, for the side to move
    positions 7-8    <elo_...> <elo_...>                      my rating bucket, then the opponent's
    positions 9-72   64 squares a8 ... h8 a7 ... h1           '.' empty, PNBRQK mine, pnbrqk opponent's
    positions 73-80  last 8 moves, oldest first               UCI move tokens, <no_move> before the game had 8 moves

The val set is the SAME 3.0M positions as boards_val.bin (same games, same plies), only encoded this way,
so val accuracy can be compared directly with the 16M (36.7%) and 30M (35.1%) board models.

Schedule (warmup -> constant -> decay, "WSD"), so the run can be stopped at any time:
    1. train_from_scratch(): warmup, then a constant lr. Stop with Ctrl+C whenever the GPU is needed;
       chess_board_fast_allpos_flip_hist8_legal_16M_latest.pt is saved every eval_interval steps.
    2. train_from_checkpoint("chess_board_fast_allpos_flip_hist8_legal_16M_latest.pt"): continue at the constant lr (repeat as needed).
    3. train_from_checkpoint("chess_board_fast_allpos_flip_hist8_legal_16M_latest.pt", decay=True): once, at the end.

Run from the repo root:  python chess_gpt/5_board_flip_hist8_legal/chess_gpt_board_fast_allpos_flip_hist8_legal_16M.py
The val files are folder 4's (built there on first use, ~1 min); training data is never written to disk.
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
# Measured on the RTX 3050 (16M, bf16): batch 128 -> 870 positions/s, 256 -> 932, 512 -> 943. The GPU is full at
# any size, so 256 is the pick: 7% faster than 128 and less noisy gradients (only one target per position).
max_iters = 150000    # 150k x 256 = 38.4M positions, ~12 h: 130k steps at constant lr + 20k decay (see __main__)
batch_size = 256      # positions per batch (2.3 GB of GPU memory)
block_size = 81       # tokens per position: 9 state + 64 squares + 8 history moves
eval_interval = 1000     # every 256k positions
learning_rate = 4e-4     # old 16M: 3e-4 at batch 128; 2x batch -> ~sqrt(2)x lr. Not higher: no gradient clipping
min_lr = 4e-5            # end of the final decay (10% of learning_rate)
warmup_iters = 2000      # linear ramp 0 -> learning_rate (512k positions)
betas = (0.9, 0.95)      # AdamW betas used for GPT training: beta2 0.95 reacts faster to big gradients, more stable
decay_start = None       # set by train_from_checkpoint(..., decay=True): step where the final decay begins
decay_iters = None       # length of the final decay
n_emb = 396    # same width as the old 16M; ~17M params (the history-move embeddings add ~0.8M)
num_heads = 6  # head_size 66
n_layer = 8    # no of multi attention and feedforward blocks
dropout = 0.0  # no dropout: ~190M train positions, the model never sees the same one twice
eval_iters = 100 # val batches per evaluation
LEGAL_WEIGHT = 0.1   # weight of the legal-move loss (summed over 1,968 yes/no answers per position, so it is ~100x
                     # bigger than a per-move average; 0.1 keeps it from drowning the move loss). See the log.
MAX_LEGAL = 256      # positions store up to 256 legal move ids (chess maximum is 218), padded with n_moves

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# bf16 autocast only on the GPU (RTX 30xx supports bf16); on CPU everything stays fp32
use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()

# Dataset settings
HERE = Path(__file__).parent         # folder of this script; its own data files (.bin) live here
DATA = HERE.parent / "data"          # shared data: games.bin, vocab.json (made by data/data_with_player_rating_start_token.ipynb)
CKPT_DIR = HERE / "inference"       # checkpoints are saved here, next to the inference code that loads them
VAL_DIR = HERE.parent / "4_board_flip_hist8"   # the val set is the same as folder 4's (same encoding), so it is shared
VAL_POSITIONS_PER_GAME = 10   # the val set: the same 10 plies per game as boards_val.bin
PLIES_PER_VISIT = 8           # training: plies taken from a game each time a worker replays it
POOL_BATCHES = 32             # a worker collects 32 batches' worth of positions, shuffles them, then hands them out
N_LOADER_WORKERS = 6          # CPU processes making training positions while the GPU trains
MOVE_BLOCK_SIZE = 256         # the move model drops games longer than 257 tokens; drop the same ones
HISTORY = 8                   # number of previous moves given to the model
NAME = "chess_board_fast_allpos_flip_hist8_legal_16M"   # checkpoint name prefix


# ---------------------------------------------------------------- vocab
# Input vocab: every token that can appear in a position sequence.
# Its id is its index in this list.
#
# 13 square tokens. After the flip, uppercase = the side to move ("my" pieces), lowercase = the opponent's.
SQUARE_TOKENS = [".", "P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]
# Position 0. In the old models it said whose turn it was; after the flip it is always "me",
# so it is a constant token whose only job is to collect the answer (the move is read from here).
CLS_TOKENS = ["<cls>"]
# Castling rights: K/Q = my kingside/queenside, k/q = the opponent's (after the flip).
CASTLE_TOKENS = ["<K>", "<noK>", "<Q>", "<noQ>", "<k>", "<nok>", "<q>", "<noq>"]
# En passant file (the board is mirrored only up/down, so files stay the same).
EP_TOKENS = ["<ep_none>"] + [f"<ep_{f}>" for f in "abcdefgh"]
# Result for the side to move. A White win is <me_wins> on White's moves and <me_loses> on Black's moves.
RESULT_TOKENS = ["<me_wins>", "<me_loses>", "<draw>"]
ELO_TOKENS = [f"<elo_{b}>" for b in range(800, 2401, 200)]   # <elo_800>, <elo_1000>, ..., <elo_2400>
# History: one token per UCI move (the same 1,968 moves as the output), plus a filler for "no move yet".
NO_MOVE = ["<no_move>"]

# Output vocab: the 1,968 UCI moves, in the same order as the move model's vocab.json.
with open(DATA / "vocab.json") as f:
    game_vocab = json.load(f)       # token id -> token, for games.bin
moves = [t for t in game_vocab if not t.startswith("<")]
move_to_id = {mv: i for i, mv in enumerate(moves)}  # "e2e4" -> output id
n_moves = len(moves)             # 1968

input_vocab = SQUARE_TOKENS + CLS_TOKENS + CASTLE_TOKENS + EP_TOKENS + RESULT_TOKENS + ELO_TOKENS + NO_MOVE + moves
stoi = {t: i for i, t in enumerate(input_vocab)}   # token -> id
vocab_size = len(input_vocab)    # 44 + 1,968 = 2,012
MOVE_OFFSET = stoi[moves[0]]     # input id of a history move = MOVE_OFFSET + its output id

# The game result as stored in games.bin (from White's point of view) -> result for the side to move.
RESULT_FOR_MOVER = {
    "<white_wins>": {chess.WHITE: "<me_wins>", chess.BLACK: "<me_loses>"},
    "<black_wins>": {chess.WHITE: "<me_loses>", chess.BLACK: "<me_wins>"},
    "<draw>": {chess.WHITE: "<draw>", chess.BLACK: "<draw>"},
}


def mirror_move(move):
    """The same move seen in the mirrored board: e7e5 -> e2e4, e1g1 (O-O) -> e8g8. Promotion piece unchanged."""
    return chess.Move(chess.square_mirror(move.from_square), chess.square_mirror(move.to_square), move.promotion)


def encode_position(board, history, result, white_elo, black_elo):
    """python-chess board + previous moves -> (81 input ids, flip).

    board:    the position BEFORE the move to predict (real orientation)
    history:  list of chess.Move played before it, oldest first (real orientation); only the last 8 are used
    result:   "<white_wins>" / "<black_wins>" / "<draw>" as stored in games.bin
    flip:     True if the board was mirrored (Black to move). The target move, and at play time the
              model's answer, must be mirrored with mirror_move() too.
    """
    flip = board.turn == chess.BLACK
    mover = board.turn
    if flip:
        # board.mirror(): ranks flipped (a8 <-> a1), colours swapped, castling rights / en passant / turn
        # flipped too. The result is the same position "seen from Black's chair": Black's pieces are now
        # uppercase on ranks 1-2 and it is "White" (= me) to move.
        board = board.mirror()
        history = [mirror_move(m) for m in history]
        white_elo, black_elo = black_elo, white_elo     # the mover's rating goes first
    my_elo, opp_elo = white_elo, black_elo

    # After a possible flip, "White" in python-chess is always me.
    ep = board.ep_square if board.has_legal_en_passant() else None
    tokens = [
        "<cls>",                                                                   # 0: where the move is read
        "<K>" if board.has_kingside_castling_rights(chess.WHITE) else "<noK>",     # 1: my O-O allowed?
        "<Q>" if board.has_queenside_castling_rights(chess.WHITE) else "<noQ>",    # 2: my O-O-O allowed?
        "<k>" if board.has_kingside_castling_rights(chess.BLACK) else "<nok>",     # 3: opponent's O-O allowed?
        "<q>" if board.has_queenside_castling_rights(chess.BLACK) else "<noq>",    # 4: opponent's O-O-O allowed?
        "<ep_none>" if ep is None else f"<ep_{chess.FILE_NAMES[chess.square_file(ep)]}>",   # 5
        RESULT_FOR_MOVER[result][mover],                                           # 6: result for me
        my_elo, opp_elo,                                                           # 7, 8
    ]
    # 9-72: the 64 squares, rank 8 to rank 1, file a to h (same order as the old models).
    piece_map = board.piece_map()            # {square: Piece}, only occupied squares
    for rank in range(7, -1, -1):
        for file in range(8):
            piece = piece_map.get(chess.square(file, rank))
            tokens.append(piece.symbol() if piece else ".")
    ids = [stoi[t] for t in tokens]

    # 73-80: the last 8 moves, oldest first, padded with <no_move> at the front early in the game.
    last = history[-HISTORY:]
    ids += [stoi["<no_move>"]] * (HISTORY - len(last))
    ids += [MOVE_OFFSET + move_to_id[m.uci()] for m in last]
    return ids, flip


def target_id(move, flip):
    """The move to predict as an output id, mirrored if the board was."""
    return move_to_id[(mirror_move(move) if flip else move).uci()]


# ---------------------------------------------------------------- games.bin
# games.bin holds the move model's games, back to back as uint16 token ids:
#     <start> <result> <elo_W> <elo_B> e2e4 e7e5 ... <end>
def game_index():
    """games.bin plus (starts, lengths) of its games, exactly as chess_gpt_move_tokens.py selects them."""
    games_data = np.memmap(DATA / "games.bin", dtype=np.uint16, mode="r")
    starts = np.flatnonzero(games_data == game_vocab.index("<start>"))
    lengths = np.diff(starts, append=len(games_data))
    keep = lengths <= MOVE_BLOCK_SIZE + 1     # same games as the move model and the old board models
    return games_data, starts[keep], lengths[keep]


def positions_from_game(ids, plies):
    """Token ids of one game + the plies (move numbers) to keep
    -> x (k, 81) uint16, y (k,) uint16, legal (k, MAX_LEGAL) uint16.

    Replays the game on a python-chess board up to the last chosen ply; at each chosen ply it saves
    (position before the move, the move played), both from the side to move's point of view, plus the ids
    of ALL legal moves in that position (the labels of the legal head), flipped the same way as y.
    legal is padded with n_moves (an id that is not a move), so every row has the same length.
    """
    tokens = [game_vocab[i] for i in ids]
    result, elo_w, elo_b, ucis = tokens[1], tokens[2], tokens[3], tokens[4:-1]   # without <start> and <end>
    board = chess.Board()
    xs, ys, legals = [], [], []
    last_ply = max(plies)
    for ply, uci in enumerate(ucis[:last_ply + 1]):
        move = chess.Move.from_uci(uci)
        if ply in plies:
            # board.move_stack = every move played so far, oldest first; encode_position uses the last 8
            x, flip = encode_position(board, board.move_stack, result, elo_w, elo_b)
            xs.append(x)
            ys.append(target_id(move, flip))
            legal = [target_id(m, flip) for m in board.legal_moves]   # e.g. 20 ids in the starting position
            legals.append(legal + [n_moves] * (MAX_LEGAL - len(legal)))
        board.push(move)
    return np.array(xs, dtype=np.uint16), np.array(ys, dtype=np.uint16), np.array(legals, dtype=np.uint16)


# ---------------------------------------------------------------- val set (built once, on disk)
def convert_val_game(item):
    """(game_number, ids) -> the same 10 plies as boards_val.bin, encoded with the flip + 8 history moves."""
    game_number, ids = item
    n_plies = len(ids) - 5          # minus <start> <result> <elo> <elo> ... <end>
    # Exactly the same random choice as build_dataset() in the old board scripts: same seed, same call.
    rng = np.random.default_rng(game_number)
    plies = set(rng.choice(n_plies, size=min(VAL_POSITIONS_PER_GAME, n_plies), replace=False).tolist())
    return positions_from_game(ids, plies)


def build_val():
    """Last 10% of games -> boards_flip_hist8_val.bin (N, 81) and moves_flip_hist8_val.bin (N,), uint16."""
    start = time.time()
    games_data, starts, lengths = game_index()
    n_train_games = int(0.9 * len(starts))
    # game_number = the game's index in the full list, as in the old build_dataset(), so the seeds match
    games = ((i, games_data[s:s+L].tolist())
             for i, (s, L) in enumerate(zip(starts.tolist(), lengths.tolist())) if i >= n_train_games)
    n = 0
    with open(VAL_DIR / "boards_flip_hist8_val.bin", "wb") as fb, open(VAL_DIR / "moves_flip_hist8_val.bin", "wb") as fm, \
            mp.get_context("fork").Pool(os.cpu_count() - 1) as pool:
        for x, y, _ in pool.imap(convert_val_game, games, chunksize=500):   # imap keeps game order; legal lists not stored
            x.tofile(fb)
            y.tofile(fm)
            n += len(y)
    print(f"val set: {len(starts) - n_train_games:,} games, {n:,} positions ({time.time() - start:.0f}s)")


def load_val():
    boards = np.memmap(VAL_DIR / "boards_flip_hist8_val.bin", dtype=np.uint16, mode="r").reshape(-1, block_size)
    moves_ = np.memmap(VAL_DIR / "moves_flip_hist8_val.bin", dtype=np.uint16, mode="r")
    return boards, moves_


# ---------------------------------------------------------------- training positions (made on the fly)
class PositionStream(torch.utils.data.IterableDataset):
    """Endless stream of training batches, made by the DataLoader worker processes.

    Each worker, forever:
      1. picks POOL_BATCHES * batch_size / PLIES_PER_VISIT random train games,
      2. replays each one and keeps PLIES_PER_VISIT random plies (different ones every visit, so over
         time all ~190M train positions get used, not just a fixed 10 per game),
      3. shuffles the pool, so one batch holds positions from many different games, and yields it
         batch by batch.
    Positions of one game are very similar; the shuffle keeps them from landing in the same batch.
    """

    def __init__(self, games_data, starts, lengths):
        # With the "fork" start method each worker gets these arrays without copying or pickling them.
        self.games_data, self.starts, self.lengths = games_data, starts, lengths

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        # Each worker gets its own seed from PyTorch (different per worker and per run).
        rng = np.random.default_rng(info.seed if info is not None else None)
        games_per_pool = POOL_BATCHES * batch_size // PLIES_PER_VISIT
        while True:
            xs, ys, legals = [], [], []
            for g in rng.integers(0, len(self.starts), games_per_pool):
                s, L = int(self.starts[g]), int(self.lengths[g])
                ids = self.games_data[s:s+L].tolist()
                n_plies = L - 5
                plies = set(rng.choice(n_plies, size=min(PLIES_PER_VISIT, n_plies), replace=False).tolist())
                x, y, legal = positions_from_game(ids, plies)
                xs.append(x)
                ys.append(y)
                legals.append(legal)
            x, y, legal = np.concatenate(xs), np.concatenate(ys), np.concatenate(legals)
            order = rng.permutation(len(y))
            for i in range(0, len(y) - batch_size + 1, batch_size):
                ix = order[i:i + batch_size]
                yield (torch.from_numpy(x[ix].astype(np.int64)), torch.from_numpy(y[ix].astype(np.int64)),
                       torch.from_numpy(legal[ix].astype(np.int64)))


def make_train_loader():
    games_data, starts, lengths = game_index()
    n_train_games = int(0.9 * len(starts))       # first 90% of games = train, as in every other script
    stream = PositionStream(games_data, starts[:n_train_games], lengths[:n_train_games])
    # batch_size=None: the stream already yields whole batches. pin_memory: faster copies to the GPU.
    loader = torch.utils.data.DataLoader(stream, batch_size=None, num_workers=N_LOADER_WORKERS,
                                         pin_memory=device == "cuda", prefetch_factor=8,
                                         multiprocessing_context="fork", persistent_workers=True)
    return iter(loader)


# ---------------------------------------------------------------- eval
def get_val_batch(batch_size):
    boards, moves_ = val_data
    ix = np.sort(np.random.randint(0, len(moves_), batch_size))
    x = torch.from_numpy(boards[ix].astype(np.int64)).to(device)
    y = torch.from_numpy(moves_[ix].astype(np.int64)).to(device)
    return x, y


def autocast():
    # bf16 on the GPU: matmuls run in bf16 (about 2x faster on tensor cores), weights stay fp32.
    # bf16 keeps fp32's exponent range, so unlike fp16 no loss scaling is needed.
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16)


@torch.no_grad
def evaluate_val():
    """Val loss and top-1 accuracy on eval_iters random val batches."""
    m.eval()
    losses = torch.zeros(eval_iters)
    correct = 0
    for k in range(eval_iters):
        x, y = get_val_batch(batch_size)
        with autocast():
            logits, loss = m(x, y)
        losses[k] = loss.item()
        correct += (logits.argmax(dim=-1) == y).sum().item()
    m.train()
    return losses.mean().item(), correct / (eval_iters * batch_size)


# ---------------------------------------------------------------- model
# Same blocks as before. The only change is inside MultiHeadAttention: instead of num_heads separate Head
# modules (each with its own small q/k/v Linear, run one after another in a Python loop), one Linear makes
# q, k, v for all heads at once, and F.scaled_dot_product_attention does
#     softmax(q @ k.T * head_size**-0.5) @ v
# for all heads in one fused GPU kernel. Same maths, far fewer kernel launches, and the (T, T) attention
# matrix never has to be stored for the backward pass.
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.qkv = nn.Linear(n_emb, 3 * num_heads * head_size)   # q, k, v of every head in one matmul
        self.proj = nn.Linear(n_emb, n_emb)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)                    # each (B, T, C)
        # (B, T, C) -> (B, num_heads, T, head_size): every head becomes its own "batch" entry
        q = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        # is_causal=False: no mask, every square sees every square (encoder attention).
        # The default scale is head_size**-0.5, the same scaling as the old Head class.
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout if self.training else 0.0, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(B, T, C)      # heads side by side again, like torch.cat
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
        self.ln1 = nn.LayerNorm(n_emb)
        self.ln2 = nn.LayerNorm(n_emb)

    def forward(self, x):
        x = x + self.multi_attension(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))

        return x


class Transormer(nn.Module):
    def __init__(self, n_emb):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, n_emb)        # (2012, n_emb): board tokens + history moves
        self.positional_embeddings = nn.Embedding(block_size, n_emb)  # (81, n_emb)

        self.blocks = nn.Sequential(*[Block(n_emb, num_heads) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_emb)
        self.linear = nn.Linear(n_emb, n_moves)                       # (n_emb, 1968): which move did the human play?
        self.legal_head = nn.Linear(n_emb, n_moves)                   # (n_emb, 1968): is each move legal? (training only)
        # Start the legal head at the base rate (~30 of 1,968 moves are legal) instead of 50/50, so the first steps
        # are not spent only on learning "most moves are illegal".
        nn.init.constant_(self.legal_head.bias, math.log(30 / (n_moves - 30)))

    def forward(self, idx, targets=None, legal_targets=None):
        """targets (B,) = move ids; legal_targets (B, 1968) = 1.0 for legal moves, 0.0 otherwise (training only)."""
        B, T = idx.shape

        token_emb = self.embedding_layer(idx)                                        # (B, T, n_emb)
        positional_emb = self.positional_embeddings(torch.arange(T, device=idx.device))  # (T, n_emb)
        x = token_emb + positional_emb

        attention = self.blocks(x)                    # (B, T, n_emb)
        x = self.ln_f(attention[:, 0, :])             # (B, n_emb): read at position 0 (<cls>)
        logits = self.linear(x)                       # (B, n_moves)

        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.float(), targets)   # loss in fp32 even under bf16 autocast
            self.move_loss = loss.detach()
            if legal_targets is not None:
                legal_logits = self.legal_head(x).float()     # (B, 1968): one yes/no score per move
                # sigmoid + binary cross entropy for each of the 1,968 moves, summed per position, averaged over B
                legal_loss = F.binary_cross_entropy_with_logits(legal_logits, legal_targets, reduction="sum") / B
                self.legal_loss = legal_loss.detach()
                loss = loss + LEGAL_WEIGHT * legal_loss
        return logits, loss

    @torch.no_grad
    def next_move(self, board, result="<white_wins>", white_elo="<elo_2000>", black_elo="<elo_2000>", greedy=False):
        """The model's move in `board` (real orientation), only among legal moves."""
        x, flip = encode_position(board, board.move_stack, result, white_elo, black_elo)
        idx = torch.tensor([x], dtype=torch.long, device=next(self.parameters()).device)
        logits, _ = self(idx)
        legal = list(board.legal_moves)
        legal_ids = torch.tensor([target_id(mv, flip) for mv in legal], device=logits.device)
        legal_logits = logits[0, legal_ids].float()
        i = legal_logits.argmax().item() if greedy else torch.multinomial(F.softmax(legal_logits, -1), 1).item()
        return legal[i]     # legal[] holds the real (unmirrored) moves, so no mirroring back is needed

    def generate(self, board, max_moves=200, **kwargs):
        """Plays a whole game from `board` against itself."""
        board = board.copy()
        for _ in range(max_moves):
            if board.is_game_over():
                break
            board.push(self.next_move(board, **kwargs))
        return board


# ---------------------------------------------------------------- training the model
def get_lr(step):
    """Linear warmup to learning_rate, then constant; during the final decay, cosine from learning_rate to min_lr."""
    if step < warmup_iters:
        return learning_rate * (step + 1) / warmup_iters
    if decay_start is None or step < decay_start:
        return learning_rate
    progress = min(1.0, (step - decay_start) / decay_iters)
    return min_lr + 0.5 * (learning_rate - min_lr) * (1 + math.cos(math.pi * progress))


def model_config():
    return {"n_emb": n_emb, "num_heads": num_heads, "n_layer": n_layer, "block_size": block_size,
            "vocab_size": vocab_size, "n_moves": n_moves, "dropout": dropout,
            "format": "side-to-move view (flip), 8 history moves, legal-move head"}


def save_checkpoint(m, opt, step, path):
    torch.save({
        "model_state_dict": m.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "step": step,
        "input_vocab": input_vocab,
        "moves": moves,
        "config": model_config(),
    }, CKPT_DIR / path)
    print(f"Model saved to {CKPT_DIR / path}")


def train(m, opt, start_step, num_iters, name):
    n_params = sum(p.numel() for p in m.parameters())
    print(f"Total parameters: {n_params/1e6:.2f}M, batch {batch_size}, bf16 {use_bf16}")

    batches = make_train_loader()
    m.train()
    best_val = float("inf")
    train_losses = []          # train move loss = average over the steps since the last eval (no extra batches)
    legal_losses, illegal_mass = [], []   # legal-head loss, and probability the MOVE head puts on illegal moves
    t0 = time.time()
    for step in range(start_step, start_step + num_iters):
        x, y, legal = next(batches)
        x, y, legal = x.to(device, non_blocking=True), y.to(device, non_blocking=True), legal.to(device, non_blocking=True)
        # (B, MAX_LEGAL) padded id lists -> (B, 1968) 0/1 targets. Column n_moves catches the padding and is dropped.
        legal_targets = torch.zeros(len(y), n_moves + 1, device=device).scatter_(1, legal, 1.0)[:, :n_moves]

        lr = get_lr(step)
        for group in opt.param_groups:
            group["lr"] = lr

        with autocast():
            logits, loss = m(x, y, legal_targets)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        train_losses.append(m.move_loss.item())
        legal_losses.append(m.legal_loss.item())
        with torch.no_grad():   # how much of the move head's probability is on illegal moves (lower = better)
            illegal_mass.append(1 - (F.softmax(logits.float(), -1) * legal_targets).sum(-1).mean().item())

        if step % eval_interval == 0:
            val_loss, val_acc = evaluate_val()
            dt = time.time() - t0
            print(f"Loss at step {step}: train {np.mean(train_losses):.4f}, val {val_loss:.4f}, "
                  f"val acc {val_acc:.1%}, legal loss {np.mean(legal_losses):.3f}, "
                  f"illegal mass {np.mean(illegal_mass):.1%}, lr {lr:.2e}, "
                  f"{len(train_losses) * batch_size / dt:,.0f} positions/s", flush=True)
            train_losses, legal_losses, illegal_mass, t0 = [], [], [], time.time()
            save_checkpoint(m, opt, step, f"{name}_latest.pt")
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(m, opt, step, f"{name}_best.pt")

    m.eval()
    board = m.generate(chess.Board())
    print(chess.Board().variation_san(board.move_stack))
    print("Result:", board.result(claim_draw=True), f"after {len(board.move_stack)} plies")
    m.train()

    save_checkpoint(m, opt, start_step + num_iters, f"{name}.pt")
    return m


def train_from_scratch(num_iters=max_iters):
    global m
    m = Transormer(n_emb).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=learning_rate, betas=betas)
    return train(m, opt, 0, num_iters, NAME)


def train_from_checkpoint(path, num_iters=None, decay=False):
    """decay=False: continue at the constant lr. decay=True: the final decay, ~15% of the steps done so far."""
    global m, decay_start, decay_iters
    ckpt = torch.load(CKPT_DIR / path, map_location=device, weights_only=True)   # names are relative to inference/
    if ckpt["config"] != model_config():
        raise ValueError(f"Checkpoint config {ckpt['config']} does not match script config {model_config()}")
    if ckpt["input_vocab"] != input_vocab or ckpt["moves"] != moves:
        raise ValueError("Checkpoint vocabulary does not match this script")

    m = Transormer(n_emb).to(device)
    m.load_state_dict(ckpt["model_state_dict"])
    opt = torch.optim.AdamW(m.parameters(), lr=learning_rate, betas=betas)
    opt.load_state_dict(ckpt["optimizer_state_dict"])
    start_step = ckpt["step"] + 1

    if decay:
        decay_start = start_step
        decay_iters = num_iters or max(5000, int(0.15 * start_step))
        print(f"Final decay from {path}: steps {start_step} -> {start_step + decay_iters}, lr {learning_rate} -> {min_lr}")
        return train(m, opt, start_step, decay_iters, f"{NAME}_decayed")

    if num_iters is None:
        num_iters = max_iters - start_step
    print(f"Resuming from {path} at step {start_step}, constant lr {learning_rate}")
    return train(m, opt, start_step, num_iters, NAME)


if __name__ == "__main__":
    if not (VAL_DIR / "boards_flip_hist8_val.bin").exists():
        build_val()
    val_data = load_val()
    print(f"val positions: {len(val_data[1]):,}")

    # 150k steps: 130k at constant lr, then 20k of decay (~15% of the run, ends at min_lr like cosine).
    train_from_scratch(130000)
    train_from_checkpoint(f"{NAME}.pt", num_iters=20000, decay=True)   # {NAME}.pt = the model at step 130000
    # Stopped early? Run only the decay from the last save: train_from_checkpoint(f"{NAME}_latest.pt", decay=True)



# legal 16M, final (after decay)                       │ 1.689(train)    │ 1.695 (val)     │ 49.9%(val accuracy) │ 86.9% │