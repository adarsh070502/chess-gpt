"""
Model classes copied from chess_gpt_board_fast_allpos_flip_hist8_16M.py (and _60M.py), plus checkpoint
loading, the 81-token encoding and one helper, move_probs(), that turns a board into move probabilities.

Hyperparameters and both vocabs are read from the checkpoint, so they always match training
(the same file works for the 16M and the 60M checkpoints).
The checkpoint is chosen as:
  1. the CHESS_BOARD_CKPT environment variable, if set
  2. otherwise the first file found from CHECKPOINT_NAMES, in this folder (the training scripts save their checkpoints here)

Differences from 3_board/inference/model.py (old 73-token board model):
  - input: 81 tokens, board seen from the side to move (mirrored when Black is to move) + the last 8 moves
  - output: when the board was mirrored, the model's move is in the mirrored board; move_probs() maps it back
  - attention: one fused qkv Linear + F.scaled_dot_product_attention (same maths, different weight names)
"""
import os
from pathlib import Path

import chess
import torch
import torch.nn as nn
from torch.nn import functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
HERE = Path(__file__).parent

CHECKPOINT_NAMES = [
    "chess_board_fast_allpos_flip_hist8_16M_decayed_best.pt"
    # "chess_board_fast_allpos_flip_hist8_16M_decayed.pt",   # end of the run (after the lr decay)
    # "chess_board_fast_allpos_flip_hist8_16M_best.pt",      # lowest val loss so far
    # "chess_board_fast_allpos_flip_hist8_16M_latest.pt",
]


def find_checkpoint():
    if os.environ.get("CHESS_BOARD_CKPT"):
        p = Path(os.environ["CHESS_BOARD_CKPT"])
        return p if p.is_file() else HERE / p   # a bare file name means: in this folder
    for name in CHECKPOINT_NAMES:
        for folder in [HERE]:
            if (folder / name).is_file():
                return folder / name
    raise FileNotFoundError("No chess_board_fast_allpos_flip_hist8_*.pt found; set CHESS_BOARD_CKPT to the checkpoint path")


CKPT_PATH = find_checkpoint()
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=True)
_cfg = ckpt["config"]
if "legal" in _cfg.get("format", ""):
    raise ValueError(f"{CKPT_PATH} has a legal-move head; use 5_board_flip_hist8_legal/inference/ for it")
if "flip" not in _cfg.get("format", ""):
    raise ValueError(f"{CKPT_PATH} is an old 73-token board checkpoint; use 3_board/inference/ for it")

# Same names as in the training script, because the classes below read them as globals
block_size = _cfg["block_size"]     # 81 tokens per position
n_emb = _cfg["n_emb"]
num_heads = _cfg["num_heads"]
n_layer = _cfg["n_layer"]
dropout = _cfg["dropout"]
vocab_size = _cfg["vocab_size"]     # 2,012 input tokens: 44 board/state tokens + <no_move> + 1,968 history moves
n_moves = _cfg["n_moves"]           # 1,968 output moves

# Both vocabs were saved in the checkpoint, so the ids here are exactly the ids used in training.
input_vocab = ckpt["input_vocab"]
stoi = {t: i for i, t in enumerate(input_vocab)}
moves = ckpt["moves"]                                   # output id -> UCI move string
move_to_id = {mv: i for i, mv in enumerate(moves)}      # UCI move string -> output id
MOVE_OFFSET = stoi[moves[0]]                            # input id of a history move = MOVE_OFFSET + its output id
HISTORY = 8

# The game result as written in training data (from White's side) -> result for the side to move.
RESULT_FOR_MOVER = {
    "<white_wins>": {chess.WHITE: "<me_wins>", chess.BLACK: "<me_loses>"},
    "<black_wins>": {chess.WHITE: "<me_loses>", chess.BLACK: "<me_wins>"},
    "<draw>": {chess.WHITE: "<draw>", chess.BLACK: "<draw>"},
}


def mirror_move(move):
    """The same move in the mirrored board: e7e5 -> e2e4, e8g8 (O-O) -> e1g1. Promotion piece unchanged."""
    return chess.Move(chess.square_mirror(move.from_square), chess.square_mirror(move.to_square), move.promotion)


# MIRROR[i] = output id of move i mirrored. Used to turn the model's answer (in the mirrored board)
# back into real moves: real_probs = probs[MIRROR]. Mirroring twice gives the move back, so one table is enough.
MIRROR = torch.tensor([move_to_id[mirror_move(chess.Move.from_uci(mv)).uci()] for mv in moves])


def encode_position(board, history, result, white_elo, black_elo):
    """python-chess board + previous moves -> (81 input ids, flip). Must be identical to the training script.

    Layout: [<cls>, 4 castling rights, en passant, result for me, my elo, opponent's elo, 64 squares, last 8 moves]
    flip = True when Black is to move: then everything is mirrored so the side to move is "White" (uppercase).
    """
    flip = board.turn == chess.BLACK
    mover = board.turn
    if flip:
        board = board.mirror()                          # ranks flipped, colours swapped, castling / ep / turn too
        history = [mirror_move(m) for m in history]
        white_elo, black_elo = black_elo, white_elo     # the mover's rating goes first
    my_elo, opp_elo = white_elo, black_elo

    ep = board.ep_square if board.has_legal_en_passant() else None
    tokens = [
        "<cls>",
        "<K>" if board.has_kingside_castling_rights(chess.WHITE) else "<noK>",     # my O-O allowed?
        "<Q>" if board.has_queenside_castling_rights(chess.WHITE) else "<noQ>",    # my O-O-O allowed?
        "<k>" if board.has_kingside_castling_rights(chess.BLACK) else "<nok>",     # opponent's O-O allowed?
        "<q>" if board.has_queenside_castling_rights(chess.BLACK) else "<noq>",    # opponent's O-O-O allowed?
        "<ep_none>" if ep is None else f"<ep_{chess.FILE_NAMES[chess.square_file(ep)]}>",
        RESULT_FOR_MOVER[result][mover],
        my_elo, opp_elo,
    ]
    piece_map = board.piece_map()
    for rank in range(7, -1, -1):                       # 64 squares, rank 8 to rank 1, file a to h
        for file in range(8):
            piece = piece_map.get(chess.square(file, rank))
            tokens.append(piece.symbol() if piece else ".")
    ids = [stoi[t] for t in tokens]

    last = history[-HISTORY:]                           # last 8 moves, oldest first, <no_move> padding in front
    ids += [stoi["<no_move>"]] * (HISTORY - len(last))
    ids += [MOVE_OFFSET + move_to_id[m.uci()] for m in last]
    return ids, flip


@torch.no_grad()
def move_probs(model, board, result, white_elo, black_elo, temperature=1.0):
    """(1968,) probabilities on the CPU, indexed by REAL moves: probs[move_to_id["e7e5"]] is the probability of e7e5,
    also when Black is to move. result / elos as in training data: "<white_wins>", "<elo_2000>", White's first."""
    ids, flip = encode_position(board, board.move_stack, result, white_elo, black_elo)
    idx = torch.tensor([ids], dtype=torch.long, device=next(model.parameters()).device)   # (1, 81)
    logits, _ = model(idx)                                                                 # (1, 1968)
    probs = F.softmax(logits[0].float() / temperature, dim=-1).cpu()
    # The model answered in the mirrored board: the real move m has the probability of mirror(m).
    return probs[MIRROR] if flip else probs


def load_model():
    model = Transormer(n_emb)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval()     # eval(): no dropout at inference


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
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2)   # (B, heads, T, head_size)
        k = k.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        # no mask (every square sees every square); default scale head_size**-0.5, as in training
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.proj(out))


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
        return self.net(x)


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
        self.embedding_layer = nn.Embedding(vocab_size, n_emb)
        self.positional_embeddings = nn.Embedding(block_size, n_emb)

        self.blocks = nn.Sequential(*[Block(n_emb, num_heads) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_emb)
        self.linear = nn.Linear(n_emb, n_moves)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_emb = self.embedding_layer(idx)
        positional_emb = self.positional_embeddings(torch.arange(T, device=idx.device))
        x = token_emb + positional_emb

        attention = self.blocks(x)
        x = self.ln_f(attention[:, 0, :])   # the move is read at position 0 (<cls>)
        logits = self.linear(x)             # (B, n_moves)

        loss = None if targets is None else F.cross_entropy(logits, targets)
        return logits, loss
