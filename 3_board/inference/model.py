"""
Model classes copied from chess_gpt_board_embeddings.py, plus checkpoint loading and board encoding.

Hyperparameters and both vocabs are read from the checkpoint, so they always match training.
The checkpoint is chosen as:
  1. the CHESS_BOARD_CKPT environment variable, if set
  2. otherwise chess_board_best.pt (lowest val loss during training), then chess_board.pt (final),
     then chess_board_latest.pt, looked for in this folder (the training scripts save their checkpoints here)
"""
import os
from pathlib import Path

import chess
import torch
import torch.nn as nn
from torch.nn import functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
HERE = Path(__file__).parent


def find_checkpoint():
    if os.environ.get("CHESS_BOARD_CKPT"):
        p = Path(os.environ["CHESS_BOARD_CKPT"])
        return p if p.is_file() else HERE / p   # a bare file name means: in this folder
    for name in ["chess_board_30M_best.pt"]: #, "chess_board.pt", "chess_board_latest.pt"]:
        for folder in [HERE]:
            if (folder / name).is_file():
                return folder / name
    raise FileNotFoundError("No chess_board*.pt found; set CHESS_BOARD_CKPT to the checkpoint path")


CKPT_PATH = find_checkpoint()
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=True)
_cfg = ckpt["config"]

# Same names as in the training script, because the classes below read them as globals
block_size = _cfg["block_size"]     # 73 tokens per board
n_emb = _cfg["n_emb"]
num_heads = _cfg["num_heads"]
n_layer = _cfg["n_layer"]
dropout = _cfg["dropout"]
vocab_size = _cfg["vocab_size"]     # 44 input tokens (squares, side to move, castling, en passant, result, ratings)
n_moves = _cfg["n_moves"]           # 1968 output moves

# Both vocabs were saved in the checkpoint, so the ids here are exactly the ids used in training.
input_vocab = ckpt["input_vocab"]                       # board tokens, id = index
stoi = {t: i for i, t in enumerate(input_vocab)}        # board token -> id
moves = ckpt["moves"]                                   # output id -> UCI move string
move_to_id = {mv: i for i, mv in enumerate(moves)}      # UCI move string -> output id


def encode_board(board, result, white_elo_tok, black_elo_tok):
    """python-chess board -> 73 token ids. Must be identical to encode_board in the training script,
    otherwise the model would see boards in a layout it never trained on.

    Layout: [side to move, 4 castling rights, en passant file, result, White elo, Black elo, 64 squares a8..h1]
    """
    # Only mark en passant when a capture is really possible (python-chess sets ep_square after
    # every double pawn push, even if no enemy pawn is next to it).
    ep = board.ep_square if board.has_legal_en_passant() else None
    tokens = [
        "<white>" if board.turn == chess.WHITE else "<black>",                     # whose move it is
        "<K>" if board.has_kingside_castling_rights(chess.WHITE) else "<noK>",     # White O-O allowed?
        "<Q>" if board.has_queenside_castling_rights(chess.WHITE) else "<noQ>",    # White O-O-O allowed?
        "<k>" if board.has_kingside_castling_rights(chess.BLACK) else "<nok>",     # Black O-O allowed?
        "<q>" if board.has_queenside_castling_rights(chess.BLACK) else "<noq>",    # Black O-O-O allowed?
        "<ep_none>" if ep is None else f"<ep_{chess.FILE_NAMES[chess.square_file(ep)]}>",
        result, white_elo_tok, black_elo_tok,                                      # what kind of game to imitate
    ]
    # 64 squares, rank 8 to rank 1, file a to h ("." = empty, uppercase = White, lowercase = Black)
    for rank in range(7, -1, -1):
        for file in range(8):
            piece = board.piece_at(chess.square(file, rank))
            tokens.append(piece.symbol() if piece else ".")
    return [stoi[t] for t in tokens]


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()

        self.query = nn.Linear(n_emb, head_size)
        self.keys = nn.Linear(n_emb, head_size)
        self.values = nn.Linear(n_emb, head_size)
        self.head_size = head_size
        # no tril buffer: every square may look at every other square

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape

        q = self.query(x)
        k = self.keys(x)

        wei = q @ k.transpose(-2,-1) * self.head_size**-0.5
        wei = F.softmax(wei, dim=-1)   # no causal mask
        wei = self.dropout(wei)

        v = self.values(x)

        out = wei @ v
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
        positional_emb = self.positional_embeddings(torch.arange(T, device=device))

        x = token_emb + positional_emb

        attention = self.blocks(x)

        x = self.ln_f(attention[:, 0, :])   # the move is read at position 0 (side to move)
        logits = self.linear(x)             # (B, n_moves)

        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits, targets)
        return logits, loss
