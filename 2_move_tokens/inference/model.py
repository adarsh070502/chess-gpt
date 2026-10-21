"""
Model classes copied from chess_gpt_move_tokens.py, plus checkpoint loading.

Hyperparameters are read from the checkpoint, so they always match training.
The checkpoint is chosen as:
  1. the CHESS_CKPT environment variable, if set
  2. otherwise the newest checkpoint (highest step) named chess_through_moves*.pt found
     in this folder (the training script saves its checkpoints here)
"""
import os
import re
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
HERE = Path(__file__).parent


def find_checkpoint():
    if os.environ.get("CHESS_CKPT"):
        return Path(os.environ["CHESS_CKPT"])
    found = []
    for folder in {HERE}:
        found += folder.glob("chess_through_moves*.pt")
    if not found:
        raise FileNotFoundError("No chess_through_moves*.pt found; set CHESS_CKPT to the checkpoint path")

    def step(p):   # chess_through_moves.pt (final) counts as newest
        m = re.search(r"_(\d+)\.pt$", p.name)
        return int(m.group(1)) if m else float("inf")
    return max(found, key=step)


CKPT_PATH = HERE / "chess_through_moves_resumed_47500.pt" #find_checkpoint()
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=True)
_cfg = ckpt["config"]

block_size = _cfg["block_size"]
n_emb = _cfg["n_emb"]
num_heads = _cfg["num_heads"]
n_layer = _cfg["n_layer"]
dropout = _cfg["dropout"]
vocab_size = _cfg["vocab_size"]


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()

        self.query = nn.Linear(n_emb, head_size)
        self.keys = nn.Linear(n_emb, head_size)
        self.values = nn.Linear(n_emb, head_size)
        self.head_size = head_size

        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape

        q = self.query(x)
        k = self.keys(x)

        wei = q @ k.transpose(-2,-1) * self.head_size**-0.5   # same scaling as in training
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
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
        self.linear = nn.Linear(n_emb, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_emb = self.embedding_layer(idx)
        positional_emb = self.positional_embeddings(torch.arange(T, device=device))

        x = token_emb + positional_emb

        attention = self.blocks(x)

        x = self.ln_f(attention)
        logits = self.linear(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape

            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets)
        return logits, loss
