"""
Paste your model classes here, copied from your training script (chess_gpt.py).

Copy ONLY:
  - the hyperparameters below (values must be identical to training)
  - the class definitions: Head, MultiHeadAttention, FeedForward, Block, and the main model

Do NOT copy data loading, get_batch, the training loop, or any code that runs at
import time, otherwise the Space will try to load data / train on startup.

Requirements app.py relies on:
  - the main model class is named GPTLanguageModel (or change the import in app.py)
  - forward(idx, targets=None) returns (logits, loss), logits shaped (B, T, vocab_size)
"""
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- hyperparameters: read from the checkpoint, so they always match training ----
CKPT_PATH = Path(__file__).parent / "chess_character_level.pt"   # next to this file
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=True)
_cfg = ckpt["config"]

block_size = _cfg["block_size"]
n_emb = _cfg["n_emb"]
num_heads = _cfg["num_heads"]
n_layer = _cfg["n_layer"]
dropout = _cfg["dropout"]
vocab_size = _cfg["vocab_size"]

# ---- paste your classes below ----


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()

        self.query = nn.Linear(n_emb, head_size)
        self.keys = nn.Linear(n_emb, head_size)
        self.values = nn.Linear(n_emb, head_size)

        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape

        q = self.query(x) # x = (B,T,n_emb), self.query = (n_emb, head_size), q = (B, T, head_size)
        k = self.keys(x) # x = (B,T,n_emb), self.keys = (n_emb, head_size), k = (B, T, head_size)

        wei = q @ k.transpose(-2,-1)  # q = (B, T, head_size), k.T = (B, head_size, T), wei = (B, T, T) where B=batch_size, T=no of examples
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
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
        self.embedding_layer = nn.Embedding(vocab_size, n_emb)  #self.embedding_layer = (vocab_size, n_emb)
        self.positional_embeddings = nn.Embedding(block_size, n_emb) #self.positional_embeddings = (block_size, n_emb)

        # self.blocks = nn.Sequential(*[Block(n_emb, num_heads) for i in range(n_layer)])
        self.blocks = nn.Sequential(*[Block(n_emb, num_heads) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_emb)
        self.linear = nn.Linear(n_emb, vocab_size) # self.linear = (n_emb, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # print(f"T size: {T}")

        token_emb = self.embedding_layer(idx) # token_emb = (B, T, n_emb)
        positional_emb = self.positional_embeddings(torch.arange(T, device=device)) # positional_emb = (T, n_emb)

        x = token_emb + positional_emb # x = (B, T, n_emb)

        attention = self.blocks(x)

        x = self.ln_f(attention)
        logits = self.linear(x) # logits = (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape

            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_num_samples):
        for i in range(max_num_samples):
            idx_cond = idx[:, -block_size:]         
            logits, _= self(idx_cond)
            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)

            ix = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, ix), dim=1)
        return idx

