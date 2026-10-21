import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

# Hyper parameters
max_iters = 50000
batch_size = 64     # games per batch (each batch = complete games, padded)
block_size = 256    # longest game (in tokens) the model can take; longer games are dropped
eval_interval = 500
learning_rate = 3e-4   # peak, reached after warmup
min_lr = 3e-5          # end of the cosine decay (10% of peak)
warmup_iters = 1000    # linear ramp 0 -> learning_rate
n_emb = 384
num_heads = 6
n_layer = 8 # no of multi attention and feedforward blocks
dropout = 0.0
eval_iters = 100 # used to decide on how many batches should the loss be calculatd on

device = 'cuda' if torch.cuda.is_available() else 'cpu' # Don't run on CPU if possible (it's slow. really.)

IGNORE = -100        # F.cross_entropy skips targets with this value (its default ignore_index)
N_CONDITION = 3      # result, white elo, black elo: given to the model, not something it should predict


def get_batch(split, batch_size):
    """batch_size complete games, each starting at position 0, padded to the longest game in the batch.

    Example (P = <pad>), one game:
        x:  <start> <white_wins> <elo_1400> <elo_1600> e2e4 e7e5 <end>  P    P
        y:   -100      -100        -100       e2e4     e7e5 <end> -100 -100 -100
    """
    starts, lengths = (train_starts, train_lengths) if split == "train" else (val_starts, val_lengths)

    ix = torch.randint(0, len(starts), (batch_size,))
    T = lengths[ix].max().item() - 1        # x and y are the game shifted by one, so one shorter

    x = torch.full((batch_size, T), pad_id, dtype=torch.long)
    y = torch.full((batch_size, T), IGNORE, dtype=torch.long)
    for row, i in enumerate(ix.tolist()):
        s, L = starts[i].item(), lengths[i].item()
        game = torch.from_numpy(data[s:s+L].astype(np.int64))   # reads just this game from games.bin
        x[row, :L-1] = game[:-1]
        y[row, :L-1] = game[1:]               # positions after the game stay IGNORE (padding)
    y[:, :N_CONDITION] = IGNORE               # don't train the model to guess result / ratings

    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad
def evaluate_loss():

    out = {}
    m.eval()

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):
            x, y = get_batch(split, batch_size)
            logits, loss = m(x, y)

            losses[k] = loss

        out[split] = losses.mean()
    m.train()
    return out


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

        q = self.query(x) # x = (B,T,n_emb), self.query = (n_emb, head_size), q = (B, T, head_size)
        k = self.keys(x) # x = (B,T,n_emb), self.keys = (n_emb, head_size), k = (B, T, head_size)

        wei = q @ k.transpose(-2,-1) * self.head_size**-0.5 # q = (B, T, head_size), k.T = (B, head_size, T), wei = (B, T, T) where B=batch_size, T=no of examples
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
            if ix.item() == stoi["<end>"]:   # game over; anything after this would be garbage
                break
        return idx



# Reading input: token ids written by data_with_player_rating_start_token.ipynb
HERE = Path(__file__).parent
DATA = HERE.parent / "data"          # shared data: games.bin, vocab.json
CKPT_DIR = HERE / "inference"       # checkpoints are saved here, next to the inference code that loads them
with open(DATA / "vocab.json") as f:
    vocab = json.load(f)             # fixed vocab: every possible UCI move + special tokens

stoi = {k:i for i, k in enumerate(vocab)}
itos = {i:k for k, i in stoi.items()}
vocab_size = len(itos)
pad_id = stoi["<pad>"]

# Encode and Decode functions
encode = lambda t: [stoi[c] for c in t]
decode = lambda idxs: " ".join([itos[i] for i in idxs])

# All games back to back, one uint16 token id per move / special token.
# np.memmap reads from disk only the parts that are used, instead of loading the whole file into RAM.
data = np.memmap(DATA / "games.bin", dtype=np.uint16, mode="r")

# Index of games: where each one starts in data and how many tokens it has
starts = torch.from_numpy(np.flatnonzero(data == stoi["<start>"]))
lengths = torch.diff(starts, append=torch.tensor([len(data)]))

# A game of L tokens gives an input of L-1 tokens, which must fit in block_size
keep = lengths <= block_size + 1
print(f"games: {len(starts):,}, dropped (longer than {block_size + 1} tokens): {(~keep).sum().item():,}")
starts, lengths = starts[keep], lengths[keep]

# train and validation splits, by game (no game is cut in half)
n = int(0.9 * len(starts))
train_starts, train_lengths = starts[:n], lengths[:n]
val_starts, val_lengths = starts[n:], lengths[n:]

#get batch
# x, y = get_batch("train", batch_size)
# # print(f"x, y shapes respectively: {x.shape}, {y.shape}")
# # print(len(y[0]))

# # for b in range(batch_size):
# #     prev = []
# #     for t in range(block_size):
# #         prev.append(x[b,t].item())
# #         print(f"Input is {prev}")
# #         print(f"Output is {y[b,t].item()}")

# # calling model
# m = Transormer(n_emb)
# m = m.to(device)
# logits, loss = m(x, y)
# print(loss)


# # Generating text
# idx = torch.zeros((1,1), dtype=torch.long, device=device)
# tokens = m.generate(idx, 100)
# print(tokens.shape)
# print(decode(tokens[0].tolist()))







# training the model
def get_lr(step):
    """Linear warmup to learning_rate over warmup_iters, then cosine decay to min_lr at max_iters."""
    if step < warmup_iters:
        return learning_rate * (step + 1) / warmup_iters
    progress = min(1.0, (step - warmup_iters) / (max_iters - warmup_iters))   # 0 -> 1
    return min_lr + 0.5 * (learning_rate - min_lr) * (1 + math.cos(math.pi * progress))


def save_checkpoint(m, opt, step, path):
    # Save everything needed for inference, plus optimizer + step so training can resume
    torch.save({
        "model_state_dict": m.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "step": step,
        "stoi": stoi,
        "itos": itos,
        "config": {
            "n_emb": n_emb, "num_heads": num_heads, "n_layer": n_layer,
            "block_size": block_size, "vocab_size": vocab_size, "dropout": dropout,
        },
    }, CKPT_DIR / path)
    print(f"Model saved to {CKPT_DIR / path}")


def train(m, opt, start_step, num_iters, name):
    # Total parameters
    n_params = sum(p.numel() for p in m.parameters())
    print(f"Total parameters: {n_params/1e6:.2f}M")

    m.train()
    losses = []
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
            print(f"Loss at step {step}: train {out['train']:.4f}, val {out['val']:.4f}, lr {lr:.2e}")
            save_checkpoint(m, opt, step, f"{name}_{step}.pt")

    # plt.plot(losses);

    # Generating text
    prompt = ["<start>", "<white_wins>", "<elo_2000>", "<elo_2000>"]
    idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    tokens = m.generate(idx, block_size - len(prompt))
    print(tokens.shape)
    print(decode(tokens[0].tolist()))

    save_checkpoint(m, opt, start_step + num_iters, f"{name}.pt")
    return m


def train_from_scratch(num_iters=max_iters):
    global m   # evaluate_loss() uses the global m
    m = Transormer(n_emb).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=learning_rate)
    return train(m, opt, 0, num_iters, "chess_through_moves")


def train_from_checkpoint(path, num_iters=None):
    global m   # evaluate_loss() uses the global m
    ckpt = torch.load(CKPT_DIR / path, map_location=device, weights_only=True)   # names are relative to inference/

    # The checkpoint must have the same architecture and vocabulary as this script
    config = {"n_emb": n_emb, "num_heads": num_heads, "n_layer": n_layer,
              "block_size": block_size, "vocab_size": vocab_size, "dropout": dropout}
    if ckpt["config"] != config:
        raise ValueError(f"Checkpoint config {ckpt['config']} does not match script config {config}")
    if ckpt["stoi"] != stoi:
        raise ValueError("Checkpoint vocabulary does not match the training text")

    m = Transormer(n_emb).to(device)
    m.load_state_dict(ckpt["model_state_dict"])
    opt = torch.optim.AdamW(m.parameters(), lr=learning_rate)
    if "optimizer_state_dict" in ckpt:   # checkpoints without it only have the weights
        opt.load_state_dict(ckpt["optimizer_state_dict"])
    start_step = ckpt.get("step", 0)
    if num_iters is None:              # by default, finish the run: stop at max_iters, where the lr schedule ends
        num_iters = max_iters - start_step
    print(f"Resuming from {path} at step {start_step}")

    return train(m, opt, start_step, num_iters, "chess_through_moves_resumed")

 
if __name__ == "__main__":
    # train_from_scratch()   # old checkpoints use the old data-derived vocab and won't load
    train_from_checkpoint("chess_through_moves_26500.pt")



#  python chess_gpt/2_move_tokens/chess_gpt_move_tokens.py 
# games: 1,879,980, dropped (longer than 257 tokens): 104
# Total parameters: 8.98M
# Model saved to chess_through_moves_47500.pt
# Loss at step 48000: train 2.2035, val 2.2041




# games: 3,001,840, dropped (longer than 257 tokens): 207
# Resuming from chess_through_moves_26500.pt at step 26500
# Total parameters: 15.82M
# Loss at step 47500: train 2.0396, val 2.0758, lr 3.17e-05
# Model saved to chess_through_moves_resumed_47500.pt