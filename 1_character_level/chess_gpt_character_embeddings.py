from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

# Hyper parameters
max_iters = 10000
batch_size = 12
block_size = 512
eval_interval = 500
learning_rate = 3e-4
n_emb = 256
num_heads = 4
n_layer = 10 # no of multi attention and feedforward blocks
dropout = 0.1
eval_iters = 200 # used to decide on how many batches should the loss be calculatd on

device = 'cuda' if torch.cuda.is_available() else 'cpu' # Don't run on CPU if possible (it's slow. really.)

def get_batch(split, batch_size):
    data = train_split if split=="train" else val_split

    ix = torch.randint(0, len(data)-block_size, (batch_size,))

    x = torch.stack([data[i:i+block_size] for i in ix])       # Stack all (ix holds batch_size many) sequences of this batch row-wise on top of each other to form a tensor
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])   # Same as x but shifted by one token
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



# Reading input
HERE = Path(__file__).parent
CKPT_DIR = HERE / "inference"       # checkpoints are saved here, next to the inference code that loads them
with open(HERE.parent / "data" / "games2.txt") as f:   # games2.txt (games as text) is not in data/ any more; recreate it to retrain
    text = f.read()

# vocab
chars = sorted(list(set(text)))
stoi = {k:i for i, k in enumerate(chars)}
itos = {i:k for k, i in stoi.items()}
vocab_size = len(itos)

# Encode and Decode functions
encode = lambda t: [stoi[c] for c in t]
decode = lambda idxs: "".join([itos[i] for i in idxs])

# Encoding input data
data = torch.tensor(encode(text), dtype=torch.long)

# train and validation splits
n = int(0.9 * len(data))
train_split = data[:n]
val_split = data[n:]

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

        logits, loss = m(x, y)

        loss.backward()
        opt.step()
        opt.zero_grad()

        if step % eval_interval == 0:
            out = evaluate_loss()
            print(f"Loss at step {step}: train {out['train']:.4f}, val {out['val']:.4f}")
            save_checkpoint(m, opt, step, f"{name}_{step}.pt")

    plt.plot(losses);

    # Generating text
    idx = torch.zeros((1,1), dtype=torch.long, device=device)
    tokens = m.generate(idx, 500)
    print(tokens.shape)
    print(decode(tokens[0].tolist()))

    save_checkpoint(m, opt, start_step + num_iters, f"{name}.pt")
    return m


def train_from_scratch(num_iters=max_iters):
    global m   # evaluate_loss() uses the global m
    m = Transormer(n_emb).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=learning_rate)
    return train(m, opt, 0, num_iters, "gpt_shakespeare")


def train_from_checkpoint(path, num_iters=max_iters):
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
    if "optimizer_state_dict" in ckpt:   # older checkpoints (e.g. chess.pt) only have the weights
        opt.load_state_dict(ckpt["optimizer_state_dict"])
    start_step = ckpt.get("step", 0)
    print(f"Resuming from {path} at step {start_step}")

    return train(m, opt, start_step, num_iters, "chess_resumed")

 
if __name__ == "__main__":
    # train_from_scratch()
    train_from_checkpoint("chess_character_level.pt")




# Character Level
# # Total parameters: 8.05M
# Total parameters: 8.05M
# Loss at step 0: train 0.6545, val 0.6531
# Model saved to chess_resumed_0.pt
# Loss at step 500: train 0.6490, val 0.6494
# Model saved to chess_resumed_500.pt
# Loss at step 1000: train 0.6417, val 0.6424
# Model saved to chess_resumed_1000.pt
# Loss at step 1500: train 0.6317, val 0.6374
# Model saved to chess_resumed_1500.pt
# Loss at step 2000: train 0.6289, val 0.6307
# Model saved to chess_resumed_2000.pt
# Loss at step 2500: train 0.6222, val 0.6246
# Model saved to chess_resumed_2500.pt
# Loss at step 3000: train 0.6160, val 0.6190
# Model saved to chess_resumed_3000.pt
# Loss at step 3500: train 0.6143, val 0.6143
# Model saved to chess_resumed_3500.pt
# Loss at step 4000: train 0.6089, val 0.6094
# Model saved to chess_resumed_4000.pt
# Loss at step 4500: train 0.6059, val 0.6054
# Model saved to chess_resumed_4500.pt
# Loss at step 5000: train 0.6001, val 0.6014
# Model saved to chess_resumed_5000.pt
# Loss at step 5500: train 0.5960, val 0.5958
# Model saved to chess_resumed_5500.pt
# Loss at step 6000: train 0.5939, val 0.5929
# Model saved to chess_resumed_6000.pt
# Loss at step 6500: train 0.5890, val 0.5916
# Model saved to chess_resumed_6500.pt
# Loss at step 7000: train 0.5870, val 0.5905
# Model saved to chess_resumed_7000.pt
# Loss at step 7500: train 0.5832, val 0.5851
# Model saved to chess_resumed_7500.pt
# Loss at step 8000: train 0.5801, val 0.5813
# Model saved to chess_resumed_8000.pt
# Loss at step 8500: train 0.5787, val 0.5795
# Model saved to chess_resumed_8500.pt
# Loss at step 9000: train 0.5750, val 0.5758
# Model saved to chess_resumed_9000.pt