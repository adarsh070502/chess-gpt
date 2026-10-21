# Plan: chess model with board input (instead of move history)

Steps to follow and code yourself. Compare the result with the move-history model in
`MOVE_TOKENS_V3_NOTES.md`.

## The idea in one picture

```
Current model:  <start> <result> <elo> <elo> e2e4 e7e5 g1f3 ...  → next move
                (the model has to rebuild the board from the move list)

Board model:    [side to move, castling, en passant] + [64 squares]  → next move
                (python-chess plays the moves; the model only sees the finished board)
```

- x = the board **before** a move, y = the move that was played on it.
- Input vocab: ~13 square tokens + a few extra tokens. Output: one of the **1,968** UCI moves.
- Same transformer blocks as Andrej's GPT, but **no causal mask** and **one prediction per board**.

---

## Step 1: Decide the board tokens (input vocab)

**Squares (13 tokens):**

| Token | Meaning |
|---|---|
| `.` | empty |
| `P N B R Q K` | White pieces (uppercase = White, the FEN standard) |
| `p n b r q k` | Black pieces |

**Extra tokens (the board alone doesn't contain this):**

| Info | Tokens | Why |
|---|---|---|
| Side to move | `<white>` / `<black>` | **Essential.** The same board can have either side to move |
| Castling rights | `<K>` `<Q>` `<k>` `<q>`, each with a "no" version (8 tokens) | Can't be seen from the pieces (the king may have moved and come back) |
| En passant | `<ep_none>`, `<ep_a>` … `<ep_h>` (9 tokens) | Whether a pawn can be taken en passant right now |
| *Optional:* ratings | `<elo_800>` … `<elo_2400>` (9 tokens) | Keeps the rating conditioning from the current model |
| *Optional:* result | `<white_wins>` `<black_wins>` `<draw>` | Keeps the result conditioning |

**Decisions to make and write down:**
- [ ] Include ratings and result, or start without them? (Without is simpler; add later.)
- [ ] Exact order of the extra tokens. It must **always be the same order**.

## Step 2: Decide the square order

Positions must always mean the same square: position 0 is always one fixed square, and so on.

- Choose: **a8, b8 … h8, a7 … h1** (reading the board top-left to bottom-right, like FEN), or
  python-chess's own order **a1, b1 … h1, a2 … h8** (square index 0 = a1, 63 = h8).
- Either works. **Pick one and never change it.** The positional embeddings learn "I am square e4"
  from this order.

Full sequence for one board (example layout):
```
[side] [K/noK] [Q/noQ] [k/nok] [q/noq] [ep]   [64 squares]          → length 70
(+ 2 elo tokens + 1 result token if you add them → length 73)
```

**Check:** encode the starting position, print it as an 8×8 grid, and compare it with a real board.

## Step 3: Build the dataset (x = board, y = move)

Reuse from `data_with_player_rating_start_token.ipynb`: `read_pgn_games`, the filters
(no bullet, only `Normal` termination, rating, minimum length), and the multiprocessing pool.

Per game:
1. Start with a new python-chess board.
2. For each move in the game:
   - **x** = encode the board as it is now (Step 2)
   - **y** = the move's id in the **output vocab** (the 1,968 moves)
   - play the move on the board
3. Keep only some positions per game (see the size table below).

**Output vocab:** the 1,968 moves. `vocab.json` has the 15 special tokens first, so either:
- build a separate move list (`moves = vocab[15:]`, move id = index in that list), or
- use the `vocab.json` ids and an output layer of size 1,983, where the 15 special tokens are simply never targets.

**Size, why you can't keep every position:**

| | Every position | 10 per game |
|---|---|---|
| Positions | ~190M (3M games × ~63 moves) | **~30M** |
| x on disk (uint16, length 70) | **~27 GB** ❌ (18 GB free) | **~4.2 GB** ✅ |
| y on disk | 380 MB | 60 MB |

Recommendations:
- [ ] Sample about **10 positions per game**, at random.
- [ ] Skip or sample less from the **first few moves**: the starting position and common openings
  repeat millions of times.
- [ ] Split train/val **by game, before sampling**, as in the current model. Positions from the same
  game must not end up in both train and val, because neighbouring positions are almost identical.

**Files:**
- `boards.bin`: uint16, shape `(N, 70)` (read back with `np.memmap(...).reshape(-1, 70)`)
- `moves.bin`: uint16, shape `(N,)`
- `board_vocab.json`: input vocab; reuse the move list for the output vocab
- a record of where the val part starts (or two separate files)

**Checks:**
- [ ] Decode 3 random rows: print the board as 8×8 and the move. Is the move legal on that board?
  (`chess.Move.from_uci(move) in board.legal_moves` after rebuilding the board from the tokens.)
- [ ] Count how often each side to move appears (should be ~50/50).
- [ ] Count the most frequent positions; check the starting position isn't a big share.

## Step 4: Batch

Easier than the current model: every example has the same length, so **no padding**.

```
ix = random row numbers                         (B,)
x  = boards[ix]   → (B, 70)  token ids
y  = moves[ix]    → (B,)     move ids
```

Batch size can be much larger than 64 games. Each example is only 70 tokens, so try 256-512.

## Step 5: Embeddings

```
token embedding:      nn.Embedding(input_vocab_size, n_emb)   ~40 × n_emb
positional embedding: nn.Embedding(70, n_emb)                 one row per position/square
x = tok_emb(x) + pos_emb(arange(70))           → (B, 70, n_emb)
```

## Step 6: Transformer blocks with no mask

- Same `Head`, `MultiHeadAttention`, `FeedForward`, `Block` as now.
- **Remove the causal mask** (`masked_fill` with `tril`). Every square must see every other square;
  a board has no "future".
- Keep the attention scaling `* head_size**-0.5`.
- After the blocks: `(B, 70, n_emb)`.

## Step 7: Output, one move per board

```
pick one position, e.g. position 0 (the side-to-move token)   → (B, n_emb)
LayerNorm
Linear(n_emb, 1968)                                              → logits (B, 1968)
loss = F.cross_entropy(logits, y)                                y: (B,)
```

Without the mask every position has seen the whole board, so any single position works. Using the
first token is the simplest (like BERT's `[CLS]` token).

**Check:** loss at step 0 ≈ ln(1968) ≈ **7.58** (uniform guessing).

## Step 8: Train

- Same AdamW + warmup + cosine schedule as the current model.
- Start with the same size as the current model (`n_emb=384, num_heads=6, n_layer=8`, ~15M params)
  so the comparison is fair.
- Evaluate on val every 500 steps, like now.
- **Memory:** sequences are 70 long instead of up to 256, so attention memory is much smaller.

## Step 9: Play (inference)

```
board = python-chess board
loop:
    x = encode(board)                → (1, 70)
    logits = model(x)                → (1, 1968)
    set logits of illegal moves to -inf (board.legal_moves)
    sample (or take the top move)
    board.push(move)
```

python-chess keeps track of the board, so the model never has to.

## Step 10: Compare with the move-history model

Use the **same val games** for both models.

| Metric | Move-history model | Board model |
|---|---|---|
| Move-match accuracy (top move = human move) | ___ | ___ |
| Top-5 accuracy | ___ | ___ |
| Probability on illegal moves (no masking), by move number 1-10 / 11-20 / 21-30 / 31+ | ___ | ___ |
| Legal-move rate when sampling without masking | ___ | ___ |
| Accuracy by rating bucket (if ratings included) | ___ | ___ |

Expected: the board model should have far less probability on illegal moves, especially late in
the game (the move model reached 45.7% on moves 10-18 in the test game).

**Note on comparing loss:** the move model's loss averages over every move of a game; the board
model's loss is per sampled position. Compare with **accuracy** on the same positions, not loss.

---

## Possible later improvements

- **Flip the board for Black** so the side to move is always "at the bottom". The model then learns
  one set of patterns instead of two. Leela Chess Zero does this.
- **Add the last few moves** as extra tokens: this gives the model some history (repetitions, style).
- **Bigger model**: attention is cheap at length 70, so there is room on the RTX 3050.
