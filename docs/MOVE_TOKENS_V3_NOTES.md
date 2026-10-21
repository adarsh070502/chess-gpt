# Chess GPT (move tokens), v3: cleaner data, bigger model, LR schedule

> **Folders moved (2026-10-08).** Paths below are from before the reorganisation. Now: move model in
> `chess_gpt/2_move_tokens/` (was `inference_move_tokens/`), 73-token board model in `3_board/` (was
> `inference_board/`), flip + history model in `4_board_flip_hist8/` (was `inference_board_flip_hist8/`), each
> with its checkpoints in its `inference/` folder; shared data in `data/`. Map: `chess_gpt/README.md`.

Follow-up to `INTERVIEW_NOTES.md` (v2: 2015-08 data, 9M model, val 2.46).
v3 rebuilt the data pipeline on a newer, larger Lichess month, then retrained a 9M and a 15.8M model.

Legend: **measured** = from my logs / files. *est.* = estimate, verify before quoting.

---

## Headline numbers

| | v2 (INTERVIEW_NOTES.md) | v3, 9M | v3, 15.8M |
|---|---|---|---|
| Data | 2015-08, 1.88M games | 2016-05, **3.00M games** | same |
| Tokens | ~135M (*est.*) | **224.5M** | same |
| Params | 8.98M | **8.98M** | **15.82M** |
| LR | constant 3e-4 | constant 3e-4 | **warmup 1k + cosine 3e-4 → 3e-5** |
| Val loss | 2.46 | **2.212** (step 49,500), best **2.204** (step 48,000) | **2.076** (step 47,500) |
| Perplexity e^val | ~11.7 | ~9.1 | **~8.0** |

Same data split, same step: the 15.8M model was **0.156 lower** at step 26,500 (2.223 vs 2.379) and
beat the 9M model's final loss by step 28,500.

---

## 1. Data pipeline v3 (`data_with_player_rating_start_token.ipynb`)

| What | Number |
|---|---|
| Source | `lichess_db_standard_rated_2016-05.pgn`, **5.77 GB**, **6,225,957 games** (measured) |
| Kept after filters | **3,001,840 games (48.2%)** |
| Total tokens | **224,502,640** |
| Output | `games.bin` (**449 MB**, uint16 token ids) + `vocab.json` |
| Train / val | 90 / 10 by game, in file (time) order |
| Dropped for length (> 257 tokens) | **207** (0.007%) |
| Game length (tokens) | median **68**, 90% **116**, 99% **161**, max 602 |
| Conversion speed | 100K games in **5 s** on 15 worker processes (test run) |

**Results:** white 1,467,484 (48.9%), black 1,387,070 (46.2%), draw 147,286 (4.9%).

**Rating tokens (white + black):**

| 800 | 1000 | 1200 | 1400 | 1600 | 1800 | 2000 | 2200 | 2400 |
|---|---|---|---|---|---|---|---|---|
| 9,326 | 109,305 | 497,710 | 1,248,424 | 1,772,487 | 1,464,252 | 676,942 | 188,790 | 36,444 |

1400-1800 is ~64% of rating tokens; 800 and 2400+ are thin, so conditioning at the extremes is weak.

### What changed vs v2, and why

1. **Dropped bullet games** (base + 40 × increment < 180 s, Lichess's own definition).
   Lichess rates bullet separately, so `<elo_1800>` meant different strengths in bullet vs blitz/classical,
   and bullet moves are mostly made under time pressure.
2. **Kept only `Termination == "Normal"`.** In a time forfeit the result token says who ran out of
   time, not who was winning on the board, which weakens the result token for ~1/3 of games.
3. **Fixed vocab: all 1,968 possible UCI moves + 15 special tokens = 1,983**, generated from chess
   geometry (queen-like + knight-like jumps + promotions) instead of from the data. Token ids no longer
   depend on which moves a month of data happens to contain, so `.bin` files from different months are
   compatible and no legal move is ever "not in vocab". (In the end all 1,968 moves appear in the data.)
4. **Binary token file + `np.memmap`** instead of `f.read()` + `split()`. Splitting ~224M tokens into
   Python strings would need ~25-30 GB RAM; the uint16 file is 449 MB and training starts in seconds.
5. **Multiprocessing** (16 cores): the old single-process conversion ran at ~1,400 games/s (measured);
   the pool version converted 100K games in 5 s (measured).

**Filter breakdown, first 100K games** (measured): kept 47.9%; time forfeit 32.3%, bullet 13.6%,
too short 5.8%, abandoned 0.4%.

---

## 2. Model + training (`chess_gpt_move_tokens.py`)

| | 9M run | 15.8M run |
|---|---|---|
| `n_emb / num_heads / n_layer` | 256 / 4 / 10 | **384 / 6 / 8** |
| Head size | 64 | 64 |
| `dropout` | 0.0 | 0.0 |
| `batch_size / block_size` | 64 games / 256 | same |
| Steps | 50,000 (≈ 1 pass over the data) | 50,000 (resumed at 26,500) |
| LR | constant 3e-4 | linear warmup 1,000 steps, cosine to 3e-5 at 50k |
| Hardware | RTX 3050 8 GB | RTX 3050, ~5.2 GB used by training |

**Sanity check:** step-0 loss 7.53 / 7.74 ≈ ln(1983) = 7.59 (uniform guessing).

### 9M run (constant LR), val loss (measured)

| step | 5k | 10k | 15k | 20k | 25k | 30k | 35k | 40k | 45k | 49.5k |
|---|---|---|---|---|---|---|---|---|---|---|
| train | 3.018 | 2.729 | 2.574 | 2.453 | 2.391 | 2.332 | 2.288 | - | 2.219 | 2.194 |
| val | 3.024 | 2.733 | 2.578 | 2.480 | 2.395 | 2.341 | 2.295 | - | 2.228 | 2.212 |

The last 2K steps bounced between val 2.204 and 2.212: with a constant LR it can't settle, which is
what motivated the cosine schedule.

### 15.8M run (warmup + cosine), val loss (measured)

| step | 26.5k | 27k | 28k | 28.5k | 47.5k |
|---|---|---|---|---|---|
| train | 2.1998 | 2.2016 | 2.1815 | 2.1891 | **2.0396** |
| val | 2.2226 | 2.2197 | 2.2055 | 2.2038 | **2.0758** |
| lr | 1.56e-4 | 1.52e-4 | 1.43e-4 | 1.39e-4 | 3.17e-5 |

Train/val gap at 47.5k: **0.036** (vs ~0.015 mid-run). Most of the training data has been seen once by then.
TODO: fill in the final step-50,000 numbers.

### Bugs and decisions in this round

- **Inference app missing the attention scaling:** `inference_move_tokens/model.py` (a copy of the model
  classes) never got the `* head_size**-0.5` fix that training had. Every app game ran a broken model
  (section 3). Caught when comparing against the board model; fixed and checked that the app's model now
  gives the same logits as the training script's.

- **Missing attention scaling** (`q @ kᵀ` without `/ √head_size`): with head size 64 the scores were
  ~8× too large, so the softmax saturated. Same bug had made an 8M names model's loss *rise*
  from 2.07 to 2.35. Fixed before the v3 runs.
- **Stale training data:** `games_uci.txt` still came from 2015-08 while the notebook pointed at
  2016-05; caught by comparing the first game in the file with the PGN.
- **`generate()` didn't stop at `<end>`**, so samples printed garbage after the game ended. Fixed.
- **Resume overran the schedule:** resuming at 26,500 with `num_iters = max_iters` would have trained
  to 76,500 at min LR. The default is now `max_iters - start_step`.
- **Disk:** each 15.8M checkpoint with optimizer state is ~190 MB, so 100 of them are ~19 GB.
- **Chose not to:** use gradient clipping (no loss spikes appeared in either run without it), or go to 20M params
  (*est.* would OOM on long-game batches without fused attention / bf16, and 224M tokens / 20M params
  ≈ 11 tokens/param is below the ~20 Chinchilla ratio).

---

## 3. Play test in the Gradio app (`inference_move_tokens/app.py`)

> ⚠️ **Invalid numbers.** This game ran with a bug in `inference_move_tokens/model.py`: the attention
> scaling (`* head_size**-0.5`) was missing, although the model was trained with it, so the app ran a
> broken version of the model. The illegal-move numbers below are far too pessimistic. Fixed afterwards;
> the correct measurement is in section 6, "Move model vs board model on the same val positions".

Checkpoint `chess_through_moves_resumed_47500.pt` (15.8M, val 2.076). Model plays Black, mode
"Sample from legal moves" (illegal moves masked, then sample by probability), temperature 1.0.

"Illegal mass" = total probability the model put on moves that are illegal in the position.

| Move | Played | p | Illegal mass | Model's top move | Top move legal? |
|---|---|---|---|---|---|
| 1... | d5 | 26.5% | 0.0% | g8f6 41% | ✅ |
| 2... | Bf5 | 3.4% | 0.1% | e7e6 31% | ✅ |
| 3... | Nf6 | 17.4% | 1.0% | e7e6 62% | ✅ |
| 4... | e6 | 12.0% | 2.0% | f5d3 80% | ✅ |
| 5... | Bd6 | 8.1% | 16.4% | b8c6 20% | ✅ |
| 6... | fxe6 | 91.9% | 7.8% | f7e6 92% | ✅ |
| 7... | Nxd5 | 14.7% | 59.2% | e6f5 58% | ❌ |
| 8... | Ne7 | 11.3% | 10.5% | d5f6 74% | ✅ |
| 9... | O-O | 9.8% | 72.0% | e6f5 71% | ❌ |
| 10... | Bc5 | 11.5% | 20.7% | d6c5 11% | ✅ |
| 11... | Qd7 | 2.8% | 15.5% | d8d5 43% | ✅ |
| 12... | Rf5 | 29.5% | 2.3% | e7f5 50% | ✅ |
| 13... | Kf7 | 8.3% | 87.5% | c5f2 25% | ❌ |
| 14... | Nc6 | 25.5% | 25.2% | b8c6 25% | ✅ |
| 15... | Bd4 | 2.2% | 26.4% | f7g6 15% | ✅ |
| 16... | Ke6 | 1.6% | 74.8% | f6g8 7% | ❌ |
| 17... | Kf7 | 1.4% | 79.0% | e7c6 16% | ✅ |
| 18... | Nxd5 | 17.4% | 79.8% | f6d5 43% | ❌ |

**Summary (one game, 18 model moves; measured):**

| Metric | Value |
|---|---|
| Avg illegal mass, moves 1-4 (opening) | **0.8%** |
| Avg illegal mass, moves 5-9 | **33.2%** |
| Avg illegal mass, moves 10-18 | **45.7%** |
| Avg illegal mass, whole game | **32.2%** |
| Top-1 move legal | **13 / 18 (72%)** |

**Reading it:** the opening is clean (book moves, ~0% illegal mass). After exchanges the model's
internal picture of the board drifts. It kept proposing `e6f5` (moves 7, 9, 11) from a stale picture of
the e6/f5 squares, and by move 16-18 it was putting ~75-80% on illegal moves. The legal-move mask hides
this in play. This is the main weakness of a move-history-only model: it has to rebuild the board from
the move list.

Earlier game (checkpoint not recorded): at `6... Qxd1+` the model put **93.4%** on illegal moves, and
all of its top 5 were *White* moves (`g1f3`, `c1f4`, ...), so it had lost track of whose turn it was.

One game is not a benchmark. Before quoting these, measure over many positions (see TODO).

---

## 4. Resume bullet (draft)

> Built a 15.8M-parameter GPT from scratch (PyTorch) that learns chess only from move sequences, on
> 3.0M Lichess games (224M tokens) with rating and result conditioning. Built a multiprocess PGN→token
> pipeline (filters for bullet/timeouts, fixed 1,983-token move vocab, memmapped uint16 data).
> Fixed a missing attention-scaling bug, added warmup + cosine LR, and scaled 9M → 15.8M params,
> lowering val loss from 2.21 to 2.08 (perplexity ~8 over ~2K moves).

---

## 5. TODO: measure before quoting

| Metric | How | Value |
|---|---|---|
| Final val loss (step 50,000) | training log | ___ |
| Legal-move rate (raw sampling) | 500+ generated games, python-chess | ___ |
| Illegal mass by move number | val positions, bucket by ply (1-10, 11-20, 21-30, 31+) | ___ |
| Move-match accuracy | % of val positions where top move = human move, per rating bucket | ___ |
| Rating conditioning works? | same positions, `<elo_1200>` vs `<elo_2200>`: blunder rate via Stockfish | ___ |
| Compare 9M vs 15.8M | all of the above for both checkpoints | ___ |

Next experiment: **Option B, board-state input** (64 square tokens + side to move/castling/en passant,
encoder without causal mask, predict one move). Same data, same size, compare against this model
on the metrics above.

---

## 6. Board-input model (`chess_gpt_board_embeddings.py`, plan in `BOARD_INPUT_PLAN.md`)

The model sees only the current board, not the move history. python-chess replays the game and
keeps the board; the model predicts one move per board.

### Data (measured)

| What | Number |
|---|---|
| Source | `games.bin` (the same 3,001,633 games as the move model, same 90/10 split by game) |
| Positions per game | 10, chosen at random, seeded by game number (reproducible) |
| Positions | train **27,014,690** / val **3,001,640** |
| One example | 73 tokens: side to move, 4 castling rights, en passant file, result, 2 rating buckets, 64 squares (a8 → h1) |
| Input vocab / output vocab | **44** board tokens / **1,968** UCI moves |
| Files | `boards_{train,val}.bin` (uint16, (N, 73)), `moves_{train,val}.bin`; ~4.4 GB boards |
| Build time | **401 s** on 15 worker processes |

### Model differences from the move model
- Same `Head` / `MultiHeadAttention` / `FeedForward` / `Block` classes, but **no causal mask** (encoder).
- The move is read from **position 0** (side-to-move token): logits `(B, 1968)`, one target per board.
- `batch_size = 128` boards (128 × 73 ≈ 9.3K tokens/step, close to the move model), 50k steps, warmup 1k + cosine 3e-4 → 3e-5.
- GPU memory **~2 GB** (vs ~5 GB for the move model): fixed length 73, attention 73×73 instead of up to 256×256.

### Results: 16M vs 30M (measured on the same 200,000 val positions)

| Checkpoint | Params (`n_emb/heads/layers`) | Step | Val loss | Top-1 acc | Top-5 acc | Train time |
|---|---|---|---|---|---|---|
| `chess_board.pt` | **15.92M** (396/6/8) | 50,000 | **2.211** | **36.7%** | **74.1%** | ~4.5 h |
| `chess_board_best.pt` | 15.92M | 49,500 | 2.214 | 36.6% | 73.9% | |
| `chess_board_30M.pt` | 29.44M (512/8/9) | 50,000 | 2.277 | 35.1% | 72.3% | 7 h 41 min |
| `chess_board_30M_best.pt` | 29.44M | 49,000 | 2.282 | 35.0% | 72.1% | |

Top-1 = the model's favourite move is the move the human played. Top-5 = the human's move is in its top 5.

**Finding: the bigger model was worse at the same budget.** Both saw 50k × 128 = **6.4M positions (~24% of
the training set)**. The 30M model was behind the 16M at every step (val 2.577 vs ~2.49 at step 25k) and
never caught up. At this budget the limit is **examples seen, not model capacity**; a bigger model needs
more data/steps to pay off.

30M training log (from `chess_board_30M.log`, 100-batch estimates):

| step | 10k | 20k | 25k | 30k | 40k | 49k | 49.5k |
|---|---|---|---|---|---|---|---|
| train | 2.905 | 2.667 | 2.566 | 2.504 | 2.336 | 2.285 | 2.303 |
| val | 2.901 | 2.685 | 2.577 | 2.482 | 2.317 | 2.269 | 2.278 |
| val acc | 26.2% | 28.8% | 30.0% | 31.2% | 34.7% | 35.3% | 35.7% |

16M at mid-run (from the terminal): step 25,000 val 2.4915, acc 31.7%; step 24,500 val 2.4788, acc 32.3%.

Loss is **not comparable** with the move model's loss (that one averages over every move of every game;
here one sampled move per board). Compare by accuracy or play metrics on the same positions.

### Move model vs board model on the same val positions (measured)

1,500 val games (the same 90/10 split), the same 10 seeded positions per game = **15,000 positions**.
Both models run with the training scripts' code. Move model: `chess_through_moves_resumed_47500.pt`
(15.8M, step 47,500). Board model: `chess_board.pt` (15.9M, step 50,000).

| | Move model | Board model |
|---|---|---|
| **Top-1 accuracy** (top move = human move) | **40.5%** | 36.6% |
| **Top-5 accuracy** | **78.4%** | 74.2% |
| Avg probability on illegal moves | **8.6%** | 10.3% |
| Top move legal | 96.4% | 96.0% |

| Moves | Move top-1 | Board top-1 | Move top-5 | Board top-5 | Move illegal | Board illegal |
|---|---|---|---|---|---|---|
| 1-10 (n=4,945) | 49.0% | 46.8% | 88.0% | 86.4% | 0.6% | 1.6% |
| 11-20 (n=4,744) | 40.9% | 35.4% | 78.1% | 73.2% | 5.6% | 9.1% |
| 21-30 (n=2,906) | 32.6% | 28.3% | 70.4% | 62.8% | 14.7% | 16.6% |
| 31-40 (n=1,315) | 29.4% | 25.7% | 64.9% | 61.1% | 22.7% | 22.1% |
| 41+ (n=1,090) | 34.5% | 30.4% | 73.5% | 68.7% | 24.3% | 23.6% |

**Finding: at the same number of training steps, move history beats board input** on accuracy at every
stage, and the board input does **not** reduce illegal moves (equal from move 31 on, worse before).

**Why:** training signal. Per step the move model predicts every move of 64 games (~4,000 targets); the
board model predicts one move for each of 128 boards (128 targets). Over 50k steps that is ~200M vs
6.4M predictions, ~30× more for the move model. The move model also sees the history (plans, how the
game went), which helps predict what a human plays next. The board model's input is "easier", but not
enough to make up for 30× less signal.

### Play tests in the app (`inference_board/app.py`, model plays Black, "Sample from legal moves", T = 1.0)

Average probability on illegal moves, per stage of the game (measured, one game each):

| Moves | Move model (15.8M, step 47.5k), ⚠️ broken app, invalid | Board 16M (step 23k) | **Board 16M (step 49.5k)** | Board 30M (step 49k) |
|---|---|---|---|---|
| 1-4 | 0.8% | 0.8% | **0.2%** | 0.4% |
| 5-9 | 33.2% | 4.4% | **0.8%** | 2.6% |
| 10-18 | 45.7% | 10.4% | **6.2%** | 13.9% |
| 19-28 | (game ended) | 12.9% | 18.8% | 15.1% |
| 29-43 | | 42.6% | 38.2% | 43.5% |
| 44-53 | | (game ended) | **62.7%** | (game ended) |
| Whole game | 32.2% (18 moves) | 20.6% (43 moves) | 27.4% (53 moves) | 21.9% (43 moves) |
| Top move legal | 13/18 (72%) | 40/43 (93%) | 44/53 (83%) | 37/43 (86%) |

(Whole-game averages aren't comparable across games of different length; compare by stage.)

**What the games show:**
- **Opening and middlegame: clean.** Moves 1-18 had 0.2-6% probability on illegal moves, and recaptures
  are confident (4...Nxe5 76%, 7...Nc6 74%, 21...Rxc2 66%). The comparison with the move-model column is
  **not valid** (that app was broken, see above); measured on the same positions, the move model is as
  good or better (table above).
- **Endgame: king moves into attacked squares.** Almost all illegal mass late in the game is the king
  moving next to the enemy king or onto a square a piece controls: 44... `e5d6` 50%, 46... `g5h4` 39%,
  48... `g5h4` **79%**, 50... `g3h4` **81%**, plus castling or non-evasions while in check (29... in the
  step-23k game). The model sees every piece but hasn't learned which squares the opponent controls,
  especially with few pieces and long lines. Endgame positions are also rarer in the data.
- Some odd moves are the sampling mode, not the model (e.g. 9...Nge7 at p = 4.3% while Nf6 had 47%);
  use "Top legal move" to judge strength.
- **30M vs 16M in play: no better, matching the val accuracy.** The 30M put more on illegal moves in
  moves 10-18 (13.9% vs 6.2%) and had the same endgame king problem (39... `e4d4` 45%, 41... `g6g5` 38%,
  43... five of its top 5 illegal). It also castled long into a king walk (12...O-O-O, then the king
  went b8 → c7 → b7 → c6 → … → h5).
- 30M self-play sample (end of training): sensible opening, a queen blunder (15. Qf6?? Nxf6), then a won
  endgame it couldn't convert; it hit the 200-move limit shuffling kings.

### Approximate Elo vs Stockfish (measured, 2026-10-07)

Script: `inference_board/elo_vs_stockfish.py`. Checkpoint `chess_board_best.pt` (16M, step 49,500).
- **Model settings:** greedy top legal move, no search; extra tokens: result = model wins, both ratings `<elo_2400>`.
- **Stockfish:** 18, `UCI_LimitStrength` at fixed `UCI_Elo`, 0.1 s/move, 1 thread.
- **Games:** 200 per level, colours alternating; games over 400 plies count as draws. Run on CPU, 376 s total.

| Stockfish UCI_Elo | Games | W | D | L | Score | as White | as Black | Avg plies |
|---|---|---|---|---|---|---|---|---|
| 1320 | 200 | 12 | 36 | 152 | 15.0% | 19.0% | 11.0% | 88 |
| 1500 | 200 | 2 | 26 | 172 | 7.5% | 6.0% | 9.0% | 77 |
| 1700 | 200 | 4 | 15 | 181 | 5.8% | 7.0% | 4.5% | 72 |

**Maximum-likelihood fit over all 600 games: ≈ 1060 Elo on Stockfish's UCI_Elo scale (95% interval 1011–1108).**

How to read it:
- **The interval covers game-to-game noise only.** It doesn't cover how well Stockfish's `UCI_Elo` is calibrated (calibrated against engines at longer time controls; at 0.1 s/move Stockfish plays a bit below its label). Quote it as "≈1050 vs Stockfish", not as a Lichess or FIDE rating.
- **Results:** 505 of 600 games were lost to checkmate, matching the known weakness: the model doesn't see tactics and king attacks. Draws: 68 by threefold repetition, 8 stalemates, 1 insufficient material.
- **Illegal mass:** the model put on average 14.0% of its probability on illegal moves in these games, more than on the val set (10.3%), because its own games reach positions unlike human games.
- **Lowest Stockfish level:** `UCI_Elo` can't go below 1320, so 1320 is the most informative level here.

To compare future checkpoints, run the same command:
`CHESS_BOARD_CKPT=<ckpt> venv/bin/python chess_gpt/inference_board/elo_vs_stockfish.py --games 200 --levels 1320 1500 1700 --workers 12`

### Resume bullet (draft, board model)

> Compared move-history vs board-state inputs for a from-scratch chess transformer on the same 3.0M
> Lichess games and the same 15K held-out positions: the move-history GPT (16M params) predicted the
> human move with 40.5% top-1 / 78.4% top-5 accuracy vs 36.6% / 74.2% for a board-input encoder of the
> same size, because it gets ~30× more training targets per step. Also found that doubling the board model
> to 29M params at a fixed budget lowered accuracy (36.7% → 35.1%): it was limited by examples seen, not
> capacity.

### TODO
- [x] Move model's top-1 / top-5 on the **same** val positions (done: table above).
- [x] Illegal mass by move number over many val positions, both models (done: table above).
- [ ] Replay a move-model game in the fixed app (old app games are invalid).
- [ ] 30M with more data / more passes (next run).

---

## 7. Board model v2: fast training + all positions + flip + 8-move history (16M)

Script: `chess_gpt_board_fast_allpos_flip_hist8_16M.py`. Inference / app / Elo test: `inference_board_flip_hist8/`.
Final checkpoint: `chess_board_fast_allpos_flip_hist8_16M_decayed.pt` (step 150,001).

### What changed vs the section 6 board model
| | Section 6 board model (16M) | This model (16M) |
|---|---|---|
| Input | 73 tokens, board from White's side | 81 tokens: board **from the side to move** (mirrored for Black) + **last 8 moves** |
| Training positions | 27M fixed (10 per game, 3.9 GB file) | made on the fly from `games.bin`; all **~189M** train positions available |
| Attention / precision | per-head loop, fp32 | fused `scaled_dot_product_attention`, bf16 autocast |
| Speed (RTX 3050) | ~395 positions/s | **~895 positions/s (2.3×)** |
| Params | 15.92M | 16.71M (+0.8M for history-move embeddings) |
| Batch / lr | 128 / 3e-4 cosine, 50k steps | 256 / 4e-4, AdamW betas (0.9, 0.95), warmup 2k, constant to 130k, decay to 4e-5 over 20k |
| Positions seen | 6.4M | **38.4M** (150k × 256) |
| Training time | ~4.5 h | ~12 h |

### Results (measured 2026-10-08, same val positions for both models)
200,000 val positions (same rows as section 6: `default_rng(0)` sample of the val file):

| Model | Val loss | Top-1 | Top-5 |
|---|---|---|---|
| Section 6 board model (`chess_board.pt`) | 2.211 | 36.7% | 74.1% |
| **This model (`..._decayed.pt`)** | **1.752** | **49.0%** | **86.0%** |

By game stage, first 1,500 val games (15,000 positions); illegal = probability put on illegal moves:

| Stage | Old top-1 | Old illegal | **New top-1** | **New illegal** |
|---|---|---|---|---|
| all | 36.6% | 10.3% | **49.2%** | **7.0%** |
| moves 1-10 | 46.8% | 1.6% | 50.4% | 1.1% |
| moves 11-20 | 35.4% | 9.1% | 47.9% | 6.2% |
| moves 21-30 | 28.3% | 16.6% | 48.6% | 11.7% |
| moves 31-40 | 25.7% | 22.1% | 47.9% | 15.4% |
| moves 41+ | 30.4% | 23.6% | 52.0% | 15.4% |
| **in check** (1,030 positions) | 45.6% | **41.1%** | 69.9% | **5.8%** |

- The gain is largest in the middlegame and endgame (+20 points from move 21 on), where the old model was weakest.
- **In check: illegal mass 41.1% → 5.8%.** The history tokens show the move that just gave check, so the model no longer
  ignores the check (the old model's main failure in play tests).
- Endgame king moves are still the weak spot (15.4% illegal mass from move 31 on).
- Move-history model (section 6 comparison, 15k positions): 40.5% top-1. This board model is now 8.7 points above it.

Training log highlights (val, 25.6k-position estimates): step 50k 2.255 / 40.0%; step 129k (end of constant lr)
1.929 / 46.1%; step 150k (end of decay) 1.744 / 49.3%. The decay alone took val loss from ~1.95 to ~1.75.
My mid-run forecast (2.00-2.10 final) was too pessimistic: the constant-lr phase kept improving longer than the fit assumed.

### Elo vs Stockfish (measured 2026-10-08, same test as section 6)
`inference_board_flip_hist8/elo_vs_stockfish.py`, greedy top legal move, no search, Stockfish 18 at 0.1 s/move,
200 games per level, colours alternating. Log: `inference_board_flip_hist8/elo_16M_decayed.log`.

| Stockfish UCI_Elo | W | D | L | Score | as White | as Black |
|---|---|---|---|---|---|---|
| 1320 | 120 | 36 | 44 | 69.0% | 70.5% | 67.5% |
| 1500 | 68 | 43 | 89 | 44.8% | 48.0% | 41.5% |
| 1700 | 44 | 36 | 120 | 31.0% | 32.0% | 30.0% |

**≈ 1490 Elo on Stockfish's UCI_Elo scale (95% interval 1461-1521), up from ≈ 1060 (1011-1108): +430.**
232 wins by checkmate (old model: 18 wins in 600 games). Same caveat as before: Stockfish scale, not a Lichess/FIDE rating.

### Resume bullet (draft, measured numbers)
> Built a chess move-prediction transformer from scratch in PyTorch, trained on 3.0M Lichess games on a single
> RTX 3050. Re-designed the input (board from the side to move + last 8 moves), generated training positions on
> the fly (27M → ~189M available) and sped up training 2.3× (fused attention + bf16). Same-size model improved from
> 36.7% to 49.0% human move-match (top-5 74.1% → 86.0%) on 200K held-out positions, and from ≈1060 to ≈1490 Elo
> against Stockfish 18 with no search (600 games per model).

### TODO
- [ ] Legal-move auxiliary target and/or in-check token (endgame king moves still 15% illegal mass).
- [ ] 60M on a rented GPU (~$10-35 for ~190M positions, see chat estimate).
