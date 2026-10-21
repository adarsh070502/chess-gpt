"""
Approximate Elo of the board model trained with the legal-move target (copy of 4_board_flip_hist8/inference/elo_vs_stockfish.py): play games against Stockfish at fixed UCI_Elo levels and fit one rating.

How it works
  1. For each Stockfish level (UCI_Elo 1320, 1500, ...) play N games, model alternating White/Black.
  2. The model plays its top legal move (greedy, no search, no sampling): its "strongest" setting.
     Every position gets the same extra tokens as in training: result = "model wins", both ratings = --model-elo,
     plus the board flip and the last 8 moves (model.move_probs).
  3. Score per level = (wins + 0.5 * draws) / games.
  4. Fit one rating R so that the Elo formula  expected = 1 / (1 + 10 ** ((sf_elo - R) / 400))
     matches all levels at once (maximum likelihood), with a 95% interval.

Notes
  - Runs on CPU only (CUDA hidden), so it can run while training uses the GPU.
  - Stockfish's UCI_Elo is calibrated to CCRL engine ratings at long time controls, not to Lichess ratings;
    with short think times Stockfish is a bit weaker than its label. Treat the result as an approximate number.
  - UCI_Elo cannot go below 1320. If the model scores ~0% at 1320, the fit only says "well below 1320".

Run from the repo root (Datascience/):
    venv/bin/python chess_gpt/5_board_flip_hist8_legal/inference/elo_vs_stockfish.py --games 200 --levels 1320 1500 1700 --workers 12
    CHESS_BOARD_CKPT=chess_board_fast_allpos_flip_hist8_legal_16M_best.pt venv/bin/python chess_gpt/5_board_flip_hist8_legal/inference/elo_vs_stockfish.py
"""
import argparse
import math
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""      # CPU only: must be set before torch is imported (in model.py)

import chess
import chess.engine
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from model import ckpt, CKPT_PATH, load_model, move_probs, move_to_id   # noqa: E402

STOCKFISH = os.environ.get("STOCKFISH_PATH", str(HERE.parent.parent / "stockfish" / "stockfish"))
MAX_PLIES = 400            # games longer than this are scored as a draw

model = None               # built once per worker process


def init_worker():
    global model
    torch.set_num_threads(1)                    # many games in parallel, one core each
    model = load_model()


@torch.no_grad()
def model_move(board, model_white, model_elo):
    # Same extra tokens as in training: the result is written from White's side, first rating is White's.
    result = "<white_wins>" if model_white else "<black_wins>"
    elo_tok = f"<elo_{model_elo}>"
    probs = move_probs(model, board, result, elo_tok, elo_tok)   # (1968,), real moves (unmirrored)
    legal = list(board.legal_moves)
    legal_ids = torch.tensor([move_to_id[m.uci()] for m in legal])
    best = probs[legal_ids].argmax().item()     # greedy: the most likely legal move
    illegal_mass = 1 - probs[legal_ids].sum().item()
    return legal[best], illegal_mass


def play_game(job):
    """job = (sf_elo, model_white, model_elo, sf_time). Returns the model's score (1 / 0.5 / 0) and stats."""
    sf_elo, model_white, model_elo, sf_time = job
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
    engine.configure({"Threads": 1, "Hash": 16, "UCI_LimitStrength": True, "UCI_Elo": sf_elo})
    board = chess.Board()
    illegal = []
    try:
        while not board.is_game_over(claim_draw=True) and board.ply() < MAX_PLIES:
            if board.turn == (chess.WHITE if model_white else chess.BLACK):
                mv, im = model_move(board, model_white, model_elo)
                illegal.append(im)
            else:
                mv = engine.play(board, chess.engine.Limit(time=sf_time)).move
            board.push(mv)
    finally:
        engine.quit()

    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:      # draw, or hit MAX_PLIES
        score, how = 0.5, outcome.termination.name.lower() if outcome else "max plies"
    else:
        score = 1.0 if outcome.winner == model_white else 0.0
        how = outcome.termination.name.lower()
    return sf_elo, model_white, score, how, board.ply(), sum(illegal) / max(len(illegal), 1)


def fit_elo(results):
    """Maximum-likelihood rating from (sf_elo, score) pairs, plus a 95% interval (log-likelihood drop of 1.92)."""
    def loglik(r):
        total = 0.0
        for sf_elo, s in results:
            p = 1 / (1 + 10 ** ((sf_elo - r) / 400))
            p = min(max(p, 1e-9), 1 - 1e-9)
            total += s * math.log(p) + (1 - s) * math.log(1 - p)   # draws count as half a win, half a loss
        return total

    grid = range(0, 3501)
    lls = [loglik(r) for r in grid]
    best = max(range(len(lls)), key=lls.__getitem__)
    inside = [r for r, ll in zip(grid, lls) if ll >= lls[best] - 1.92]
    return grid[best], inside[0], inside[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, nargs="+", default=[1320, 1500, 1700])
    ap.add_argument("--games", type=int, default=40, help="games per level (half as White, half as Black)")
    ap.add_argument("--model-elo", default="2400", help="rating bucket token given to the model (800..2400)")
    ap.add_argument("--sf-time", type=float, default=0.1, help="Stockfish seconds per move")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    jobs = [(lv, g % 2 == 0, args.model_elo, args.sf_time) for lv in args.levels for g in range(args.games)]
    print(f"checkpoint {CKPT_PATH} (step {ckpt.get('step', '?')}), {len(jobs)} games, "
          f"model elo token {args.model_elo}, Stockfish {args.sf_time}s/move", flush=True)

    t0 = time.time()
    per_level = {lv: [] for lv in args.levels}
    with Pool(args.workers, initializer=init_worker) as pool:
        for i, (lv, white, score, how, plies, im) in enumerate(pool.imap_unordered(play_game, jobs), 1):
            per_level[lv].append((white, score, how, plies, im))
            print(f"[{i:3d}/{len(jobs)}] SF {lv}  model {'W' if white else 'B'}  "
                  f"score {score}  {how:20s} plies {plies:3d}  illegal mass {im:.1%}", flush=True)

    print(f"\n{'SF Elo':>7} {'games':>6} {'W':>4} {'D':>4} {'L':>4} {'score':>7} "
          f"{'as White':>9} {'as Black':>9} {'avg plies':>10}")
    fit_input = []
    for lv, games in per_level.items():
        scores = [g[1] for g in games]
        w, d, l = scores.count(1.0), scores.count(0.5), scores.count(0.0)
        sw = [g[1] for g in games if g[0]]
        sb = [g[1] for g in games if not g[0]]
        print(f"{lv:>7} {len(games):>6} {w:>4} {d:>4} {l:>4} {sum(scores) / len(scores):>7.1%} "
              f"{sum(sw) / len(sw):>9.1%} {sum(sb) / len(sb):>9.1%} {sum(g[3] for g in games) / len(games):>10.0f}")
        fit_input += [(lv, s) for s in scores]

    r, lo, hi = fit_elo(fit_input)
    print(f"\nEstimated Elo (Stockfish UCI_Elo scale): {r}  (95% interval {lo}-{hi})")
    print(f"took {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
