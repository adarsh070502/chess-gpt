# Illegal mass + top-1 by stage for folders 3, 4, 5 on the same val positions (first N val games, CPU only).
# Run from the repo root: venv/bin/python chess_gpt/5_board_flip_hist8_legal/inference/eval_by_stage.py 1500
import os, sys, importlib.util
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, torch, torch.nn.functional as F, chess
torch.set_num_threads(4)
R = "/home/aidev4/PycharmProjects/Datascience/chess_gpt/"
N_GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
ONLY = sys.argv[2:]   # optional model labels to run, e.g. "5: + legal head 30M"; default all

def load(path, name, ckpt=None):
    if ckpt: os.environ["CHESS_BOARD_CKPT"] = ckpt
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    os.environ.pop("CHESS_BOARD_CKPT", None)
    return mod

T = load(R + "5_board_flip_hist8_legal/chess_gpt_board_fast_allpos_flip_hist8_legal_16M.py", "T")   # replay helpers
games_data, starts, lengths = T.game_index()
n_train = int(0.9 * len(starts))

# Replay the same 10 plies per val game as the val files; keep labels per position.
info = []   # (move_no, in_check, n_pieces, legal_real_ids, legal_flip_ids, y_real, y_flip)
for g in range(n_train, n_train + N_GAMES):
    s, L = int(starts[g]), int(lengths[g])
    ids = games_data[s:s+L].tolist()
    rng = np.random.default_rng(g)
    plies = set(rng.choice(L - 5, size=min(10, L - 5), replace=False).tolist())
    ucis = [T.game_vocab[i] for i in ids][4:-1]
    b = chess.Board()
    for ply, u in enumerate(ucis[:max(plies) + 1]):
        mv = chess.Move.from_uci(u)
        if ply in plies:
            flip = b.turn == chess.BLACK
            info.append((ply // 2 + 1, b.is_check(), len(b.piece_map()),
                         [T.move_to_id[m.uci()] for m in b.legal_moves], [T.target_id(m, flip) for m in b.legal_moves],
                         T.move_to_id[mv.uci()], T.target_id(mv, flip)))
        b.push(mv)
K = len(info)
x81 = np.memmap(R + "4_board_flip_hist8/boards_flip_hist8_val.bin", dtype=np.uint16, mode="r").reshape(-1, 81)[:K].astype(np.int64)
y81 = np.memmap(R + "4_board_flip_hist8/moves_flip_hist8_val.bin", dtype=np.uint16, mode="r")[:K]
x73 = np.memmap(R + "3_board/boards_val.bin", dtype=np.uint16, mode="r").reshape(-1, 73)[:K].astype(np.int64)
y73 = np.memmap(R + "3_board/moves_val.bin", dtype=np.uint16, mode="r")[:K]
assert all(y81[i] == info[i][6] for i in range(K)), "folder 4/5 val rows do not line up"
assert all(y73[i] == info[i][5] for i in range(K)), "folder 3 val rows do not line up"

models = [
    ("3: board 16M", load(R + "3_board/inference/model.py", "m3", "chess_board_best.pt"), x73, 3),
    ("4: flip+hist8 16M", load(R + "4_board_flip_hist8/inference/model.py", "m4", "chess_board_fast_allpos_flip_hist8_16M_decayed.pt"), x81, 4),
    ("5: + legal head 16M", load(R + "5_board_flip_hist8_legal/inference/model.py", "m5", "chess_board_fast_allpos_flip_hist8_legal_16M_decayed.pt"), x81, 4),
    ("5: + legal head 30M", load(R + "5_board_flip_hist8_legal/inference/model.py", "m5b", "chess_board_fast_allpos_flip_hist8_legal_30M_decayed.pt"), x81, 4),
]
models = [mm for mm in models if not ONLY or mm[0] in ONLY]
groups = {"all": lambda r: True, "moves 1-10": lambda r: r[0] <= 10, "moves 11-20": lambda r: 11 <= r[0] <= 20,
          "moves 21-30": lambda r: 21 <= r[0] <= 30, "moves 31+": lambda r: r[0] >= 31,
          "in check": lambda r: r[1], "endgame (<=10 pieces)": lambda r: r[2] <= 10}
print(f"{K:,} positions from {N_GAMES:,} val games; group sizes: " +
      ", ".join(f"{k} {sum(f(r) for r in info):,}" for k, f in groups.items()), flush=True)
for label, mod, X, which in models:
    net = mod.Transormer(mod.n_emb); net.load_state_dict(mod.ckpt["model_state_dict"]); net.eval()
    im, top1, top_illegal = np.zeros(K), np.zeros(K), np.zeros(K)
    with torch.no_grad():
        for i in range(0, K, 512):
            out = net(torch.from_numpy(X[i:i+512]))
            p = F.softmax((out[0] if isinstance(out, tuple) else out).float(), -1).numpy()
            for j in range(len(p)):
                r = info[i + j]
                legal, y = (r[3], r[5]) if which == 3 else (r[4], r[6])
                im[i+j] = 1 - p[j, legal].sum()
                a = p[j].argmax(); top1[i+j] = a == y; top_illegal[i+j] = a not in set(legal)
    print(f"\n{label}  ({mod.CKPT_PATH.name})")
    print(f"  {'group':22s} {'illegal mass':>12s} {'top-1':>7s} {'top pick illegal':>17s}")
    for k, f in groups.items():
        sel = np.array([f(r) for r in info])
        print(f"  {k:22s} {im[sel].mean():12.1%} {top1[sel].mean():7.1%} {top_illegal[sel].mean():17.2%}", flush=True)
