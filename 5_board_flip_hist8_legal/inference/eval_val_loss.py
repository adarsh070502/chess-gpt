# Val loss / top-1 / top-5 on the 200k val rows (default_rng(0), same as folder 4) + train loss, legal loss and
# illegal mass on 51,200 fresh train positions. Uses the GPU if free.
# Run from the repo root: venv/bin/python chess_gpt/5_board_flip_hist8_legal/inference/eval_val_loss.py chess_board_fast_allpos_flip_hist8_legal_16M_decayed.pt
import importlib.util, os, sys, numpy as np, torch, torch.nn.functional as F
spec = importlib.util.spec_from_file_location("L", "chess_gpt/5_board_flip_hist8_legal/chess_gpt_board_fast_allpos_flip_hist8_legal_" + os.environ.get("SIZE", "16M") + ".py")
L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
boards, moves_ = L.load_val()
rows = np.sort(np.random.default_rng(0).choice(len(moves_), 200_000, replace=False))   # same rows as folder 4's eval
for name in sys.argv[1:]:
    ck = torch.load(L.CKPT_DIR / name, map_location=L.device, weights_only=False)
    m = L.Transormer(L.n_emb).to(L.device); m.load_state_dict(ck["model_state_dict"]); m.eval()
    tot, top1, top5 = 0.0, 0, 0
    with torch.no_grad():
        for i in range(0, len(rows), 1000):
            ix = rows[i:i+1000]
            x = torch.from_numpy(boards[ix].astype(np.int64)).to(L.device); y = torch.from_numpy(moves_[ix].astype(np.int64)).to(L.device)
            with L.autocast():
                logits, _ = m(x)
            logits = logits.float()
            tot += F.cross_entropy(logits, y, reduction="sum").item()
            top1 += (logits.argmax(-1) == y).sum().item()
            top5 += (logits.topk(5, -1).indices == y[:, None]).any(-1).sum().item()
    n = len(rows)
    L.batch_size = 256
    batches = L.make_train_loader()
    tl, ll, im = [], [], []
    with torch.no_grad():
        for _ in range(200):     # 51,200 fresh train positions
            x, y, legal = (t.to(L.device) for t in next(batches))
            lt = torch.zeros(len(y), L.n_moves + 1, device=L.device).scatter_(1, legal, 1.0)[:, :L.n_moves]
            with L.autocast():
                logits, loss = m(x, y, lt)
            tl.append(m.move_loss.item()); ll.append(m.legal_loss.item())
            im.append(1 - (F.softmax(logits.float(), -1) * lt).sum(-1).mean().item())
    print(f"{name} (step {ck['step']}): val loss {tot/n:.4f}, top-1 {top1/n:.1%}, top-5 {top5/n:.1%} | "
          f"train loss {np.mean(tl):.4f}, legal loss {np.mean(ll):.3f}, illegal mass {np.mean(im):.1%}", flush=True)
