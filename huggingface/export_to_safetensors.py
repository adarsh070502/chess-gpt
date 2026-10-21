"""
Turn the folder-5 training checkpoints into the files uploaded to the Hugging Face model repo.

For each size (30M, 16M):
    chess_board_fast_allpos_flip_hist8_legal_<size>_decayed.pt   (~378 MB / ~210 MB: weights + AdamW state)
 -> model_repo/model_<size>.safetensors                            weights only (fp32), no optimizer state
 -> model_repo/config_<size>.json                                  hyperparameters, step, both vocabs

Then it checks the export: the Space's model.py, loading the new files, must give the same move probabilities
as the existing 5_board_flip_hist8_legal/inference/model.py loading the original .pt, on positions from a few games.

Run from anywhere:  python chess_gpt/huggingface/export_to_safetensors.py
"""
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

import chess
import torch
from safetensors.torch import save_file

HERE = Path(__file__).parent
SRC_DIR = HERE.parent / "5_board_flip_hist8_legal" / "inference"
OUT_DIR = HERE / "model_repo"
SIZES = ["30M", "16M"]
HISTORY = 8


def source_path(size):
    return SRC_DIR / f"chess_board_fast_allpos_flip_hist8_legal_{size}_decayed.pt"


def export(size):
    src = source_path(size)
    ckpt = torch.load(src, map_location="cpu", weights_only=True)

    # Weights only. contiguous() + clone(): safetensors refuses views / tensors that share memory.
    state = {k: v.detach().cpu().contiguous().clone() for k, v in ckpt["model_state_dict"].items()}
    n_params = sum(v.numel() for v in state.values())
    save_file(state, OUT_DIR / f"model_{size}.safetensors", metadata={"step": str(ckpt["step"])})

    config = {
        **ckpt["config"],                 # n_emb, num_heads, n_layer, block_size, vocab_size, n_moves, dropout, format
        "history": HISTORY,
        "step": ckpt["step"],
        "n_params": n_params,
        "input_vocab": ckpt["input_vocab"],   # 2,012 input tokens; list index = token id
        "moves": ckpt["moves"],               # 1,968 UCI moves; list index = output id
    }
    (OUT_DIR / f"config_{size}.json").write_text(json.dumps(config, indent=1))

    mb = (OUT_DIR / f"model_{size}.safetensors").stat().st_size / 1e6
    print(f"{size}: {src.name} (step {ckpt['step']}) -> model_{size}.safetensors ({mb:.0f} MB, "
          f"{n_params:,} params) + config_{size}.json")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify(size, space_model):
    """Same probabilities from (old model.py + .pt) and (Space model.py + .safetensors)?"""
    os.environ["CHESS_BOARD_CKPT"] = str(source_path(size))      # old model.py picks its checkpoint from this
    old = load_module(f"old_model_{size}", SRC_DIR / "model.py")
    old_net = old.load_model().cpu()
    new = space_model.ChessModel(size)

    rng = random.Random(0)
    worst, n = 0.0, 0
    for game in range(5):
        board = chess.Board()
        for ply in range(60):
            if board.is_game_over():
                break
            args = (board, "<white_wins>", "<elo_2000>", "<elo_1600>")
            p_old = old.move_probs(old_net, *args)
            p_new = new.move_probs(*args)
            worst, n = max(worst, (p_old - p_new).abs().max().item()), n + 1
            board.push(rng.choice(list(board.legal_moves)))
    ok = worst < 1e-4
    print(f"{size}: {n} positions, max probability difference {worst:.2e} -> {'OK' if ok else 'MISMATCH'}")
    return ok


def check_usage_example(space_model, size):
    """The 'Usage' snippet from model_repo/README.md, minus the download (files are read locally)."""
    model = space_model.ChessModel(size)
    board = chess.Board()
    board.push_san("e4")
    probs = model.move_probs(board, "<black_wins>", "<elo_1600>", "<elo_2000>")
    best = max(board.legal_moves, key=lambda m: probs[model.move_to_id[m.uci()]])
    ok = probs.shape == (1968,) and abs(probs.sum().item() - 1) < 1e-4 and best in board.legal_moves
    print(f"usage example ({size}): after 1. e4 the model plays 1... {board.san(best)} -> {'OK' if ok else 'FAILED'}")
    return ok


if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    sizes = [s for s in SIZES if source_path(s).is_file()]
    for s in SIZES:
        if s not in sizes:
            print(f"{s}: skipped, {source_path(s)} not found (copy it there and run again)")
    if not sizes:
        sys.exit("No checkpoints found.")

    for size in sizes:
        export(size)

    # The Space's model.py reads the files from this folder instead of the Hub when CHESS_MODEL_DIR is set
    os.environ["CHESS_MODEL_DIR"] = str(OUT_DIR)
    space_model = load_module("space_model", HERE / "space" / "model.py")
    results = [verify(size, space_model) for size in sizes]
    results.append(check_usage_example(space_model, sizes[0]))
    if not all(results):
        print("\nCheck FAILED, don't upload.")
    elif len(sizes) < len(SIZES):
        print("\nChecks passed for " + ", ".join(sizes) + "; export the missing size before uploading.")
    else:
        print("\nAll checks passed, model_repo/ is ready to upload.")
