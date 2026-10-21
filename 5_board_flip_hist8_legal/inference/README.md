# Chess GPT (board + 8-move history + legal-move training)

Play against the board model trained by
`chess_gpt/5_board_flip_hist8_legal/chess_gpt_board_fast_allpos_flip_hist8_legal_16M.py` (the same code works for the
`_30M` checkpoints). Input and output are the same as `4_board_flip_hist8/inference/` (81 tokens: board from the side
to move + last 8 moves, 1,968 UCI moves out). The only difference is training: the model also learned to predict which
moves are legal (`legal_head`). That layer is loaded but not used here; the app uses the move head as before.

The training scripts save their checkpoints in this folder.

Run from the repo root (`Datascience/`):

    venv/bin/python chess_gpt/5_board_flip_hist8_legal/inference/app.py

The app opens on http://127.0.0.1:7863 (7860 move tokens, 7861 old board, 7862 flip + history).

Checkpoint: `chess_board_fast_allpos_flip_hist8_legal_16M_decayed.pt`, else `_best.pt`, else `_latest.pt`, in this
folder. To pick another: `CHESS_BOARD_CKPT=<file name>.pt` (a bare name means this folder) before the command.

## Approximate Elo against Stockfish

    CHESS_BOARD_CKPT=chess_board_fast_allpos_flip_hist8_legal_16M_decayed.pt venv/bin/python chess_gpt/5_board_flip_hist8_legal/inference/elo_vs_stockfish.py --games 200 --levels 1320 1500 1700 --workers 12

Same test as `4_board_flip_hist8/inference/elo_vs_stockfish.py`, so the numbers compare directly
(folder 4's 16M: ≈1490). Stockfish binary: `chess_gpt/stockfish/stockfish`.
