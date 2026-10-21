# Chess GPT (board + 8-move history)

Play against the board model trained by `chess_gpt/4_board_flip_hist8/chess_gpt_board_fast_allpos_flip_hist8_16M.py`
(the same code works for the `_60M` checkpoints). Each turn the model sees 81 tokens: the current board
seen from the side to move (mirrored when Black is to move), castling rights, en passant, the asked result,
both ratings, and the last 8 moves. It outputs one of 1,968 UCI moves; `model.move_probs()` maps the answer
back to real moves when the board was mirrored.

The old 73-token board model stays in `3_board/inference/`; its checkpoints don't load here (and the other way round).

Run from the repo root (`Datascience/`):

    venv/bin/python chess_gpt/4_board_flip_hist8/inference/app.py

The app opens on http://127.0.0.1:7862 (7860 = move-token app, 7861 = old board app).

Checkpoint: `chess_board_fast_allpos_flip_hist8_16M_decayed.pt`, else `_best.pt`, else `_latest.pt`, looked for
in this folder, where the training scripts save their checkpoints. To pick another: `CHESS_BOARD_CKPT=<file name>.pt`
(a bare name means this folder) before the command.

## Approximate Elo against Stockfish

    CHESS_BOARD_CKPT=chess_board_fast_allpos_flip_hist8_16M_best.pt venv/bin/python chess_gpt/4_board_flip_hist8/inference/elo_vs_stockfish.py --games 200 --levels 1320 1500 1700 --workers 12

Same test as `3_board/inference/elo_vs_stockfish.py` (greedy top legal move, no search, Stockfish 18 at fixed
`UCI_Elo`, CPU only), so the numbers can be compared directly. It uses the Stockfish binary in
`chess_gpt/stockfish/stockfish`.
