# Chess GPT (move tokens)

Play against the move-level Chess GPT (one token per UCI move, trained by
`chess_gpt/2_move_tokens/chess_gpt_move_tokens.py` on `games_uci.txt`).

Run from the `DeepLearning` folder:

    venv\Scripts\python chess_gpt\2_move_tokens\inference\app.py

Checkpoint: the newest `chess_through_moves*.pt` in the current folder, `chess_gpt/` or this
folder is used automatically. To pick one: `set CHESS_CKPT=path\to\file.pt` before running.

Stockfish (only for the Stockfish mode) is taken from `STOCKFISH_PATH`, `./stockfish/`,
`chess_gpt/stockfish/`, or the system PATH.
