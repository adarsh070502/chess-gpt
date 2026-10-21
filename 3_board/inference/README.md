# Chess GPT (board input)

Play against the board-input Chess GPT, trained by `chess_gpt/3_board/chess_gpt_board_embeddings.py`.
Each turn the model sees only the current board (73 tokens: side to move, castling rights,
en passant, asked result, two ratings, 64 squares) and outputs one of 1,968 UCI moves.

Run from the repo root (`Datascience/`):

    venv/bin/python chess_gpt/3_board/inference/app.py

The app opens on http://127.0.0.1:7861, so it can run next to the move-token app (7860).

Checkpoint: set in `find_checkpoint()` of `model.py` (currently `chess_board_30M_best.pt`), looked for in this
folder, where the training scripts save their checkpoints. To pick another: `CHESS_BOARD_CKPT=chess_board.pt`
(a bare name means this folder) before the command.

Stockfish (only for the Stockfish mode) is taken from `STOCKFISH_PATH`, `chess_gpt/stockfish/`, the system
PATH or `/usr/games/stockfish` (`sudo apt install stockfish`).

The log has the same format as the move-token app (`p`, `on illegal moves`, `top`), so the two models
can be compared game by game.

## Approximate Elo against Stockfish

    CHESS_BOARD_CKPT=chess_board_best.pt venv/bin/python chess_gpt/3_board/inference/elo_vs_stockfish.py --games 200 --levels 1320 1500 1700 --workers 12

Plays the model (greedy top legal move, no search) against Stockfish at fixed `UCI_Elo` levels on CPU,
then fits one rating to all results. A Linux Stockfish 18 binary is in `chess_gpt/stockfish/stockfish`.
Results are in `chess_gpt/docs/MOVE_TOKENS_V3_NOTES.md`, section 6.
