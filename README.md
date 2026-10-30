# Chess GPT

Transformers trained from scratch on Lichess games to predict the move a human plays. Each numbered folder is one
model family: its training script(s) at the top, and an `inference/` folder with the app, the Elo test and the
checkpoints (`.pt`). Training scripts save their checkpoints straight into their own `inference/` folder.

```
chess_gpt/
  data/                      shared data
      data_with_player_rating_start_token.ipynb    Lichess PGN -> games.bin + vocab.json
      games.bin, vocab.json                        3.0M games as uint16 move tokens; the token list
      demo.ipynb                                   small example of reading games.bin
  docs/                      notes and results (MOVE_TOKENS_V3_NOTES.md has every measured number)
  stockfish/                 Stockfish 18 (Linux build + Windows .exe), used by every app / Elo test

  1_character_level/         first model: PGN text, one character per token
  2_move_tokens/             GPT over the whole game as move tokens (15.8M, 40.5% top-1)
  3_board/                   board input, 73 tokens (16M: 36.7% top-1, ≈1060 Elo; 30M runs)
      boards_*.bin, moves_*.bin   fixed dataset, 10 positions per game (built by the scripts)
  4_board_flip_hist8/        board from the side to move + last 8 moves, positions made on the fly
                             (16M: 49.0% top-1, 86.0% top-5, ≈1490 Elo vs Stockfish)
      boards_flip_hist8_val.bin / moves_flip_hist8_val.bin   val set (also used by folder 5)
  5_board_flip_hist8_legal/  folder 4 + a second training target: which moves are legal
```

Run everything from the parent folder of the repo, for example:

    venv/bin/python chess_gpt/4_board_flip_hist8/inference/app.py
    venv/bin/python chess_gpt/5_board_flip_hist8_legal/chess_gpt_board_fast_allpos_flip_hist8_legal_16M.py

App ports: 7860 move tokens, 7861 board (3), 7862 flip + history (4), 7863 legal (5).

`huggingface/space/` is the public demo ([adarsh-007/chess-gpt](https://huggingface.co/spaces/adarsh-007/chess-gpt)).
Its *Varied* mode uses top-k sampling: the 5 most likely legal moves, drawn by their probabilities (`SAMPLE_TOP_K`
in `app.py`). Run it locally from `huggingface/space/` with `python app.py` (downloads the weights on first start).

Notes:
- `1_character_level/`: its training text (`games2.txt`) no longer exists; the checkpoint and app still work.
- `4_board_flip_hist8/inference/chess_board_v3_60M_*.pt` are from a 500-step test of the 60M under an old name;
  they cannot be resumed by the current scripts and can be deleted.

## Not included in this repo

Checkpoints (`*.pt`, 100–700 MB each), the training datasets (`*.bin`, up to 3.7 GB) and the Stockfish binaries are too
large for GitHub.

- **Trained weights:** [huggingface.co/adarsh-007/chess-gpt-board](https://huggingface.co/adarsh-007/chess-gpt-board)
- **Data:** rebuild `data/games.bin` from a Lichess PGN dump with `data/data_with_player_rating_start_token.ipynb`
- **Stockfish:** download from [stockfishchess.org](https://stockfishchess.org/download/) into `stockfish/`
