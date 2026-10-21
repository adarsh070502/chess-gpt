---
title: Chess GPT
emoji: ♟️
colorFrom: gray
colorTo: green
sdk: gradio
sdk_version: 6.29.0
python_version: '3.12'
app_file: app.py
pinned: false
license: mit
models:
  - adarsh-007/chess-gpt-board
short_description: Play chess against a transformer trained from scratch
---

# Chess GPT

Play against a transformer trained from scratch on 3.0M Lichess games to predict the move a human would play.
No search, no rules given: every move is one forward pass over the board, the asked result and ratings, and the
last 8 moves.

Two sizes: **30M** (≈1558 Elo vs Stockfish, 50.8% human move-match) and **16M** (≈1492, 49.9%).

Model card, numbers and how they were measured: [adarsh-007/chess-gpt-board](https://huggingface.co/adarsh-007/chess-gpt-board).
