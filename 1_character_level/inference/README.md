---
title: Chess GPT
emoji: ♟️
colorFrom: gray
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# Chess GPT

A small GPT trained from scratch on Lichess games (move text only, SAN notation).
Play against it in the app. If the model produces an illegal move 3 times in a row,
a random legal move is played instead, and the app shows how often that happens.
