"""
Chess GPT board model (board from the side to move + last 8 moves + legal-move training), for the Space.

Weights and config are downloaded from the model repo (REPO_ID) on first use and cached by huggingface_hub:
    model_<size>.safetensors   weights (fp32)
    config_<size>.json         hyperparameters + both vocabs (input tokens, output moves)

Same network and encoding as chess_gpt/5_board_flip_hist8_legal/inference/model.py, except that each model
carries its own config (so the 30M and the 16M can be loaded side by side) instead of module-level globals.

One position = 81 input tokens, all from the side to move's point of view ("me"):
    0       <cls>                                   the move is read from here
    1-4     <K>/<noK> <Q>/<noQ> <k>/<nok> <q>/<noq>  castling rights: uppercase = mine, lowercase = opponent's
    5       <ep_none> / <ep_a> ... <ep_h>            en passant file
    6       <me_wins> / <me_loses> / <draw>          asked result, for the side to move
    7-8     <elo_...> <elo_...>                      my rating bucket, then the opponent's
    9-72    64 squares a8 ... h1                     '.' empty, PNBRQK mine, pnbrqk opponent's
    73-80   last 8 moves, oldest first               UCI move tokens, <no_move> before move 8
When Black is to move, the board is mirrored so the side to move always plays "upwards" as White;
move_probs() maps the answer back to real moves.
"""
import json
import os
from pathlib import Path

import chess
import torch
import torch.nn as nn
from torch.nn import functional as F
from safetensors.torch import load_file

REPO_ID = "adarsh-007/chess-gpt-board"
LOCAL_DIR = os.environ.get("CHESS_MODEL_DIR")   # set to a folder with the files to skip the download (local preview)
# Locally only the sizes that were exported are offered; on the Space, both.
MODEL_SIZES = [s for s in ["30M", "16M"] if not LOCAL_DIR or (Path(LOCAL_DIR) / f"model_{s}.safetensors").is_file()]

device = "cpu"   # one 81-token forward pass is fast on CPU; on the ZeroGPU Space, CUDA only exists inside @spaces.GPU calls

# The game result as the user asks for it (from White's side, like the training data) -> for the side to move.
RESULT_FOR_MOVER = {
    "<white_wins>": {chess.WHITE: "<me_wins>", chess.BLACK: "<me_loses>"},
    "<black_wins>": {chess.WHITE: "<me_loses>", chess.BLACK: "<me_wins>"},
    "<draw>": {chess.WHITE: "<draw>", chess.BLACK: "<draw>"},
}


def model_files(size):
    names = [f"config_{size}.json", f"model_{size}.safetensors"]
    if LOCAL_DIR:
        return [Path(LOCAL_DIR) / n for n in names]
    from huggingface_hub import hf_hub_download
    return [Path(hf_hub_download(REPO_ID, n)) for n in names]


def mirror_move(move):
    """The same move in the mirrored board: e7e5 -> e2e4, e8g8 (O-O) -> e1g1. Promotion piece unchanged."""
    return chess.Move(chess.square_mirror(move.from_square), chess.square_mirror(move.to_square), move.promotion)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_emb, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = n_emb // num_heads
        self.qkv = nn.Linear(n_emb, 3 * num_heads * self.head_size)   # q, k, v of every head in one matmul
        self.proj = nn.Linear(n_emb, n_emb)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2)   # (B, heads, T, head_size)
        k = k.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)     # no mask: every square sees every square
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(self, n_emb):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_emb, 4*n_emb),
            nn.ReLU(),
            nn.Linear(4*n_emb, n_emb),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_emb, num_heads):
        super().__init__()
        self.multi_attension = MultiHeadAttention(n_emb, num_heads)
        self.feed_forward = FeedForward(n_emb)
        self.ln1 = nn.LayerNorm(n_emb)
        self.ln2 = nn.LayerNorm(n_emb)

    def forward(self, x):
        x = x + self.multi_attension(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x


class Transormer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        n_emb = cfg["n_emb"]
        self.embedding_layer = nn.Embedding(cfg["vocab_size"], n_emb)
        self.positional_embeddings = nn.Embedding(cfg["block_size"], n_emb)
        self.blocks = nn.Sequential(*[Block(n_emb, cfg["num_heads"]) for _ in range(cfg["n_layer"])])
        self.ln_f = nn.LayerNorm(n_emb)
        self.linear = nn.Linear(n_emb, cfg["n_moves"])       # move head: which move would a human play?
        self.legal_head = nn.Linear(n_emb, cfg["n_moves"])   # legal head: training only, kept so the weights load as-is

    def forward(self, idx):
        B, T = idx.shape
        x = self.embedding_layer(idx) + self.positional_embeddings(torch.arange(T, device=idx.device))
        x = self.blocks(x)
        x = self.ln_f(x[:, 0, :])     # the move is read at position 0 (<cls>)
        return self.linear(x)         # (B, n_moves) logits


class ChessModel:
    """One downloaded model (network + its vocabs), with the board encoding used in training."""

    def __init__(self, size):
        config_path, weights_path = model_files(size)
        cfg = json.loads(config_path.read_text())
        self.size, self.step, self.n_params = size, cfg["step"], cfg["n_params"]
        self.history = cfg["history"]

        self.stoi = {t: i for i, t in enumerate(cfg["input_vocab"])}
        self.moves = cfg["moves"]                                        # output id -> UCI move
        self.move_to_id = {mv: i for i, mv in enumerate(self.moves)}     # UCI move -> output id
        self.move_offset = self.stoi[self.moves[0]]                      # input id of a history move = offset + output id
        # mirror[i] = output id of move i mirrored; real_probs = probs[mirror] when the board was mirrored
        self.mirror = torch.tensor([self.move_to_id[mirror_move(chess.Move.from_uci(mv)).uci()] for mv in self.moves])

        self.net = Transormer(cfg)
        self.net.load_state_dict(load_file(weights_path))
        self.net.to(device).eval()

    def encode_position(self, board, history, result, white_elo, black_elo):
        """python-chess board + previous moves -> (81 input ids, flip). Identical to the training script."""
        flip = board.turn == chess.BLACK
        mover = board.turn
        if flip:
            board = board.mirror()                          # ranks flipped, colours swapped, castling / ep / turn too
            history = [mirror_move(m) for m in history]
            white_elo, black_elo = black_elo, white_elo     # the mover's rating goes first
        my_elo, opp_elo = white_elo, black_elo

        ep = board.ep_square if board.has_legal_en_passant() else None
        tokens = [
            "<cls>",
            "<K>" if board.has_kingside_castling_rights(chess.WHITE) else "<noK>",
            "<Q>" if board.has_queenside_castling_rights(chess.WHITE) else "<noQ>",
            "<k>" if board.has_kingside_castling_rights(chess.BLACK) else "<nok>",
            "<q>" if board.has_queenside_castling_rights(chess.BLACK) else "<noq>",
            "<ep_none>" if ep is None else f"<ep_{chess.FILE_NAMES[chess.square_file(ep)]}>",
            RESULT_FOR_MOVER[result][mover],
            my_elo, opp_elo,
        ]
        piece_map = board.piece_map()
        for rank in range(7, -1, -1):                       # rank 8 to rank 1, file a to h
            for file in range(8):
                piece = piece_map.get(chess.square(file, rank))
                tokens.append(piece.symbol() if piece else ".")
        ids = [self.stoi[t] for t in tokens]

        last = history[-self.history:]                      # last 8 moves, oldest first, <no_move> padding in front
        ids += [self.stoi["<no_move>"]] * (self.history - len(last))
        ids += [self.move_offset + self.move_to_id[m.uci()] for m in last]
        return ids, flip

    @torch.no_grad()
    def move_probs(self, board, result, white_elo, black_elo, temperature=1.0):
        """(1968,) probabilities on the CPU, indexed by REAL moves (also when Black is to move).
        result / elos as in training data: "<white_wins>", "<elo_2000>", White's rating first."""
        ids, flip = self.encode_position(board, board.move_stack, result, white_elo, black_elo)
        logits = self.net(torch.tensor([ids], dtype=torch.long, device=device))
        probs = F.softmax(logits[0].float() / temperature, dim=-1).cpu()
        return probs[self.mirror] if flip else probs
