from datasets import load_dataset
import chess
import chess.pgn
import json
import sys
import torch
import numpy as np

PIECE_MAPPING = {'P': 0, 'p': 1, 'N': 2, 'n': 3, 'B': 4, 'b': 5, 'R': 6, 'r': 7, 'Q': 8, 'q': 9, 'K': 10, 'k': 11}


def fen_to_tensor(fen: str) -> torch.Tensor:
    tensor = torch.zeros((12, 8, 8), dtype=torch.float32)

    board = fen.split()[0]
    ranks = board.split('/')

    for i, rank in enumerate(ranks):
        j = 0
        for char in rank:
            if char.isdigit():
                j += int(char)
            else:
                plane = PIECE_MAPPING[char]
                tensor[plane, i, j] = 1
                j += 1

    return tensor


stream = load_dataset("Lichess/chess-position-evaluations", split="train", streaming=True)

N = 1000
arr_evaluations = []
for i, evaluation in enumerate(stream):
    if i >= N:
        break
    if evaluation["cp"]:
        matrix = fen_to_tensor(evaluation["fen"])
        arr_evaluations.append(matrix)

print(arr_evaluations)