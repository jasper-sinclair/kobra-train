import torch
import numpy as np
import matplotlib.pyplot as plt

INPUT_SIZE = 768
L1_SIZE = 256   # adjust if needed

PIECE_NAMES = [
    "Pawn",
    "Knight",
    "Bishop",
    "Rook",
    "Queen",
    "King"
]


def load_weights(path):

    model = torch.load(path, map_location="cpu")

    if "fc1.weight" in model:
        w = model["fc1.weight"].numpy()
    else:
        raise RuntimeError("Cannot find fc1 weights")

    return w


def visualize_piece(weights, piece_index):

    start = piece_index * 64
    end = start + 64

    piece_weights = weights[:, start:end]

    piece_weights = np.mean(piece_weights, axis=0)

    board = piece_weights.reshape(8,8)

    plt.imshow(board, cmap="coolwarm")
    plt.colorbar()
    plt.title(PIECE_NAMES[piece_index])
    plt.show()


def main():

    path = "best_model.pt"

    weights = load_weights(path)

    for i in range(6):
        visualize_piece(weights, i)


if __name__ == "__main__":
    main()