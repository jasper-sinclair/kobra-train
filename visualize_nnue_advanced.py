import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import json

INPUT_SIZE = 768
PIECES = ["Pawn", "Knight", "Bishop", "Rook", "Queen", "King"]


def load_model(path):

    model = torch.load(path, map_location="cpu")

    if isinstance(model, dict) and "fc1.weight" in model:
        weights = model["fc1.weight"].numpy()
    else:
        raise RuntimeError("Could not find fc1 weights")

    return weights


def visualize_neuron(weights, neuron):

    fig, axes = plt.subplots(2, 3, figsize=(10, 6))

    for piece in range(6):

        start = piece * 64
        end = start + 64

        data = weights[neuron, start:end]
        board = data.reshape(8, 8)

        ax = axes[piece // 3][piece % 3]
        im = ax.imshow(board, cmap="coolwarm")

        ax.set_title(PIECES[piece])
        ax.axis("off")

    fig.colorbar(im)
    fig.suptitle(f"Neuron {neuron}")

    plt.show()


def analyze_network(weights):

    neuron_strength = np.mean(np.abs(weights), axis=1)

    dead = np.sum(neuron_strength < 1e-5)

    print("Neurons:", len(neuron_strength))
    print("Dead neurons:", dead)

    plt.hist(neuron_strength, bins=50)
    plt.title("Neuron strength distribution")
    plt.show()

def neuron_specialization(weights):

    print("\nNeuron specialization:\n")

    for n in range(weights.shape[0]):

        scores = []

        for piece in range(6):

            start = piece * 64
            end = start + 64

            val = np.mean(np.abs(weights[n, start:end]))
            scores.append(val)

        best_piece = np.argmax(scores)

        print(f"Neuron {n:3d} -> {PIECES[best_piece]}")
        
def interactive_view(weights):

    while True:

        try:
            n = int(input("Neuron index (-1 to quit): "))

            if n < 0:
                break

            visualize_neuron(weights, n)

        except Exception as e:
            print("Error:", e)


def global_heatmap(weights):

    heatmap = np.zeros((6, 64))

    for piece in range(6):

        start = piece * 64
        end = start + 64

        piece_weights = weights[:, start:end]

        heatmap[piece] = np.mean(np.abs(piece_weights), axis=0)

    fig, axes = plt.subplots(2, 3, figsize=(10,6))

    for piece in range(6):

        board = heatmap[piece].reshape(8,8)

        ax = axes[piece//3][piece%3]

        im = ax.imshow(board, cmap="inferno")

        ax.set_title(PIECES[piece])
        ax.axis("off")

    fig.colorbar(im)

    plt.suptitle("Global NNUE Feature Importance")

    plt.show()
    
    
def main():

    # Load config.json
    with open("config.json") as f:
        config = json.load(f)

    path = config.get("best_model_path", "best_model.pt")

    print("Loading model:", path)

    weights = load_model(path)

    analyze_network(weights)
    neuron_specialization(weights)
    global_heatmap(weights)
    
    interactive_view(weights)


if __name__ == "__main__":
    main()