# shuffle_dataset.py
# jasper sinclair

import json
import random
import os
import time


# =========================
# Config
# =========================

def load_config(path="config.json"):
    if not os.path.exists(path):
        print("Config not found, using defaults.")
        return {}
    with open(path) as f:
        return json.load(f)


# =========================
# Main
# =========================

def main():

    config = load_config()

    input_file = config.get("training_txt", "training.txt")
    output_file = config.get("shuffle_output", "training_shuffled.txt")

    print("Input :", input_file)
    print("Output:", output_file)

    # ---- load file ----
    print("\nLoading dataset...")

    start = time.time()

    with open(input_file) as f:
        lines = f.readlines()

    count = len(lines)

    elapsed = time.time() - start
    print(f"Loaded {count:,} positions in {elapsed:.1f}s")

    # ---- shuffle ----
    print("\nShuffling dataset...")

    start = time.time()

    random.shuffle(lines)

    elapsed = time.time() - start
    print(f"Shuffle complete in {elapsed:.1f}s")

    # ---- write output ----
    print("\nWriting shuffled dataset...")

    start = time.time()

    with open(output_file, "w") as f:

        for i, line in enumerate(lines):

            f.write(line)

            if (i + 1) % 1_000_000 == 0:
                pct = (i + 1) / count * 100
                print(f"{i+1:,} / {count:,} ({pct:.1f}%)")

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()