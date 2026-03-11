# normalize_dataset.py
# jasper sinclair
# Normalize mixed chess training datasets into:
#
#     FEN | result
#
# result [0,1]
# Config is read from config.json.
# Designed for very large datasets (10M+ positions)

import json
import math
import os
import random

# -----------------------------
# Config loader
# -----------------------------

def load_config(path="config.json"):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


# -----------------------------
# Centipawn probability
# -----------------------------

def cp_to_prob(cp):
    return 1.0 / (1.0 + math.exp(-cp / 400.0))


# -----------------------------
# Parse dataset line
# -----------------------------

def parse_line(line):

    line = line.strip()
    if not line:
        return None, None

    # --------------------------
    # FEN | result
    # --------------------------
    if "|" in line:
        try:
            fen_part, score_part = line.split("|", 1)
            fen = " ".join(fen_part.split()[:4])
            result = float(score_part.strip())
            return fen, max(0.0, min(1.0, result))
        except:
            return None, None

    # --------------------------
    # EPD result "1-0"
    # --------------------------
    if '"' in line:
        try:
            r = line.split('"')[1]

            if r == "1-0":
                result = 1.0
            elif r == "0-1":
                result = 0.0
            elif r == "1/2-1/2":
                result = 0.5
            else:
                return None, None

            fen = " ".join(line.split()[:4])
            return fen, result
        except:
            return None, None

    # --------------------------
    # bracket result [0.5]
    # --------------------------
    if "[" in line:
        try:
            result = float(line.split("[")[1].split("]")[0])
            fen = " ".join(line.split()[:4])
            return fen, result
        except:
            return None, None

    # --------------------------
    # numeric eval
    # --------------------------
    tokens = line.split()

    if len(tokens) >= 5:
        try:
            val = float(tokens[-1])
            result = val if 0 <= val <= 1 else cp_to_prob(val)
            fen = " ".join(tokens[:4])
            return fen, result
        except:
            pass

    return None, None

# -----------------------------
# Normalize dataset
# -----------------------------

def normalize(input_path, output_path, dataset_sample_limit, skip_invalid):

    seen = set()
    MAX_HASH = 5_000_000

    valid = 0
    invalid = 0

    draw_drop_rate = config.get("draw_drop_rate", 0.0)

    with open(input_path) as fin, open(output_path, "w") as fout:

        for i, line in enumerate(fin, 1):

            # Optional progress reporting for very large datasets
            if i % 5_000_000 == 0:
                print(f"{i:,} lines processed | {valid:,} kept")

            # Optional sample limit
            if dataset_sample_limit and valid >= dataset_sample_limit:
                break

            fen, result = parse_line(line)

            if fen is None:
                invalid += 1
                if skip_invalid:
                    continue
                else:
                    raise ValueError("Invalid line")

            # -----------------------------
            # Optional: reduce draw frequency
            # -----------------------------
            if result == 0.5 and random.random() < draw_drop_rate:
                continue

            # Split FEN once for speed
            parts = fen.split()

            # Deduplicate by board + side-to-move
            key = " ".join(parts[:2])

            if key in seen:
                continue

            if len(seen) < MAX_HASH:
                seen.add(key)

            fout.write(f"{fen} | {result:.6f}\n")

            valid += 1

            if valid % 1_000_000 == 0:
                print(f"{valid:,} positions normalized")

    return valid, invalid


# -----------------------------
# Entry point
# -----------------------------

if __name__ == "__main__":

    config = load_config()

    input_path = config.get("raw_training_txt", "training.txt")
    output_path = config.get("normalized_txt", "training_normalized.txt")

    dataset_sample_limit = config.get("dataset_sample_limit", 0)
    skip_invalid = config.get("skip_invalid", True)

    print("Input:", input_path)
    print("Output:", output_path)

    valid, invalid = normalize(
        input_path,
        output_path,
        dataset_sample_limit,
        skip_invalid
    )

    print("\nNormalization complete")
    print("Valid positions:", valid)
    print("Skipped:", invalid)