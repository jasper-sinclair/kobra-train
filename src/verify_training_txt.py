# verify_training_txt.py
# jasper sinclair
# quick NNUE dataset validation

import sys
import json
import os
import time
from collections import Counter


# =========================
# Config
# =========================

def load_config(path="config.json"):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


# =========================
# Helpers
# =========================

def piece_count(board):
    count = 0
    for c in board:
        if c.isalpha():
            count += 1
    return count


# =========================
# Main
# =========================

def main():

    config = load_config()

    dataset_path = config.get("normalized_txt", "training_normalized.txt")
    max_sample = config.get("verify_sample_limit", 500000)

    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]

    print("Checking dataset:", dataset_path)
    print("Sample limit:", max_sample)

    total = 0
    bad_fen = 0
    bad_label = 0
    ctrlz = 0

    label_counts = Counter()
    piece_counts = []
    boards_seen = set()
    duplicates = 0

    file_size = os.path.getsize(dataset_path)

    start = time.time()
    last_print = start

    with open(dataset_path, "rb") as f:

        while True:

            line = f.readline()
            if not line:
                break

            total += 1

            if b'\x1a' in line:
                ctrlz += 1

            try:
                line = line.decode().strip()
            except:
                continue

            if "|" in line:

                fen, label = line.split("|", 1)
                label = label.strip()

            elif "[" in line:

                fen = " ".join(line.split()[:4])
                label = line.split("[")[1].split("]")[0]

            elif '"' in line:

                fen = " ".join(line.split()[:4])
                result_str = line.split('"')[1]

                if result_str == "1-0":
                    label = 1.0
                elif result_str == "0-1":
                    label = 0.0
                elif result_str == "1/2-1/2":
                    label = 0.5
                else:
                    bad_label += 1
                    continue

            else:

                bad_label += 1
                continue

            try:
                label = float(label)
            except:
                bad_label += 1
                continue

            label_counts[label] += 1

            tokens = fen.split()
            if len(tokens) < 1:
                bad_fen += 1
                continue

            board = tokens[0] + " " + tokens[1]
            side = tokens[1] if len(tokens) > 1 else "?"

            pc = piece_count(board)
            piece_counts.append(pc)

            key = board + side
            
            MAX_HASH = 5_000_000
            
            if key in boards_seen:
                duplicates += 1
            else:
                if len(boards_seen) < MAX_HASH:
                    boards_seen.add(key)

            # progress display
            now = time.time()
            if now - last_print > 1:

                pos = f.tell()
                pct = pos / file_size * 100

                elapsed = now - start
                speed = total / elapsed if elapsed > 0 else 0

                print(
                    f"\rProcessed {total:,} "
                    f"({pct:.2f}%) "
                    f"{speed:,.0f} pos/s",
                    end=""
                )

                last_print = now

            if max_sample and total >= max_sample:
                break


    print("\n\nSample size:", total)

    print("\nLabel distribution:")
    for k in sorted(label_counts):
        pct = label_counts[k] / total * 100
        print(f" {k:.3f} : {pct:.2f}%")

    if piece_counts:
        avg = sum(piece_counts) / len(piece_counts)
        print("\nAverage piece count:", round(avg, 2))

    print("\nDuplicates:", duplicates)
    print("Broken FEN:", bad_fen)
    print("Bad labels:", bad_label)
    print("Ctrl-Z markers:", ctrlz)

    elapsed = time.time() - start
    print(f"\nFinished in {elapsed:.1f}s")


if __name__ == "__main__":
    main()