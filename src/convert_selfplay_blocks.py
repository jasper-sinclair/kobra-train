# convert_selfplay_blocks_batch.py

import glob
import math


# sigmoid scaling constant (commonly 400–600 for chess evals)
SCORE_SCALING = 400.0

# clamp scores to avoid extreme probabilities
MAX_CP = 2000

# filters
MIN_PLY = 8
MAX_SCORE = 2000


def result_to_prob(r):
    r = int(r)

    if r == 1:
        return 1.0
    elif r == 0:
        return 0.5
    elif r == -1:
        return 0.0

    return None


def score_to_prob(score_cp):
    score_cp = max(-MAX_CP, min(MAX_CP, score_cp))
    return 1.0 / (1.0 + math.exp(-score_cp / SCORE_SCALING))


def convert_file(input_path):

    output_path = input_path.replace("_plain.txt", "_training.txt")

    fen = None
    score = None
    result = None
    ply = None

    written = 0
    skipped_ply = 0
    skipped_score = 0

    with open(input_path, "r") as fin, open(output_path, "w") as fout:

        for line in fin:

            line = line.strip()

            if line.startswith("fen "):
                fen = line[4:].strip()

            elif line.startswith("score "):
                try:
                    score = int(line.split()[1])
                except:
                    score = None

            elif line.startswith("result "):
                result = line.split()[1]

            elif line.startswith("ply "):
                try:
                    ply = int(line.split()[1])
                except:
                    ply = None

            elif line == "e":

                if fen and ply is not None:

                    # filter opening noise
                    if ply < MIN_PLY:
                        skipped_ply += 1
                        fen = score = result = ply = None
                        continue

                    parts = fen.split()
                    stm = parts[1]  # side to move

                    prob = None

                    if score is not None:

                        # filter extreme scores
                        if abs(score) > MAX_SCORE:
                            skipped_score += 1
                            fen = score = result = ply = None
                            continue

                        # convert score to white perspective
                        if stm == "b":
                            score_adj = -score
                        else:
                            score_adj = score

                        prob = score_to_prob(score_adj)

                    elif result is not None:

                        prob = result_to_prob(result)

                    if prob is not None:

                        fen4 = " ".join(parts[:4])
                        fout.write(f"{fen4} | {prob:.6f}\n")
                        written += 1

                fen = None
                score = None
                result = None
                ply = None

    print("skipped ply <", MIN_PLY, ":", skipped_ply)
    print("skipped abs(score) >", MAX_SCORE, ":", skipped_score)

    return output_path, written


def main():

    files = sorted(glob.glob("*_plain.txt"))

    print("files found:", len(files))

    total = 0

    for f in files:

        print("processing:", f)

        output_path, n = convert_file(f)

        print("written:", n, "→", output_path)

        total += n

    print("\nTOTAL POSITIONS:", total)


if __name__ == "__main__":
    main()