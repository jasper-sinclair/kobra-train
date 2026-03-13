# convert_selfplay_blocks.py

import glob
import math


def result_to_prob(r):
    try:
        r = int(r.strip())
    except:
        return None

    if r == 1:
        return 1.0
    elif r == 0:
        return 0.5
    elif r == -1:
        return 0.0

    return None


def score_to_prob(score_str):
    try:
        score = int(score_str)
    except:
        return None

    # Clamp mate scores so sigmoid doesn't explode
    score = max(-4000, min(4000, score))

    return 1.0 / (1.0 + math.exp(-score / 400.0))


def convert_file(input_path):

    output_path = input_path.replace("_plain.txt", "_training.txt")

    fen = None
    result = None
    score = None

    written = 0
    bad_results = 0
    bad_scores = 0

    with open(input_path, "r", buffering=1024 * 1024) as fin, \
         open(output_path, "w", buffering=1024 * 1024) as fout:

        write = fout.write

        for line in fin:

            line = line.strip()

            if line.startswith("fen "):
                fen = line[4:].strip()

            elif line.startswith("result "):

                parts = line.split()
                if len(parts) >= 2:
                    result = parts[1].strip()
                else:
                    result = None

            elif line.startswith("score "):

                parts = line.split()
                if len(parts) >= 2:
                    score = parts[1].strip()
                else:
                    score = None

            elif line == "e":

                if fen and result and score:

                    p_result = result_to_prob(result)
                    p_eval = score_to_prob(score)

                    # use eval only (STM perspective already)
                    if p_eval is not None:

                        tokens = fen.split()
                        fen4 = " ".join(tokens[:4])

                        target = p_eval

                        write(f"{fen4} | {target:.6f}\n")
                        written += 1

                    else:
                        if p_result is None:
                            bad_results += 1
                        if p_eval is None:
                            bad_scores += 1

                fen = None
                result = None
                score = None

    print("skipped corrupt results:", bad_results)
    print("skipped corrupt scores:", bad_scores)

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