# convert_selfplay_blocks.py

import glob
import os


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


def convert_file(input_path):

    output_path = input_path.replace("_plain.txt", "_training.txt")

    fen = None
    result = None
    written = 0
    bad_results = 0

    # large buffer for faster writes on huge datasets
    with open(input_path, "r", buffering=1024*1024) as fin, \
         open(output_path, "w", buffering=1024*1024) as fout:

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

            elif line == "e":

                if fen and result:

                    prob = result_to_prob(result)

                    if prob is not None:

                        fen4 = " ".join(fen.split()[:4])

                        write(f"{fen4} | {prob:.6f}\n")
                        written += 1
                    else:
                        bad_results += 1

                fen = None
                result = None

    print("skipped corrupt results:", bad_results)

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