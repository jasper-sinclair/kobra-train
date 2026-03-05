import random

bad = 0
total = 0

with open("training.txt") as f:

    for line in f:

        if "|" not in line:
            continue

        fen, val = line.split("|",1)
        label = float(val.strip())

        stm = fen.split()[1]

        if stm == "w" and label < 0.2:
            bad += 1

        if stm == "b" and label > 0.8:
            bad += 1

        total += 1

        if total > 20000:
            break

print("checked:", total)
print("suspicious:", bad)

if bad > total * 0.2:
    print("⚠ perspective likely broken")
else:
    print("dataset looks healthy")