import struct

with open("training_sparse.bin","rb") as f:

    while True:

        header = f.read(2)
        if not header:
            break

        n_white, n_black = header

        for _ in range(n_white):
            idx = struct.unpack("<H", f.read(2))[0]
            if idx >= 768:
                print("BAD INDEX", idx)
                exit()

        for _ in range(n_black):
            idx = struct.unpack("<H", f.read(2))[0]
            if idx >= 768:
                print("BAD INDEX", idx)
                exit()

        f.read(4)

print("Dataset OK")