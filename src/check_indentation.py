# check_indentation.py
# jasper sinclair

import os
import re

def check_file(path):
    has_tabs = False
    has_spaces = False
    mixed_lines = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f, 1):
            if line.startswith("\t"):
                has_tabs = True
            if re.match(r"^ {1,8}\S", line):
                has_spaces = True
            if re.match(r"^\t+ +", line) or re.match(r"^ +\t+", line):
                mixed_lines.append(i)

    return has_tabs, has_spaces, mixed_lines


def scan_directory(root):
    print("Scanning:", root)
    print()

    for dirpath, _, filenames in os.walk(root):
        if "nnue_env" in dirpath:
            continue
            
            for name in filenames:
                if name.endswith(".py"):
                    path = os.path.join(dirpath, name)

                has_tabs, has_spaces, mixed = check_file(path)

                if mixed or (has_tabs and has_spaces):
                    print(f"⚠ {path}")

                    if has_tabs and has_spaces:
                        print("  File contains BOTH tabs and spaces")

                    if mixed:
                        print("  Mixed indentation lines:", mixed)

                    print()


if __name__ == "__main__":
    scan_directory(".")