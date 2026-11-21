#!/usr/bin/env python3
"""List the model files that are not whole models.

Finds downloads still in flight (`.fdmdownload`, `.part`, `.tmp`, `.download`)
and `.gguf` files that are empty, tiny or do not start with the GGUF magic,
which no loader will open. Read-only: it prints a `rm` line for each file and
leaves deletion to you.

    python scripts/check_downloads.py ~/Devs/models
"""

import argparse
import datetime
import os
import sys
from pathlib import Path

PARTIAL = (".fdmdownload", ".part", ".tmp", ".download")


def files(root):
    for dirpath, _, names in os.walk(root):
        for name in names:
            yield Path(dirpath) / name


def problem(path):
    if path.name.lower().endswith(PARTIAL):
        return "téléchargement en cours"
    if path.suffix.lower() != ".gguf":
        return None
    if path.stat().st_size < 4:
        return "fichier vide"
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            return "pas un GGUF"
    return None


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("roots", nargs="*", default=[os.path.expanduser("~/Devs/models")])
    args = args.parse_args()

    found = 0
    for root in args.roots:
        for path in sorted(files(root)):
            try:
                why = problem(path)
            except OSError as e:
                print(f"{path}: {e}", file=sys.stderr)
                continue
            if not why:
                continue
            found += 1
            age = datetime.date.today() - datetime.date.fromtimestamp(
                path.stat().st_mtime
            )
            print(f"{path} ({path.stat().st_size >> 20} Mio, {age.days} j): {why}")
            print(f"  rm '{path}'")
    if not found:
        print("aucun fichier incomplet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
