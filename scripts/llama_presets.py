#!/usr/bin/env python3
"""Check that every path a llama.cpp router preset names exists.

Reads `presets.ini` (the file `llama-server --models-preset` takes), and
reports the `model`, `mmproj`, `lora` and `chat-template-file` paths that are
missing, which are exactly the presets the router cannot load.

With `--write`, also appends a section for each GGUF model folder under
`--models` that no existing section already names, so a model added to the
kinetic folder shows up in the router without hand-editing the file. Existing
sections, and their tuned options, are never rewritten.

    python scripts/llama_presets.py
    python scripts/llama_presets.py --write
"""

import argparse
import configparser
import os
import sys
from pathlib import Path

PATHS = ("model", "mmproj", "lora", "chat-template-file")


def sections(path):
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read(path)
    return parser


def check(path):
    parser = sections(path)
    missing = 0
    for section in parser.sections():
        if section == "*":
            continue
        for key in PATHS:
            value = parser[section].get(key)
            if value and not Path(value).exists():
                print(f"{section}: {key} is missing: {value}")
                missing += 1
    return missing


def named(parser):
    paths = set()
    for section in parser.sections():
        for key in ("model", "lora"):
            value = parser[section].get(key)
            if value:
                paths.add(os.path.realpath(value))
    return paths


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument(
        "--presets", default=os.path.expanduser("~/Devs/models/presets.ini")
    )
    args.add_argument(
        "--models",
        default=os.path.expanduser("~/.local/share/kinetic/models/local"),
    )
    args.add_argument("--write", action="store_true")
    args = args.parse_args()

    parser = sections(args.presets)
    missing = check(args.presets)

    if args.write:
        known = named(parser)
        added = []
        for folder in sorted(Path(args.models).iterdir()):
            if not folder.is_dir():
                continue
            weights = sorted(folder.glob("*.gguf"), key=lambda p: -p.stat().st_size)
            weights = [w for w in weights if "mmproj" not in w.name.lower()]
            if not weights or os.path.realpath(weights[0]) in known:
                continue
            projectors = sorted(folder.glob("*mmproj*.gguf"))
            section = folder.name.replace(".", "-").lower()
            lines = [f"[{section}]", f"model = {weights[0]}"]
            if projectors:
                lines.append(f"mmproj = {projectors[0]}")
            added += ["", *lines]
        if added:
            with open(args.presets, "a") as out:
                out.write("\n# Ajoutés par scripts/llama_presets.py\n")
                out.write("\n".join(added) + "\n")
            print(f"{len(added) // 3} présets ajoutés à {args.presets}")
        else:
            print("aucun preset à ajouter")

    if missing:
        print(f"{missing} chemin(s) manquant(s)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
