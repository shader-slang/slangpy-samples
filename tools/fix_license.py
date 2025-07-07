#!/usr/bin/env python

"""
A utility for adding SPDX license identifer to source files.
"""

from __future__ import annotations
import sys
import os
import re
import argparse
from typing import Any, Sequence
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.resolve()

INCLUDE_PATHS = [
    "examples",
    "experiments",
    "tools",
    "tutorials",
]

EXCLUDE_PATHS = []

EXTENSIONS = ["h", "cpp", "slang", "slangh", "py"]

SPDX_REMOVE_REGEX = re.compile(r"^((// )|(# ))SPDX-License-Identifier: .*\n\n")
SPDX_IDENTIFIER = "SPDX-License-Identifier: Apache-2.0"
SPDX_IDENTIFIER_C_LIKE = f"// {SPDX_IDENTIFIER}\n"
SPDX_IDENTIFIER_PYTHON = f"# {SPDX_IDENTIFIER}\n"


def list_files(
    root: Path,
    include: Sequence[Path | str],
    exclude: Sequence[Path | str],
    extensions: Sequence[str] = [],
):
    # collect files
    files: list[str] = []
    for path in include:
        path = os.path.normpath(root / path)
        if os.path.isdir(path):
            for dirpath, _, fnames in os.walk(path):
                fpaths = [os.path.join(dirpath, fname) for fname in fnames]
                for f in fpaths:
                    ext = os.path.splitext(f)[1][1:]
                    if ext in extensions:
                        files.append(f)
        elif os.path.isfile(path):
            files.append(path)
    # filter excluded paths
    exclude = [os.path.normpath(root / p) for p in exclude]
    files = [f for f in files if not any(f.startswith(p) for p in exclude)]
    return files


def add_spdx_identifier(path: str, text: str):
    if not SPDX_IDENTIFIER in text:
        # remove lines containing SPDX identifer
        text = SPDX_REMOVE_REGEX.sub("", text)
        # add SPDX_IDENTIFIER to the top of the file
        identifier = SPDX_IDENTIFIER_PYTHON if path.endswith(".py") else SPDX_IDENTIFIER_C_LIKE
        # add extra newline if file is not empty
        if text != "":
            identifier += "\n"
        return identifier + text
    return text


def process_file(path: str, dry_run: bool = False):
    ext = os.path.splitext(path)[1][1:]
    if not ext in EXTENSIONS:
        return

    text = open(path, "r").read()
    edited = add_spdx_identifier(path, text)
    if edited != text:
        if dry_run:
            print(edited[0:100])
        else:
            print(f"Writing file '{path}'")
            open(path, "w").write(edited)


def run(args: Any):
    files = list_files(
        root=PROJECT_DIR,
        include=INCLUDE_PATHS,
        exclude=EXCLUDE_PATHS,
        extensions=EXTENSIONS,
    )

    for file in files:
        process_file(file, args.dry_run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-d",
        "--dry-run",
        action="store_true",
        default=False,
        help="run without writing files",
    )

    args = parser.parse_args()

    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
