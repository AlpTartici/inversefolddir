#!/usr/bin/env python3
"""
add_license_headers.py

Add SPDX license headers to Python files that are missing them.

The original Microsoft repo established a per-file header convention. Files
that came from that repo keep the Microsoft copyright; files written at
Stanford get their own copyright line. Both are MIT, matching the root
LICENSE, so this is a consistency pass rather than a licensing change.

Provenance is determined from git: if a path exists in the last upstream
commit, it is Microsoft-origin.

Usage:
    python scripts/add_license_headers.py --dry-run
    python scripts/add_license_headers.py
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Last commit authored by the original Microsoft repo maintainer. Anything
# present at this commit predates the Stanford work.
UPSTREAM_REF = "5193f80"

MSR_HEADER = """# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""

STANFORD_HEADER = """# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025-2026 Alp Tartici, Stanford University.
# Licensed under the MIT License.
"""

HEADER_MARKERS = ("copyright (c) microsoft", "licensed under the mit")

# Vendored or generated code we should not annotate.
SKIP_DIRS = {"deletable", "datasets", ".git", "__pycache__", "gvp"}


def run(cmd):
    """Run a command given as an argument list, returning stdout.

    Deliberately not shell=True: the arguments include repository paths, and
    interpolating those into a shell string is a command-injection vector for
    any path containing shell metacharacters.
    """
    return subprocess.run(
        cmd, capture_output=True, text=True, check=False
    ).stdout.strip()


def tracked_python_files():
    out = run(["git", "ls-files", "*.py"])
    return [Path(p) for p in out.split("\n") if p]


def has_header(path):
    try:
        head = path.read_text(errors="ignore")[:600].lower()
    except OSError:
        return True  # unreadable: leave it alone
    return any(marker in head for marker in HEADER_MARKERS)


def is_upstream(path):
    """True if the file existed in the original Microsoft repo."""
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{UPSTREAM_REF}:{path}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
    )
    return result.returncode == 0


def insert_header(text, header):
    """
    Place the header at the top, but below a shebang or coding line so the
    file stays executable and correctly decoded.
    """
    lines = text.split("\n")
    insert_at = 0
    if lines and lines[0].startswith("#!"):
        insert_at = 1
    if len(lines) > insert_at and "coding" in lines[insert_at] and lines[insert_at].startswith("#"):
        insert_at += 1

    before = lines[:insert_at]
    after = lines[insert_at:]
    # Keep exactly one blank line between the header and what follows.
    while after and after[0].strip() == "":
        after.pop(0)
    return "\n".join(before + header.rstrip("\n").split("\n") + [""] + after)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change without writing files.")
    args = parser.parse_args()

    files = tracked_python_files()
    msr, stanford, skipped = [], [], 0

    for path in files:
        if set(path.parts) & SKIP_DIRS:
            continue
        if not path.exists() or has_header(path):
            skipped += 1
            continue
        (msr if is_upstream(path) else stanford).append(path)

    print(f"Scanned {len(files)} tracked Python files")
    print(f"  already have a header : {skipped}")
    print(f"  need Microsoft header : {len(msr)}")
    print(f"  need Stanford header  : {len(stanford)}")

    if args.dry_run:
        for path in msr:
            print(f"  [MSR]      {path}")
        for path in stanford:
            print(f"  [STANFORD] {path}")
        print("\nDry run: no files modified.")
        return 0

    changed = 0
    for path, header in [(p, MSR_HEADER) for p in msr] + \
                        [(p, STANFORD_HEADER) for p in stanford]:
        try:
            original = path.read_text()
            path.write_text(insert_header(original, header))
            changed += 1
        except OSError as exc:
            print(f"  ERROR writing {path}: {exc}", file=sys.stderr)

    print(f"\nAdded headers to {changed} files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
