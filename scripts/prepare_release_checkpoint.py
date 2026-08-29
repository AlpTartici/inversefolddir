#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025-2026 Alp Tartici, Stanford University.
# Licensed under the MIT License.

"""
prepare_release_checkpoint.py

Turn a training checkpoint into one fit for public distribution.

Training checkpoints carry optimizer state, LR-scheduler state, RNG state, and
absolute paths from the machine that trained them. None of that is used at
inference, it roughly doubles the download, and the paths leak local directory
layout. This script keeps only what the sampler reads and rewrites the paths.

    python scripts/prepare_release_checkpoint.py IN.pt OUT.pt
    python scripts/prepare_release_checkpoint.py IN.pt OUT.pt --dry-run

Verify the result before publishing it:

    python training/inpainting.py --pdb_input 3OGO --model OUT.pt \\
        --mask-ratio 1.0 --output-dir output/verify_release
"""

import argparse
import re
import sys
from pathlib import Path

import torch

# The only keys the design path reads. Confirmed against sample_utils.py and
# inpainting.py -- everything else is training bookkeeping.
REQUIRED_KEYS = [
    "model_state_dict",
    "args",
    "graph_builder_params",
    "model_architecture_params",
]

# Small, useful provenance worth keeping for users and reviewers.
METADATA_KEYS = ["epoch", "model_name", "metrics", "timestamp"]

# Dropped: optimizer_state_dict, scheduler_state_dict, training_state,
# rng_states, is_best.

# Argument values that are filesystem paths get replaced with a relative
# placeholder, so a downloaded checkpoint does not point at directories that
# only existed on the training machine.
PATH_PATTERN = re.compile(r"^(/oak/|/scratch/|/home/|/tmp/|[A-Za-z]:\\)")

PATH_REPLACEMENTS = {
    "af2_chunk_dir": "datasets/af2_pkl",
    "val_gen_pkl": "datasets/cath-4.2/val_entries.pkl",
    "output_dir": "output",
    "split_json": "datasets/cath-4.2/chain_set_splits.json",
    "map_pkl": "datasets/cath-4.2/chain_set_map_with_b_factors_dssp.pkl",
    "data_dir": "datasets",
    "checkpoint_dir": "ckpts",
}


def scrub_paths(args_obj, verbose=True):
    """Replace absolute training-machine paths in the stored args."""
    is_namespace = hasattr(args_obj, "__dict__") and not isinstance(args_obj, dict)
    values = vars(args_obj) if is_namespace else dict(args_obj)

    changed = []
    for key, value in list(values.items()):
        if not isinstance(value, str) or not PATH_PATTERN.match(value):
            continue
        replacement = PATH_REPLACEMENTS.get(key)
        if replacement is None:
            # Unknown path-valued argument: keep only the final component so
            # the value stays meaningful without exposing the directory tree.
            replacement = Path(value).name or "unset"
        values[key] = replacement
        changed.append((key, value, replacement))

    if verbose and changed:
        print(f"\nRewrote {len(changed)} path argument(s):")
        for key, old, new in changed:
            print(f"  {key}")
            print(f"    from: {old}")
            print(f"    to  : {new}")

    if is_namespace:
        for key, value in values.items():
            setattr(args_obj, key, value)
        return args_obj
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="training checkpoint (.pt)")
    parser.add_argument("output", nargs="?", help="path for the release checkpoint")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would change without writing")
    parser.add_argument("--keep-paths", action="store_true",
                        help="do not rewrite absolute paths in args")
    args = parser.parse_args()

    source = Path(args.input)
    if not source.exists():
        print(f"ERROR: {source} not found", file=sys.stderr)
        return 1

    if not args.dry_run and not args.output:
        print("ERROR: give an output path, or use --dry-run", file=sys.stderr)
        return 1

    size_before = source.stat().st_size
    print(f"Reading {source.name}  ({size_before / 1e6:.1f} MB)")

    checkpoint = torch.load(source, map_location="cpu", weights_only=False)

    missing = [k for k in REQUIRED_KEYS if k not in checkpoint]
    if missing:
        print(f"ERROR: checkpoint is missing required key(s): {missing}",
              file=sys.stderr)
        print("This does not look like a checkpoint the sampler can load.",
              file=sys.stderr)
        return 1

    release = {k: checkpoint[k] for k in REQUIRED_KEYS}
    for key in METADATA_KEYS:
        if key in checkpoint:
            release[key] = checkpoint[key]

    dropped = [k for k in checkpoint if k not in release]
    print(f"\nKeeping {len(release)} key(s): {', '.join(release)}")
    print(f"Dropping {len(dropped)} key(s): {', '.join(dropped)}")

    n_params = sum(p.numel() for p in release["model_state_dict"].values()
                   if hasattr(p, "numel"))
    print(f"\nModel: {n_params / 1e6:.1f}M parameters")

    if not args.keep_paths:
        release["args"] = scrub_paths(release["args"])

    if args.dry_run:
        print("\nDry run: nothing written.")
        return 0

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(release, destination)

    size_after = destination.stat().st_size
    print(f"\nWrote {destination}  ({size_after / 1e6:.1f} MB, "
          f"{100 * (1 - size_after / size_before):.0f}% smaller)")
    print("\nVerify it before publishing:")
    print(f"  python training/inpainting.py --pdb_input 3OGO \\")
    print(f"      --model {destination} --mask-ratio 1.0 \\")
    print(f"      --output-dir output/verify_release")
    return 0


if __name__ == "__main__":
    sys.exit(main())
