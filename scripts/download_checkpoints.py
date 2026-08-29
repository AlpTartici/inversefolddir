#!/usr/bin/env python3
# Copyright (c) 2026 Alp Tartici and contributors.
# Licensed under the MIT License.

"""Download pretrained InverseFoldDir checkpoints from the Hugging Face Hub.

Model weights are distributed separately from this repository because they are
too large to version in git. By default the checkpoint is written to ``ckpts/``
alongside this repository, which is where the sampling and inpainting entry
points look for it.

Examples
--------
Download the default checkpoint::

    python scripts/download_checkpoints.py

Download every available checkpoint::

    python scripts/download_checkpoints.py --all

Download to a custom location::

    python scripts/download_checkpoints.py --output-dir /path/to/ckpts

The download location can also be set with the ``IFD_CKPT_DIR`` environment
variable, which the rest of the codebase honours as well.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from paths import ckpt_dir  # noqa: E402

# Hugging Face repository holding the released weights.
HF_REPO_ID = "AlpTartici/inversefolddir"

# Checkpoint filename served from that repository. One checkpoint is published:
# epoch 1353 of the run described by configs/released_model.json, prepared for
# distribution by scripts/prepare_release_checkpoint.py. Roughly 38 MB.
DEFAULT_CHECKPOINT = "inverse_folddir_model.pt"

AVAILABLE_CHECKPOINTS = [
    DEFAULT_CHECKPOINT,
]


def download(filenames, output_dir, repo_id=HF_REPO_ID):
    """Fetch ``filenames`` from ``repo_id`` into ``output_dir``."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit(
            "huggingface_hub is required to download checkpoints.\n"
            "Install it with:  pip install huggingface_hub"
        )

    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    for name in filenames:
        print(f"Downloading {name} from {repo_id} ...", flush=True)
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=name,
                local_dir=str(output_dir),
            )
        except Exception as exc:  # network, auth, or missing-file errors
            sys.exit(
                f"Failed to download {name}: {exc}\n\n"
                f"Check that https://huggingface.co/{repo_id} exists and is public, "
                "and that you have network access."
            )
        size_mb = Path(path).stat().st_size / (1024 * 1024)
        print(f"  saved to {path} ({size_mb:.0f} MB)", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Download pretrained InverseFoldDir model checkpoints.",
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help=f"Checkpoint filename to download (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download every available checkpoint instead of just one.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Destination directory (default: $IFD_CKPT_DIR, else ckpts/ in the repository).",
    )
    parser.add_argument(
        "--repo-id",
        default=HF_REPO_ID,
        help=f"Hugging Face repository to download from (default: {HF_REPO_ID})",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the available checkpoints and exit.",
    )
    args = parser.parse_args()

    if args.list:
        print("Available checkpoints:")
        for name in AVAILABLE_CHECKPOINTS:
            default = "  (default)" if name == DEFAULT_CHECKPOINT else ""
            print(f"  {name}{default}")
        return

    filenames = AVAILABLE_CHECKPOINTS if args.all else [args.checkpoint]
    output_dir = args.output_dir or ckpt_dir()
    download(filenames, output_dir, repo_id=args.repo_id)
    print("\nDone. Point --model at the downloaded file, for example:")
    print(f"  python training/inpainting.py --pdb_input 3OGO --model {Path(output_dir) / filenames[0]}")


if __name__ == "__main__":
    main()
