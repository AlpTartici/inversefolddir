# Copyright (c) Microsoft Corporation.
# Copyright (c) 2026 Alp Tartici and contributors.
# Licensed under the MIT License.

"""Filesystem path resolution for datasets and model checkpoints.

Paths are resolved in this order:

1. The ``IFD_DATA_DIR`` / ``IFD_CKPT_DIR`` environment variables, if set.
2. ``datasets/`` and ``ckpts/`` directories alongside this file.

This keeps the repository free of machine-specific absolute paths. To point at
data held outside the repository, export the variables before running::

    export IFD_DATA_DIR=/path/to/datasets
    export IFD_CKPT_DIR=/path/to/ckpts
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def data_dir() -> Path:
    """Root directory holding the datasets (``cath-4.2/``, ``af2_pkl/``, ...)."""
    env = os.environ.get('IFD_DATA_DIR')
    return Path(env).expanduser() if env else REPO_ROOT / 'datasets'


def ckpt_dir() -> Path:
    """Root directory holding model checkpoints."""
    env = os.environ.get('IFD_CKPT_DIR')
    return Path(env).expanduser() if env else REPO_ROOT / 'ckpts'


def data_path(*parts: str) -> str:
    """Resolve a path beneath :func:`data_dir`, e.g. ``data_path('cath-4.2')``."""
    return str(data_dir().joinpath(*parts))


def ckpt_path(*parts: str) -> str:
    """Resolve a path beneath :func:`ckpt_dir`, e.g. ``ckpt_path('inverse_folddir_model.pt')``."""
    return str(ckpt_dir().joinpath(*parts))


# Frequently referenced dataset files.
CATH_DIR = 'cath-4.2'
AF2_CHUNK_DIRNAME = 'af2_pkl'
VAL_GEN_PKL_NAME = 'val_entries_with_less_than_300_continuous_coords.pkl'
CHAIN_SET_MAP_NAME = 'chain_set_map_with_b_factors_dssp.pkl'


def default_af2_chunk_dir() -> str:
    return data_path(AF2_CHUNK_DIRNAME)


def default_val_gen_pkl() -> str:
    return data_path(CATH_DIR, VAL_GEN_PKL_NAME)


def default_chain_set_map() -> str:
    return data_path(CATH_DIR, CHAIN_SET_MAP_NAME)
