#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025-2026 Alp Tartici, Stanford University.
# Licensed under the MIT License.

"""
Score ProteinGym DMS CSVs using inpainting probability NPZ files.

For each row in a DMS CSV (single mutant), extracts:
  - wt_prob:  probability assigned to the wild-type AA at the mutated position
  - mut_prob: probability assigned to the mutant AA at the mutated position
  - llr:      log(mut_prob / wt_prob)

Reads consolidated NPZ files produced by batch_proteingym_inpainting.py and
writes a new scored CSV alongside the NPZ in the output directory.

Usage:
    python scripts/score_proteingym_inpainting.py \
        --npz-dir /path/to/zeroshot/config_129 \
        --csv-dir /path/to/proteingym/dms_subs_processed_single_mutant \
        --output-dir /path/to/zeroshot/config_129/scored
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Amino acid single-letter to three-letter mapping (matches sample_utils.py)
ONE_TO_THREE = {
    'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE',
    'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU',
    'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG',
    'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR',
}

# Fixed index order from sample_utils.py IDX_TO_AA
IDX_TO_AA = [
    'ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE',
    'LYS', 'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER',
    'THR', 'VAL', 'TRP', 'TYR', 'XXX'
]
AA_TO_IDX = {aa: i for i, aa in enumerate(IDX_TO_AA)}


def setup_logging(verbose: bool = False) -> logging.Logger:
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)


def parse_mutant(mutant_str: str):
    """
    Parse a mutant string like 'A25C' into (wt_aa_1letter, position_str, mut_aa_1letter).

    Returns None if the string cannot be parsed.
    """
    if len(mutant_str) < 3:
        return None
    wt = mutant_str[0]
    mut = mutant_str[-1]
    pos = mutant_str[1:-1]
    if not wt.isalpha() or not mut.isalpha() or not pos.isdigit():
        return None
    return wt, pos, mut


def load_npz(npz_path: str, logger: logging.Logger) -> dict:
    """Load NPZ and build a pos_valid -> prob_vector dict."""
    data = np.load(npz_path, allow_pickle=True)

    # Verify AA ordering matches expectation (warn if it differs)
    if 'aa_names' in data:
        stored_aa = list(data['aa_names'])
        if stored_aa != IDX_TO_AA:
            logger.warning(
                f"AA ordering in NPZ differs from expected! "
                f"NPZ: {stored_aa}, expected: {IDX_TO_AA}. "
                f"Using NPZ ordering."
            )
            # Build index map from the stored ordering
            aa_to_idx = {aa: i for i, aa in enumerate(stored_aa)}
        else:
            aa_to_idx = AA_TO_IDX
    else:
        aa_to_idx = AA_TO_IDX

    positions = list(data['positions'])  # e.g. ['A25', 'K30', ...]

    prob_map = {}
    for pos_valid in positions:
        key = f'{pos_valid}_probs'
        if key in data:
            prob_map[pos_valid] = (data[key], aa_to_idx)
        else:
            logger.warning(f"  Key '{key}' not found in NPZ, skipping position {pos_valid}")

    return prob_map


def score_csv(
    csv_path: str,
    npz_path: str,
    output_path: str,
    logger: logging.Logger
) -> bool:
    """
    Score a single DMS CSV using the corresponding NPZ file.

    Returns True on success, False on failure.
    """
    csv_name = Path(csv_path).stem
    logger.info(f"Scoring: {csv_name}")

    # Load CSV
    df = pd.read_csv(csv_path)
    required = ['mutant', 'pos_valid']
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.error(f"  CSV missing required columns: {missing}")
        return False

    # Load NPZ probability map
    logger.info(f"  Loading NPZ: {npz_path}")
    prob_map = load_npz(npz_path, logger)
    logger.info(f"  NPZ contains {len(prob_map)} positions")

    wt_probs = []
    mut_probs = []
    llrs = []
    missing_positions = set()

    for _, row in df.iterrows():
        mutant_str = row['mutant']
        pos_valid = row['pos_valid']

        parsed = parse_mutant(mutant_str)
        if parsed is None:
            logger.warning(f"  Cannot parse mutant '{mutant_str}', filling NaN")
            wt_probs.append(np.nan)
            mut_probs.append(np.nan)
            llrs.append(np.nan)
            continue

        wt_1, _, mut_1 = parsed

        if pos_valid not in prob_map:
            if pos_valid not in missing_positions:
                logger.warning(f"  Position '{pos_valid}' not in NPZ")
                missing_positions.add(pos_valid)
            wt_probs.append(np.nan)
            mut_probs.append(np.nan)
            llrs.append(np.nan)
            continue

        probs, aa_to_idx = prob_map[pos_valid]  # shape [21]

        wt_3 = ONE_TO_THREE.get(wt_1)
        mut_3 = ONE_TO_THREE.get(mut_1)

        if wt_3 is None or wt_3 not in aa_to_idx:
            logger.warning(f"  Unknown WT AA '{wt_1}' in '{mutant_str}', filling NaN")
            wt_probs.append(np.nan)
            mut_probs.append(np.nan)
            llrs.append(np.nan)
            continue

        if mut_3 is None or mut_3 not in aa_to_idx:
            logger.warning(f"  Unknown mutant AA '{mut_1}' in '{mutant_str}', filling NaN")
            wt_probs.append(np.nan)
            mut_probs.append(np.nan)
            llrs.append(np.nan)
            continue

        wt_p = float(probs[aa_to_idx[wt_3]])
        mut_p = float(probs[aa_to_idx[mut_3]])

        wt_probs.append(wt_p)
        mut_probs.append(mut_p)

        # LLR: guard against zero probabilities
        if wt_p > 0 and mut_p > 0:
            llrs.append(np.log(mut_p / wt_p))
        else:
            llrs.append(np.nan)

    df['wt_prob'] = wt_probs
    df['mut_prob'] = mut_probs
    df['llr'] = llrs

    n_scored = df['llr'].notna().sum()
    n_total = len(df)
    logger.info(f"  Scored {n_scored}/{n_total} rows")

    if missing_positions:
        logger.warning(f"  {len(missing_positions)} positions missing from NPZ: {sorted(missing_positions)}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"  Saved scored CSV: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Score ProteinGym CSVs with inpainting LLR scores'
    )
    parser.add_argument(
        '--npz-dir',
        type=str,
        required=True,
        help='Directory containing per-protein subdirs with <name>_inpainting_probs.npz files'
    )
    parser.add_argument(
        '--csv-dir',
        type=str,
        required=True,
        help='Directory containing original ProteinGym DMS CSV files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to write scored CSV files'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()
    logger = setup_logging(args.verbose)

    npz_dir = Path(args.npz_dir)
    csv_dir = Path(args.csv_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all CSV files
    csv_files = sorted(csv_dir.glob('*.csv'))
    if not csv_files:
        logger.error(f"No CSV files found in {csv_dir}")
        sys.exit(1)

    logger.info(f"Found {len(csv_files)} CSV files")
    logger.info(f"NPZ directory: {npz_dir}")
    logger.info(f"Output directory: {output_dir}")

    n_success = 0
    n_failed = 0
    n_skipped = 0

    for csv_path in csv_files:
        csv_name = csv_path.stem

        # NPZ is at <npz_dir>/<csv_name>/<csv_name>_inpainting_probs.npz
        npz_path = npz_dir / csv_name / f"{csv_name}_inpainting_probs.npz"

        if not npz_path.exists():
            logger.warning(f"NPZ not found for {csv_name}, skipping: {npz_path}")
            n_skipped += 1
            continue

        output_path = output_dir / f"{csv_name}_scored.csv"

        success = score_csv(str(csv_path), str(npz_path), str(output_path), logger)
        if success:
            n_success += 1
        else:
            n_failed += 1

    logger.info("=" * 60)
    logger.info(f"Done. Success: {n_success}, Failed: {n_failed}, Skipped (no NPZ): {n_skipped}")
    logger.info("=" * 60)

    if n_failed > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
