#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025-2026 Alp Tartici, Stanford University.
# Licensed under the MIT License.

"""
Convert inpainting JSON results to CSV format.

Parses output directory trees produced by run_expval_nanobody.sh (or any sweep
using the same config_N/tier_M/rep_P directory structure) and writes one row per
replicate with the following columns:

    structure_idx, structure_name, config_num, tier, rep,
    length, predicted_sequence, true_sequence, accuracy,
    seq_sim_blosum_frac, seq_sim_blosum_mean,
    tier_1_recovery, tier_2_recovery, tier_3_recovery,
    tier_4_recovery, tier_5_recovery

Tier recovery (tier_K_recovery): fraction of all positions accumulated up to
tier K that the generated sequence predicts correctly.  This is computed
regardless of whether the position was fixed or sampled, using the 1-indexed
PDB residue numbers stored in the tier .txt files.

Position mapping with flank filtering
--------------------------------------
When --filter_out_missing_flanks is active, inpainting.py trims leading/trailing
residues and saves the offset in the JSON under:
    flank_filtering.start_offset   (int, 0 if not present)
A tier residue at 1-indexed position P in the *full* sequence corresponds to
0-indexed position (P - 1 - start_offset) in the trimmed output sequence.
Positions that fall outside the trimmed range are skipped (not counted as
incorrect).
"""

import argparse
import csv
import glob
import json
import os
import re
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Tier file parsing
# ---------------------------------------------------------------------------

def load_tier_positions(tier_dir: str, max_tier: int = 5) -> dict:
    """
    Load cumulative tier positions from tier_0.txt … tier_N.txt.

    Returns a dict: {tier_num (int): [list of 1-indexed positions (int)]}.
    The list for tier K contains ALL positions accumulated from tier_0 through
    tier_K (cumulative).

    Positions are extracted by stripping the amino-acid prefix from tokens like
    'C23', 'R36' → 23, 36.  Tier 0 is assumed to be empty (no fixed positions).
    """
    cumulative = []
    tier_positions = {}

    for t in range(0, max_tier + 1):
        tier_file = os.path.join(tier_dir, f"tier_{t}.txt")
        if not os.path.isfile(tier_file):
            tier_positions[t] = list(cumulative)
            continue

        content = open(tier_file).read().strip()
        if not content:
            tier_positions[t] = list(cumulative)
            continue

        new_positions = []
        for token in re.split(r'[,\s]+', content):
            token = token.strip()
            if not token:
                continue
            # Strip leading letters (amino-acid prefix), keep digits
            m = re.match(r'^[A-Za-z]*(\d+)$', token)
            if m:
                pos = int(m.group(1))
                if pos not in cumulative:
                    new_positions.append(pos)

        cumulative.extend(new_positions)
        tier_positions[t] = list(cumulative)

    return tier_positions


# ---------------------------------------------------------------------------
# Sequence string helpers
# ---------------------------------------------------------------------------

def _compute_accuracy(pred: str, true: str) -> float:
    """Overall per-position accuracy (0–100)."""
    if not pred or not true:
        return 0.0
    n = min(len(pred), len(true))
    matches = sum(1 for i in range(n) if pred[i] == true[i])
    return (matches / len(true)) * 100.0


def _tier_recovery(pred: str, true: str, positions_1indexed: list,
                   start_offset: int, seq_length: int,
                   pdb_res_offset: int = 0) -> Optional[float]:
    """
    Fraction of the given tier positions that are correctly predicted.

    Args:
        pred:              predicted sequence string (trimmed length)
        true:              true sequence string (trimmed length)
        positions_1indexed: list of raw numbers from tier files (PDB residue
                            numbers if the PDB doesn't start at 1)
        start_offset:      residues trimmed from the N-terminus by flank filter
        seq_length:        length of pred/true (trimmed)
        pdb_res_offset:    (first_PDB_residue_number - 1).  Subtracting this
                           converts a PDB residue number to a 1-based sequence
                           index.  E.g. if PDB starts at residue 2, pass 1.

    Returns float in [0,1] or None if no positions fall within range.
    """
    if not positions_1indexed:
        return None

    correct = 0
    total = 0
    for pdb_pos in positions_1indexed:
        # PDB res → 1-based seq idx → 0-based idx in trimmed sequence
        idx = (pdb_pos - pdb_res_offset) - 1 - start_offset
        if idx < 0 or idx >= seq_length:
            continue  # outside trimmed range – skip
        total += 1
        if idx < len(pred) and idx < len(true) and pred[idx] == true[idx]:
            correct += 1

    if total == 0:
        return None
    return correct / total


# ---------------------------------------------------------------------------
# Path parsing
# ---------------------------------------------------------------------------

def _parse_path_metadata(json_path: str, base_dir: str):
    """
    Extract config_num, tier, rep from the relative path.

    Expected structure:  <base_dir>/config_N/tier_M/rep_P/inpainting_results.json
    or (num_repeats==1): <base_dir>/config_N/tier_M/inpainting_results.json

    Returns (structure_name, config_num, tier, rep) where unknowns are None.
    """
    rel = os.path.relpath(json_path, base_dir)
    parts = Path(rel).parts  # e.g. ('config_129', 'tier_2', 'rep_1', 'inpainting_results.json')

    config_num = None
    tier = None
    rep = None

    for part in parts:
        m = re.match(r'^config_(\d+)$', part)
        if m:
            config_num = int(m.group(1))
            continue
        m = re.match(r'^tier_(\d+)$', part)
        if m:
            tier = int(m.group(1))
            continue
        m = re.match(r'^rep_(\d+)$', part)
        if m:
            rep = int(m.group(1))
            continue

    # Structure name = path without filename
    structure_name = str(Path(rel).parent)

    return structure_name, config_num, tier, rep


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_directory(base_dir: str, output_csv: str, tier_dir: str = None,
                      max_tier: int = 5, reference_name: str = None,
                      pdb_res_offset: int = 0):
    """
    Walk base_dir for inpainting_results.json files and write a CSV.

    Args:
        base_dir:        Root sweep output directory.
        output_csv:      Destination CSV path.
        tier_dir:        Directory with tier_0.txt … tier_N.txt.  If None,
                         tier recovery columns are omitted.
        max_tier:        Highest tier to compute recovery for (default 5).
        reference_name:  If provided, write this value in the 'structure_name'
                         column so that evaluation_pipeline.py can look up the
                         correct reference PDB by stem name.  The path-based
                         identifier is stored separately in 'run_path'.
                         If None, structure_name and run_path are identical.
        pdb_res_offset:  (first_PDB_residue_number - 1).  Used to convert the
                         PDB residue numbers in tier files to 1-based sequence
                         indices.  E.g. if the PDB starts at residue 2, pass 1.
                         Default 0 (PDB starts at residue 1).
    """
    json_files = sorted(
        glob.glob(os.path.join(base_dir, '**/inpainting_results.json'), recursive=True)
    )

    if not json_files:
        print(f"No inpainting_results.json files found under: {base_dir}")
        return

    print(f"Found {len(json_files)} result files")

    # Load tier positions once
    tier_positions = {}
    if tier_dir and os.path.isdir(tier_dir):
        tier_positions = load_tier_positions(tier_dir, max_tier)
        print(f"Loaded tier positions from: {tier_dir}")
        for t, pos in tier_positions.items():
            print(f"  tier_{t}: {len(pos)} cumulative positions")
    else:
        if tier_dir:
            print(f"WARNING: tier_dir not found: {tier_dir}  – recovery columns will be None")

    # Build fieldnames
    tier_recovery_cols = [f"tier_{t}_recovery" for t in range(1, max_tier + 1)]
    fieldnames = [
        'structure_idx', 'structure_name', 'run_path',
        'config_num', 'tier', 'rep',
        'length', 'predicted_sequence', 'true_sequence', 'accuracy',
        'seq_sim_blosum_frac', 'seq_sim_blosum_mean',
    ] + tier_recovery_cols

    rows = []
    for idx, json_path in enumerate(json_files):
        structure_name, config_num, tier_num, rep = _parse_path_metadata(json_path, base_dir)

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"  ERROR reading {json_path}: {e}")
            continue

        pred_seq = data.get('predicted_aa_single', '')
        true_seq = data.get('true_aa_single', '')

        if not pred_seq:
            print(f"  WARNING: no predicted_aa_single in {json_path}, skipping")
            continue

        length = len(pred_seq)
        accuracy = _compute_accuracy(pred_seq, true_seq) if true_seq else None

        # BLOSUM metrics (written by inpainting.py since our patch)
        blosum_frac = data.get('seq_sim_blosum_frac')
        blosum_mean = data.get('seq_sim_blosum_mean')

        # Flank filtering offset
        flank = data.get('flank_filtering', {}) or {}
        start_offset = flank.get('start_offset', 0) if isinstance(flank, dict) else 0

        # Tier recovery columns
        tier_recovery = {}
        for t in range(1, max_tier + 1):
            col = f"tier_{t}_recovery"
            positions = tier_positions.get(t, [])
            if positions and true_seq:
                tier_recovery[col] = _tier_recovery(
                    pred_seq, true_seq, positions, start_offset, length,
                    pdb_res_offset=pdb_res_offset
                )
            else:
                tier_recovery[col] = None

        row = {
            'structure_idx': idx,
            'structure_name': reference_name if reference_name else structure_name,
            'run_path': structure_name,
            'config_num': config_num,
            'tier': tier_num,
            'rep': rep,
            'length': length,
            'predicted_sequence': pred_seq,
            'true_sequence': true_seq,
            'accuracy': f"{accuracy:.4f}" if accuracy is not None else '',
            'seq_sim_blosum_frac': f"{blosum_frac:.6f}" if blosum_frac is not None else '',
            'seq_sim_blosum_mean': f"{blosum_mean:.6f}" if blosum_mean is not None else '',
        }
        row.update({col: (f"{v:.4f}" if v is not None else '') for col, v in tier_recovery.items()})

        rows.append(row)
        print(f"  [{idx+1}/{len(json_files)}] {structure_name}  "
              f"acc={row['accuracy']}  "
              f"blosum_frac={row['seq_sim_blosum_frac']}")

    if not rows:
        print("No valid results to write.")
        return

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to: {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert inpainting sweep results to a single CSV.'
    )
    parser.add_argument('directory',
                        help='Root sweep output directory (contains config_*/tier_*/rep_*/ trees).')
    parser.add_argument('-o', '--output', default='inpainting_sequences.csv',
                        help='Output CSV file path.')
    parser.add_argument('--tier_dir', type=str, default=None,
                        help='Directory containing tier_0.txt … tier_5.txt for recovery metrics.')
    parser.add_argument('--max_tier', type=int, default=5,
                        help='Highest tier number to report recovery for (default: 5).')
    parser.add_argument('--pdb_res_offset', type=int, default=0,
                        help='first_PDB_residue_number - 1.  Converts PDB residue numbers '
                             'in tier files to 1-based sequence indices for recovery metrics. '
                             'E.g. pass 1 if the target PDB starts at residue 2 (default: 0).')
    parser.add_argument('--reference_name', type=str, default=None,
                        help='Fixed value to write in the structure_name column (should match '
                             'the stem of the reference PDB, e.g. "nanobody_antiGFP.E"). '
                             'The path-based identifier is preserved in run_path. '
                             'Required when feeding the CSV into evaluation_pipeline.py.')

    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"ERROR: Directory not found: {args.directory}")
        return 1

    process_directory(
        base_dir=args.directory,
        output_csv=args.output,
        tier_dir=args.tier_dir,
        max_tier=args.max_tier,
        reference_name=args.reference_name,
        pdb_res_offset=args.pdb_res_offset,
    )
    return 0


if __name__ == '__main__':
    exit(main())
