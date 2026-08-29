# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import argparse
import os
import sys
import tempfile
from glob import glob
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import tmtools

AA_MAP = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
}


def parse_pdb_ca(pdb_path: str) -> Optional[Tuple[np.ndarray, str]]:
    """Extract CA coordinates and sequence from a PDB file (first chain, first model)."""
    coords = []
    sequence = []
    target_chain = None

    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ENDMDL'):
                break
            if line.startswith('ATOM') and line[12:16].strip() == 'CA' and line[16] in (' ', 'A'):
                current_chain = line[21].strip()
                if target_chain is None:
                    target_chain = current_chain
                if current_chain != target_chain:
                    continue
                coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                sequence.append(AA_MAP.get(line[17:20].strip(), 'X'))

    if not coords:
        return None
    return np.array(coords, dtype=np.float64), ''.join(sequence)


def ensure_pdb(path: str) -> Tuple[str, Optional[str]]:
    """Return a PDB path for the input. Converts CIF to a temp PDB if needed.
    Returns (pdb_path, temp_path_or_None) — caller must clean up temp."""
    p = Path(path)
    if p.suffix.lower() in ('.cif', '.mmcif'):
        sys.path.insert(0, str(Path(__file__).parent.parent / 'helpers'))
        from convert_cif_to_pdb import convert_cif_to_pdb

        tmp = tempfile.NamedTemporaryFile(suffix='.pdb', delete=False)
        tmp.close()
        ok = convert_cif_to_pdb(str(p), tmp.name, verbose=False)
        if not ok:
            os.unlink(tmp.name)
            print(f"ERROR: could not convert {path} to PDB", file=sys.stderr)
            sys.exit(1)
        return tmp.name, tmp.name
    return str(p), None


def run_tmalign(file1: str, file2: str) -> Optional[float]:
    """Compute TM-score between two PDB files using tmtools. Returns tm_score_chain1."""
    try:
        pdb1, tmp1 = ensure_pdb(file1)
        pdb2, tmp2 = ensure_pdb(file2)
        try:
            data1 = parse_pdb_ca(pdb1)
            data2 = parse_pdb_ca(pdb2)
            if data1 is None or data2 is None:
                return None
            result = tmtools.tm_align(data1[0], data2[0], data1[1], data2[1])
            return result.tm_norm_chain1
        finally:
            for tmp in (tmp1, tmp2):
                if tmp and os.path.exists(tmp):
                    os.unlink(tmp)
    except Exception as e:
        print(f"Error computing TM-score: {e}")
        return None


def process_files():
    from cif_parser import parse_cif_backbone_auto  # only needed in batch mode

    ground_truth_files = glob("*.cif")
    data = []

    for gt_file in ground_truth_files:
        pdb_id = os.path.basename(gt_file).split(".")[0]
        pred_file = f"pred_for_{pdb_id}.pdb"
        if not os.path.exists(pred_file):
            pred_file = f"{pdb_id}.pdb"

        if os.path.exists(pred_file):
            seq_length = parse_cif_backbone_auto(gt_file)
            tm_score = run_tmalign(gt_file, pred_file)
            data.append([pdb_id, seq_length, tm_score])

    df = pd.DataFrame(data, columns=["PDB ID", "Sequence Length", "TM-score"])

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(f"{timestamp}_tm_align_scores.csv", index=False)


def main_cli():
    parser = argparse.ArgumentParser(
        description="Compute TM-score between two structures (PDB or CIF)."
    )
    parser.add_argument("file1", help="Reference structure (.pdb or .cif)")
    parser.add_argument("file2", help="Query structure (.pdb or .cif)")
    args = parser.parse_args()

    for f in (args.file1, args.file2):
        if not Path(f).exists():
            print(f"ERROR: file not found: {f}", file=sys.stderr)
            sys.exit(1)

    pdb1, tmp1 = ensure_pdb(args.file1)
    pdb2, tmp2 = ensure_pdb(args.file2)

    try:
        data1 = parse_pdb_ca(pdb1)
        data2 = parse_pdb_ca(pdb2)

        if data1 is None:
            print(f"ERROR: no CA atoms found in {args.file1}", file=sys.stderr)
            sys.exit(1)
        if data2 is None:
            print(f"ERROR: no CA atoms found in {args.file2}", file=sys.stderr)
            sys.exit(1)

        coords1, seq1 = data1
        coords2, seq2 = data2

        result = tmtools.tm_align(coords1, coords2, seq1, seq2)

        print(f"File 1:           {args.file1}  ({len(seq1)} residues)")
        print(f"File 2:           {args.file2}  ({len(seq2)} residues)")
        print(f"TM-score (norm by chain 1): {result.tm_norm_chain1:.4f}")
        print(f"TM-score (norm by chain 2): {result.tm_norm_chain2:.4f}")
        print(f"RMSD:             {result.rmsd:.4f}")
    finally:
        for tmp in (tmp1, tmp2):
            if tmp and os.path.exists(tmp):
                os.unlink(tmp)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main_cli()
    else:
        process_files()
