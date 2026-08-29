#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025-2026 Alp Tartici, Stanford University.
# Licensed under the MIT License.

"""
Convert CIF file to PDB format for DSSP compatibility.

DSSP binary only works with PDB format, so we need to convert CIF files
before processing them with DSSP-enabled configs.

Usage:
    python helpers/convert_cif_to_pdb.py input.cif output.pdb
"""

import sys
from pathlib import Path
from Bio.PDB import MMCIFParser, PDBIO, Select


class ProteinSelect(Select):
    """Select only protein residues (exclude water, ligands, etc.)."""
    def accept_residue(self, residue):
        # Only keep standard amino acid residues
        return residue.get_id()[0] == ' '  # ' ' indicates standard residue


def convert_cif_to_pdb(cif_path, pdb_path, verbose=False):
    """
    Convert CIF file to PDB format.

    Args:
        cif_path: Path to input CIF file
        pdb_path: Path to output PDB file
        verbose: Print status messages

    Returns:
        True if successful, False otherwise
    """
    try:
        if verbose:
            print(f"Converting {cif_path} -> {pdb_path}")

        # Parse CIF file
        parser = MMCIFParser(QUIET=not verbose)
        structure = parser.get_structure("structure", cif_path)

        # Write as PDB
        io = PDBIO()
        io.set_structure(structure)
        io.save(str(pdb_path), select=ProteinSelect())

        if verbose:
            print(f"Successfully converted to PDB format")

        return True

    except Exception as e:
        print(f"ERROR: Failed to convert CIF to PDB: {e}", file=sys.stderr)
        return False


def main():
    """CLI entry point."""
    if len(sys.argv) != 3:
        print("Usage: python convert_cif_to_pdb.py input.cif output.pdb")
        sys.exit(1)

    cif_path = Path(sys.argv[1])
    pdb_path = Path(sys.argv[2])

    if not cif_path.exists():
        print(f"ERROR: CIF file not found: {cif_path}", file=sys.stderr)
        sys.exit(1)

    success = convert_cif_to_pdb(cif_path, pdb_path, verbose=True)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
