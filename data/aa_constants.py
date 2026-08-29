# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025-2026 Alp Tartici, Stanford University.
# Licensed under the MIT License.

"""
aa_constants.py

Amino acid index mappings for auxiliary prediction heads.

AA_TO_IDX ordering (canonical, from cath_dataset.py):
  ALA=0, CYS=1, ASP=2, GLU=3, PHE=4, GLY=5, HIS=6, ILE=7, LYS=8, LEU=9,
  MET=10, ASN=11, PRO=12, GLN=13, ARG=14, SER=15, THR=16, VAL=17, TRP=18,
  TYR=19, XXX=20
"""

import torch

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None

# ---------------------------------------------------------------------------
# HEAD 1: ELECTROSTATIC / POLARITY (5-class)
# ---------------------------------------------------------------------------
# Classes:
#   0 = ACIDIC         D(2), E(3)
#   1 = BASIC          K(8), R(14)
#   2 = NEUTRAL_POLAR  N(11), Q(13), S(15), T(16)
#   3 = HISTIDINE      H(6)   -- own class: 9.5% recall, chemically borderline
#   4 = OTHER          A, C, F, G, I, L, M, P, V, W, Y
#  -1 = MASK           X(20)  -- excluded from loss
ELECTROSTATIC_LABEL = torch.tensor([
    4,   # A ( 0) ALA -> OTHER
    4,   # C ( 1) CYS -> OTHER
    0,   # D ( 2) ASP -> ACIDIC
    0,   # E ( 3) GLU -> ACIDIC
    4,   # F ( 4) PHE -> OTHER
    4,   # G ( 5) GLY -> OTHER
    3,   # H ( 6) HIS -> HISTIDINE
    4,   # I ( 7) ILE -> OTHER
    1,   # K ( 8) LYS -> BASIC
    4,   # L ( 9) LEU -> OTHER
    4,   # M (10) MET -> OTHER
    2,   # N (11) ASN -> NEUTRAL_POLAR
    4,   # P (12) PRO -> OTHER
    2,   # Q (13) GLN -> NEUTRAL_POLAR
    1,   # R (14) ARG -> BASIC
    2,   # S (15) SER -> NEUTRAL_POLAR
    2,   # T (16) THR -> NEUTRAL_POLAR
    4,   # V (17) VAL -> OTHER
    4,   # W (18) TRP -> OTHER
    4,   # Y (19) TYR -> OTHER
    -1,  # X (20) XXX -> MASK
], dtype=torch.long)

NUM_ELECTROSTATIC_CLASSES = 5

ELECTROSTATIC_CLASS_NAMES = ['ACIDIC', 'BASIC', 'NEUTRAL_POLAR', 'HISTIDINE', 'OTHER']

# ---------------------------------------------------------------------------
# HEAD 3: GEOMETRIC TOPOLOGY (5-class)
# ---------------------------------------------------------------------------
# Classes:
#   0 = AROMATIC      F(4), W(18), Y(19)   -- large flat sidechains, pi-stacking
#   1 = CB_BRANCHED   I(7), V(17)          -- bifurcated packing at Cb
#   2 = EXTENDED      L(9), M(10)          -- distal branch or flexible
#   3 = SMALL         A(0), G(5)           -- minimal sidechain, backbone flexibility
#   4 = OTHER         everything else (charged, polar, C, P)
#  -1 = MASK          X(20)
#
# NOTE: T is OTHER (not CB_BRANCHED) -- its confusion is polar (S/D/E/R), not hydrophobic.
# NOTE: P is OTHER -- already 84.9% recall, confusion pattern is scattered.
GEOM_TOPOLOGY_LABEL = torch.tensor([
    3,   # A ( 0) ALA -> SMALL
    4,   # C ( 1) CYS -> OTHER
    4,   # D ( 2) ASP -> OTHER
    4,   # E ( 3) GLU -> OTHER
    0,   # F ( 4) PHE -> AROMATIC
    3,   # G ( 5) GLY -> SMALL
    4,   # H ( 6) HIS -> OTHER
    1,   # I ( 7) ILE -> CB_BRANCHED
    4,   # K ( 8) LYS -> OTHER
    2,   # L ( 9) LEU -> EXTENDED
    2,   # M (10) MET -> EXTENDED
    4,   # N (11) ASN -> OTHER
    4,   # P (12) PRO -> OTHER
    4,   # Q (13) GLN -> OTHER
    4,   # R (14) ARG -> OTHER
    4,   # S (15) SER -> OTHER
    4,   # T (16) THR -> OTHER  (polar confusion pattern, not hydrophobic)
    1,   # V (17) VAL -> CB_BRANCHED
    0,   # W (18) TRP -> AROMATIC
    0,   # Y (19) TYR -> AROMATIC
    -1,  # X (20) XXX -> MASK
], dtype=torch.long)

NUM_GEOM_TOPOLOGY_CLASSES = 5

GEOM_TOPOLOGY_CLASS_NAMES = ['AROMATIC', 'CB_BRANCHED', 'EXTENDED', 'SMALL', 'OTHER']

# ---------------------------------------------------------------------------
# HEAD 2: BURIAL / EXPOSURE (3-class) -- placeholder, requires preprocessing
# ---------------------------------------------------------------------------
NUM_BURIAL_CLASSES = 3  # 0=EXPOSED, 1=INTERMEDIATE, 2=BURIED


_AA_WEIGHTS_20 = [
    0.703947,  # A (index  0) — 8.24% freq
    1.791049,  # C (index  1) — 1.27% freq
    0.840191,  # D (index  2) — 5.79% freq
    0.769309,  # E (index  3) — 6.90% freq
    1.029705,  # F (index  4) — 3.85% freq
    0.745034,  # G (index  5) — 7.36% freq
    1.180800,  # H (index  6) — 2.93% freq
    0.847131,  # I (index  7) — 5.69% freq
    0.840948,  # K (index  8) — 5.77% freq
    0.661782,  # L (index  9) — 9.33% freq
    1.323900,  # M (index 10) — 2.33% freq
    0.993294,  # N (index 11) — 4.14% freq
    0.951446,  # P (index 12) — 4.51% freq
    1.052249,  # Q (index 13) — 3.69% freq
    0.891999,  # R (index 14) — 5.13% freq
    0.806426,  # S (index 15) — 6.28% freq
    0.878327,  # T (index 16) — 5.29% freq
    0.765626,  # V (index 17) — 6.97% freq
    1.811629,  # W (index 18) — 1.24% freq
    1.115209,  # Y (index 19) — 3.28% freq
]

# Full 21-class weight tensor (index 20 = X = 0.0, masked from loss)
FIXED_AA_CLASS_WEIGHTS = None
if _TORCH_AVAILABLE:
    FIXED_AA_CLASS_WEIGHTS = torch.tensor(
        _AA_WEIGHTS_20 + [0.0],  # append 0.0 for X at index 20
        dtype=torch.float32
    )

# ---------------------------------------------------------------------------
# Verification (run as script)
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # Canonical AA_TO_IDX: ALA=0,CYS=1,ASP=2,GLU=3,PHE=4,GLY=5,HIS=6,ILE=7,
    #                      LYS=8,LEU=9,MET=10,ASN=11,PRO=12,GLN=13,ARG=14,
    #                      SER=15,THR=16,VAL=17,TRP=18,TYR=19,XXX=20
    print("Verifying ELECTROSTATIC_LABEL...")
    assert len(ELECTROSTATIC_LABEL) == 21, f"Expected 21, got {len(ELECTROSTATIC_LABEL)}"
    assert ELECTROSTATIC_LABEL[2].item() == 0,  "D(ASP=2) should be ACIDIC (0)"
    assert ELECTROSTATIC_LABEL[3].item() == 0,  "E(GLU=3) should be ACIDIC (0)"
    assert ELECTROSTATIC_LABEL[14].item() == 1, "R(ARG=14) should be BASIC (1)"
    assert ELECTROSTATIC_LABEL[8].item() == 1,  "K(LYS=8) should be BASIC (1)"
    assert ELECTROSTATIC_LABEL[11].item() == 2, "N(ASN=11) should be NEUTRAL_POLAR (2)"
    assert ELECTROSTATIC_LABEL[13].item() == 2, "Q(GLN=13) should be NEUTRAL_POLAR (2)"
    assert ELECTROSTATIC_LABEL[15].item() == 2, "S(SER=15) should be NEUTRAL_POLAR (2)"
    assert ELECTROSTATIC_LABEL[16].item() == 2, "T(THR=16) should be NEUTRAL_POLAR (2)"
    assert ELECTROSTATIC_LABEL[6].item() == 3,  "H(HIS=6) should be HISTIDINE (3)"
    assert ELECTROSTATIC_LABEL[20].item() == -1, "X(XXX=20) should be MASK (-1)"
    print("  ELECTROSTATIC_LABEL OK")

    print("Verifying GEOM_TOPOLOGY_LABEL...")
    assert len(GEOM_TOPOLOGY_LABEL) == 21, f"Expected 21, got {len(GEOM_TOPOLOGY_LABEL)}"
    assert GEOM_TOPOLOGY_LABEL[4].item() == 0,  "F(PHE=4) should be AROMATIC (0)"
    assert GEOM_TOPOLOGY_LABEL[18].item() == 0, "W(TRP=18) should be AROMATIC (0)"
    assert GEOM_TOPOLOGY_LABEL[19].item() == 0, "Y(TYR=19) should be AROMATIC (0)"
    assert GEOM_TOPOLOGY_LABEL[7].item() == 1,  "I(ILE=7) should be CB_BRANCHED (1)"
    assert GEOM_TOPOLOGY_LABEL[17].item() == 1, "V(VAL=17) should be CB_BRANCHED (1)"
    assert GEOM_TOPOLOGY_LABEL[9].item() == 2,  "L(LEU=9) should be EXTENDED (2)"
    assert GEOM_TOPOLOGY_LABEL[10].item() == 2, "M(MET=10) should be EXTENDED (2)"
    assert GEOM_TOPOLOGY_LABEL[0].item() == 3,  "A(ALA=0) should be SMALL (3)"
    assert GEOM_TOPOLOGY_LABEL[5].item() == 3,  "G(GLY=5) should be SMALL (3)"
    assert GEOM_TOPOLOGY_LABEL[16].item() == 4, "T(THR=16) should be OTHER (4)"
    assert GEOM_TOPOLOGY_LABEL[12].item() == 4, "P(PRO=12) should be OTHER (4)"
    assert GEOM_TOPOLOGY_LABEL[1].item() == 4,  "C(CYS=1) should be OTHER (4)"
    assert GEOM_TOPOLOGY_LABEL[20].item() == -1, "X(XXX=20) should be MASK (-1)"
    print("  GEOM_TOPOLOGY_LABEL OK")

    print("All verifications passed!")
