# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025-2026 Alp Tartici, Stanford University.
# Licensed under the MIT License.

"""
Sequence-level metrics (BLOSUM62 similarity) with no heavy structural dependencies.
Safe to import from any environment that has Biopython.
"""

import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from Bio import Align
    from Bio.Align import substitution_matrices
    _BIOPYTHON_AVAILABLE = True
except ImportError:
    _BIOPYTHON_AVAILABLE = False
    logger.warning("Biopython not available — BLOSUM62 similarity metrics will be None")


def _gapped_sequences_from_alignment(alignment) -> Tuple[str, str]:
    """
    Reconstruct gapped (dash-containing) aligned strings from a Biopython
    PairwiseAligner Alignment object using its coordinate array.
    """
    coords = alignment.coordinates  # shape (2, n_segments+1)
    target = str(alignment.target)
    query = str(alignment.query)
    t_parts, q_parts = [], []
    for i in range(coords.shape[1] - 1):
        t_start, t_end = int(coords[0, i]), int(coords[0, i + 1])
        q_start, q_end = int(coords[1, i]), int(coords[1, i + 1])
        t_len = t_end - t_start
        q_len = q_end - q_start
        block_len = max(t_len, q_len)
        t_parts.append(target[t_start:t_end] + '-' * (block_len - t_len))
        q_parts.append(query[q_start:q_end] + '-' * (block_len - q_len))
    return ''.join(t_parts), ''.join(q_parts)


def compute_blosum62_similarity(predicted_sequence: str, true_sequence: str) -> Dict[str, Optional[float]]:
    """
    Compute BLOSUM62-based sequence similarity between two amino acid sequences.

    Uses Needleman-Wunsch global alignment with BLOSUM62 as the substitution matrix.

    Returns a dict with:
        seq_sim_blosum_frac  — fraction of aligned positions where BLOSUM62(pred, true) > 0
        seq_sim_blosum_mean  — mean BLOSUM62 score over all aligned non-gap positions

    Returns None values for both if sequences are empty or Biopython is unavailable.
    """
    null = {'seq_sim_blosum_frac': None, 'seq_sim_blosum_mean': None}
    if not predicted_sequence or not true_sequence:
        return null
    if not _BIOPYTHON_AVAILABLE:
        return null

    try:
        blosum62 = substitution_matrices.load("BLOSUM62")
        aligner = Align.PairwiseAligner()
        aligner.substitution_matrix = blosum62
        aligner.open_gap_score = -11
        aligner.extend_gap_score = -1
        aligner.mode = 'global'
        best = next(iter(aligner.align(predicted_sequence, true_sequence)))
        aligned_pred, aligned_true = _gapped_sequences_from_alignment(best)

        positive_count = 0
        total_aligned = 0
        score_sum = 0.0
        for aa_p, aa_t in zip(aligned_pred, aligned_true):
            if aa_p == '-' or aa_t == '-':
                continue
            total_aligned += 1
            try:
                s = blosum62[aa_p, aa_t]
            except KeyError:
                s = blosum62.get((aa_t, aa_p), 0)
            score_sum += s
            if s > 0:
                positive_count += 1

        if total_aligned == 0:
            return null
        return {
            'seq_sim_blosum_frac': positive_count / total_aligned,
            'seq_sim_blosum_mean': score_sum / total_aligned,
        }
    except Exception as e:
        logger.warning(f"BLOSUM62 similarity failed: {e}")
        return null
