# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025-2026 Alp Tartici, Stanford University.
# Licensed under the MIT License.

"""
soft_priors.py

Soft residue-prior conditioning for Inverse FoldDir sampling.

Fixed-residue inpainting clamps a position to one amino acid for the whole
trajectory. Soft priors are the intermediate level of control: the user biases
selected positions toward a residue *class* (or a custom amino-acid
distribution) without committing them to a single identity. Those positions
start from the user's distribution instead of the diffuse default and remain
free to change during denoising.

Two ways to specify priors, both accepted everywhere:

  1. Named residue classes (for the common cases)::

        --soft-priors "34:polar, 57:metal_binding, 88:aromatic"

  2. Explicit per-amino-acid weights (for full control)::

        --soft-priors "88:H0.5|E0.3|D0.2"

  3. A JSON file, when there are many positions (mixes both forms freely)::

        --soft-priors-json priors.json

        {
          "34": "polar",
          "57": {"HIS": 0.4, "CYS": 0.2, "ASP": 0.2, "GLU": 0.2},
          "88": "H0.5|E0.3|D0.2"
        }

Partial specifications are allowed and are the recommended default. If the
weights at a position sum to less than 1.0, the remaining probability mass is
spread evenly over all *unspecified* amino acids rather than being discarded.
So ``57:H0.4`` means "40% histidine, and the other 60% spread across the
other 19 residues" -- a nudge, not a constraint. Weights summing to exactly
1.0 concentrate all mass on the named residues; weights summing above 1.0 are
normalized down to 1.0 with a warning.

Positions are 1-indexed to match --fixed-positions and --forced-positions.
"""

import json
import os
import re
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# AA_TO_IDX / SINGLE_TO_TRIPLE are defined in sample_utils, which is the
# canonical source for these mappings across the sampling code.
from training.sample_utils import AA_TO_IDX, SINGLE_TO_TRIPLE

# Number of amino-acid classes the model emits (20 canonical + unknown).
K_CLASSES = 21
# Index of the unknown/mask class, which never receives prior mass.
UNKNOWN_IDX = 20

# Canonical single-letter codes in model index order (index 0..19).
_CANONICAL_AAS = "ACDEFGHIKLMNPQRSTVWY"


# ---------------------------------------------------------------------------
# Named residue classes
# ---------------------------------------------------------------------------
# Chosen to cover the design scenarios the method is meant to support:
# biasing exposed positions toward polar residues, candidate metal sites toward
# coordinating residues, nucleic-acid-binding surfaces toward basic residues,
# and so on. Mass is spread evenly across the members of a class.
RESIDUE_CLASSES = {
    "polar":          "STNQ",
    "hydrophobic":    "AVILMFWC",
    "aromatic":       "FWYH",
    "charged":        "DEKR",
    "positive":       "KR",
    "basic":          "KRH",
    "negative":       "DE",
    "acidic":         "DE",
    "metal_binding":  "HCDE",
    "small":          "AGS",
    "tiny":           "AG",
    "aliphatic":      "AVILM",
    "helix_former":   "AELMQK",
    "sheet_former":   "VIYFWT",
    "turn_former":    "GNPSD",
    "flexible":       "GS",
    "rigid":          "P",
    "cysteine":       "C",
    "proline":        "P",
    "glycine":        "G",
    "any":            _CANONICAL_AAS,
}

# Human-readable descriptions, surfaced in --list-residue-classes and errors.
RESIDUE_CLASS_DESCRIPTIONS = {
    "polar":         "Neutral polar (S, T, N, Q) -- solvent-exposed surfaces",
    "hydrophobic":   "Hydrophobic (A, V, I, L, M, F, W, C) -- buried core",
    "aromatic":      "Aromatic (F, W, Y, H) -- stacking, interface hot spots",
    "charged":       "Charged (D, E, K, R) -- salt bridges, solubility",
    "positive":      "Positively charged (K, R) -- nucleic-acid contacts",
    "basic":         "Basic (K, R, H) -- includes histidine",
    "negative":      "Negatively charged (D, E)",
    "acidic":        "Acidic (D, E) -- synonym of 'negative'",
    "metal_binding": "Metal-coordinating (H, C, D, E) -- candidate metal sites",
    "small":         "Small (A, G, S) -- tight packing",
    "tiny":          "Tiny (A, G) -- minimal side chain",
    "aliphatic":     "Aliphatic (A, V, I, L, M)",
    "helix_former":  "Helix-favoring (A, E, L, M, Q, K)",
    "sheet_former":  "Sheet-favoring (V, I, Y, F, W, T)",
    "turn_former":   "Turn-favoring (G, N, P, S, D)",
    "flexible":      "Flexible backbone (G, S)",
    "rigid":         "Backbone-rigidifying (P)",
    "cysteine":      "Cysteine only (C) -- disulfide partner, still soft",
    "proline":       "Proline only (P)",
    "glycine":       "Glycine only (G)",
    "any":           "All 20 canonical amino acids (uniform)",
}


class SoftPriorError(ValueError):
    """Raised when a soft-prior specification cannot be parsed or validated."""


def list_residue_classes():
    """Return a printable listing of the named residue classes."""
    lines = ["Available residue classes for --soft-priors:", ""]
    width = max(len(name) for name in RESIDUE_CLASSES)
    for name in sorted(RESIDUE_CLASSES):
        desc = RESIDUE_CLASS_DESCRIPTIONS.get(name, "")
        lines.append(f"  {name:<{width}}  {desc}")
    lines += [
        "",
        "Usage:",
        '  --soft-priors "34:polar, 57:metal_binding"',
        '  --soft-priors "88:H0.5|E0.3|D0.2"        (explicit weights)',
        "  --soft-priors-json priors.json            (many positions)",
        "",
        "Unspecified probability mass is spread evenly over the remaining",
        "amino acids, so partial specs such as '57:H0.4' act as a nudge.",
    ]
    return "\n".join(lines)


def _aa_to_index(token):
    """Map a 1- or 3-letter amino-acid code to its model index."""
    token = token.strip().upper()
    if not token:
        raise SoftPriorError("Empty amino-acid code")

    if len(token) == 1:
        triple = SINGLE_TO_TRIPLE.get(token)
        if triple is None:
            raise SoftPriorError(
                f"Unknown amino-acid code '{token}'. "
                f"Expected one of: {_CANONICAL_AAS}"
            )
    elif len(token) == 3:
        triple = token
    else:
        raise SoftPriorError(
            f"Unknown amino-acid code '{token}'. Use 1-letter (H) or 3-letter (HIS)."
        )

    idx = AA_TO_IDX.get(triple)
    if idx is None or idx >= UNKNOWN_IDX:
        raise SoftPriorError(
            f"Unknown amino-acid code '{token}'. "
            f"Expected one of: {_CANONICAL_AAS}"
        )
    return idx


def _weights_from_class(class_name):
    """Expand a named residue class into an even {aa_index: weight} mapping."""
    key = class_name.strip().lower()
    if key not in RESIDUE_CLASSES:
        close = [c for c in RESIDUE_CLASSES if c.startswith(key[:3])]
        hint = f" Did you mean: {', '.join(sorted(close))}?" if close else ""
        raise SoftPriorError(
            f"Unknown residue class '{class_name}'.{hint}\n"
            f"Available classes: {', '.join(sorted(RESIDUE_CLASSES))}\n"
            f"(Run with --list-residue-classes for descriptions.)"
        )
    members = RESIDUE_CLASSES[key]
    share = 1.0 / len(members)
    return {_aa_to_index(aa): share for aa in members}


def _weights_from_spec_string(spec):
    """
    Parse the right-hand side of one position's prior.

    Accepts either a class name ("polar") or explicit weights
    ("H0.5|E0.3|D0.2", also tolerating commas or spaces as separators).
    """
    spec = spec.strip()
    if not spec:
        raise SoftPriorError("Empty prior specification")

    # A bare alphabetic token is a class name.
    if re.fullmatch(r"[A-Za-z_]+", spec):
        return _weights_from_class(spec)

    weights = {}
    for token in re.split(r"[|,]", spec):
        token = token.strip()
        if not token:
            continue
        match = re.fullmatch(r"([A-Za-z]{1,3})\s*([0-9]*\.?[0-9]+)", token)
        if not match:
            raise SoftPriorError(
                f"Cannot parse weight '{token}'. "
                f"Expected forms like 'H0.5' or 'HIS0.5', joined by '|'."
            )
        aa_token, weight_str = match.groups()
        weight = float(weight_str)
        if weight < 0:
            raise SoftPriorError(f"Negative weight in '{token}' is not allowed")
        idx = _aa_to_index(aa_token)
        weights[idx] = weights.get(idx, 0.0) + weight

    if not weights:
        raise SoftPriorError(f"No amino-acid weights found in '{spec}'")
    return weights


def _weights_from_mapping(mapping):
    """Parse an explicit {"HIS": 0.4, ...} JSON object into indices."""
    weights = {}
    for aa_token, weight in mapping.items():
        try:
            weight = float(weight)
        except (TypeError, ValueError):
            raise SoftPriorError(
                f"Weight for '{aa_token}' must be a number, got {weight!r}"
            )
        if weight < 0:
            raise SoftPriorError(f"Negative weight for '{aa_token}' is not allowed")
        idx = _aa_to_index(aa_token)
        weights[idx] = weights.get(idx, 0.0) + weight
    if not weights:
        raise SoftPriorError("Empty amino-acid weight mapping")
    return weights


def _normalize_weights(weights, position_1indexed, verbose=False):
    """
    Turn raw weights into a full length-21 probability vector.

    Any mass not assigned by the user is spread evenly across the amino acids
    they did not mention, which is what makes a partial spec behave as a nudge
    rather than a hard restriction.
    """
    total = sum(weights.values())
    if total <= 0:
        raise SoftPriorError(
            f"Position {position_1indexed}: prior weights sum to {total}, must be > 0"
        )

    if total > 1.0 + 1e-6:
        if verbose:
            print(
                f"  [soft-prior] Position {position_1indexed}: weights sum to "
                f"{total:.3f} (> 1.0); normalizing to 1.0."
            )
        weights = {idx: w / total for idx, w in weights.items()}
        total = 1.0

    probs = torch.zeros(K_CLASSES, dtype=torch.float32)
    for idx, weight in weights.items():
        probs[idx] = weight

    # Spread the leftover mass over the amino acids the user did not name.
    remainder = 1.0 - total
    if remainder > 1e-9:
        unspecified = [
            i for i in range(UNKNOWN_IDX) if i not in weights
        ]
        if unspecified:
            share = remainder / len(unspecified)
            for i in unspecified:
                probs[i] = share
        else:
            # User named all 20 residues but undershot 1.0; rescale instead.
            probs[:UNKNOWN_IDX] /= probs[:UNKNOWN_IDX].sum()

    probs[UNKNOWN_IDX] = 0.0
    probs = probs / probs.sum()
    return probs


def parse_soft_priors(spec_string=None, json_path=None, sequence_length=None,
                      verbose=False):
    """
    Parse soft-prior specifications into per-position probability vectors.

    Args:
        spec_string: Inline spec, e.g. "34:polar, 57:H0.4|C0.2".
        json_path: Path to a JSON file mapping positions to specs.
        sequence_length: If given, validate positions against it.
        verbose: Print a per-position summary.

    Returns:
        dict {position_0indexed: FloatTensor[21]}, or {} if nothing specified.
    """
    raw = {}

    if json_path:
        if not os.path.exists(json_path):
            raise SoftPriorError(f"Soft-prior JSON file not found: {json_path}")
        try:
            with open(json_path) as handle:
                payload = json.load(handle)
        except json.JSONDecodeError as exc:
            raise SoftPriorError(f"Invalid JSON in {json_path}: {exc}")
        if not isinstance(payload, dict):
            raise SoftPriorError(
                f"{json_path}: expected a JSON object mapping positions to priors, "
                f"got {type(payload).__name__}"
            )
        # Allow an optional wrapper key so users can keep notes alongside priors.
        if "priors" in payload and isinstance(payload["priors"], dict):
            payload = payload["priors"]
        raw.update(payload)

    if spec_string:
        # Split on commas that separate "pos:spec" pairs, but not on commas
        # inside an explicit weight list.
        for chunk in re.split(r",(?=\s*[0-9]+\s*:)", spec_string):
            chunk = chunk.strip()
            if not chunk:
                continue
            if ":" not in chunk:
                raise SoftPriorError(
                    f"Cannot parse soft prior '{chunk}'. "
                    f"Expected 'POSITION:SPEC', e.g. '34:polar' or '88:H0.5|E0.3'."
                )
            pos_part, spec_part = chunk.split(":", 1)
            raw[pos_part.strip()] = spec_part.strip()

    if not raw:
        return {}

    priors = {}
    errors = []
    for pos_key, spec in raw.items():
        try:
            try:
                position_1indexed = int(str(pos_key).strip())
            except ValueError:
                raise SoftPriorError(
                    f"Position '{pos_key}' is not an integer. Positions are 1-indexed."
                )

            if position_1indexed < 1:
                raise SoftPriorError(
                    f"Position {position_1indexed} is invalid; positions are 1-indexed."
                )
            position_0indexed = position_1indexed - 1

            if sequence_length is not None and position_0indexed >= sequence_length:
                raise SoftPriorError(
                    f"Position {position_1indexed} is out of range "
                    f"(sequence length: {sequence_length})"
                )

            if isinstance(spec, dict):
                weights = _weights_from_mapping(spec)
            elif isinstance(spec, str):
                weights = _weights_from_spec_string(spec)
            else:
                raise SoftPriorError(
                    f"Prior for position {position_1indexed} must be a class name, "
                    f"a weight string, or an object; got {type(spec).__name__}"
                )

            priors[position_0indexed] = _normalize_weights(
                weights, position_1indexed, verbose=verbose
            )
        except SoftPriorError as exc:
            errors.append(str(exc))

    if errors:
        raise SoftPriorError(
            "Soft-prior validation failed:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    if verbose and priors:
        print(f"\nParsed {len(priors)} soft residue prior(s):")
        for pos in sorted(priors):
            print(f"  {describe_prior(pos, priors[pos])}")

    return priors


def describe_prior(position_0indexed, prob_vector, top_n=4):
    """One-line human-readable summary of a position's prior."""
    top = torch.topk(prob_vector[:UNKNOWN_IDX], min(top_n, UNKNOWN_IDX))
    parts = [
        f"{_CANONICAL_AAS[idx]}={val:.2f}"
        for val, idx in zip(top.values.tolist(), top.indices.tolist())
        if val > 1e-4
    ]
    entropy = -(prob_vector[:UNKNOWN_IDX].clamp_min(1e-12).log()
                * prob_vector[:UNKNOWN_IDX]).sum().item()
    return (
        f"Position {position_0indexed + 1}: "
        + ", ".join(parts)
        + f" (entropy {entropy:.2f} nats)"
    )


def validate_prior_conflicts(priors, fixed_positions=None, forced_positions=None,
                             verbose=False):
    """
    Reject positions that are both softly biased and hard-clamped.

    A fixed or forced position is held constant for the whole trajectory, so a
    soft prior there would be silently ignored. Failing loudly is better than
    letting a user believe a prior took effect when it did not.
    """
    if not priors:
        return

    hard = set()
    for group in (fixed_positions, forced_positions):
        if group:
            hard.update(group)

    overlap = sorted(set(priors) & hard)
    if overlap:
        raise SoftPriorError(
            "These positions are both soft-prior and fixed/forced, which is "
            "contradictory (fixed positions never change, so the prior would "
            "be ignored):\n"
            f"  Positions (1-indexed): {[p + 1 for p in overlap]}\n"
            "Remove them from either --soft-priors or --fixed-positions/"
            "--forced-positions."
        )

    if verbose:
        print(f"  [soft-prior] No conflicts with fixed/forced positions.")


def apply_soft_priors(x, priors, prior_strength=1.0, dirichlet_concentration=20.0,
                      inpainting_mask=None, generator=None, verbose=False):
    """
    Overwrite the initial state at prior positions with samples from the
    user's distribution.

    The prior replaces the default diffuse initialization at those positions
    only. Every prior position stays free to move during denoising -- this
    biases where the trajectory starts, it does not clamp where it ends.

    Args:
        x: Initial state, [1, N, K] or [B, N, K] or [N, K]. Modified in place.
        priors: {position_0indexed: FloatTensor[K]} from parse_soft_priors.
        prior_strength: Dirichlet concentration multiplier for prior positions.
            Higher values start closer to the stated distribution; lower values
            leave more initial noise. 1.0 reuses dirichlet_concentration.
        dirichlet_concentration: Baseline concentration used elsewhere.
        inpainting_mask: Optional bool [N]; True = position is being sampled.
            Priors on non-sampled positions are skipped with a warning.
        generator: Optional torch.Generator for reproducibility.
        verbose: Print what was applied.

    Returns:
        The modified tensor (same object as ``x``).
    """
    if not priors:
        return x

    from torch.distributions import Dirichlet

    squeeze_back = False
    if x.dim() == 2:
        x = x.unsqueeze(0)
        squeeze_back = True

    batch_size, seq_len, num_classes = x.shape
    device = x.device

    concentration = float(prior_strength) * float(dirichlet_concentration)
    if concentration <= 0:
        raise SoftPriorError(
            f"prior_strength * dirichlet_concentration must be > 0, got {concentration}"
        )

    applied, skipped = [], []
    for position, prob_vector in sorted(priors.items()):
        if position >= seq_len:
            skipped.append((position, "beyond sequence length"))
            continue
        if inpainting_mask is not None and not bool(inpainting_mask[position]):
            skipped.append((position, "position is not being sampled"))
            continue

        # Dirichlet requires strictly positive concentration on every class, so
        # give unnamed residues a floor rather than exactly zero.
        alpha = prob_vector.to(device=device, dtype=torch.float32) * concentration
        alpha = alpha.clamp_min(1e-6)

        dist = Dirichlet(alpha)
        for b in range(batch_size):
            sample = (dist.sample() if generator is None
                      else _sample_with_generator(alpha, generator, device))
            x[b, position] = sample
        applied.append(position)

    if verbose:
        if applied:
            print(f"  [soft-prior] Applied {len(applied)} prior(s) at "
                  f"concentration {concentration:.2f} "
                  f"(strength {prior_strength} x base {dirichlet_concentration})")
            for position in applied:
                print(f"    {describe_prior(position, priors[position])}")
        for position, reason in skipped:
            print(f"  [soft-prior] WARNING: skipped position {position + 1} ({reason})")

    if skipped and not verbose:
        print(f"  [soft-prior] WARNING: {len(skipped)} prior position(s) skipped; "
              f"re-run with --verbose for details.")

    return x.squeeze(0) if squeeze_back else x


def _sample_with_generator(alpha, generator, device):
    """Dirichlet sample honoring an explicit generator (via Gamma ratios)."""
    gamma = torch._standard_gamma(alpha, generator=generator) \
        if hasattr(torch, "_standard_gamma") else None
    if gamma is None:
        gamma = torch.stack([
            torch.distributions.Gamma(a, torch.ones_like(a)).sample()
            for a in alpha
        ])
    return gamma / gamma.sum().clamp_min(1e-12)
