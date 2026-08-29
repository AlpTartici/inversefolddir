# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025-2026 Alp Tartici, Stanford University.
# Licensed under the MIT License.

"""
inversefolddir_tools.py

Plain-language helpers for reading Inverse FoldDir output.

The sampler writes ``inpainting_results.json`` with sequences stored as
integer amino-acid indices, which is convenient for code and useless at the
bench. These helpers turn a results directory into the things you actually
need: a one-letter sequence, a FASTA file, a per-position confidence table,
and a comparison against the starting sequence.

Typical use, from a notebook or a Python prompt::

    from inversefolddir_tools import load_results, write_fasta, compare

    r = load_results("output/my_design")
    print(r.sequence)              # MKVLA...
    write_fasta(r, "designs.fasta")
    compare(r)                     # table of what changed

Nothing here imports torch or the model, so it runs on a login node without
a GPU and without the inv_fold environment.
"""

import json
from pathlib import Path

# Model index order (20 canonical amino acids, then unknown).
IDX_TO_LETTER = list("ACDEFGHIKLMNPQRSTVWY") + ["X"]

THREE_LETTER = {
    "A": "Ala", "C": "Cys", "D": "Asp", "E": "Glu", "F": "Phe",
    "G": "Gly", "H": "His", "I": "Ile", "K": "Lys", "L": "Leu",
    "M": "Met", "N": "Asn", "P": "Pro", "Q": "Gln", "R": "Arg",
    "S": "Ser", "T": "Thr", "V": "Val", "W": "Trp", "Y": "Tyr",
    "X": "Unk",
}

# Rough physicochemical grouping, used only for the human-readable
# "did the chemistry change?" column in compare().
PROPERTY = {}
for _letters, _name in [
    ("STNQ", "polar"), ("AVILMFWC", "hydrophobic"), ("DE", "negative"),
    ("KR", "positive"), ("H", "positive"), ("G", "flexible"),
    ("P", "rigid"), ("Y", "aromatic"), ("X", "unknown"),
]:
    for _letter in _letters:
        PROPERTY.setdefault(_letter, _name)


def indices_to_sequence(indices):
    """Turn a list of model indices into a one-letter sequence string."""
    return "".join(
        IDX_TO_LETTER[i] if 0 <= i < len(IDX_TO_LETTER) else "X"
        for i in indices
    )


class Results:
    """
    One Inverse FoldDir run, in bench-friendly form.

    Attributes:
        sequence: designed sequence, one-letter codes
        native: starting/native sequence if the run recorded one, else None
        confidence: per-position probability of the chosen residue (0-1)
        redesigned: bool per position -- True where the model was free to design
        directory: where the results were loaded from
    """

    def __init__(self, payload, directory):
        self.directory = Path(directory)
        self._raw = payload

        self.sequence = indices_to_sequence(payload["predicted_sequence"])

        metrics = payload.get("evaluation_metrics") or {}
        native_indices = metrics.get("true_sequence")
        self.native = indices_to_sequence(native_indices) if native_indices else None

        # A backbone with no residue identities -- a de novo design, or a file
        # whose residues are UNK -- parses as all-X. There is no starting
        # sequence to compare against in that case, and reporting an identity
        # of ~0% against a placeholder would be actively misleading.
        self.known_positions = (
            [aa != "X" for aa in self.native] if self.native
            else [False] * len(self.sequence)
        )
        self.has_native_sequence = any(self.known_positions)

        probabilities = payload.get("final_probabilities") or []
        self.confidence = [max(row) for row in probabilities] if probabilities else []

        mask = payload.get("inpainting_mask")
        self.redesigned = [bool(m) for m in mask] if mask else [True] * len(self.sequence)

        self.metrics = metrics

    def __len__(self):
        return len(self.sequence)

    def __repr__(self):
        n_redesigned = sum(self.redesigned)
        return (f"<Results {len(self)} residues, {n_redesigned} redesigned, "
                f"from {self.directory.name}>")

    def summary(self):
        """Print a short plain-language description of the run."""
        n = len(self)
        n_redesigned = sum(self.redesigned)
        mean_conf = sum(self.confidence) / len(self.confidence) if self.confidence else 0.0

        print(f"Design from : {self.directory}")
        print(f"Length      : {n} residues")
        print(f"Redesigned  : {n_redesigned} positions "
              f"({100 * n_redesigned / n:.0f}%), {n - n_redesigned} held fixed")
        print(f"Confidence  : {mean_conf:.2f} average "
              f"(1.00 = fully certain, 0.05 = no idea)")

        if self.has_native_sequence:
            comparable = sum(self.known_positions)
            identical = sum(
                a == b for a, b, known in
                zip(self.sequence, self.native, self.known_positions) if known
            )
            note = ""
            if comparable < n:
                note = f"  (over {comparable} of {n} positions with a known residue)"
            print(f"Identity    : {100 * identical / comparable:.0f}% "
                  f"to the starting sequence{note}")
        else:
            print("Identity    : not applicable -- the input backbone carries no "
                  "residue identities,")
            print("              so there is no starting sequence to compare against.")

        low = sum(1 for c in self.confidence if c < 0.5)
        if low:
            print(f"\nNote: {low} position(s) below 0.5 confidence. "
                  f"These are the model's least certain choices.")
        print(f"\nSequence:\n{self.sequence}")


def load_results(directory):
    """
    Load one run's output.

    Args:
        directory: the --output-dir you passed to the sampler, or the path to
            an inpainting_results.json directly.

    Returns:
        Results
    """
    path = Path(directory)
    if path.is_dir():
        path = path / "inpainting_results.json"

    if not path.exists():
        raise FileNotFoundError(
            f"No results found at {path}.\n"
            f"Expected an 'inpainting_results.json' inside the output directory "
            f"you gave the sampler. Check that the run finished without errors."
        )

    with open(path) as handle:
        payload = json.load(handle)

    return Results(payload, path.parent)


def write_fasta(results, output_path, name="design"):
    """
    Write the designed sequence as FASTA -- the format ordering services,
    BLAST, and alignment tools expect.

    Accepts a single Results or a list of them.
    """
    if isinstance(results, Results):
        results = [results]

    lines = []
    for i, result in enumerate(results, start=1):
        label = name if len(results) == 1 else f"{name}_{i}"
        lines.append(f">{label} length={len(result)}")
        # Wrap at 60 columns, the FASTA convention.
        for start in range(0, len(result.sequence), 60):
            lines.append(result.sequence[start:start + 60])

    output_path = Path(output_path)
    output_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(results)} sequence(s) to {output_path}")
    return output_path


def compare(results, show="changed", limit=40):
    """
    Print a position-by-position comparison against the starting sequence.

    Args:
        results: a Results object
        show: "changed" (default), "all", or "low-confidence"
        limit: maximum rows to print
    """
    if not results.has_native_sequence:
        print("The input backbone carries no residue identities (a de novo "
              "design, or a file whose residues are UNK), so there is no "
              "starting sequence to compare against.")
        print("\nThe design is:")
        print(results.sequence)
        return

    rows = []
    for i, (new, old) in enumerate(zip(results.sequence, results.native)):
        # Skip positions with no known starting residue -- comparing against a
        # placeholder X would report a spurious "change".
        if not results.known_positions[i]:
            continue
        conf = results.confidence[i] if i < len(results.confidence) else float("nan")
        changed = new != old
        if show == "changed" and not changed:
            continue
        if show == "low-confidence" and conf >= 0.5:
            continue
        rows.append((i + 1, old, new, conf, changed))

    if not rows:
        print(f"No positions matched '{show}'.")
        return

    print(f"{'Pos':>5}  {'From':<5} {'To':<5} {'Conf':>6}  Note")
    print("-" * 58)
    for position, old, new, conf, changed in rows[:limit]:
        note = ""
        if changed:
            old_property = PROPERTY.get(old, "?")
            new_property = PROPERTY.get(new, "?")
            note = ("same chemistry" if old_property == new_property
                    else f"{old_property} -> {new_property}")
        else:
            note = "unchanged"
        print(f"{position:>5}  {THREE_LETTER.get(old, old):<5} "
              f"{THREE_LETTER.get(new, new):<5} {conf:>6.2f}  {note}")

    if len(rows) > limit:
        print(f"\n... and {len(rows) - limit} more. "
              f"Raise limit= to see them all.")


def low_confidence_positions(results, threshold=0.5):
    """
    Return 1-indexed positions the model was unsure about.

    These are the positions most worth a second look, or worth fixing to the
    native residue on a follow-up run.
    """
    return [
        i + 1 for i, conf in enumerate(results.confidence)
        if conf < threshold
    ]


def load_many(parent_directory, pattern="*"):
    """
    Load every run under a parent directory, e.g. a batch of designs.

    Returns a list of Results, skipping directories with no results file.
    """
    parent = Path(parent_directory)
    found = []
    for candidate in sorted(parent.glob(pattern)):
        if (candidate / "inpainting_results.json").exists():
            found.append(load_results(candidate))
    if not found:
        print(f"No results found under {parent}. "
              f"Each run should be its own subdirectory.")
    return found
