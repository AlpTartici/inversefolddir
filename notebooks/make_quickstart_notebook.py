# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025-2026 Alp Tartici, Stanford University.
# Licensed under the MIT License.

"""
Generates notebooks/quickstart.ipynb.

The notebook is the main entry point for users who prefer a guided,
run-one-cell-at-a-time interface over the command line. Keeping it generated
from this script means the content stays reviewable in plain text and diffs
cleanly, instead of being an unreadable blob of notebook JSON.

Regenerate with:
    python notebooks/make_quickstart_notebook.py
"""

import json
from pathlib import Path


def source_lines(text):
    """
    Split a cell body into nbformat "source" lines.

    Every line keeps its trailing newline except the last one. Without the
    newlines, viewers join the lines into a single run-on line, which breaks
    markdown rendering and makes code cells syntactically invalid.
    """
    lines = text.strip("\n").split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": source_lines(text)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source_lines(text)}


CELLS = [
    markdown("""
# Inverse FoldDir quickstart

Give this notebook a protein backbone. It gives you back amino acid sequences
predicted to fold into that backbone.

## How to use it

Run the cells in order from the top. `Shift + Enter` runs the selected cell and
moves to the next one.

There is **one cell you need to edit**: the settings cell in step 2. Every
other cell runs as written.

## What you need before you start

- The `inv_fold` environment installed, and selected as this notebook's kernel.
  To select it: `Kernel` menu, then `Change kernel`, then `inv_fold`.
  Installation is covered in `docs/INSTALL.md`.
- The model weights. They are distributed separately from the code, and one
  command fetches them. Run this in a terminal at the repository root:

```
python scripts/download_checkpoints.py
```

Step 1 checks both of these and tells you if something is missing.
"""),

    markdown("""
## Step 1 — Check the setup

Run the cell below. It prints a short report of what it found. If anything says
`not found`, fix that before going on.
"""),

    code("""
import os
import sys
from pathlib import Path

# Run from the repository root, whichever folder Jupyter happened to start in.
if Path.cwd().name == "notebooks":
    os.chdir("..")
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))


def report(label, value, *notes):
    \"\"\"Print one aligned 'label  value' line, plus any indented notes.\"\"\"
    print(f"{label:<16}{value}")
    for note in notes:
        print(f"{'':<16}{note}")


report("Repository", ROOT)

# paths.py is the single place that resolves where data and checkpoints live,
# so setting IFD_CKPT_DIR or IFD_DATA_DIR works here exactly as it does on the
# command line.
from paths import CATH_DIR, CHAIN_SET_MAP_NAME, ckpt_dir, data_dir

try:
    import torch
except ImportError:
    report("Environment", "not ready",
           "PyTorch could not be imported, which means the inv_fold",
           "kernel is not active. Use Kernel > Change kernel >",
           "inv_fold, then run this cell again.")
else:
    report("Environment", f"ready (PyTorch {torch.__version__})")
    if torch.cuda.is_available():
        report("Hardware", f"GPU ({torch.cuda.get_device_name(0)})")
    else:
        report("Hardware", "CPU", "No GPU detected. Everything still works, just slower.")

# The weights ship separately from the code. A .pt file of only a few hundred
# bytes is a placeholder left by a failed download, not real weights.
CKPT_DIR = ckpt_dir()
weights = sorted(p for p in CKPT_DIR.glob("*.pt") if p.stat().st_size > 1_000_000)

if weights:
    MODEL = str(weights[0])
    report("Weights", Path(MODEL).name,
           *(f"also found {other.name}" for other in weights[1:]))
else:
    MODEL = str(CKPT_DIR / "inverse_folddir_model.pt")
    report("Weights", "not found",
           f"Looked in {CKPT_DIR}",
           "To fetch them, run this in a terminal at the repository root:",
           "    python scripts/download_checkpoints.py",
           "then run this cell again.")

# The CATH reference dataset is optional. It is only needed to pull a structure
# out of that dataset by UniProt or PDB ID. Designing from a structure file, or
# from a PDB ID downloaded on the fly, does not use it.
CATH = data_dir() / CATH_DIR
SPLIT_JSON = CATH / "chain_set_splits.json"
MAP_PKL = CATH / CHAIN_SET_MAP_NAME
HAVE_DATASET = (
    SPLIT_JSON.exists()
    and MAP_PKL.exists()
    and MAP_PKL.stat().st_size > 1_000_000
)

if HAVE_DATASET:
    report("Reference data", "present")
else:
    report("Reference data", "not present",
           "This is optional and this notebook does not need it.")
"""),

    markdown("""
## Step 2 — Settings

This is the cell to edit. Each setting is explained here, and repeated as a
comment in the cell itself.

| Setting | What it does |
|---|---|
| `STRUCTURE` | The backbone to design for. Either a four-character PDB ID such as `"3OGO"`, which is downloaded for you, or a path to your own `.pdb` or `.cif` file. |
| `RUN_NAME` | A short label for this run. Results are written to `output/RUN_NAME/`. |
| `NUM_DESIGNS` | How many sequences to produce. Each is generated independently, so they come out different from one another. |
| `KEEP` | Positions to leave untouched, for example `"C22,C96"`. Leave it as `""` to redesign every position. |
| `CHEMISTRY` | Optional. Steers a position towards a kind of amino acid without pinning an exact one, for example `"34:polar"`. Leave it as `""` to skip. |
| `SEED` | Starting point for the random number generator. The same settings and seed give you the same sequences back. |

### If your file has more than one chain

Only one chain is designed. If the file holds several, the longest one is
picked and the rest are ignored, so check that the longest chain is the one you
meant. `--target-chain` and `--context-chains` on `training/inpainting.py` let
you choose a different chain, or keep the others as structural context.

### How positions are numbered

Positions count from 1 along the chain as read out of your structure file. If
your file is missing residues, or its numbering does not start at 1, these
numbers will not match the residue numbers written in the file. Count along the
chain instead.

Three ways to write a position in `KEEP`:

- `C22` — position 22, and check that it really is a cysteine
- `CYS22` — the same thing with the three-letter code
- `22` — position 22, no check

The first two forms are worth the extra typing. If position 22 turns out not to
be a cysteine, the run stops immediately with an error, which is how you find a
numbering mistake before waiting on a design rather than after.

### Deciding what to keep

Whatever the protein has to keep doing belongs in `KEEP`: catalytic residues,
both halves of a disulfide, positions that sit on a binding interface.
Positions you leave out are free to change.

If the backbone carries no sequence at all (a de novo design, or a file whose
residues are all `UNK`), leave `KEEP` empty. There is nothing there to preserve.
"""),

    code("""
# ---------------------------------------------------------------------------
# Settings. This is the only cell you need to edit.
# ---------------------------------------------------------------------------

# A four-character PDB ID, or a path to your own .pdb or .cif file.
STRUCTURE = "3OGO"

# Label for this run. Results go to output/RUN_NAME/
RUN_NAME = "my_design"

# How many sequences to generate.
NUM_DESIGNS = 3

# Positions to leave untouched, e.g. "C22,C96,W47". Empty means redesign
# everything.
KEEP = ""

# Optional chemistry preferences, e.g. "34:polar, 57:metal_binding". Empty
# skips this. The cell further down lists the names you can use here.
CHEMISTRY = ""

# Change this for a different set of designs; keep it to reproduce these ones.
SEED = 1

# ---------------------------------------------------------------------------

OUTPUT_DIR = f"output/{RUN_NAME}"
"""),

    markdown("""
Run the next cell to check the settings before committing to a run. It confirms
the structure can be found and repeats back what is about to happen.
"""),

    code("""
looks_like_pdb_id = (
    len(STRUCTURE) == 4 and not STRUCTURE.lower().endswith((".pdb", ".cif"))
)

if looks_like_pdb_id:
    report("Structure", f"{STRUCTURE} (will be downloaded)")
elif Path(STRUCTURE).exists():
    if STRUCTURE.lower().endswith((".cif", ".mmcif")):
        report("Structure", f"{STRUCTURE} (mmCIF, converted automatically)")
    else:
        with open(STRUCTURE) as handle:
            n_atoms = sum(1 for line in handle if line.startswith("ATOM"))
        report("Structure", f"{STRUCTURE} ({n_atoms} atom records)",
               *([] if n_atoms else
                 ["This file has no atom records, so it will not work."]))
else:
    report("Structure", f"'{STRUCTURE}' not found",
           "Check the path, or use a four-character PDB ID instead.")

report("Designs", NUM_DESIGNS)
report("Keeping", KEEP if KEEP else "nothing, so every position is redesigned")
if CHEMISTRY:
    report("Chemistry", CHEMISTRY)
report("Results in", OUTPUT_DIR)
"""),

    markdown("""
### Names you can use in `CHEMISTRY`

Only needed if you are using the `CHEMISTRY` setting. Run this cell for the
list. `docs/soft_residue_priors.md` explains the options in full.
"""),

    code("""
from training.soft_priors import list_residue_classes

print(list_residue_classes())
"""),

    markdown("""
## Step 3 — Generate the designs

Run the cell below and wait. One design takes anywhere from under a minute to
several minutes, depending on how large the protein is and whether you have a
GPU. Progress is printed as each design finishes.
"""),

    code("""
import subprocess
import time


def build_command(output_dir, seed):
    command = [
        sys.executable, "training/inpainting.py",
        "--pdb_input", STRUCTURE,
        "--model", MODEL,
        "--output-dir", output_dir,
        "--seed", str(seed),
    ]
    if HAVE_DATASET:
        command += ["--split_json", str(SPLIT_JSON), "--map_pkl", str(MAP_PKL)]
    if KEEP:
        command += ["--fixed-positions", KEEP]
    else:
        # The sampler has to be told which positions to design. A mask ratio of
        # 1.0 means all of them.
        command += ["--mask-ratio", "1.0"]
    if CHEMISTRY:
        command += ["--soft-priors", CHEMISTRY, "--prior-strength", "5.0"]
    return command


completed = []

for i in range(1, NUM_DESIGNS + 1):
    run_dir = f"{OUTPUT_DIR}_{i}" if NUM_DESIGNS > 1 else OUTPUT_DIR
    print(f"Design {i} of {NUM_DESIGNS} ... ", end="", flush=True)

    started = time.time()
    result = subprocess.run(
        build_command(run_dir, SEED + i - 1), capture_output=True, text=True
    )
    elapsed = time.time() - started

    # Check for the results file rather than trusting the exit status, so a run
    # that stopped early cannot be reported as a success.
    if (Path(run_dir) / "inpainting_results.json").exists():
        completed.append(run_dir)
        print(f"done in {elapsed:.0f}s -> {run_dir}")
    else:
        output = result.stdout.splitlines() + result.stderr.splitlines()
        complaints = [line.strip() for line in output if "rror" in line]
        print("did not finish")
        print(f"  {complaints[-1] if complaints else 'No error message was printed.'}")
        FAILED_OUTPUT = output  # kept so you can inspect it if you need to

print()
print(f"{len(completed)} of {NUM_DESIGNS} design(s) finished.")
if len(completed) < NUM_DESIGNS:
    print("Troubleshooting is in docs/GETTING_STARTED.md. For the full output of")
    print("the last failed run, print FAILED_OUTPUT.")
"""),

    markdown("""
## Step 4 — Look at the results

The sampler writes sequences out as numbers. The next cell converts them into
letters and prints a summary of each design.
"""),

    code("""
from inversefolddir_tools import load_results

designs = [load_results(d) for d in completed]

if not designs:
    print("There are no results to show, because no design finished in step 3.")

for i, design in enumerate(designs, start=1):
    print(f"Design {i}")
    print("-" * 60)
    design.summary()
    print()
"""),

    markdown("""
### What the summary is telling you

**Length** is the number of residues in the backbone you gave it.

**Redesigned** is how many of those positions the model was free to change. The
rest are the ones you listed in `KEEP`.

**Confidence** is how strongly the model preferred the residue it picked,
averaged over the sequence, on a scale from 0 to 1. It reflects how constrained
each position looked to the model. It is not a prediction that the protein will
work.

**Identity** is the percentage of positions where the design matches the
sequence your structure came with. It only appears when the input had a
sequence to compare against. A low value is not a problem in itself, since the
backbone is what the design is built to fit, not the original sequence.
"""),

    markdown("""
### Which positions changed

Prints a position-by-position comparison of the first design against the
sequence your structure came with. Change the index to look at a different
design, or raise `limit` to see more rows.

If the backbone has no sequence of its own, there is nothing to compare
against, so this prints the designed sequence instead of a table.
"""),

    code("""
from inversefolddir_tools import compare

if designs:
    compare(designs[0], limit=25)
"""),

    markdown("""
### Positions the model was least sure about

These are worth a look. If any of them land on a residue you care about, add it
to `KEEP` in step 2 and run the notebook again from there.
"""),

    code("""
from inversefolddir_tools import low_confidence_positions

for i, design in enumerate(designs, start=1):
    uncertain = low_confidence_positions(design, threshold=0.5)
    if uncertain:
        print(f"Design {i}: {len(uncertain)} position(s) below 0.5 -> {uncertain[:20]}")
    else:
        print(f"Design {i}: nothing below 0.5")
"""),

    markdown("""
## Step 5 — Save the sequences

Writes the designs to a FASTA file, which is the format sequence-ordering
services and alignment tools expect.
"""),

    code("""
from inversefolddir_tools import write_fasta

if designs:
    fasta_path = write_fasta(designs, f"{RUN_NAME}.fasta", name=RUN_NAME)
    print()
    print(fasta_path.read_text())
"""),

    markdown("""
## Step 6 — Before you order anything

These sequences are the model's answer to one question: which residues suit
this backbone. That is not the same as a protein that expresses, folds, stays
soluble, or does its job. None of those are being predicted here.

So screen the designs on a computer before you spend money on any of them.
Predict a structure for each sequence and compare it against the backbone you
started from; the ones that disagree with the input are the cheapest to drop.
Then order a panel rather than a single sequence, since some designs will not
work out and you want enough candidates that this is survivable.

If a specific function has to survive the redesign, the most direct control you
have over that is `KEEP`. Put the residues you know are essential in it.

## Where to go next

| If you want to | Read |
|---|---|
| Understand every available option | `docs/GETTING_STARTED.md` |
| Steer chemistry without fixing exact residues | `docs/soft_residue_priors.md` |
| Run many structures in one go | `example_scripts_for_prediction/batch_processing.sh` |
| Report a problem | <https://github.com/AlpTartici/inversefolddir/issues> |
"""),
]


def main():
    notebook = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {
                "display_name": "inv_fold",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    output = Path(__file__).parent / "quickstart.ipynb"
    output.write_text(json.dumps(notebook, indent=1) + "\n")
    print(f"Wrote {output} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
