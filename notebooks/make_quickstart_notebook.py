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


def markdown(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip().split("\n")}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip().split("\n")}


CELLS = [
    markdown("""
# Inverse FoldDir — Quickstart

**Design protein sequences from a backbone structure.**

This notebook takes you from a structure to sequences you can order. You do not
need to know Python — run each cell in order with `Shift + Enter` and change
only the settings marked **EDIT THIS**.

### Before you start

You need the `inv_fold` environment installed and selected as this notebook's
kernel. If you have not installed it yet, see `docs/GETTING_STARTED.md` Step 2.

### What you will do

1. Check the setup works
2. Point at your structure
3. Generate designs
4. Read and export the results
"""),

    markdown("""
---
## 1. Check the setup

Run this cell. It confirms the environment is working and tells you whether a
GPU is available.
"""),

    code("""
import sys, os
from pathlib import Path

# Work from the repository root, regardless of where Jupyter was launched.
if Path.cwd().name == "notebooks":
    os.chdir("..")
sys.path.insert(0, str(Path.cwd()))

try:
    import torch
    gpu = torch.cuda.is_available()
    print("Environment OK")
    print(f"GPU available: {gpu}" + ("" if gpu else "  (CPU works too, just slower)"))
except ImportError:
    print("PROBLEM: the inv_fold environment is not active.")
    print("Fix: select the 'inv_fold' kernel (Kernel > Change Kernel), then re-run.")

print(f"Working directory: {Path.cwd()}")
"""),

    markdown("""
### Find the model weights

The trained model is a separate ~38 MB file in `ckpts/`. This cell looks for it.
"""),

    code("""
checkpoints = sorted(Path("ckpts").glob("*.pt")) if Path("ckpts").exists() else []
usable = [c for c in checkpoints if c.stat().st_size > 1_000_000]

if usable:
    MODEL = str(usable[0])
    print(f"Using model: {MODEL}")
    if len(usable) > 1:
        print("Other available models:")
        for c in usable[1:]:
            print(f"   {c}")
else:
    MODEL = "ckpts/inverse_folddir_model.pt"
    print("No model weights found in ckpts/.")
    print("Download the checkpoint from the repository releases page first.")
    print("(Files of only a few hundred bytes are placeholders, not real weights.)")

# The CATH reference dataset is optional: it is only needed to look structures
# up by UniProt or PDB ID. Designing from a structure file does not use it.
SPLIT_JSON = Path("datasets/cath-4.2/chain_set_splits.json")
MAP_PKL = Path("datasets/cath-4.2/chain_set_map_with_b_factors_dssp.pkl")

have_dataset = (
    SPLIT_JSON.exists()
    and MAP_PKL.exists()
    and MAP_PKL.stat().st_size > 1_000_000
)

if have_dataset:
    print("CATH reference dataset: found (ID lookup available)")
else:
    print("CATH reference dataset: not present -- that is fine.")
    print("  Designing from a structure file works without it.")
    print("  It is only needed to look structures up by UniProt/PDB ID.")
"""),

    markdown("""
---
## 2. Choose your structure  <span style="color:#c00">EDIT THIS</span>

Two options:

- **A PDB ID** such as `"3OGO"` — downloaded automatically, nothing to prepare.
- **Your own file** — `.pdb` or `.cif` both work. mmCIF files are converted
  automatically. The file must contain backbone atoms (N, CA, C).

Backbones with no residue identities (de novo designs, or files whose residues
are `UNK`) are supported — that is the intended de novo case.
"""),

    code("""
# ============ EDIT THIS ============
STRUCTURE = "3OGO"          # a PDB ID, or a path like "mystructure.pdb"
OUTPUT_NAME = "my_design"   # a name for this experiment
# ===================================

OUTPUT_DIR = f"output/{OUTPUT_NAME}"

if len(STRUCTURE) == 4 and not STRUCTURE.lower().endswith((".pdb", ".cif")):
    print(f"Will download PDB ID: {STRUCTURE}")
elif Path(STRUCTURE).exists():
    if STRUCTURE.lower().endswith((".cif", ".mmcif")):
        print(f"Found {STRUCTURE} (mmCIF -- will be converted automatically)")
    else:
        n_atoms = sum(1 for line in open(STRUCTURE) if line.startswith("ATOM"))
        print(f"Found {STRUCTURE} with {n_atoms} atom records")
        if n_atoms == 0:
            print("WARNING: no ATOM records. This file will not work.")
else:
    print(f"Cannot find '{STRUCTURE}'.")
    print("Check the path, or use a 4-character PDB ID instead.")

print(f"Results will go to: {OUTPUT_DIR}")
"""),

    markdown("""
---
## 3. Choose how much to redesign  <span style="color:#c00">EDIT THIS</span>

This is the most important decision you make.

| Setting | What happens | When to use it |
|---|---|---|
| `FIXED = ""` | Every position redesigned | New backbones with no sequence to preserve |
| `FIXED = "C22,C96"` | Those residues kept exactly | Disulfides, catalytic residues, binding hot spots |

**Positions count from 1.** The letter is checked against your structure, so
`C22` fails loudly if position 22 is not a cysteine — that catches numbering
mistakes before you waste a run.

**If function matters, fix the residues you know are important.** Fully
unconstrained designs can recover a fold while losing the specific activity you
care about — catalytic sites, disulfides, and interface positions are the usual
ones to preserve.
"""),

    code("""
# ============ EDIT THIS ============
FIXED = ""            # e.g. "C22,C96,W47" to preserve those residues
SOFT_PRIORS = ""      # e.g. "34:polar, 57:metal_binding" to nudge chemistry
NUM_DESIGNS = 3       # how many different sequences to generate
# ===================================

if FIXED:
    print(f"Keeping fixed: {FIXED}")
else:
    print("Redesigning every position (nothing held fixed)")

if SOFT_PRIORS:
    print(f"Chemistry preferences: {SOFT_PRIORS}")

print(f"Generating {NUM_DESIGNS} design(s)")
"""),

    markdown("""
**Which chemistry names can I use?** Run the cell below to list them.
Details in `docs/soft_residue_priors.md`.
"""),

    code("""
from training.soft_priors import list_residue_classes
print(list_residue_classes())
"""),

    markdown("""
---
## 4. Generate the designs

Run this cell and wait. Each design takes 30 seconds to 5 minutes depending on
protein size and whether you have a GPU.

Each run is random, so the designs will differ from each other. That is intended
— it gives you a panel of candidates rather than one guess.
"""),

    code("""
import subprocess, time

def build_command(output_dir):
    cmd = [sys.executable, "training/inpainting.py",
           "--pdb_input", STRUCTURE,
           "--model", MODEL,
           "--output-dir", output_dir]
    if have_dataset:
        cmd += ["--split_json", str(SPLIT_JSON), "--map_pkl", str(MAP_PKL)]
    if FIXED:
        cmd += ["--fixed-positions", FIXED]
    else:
        # The sampler needs to be told what to design; 1.0 means everything.
        cmd += ["--mask-ratio", "1.0"]
    if SOFT_PRIORS:
        cmd += ["--soft-priors", SOFT_PRIORS, "--prior-strength", "5.0"]
    return cmd

successes = []
for i in range(1, NUM_DESIGNS + 1):
    run_dir = f"{OUTPUT_DIR}_{i}" if NUM_DESIGNS > 1 else OUTPUT_DIR
    print(f"[{i}/{NUM_DESIGNS}] designing -> {run_dir} ...", flush=True)
    started = time.time()
    result = subprocess.run(build_command(run_dir), capture_output=True, text=True)

    if result.returncode == 0:
        successes.append(run_dir)
        print(f"    done in {time.time() - started:.0f}s")
    else:
        # Surface the real error rather than a stack trace.
        tail = [l for l in result.stdout.split("\\n") + result.stderr.split("\\n")
                if "Error" in l or "error" in l]
        print(f"    FAILED: {tail[-1] if tail else 'see docs/GETTING_STARTED.md'}")

print(f"\\n{len(successes)} of {NUM_DESIGNS} design(s) completed.")
"""),

    markdown("""
---
## 5. Read the results

The raw output stores sequences as numbers. These helpers convert them into
something you can actually use.
"""),

    code("""
from inversefolddir_tools import load_results

designs = [load_results(d) for d in successes]

for i, design in enumerate(designs, start=1):
    print(f"{'=' * 60}\\nDESIGN {i}\\n{'=' * 60}")
    design.summary()
    print()
"""),

    markdown("""
**Reading the numbers:**

- **Confidence** — how certain the model was, averaged across positions. Above
  0.8 is typical. Individual positions below 0.5 are the model's weakest guesses.
- **Identity** — similarity to the original sequence. 20–40% is normal for a full
  redesign. The *fold* is preserved, not the sequence.
"""),

    markdown("""
### What changed, position by position
"""),

    code("""
from inversefolddir_tools import compare

if designs:
    compare(designs[0], limit=25)   # change the index to inspect another design
"""),

    markdown("""
### Positions the model was unsure about

These are worth a second look. If any sit at a site you care about, consider
adding them to `FIXED` above and re-running.
"""),

    code("""
from inversefolddir_tools import low_confidence_positions

for i, design in enumerate(designs, start=1):
    uncertain = low_confidence_positions(design, threshold=0.5)
    if uncertain:
        print(f"Design {i}: {len(uncertain)} uncertain position(s) -> {uncertain[:20]}")
    else:
        print(f"Design {i}: no low-confidence positions")
"""),

    markdown("""
---
## 6. Export for ordering

Writes a FASTA file — the format synthesis services, BLAST, and alignment tools
expect.
"""),

    code("""
from inversefolddir_tools import write_fasta

if designs:
    write_fasta(designs, f"{OUTPUT_NAME}.fasta", name=OUTPUT_NAME)
    print()
    print(open(f"{OUTPUT_NAME}.fasta").read()[:400])
"""),

    markdown("""
---
## 7. Before you order — please read

The model says these sequences fit the backbone. It does **not** say they will
express, fold, be soluble, or work.

**Filter computationally first.** Fold each design with ESMFold or AlphaFold and
compare to your input backbone. A TM-score above 0.9 means the design recovers
the intended structure. This is the single most useful check you can run before
spending money.

**Order several.** Structural self-consistency does not guarantee expression,
solubility, or function, so expect only a fraction of designs to work. Test a
panel rather than one candidate.

**Preserve what matters.** Fixing residues you know are important — catalytic
sites, disulfides, interface positions — is the most direct way to retain a
specific function.

---

### Where to go next

| I want to... | See |
|---|---|
| Understand every option | `docs/GETTING_STARTED.md` |
| Bias chemistry without fixing residues | `docs/soft_residue_priors.md` |
| Run many structures at once | `example_scripts_for_prediction/batch_processing.sh` |
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
    output.write_text(json.dumps(notebook, indent=1))
    print(f"Wrote {output} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
