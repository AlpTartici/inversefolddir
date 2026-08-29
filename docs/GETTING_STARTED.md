# Getting Started with Inverse FoldDir

**For experimentalists.** This guide assumes you know proteins but don't develop computational tools. It
walks from a fresh machine to a sequence you can order, with copy-paste commands
at every step.

If a command fails, jump to [When something goes wrong](#when-something-goes-wrong)
at the bottom — the common failures are all listed there with fixes.

---

## What this tool does

You give it **a protein structure**. It gives you **amino-acid sequences
predicted to fold into that structure.**

That's it. Some things it is useful for:

- You have a backbone from a design tool and need a sequence for it.
- You want variants of an existing protein that keep the fold but change the sequence.
- You want to redesign a loop or surface while keeping a binding site intact.

Some things it does **not** do:

- It does not predict whether your protein will express, be soluble, or be stable.
- It does not design binding to a new target.
- It does not guarantee function. A sequence that fits the backbone may still
  fail experimentally. Plan to test several candidates rather than one.

---

## Step 1 — Check what you need

You need:

| Requirement | Why | How to check |
|---|---|---|
| Python 3.9 or newer | Runs the tool | `python --version` |
| ~5 GB disk space | Model weights and dependencies | `df -h .` |
| A GPU | **Optional.** ~30 s per design instead of ~5 min | `nvidia-smi` — a table means you have one |

Linux, macOS (Intel and Apple Silicon), and Windows all work. **A GPU is not
required** — everything runs on CPU, just slower.

**On a shared cluster?** Do the install and the runs inside an interactive job
rather than on the login node. On SLURM systems that is usually
`srun --pty bash` or your site's equivalent.

---

## Step 2 — Install

Works on Linux, macOS, and Windows, with or without a GPU.

```bash
git clone https://github.com/AlpTartici/inversefolddir.git
cd inversefolddir

python -m venv ifd-env
source ifd-env/bin/activate        # Windows: ifd-env\Scripts\activate

# Install PyTorch for your machine -- pick one
pip install torch --index-url https://download.pytorch.org/whl/cpu     # no GPU / macOS
pip install torch --index-url https://download.pytorch.org/whl/cu126   # NVIDIA GPU

pip install -r requirements.txt

# Optional: only if you want to use the notebooks
pip install -r requirements-notebooks.txt
```

**Remember to activate the environment in every new terminal**
(`source ifd-env/bin/activate`).

Check it worked:

```bash
python check_install.py
```

This prints exactly what is installed and what is missing. If it says
**"Ready to design sequences"**, continue to Step 3.

> **Prefer conda?** `bash install_inv_fold_dir.sh` then `conda activate inv_fold`
> does the same thing and picks a PyTorch build matching your GPU driver.
>
> **Install problems?** [docs/INSTALL.md](INSTALL.md) covers platform-specific
> notes and a troubleshooting table. Errors mentioning `torch-scatter`,
> `torch-sparse`, or `torch-cluster` can be ignored — those are optional.

---

## Step 3 — Get the model weights

The trained model is a separate file (too large for the code repository). It is
hosted on the Hugging Face Hub and downloaded with a helper script:

```bash
pip install huggingface_hub          # one time, if not already installed
python scripts/download_checkpoints.py
```

This writes the checkpoint into `ckpts/`. To check what you got:

```bash
ls -lh ckpts/*.pt
```

You should see a `.pt` file of roughly 38 MB. If the file is only a few hundred
bytes, the download did not complete — run the script again.

To put the weights somewhere else (a shared scratch directory, say), either pass
`--output-dir` or set `IFD_CKPT_DIR`:

```bash
export IFD_CKPT_DIR=/path/to/shared/ckpts
python scripts/download_checkpoints.py
```

Every command below assumes the checkpoint is in `ckpts/`. If you used a
different directory, adjust the `--model` path accordingly.

### Do you need the CATH reference dataset?

**If you are designing from your own structure file: no.** Nothing else to
download. Skip to Step 4.

You need it to look a structure up by **UniProt accession** (`--uniprot`) or
**PDB ID** (`--pdb-id`), because those modes search a prebuilt index; to run the
CATH sequence-recovery benchmark; and to train. Designing from a file with
`--pdb_input` reads your structure directly and never touches it.

To fetch it:

```bash
./datasets/download_data.sh
```

That pulls the files from the same public Hugging Face repository as the
weights and checks each one against a SHA256 recorded in the script — no
credentials, and a truncated download fails loudly instead of producing a
confusing error later. `--verify` re-checks what you already have without
downloading, and `IFD_DATA_DIR` sends the files somewhere other than
`datasets/`.

| File | Size | Needed for |
|---|---|---|
| `chain_set_splits.json` | ~1 MB | the train/validation/test split (already in the repo) |
| `chain_set_map_with_b_factors_dssp.pkl` | ~280 MB | ID lookup, the recovery benchmark, training |
| `val_entries_with_less_than_300_continuous_coords.pkl` | ~3 MB | generative validation during training |

The entry points look in `datasets/cath-4.2/` by default. To point at a copy
held elsewhere, pass the paths explicitly:

```bash
--split_json datasets/cath-4.2/chain_set_splits.json \
--map_pkl datasets/cath-4.2/chain_set_map_with_b_factors_dssp.pkl
```

> **No DSSP program is required.** The secondary-structure annotations are
> already inside the pickle, and the released checkpoint was trained with the
> DSSP auxiliary head disabled, so nothing recomputes them.

> **Tip:** you can also fetch any PDB entry as a file and use `--pdb_input`,
> which avoids the download entirely:
> `wget https://files.rcsb.org/download/3OGO.cif`

---

## Step 4 — Prepare your input

### The simplest case: a structure already in the PDB

You need nothing. Just use the 4-character PDB ID, for example `3OGO`. The tool
downloads it for you.

### Your own structure file

**`.pdb` and `.cif` (mmCIF) both work.** Pass either one to `--pdb_input`;
mmCIF files are converted automatically, so there is no separate step.

| Requirement | Detail |
|---|---|
| Format | `.pdb` or `.cif` — both accepted directly |
| Backbone atoms | Must contain N, CA, C atoms. Side chains are ignored |
| One protein | If your file has several chains, the longest is used automatically |
| No gaps preferred | Missing residues are handled but reduce quality around the gap |

```bash
python training/inpainting.py --pdb_input mystructure.cif ...   # works
python training/inpainting.py --pdb_input mystructure.pdb ...   # also works
```

Converted files are written alongside the original as `mystructure.pdb` and
reused on later runs. Waters, ligands, and other heteroatoms are dropped during
conversion.

**Coming from AlphaFold?** Files from AlphaFold DB work directly, in either format.

**Coming from a backbone-design tool?** Structures with no residue identities
(all `UNK`, or backbone-only) work — that is the intended de novo case. See
["Identity" when there is no starting sequence](#identity-when-there-is-no-starting-sequence)
below for how the output differs.

**Checking your file is valid:**

```bash
grep -c "^ATOM" mystructure.pdb
```

If this prints `0`, the file has no atom records and will not work.

---

## Step 5 — Run your first design

Start with this. It downloads a small example structure and designs a complete
new sequence for it:

```bash
python training/inpainting.py \
    --pdb_input 3OGO \
    --model ckpts/inverse_folddir_model.pt \
    --mask-ratio 1.0 \
    --output-dir output/my_first_design
```

**What each line means:**

| Flag | Meaning |
|---|---|
| `--pdb_input` | Your structure: a PDB ID, or a path like `mystructure.pdb` |
| `--model` | The model weights file from Step 3 |
| `--mask-ratio 1.0` | Redesign **every** position. `1.0` = full redesign |
| `--output-dir` | Where results go. Use a new folder each time |

It takes 30 seconds to 5 minutes. When it finishes you will see
`INPAINTING COMPLETED SUCCESSFULLY!` and a sequence printed on screen.

---

## Step 6 — Read your results

The output folder contains two files. The JSON stores sequences as numbers,
which is not useful at the bench, so use the helper:

```bash
python -c "
from inversefolddir_tools import load_results
r = load_results('output/my_first_design')
r.summary()
"
```

This prints a plain summary:

```
Length      : 231 residues
Redesigned  : 231 positions (100%), 0 held fixed
Confidence  : 0.94 average (1.00 = fully certain, 0.05 = no idea)
Identity    : 24% to the starting sequence

Sequence:
LKGEKLLSGKLPLSLSLEFKVNGKPGSLESSGSL...
```

**How to read this:**

- **Confidence** — how sure the model is, averaged over positions. Above 0.8 is
  typical and good. Below 0.5 at a position means the model had no strong
  preference there.
- **Identity** — how similar the design is to the original sequence. Low identity
  (20–40%) is normal and expected for full redesign; the fold is conserved, not
  the sequence.

#### "Identity" when there is no starting sequence

Identity compares your design against the residues in the **input structure**.
It comes from the residue names in the file (`ALA`, `GLY`, ...), not from side
chain coordinates — so stripping side chains does not affect it.

But if your structure has no residue identities at all — a de novo backbone, or
a file whose residues are `UNK` — there is nothing to compare against. In that
case the summary says so explicitly rather than printing a meaningless number:

```
Identity    : not applicable -- the input backbone carries no residue identities,
              so there is no starting sequence to compare against.
```

If only *some* positions have known residues, identity is computed over just
those, and the count is shown:

```
Identity    : 22% to the starting sequence  (over 115 of 231 positions with a known residue)
```

Note that the `accuracy` field in the raw JSON behaves the same way — it also
ignores unknown positions — so on a fully de novo backbone it is not a
meaningful number. **For de novo designs, judge the result by refolding it and
comparing to your target backbone, not by identity or accuracy.**

**To get a FASTA file** (what you paste into an ordering form or BLAST):

```bash
python -c "
from inversefolddir_tools import load_results, write_fasta
write_fasta(load_results('output/my_first_design'), 'my_design.fasta')
"
```

**To see exactly what changed:**

```bash
python -c "
from inversefolddir_tools import load_results, compare
compare(load_results('output/my_first_design'))
"
```

```
  Pos  From  To      Conf  Note
----------------------------------------------------------
    1  Ser   Leu     0.93  polar -> hydrophobic
    5  Glu   Lys     0.94  negative -> positive
    7  Phe   Leu     0.96  same chemistry
```

---

## Step 7 — Control the design

Full redesign is rarely what you want. Usually some residues must be preserved.

### Keep specific residues exactly

Catalytic residues, disulfide cysteines, binding hot spots:

```bash
python training/inpainting.py \
    --pdb_input 3OGO \
    --model ckpts/inverse_folddir_model.pt \
    --fixed-positions "C22,C96,W47" \
    --output-dir output/design_fixed
```

`"C22,C96,W47"` means *position 22 must stay cysteine, 96 cysteine, 47
tryptophan*. **Positions count from 1**, and the letters are checked against
your structure — if position 22 is not actually a cysteine, you get an error
instead of a silently wrong design. That check is deliberate; it catches
numbering mistakes.

You can also write positions without letters (`"22,96,47"`) to skip the check.

### Nudge chemistry without fixing an exact residue

When you know what *kind* of residue you want but not which one:

```bash
python training/inpainting.py \
    --pdb_input 3OGO \
    --model ckpts/inverse_folddir_model.pt \
    --fixed-positions "C22,C96" \
    --soft-priors "34:polar, 57:metal_binding" \
    --prior-strength 5.0 \
    --output-dir output/design_priors
```

This keeps the cysteines, and *prefers* polar residues at 34 and
metal-coordinating residues at 57 — without forcing a specific one.

To see all the available class names (`polar`, `hydrophobic`, `aromatic`,
`charged`, `metal_binding`, ...):

```bash
python training/inpainting.py --list-residue-classes
```

Full details: [docs/soft_residue_priors.md](soft_residue_priors.md).

### Make several different designs

Each run is random, so running the same command repeatedly gives you different
sequences. This is the normal way to build a candidate panel:

```bash
for i in 1 2 3 4 5; do
    python training/inpainting.py \
        --pdb_input 3OGO \
        --model ckpts/inverse_folddir_model.pt \
        --fixed-positions "C22,C96" \
        --output-dir output/candidate_$i
done
```

Then collect them into one FASTA:

```bash
python -c "
from inversefolddir_tools import load_many, write_fasta
write_fasta(load_many('output', 'candidate_*'), 'all_candidates.fasta')
"
```

---

## Step 8 — Decide what to order

The model tells you a sequence fits the backbone. It does **not** tell you the
protein will work. Before ordering:

1. **Check the fold computationally.** Fold your designs with ESMFold or
   AlphaFold and compare to your input backbone. A TM-score above 0.9 means the
   design recovers the intended structure. This is the single most useful filter
   available before you spend money.

2. **Look at low-confidence positions.** These are the model's weakest choices:

   ```bash
   python -c "
   from inversefolddir_tools import load_results, low_confidence_positions
   print(low_confidence_positions(load_results('output/my_first_design')))
   "
   ```

   Consider fixing these to the native residue with `--fixed-positions`.

3. **Order several.** Structural self-consistency does not guarantee expression,
   solubility, or function, so expect only a fraction of designs to work. Test a
   panel rather than a single candidate.

4. **Keep more residues fixed if function matters.** Preserving residues you know
   are important — catalytic sites, disulfides, interface positions — is the most
   direct way to retain a specific function. Fully unconstrained designs can
   recover a fold while losing the activity you care about.

---

## Settings that do not travel with the checkpoint

Most of what the sampler needs — layer sizes, graph featurization, the flow's
alpha grid — is stored **inside** the checkpoint and read back automatically.
You cannot get those wrong.

Three things are **not** stored, and take command-line defaults instead. If you
use a checkpoint trained at other settings, or copy a command from a paper,
these are the ones to check.

| Setting | `inpainting.py` | `sample.py` | Default | What it is |
|---|---|---|---|---|
| Flow horizon | `--t_max` | `--T` | 8.0 | How far the flow integrates. Should match what the model was trained at, unless you mean to extrapolate |
| Step count | `--steps` | `--steps` | 20 | Euler steps across the horizon; `dt = T / (steps - 1)` |
| Sampling temperature | `--flow_temp` | `--flow_temp` | 1.0 | Higher is more diverse, lower more conservative |

Two consequences worth spelling out:

- **The horizon flag is spelled differently by the two entry points.**
  `inpainting.py` takes `--t_max`, `sample.py` takes `--T`. They mean the same
  thing. This is the easiest way to accidentally sample a model over the wrong
  horizon.
- **Raising the horizon is supported, but it is extrapolation.** The released
  checkpoint was trained at `t_max = 8`. Sampling at a longer horizon rebuilds
  the Dirichlet c_factor table to cover it — you will see a message saying so,
  and model load takes a few seconds longer — but the model is being asked about
  times it never saw in training. Validate the output if you go there.

### Other flags with sharp edges

- **`--no_probabilities`** skips writing the per-residue probability NPZ. It
  does not stop the probabilities being computed — sample selection and
  perplexity need them. Use it to save disk, not time.
- **`--output_prefix`** names the output files. `sample.py` falls back to
  `protein_sampling` when you leave it unset.
- **Designs are stochastic, and that is deliberate.** Running the same command
  twice gives two different sequences -- that is how you get a panel of
  candidates from one backbone. If you want a specific design back, pass
  `--seed`:

  ```bash
  python training/inpainting.py --pdb_input MYFILE.pdb \
      --model ckpts/inverse_folddir_model.pt --mask-ratio 1.0 \
      --seed 42 --output-dir output/run1
  ```

  Record the seed alongside any sequence you order. Both entry points take
  `--seed`; without it, neither is reproducible.
- **`--fixed-positions` and `--forced-positions` cannot be combined with
  `--mask-ratio`.** They already say which positions are held, so the mask is
  implied. Pick one or the other.
- **The validated position format checks you.** `--fixed-positions "C22,C96"`
  asserts that position 22 really is a cysteine and fails loudly if it is not.
  That is the point of it -- use the bare form `"22,96"` if you do not want the
  check.
- **Exit codes mean something.** `sample.py` and batch `inpainting.py` return
  `0` when every structure was designed, `2` when some failed (the output files
  then hold only the successful ones), and `1` when nothing was produced. Check
  the status if you are driving either from a script.

---

## When something goes wrong

| Message | What it means | Fix |
|---|---|---|
| `Must specify either mask_positions, known_sequence, or positive mask_ratio` | You did not say which positions to design | Add `--mask-ratio 1.0` for full redesign, or `--fixed-positions "..."` |
| `No module named 'torch_geometric'` | Environment not active | Run `conda activate inv_fold` |
| `invalid load key, 'v'` | A downloaded file is a Git LFS placeholder, not the real thing | Re-download it. Real checkpoints are tens of MB; a few-hundred-byte file is a placeholder |
| `needs the CATH reference dataset` | You used `--uniprot` or `--pdb-id` without the reference files | Design from a file with `--pdb_input yourfile.pdb`, or run `./datasets/download_data.sh` (Step 3) |
| `Could not convert ... from mmCIF to PDB` | The `.cif` is malformed or has no standard residues | Check the file opens in PyMOL/ChimeraX; try re-downloading it |
| `Position 22: Expected C but found A in structure` | Your numbering does not match the structure | Check your PDB numbering. Positions count from 1 at the first residue **present in the file** |
| `CUDA out of memory` | Protein too large for your GPU | Add `--steps 10`, or run on CPU |
| `These positions are both soft-prior and fixed` | A position appears in both flags | Remove it from one of them |
| Nothing happens for a long time | Normal on CPU | Wait ~5 min. Add `--verbose` to watch progress |

**Still stuck?** Open an issue at
<https://github.com/AlpTartici/inversefolddir/issues> and include the full
command you ran and the complete error message.

---

## Quick reference

```bash
conda activate inv_fold                       # every new terminal

# Full redesign
python training/inpainting.py --pdb_input MYFILE.pdb \
    --model ckpts/inverse_folddir_model.pt --mask-ratio 1.0 --output-dir output/run1

# Keep key residues
python training/inpainting.py --pdb_input MYFILE.pdb \
    --model ckpts/inverse_folddir_model.pt --fixed-positions "C22,C96" --output-dir output/run2

# Read results
python -c "from inversefolddir_tools import load_results; load_results('output/run1').summary()"
```

| I want to... | Flag |
|---|---|
| Redesign everything | `--mask-ratio 1.0` |
| Keep certain residues | `--fixed-positions "C22,W47"` |
| Force specific residues | `--forced-positions "A22,K47"` |
| Prefer a chemistry | `--soft-priors "34:polar"` |
| See more detail while running | `--verbose` |
| Run faster / rougher | `--steps 10` |
