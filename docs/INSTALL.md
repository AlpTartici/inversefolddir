# Installation

Works on **Linux, macOS (Intel and Apple Silicon), and Windows**, with or
without a GPU. Pick one of the two routes below.

If anything goes wrong, run `python check_install.py` — it reports exactly what
is missing and how to fix it.

---

## Route 1 — pip (recommended, most portable)

Three commands. Works in any Python 3.9+ environment: venv, conda, or system
Python.

```bash
# 1. Create and activate an environment
python -m venv ifd-env
source ifd-env/bin/activate        # Windows: ifd-env\Scripts\activate

# 2. Install PyTorch for YOUR machine -- pick one line
pip install torch --index-url https://download.pytorch.org/whl/cpu     # no GPU / macOS
pip install torch --index-url https://download.pytorch.org/whl/cu126   # NVIDIA GPU

# 3. Install everything else
pip install -r requirements.txt
```

Verify:

```bash
python check_install.py
```

**For the notebooks** (optional — not needed to design sequences):

```bash
pip install -r requirements-notebooks.txt
```

This is kept separate on purpose. It pulls in matplotlib, which depends on
`pillow`, which compiles from source when no prebuilt wheel matches your Python
version. Since pip installs atomically, bundling it would let a pillow build
failure block numpy and torch-geometric and leave you unable to run anything.
Keeping it separate means a plotting problem never blocks sequence design.

**Why PyTorch separately?** The right build depends on your operating system and
GPU driver, and a wrong guess is the most common installation failure. Choosing
it yourself takes one line and avoids the problem. If unsure, use the CPU line —
it works everywhere, just slower. The chooser at
<https://pytorch.org/get-started/locally/> gives the exact command for your setup.

---

## Route 2 — conda script

If you already use conda and want one command:

```bash
bash install_inv_fold_dir.sh
conda activate inv_fold
```

The script detects your GPU driver and installs a matching PyTorch build. To
override:

```bash
bash install_inv_fold_dir.sh --cpu          # force CPU-only
bash install_inv_fold_dir.sh --cuda 121     # force a CUDA version
bash install_inv_fold_dir.sh --name myenv   # different environment name
```

---

## Platform notes

### macOS (including Apple Silicon)

Works. Use the CPU PyTorch line — it includes Apple's MPS backend, which the
code will use automatically if available. Do **not** install `torch-scatter` or
the other compiled extensions; they frequently fail to build on macOS and are
not required.

### Windows

Works. Use PowerShell or WSL. In PowerShell, activate with
`ifd-env\Scripts\activate`. Otherwise identical.

### Linux / HPC clusters

Works. On a shared cluster, do the install inside an interactive job rather than
on the login node, and put the environment on a filesystem with quota to spare.
On SLURM systems, `srun --pty bash` or your site's equivalent gets you a shell.

**On an old distribution (CentOS/RHEL 7, glibc 2.17):** current NumPy and SciPy
wheels target `manylinux_2_28`, which needs glibc 2.28 or newer. On glibc 2.17
none of those wheels match, so pip silently falls back to building SciPy from
source and fails with `Dependency "OpenBLAS" not found`. Force wheels and let
pip resolve to versions that still ship `manylinux2014` builds:

```bash
pip install --only-binary=:all: -r requirements.txt
```

Check your glibc with `ldd --version`. This does not affect current
distributions, macOS, or Windows.

### No GPU

Fully supported. A design takes roughly 5 minutes on CPU versus about 30 seconds
on GPU. Everything else is identical.

---

## Optional extras

| File | What it adds | Needed for |
|---|---|---|
| `requirements.txt` | Core | Designing sequences. This is all most people need. |
| `requirements-notebooks.txt` | matplotlib, Jupyter | `notebooks/quickstart.ipynb` and its plots |
| `requirements-eval.txt` | tmtools, transformers, parasail | `eval/evaluation_pipeline.py` -- refolding designs and scoring them against the input backbone |

Designing works with `requirements.txt` alone; the other two are opt-in because
they are heavier and less portable.

---

## About the compiled extensions

Older environment files in this repository list `torch-scatter`,
`torch-cluster`, `torch-sparse`, and `torch-spline-conv`. **You do not need
them.** They are compiled C++/CUDA extensions without prebuilt wheels for every
platform, and they are the single most common cause of failed installs.

- `torch-geometric` falls back to pure-PyTorch implementations without them.
- The one place this codebase used `torch-scatter` now has a built-in fallback
  that produces identical results.

If you want the extra speed on a supported platform:

```bash
pip install torch-scatter torch-cluster \
    -f "https://data.pyg.org/whl/torch-$(python -c 'import torch; print(torch.__version__)').html"
```

If that fails, ignore it and carry on.

---

## Model weights

The trained model is distributed separately (~38 MB) and goes in `ckpts/`.

```bash
ls -la ckpts/*.pt
```

A real checkpoint is around 38 MB. **If the file is only a few hundred bytes it
is a Git LFS pointer, not the model** — re-download it. `check_install.py`
detects this case explicitly.

---

## Optional: the CATH reference dataset

Only needed to look structures up by **UniProt accession** or **PDB ID**.
Designing from your own `.pdb` or `.cif` file requires no extra download.

If you want ID lookup, place `chain_set_splits.json` and
`chain_set_map_with_b_factors_dssp.pkl` in `datasets/cath-4.2/` and pass
`--split_json` / `--map_pkl`.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `No module named 'torch_geometric'` | Environment not activated | Activate it, then `pip install -r requirements.txt` |
| `No module named 'torch_scatter'` | Compiled extension absent | Nothing to do — a fallback handles it |
| `error: Microsoft Visual C++ 14.0 required` | Windows trying to compile an extension | Skip the compiled extensions; they are optional |
| Build errors mentioning `torch-sparse` / `torch-cluster` | Same | Same — not required |
| `CUDA error: no kernel image` | PyTorch build does not match your GPU | Reinstall PyTorch with the right CUDA version, or use the CPU build |
| `invalid load key, 'v'` | A downloaded file is a Git LFS pointer | Re-download it |
| `Dependency "OpenBLAS" not found` while installing SciPy | glibc too old for current wheels, so pip is building from source | `pip install --only-binary=:all: -r requirements.txt` — see Linux / HPC clusters above |
| `Failed building wheel for pillow` | matplotlib dependency compiling from source | Only affects notebook plots. Skip `requirements-notebooks.txt`, or install `libjpeg-dev zlib1g-dev` (Linux) / `brew install jpeg zlib` (macOS) |
| `Weights only load failed` / `WeightsUnpickler error` | PyTorch 2.6+ refuses to unpickle checkpoint metadata by default | Fixed in this codebase. If you see it, you are on an older copy — `git pull` |
| `killed` / out of memory | Protein too large | Add `--steps 10`, or use a machine with more RAM |

Still stuck? Run `python check_install.py` and include its full output when
opening an issue at <https://github.com/AlpTartici/inversefolddir/issues>.
