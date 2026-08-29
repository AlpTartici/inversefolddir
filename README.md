# Inverse Folding: Structure-Conditioned Protein Sequence Sampling

A generative protein sequence design framework using **Dirichlet Flow Matching (DFM)** for structure-conditioned sequence generation. This repository implements deep generative models that learn to predict amino acid sequences from protein backbone structures.

## New here?

**If you are a bench scientist and mainly want sequences to order, start with
one of these two — they assume no Python experience:**

| Start here | What it is |
|---|---|
| **[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)** | Step-by-step guide: install, prepare input, run, read output, troubleshoot |
| **[notebooks/quickstart.ipynb](notebooks/quickstart.ipynb)** | Same workflow as a notebook — edit two cells, run the rest |
| **[docs/INSTALL.md](docs/INSTALL.md)** | Platform notes (Linux/macOS/Windows, CPU or GPU) and install troubleshooting |

Already installed? Check it with `python check_install.py`.

The rest of this README is the full technical reference.

---

## Quick Start

```bash
# Clone repository
git clone https://github.com/AlpTartici/inversefolddir.git
cd inversefolddir

# Option 1: pip (most portable -- Linux, macOS, Windows, CPU or GPU)
python -m venv ifd-env && source ifd-env/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or .../cu126 for NVIDIA
pip install -r requirements.txt

# Option 2: conda (detects your GPU driver automatically)
bash install_inv_fold_dir.sh
conda activate inv_fold

# Verify either way
python check_install.py

# Get the model weights (~38 MB, from Hugging Face)
pip install huggingface_hub
python scripts/download_checkpoints.py

# Interactive notebook (recommended for first-time users)
jupyter lab notebooks/quickstart.ipynb

# Design a full sequence for a structure
python training/inpainting.py --pdb_input 3OGO --model ckpts/inverse_folddir_model.pt \
    --mask-ratio 1.0 --output-dir output/my_design

# Redesign everything except residues you want to keep
python training/inpainting.py --pdb_input 3OGO --model ckpts/inverse_folddir_model.pt \
    --fixed-positions "C22,C96" --output-dir output/my_design

# Turn the results into a readable sequence and a FASTA file
python -c "
from inversefolddir_tools import load_results, write_fasta
r = load_results('output/my_design'); r.summary(); write_fasta(r, 'design.fasta')"
```

> **Structure input:** `--pdb_input` accepts a PDB ID, a `.pdb` file, or a
> `.cif` file (converted automatically). Backbones with no residue identities
> (de novo designs, `UNK` residues) are supported.
>
> **CATH reference dataset:** only needed for `--uniprot` / `--pdb-id` lookup,
> for the recovery benchmark, and for training. Fetch it with
> `./datasets/download_data.sh`. Designing from a structure file requires no
> extra download.

---

## Table of Contents

0. **[Getting Started (for experimentalists)](docs/GETTING_STARTED.md)** ← start here if you are new
1. [Installation](#installation)
2. [Quick Usage Examples](#quick-usage-examples)
3. [Sampling Modes](#sampling-modes)
4. [Example Scripts](#example-scripts)
5. [Input Formats](#input-formats)
6. [Advanced Configuration](#advanced-configuration)
7. [Output Formats](#output-formats)
8. [Training Your Own Models](#training-your-own-models)
9. [Troubleshooting](#troubleshooting)

---

## Installation

Works on **Linux, macOS (Intel and Apple Silicon), and Windows**, with or
without a GPU. Full details, platform notes, and troubleshooting:
**[docs/INSTALL.md](docs/INSTALL.md)**.

### Option 1: pip (recommended, most portable)

```bash
python -m venv ifd-env
source ifd-env/bin/activate        # Windows: ifd-env\Scripts\activate

# PyTorch build for your machine -- pick one
pip install torch --index-url https://download.pytorch.org/whl/cpu     # no GPU / macOS
pip install torch --index-url https://download.pytorch.org/whl/cu126   # NVIDIA GPU

pip install -r requirements.txt
```

### Option 2: conda script

Detects your GPU driver and installs a matching PyTorch build.

```bash
bash install_inv_fold_dir.sh          # --cpu or --cuda 121 to override
conda activate inv_fold
```

### Verify

```bash
python check_install.py
```

Reports what is installed, what is missing, and how to fix it.

### Requirements

| | |
|---|---|
| Python | 3.9 or newer |
| GPU | Optional. ~30 s per design with one, ~5 min without |
| Disk | ~5 GB |

> **`torch-scatter` / `torch-sparse` / `torch-cluster` are not required.**
> They are compiled extensions that often fail to build, and this codebase
> has fallbacks. Build errors mentioning them can be ignored.


---

## Running Jupyter Notebooks

The environment includes Jupyter Lab and Jupyter Notebook for interactive experimentation:

```bash
# Activate environment
conda activate inv_fold        # or: source ifd-env/bin/activate

# Start Jupyter Lab (recommended)
jupyter lab

# Or start classic Jupyter Notebook
jupyter notebook

# Run a specific notebook
jupyter lab notebooks/quickstart.ipynb
```

### Available Notebooks

- **`notebooks/quickstart.ipynb`** - Guided walkthrough from a structure to an
  orderable sequence. Start here.

---

## Quick Usage Examples

### Full Sequence Sampling

Generate complete sequences conditioned on protein backbone structure:

```bash
# Navigate to training directory
cd training

# Using PDB ID
python sample.py --pdb_input 1abc --steps 20 --flow_temp 0.2

# Using local PDB file
python sample.py --pdb_input /path/to/protein.pdb --steps 30

# Multiple structures with ensemble sampling
python sample.py --pdb_input 1fcd.C --ensemble_size 5 --steps 20
```

### Sequence Inpainting

Predict specific amino acids while keeping others fixed:

```bash
# Navigate to training directory (if not already there)
cd training

# Mask specific positions (with validation)
python inpainting.py --pdb_input 1abc --mask-positions "D45,Y67,K89"

# Mask positions without validation
python inpainting.py --pdb_input 1abc --mask-positions "45,67,89"

# Template-based inpainting
python inpainting.py --pdb_input 1abc --template-sequence "ACDEFXHIKLXNPQXSTVWY"

# Random masking
python inpainting.py --pdb_input 1abc --mask-ratio 0.15
```

---

## Sampling Modes

### 1. Full Sequence Sampling

- **Purpose**: Generate complete amino acid sequences that fold to given structures
- **Use Cases**: De novo protein design, sequence optimization
- **Command**: `cd training && python sample.py`

### 2. Sequence Inpainting

- **Purpose**: Predict masked positions while conditioning on known residues
- **Use Cases**: Variant effect prediction, protein completion, mutation design
- **Command**: `cd training && python inpainting.py`

### 3. Soft Residue Priors

- **Purpose**: Bias positions toward a residue *class* or custom amino-acid
  distribution without fixing them to one identity
- **Use Cases**: Favoring polar residues on exposed surfaces, biasing a
  candidate metal site toward His/Cys/Asp/Glu, basic patches on
  nucleic-acid-binding surfaces
- **Parameters**: `--soft-priors`, `--soft-priors-json`, `--prior-strength`

```bash
# Nudge position 34 polar and 57 toward metal-coordinating residues,
# while keeping the disulfide cysteines fixed
python training/inpainting.py \
    --pdb_input 3OGO --model ckpts/inverse_folddir_model.pt \
    --fixed-positions "C22,C96" \
    --soft-priors "34:polar, 57:metal_binding" \
    --prior-strength 5.0

# See all available residue class names
python training/inpainting.py --list-residue-classes
```

Unlike fixed positions, soft-prior positions remain free to change during
denoising. Weights that sum to less than 1.0 have the remainder spread
evenly over the unnamed amino acids, so `"57:H0.4"` is a nudge rather than
a restriction. Full reference: [docs/soft_residue_priors.md](docs/soft_residue_priors.md).

### 4. Ensemble Sampling

- **Purpose**: Generate multiple structural variants for robust predictions
- **Use Cases**: Uncertainty quantification, consensus design
- **Parameters**: `--ensemble_size`, `--ensemble_consensus_strength`

---

## Example Scripts

Four runnable scripts live in `example_scripts_for_prediction/`. Each one sets
its own paths at the top and can be run directly:

| Script | What it does |
|---|---|
| `full_sampling.sh` | Full-sequence design over a structure file, with ensemble sampling |
| `inpainting_validated.sh` | Redesign specific positions, validating that each position holds the residue you expect (`D45` = position 45 must be D) |
| `inpainting_positions.sh` | The same, by position number only, without validation |
| `batch_processing.sh` | Many protein/position combinations from a CSV in one run |

```bash
cd example_scripts_for_prediction
bash full_sampling.sh
```

They are deliberately not reproduced here: an inline copy drifts from the file
it is quoting, and the flag spellings differ between the two entry points
(`sample.py` takes `--output_dir`, `inpainting.py` takes `--output-dir`). Read
the script you are about to run.

---

## Input Formats

### Mask Position Formats

#### 1. Position Only Format

```bash
--mask-positions "45,67,89"
```

- **Usage**: Mask these positions without validation
- **Format**: Comma-separated position numbers, **1-indexed** — the first
  residue of the sequence is position 1, matching PDB numbering
- **Example**: `"45,67,89"` masks the 45th, 67th and 89th residues

> **Positions are 1-indexed everywhere**, in this format and in the validated
> format below. Passing a 0-indexed position silently designs its neighbour, so
> check your numbering against the sequence before ordering anything.

#### 2. Position + Validation Format

```bash
--mask-positions "D45,Y67,K89"
```

- **Usage**: Mask positions 45, 67, 89 but first verify amino acids
- **Format**: `{amino_acid}{position}` format, 1-indexed
- **Validation**: Position 45 must have D, position 67 must have Y, position 89 must have K
- **Safety**: Program errors out if validation fails (perfect for variant effect studies)

### CSV File Format

For batch processing with `--list_csv`:

```csv
mutant,mutated_sequence,DMS_score,DMS_score_bin,mask-positions,protein
A1V,MVQPQVQHPIQ...,-2.1,low,1,PIN1_HUMAN
L2P,MPQPQVQHPIQ...,0.5,medium,2,PIN1_HUMAN
G3A,MVQAQVQHPIQ...,1.2,high,3,PIN1_HUMAN
D45A,MVQPQVQHPIQXIKLMNPQ...,0.8,medium,D45,PIN1_HUMAN
```

**Required columns:**

- `mutant`: Mutation identifier
- `mutated_sequence`: Sequence with mutations (optional for validation)
- `DMS_score`: Deep Mutational Scanning score
- `DMS_score_bin`: Score category (low/medium/high)
- `mask-positions`: Positions to mask (same format as `--mask-positions`)
- `protein`: UniProt ID (corresponds to `--uniprot` argument)

### Structure Input Formats

```bash
# Local PDB file
--pdb_input /path/to/protein.pdb

# Local CIF file
--pdb_input /path/to/protein.cif

# PDB ID from RCSB
--pdb_input 1abc

# PDB ID with specific chain
--pdb_input 1fcd.C
```

---

## Advanced Configuration

### Sampling Parameters

```bash
# Temperature control - Trade-off between diversity and structural recovery
--flow_temp 0.2          # Conservative sampling (better structure recovery, less diversity)
--flow_temp 1.0          # Balanced sampling
--flow_temp 2.0          # Diverse sampling (higher diversity, may compromise structure recovery)

# Integration steps (more = higher quality, slower)
--steps 10               # Fast sampling
--steps 20               # Balanced quality/speed
--steps 50               # High quality

# Flow horizon. NOTE the spelling differs between entry points:
--T 8.0                  # sample.py
--t_max 8.0              # inpainting.py -- same quantity
--t_min 0.0              # Minimum noise level

# Initial distribution - Controls noise variance and diversity
--dirichlet_concentration 20.0    # Default. Lower noise variance, less diversity
--dirichlet_concentration 1.0     # Higher noise variance, more diversity
--dirichlet_concentration 0.5     # Higher still
```

> The flow horizon and the step count are **not** stored in the checkpoint —
> unlike the architecture and featurization, which are. See
> [Settings that do not travel with the checkpoint](docs/GETTING_STARTED.md#settings-that-do-not-travel-with-the-checkpoint).

#### Parameter Guidelines:

**Temperature (`--flow_temp`)**:

- **Lower values (0.1-0.3)**: More conservative predictions with better average structural recovery but reduced sequence diversity
- **Higher values (1.0-2.0)**: More diverse predictions but may compromise average structural recovery
- **Recommended**: Start with 0.2-0.3 for most applications

**Dirichlet Concentration (`--dirichlet_concentration`)**:

- **Default**: 20.0, on both entry points. Concentrated initial noise; this is
  the setting the released checkpoint was evaluated at.
- **Lower values**: Increase noise variance -- more diversity, potentially lower
  structural recovery.
- **Recommended**: leave at 20.0 unless you are specifically chasing diversity.

### Ensemble Sampling

```bash
# Ensemble size (1-10 replicas)
--ensemble_size 5

# Consensus strength (0=independent, 1=full consensus)
--ensemble_consensus_strength 0.3

# Ensemble method
--ensemble_method arithmetic     # Mean in probability space
--ensemble_method geometric      # Mean in log space

# Structure noise for diversity
--structure_noise_mag_std 1.0    # Standard deviation (Angstroms)
--uncertainty_struct_noise_scaling  # Scale by B-factors
```

### Output Control

```bash
# Output directory
--output_dir ./results/my_experiment

# Output prefix
--output_prefix my_protein_design

# File formats
--save_probabilities        # Write the per-residue probability NPZ (default)
--no_probabilities          # Skip writing that file. Saves disk, not time:
                            # the probabilities are still computed, because
                            # sample selection and perplexity need them.
--detailed_json             # Generate time-step trajectories
```

---

## Output Formats

### 0. Single-run results (`inpainting_results.json` / `.npz`)

Every `inpainting.py` run writes these two files into `--output-dir`:

| Key | Contents |
|---|---|
| `predicted_sequence` | Designed sequence as **integer amino-acid indices**, not letters |
| `final_probabilities` | `[length, 21]` probability per position |
| `inpainting_mask` | `true` where the position was redesigned |
| `evaluation_metrics` | Accuracy, confidence, entropy, and `true_sequence` |

The sequence is stored as indices (`ACDEFGHIKLMNPQRSTVWY` + `X`, in that order),
so read it with the helper rather than by hand:

```python
from inversefolddir_tools import load_results, write_fasta, compare

r = load_results("output/my_design")
r.summary()                      # length, % redesigned, confidence, identity
print(r.sequence)                # one-letter string
write_fasta(r, "design.fasta")   # FASTA for ordering / BLAST
compare(r)                       # per-position table of what changed
```

`inversefolddir_tools.py` imports no torch and no model, so it runs anywhere,
including a login node without the `inv_fold` environment.

### 1. Sequence Files (CSV)

```csv
structure_idx,structure_name,length,predicted_sequence,true_sequence,accuracy
0,1abc,150,MKFLVLLFNISCV...,MKFLVLLFNISCV...,94.67
```

### 2. Probability Distributions (NPZ)

```python
import numpy as np

# Load results
data = np.load('results_probabilities.npz')

# Access probabilities for structure 0
probs = data['struct_0_probabilities']  # Shape: [seq_len, 21]
true_indices = data['struct_0_true_indices']  # Ground truth
predicted = data['struct_0_predicted_indices']  # Predictions
```

### 3. Trajectory Analysis (JSON)

```json
{
  "1abc": {
    "0": {
      "trajectory": [
        {
          "time_point": 0.0,
          "most_likely_amino_acid": "M",
          "current_probability": 0.456789,
          "amino_acid_breakdown": {
            "A": { "predicted_prob": 0.023, "current_prob": 0.045 },
            "M": { "predicted_prob": 0.678, "current_prob": 0.456 }
          }
        }
      ],
      "final_prediction": "M",
      "ground_truth": "M"
    }
  }
}
```

### 4. Metadata Files (TXT)

A plain-text sidecar recording the timestamp, the checkpoint, the split, and
the sampling parameters (steps, `T`) used for the run, followed by how many
structures succeeded and failed.

---

## Training Your Own Models

Training needs the CATH reference dataset:

```bash
./datasets/download_data.sh
```

That fetches the coordinates, splits and generative validation set from the
same Hugging Face repository as the weights, verifying each against a SHA256
(see [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md), Step 3). The AlphaFold
half of the corpus additionally needs pre-built chunk files; `helpers/` holds
the scripts that build those.

### Configuration

A training run is described by a JSON file in `configs/`. Command-line flags
override whatever the file sets.

```bash
python training/train.py --config_file configs/released_model.json
```

`configs/released_model.json` is the configuration the released checkpoint was
trained with, so it is the sensible starting point.

### Common flags

```bash
python training/train.py \
    --config_file configs/released_model.json \
    --batch 20 \
    --lr 5e-4 \
    --epochs 600 \
    --num_layers_gvp 5 \
    --hidden_dim 288
```

Run `python training/train.py --help` for the full list. Note the spellings:
`--batch` (not `--batch_size`) and `--lr` (not `--learning_rate`).

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

```bash
# Reduce the batch size
--batch 2

# Use an ensemble instead of a large batch
--ensemble_size 3 --batch 1

# Reduce the number of integration steps
--steps 10
```

#### 2. Model Loading Errors

```bash
# Navigate to training directory
cd training

# Check model path
ls -la /path/to/model.pt

# Try auto-discovery
python sample.py --pdb_input 1abc  # Will find *best*.pt files

# Verify model architecture
python sample.py --pdb_input 1abc --verbose
```

#### 3. PDB Download Issues

```bash
# Use local files instead
--pdb_input /path/to/local/file.pdb

# Check internet connection
curl -I https://files.rcsb.org/download/1abc.pdb

# Try different PDB ID format
--pdb_input 1ABC  # Sometimes case matters
```

#### 4. Dependency Conflicts

First run the diagnostic -- it names the specific problem:

```bash
python check_install.py
```

Most conflicts come from the compiled PyTorch Geometric extensions, which are
**not required**. If you see build errors mentioning `torch-scatter`,
`torch-sparse`, `torch-cluster`, or `torch-spline-conv`, skip them.

Clean reinstall:

```bash
python -m venv ifd-env && source ifd-env/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or .../cu126
pip install -r requirements.txt
python check_install.py
```

Full platform notes: [docs/INSTALL.md](docs/INSTALL.md).

#### 5. Performance Issues

```bash
# Enable GPU acceleration
nvidia-smi  # Check GPU availability

# Optimize parameters
--batch_size 4     # Balance memory/speed
--steps 20         # Reasonable quality/speed
--flow_temp 0.3    # Faster convergence
```

### Debug Mode

```bash
# Enable verbose output
--verbose

# Test with simple structure
--pdb_input 1abc --steps 5 --verbose

# Check model parameters
python -c "
import torch
checkpoint = torch.load('model.pt', map_location='cpu')
print('Available keys:', checkpoint.keys())
if 'args' in checkpoint:
    print('Model args:', checkpoint['args'])
"
```

### Performance Benchmarks

| Configuration   | Speed (seq/min) | Memory (GB) | Quality   |
| --------------- | --------------- | ----------- | --------- |
| CPU, steps=10   | 5               | 2           | Basic     |
| GPU, steps=20   | 50              | 4           | Good      |
| GPU, ensemble=5 | 25              | 8           | Excellent |

---

## Citation

The accompanying paper is not out yet. Until it is, cite the software:

```bibtex
@software{inversefolddir,
  title  = {Inverse FoldDir: Flexibility-Aware Structure-Conditioned Protein
            Sequence Design with Dirichlet Flow Matching},
  author = {Tartici, Alp and contributors},
  year   = {2026},
  url    = {https://github.com/AlpTartici/inversefolddir},
  note   = {Software. Model weights:
            https://huggingface.co/AlpTartici/inversefolddir}
}
```

A paper citation will replace this once the preprint is posted. If you are
reading this after that point and the block still says "not out yet", please
open an issue -- it means this file went stale.

---

## License

Released under the [MIT License](LICENSE).

### Relationship to the original repository

This work builds on [microsoft/InverseFoldDir](https://github.com/microsoft/InverseFoldDir),
released by Microsoft Corporation under the MIT License. This repository is a
derivative work that continues that codebase.

It is published as a curated snapshot rather than as a continuation of the
original git history: the development history contained large intermediate data
files and internal cluster configuration that are not appropriate to
redistribute. The original copyright notice, license terms, per-file license
headers, and third-party attributions are all retained, and the upstream
repository above remains the reference for the original commit history.

Substantial changes made in this work include a revised training objective and
learning algorithm, updated model architecture and graph featurization,
soft residue-prior conditioning, multichain context conditioning, and an
expanded evaluation and sampling pipeline.

The base method also incorporates:

- [gvp-pytorch](https://github.com/drorlab/gvp-pytorch) — Geometric Vector Perceptron (Jing et al.)
- [dirichlet-flow-matching](https://github.com/HannesStark/dirichlet-flow-matching) — Dirichlet Flow Matching (Stärk et al.)

See [NOTICE](NOTICE) for full third-party attribution, including the
upstream revisions these components were taken from.

---

## Contributing

Contributions and suggestions are welcome — please open an issue or pull
request.

> **Note:** this repository is independent of the Microsoft-hosted original.
> Contributions here are not covered by, and do not require, the Microsoft
> Contributor License Agreement. To contribute to the original project
> instead, see [microsoft/InverseFoldDir](https://github.com/microsoft/InverseFoldDir).

Participation is governed by the [Code of Conduct](CODE_OF_CONDUCT.md).
Report concerns about conduct in this repository to tartici@stanford.edu;
Microsoft does not moderate this repository.

## Trademarks

This project is not affiliated with or endorsed by Microsoft. Microsoft
trademarks and logos are not used here, and any use of them in a further
modified version must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/legal/intellectualproperty/trademarks/usage/general)
and must not imply Microsoft sponsorship. Third-party trademarks remain subject
to their owners' policies.

---

## Support

- **Issues**: [GitHub Issues](https://github.com/AlpTartici/inversefolddir/issues)

---
