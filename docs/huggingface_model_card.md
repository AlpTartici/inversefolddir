---
license: mit
tags:
  - protein-design
  - inverse-folding
  - protein-sequence-design
  - flow-matching
  - biology
library_name: pytorch
pipeline_tag: other
---

<!--
This file is published verbatim as README.md of the Hugging Face model
repository. The YAML front matter above must stay first: HF parses it for the
license badge, tags, and search filters.

Replace the Citation section with the paper once it has a DOI or arXiv ID.
-->

# Inverse FoldDir

Structure-conditioned protein sequence design by Dirichlet flow matching.

Give it a protein backbone; it generates amino-acid sequences predicted to fold
into that backbone. Supports full-sequence generation, fixed-residue inpainting,
and soft residue-prior conditioning.

- **Code:** https://github.com/AlpTartici/inversefolddir
- **License:** MIT
- **Architecture:** SE(3)-equivariant GVP graph encoder + Dirichlet flow matching
- **Parameters:** 9.6M
- **Input:** protein backbone (`.pdb` or `.cif`); side chains not required
- **Output:** amino-acid sequence(s)

## Quick start

```bash
git clone https://github.com/AlpTartici/inversefolddir.git
cd inversefolddir

python -m venv ifd-env && source ifd-env/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Download the checkpoint:

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="AlpTartici/inversefolddir",
    filename="inverse_folddir_model.pt",
)
print(path)
```

Design a sequence:

```bash
python training/inpainting.py \
    --pdb_input your_structure.pdb \
    --model /path/from/hf_hub_download \
    --mask-ratio 1.0 \
    --output-dir output/my_design
```

Read the result:

```python
from inversefolddir_tools import load_results, write_fasta

r = load_results("output/my_design")
r.summary()
write_fasta(r, "design.fasta")
```

Full walkthrough: [docs/GETTING_STARTED.md](https://github.com/AlpTartici/inversefolddir/blob/main/docs/GETTING_STARTED.md).

## Design modes

**Full generation** — redesign every position:

```bash
--mask-ratio 1.0
```

**Fixed-residue inpainting** — preserve catalytic residues, disulfides, or
interface positions while redesigning the rest:

```bash
--fixed-positions "C22,C96,W47"
```

**Soft residue priors** — bias positions toward a residue class without fixing
them to one identity:

```bash
--soft-priors "34:polar, 57:metal_binding"
```

Positions are 1-indexed. Run `python training/inpainting.py --list-residue-classes`
for the available classes. `--fixed-positions` and `--forced-positions` cannot be
combined with `--mask-ratio`; they already imply the mask.

Sampling is stochastic, so the same command twice gives two different sequences --
that is how you get a panel from one backbone. Pass `--seed` to reproduce a
specific design, and record it alongside anything you order.

## Sampling settings

Architecture, graph featurization, and the flow's alpha grid are stored inside
the checkpoint and read back automatically. Three things are **not** stored and
take command-line defaults, which matters if you copy a command from elsewhere:

| Setting | `inpainting.py` | `sample.py` | Default |
|---|---|---|---|
| Flow horizon | `--t_max` | `--T` | 8.0 |
| Step count | `--steps` | `--steps` | 20 |
| Sampling temperature | `--flow_temp` | `--flow_temp` | 1.0 |

Note the horizon flag is spelled differently by the two entry points. This
checkpoint was trained at `t_max = 8`; sampling at a longer horizon works and
the c_factor table is rebuilt to cover it, but it is extrapolation beyond the
trained range.

## Training data

**CATH 4.2**, stock Ingraham splits, never re-split: **18,024 train / 608
validation / 1,120 test** chains.

The CATH artifacts are published in this repository under `datasets/cath-4.2/`,
and `./datasets/download_data.sh` in the code repository fetches and checksums
them:

| File | SHA256 |
|---|---|
| `chain_set_splits.json` | `a2d47e11a60eb93e17dd43f5b99754539114d2c6f9761e8f9ea57b141331a155` |
| `chain_set_map_with_b_factors_dssp.pkl` | `c97cd7b211076aed80b5d554fd31c28973415f7a86682ce55ddbcca23b1be7b1` |
| `val_entries_with_less_than_300_continuous_coords.pkl` | `94b0e21101088f78e7ab96b7571247aab527bdfbd3082541a09831e12d8d7e5c` |

Coordinates and splits derive from the CATH 4.2 dataset of Ingraham et al.
(2019); the per-residue B-factor and DSSP annotations were added here. The
B-factors drive the uncertainty features; the DSSP annotations are inert for
this checkpoint, which was trained with the DSSP auxiliary head disabled, so no
DSSP program is needed to use the data.

**AlphaFold DB: 2,824,736 structures.** Built from the Barrio-Hernandez et al.
AFDB cluster release (2,302,907 clusters with at least two members); no
clustering was run here, so there is no clustering threshold of ours to quote.

```
2,302,907  AFDB clusters (>= 2 members)
             pLDDT gate: representative > 70 AND cluster average > 70
             leakage removal (below)
1,211,316  usable clusters
             expanded to members
3,199,383  candidate structures
             per-structure gate: mean pLDDT >= 70, length <= 700 residues
2,824,736  structures used for training
```

**Three separate pLDDT-70 gates**, easy to conflate and worth stating apart:
at the cluster level (representative and average pLDDT > 70), at the structure
level (mean pLDDT >= 70), and again at load time per residue (residues below
pLDDT 70 are chopped out; resulting contiguous segments shorter than 100
residues are discarded). The residues the model actually sees are therefore a
subset of the structures counted above.

### Leakage control

Held-out CATH chains were mapped to AFDB cluster representatives through SIFTS,
and every cluster within **Foldseek E-value < 0.1** of a held-out representative
was removed **in whole** — members included, not just the matching
representative. That excluded 8,190 representatives, removing 4,355 of the
1,215,671 pLDDT-passing clusters (0.358%).

Three things are worth knowing about that cutoff, because they are what makes it
checkable rather than merely stated:

- It is **10x broader than the E < 0.01** used to build the AFDB clusters in the
  first place, and it removes whole clusters rather than single representatives.
- It is **not load-bearing**. Sweeping from "remove nothing detectable" to
  "remove any hit at all (E <= 10)" moves the amount removed only from 0.002%
  to 0.86% of training clusters. The corpus does not depend on a narrow band of
  benchmark-adjacent structures.
- **What survives is normal for this benchmark.** After filtering, no held-out
  CATH chain has a retained training cluster within E < 0.1, and 97.3% have no
  structurally detectable neighbour at all. Of the 43 residual neighbours, 13
  sit at TM >= 0.5 — but CATH 4.2 itself places **51% of held-out chains within
  TM >= 0.5 of a CATH *training* chain** (median 0.504, max 0.979). For 32 of
  the 39 at-risk chains, CATH's own training split already contains something at
  least as close.

Two holes in the filter, stated plainly: removal operated at
**cluster-representative resolution** (the Foldseek table covers
representatives, not members, and is truncated at E <= 10), and 60 of the 1,728
held-out chains had no SIFTS mapping and so never entered the avoid set.

## Intended use and limitations

Intended for research in protein design: assigning sequences to generated
backbones, redesigning existing scaffolds, and exploring sequence diversity
compatible with a fold.

**It does not predict** whether a designed protein will express, fold
efficiently, remain soluble, or retain function. Structural self-consistency is
a necessary but not sufficient criterion.

Practical guidance:

- **Filter by refolding.** Fold designs with ESMFold or AlphaFold and compare to
  the input backbone. This is the most informative check available before
  ordering.
- **Test a panel, not one design.** Expect a fraction of designs to work.
- **Preserve what matters.** Fixing residues you know are functionally important
  is the most direct way to retain a specific activity.
- Trained primarily on single chains; not a general protein-complex design model.
- Position-wise likelihoods correlate best with stability-like phenotypes and
  less well with binding or activity. Not a general variant-effect predictor.

### Known limitations, specifically

- **Residual benchmark overlap is bounded but non-zero.** Five CATH test chains
  (0.45%) have an AFDB training structure closer to them than anything in the
  CATH training split. Their contribution to mean sequence recovery is bounded
  at roughly 0.3 percentage points.
- **Leakage filtering was at cluster-representative resolution**, not
  per-member, so a member of a retained cluster could in principle be closer to
  a held-out chain than its representative was.
- **Sampling is step-count sensitive when the c_factor is enabled.** The default
  20 steps over horizon T = 8 gives dt ≈ 0.42, which has not converged for that
  integration. Results move with step count in that mode; the default path
  (c_factor off) does not use it.
- **Designed sequences are not calibrated probabilities.** With the c_factor
  enabled the terminal simplex state is diffuse by construction — the Dirichlet
  marginal at K = 21, T = 8 is 0.31 — so per-residue confidences read from the
  saved state should not be treated as well-calibrated. The argmax sequence is
  unaffected.
- **No sequence-identity deduplication within retained AFDB clusters.**
  Redundancy is controlled at cluster granularity only.

## Checkpoint contents

`inverse_folddir_model.pt`, 38 MB, 9.6M parameters. Epoch 1353, selected on the
generative-validation metric. The training run is described by
`configs/released_model.json` in the code repository; the file itself records
its internal run identifier and epoch under `model_name` and `epoch`.

Distributed for inference. Optimizer state, scheduler state, and RNG state have
been stripped, and absolute training paths rewritten. Keys retained:

| Key | Purpose |
|---|---|
| `model_state_dict` | Model weights |
| `args` | Training configuration the sampler reads |
| `graph_builder_params` | Graph featurization settings |
| `model_architecture_params` | Layer sizes and architecture |
| `epoch`, `model_name`, `metrics`, `timestamp` | Provenance |

Loaded with `weights_only=False`, since `args` contains a configuration object
rather than plain tensors.

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

## Acknowledgements

Builds on [microsoft/InverseFoldDir](https://github.com/microsoft/InverseFoldDir)
(MIT, © Microsoft Corporation), which in turn incorporates
[gvp-pytorch](https://github.com/drorlab/gvp-pytorch) (Jing et al.) and
[dirichlet-flow-matching](https://github.com/HannesStark/dirichlet-flow-matching)
(Stärk et al.).
