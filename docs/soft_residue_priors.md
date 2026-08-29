# Soft Residue Priors

Soft priors let you say *"I'd like this position to be polar"* without saying
*"this position must be serine."* They sit between the two hard controls:

| Control | What it does | Position can change during generation? |
|---|---|---|
| `--fixed-positions` | Keeps the native residue | No — clamped the whole time |
| `--forced-positions` | Sets a residue you choose | No — clamped the whole time |
| **`--soft-priors`** | **Biases toward a residue class or distribution** | **Yes — free to change** |

Use a soft prior when you know the *chemistry* you want but not the exact amino
acid: polar residues on an exposed loop, a candidate metal site that could be
His/Cys/Asp/Glu, or a basic patch on a nucleic-acid-binding surface.

---

## Quick start

```bash
# Bias position 34 toward polar residues
python training/inpainting.py --pdb_input 3OGO --model ckpts/inverse_folddir_model.pt \
    --soft-priors "34:polar"
```

See every available class name:

```bash
python training/inpainting.py --list-residue-classes
```

---

## Three ways to specify a prior

### 1. Named residue class — easiest

```bash
--soft-priors "34:polar, 57:metal_binding, 88:aromatic"
```

Positions are **1-indexed**, matching `--fixed-positions`. Available classes:

| Class | Amino acids | Typical use |
|---|---|---|
| `polar` | S, T, N, Q | Solvent-exposed surfaces |
| `hydrophobic` | A, V, I, L, M, F, W, C | Buried core |
| `aromatic` | F, W, Y, H | Stacking, interface hot spots |
| `charged` | D, E, K, R | Salt bridges, solubility |
| `positive` / `basic` | K, R (+H) | Nucleic-acid contacts |
| `negative` / `acidic` | D, E | Acidic patches |
| `metal_binding` | H, C, D, E | Candidate metal sites |
| `small` / `tiny` | A, G, S / A, G | Tight packing |
| `helix_former` | A, E, L, M, Q, K | Helical segments |
| `sheet_former` | V, I, Y, F, W, T | Strands |
| `turn_former` | G, N, P, S, D | Turns and loops |
| `flexible` / `rigid` | G, S / P | Backbone flexibility |

### 2. Explicit weights — full control

```bash
--soft-priors "88:H0.5|E0.3|D0.2"
```

Amino acids are 1-letter (`H`) or 3-letter (`HIS`), separated by `|`.

### 3. JSON file — many positions

```bash
--soft-priors-json priors.json
```

```json
{
  "34": "polar",
  "57": {"HIS": 0.4, "CYS": 0.2, "ASP": 0.2, "GLU": 0.2},
  "88": "H0.5|E0.3|D0.2"
}
```

All three forms mix freely, and `--soft-priors` can be combined with
`--soft-priors-json` (inline entries win on conflict).

---

## The one rule worth understanding: leftover mass

**If your weights sum to less than 1.0, the rest is spread evenly over the
amino acids you did not mention.** This is what makes a prior a *nudge* rather
than a restriction.

```
--soft-priors "57:H0.4"
```

means: **40%** histidine, and the remaining **60%** spread evenly across the
other 19 residues (~3.2% each). The model can still choose anything.

```
--soft-priors "57:H0.5|E0.3|D0.2"      # sums to 1.0
```

means: all mass on H/E/D. Everything else starts at zero — a strong bias,
though the position can still drift during denoising.

| Weights sum to | Behavior |
|---|---|
| < 1.0 | Gentle nudge; remainder spread over unnamed residues |
| = 1.0 | Strong bias; only the named residues get initial mass |
| > 1.0 | Normalized down to 1.0, with a warning |

Named classes always sum to 1.0 across their members, so `34:polar` puts all
initial mass on S/T/N/Q.

---

## Controlling prior strength

`--prior-strength` (default `1.0`) scales how tightly the starting state
concentrates on your distribution:

```bash
--soft-priors "34:polar" --prior-strength 5.0
```

| Value | Effect |
|---|---|
| `0.5` | Loose — lots of initial noise, prior barely felt |
| `1.0` | Default — same concentration as normal sampling |
| `5.0` | Firm — starts close to your stated distribution |
| `20.0` | Very tight — near-deterministic start (still free to move) |

A soft prior shapes **where the trajectory starts**, not where it ends.

**Raising the strength does not guarantee the prior wins.** Where the backbone
geometry strongly determines a residue, the structural signal overrides the
prior no matter how tightly you initialize. Measured on 3OGO with 20 denoising
steps:

| Position | Prior | strength 1 | strength 20 | strength 100 |
|---|---|---|---|---|
| 60 | `H1.0` | H | H | H |
| 50 | `metal_binding` | G | G | G |

Position 60 accepts the prior at any strength. Position 50 sits in a tight turn
where glycine is geometrically required, and rejects it at every strength.

This is the intended behavior: a soft prior is a *preference*, and the model is
free to overrule it when the structure says otherwise. If a position refuses
your prior, that is informative — the backbone likely does not tolerate the
chemistry you asked for. Use `--fixed-positions` or `--forced-positions` when
you need a guarantee rather than a preference.

---

## Combining with fixed positions

Soft priors compose with `--fixed-positions`:

```bash
python training/inpainting.py --pdb_input 3OGO --model ckpts/inverse_folddir_model.pt \
    --fixed-positions "C22,C96" \
    --soft-priors "34:polar, 57:metal_binding" \
    --prior-strength 5.0
```

Keeps the disulfide cysteines exactly, nudges 34 polar and 57 toward metal
coordination, redesigns everything else.

**A position cannot be both fixed and soft-prior.** That request is
contradictory — fixed positions never change, so the prior could not apply —
and the run stops with an error naming the offending positions rather than
silently ignoring them.

---

## Checking that it worked

Run with `--verbose` to see each parsed prior and its entropy:

```
Parsed 3 soft residue prior(s):
  Position 30: S=0.25, T=0.25, N=0.25, Q=0.25 (entropy 1.39 nats)
  Position 50: H=0.25, C=0.25, D=0.25, E=0.25 (entropy 1.39 nats)
  Position 60: H=0.50, E=0.30, D=0.20 (entropy 1.03 nats)
  [soft-prior] Applied 3 prior(s) at concentration 100.00 (strength 5.0 x base 20.0)
```

Lower entropy means a more committed prior. A uniform distribution over all 20
residues is ~3.00 nats; a single residue is 0.00.

If a prior position is not being sampled (because it is fixed, or outside the
target chain), it is reported as skipped rather than applied.

---

## Limitations

- Priors apply to **initialization**, not to every denoising step. They bias
  the start of the trajectory; they do not constrain the final sequence. For a
  hard guarantee use `--fixed-positions` or `--forced-positions`.
- Not wired into the batched ProteinGym variant-scoring path
  (`sample_chain_inpainting_batched`), which is used for likelihood scoring
  rather than design.
- With `--context-chains`, prior positions are interpreted as 1-indexed within
  the **target chain**, consistent with `--fixed-positions`.
