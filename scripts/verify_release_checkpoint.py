# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025-2026 Alp Tartici, Stanford University.
# Licensed under the MIT License.

"""
Prove a release checkpoint is functionally identical to the training
checkpoint it was derived from.

    python scripts/verify_release_checkpoint.py ORIGINAL.pt RELEASE.pt

Checks that every weight tensor is bit-identical, that the architecture and
featurization config are unchanged, that only the intended path arguments
were rewritten, and that no absolute paths survive.
"""
import sys
import torch, os

if len(sys.argv) != 3:
    print(__doc__)
    sys.exit(1)
ORIG, REL = os.path.expandvars(sys.argv[1]), os.path.expandvars(sys.argv[2])

o = torch.load(ORIG, map_location="cpu", weights_only=False)
r = torch.load(REL,  map_location="cpu", weights_only=False)

fails = []
def check(name, ok, extra=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))
    if not ok: fails.append(name)

# 1. Every weight tensor bit-identical
so, sr = o["model_state_dict"], r["model_state_dict"]
check("same tensor names", set(so) == set(sr), f"{len(so)} vs {len(sr)}")
mismatch = [k for k in so if not torch.equal(so[k], sr[k])] if set(so)==set(sr) else ["<keyset differs>"]
check("all weights bit-identical", not mismatch,
      f"{len(so)} tensors, {sum(t.numel() for t in so.values())/1e6:.1f}M params")
if mismatch: print("   first mismatches:", mismatch[:5])

# 2. Architecture / featurization config preserved exactly
for k in ["graph_builder_params", "model_architecture_params"]:
    check(f"{k} identical", o.get(k) == r.get(k))

# 3. args: only the intended path keys differ
ao = vars(o["args"]) if hasattr(o["args"], "__dict__") else dict(o["args"])
ar = vars(r["args"]) if hasattr(r["args"], "__dict__") else dict(r["args"])
check("args keys unchanged", set(ao) == set(ar), f"{len(ao)} keys")
diff = {k for k in ao if ao.get(k) != ar.get(k)}
expected = {"af2_chunk_dir", "val_gen_pkl", "output_dir"}
check("only intended paths rewritten", diff == expected, f"differing: {sorted(diff)}")

# 4. No absolute paths remain
import re
leftover = {k: v for k, v in ar.items()
            if isinstance(v, str) and re.match(r"^(/oak/|/scratch/|/home/|/tmp/)", v)}
check("no absolute paths remain", not leftover, str(leftover) if leftover else "")

# 5. Training-only state gone
check("optimizer/scheduler/rng dropped",
      not any(k in r for k in ["optimizer_state_dict","scheduler_state_dict","rng_states","training_state"]))

# 6. Provenance kept
check("provenance kept", all(k in r for k in ["epoch","model_name","metrics"]),
      f"epoch={r.get('epoch')}")

print("\n" + "="*58)
print("VERIFIED IDENTICAL" if not fails else "FAILURES: " + ", ".join(fails))
sys.exit(1 if fails else 0)
