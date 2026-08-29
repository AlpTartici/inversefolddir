# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025-2026 Alp Tartici, Stanford University.
# Licensed under the MIT License.

import os
import sys

import torch

# Import from the repository root regardless of where this is run from: under
# pytest the working directory is not necessarily the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.soft_priors import (
    parse_soft_priors, apply_soft_priors, validate_prior_conflicts,
    describe_prior, SoftPriorError, list_residue_classes, _CANONICAL_AAS,
)

fails = []
def check(name, cond, extra=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))
    if not cond: fails.append(name)

# 1. Named class: mass concentrated on class members, remainder spread
p = parse_soft_priors(spec_string="34:polar")
v = p[33]
polar_idx = [_CANONICAL_AAS.index(a) for a in "STNQ"]
polar_mass = sum(v[i].item() for i in polar_idx)
check("named class 'polar' sums to 1", abs(v.sum().item()-1.0) < 1e-5)
check("polar mass == 1.0 (full class spec)", abs(polar_mass-1.0) < 1e-5, f"got {polar_mass:.4f}")
check("unknown class idx20 == 0", v[20].item() == 0.0)

# 2. PARTIAL spec: the key usability requirement
p = parse_soft_priors(spec_string="57:H0.4")
v = p[56]
h = v[_CANONICAL_AAS.index('H')].item()
others = [v[i].item() for i in range(20) if i != _CANONICAL_AAS.index('H')]
check("partial 'H0.4' -> H==0.4", abs(h-0.4) < 1e-5, f"got {h:.4f}")
check("remainder spread evenly over other 19", abs(others[0]-0.6/19) < 1e-5 and max(others)-min(others) < 1e-6,
      f"each={others[0]:.4f} expected={0.6/19:.4f}")
check("partial sums to 1", abs(v.sum().item()-1.0) < 1e-5)

# 3. Explicit multi-AA weights summing to 1
p = parse_soft_priors(spec_string="88:H0.5|E0.3|D0.2")
v = p[87]
check("explicit H0.5", abs(v[_CANONICAL_AAS.index('H')].item()-0.5) < 1e-5)
check("explicit E0.3", abs(v[_CANONICAL_AAS.index('E')].item()-0.3) < 1e-5)
check("explicit D0.2", abs(v[_CANONICAL_AAS.index('D')].item()-0.2) < 1e-5)
check("explicit others==0", all(v[i].item() < 1e-6 for i in range(20)
      if _CANONICAL_AAS[i] not in "HED"))

# 4. Multi-position inline, mixed forms (comma-splitting must not break H0.5|E0.5)
p = parse_soft_priors(spec_string="34:polar, 57:metal_binding, 88:H0.5|E0.5")
check("multi-position parse -> 3 priors", len(p)==3, f"got {len(p)}")
check("pos88 split correctly", abs(p[87][_CANONICAL_AAS.index('H')].item()-0.5)<1e-5)

# 5. 3-letter codes + dict form
p = parse_soft_priors(spec_string="10:HIS0.4|CYS0.2")
check("3-letter codes parse", abs(p[9][_CANONICAL_AAS.index('H')].item()-0.4)<1e-5)

# 6. Over-1.0 normalization
p = parse_soft_priors(spec_string="5:H0.8|E0.8")
v = p[4]
check("over-1.0 normalized", abs(v.sum().item()-1.0)<1e-5 and abs(v[_CANONICAL_AAS.index('H')].item()-0.5)<1e-5)

# 7. Errors
for bad, label in [("34:notaclass","unknown class"), ("34:Z0.5","bad AA"),
                   ("0:polar","zero position"), ("abc:polar","non-int position"),
                   ("34polar","missing colon")]:
    try:
        parse_soft_priors(spec_string=bad); check(f"rejects {label}", False)
    except SoftPriorError: check(f"rejects {label}", True)

# 8. Bounds check
try:
    parse_soft_priors(spec_string="500:polar", sequence_length=100); check("rejects OOB position", False)
except SoftPriorError: check("rejects OOB position", True)

# 9. Conflict detection
try:
    validate_prior_conflicts({33: torch.ones(21)/21}, fixed_positions=[33]); check("detects fixed/prior conflict", False)
except SoftPriorError: check("detects fixed/prior conflict", True)
validate_prior_conflicts({33: torch.ones(21)/21}, fixed_positions=[10]); check("no false conflict", True)

# 10. apply_soft_priors: does it actually bias the state?
torch.manual_seed(0)
N, K = 100, 21
x = torch.distributions.Dirichlet(20.0*torch.ones(K)).sample((1,N))
before = x[0,56].argmax().item()
priors = parse_soft_priors(spec_string="57:H0.9")
mask = torch.ones(N, dtype=torch.bool)
x = apply_soft_priors(x, priors, prior_strength=5.0, dirichlet_concentration=20.0, inpainting_mask=mask)
after_h = x[0,56][_CANONICAL_AAS.index('H')].item()
check("apply: prior position biased toward H", after_h > 0.5, f"H prob={after_h:.3f}")
check("apply: shape preserved", x.shape==(1,N,K))
check("apply: simplex valid", abs(x[0,56].sum().item()-1.0)<1e-4)
check("apply: other positions untouched", abs(x[0,10].sum().item()-1.0)<1e-4)

# 11. Skips non-sampled positions
mask2 = torch.ones(N, dtype=torch.bool); mask2[56]=False
x2 = torch.distributions.Dirichlet(20.0*torch.ones(K)).sample((1,N))
orig = x2[0,56].clone()
x2 = apply_soft_priors(x2, priors, inpainting_mask=mask2, verbose=False)
check("skips non-sampled position", torch.allclose(orig, x2[0,56]))

print("\n" + "="*60)
print(f"{'ALL PASSED' if not fails else 'FAILURES: '+', '.join(fails)}")


def test_soft_priors():
    """Expose the checks above to pytest, so a failure fails the suite."""
    assert not fails, "failed checks: " + ", ".join(fails)


if __name__ == "__main__":
    # Exit non-zero on failure. Printing "FAILURES" and returning 0 makes this
    # invisible to CI and to anyone running it from a script.
    sys.exit(1 if fails else 0)
