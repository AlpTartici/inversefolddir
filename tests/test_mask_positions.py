#!/usr/bin/env python3
# Copyright (c) 2025-2026 Alp Tartici, Stanford University.
# Licensed under the MIT License.

"""Index-base checks for ``--mask-positions``.

The parser takes 1-indexed positions and returns 0-indexed ones. Getting this
wrong is silent: nothing raises, the sampler happily designs the neighbouring
residue, and the mistake only shows up in the wet lab. These checks pin the
convention down so it cannot drift.
"""

import os
import sys

# Import from the repository root regardless of where this is run from: under
# pytest the working directory is not necessarily the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.inpainting import parse_and_validate_mask_positions

# A sequence whose residues are all distinct, so an off-by-one cannot pass by
# landing on an identical neighbour.
SEQ = "MADEQKFHIL"

fails = []


def check(name, cond, extra=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))
    if not cond:
        fails.append(name)


# 1. THE regression check: position 1 is the first residue, index 0.
check("position-only '1' -> index 0",
      parse_and_validate_mask_positions("1", SEQ) == [0],
      f"got {parse_and_validate_mask_positions('1', SEQ)}")

# 2. The validated format uses the same base.
check("validated 'M1' -> index 0",
      parse_and_validate_mask_positions("M1", SEQ) == [0],
      f"got {parse_and_validate_mask_positions('M1', SEQ)}")

# 3. Both formats agree on every position, not just the first.
check("position-only '1,5,10' -> [0,4,9]",
      parse_and_validate_mask_positions("1,5,10", SEQ) == [0, 4, 9])
check("validated 'M1,Q5,L10' -> [0,4,9]",
      parse_and_validate_mask_positions("M1,Q5,L10", SEQ) == [0, 4, 9])

# 4. The last residue is reachable, and one past it is not.
check("last position is in range",
      parse_and_validate_mask_positions(str(len(SEQ)), SEQ) == [len(SEQ) - 1])
try:
    parse_and_validate_mask_positions(str(len(SEQ) + 1), SEQ)
    check("position len+1 is rejected", False, "no error raised")
except ValueError:
    check("position len+1 is rejected", True)

# 5. Position 0 is not a valid 1-indexed position and must be refused rather
#    than wrapping to the end of the sequence.
try:
    parse_and_validate_mask_positions("0", SEQ)
    check("position 0 is rejected", False, "no error raised")
except ValueError:
    check("position 0 is rejected", True)

# 6. Validation reads the residue at the 1-indexed position. 'A2' is correct
#    here; 'A1' names the residue at index 0, which is M, and must fail.
check("validated 'A2' accepted (SEQ[1] == 'A')",
      parse_and_validate_mask_positions("A2", SEQ) == [1])
try:
    parse_and_validate_mask_positions("A1", SEQ)
    check("validated 'A1' rejected (SEQ[0] == 'M')", False, "no error raised")
except ValueError:
    check("validated 'A1' rejected (SEQ[0] == 'M')", True)

# 7. Whitespace and ordering are handled without shifting the index base.
check("whitespace tolerated", parse_and_validate_mask_positions(" 1 , 3 ", SEQ) == [0, 2])

print("\n" + "=" * 60)
print(f"{'ALL PASSED' if not fails else 'FAILURES: ' + ', '.join(fails)}")


def test_mask_positions_are_one_indexed():
    """Expose the checks above to pytest, so a failure fails the suite."""
    assert not fails, "failed checks: " + ", ".join(fails)


if __name__ == "__main__":
    # Exit non-zero on failure. Printing "FAILURES" and returning 0 makes this
    # invisible to CI and to anyone running it from a script.
    sys.exit(1 if fails else 0)
