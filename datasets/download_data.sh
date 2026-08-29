#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025-2026 Alp Tartici, Stanford University.
# Licensed under the MIT License.
#
# Download the CATH 4.2 reference artifacts.
#
# They are served from the same Hugging Face repository as the released
# weights, and every file is checked against a SHA256 recorded below. No
# credentials are needed; the repository is public.
#
#   ./datasets/download_data.sh            # fetch whatever is missing
#   ./datasets/download_data.sh --force    # re-fetch even if present
#   ./datasets/download_data.sh --verify   # check what is there, download nothing
#
# By default the files land in `datasets/cath-4.2/` next to this script. Set
# IFD_DATA_DIR to put them elsewhere; the rest of the codebase honours the same
# variable.
#
# What each file is for:
#
#   chain_set_splits.json                   the CATH 4.2 train/validation/test
#                                           split. Small, and already tracked in
#                                           git -- listed here so its integrity
#                                           can be checked too.
#   chain_set_map_with_b_factors_dssp.pkl   backbone coordinates, sequences,
#                                           per-residue B-factors and DSSP
#                                           annotations. Needed to train, to run
#                                           the CATH recovery benchmark, and for
#                                           --pdb-id / --uniprot lookup. Not
#                                           needed to design from your own
#                                           structure file.
#   val_entries_...continuous_coords.pkl    the 608-chain generative validation
#                                           set used by
#                                           --validation_with_full_generation.
#
# The B-factors are load-bearing: they drive the uncertainty features and the
# uncertainty-scaled structure noise. The DSSP annotations are not -- the
# released checkpoint was trained with the DSSP auxiliary head disabled
# (--lambda_dssp_loss 0), and the loader treats the field as optional. They ship
# inside the same pickle, so no DSSP program needs to be installed.
#
# Coordinates and splits derive from the CATH 4.2 dataset of Ingraham et al.
# (2019); the B-factor and DSSP annotations were added here.

set -euo pipefail

REPO_ID="AlpTartici/inversefolddir"
BASE_URL="https://huggingface.co/${REPO_ID}/resolve/main/datasets/cath-4.2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${IFD_DATA_DIR:-$SCRIPT_DIR}"
DEST="${DATA_DIR%/}/cath-4.2"

# filename:sha256
FILES=(
  "chain_set_splits.json:a2d47e11a60eb93e17dd43f5b99754539114d2c6f9761e8f9ea57b141331a155"
  "chain_set_map_with_b_factors_dssp.pkl:c97cd7b211076aed80b5d554fd31c28973415f7a86682ce55ddbcca23b1be7b1"
  "val_entries_with_less_than_300_continuous_coords.pkl:94b0e21101088f78e7ab96b7571247aab527bdfbd3082541a09831e12d8d7e5c"
)

FORCE=0
VERIFY_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --force)  FORCE=1 ;;
    --verify) VERIFY_ONLY=1 ;;
    -h|--help)
      sed -n '6,18p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *)
      echo "Unknown option: $arg (try --help)" >&2
      exit 2 ;;
  esac
done

# sha256sum on Linux, shasum on macOS.
if command -v sha256sum >/dev/null 2>&1; then
  sha256_of() { sha256sum "$1" | cut -d' ' -f1; }
elif command -v shasum >/dev/null 2>&1; then
  sha256_of() { shasum -a 256 "$1" | cut -d' ' -f1; }
else
  echo "Need sha256sum or shasum to verify downloads." >&2
  exit 1
fi

if command -v curl >/dev/null 2>&1; then
  fetch() { curl -fL --retry 3 --retry-delay 5 -o "$1" "$2"; }
elif command -v wget >/dev/null 2>&1; then
  fetch() { wget -q --tries=3 -O "$1" "$2"; }
else
  echo "Need curl or wget to download." >&2
  exit 1
fi

mkdir -p "$DEST"
echo "Destination: $DEST"
echo

failed=0
for entry in "${FILES[@]}"; do
  name="${entry%%:*}"
  want="${entry##*:}"
  path="$DEST/$name"

  if [ -f "$path" ] && [ "$FORCE" -eq 0 ]; then
    got="$(sha256_of "$path")"
    if [ "$got" = "$want" ]; then
      echo "ok        $name (already present, checksum matches)"
      continue
    fi
    if [ "$VERIFY_ONLY" -eq 1 ]; then
      echo "MISMATCH  $name"
      echo "          expected $want"
      echo "          found    $got"
      failed=1
      continue
    fi
    echo "stale     $name (checksum differs, re-downloading)"
  elif [ "$VERIFY_ONLY" -eq 1 ]; then
    echo "MISSING   $name"
    failed=1
    continue
  fi

  echo "download  $name"
  if ! fetch "$path.part" "$BASE_URL/$name"; then
    echo "          download failed" >&2
    rm -f "$path.part"
    failed=1
    continue
  fi

  got="$(sha256_of "$path.part")"
  if [ "$got" != "$want" ]; then
    echo "          CHECKSUM MISMATCH -- discarding" >&2
    echo "          expected $want" >&2
    echo "          found    $got" >&2
    rm -f "$path.part"
    failed=1
    continue
  fi
  mv "$path.part" "$path"
  echo "ok        $name (checksum verified)"
done

echo
if [ "$failed" -ne 0 ]; then
  echo "One or more files are missing or corrupt."
  exit 1
fi
echo "All CATH reference files are present and verified."
