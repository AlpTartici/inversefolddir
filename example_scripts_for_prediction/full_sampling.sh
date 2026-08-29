#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Full Sequence Sampling Example Script
# Generate complete amino acid sequences conditioned on protein backbone structure

# Set script options
set -e  # Exit on any error
set -u  # Exit on undefined variable

# Configuration
PDB_INPUT="${1:-1fcd.C}"           # PDB ID or file path (default: 1fcd.C)
OUTPUT_DIR="${2:-./results/full_sampling}"  # Output directory
STEPS="${3:-20}"                   # Sampling steps (default: 20)
FLOW_TEMP="${4:-0.1}"             # Temperature (default: 0.1)
ENSEMBLE_SIZE="${5:-3}"            # Ensemble size (default: 3)

echo "============================================="
echo "Full Sequence Sampling"
echo "============================================="
echo "PDB Input: $PDB_INPUT"
echo "Output Directory: $OUTPUT_DIR"
echo "Sampling Steps: $STEPS"
echo "Flow Temperature: $FLOW_TEMP"
echo "Ensemble Size: $ENSEMBLE_SIZE"
echo "============================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run full sequence sampling
python ../training/sample.py \
    --pdb_input "$PDB_INPUT" \
    --steps "$STEPS" \
    --flow_temp "$FLOW_TEMP" \
    --model_path "../ckpts/inverse_folddir_model.pt" \
    --ensemble_size "$ENSEMBLE_SIZE" \
    --ensemble_consensus_strength 0.3 \
    --output_dir "$OUTPUT_DIR" \
    --output_prefix "full_sampling" \
    --save_probabilities \
    --detailed_json

echo ""
echo "Full sequence sampling completed!"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Output files:"
echo "  - Sequences: $OUTPUT_DIR/full_sampling_sequences.csv"
echo "  - Probabilities: $OUTPUT_DIR/*_probabilities.npz"
echo "  - Metadata: $OUTPUT_DIR/*_metadata.txt"
echo ""

# Show quick results summary if CSV exists
if [ -f "$OUTPUT_DIR/full_sampling_sequences.csv" ]; then
    echo "Quick Results Summary:"
    echo "Total sequences generated: $(tail -n +2 "$OUTPUT_DIR/full_sampling_sequences.csv" | wc -l)"
    echo ""
    echo "Sample sequences (first 3):"
    head -n 4 "$OUTPUT_DIR/full_sampling_sequences.csv" | column -t -s ','
fi
