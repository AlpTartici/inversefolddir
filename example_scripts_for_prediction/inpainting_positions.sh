#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Inpainting with Position-Only Format Example Script
# Predict masked positions without validation (faster, more flexible)

# Set script options
set -e  # Exit on any error
set -u  # Exit on undefined variable

# Configuration
PDB_INPUT="${1:-1fcd.C}"                 # PDB ID or file path (default: 1fcd.C)
MASK_POSITIONS="${2:-16,42}"             # Positions to mask (default: 16,42)
OUTPUT_DIR="${3:-./results/inpainting_positions}"  # Output directory
STEPS="${4:-20}"                         # Sampling steps (default: 20)
FLOW_TEMP="${5:-0.3}"                   # Temperature (default: 0.3)
ENSEMBLE_SIZE="${6:-2}"                  # Ensemble size (default: 2)

echo "============================================="
echo "Position-Only Inpainting"
echo "============================================="
echo "PDB Input: $PDB_INPUT"
echo "Mask Positions: $MASK_POSITIONS (no validation)"
echo "Output Directory: $OUTPUT_DIR"
echo "Sampling Steps: $STEPS"
echo "Flow Temperature: $FLOW_TEMP"
echo "Ensemble Size: $ENSEMBLE_SIZE"
echo "============================================="
echo ""
echo "Mask Format Explanation:"
echo "  Positions are 1-indexed (matches PDB numbering)"
echo "  16 = Mask the 16th residue"
echo "  42 = Mask the 42nd residue"
echo ""
echo "Benefits of Position-Only Format:"
echo "  - No amino acid validation required"
echo "  - Faster execution (no structure checking)"
echo "  - Good for exploratory design"
echo "  - Works with any position numbers"
echo "============================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run position-only inpainting with ensemble
python ../training/inpainting.py \
    --pdb-id "$PDB_INPUT" \
    --mask-positions "$MASK_POSITIONS" \
    --model "../ckpts/inverse_folddir_model.pt" \
    --split_json ../datasets/cath-4.2/chain_set_splits.json \
    --map_pkl ../datasets/cath-4.2/chain_set_map_with_b_factors_dssp.pkl \
    --steps "$STEPS" \
    --flow_temp "$FLOW_TEMP" \
    --ensemble_size "$ENSEMBLE_SIZE" \
    --ensemble_consensus_strength 0.4 \
    --ensemble_method arithmetic \
    --output-dir "$OUTPUT_DIR" \
    --detailed_json \
    --verbose

echo ""
echo "Position-based inpainting completed!"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Output files:"
echo "  - Sequences: $OUTPUT_DIR/inpainting_positions_sequences.csv"
echo "  - Probabilities: $OUTPUT_DIR/*_probabilities.npz"
echo "  - Detailed JSON: $OUTPUT_DIR/*_detailed_predictions.json"
echo "  - Metadata: $OUTPUT_DIR/*_metadata.txt"
echo ""

# Show quick results summary if files exist
if [ -f "$OUTPUT_DIR/inpainting_positions_sequences.csv" ]; then
    echo "Inpainting Results Summary:"

    # Count successful predictions
    total_sequences=$(tail -n +2 "$OUTPUT_DIR/inpainting_positions_sequences.csv" | wc -l)
    echo "Total sequences processed: $total_sequences"

    # Show sample results
    if [ $total_sequences -gt 0 ]; then
        echo ""
        echo "Sample results:"
        head -n 4 "$OUTPUT_DIR/inpainting_positions_sequences.csv" | column -t -s ','
    fi

    echo ""
    echo "Ensemble Information:"
    echo "  - Used $ENSEMBLE_SIZE structural variants"
    echo "  - Consensus strength: 0.4 (balanced diversity/agreement)"
    echo "  - Method: arithmetic averaging"

    echo ""
    echo "Analysis Tips:"
    echo "  1. Compare predictions across ensemble members"
    echo "  2. Look at confidence scores in detailed JSON"
    echo "  3. Check probability distributions for uncertainty"
fi
