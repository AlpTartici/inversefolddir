#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Inpainting with Position+Validation Format Example Script
# Predict masked positions with amino acid validation (perfect for variant effect studies)

# Set script options
set -e  # Exit on any error
set -u  # Exit on undefined variable

# Configuration
PDB_INPUT="${1:-1fcd.C}"                    # PDB ID or file path (default: 1abc)
MASK_POSITIONS="${2:-E42,G16}"        # Positions with validation (default: D45,Y67,K89)
OUTPUT_DIR="${3:-./results/inpainting_validated}"  # Output directory
STEPS="${4:-20}"                          # Sampling steps (default: 20)
FLOW_TEMP="${5:-0.3}"                    # Temperature (default: 0.3)

echo "============================================="
echo " Inpainting with Position Validation"
echo "============================================="
echo "PDB Input: $PDB_INPUT"
echo "Mask Positions: $MASK_POSITIONS (with amino acid validation)"
echo "Output Directory: $OUTPUT_DIR"
echo "Sampling Steps: $STEPS"
echo "Flow Temperature: $FLOW_TEMP"
echo "============================================="
echo ""
echo " Mask Format Explanation:"
echo "  E42 = Mask position 42, but first verify it has amino acid E"
echo "  G16 = Mask position 16, but first verify it has amino acid G"
echo "  Positions are 1-indexed (matches PDB numbering) and separated by commas"
echo ""
echo "  Validation Safety:"
echo "  - Program will ERROR if validation fails"
echo "  - Perfect for variant effect prediction studies"
echo "  - Ensures you're masking the expected amino acids"
echo "============================================="

# Activate conda environment (skip if you installed with pip/venv instead)
echo "Activating inv_fold conda environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate inv_fold

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run inpainting with validation
python ../training/inpainting.py \
    --pdb-id "$PDB_INPUT" \
    --mask-positions "$MASK_POSITIONS" \
    --steps "$STEPS" \
    --flow_temp "$FLOW_TEMP" \
    --output-dir "$OUTPUT_DIR" \
    --model "../ckpts/inverse_folddir_model.pt" \
    --split_json ../datasets/cath-4.2/chain_set_splits.json \
    --map_pkl ../datasets/cath-4.2/chain_set_map_with_b_factors_dssp.pkl \
    --detailed_json \
    --verbose

echo ""
echo "Validated inpainting completed!"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Output files:"
echo "  - Sequences: $OUTPUT_DIR/inpainting_validated_sequences.csv"
echo "  - Probabilities: $OUTPUT_DIR/*_probabilities.npz"
echo "  - Detailed JSON: $OUTPUT_DIR/*_detailed_predictions.json"
echo "  - Metadata: $OUTPUT_DIR/*_metadata.txt"
echo ""

# Show quick results summary if files exist
if [ -f "$OUTPUT_DIR/inpainting_validated_sequences.csv" ]; then
    echo "Inpainting Results Summary:"

    # Count successful predictions
    total_sequences=$(tail -n +2 "$OUTPUT_DIR/inpainting_validated_sequences.csv" | wc -l)
    echo "Total sequences processed: $total_sequences"

    # Show sample results
    if [ $total_sequences -gt 0 ]; then
        echo ""
        echo "Sample results:"
        head -n 4 "$OUTPUT_DIR/inpainting_validated_sequences.csv" | column -t -s ','
    fi

    echo ""
    echo "Next Steps:"
    echo "  1. Check detailed JSON for position-by-position predictions"
    echo "  2. Analyze probabilities in the NPZ file"
    echo "  3. Compare predictions with known effects"
fi
