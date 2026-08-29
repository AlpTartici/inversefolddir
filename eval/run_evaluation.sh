#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Test script for the evaluation pipeline
# This script demonstrates how to use the evaluation pipeline

set -e  # Exit on any error

# Configuration
CSV_PATH="../output/prediction/batch_validation_sampling_virtual_sequences_20250709_233849.csv"
REFERENCE_DIR="../datasets/esmfold_predictions/esmfold_predictions_on_ref_valid"
OUTPUT_DIR="../output/evaluation/test_run_$(date +%Y%m%d_%H%M%S)"

# Default to verbose mode, can be overridden
VERBOSE=${1:-"--verbose"}

echo "=========================================="
echo "EVALUATION PIPELINE TEST"
echo "=========================================="
echo "CSV file: $CSV_PATH"
echo "Reference directory: $REFERENCE_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Verbose mode: $VERBOSE"
echo ""

# Activate the esmfold environment
echo "Activating esmfold environment..."
source activate esmfold

# Check if required files exist
if [ ! -f "$CSV_PATH" ]; then
    echo "Error: CSV file not found: $CSV_PATH"
    exit 1
fi

if [ ! -d "$REFERENCE_DIR" ]; then
    echo "Error: Reference directory not found: $REFERENCE_DIR"
    exit 1
fi

# Run the evaluation pipeline
echo "Starting evaluation pipeline..."
if [ "$VERBOSE" = "--verbose" ]; then
    echo "Running in VERBOSE mode - detailed progress will be shown"
    python evaluation_pipeline.py \
        --csv_path "$CSV_PATH" \
        --reference_dir "$REFERENCE_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --verbose
else
    echo "Running in normal mode"
    python evaluation_pipeline.py \
        --csv_path "$CSV_PATH" \
        --reference_dir "$REFERENCE_DIR" \
        --output-dir "$OUTPUT_DIR"
fi

echo ""
echo "=========================================="
echo "EVALUATION COMPLETED"
echo "=========================================="
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Key files:"
echo "  - structure_comparison_results.csv  (detailed results)"
echo "  - evaluation_summary.txt           (summary report)"
echo "  - prediction_log.txt               (prediction log)"
echo "  - predicted_structures/             (predicted PDB files)"

# Show summary if available
SUMMARY_FILE="$OUTPUT_DIR/results/evaluation_summary.txt"
if [ -f "$SUMMARY_FILE" ]; then
    echo ""
    echo "Quick summary:"
    echo "---------------"
    grep -E "(Average|TM-score >)" "$SUMMARY_FILE" | head -10
fi
