#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Batch Processing from CSV Example Script
# Process multiple mutations from a CSV file with custom batch size

# Set script options
set -e  # Exit on any error
set -u  # Exit on undefined variable

# Configuration
CSV_FILE="${1:-mutations.csv}"           # CSV file with mutations (default: mutations.csv)
OUTPUT_DIR="${2:-./results/batch_processing}"  # Output directory
STEPS="${3:-20}"                         # Sampling steps (default: 20)
FLOW_TEMP="${4:-0.3}"                   # Temperature (default: 0.3)
BATCH_SIZE="${5:-8}"                     # Batch size (default: 8)

echo "============================================="
echo "Batch Processing from CSV"
echo "============================================="
echo "CSV File: $CSV_FILE"
echo "Output Directory: $OUTPUT_DIR"
echo "Sampling Steps: $STEPS"
echo "Flow Temperature: $FLOW_TEMP"
echo "Batch Size: $BATCH_SIZE"
echo "============================================="

# Check if CSV file exists
if [ ! -f "$CSV_FILE" ]; then
    echo ""
    echo "Error: CSV file '$CSV_FILE' not found!"
    echo ""
    echo "Creating example CSV file..."

    # Create example CSV file
    cat > "$CSV_FILE" << 'EOF'
mutant,mutated_sequence,DMS_score,DMS_score_bin,mask-positions,protein
A1V,MVQPQVQHPIQSIKAFN,-2.1,low,1,PIN1_HUMAN
L2P,MPQPQVQHPIQSIKAFN,0.5,medium,2,PIN1_HUMAN
G3A,MVQAQVQHPIQSIKAFN,1.2,high,3,PIN1_HUMAN
D45A,MVQPQVQHPIQSIKAFN,0.8,medium,D45,PIN1_HUMAN
Y67F,MVQPQVQHPIQSIKAFN,-0.3,low,Y67,PIN1_HUMAN
K89R,MVQPQVQHPIQSIKAFN,1.5,high,K89,PIN1_HUMAN
EOF

    echo "Created example CSV file: $CSV_FILE"
    echo ""
    echo "CSV Format Requirements:"
    echo "  - mutant: Mutation identifier"
    echo "  - mutated_sequence: Sequence with mutations (optional)"
    echo "  - DMS_score: Deep Mutational Scanning score"
    echo "  - DMS_score_bin: Score category (low/medium/high)"
    echo "  - mask-positions: Positions to mask (e.g., '45' or 'D45')"
    echo "  - protein: UniProt ID"
    echo ""
fi

# Show CSV file contents
if [ -f "$CSV_FILE" ]; then
    mutation_count=$(tail -n +2 "$CSV_FILE" | wc -l)
    echo "CSV File Analysis:"
    echo "  Total mutations: $mutation_count"
    echo "  Estimated batches: $(echo "($mutation_count + $BATCH_SIZE - 1) / $BATCH_SIZE" | bc)"
    echo ""

    echo "Sample entries (first 5 lines):"
    head -n 6 "$CSV_FILE" | column -t -s ','
    echo ""
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Starting batch processing..."
echo "============================================="

# Run batch processing
python ../training/inpainting.py \
    --list_csv "$CSV_FILE" \
    --steps "$STEPS" \
    --flow_temp "$FLOW_TEMP" \
    --batch_size "$BATCH_SIZE" \
    --output-dir "$OUTPUT_DIR" \
    --model "../ckpts/inverse_folddir_model.pt" \
    --split_json ../datasets/cath-4.2/chain_set_splits.json \
    --map_pkl ../datasets/cath-4.2/chain_set_map_with_b_factors_dssp.pkl \
    --detailed_json \
    --verbose

echo ""
echo "Batch processing completed!"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Output files:"
echo "  - Sequences: $OUTPUT_DIR/batch_processing_sequences.csv"
echo "  - Probabilities: $OUTPUT_DIR/*_probabilities.npz"
echo "  - Detailed JSON: $OUTPUT_DIR/*_detailed_predictions.json"
echo "  - Metadata: $OUTPUT_DIR/*_metadata.txt"
echo ""

# Show processing summary if results exist
if [ -f "$OUTPUT_DIR/batch_processing_sequences.csv" ]; then
    echo "Batch Processing Results:"

    # Count results
    total_processed=$(tail -n +2 "$OUTPUT_DIR/batch_processing_sequences.csv" | wc -l)
    echo "Total mutations processed: $total_processed"

    # Show sample results
    if [ $total_processed -gt 0 ]; then
        echo ""
        echo "Sample results (first 3):"
        head -n 4 "$OUTPUT_DIR/batch_processing_sequences.csv" | column -t -s ','

        # Basic accuracy statistics if available
        if command -v awk > /dev/null; then
            echo ""
            echo "Quick Statistics:"

            # Average accuracy (if accuracy column exists)
            avg_accuracy=$(tail -n +2 "$OUTPUT_DIR/batch_processing_sequences.csv" | awk -F',' '
                BEGIN {sum=0; count=0}
                $6 != "" && $6 != "None" {sum+=$6; count++}
                END {if(count>0) printf "%.2f", sum/count; else print "N/A"}
            ')
            echo "  Average accuracy: ${avg_accuracy}%"
        fi
    fi

    echo ""
    echo "Analysis Recommendations:"
    echo "  1. Compare predictions across DMS score bins"
    echo "  2. Analyze position-specific accuracy patterns"
    echo "  3. Correlate predictions with experimental scores"
    echo "  4. Look for systematic biases or trends"

    echo ""
    echo "Next Steps for Analysis:"
    echo "  python -c \"from inversefolddir_tools import load_results; load_results('$OUTPUT_DIR').summary()\""
    echo "  See the batch summary CSV and NPZ written into $OUTPUT_DIR"
fi
