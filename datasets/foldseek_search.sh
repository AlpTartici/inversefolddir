# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# write an sh file that runs foldseek search
#!/bin/bash

# Parse command line arguments
QUERY_DIR="${1:-test_val_chain_pdbs}"
TARGET_DB="${2:-pdb_pad}"
OUTPUT_DIR="${3:-foldseek_search_results}"
NUM_GPUS="${4:-1}"

# Check if query directory exists
if [ ! -d "$QUERY_DIR" ]; then
    echo "Error: Query directory '$QUERY_DIR' does not exist"
    echo "Usage: $0 [query_directory] [target_database] [output_directory] [num_gpus]"
    echo "  query_directory: Directory containing PDB files to search (default: test_val_chain_pdbs)"
    echo "  target_database: Target database for search (default: pdb_pad)"
    echo "  output_directory: Directory to save search results (default: foldseek_search_results)"
    echo "  num_gpus: Number of GPUs to use (default: 1)"
    exit 1
fi

# get a list of files in the query directory
query_files=$(ls "$QUERY_DIR")

# put the code below in a for loop to run foldseek easy search on each file
# in the list of files in the query directory
# and save the results in the specified output directory
# create the output directory if it does not exist
mkdir -p "$OUTPUT_DIR"
for query_file in $query_files; do
    # get the first file in the list with relative path
    #query_file=$(echo "$query_files" | head -n 1)
    # get the first file in the list with absolute path
    query_file_abs=$(realpath "$QUERY_DIR"/"$query_file")

    # write foldseek easy search command with flag names
    # run foldseek easy search with the first file in the list
    # and the target database
    # and save the results in the specified output directory

    echo "Running foldseek easy search on $query_file_abs"

    foldseek easy-search \
        "$query_file_abs" \
        "$TARGET_DB" \
        "$OUTPUT_DIR"/testpdb_"$query_file".txt \
        "$OUTPUT_DIR"/tmp_fs_search \
        --format-output "query,target,alntmscore,lddt,evalue,prob,alnlen"
        #\
        #--gpu "$NUM_GPUS"


done

# example run
# ./foldseek_search.sh test_val_chain_pdbs pdb_pad foldseek_search_results 1
