#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Single process training (existing behavior)

echo "Starting single process training..."

python train.py \
  --split_json ../datasets/cath-4.2/chain_set_splits.json \
  --map_pkl ../datasets/cath-4.2/chain_set_map.pkl \
  --epochs 50 \
  --lr 1e-3 \
  --batch 64 \
  --use_wandb \
  --model_name_prefix single_process_training
