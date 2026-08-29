#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Distributed training with current CATH dataset

echo "Starting distributed training..."

export MASTER_ADDR=localhost
export MASTER_PORT=12355

# Launch distributed training with 4 processes
torchrun --nproc_per_node=4 \
         --nnodes=1 \
         --node_rank=0 \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         train.py \
         --distributed \
         --split_json ../datasets/cath-4.2/chain_set_splits.json \
         --map_pkl ../datasets/cath-4.2/chain_set_map.pkl \
         --epochs 50 \
         --lr 1e-3 \
         --batch 64 \
         --use_wandb \
         --model_name_prefix distributed_training
