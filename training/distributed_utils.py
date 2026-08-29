# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
distributed_utils.py

Utilities for distributed training support with backward compatibility.
"""

import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Verbose chunk-loading and startup tracing, off unless asked for. These lines
# are genuinely useful when a training run misbehaves, but they should not be
# the default experience. Set IFD_DEBUG=1 to turn them back on.
_IFD_DEBUG = bool(os.environ.get("IFD_DEBUG"))


def setup_distributed(device_arg='auto'):
    """
    Setup distributed training if environment variables are present.

    Args:
        device_arg: Device specification ('auto', 'cpu', 'cuda', 'cuda:0', etc.)

    Returns:
        tuple: (is_distributed, rank, world_size, local_rank, device)
    """
    if _IFD_DEBUG: print("DEBUG: setup_distributed called", flush=True)

    # Check if we're in a distributed environment (set by torchrun)
    # All three environment variables must be present for distributed training
    world_size_env = os.environ.get('WORLD_SIZE')
    rank_env = os.environ.get('RANK')
    local_rank_env = os.environ.get('LOCAL_RANK')

    if _IFD_DEBUG: print(f"DEBUG: Environment variables - WORLD_SIZE={world_size_env}, RANK={rank_env}, LOCAL_RANK={local_rank_env}", flush=True)

    if (world_size_env and rank_env and local_rank_env):
        if _IFD_DEBUG: print("DEBUG: Distributed environment detected...", flush=True)
        # NOTE: container-specific NCCL overrides are intentionally not set here.
        # Default NCCL settings support InfiniBand on most HPC clusters.
        # To customize NCCL, set environment variables in your job script:
        #   export NCCL_DEBUG=INFO  # for debugging
        #   export NCCL_IB_DISABLE=0  # enable InfiniBand (the default on most clusters)

        if _IFD_DEBUG: print("DEBUG: Initializing distributed process group...", flush=True)
        # Initialize distributed process group
        try:
            dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')
            if _IFD_DEBUG: print("DEBUG: Successfully initialized distributed process group with NCCL/gloo", flush=True)
        except Exception as e:
            if _IFD_DEBUG: print(f"DEBUG: Failed to initialize NCCL backend: {e}", flush=True)
            if _IFD_DEBUG: print("DEBUG: Falling back to gloo backend for distributed training", flush=True)
            dist.init_process_group(backend='gloo')

        rank = int(rank_env)
        world_size = int(world_size_env)
        local_rank = int(local_rank_env)

        if _IFD_DEBUG: print(f"DEBUG: Parsed distributed parameters - rank={rank}, world_size={world_size}, local_rank={local_rank}", flush=True)

        # Set device for this process
        if _IFD_DEBUG: print("DEBUG: Getting device for distributed training...", flush=True)
        device = _get_device(device_arg, local_rank, distributed=True)
        if device.type == 'cuda':
            if _IFD_DEBUG: print(f"DEBUG: Setting CUDA device to {device}", flush=True)
            torch.cuda.set_device(device)

        print(f"Distributed training: rank {rank}/{world_size}, local_rank {local_rank}, device {device}")
        return True, rank, world_size, local_rank, device
    else:
        if _IFD_DEBUG: print("DEBUG: No distributed environment detected, using single process...", flush=True)
        # Single process training
        device = _get_device(device_arg, 0, distributed=False)
        print(f"Single process training on device: {device}")
        return False, 0, 1, 0, device

def _get_device(device_arg, local_rank, distributed=False):
    """
    Determine the appropriate device based on the device argument.

    Args:
        device_arg: Device specification ('auto', 'cpu', 'cuda', 'cuda:0', etc.)
        local_rank: Local rank for distributed training
        distributed: Whether in distributed mode

    Returns:
        torch.device: The device to use
    """
    if device_arg == 'auto':
        # Automatic device selection
        if torch.cuda.is_available():
            if distributed:
                return torch.device(f'cuda:{local_rank}')
            else:
                return torch.device('cuda')
        else:
            print("WARNING: CUDA not available, falling back to CPU")
            return torch.device('cpu')
    elif device_arg == 'cpu':
        return torch.device('cpu')
    elif device_arg.startswith('cuda'):
        if not torch.cuda.is_available():
            print(f"ERROR: CUDA requested ({device_arg}) but not available. Available devices:")
            print(f"  CUDA available: {torch.cuda.is_available()}")
            print(f"  CUDA device count: {torch.cuda.device_count()}")
            raise RuntimeError(f"CUDA requested but not available")
        return torch.device(device_arg)
    else:
        # Try to create device directly
        try:
            return torch.device(device_arg)
        except Exception as e:
            print(f"ERROR: Invalid device specification: {device_arg}")
            print(f"Error: {e}")
            raise

def wrap_model_for_distributed(model, is_distributed, device):
    """
    Wrap model with DDP if distributed training is enabled.

    Args:
        model: PyTorch model
        is_distributed: Whether distributed training is enabled
        device: Device to move model to

    Returns:
        Wrapped model (DDP if distributed, otherwise original)
    """
    model = model.to(device)

    if is_distributed:
        # Find unused parameters (like virtual node features that might not get gradients)
        model = DDP(model, device_ids=[device.index] if device.type == 'cuda' else None,
                   find_unused_parameters=True)
        print(f"Model wrapped with DistributedDataParallel")

    return model

def create_distributed_sampler(dataset, is_distributed, shuffle=True):
    """
    Create distributed sampler if needed, otherwise return None.

    Args:
        dataset: PyTorch dataset
        is_distributed: Whether distributed training is enabled
        shuffle: Whether to shuffle data

    Returns:
        DistributedSampler if distributed, None otherwise
    """
    if is_distributed:
        return DistributedSampler(dataset, shuffle=shuffle)
    return None

def cleanup_distributed():
    """Clean up distributed training resources."""
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process():
    """Check if this is the main process (rank 0)."""
    return not dist.is_initialized() or dist.get_rank() == 0

def reduce_loss_across_processes(loss_tensor):
    """
    Reduce loss across all processes for logging.

    Args:
        loss_tensor: Loss tensor to reduce

    Returns:
        Reduced loss tensor
    """
    if dist.is_initialized():
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        loss_tensor /= dist.get_world_size()
    return loss_tensor
