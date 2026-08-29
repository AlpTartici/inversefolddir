# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import gc
import json
import os
import random
import sys
import time
from datetime import datetime

import torch
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import multiprocessing as mp
import warnings

import numpy as np
import torch.nn.functional as F
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.aa_constants import FIXED_AA_CLASS_WEIGHTS
from data.cath_dataset import CathDataset
from flow.sampler import sample_forward
from models.dfm_model import DFMNodeClassifier
from training.collate import collate_fn
from training.config import get_training_config

# Add distributed training imports
from training.distributed_utils import (
    cleanup_distributed,
    create_distributed_sampler,
    is_main_process,
    reduce_loss_across_processes,
    setup_distributed,
    wrap_model_for_distributed,
)
from training.training_utils import (
    apply_dssp_masking,
    apply_geometry_masking,
    apply_inpainting_noise,
    apply_same_masking_to_weights,
    apply_virtual_node_masking,
    cleanup_old_checkpoints,
    get_rank,
    get_valid_node_mask,
    load_checkpoint_for_training,
    load_optimizer_and_scheduler_state,
    mask_virtual_nodes_from_batch,
    run_generation_validation,
    run_validation_phase,
    save_model_with_metadata,
    setup_generation_validation,
    should_log_to_wandb,
    update_config_with_best_metrics,
)

# Verbose chunk-loading and startup tracing, off unless asked for. These lines
# are genuinely useful when a training run misbehaves, but they should not be
# the default experience. Set IFD_DEBUG=1 to turn them back on.
_IFD_DEBUG = bool(os.environ.get("IFD_DEBUG"))


def reset_optimizer_momentum_selectively(optimizer, target_device=None):
    """
    Reset problematic optimizer state components while preserving valuable adaptive information.

    Resets:
    - exp_avg (momentum): Prevents momentum-LR mismatch discontinuity
    - step counters: Ensures proper bias correction with fresh start

    Preserves:
    - exp_avg_sq (variance estimates): Keeps learned per-parameter adaptive scaling

    Args:
        optimizer: PyTorch optimizer to reset
        target_device: Device to move optimizer state tensors to (for device consistency)
    """
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p in optimizer.state:
                state = optimizer.state[p]

                # Ensure parameter itself is on target device
                if target_device is not None and p.device != target_device:
                    # This shouldn't happen in normal rollback, but let's be safe
                    print(
                        f"WARNING: Parameter device mismatch detected: {p.device} != {target_device}"
                    )

                # Reset momentum (first moment) - this was causing the loss spike
                if "exp_avg" in state:
                    state["exp_avg"].zero_()
                    # Ensure device consistency
                    if target_device is not None:
                        state["exp_avg"] = state["exp_avg"].to(target_device)

                # Keep exp_avg_sq (second moment) but ensure it's on correct device.
                # Do NOT reset step: bias_correction2 = 1 - beta2^step must stay
                # consistent with the accumulated exp_avg_sq statistics; resetting step
                # to 0 while keeping exp_avg_sq makes the effective denominator ~31x too
                # large on the first post-rollback step (M-1 fix).
                if "exp_avg_sq" in state and target_device is not None:
                    state["exp_avg_sq"] = state["exp_avg_sq"].to(target_device)

                # Handle any other state tensors that might exist (comprehensive)
                for key, value in list(
                    state.items()
                ):  # Use list() to avoid dict modification during iteration
                    if torch.is_tensor(value) and key not in [
                        "exp_avg",
                        "exp_avg_sq",
                        "step",
                    ]:
                        if target_device is not None:
                            state[key] = value.to(target_device)

                # Additional Adam-specific state handling
                if hasattr(optimizer, "state_dict"):
                    # Some optimizers have additional hidden state
                    try:
                        # Force a state dict rebuild to ensure consistency
                        temp_state = optimizer.state_dict()
                    except Exception:
                        pass  # Ignore if state_dict fails

    # Final verification step
    if target_device is not None:
        device_mismatches = []
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p in optimizer.state:
                    state = optimizer.state[p]
                    for key, value in state.items():
                        if (
                            torch.is_tensor(value) and key != "step"
                        ):  # step can be on CPU
                            if value.device != target_device:
                                device_mismatches.append(
                                    f"Param {id(p)}, {key}: {value.device} != {target_device}"
                                )

        if device_mismatches:
            print(
                f"WARNING: Device mismatches after selective reset: {device_mismatches}"
            )
            # Force move any remaining mismatched tensors
            for group in optimizer.param_groups:
                for p in group["params"]:
                    if p in optimizer.state:
                        state = optimizer.state[p]
                        for key, value in state.items():
                            if torch.is_tensor(value) and key != "step":
                                if value.device != target_device:
                                    state[key] = value.to(target_device)
                                    print(f"Force-moved {key} to {target_device}")
        else:
            print(
                f"Device consistency verified in selective reset: all tensors on {target_device}"
            )


def main():
    """
    Main function to run the training process for the DFMNodeClassifier.

    This script handles:
    - Setting up configuration and output directories.
    - Setting up the CATH dataset and DataLoader.
    - Initializing the model, optimizer.
    - Running the training loop for a specified number of epochs.
    - Optional distributed training support.
    """

    if _IFD_DEBUG: print("DEBUG: Starting main() function", flush=True)

    # Configuration Setup
    if _IFD_DEBUG: print("DEBUG: Getting training config...", flush=True)
    args, output_base, model_name = get_training_config()
    if _IFD_DEBUG: print("DEBUG: Training config obtained successfully", flush=True)

    # Job timestamp for consistent file naming
    job_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if _IFD_DEBUG: print(f"DEBUG: Job timestamp: {job_timestamp}", flush=True)

    # Set global debug flag for graph_builder
    if args.debug_mode:
        torch._debug_mode = True

    # Distributed Setup
    if _IFD_DEBUG: print("DEBUG: Setting up distributed training...", flush=True)
    is_distributed, rank, world_size, local_rank, device = setup_distributed(
        args.device
    )
    print(
        f"DEBUG: Distributed setup complete. is_distributed={is_distributed}, rank={rank}, world_size={world_size}",
        flush=True,
    )

    # Re-setup reproducibility with rank-aware seeding for distributed training
    from training.config import setup_reproducibility

    setup_reproducibility(args.seed, rank if is_distributed else None)
    # Create a model_config directory to save run details
    config_dir = os.path.join(output_base, "model_config")
    os.makedirs(config_dir, exist_ok=True)
    # Dump hyperparameters & seed (use job_timestamp in filename)
    config_path = os.path.join(config_dir, f"config_{job_timestamp}_{model_name}.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    # Only main process prints configuration
    if is_main_process():
        print(f"Training configuration loaded. Model name: {model_name}")
        # print(f"Device configuration: {args.device} -> {device}")
        # print(f"CUDA available: {torch.cuda.is_available()}")
        # if torch.cuda.is_available():
        #     print(f"CUDA device count: {torch.cuda.device_count()}")
        #     if device.type == 'cuda':
        #         print(f"Using GPU: {torch.cuda.get_device_name(device)}")
        # if is_distributed:
        #     print(f"Distributed training enabled: {world_size} processes")
        # else:
        #     print("Single process training")

    # Set smoke test mode flag
    is_smoke_test = args.run_mode == "smoke_test"

    # Wandb Setup
    current_rank = get_rank()
    print(
        f"[DEBUG] Wandb setup: use_wandb={args.use_wandb}, is_smoke_test={is_smoke_test}, rank={current_rank}",
        flush=True,
    )
    print(
        f"[DEBUG] should_log_to_wandb result: {should_log_to_wandb(args, is_smoke_test)}",
        flush=True,
    )

    if should_log_to_wandb(args, is_smoke_test):
        if not WANDB_AVAILABLE:
            print("WARNING: wandb requested but not installed. Continuing without wandb logging.", flush=True)
            args.use_wandb = False
        else:
            print(
                f"[DEBUG] Attempting wandb initialization on rank {current_rank}",
                flush=True,
            )
            try:
                import signal

                def timeout_handler(signum, frame):
                    raise TimeoutError("Wandb initialization timed out after 120 seconds")

                # Use wandb API key from environment (set via `wandb login` or WANDB_API_KEY env var)
                print(f"[DEBUG] Attempting wandb.login (using system credentials)...", flush=True)
                wandb.login()  # Uses WANDB_API_KEY from environment or cached credentials
                print(
                    f"[DEBUG] wandb.login successful, attempting wandb.init with 120s timeout...",
                    flush=True,
                )

                # Set a 120-second timeout for wandb.init
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(120)

                try:
                    wandb_kwargs = {"project": args.wandb_project, "config": vars(args)}
                    if getattr(args, 'wandb_run_name', None):
                        wandb_kwargs["name"] = args.wandb_run_name
                    wandb.init(**wandb_kwargs)
                    signal.alarm(0)  # Cancel the alarm
                    print(
                        f"Successfully initialized wandb in online mode (rank: {current_rank})",
                        flush=True,
                    )
                except TimeoutError:
                    signal.alarm(0)  # Cancel the alarm
                    print(
                        f"Wandb initialization timed out after 120 seconds, falling back to offline mode",
                        flush=True,
                    )
                    raise Exception("Wandb init timeout")

            except Exception as e:
                print(f"Failed to initialize wandb in online mode: {e}", flush=True)
                print("Falling back to offline mode", flush=True)
                try:
                    os.environ["WANDB_MODE"] = "offline"
                    wandb_kwargs = {"project": args.wandb_project, "config": vars(args)}
                    if getattr(args, 'wandb_run_name', None):
                        wandb_kwargs["name"] = args.wandb_run_name
                    wandb.init(**wandb_kwargs)
                    print(
                        f"Successfully initialized wandb in offline mode (rank: {current_rank})",
                        flush=True,
                    )
                except Exception as e2:
                    print(f"Failed to initialize wandb in offline mode: {e2}", flush=True)
                    print("Continuing without wandb logging", flush=True)
                    args.use_wandb = False
    elif args.use_wandb:
        print(
            f"Skipping wandb initialization on rank {current_rank} (only rank 0 initializes wandb)",
            flush=True,
        )

    # Wandb initialization complete - proceeding to dataset setup
    if is_main_process():
        print(f"Wandb setup complete, proceeding to dataset creation...", flush=True)

    # Dataset and DataLoader Setup - Hybrid Training System
    if is_main_process():
        print(f"Training mode: ratio_af2_pdb={args.ratio_af2_pdb}")
        if args.ratio_af2_pdb == 0:
            print("Using PDB-only training (B-factors)")
        elif args.ratio_af2_pdb == -1:
            print("Using AF2-only training (pLDDT scores)")
        else:
            print(
                f"Using homogeneous alternating training: {args.ratio_af2_pdb} pure AF2 batches per 1 pure PDB batch"
            )

        # Show epoch timeout configuration (disabled per user request)
        print("Epoch timeout: DISABLED (infinite duration allowed)")

    # Include graph builder parameters
    graph_builder_kwargs = {
        "smoke_test": is_smoke_test,
        "use_virtual_node": args.use_virtual_node,
        "k": args.k_neighbors,
        "k_farthest": args.k_farthest,
        "k_random": args.k_random,
        "max_edge_dist": getattr(
            args, "max_edge_dist", None
        ),  # Distance-based edge building
        "num_rbf_3d": args.num_rbf_3d,
        "num_rbf_seq": args.num_rbf_seq,
        # RBF distance range parameters
        "rbf_3d_min": args.rbf_3d_min,
        "rbf_3d_max": args.rbf_3d_max,
        "rbf_3d_spacing": args.rbf_3d_spacing,
        "verbose": False,
        # Structure noise parameters - enabled for training
        "structure_noise_mag_std": args.structure_noise_mag_std,
        "time_based_struct_noise": args.time_based_struct_noise,
        "uncertainty_struct_noise_scaling": args.uncertainty_struct_noise_scaling,
        # Uncertainty processing parameters
        "blur_uncertainty": args.blur_uncertainty,
        # Node degree / packing feature
        "include_node_degree": getattr(args, "include_node_degree", False),
    }

    # Import unified dataset (simple and elegant like original CATH)
    from data.unified_dataset import create_unified_dataloader

    # Create training DataLoader using unified approach
    print(
        f"Creating unified dataloader (ratio_af2_pdb={args.ratio_af2_pdb})...",
        flush=True,
    )

    # Use unified dataset (mixed within batches)
    # Multi-worker mode enabled - proper file handle management ensures compatibility

    train_loader = create_unified_dataloader(
        # PDB parameters
        split_json=args.split_json,
        map_pkl=args.map_pkl,
        split="train",
        # AF2 parameters
        af2_chunk_dir=args.af2_chunk_dir if args.ratio_af2_pdb != 0 else None,
        af2_chunk_limit=getattr(
            args, "af2_chunk_limit", None
        ),  # Limit chunks for testing
        # AF2 data is always loaded lazily - no upfront loading option
        # Mixing parameters
        ratio_af2_pdb=args.ratio_af2_pdb,
        heterogeneous_batches=getattr(
            args, "heterogeneous_batches", True
        ),  # Default to heterogeneous
        # DataLoader parameters
        batch_size=args.batch,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,  # Deterministic iteration
        # Common parameters
        max_len=getattr(args, "max_len", None),
        graph_builder_kwargs=graph_builder_kwargs,
        verbose=getattr(args, "verbose", False),  # Pass verbose flag for debug output
        # Time sampling parameters
        time_sampling_strategy=args.time_sampling_strategy,
        t_min=args.t_min,
        t_max=args.t_max,
        alpha_range=args.alpha_range,
        # Iteration control
        deterministic=True,  # Cycle through data deterministically
        # Distributed training
        rank=rank,
        world_size=world_size,
        seed=args.seed,
        # AF2 pLDDT-based segment chopping
        af2_plddt_threshold=getattr(args, 'af2_plddt_threshold', 0.0),
        af2_min_segment_len=getattr(args, 'af2_min_segment_len', 60),
    )

    if args.ratio_af2_pdb == -1:
        print("AF2-only unified dataloader created (deterministic iteration)")
    elif args.ratio_af2_pdb == 0:
        print("PDB-only unified dataloader created (deterministic iteration)")
    else:
        batch_type = (
            "heterogeneous"
            if getattr(args, "heterogeneous_batches", True)
            else "homogeneous"
        )
        print(
            f"Mixed unified dataloader created ({args.ratio_af2_pdb} AF2 per 1 PDB, {batch_type} batches, deterministic iteration)"
        )

    print(
        f"Training DataLoader created successfully! Starting validation dataset setup...",
        flush=True,
    )

    # Protein tracking now handled at dataset level
    # (no need for training-loop level protein tracker)

    # Use unified dataset for validation (PDB-only as requested)
    if (args.split_json is not None) and (args.map_pkl is not None):
        if is_main_process():
            print(
                f"Creating PDB-only validation dataset using unified approach...",
                flush=True,
            )

        # Create validation dataset using unified approach with ratio_af2_pdb=0 (PDB-only)
        from data.unified_dataset import UnifiedDataset

        # Create TRAINING validation dataset (same noise as training)
        # This preserves the original working approach that gave 90% DSSP accuracy
        val_ds = UnifiedDataset(
            # PDB parameters for validation
            split_json=args.split_json,
            map_pkl=args.map_pkl,
            split="validation",
            # AF2 parameters (disabled for validation)
            af2_chunk_dir=None,  # No AF2 data for validation
            af2_chunk_limit=None,
            # Mixing parameters (PDB-only)
            ratio_af2_pdb=0,  # PDB-only for validation as requested
            heterogeneous_batches=False,  # Not relevant for PDB-only
            # Common parameters
            max_len=getattr(args, "max_len", None),
            graph_builder_kwargs=graph_builder_kwargs,  # SAME as training (original approach)
            # Time sampling parameters
            time_sampling_strategy=args.time_sampling_strategy,
            t_min=args.t_min,
            t_max=args.t_max,
            alpha_range=args.alpha_range,
            # Iteration control
            deterministic=True,  # Cycle through data deterministically
            # Distributed training
            rank=rank,
            world_size=world_size,
            seed=args.seed,
        )

        print(f"PDB-only unified validation dataset created successfully")
    else:
        raise ValueError("CATH split_json and map_pkl required for validation")

    # No train_ds for unified mode, only train_loader
    train_ds = None

    # Handle special overfit modes if needed
    if args.run_mode == "overfit_on_one" and args.overfit_protein_name:
        from data.single_protein_dataset import SingleProteinDataset

        # Create dataset with only the target protein
        try:
            val_ds = SingleProteinDataset(
                split_json=args.split_json,
                map_pkl=args.map_pkl,
                protein_name=args.overfit_protein_name,
                split="validation",
                graph_builder_kwargs=graph_builder_kwargs,
            )
            print(
                f"[SUCCESS] Single protein dataset created: 1 protein only ({args.overfit_protein_name})"
            )
        except ValueError as e:
            raise ValueError(
                f"Error creating single protein dataset: {e}. Check protein name and data availability."
            )

    # Create validation DataLoader
    if val_ds is not None:
        val_sampler = (
            create_distributed_sampler(val_ds, is_distributed, shuffle=False)
            if is_distributed
            else None
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.val_batch,
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )
    else:
        raise ValueError("Validation dataset creation failed")

    # Worker initialization function for reproducible DataLoader behavior
    def worker_init_fn(worker_id):
        # Ensure each worker on each process gets a unique seed
        worker_seed = args.seed + worker_id + (rank * 1000 if is_distributed else 0)
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    # Set epoch on hybrid train loader for distributed training
    if hasattr(train_loader, "set_epoch"):
        train_loader.set_epoch(0)  # Will be updated each epoch

    # Create dual validation loaders for different noise schedules
    # Use original batch size for performance
    val_fixed_loader = DataLoader(
        val_ds,
        batch_size=args.batch,  # Original approach
        shuffle=False,
        sampler=val_sampler,  # Use distributed sampler if available
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
    )
    val_unfixed_loader = DataLoader(
        val_ds,
        batch_size=args.batch,  # Original approach
        shuffle=False,
        sampler=val_sampler,  # Use distributed sampler if available
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
    )

    if args.run_mode == "overfit":
        train_loader = (
            val_fixed_loader  # Use fixed validation loader for training in overfit mode
        )

    if args.run_mode == "overfit_on_one":
        # Find the protein by name if specified, otherwise use first validation sample
        sample_idx = 0  # Default to first sample
        single_sample = None
        protein_name = "unknown"
        protein_source = "unknown"
        sequence_length = 0

        # Check if we're using the single protein dataset
        from data.single_protein_dataset import SingleProteinDataset

        if args.overfit_protein_name and isinstance(val_ds, SingleProteinDataset):
            # We already have the single protein dataset
            single_sample = [val_ds[0]]
            sample_data, sample_y, sample_mask, _ = val_ds[0]
            protein_name = getattr(sample_data, "name", "unknown")
            protein_source = getattr(sample_data, "source", "unknown")
            sequence_length = sample_data.num_nodes - (
                1 if getattr(sample_data, "use_virtual_node", False) else 0
            )
            sample_idx = 0

            if is_main_process():
                print(f"Using single protein dataset for overfit: {protein_name}")

        elif args.overfit_protein_name:
            # Search for the protein by name in the validation dataset (fallback)
            found_idx = None
            if is_main_process():
                print(f"Searching for protein: {args.overfit_protein_name}")

            for idx in range(len(val_ds)):
                try:
                    sample_data, _, _, _ = val_ds[idx]
                    sample_name = getattr(sample_data, "name", "")
                    if sample_name == args.overfit_protein_name:
                        found_idx = idx
                        if is_main_process():
                            print(
                                f"Found protein '{args.overfit_protein_name}' at index {idx}"
                            )
                        break
                except Exception as e:
                    if is_main_process():
                        print(f"Warning: Could not load sample at index {idx}: {e}")
                    continue

            if found_idx is not None:
                sample_idx = found_idx
            else:
                if is_main_process():
                    print(
                        f"Warning: Protein '{args.overfit_protein_name}' not found in validation set."
                    )
                    print("Available proteins in first 10 validation samples:")
                    for i in range(min(10, len(val_ds))):
                        try:
                            sample_data, _, _, _ = val_ds[i]
                            sample_name = getattr(sample_data, "name", f"sample_{i}")
                            print(f"  Index {i}: {sample_name}")
                        except:
                            print(f"  Index {i}: <could not load>")
                    print("Using first validation sample instead.")

        # Create a single-sample dataset from the selected sample (if not already set)
        if single_sample is None:
            try:
                single_sample = [val_ds[sample_idx]]
                # Get information about the protein we're overfitting on
                sample_data, sample_y, sample_mask, _ = val_ds[sample_idx]
                protein_name = getattr(sample_data, "name", "unknown")
                protein_source = getattr(sample_data, "source", "unknown")
                sequence_length = sample_data.num_nodes - (
                    1 if getattr(sample_data, "use_virtual_node", False) else 0
                )
            except Exception as e:
                if is_main_process():
                    print(f"Error loading sample at index {sample_idx}: {e}")
                    print("Falling back to first validation sample.")
                single_sample = [val_ds[0]]
                sample_data, sample_y, sample_mask, _ = val_ds[0]
                protein_name = getattr(sample_data, "name", "unknown")
                protein_source = getattr(sample_data, "source", "unknown")
                sequence_length = sample_data.num_nodes - (
                    1 if getattr(sample_data, "use_virtual_node", False) else 0
                )
                sample_idx = 0

        train_loader = DataLoader(
            single_sample,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )
        val_fixed_loader = DataLoader(
            single_sample,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )
        val_unfixed_loader = DataLoader(
            single_sample,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )
        if is_main_process():
            print(
                "Running in overfit_on_one mode: using single sample for both training and validation."
            )
            print(
                f"  Overfitting on protein: {protein_name} (source: {protein_source})"
            )
            print(f"  Sequence length: {sequence_length} residues")
            print(f"  Dataset index: {sample_idx}")
            if args.use_wandb:
                # Log the protein info to wandb as well

                wandb.config.update(
                    {
                        "overfit_protein_name": protein_name,
                        "overfit_protein_source": protein_source,
                        "overfit_sequence_length": sequence_length,
                        "overfit_dataset_index": sample_idx,
                    }
                )

    # Model Initialization
    if is_main_process():
        print(f"Using device: {device}")

    # Compute effective edge dimension (add 2 if entropy features are enabled)
    effective_edge_dim_s = args.edge_dim_s
    if getattr(args, 'use_entropy_features', False):
        effective_edge_dim_s = args.edge_dim_s + 2  # +2 for source and target entropy
        if is_main_process():
            print(f"Entropy edge features enabled: edge_dim_s increased from {args.edge_dim_s} to {effective_edge_dim_s}")

    if getattr(args, 'include_node_degree', False):
        args.node_dim_s += 2  # log-packing score + within-structure percentile rank
        if is_main_process():
            print(f"Node packing features enabled (log_score + percentile): node_dim_s increased to {args.node_dim_s}")

    # Use command-line arguments to control model architecture
    model = DFMNodeClassifier(
        gvp_kwargs=dict(
            node_dims=(
                args.node_dim_s,
                args.node_dim_v,
            ),  # Configurable node feature dimensions
            edge_dims=(
                effective_edge_dim_s,
                args.edge_dim_v,
            ),  # Configurable edge feature dimensions (adjusted for entropy features)
            hidden_dims=(
                args.hidden_dim,
                args.hidden_dim_v,
            ),  # Use configurable hidden dimensions
            num_layers=args.num_layers_gvp,  # Use configurable number of GVP layers
            num_message_layers=args.num_message_layers,  # Use configurable number of message passing layers
            use_qkv=args.use_qkv,
            dropout=args.dropout,
            architecture=args.architecture,  # Architecture type: 'blocked' or 'interleaved'
            update_edge_feats=args.update_edge_feats,  # SE(3)-equivariant edge updates
            attention_skip_connection=getattr(args, 'attention_skip_connection', False),
        ),
        dfm_kwargs=dict(
            K=21,
            alpha_min=args.alpha_min,
            alpha_max=args.alpha_max,
            alpha_spacing=args.alpha_spacing,
            label_similarity_csv=(
                args.label_similarity_csv if args.use_smoothed_labels else None
            ),
        ),
        time_dim=args.time_dim,
        time_scale=args.time_scale,
        head_hidden=args.head_hidden,
        head_dropout=args.dropout,
        head_depth=args.num_layers_prediction,  # Add configurable prediction head depth
        recycle_steps=args.recycle_steps,  # Add recycling steps parameter
        use_time_conditioning=not args.disable_time_conditioning,  # enable time conditioning by default
        time_integration=args.time_integration,  #  time integration method ('film' or 'add')
        lambda_dssp_loss=args.lambda_dssp_loss,  # Enable DSSP multitask learning
        lambda_electrostatic_loss=args.lambda_electrostatic_loss,
        lambda_geom_topology_loss=args.lambda_geom_topology_loss,
        use_entropy_features=getattr(args, 'use_entropy_features', False),
        entropy_smoothing=getattr(args, 'entropy_smoothing', 0.1),
    )

    # Print model parameter count for debugging
    if is_main_process():
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameter count:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Non-trainable parameters: {total_params - trainable_params:,}")

        # Check if any parameters are frozen
        frozen_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                frozen_params.append(name)

        if frozen_params:
            print(
                f"  WARNING: {len(frozen_params)} parameters are frozen (requires_grad=False):"
            )
            for name in frozen_params[:10]:  # Show first 10
                print(f"    {name}")
            if len(frozen_params) > 10:
                print(f"    ... and {len(frozen_params) - 10} more")

        # Print model architecture summary
        print(f"Model architecture:")
        print(f"  GVP layers: {args.num_layers_gvp}")
        print(f"  Prediction head layers: {args.num_layers_prediction}")
        print(f"  Hidden dimensions: {args.hidden_dim}")
        print(f"  Use QKV: {args.use_qkv}")
        print(f"  Use virtual node: {args.use_virtual_node}")
        print(f"  Recycle steps: {args.recycle_steps}")

    # Wrap model for distributed training
    model = wrap_model_for_distributed(model, is_distributed, device)

    # Print model parameter count for debugging
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main_process():
        print(f"Model parameter count:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(
            f"  Model architecture: {args.num_layers_gvp} GVP layers, {args.num_layers_prediction} head layers, {args.hidden_dim} hidden dim, QKV={args.use_qkv}"
        )

    # Optimizer Setup
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(
        opt,
        mode="max" if getattr(args, 'val_metric', 'val_fixed') == 'val_gen' else "min",
        factor=0.25,
        patience=args.scheduler_patience,
        min_lr=1e-7,
    )

    # Training Loop
    use_val_gen_metric = getattr(args, 'val_metric', 'val_fixed') == 'val_gen'
    best_val_loss = float("-inf") if use_val_gen_metric else float("inf")
    best_val_accuracy = 0.0
    best_epoch = 0
    patience_counter = 0
    best_model_path = None  # Track the path to the saved best model
    # Rolling history for val_gen smoothing (stores raw mean+median scores)
    val_gen_score_history = []

    # Initialize starting epoch and training state
    start_epoch = 1
    training_state = {}

    # Checkpoint rollback tracking variables (initialize for all cases)
    current_lr = args.lr
    epochs_since_lr_reduction = 0
    epochs_since_best_checkpoint = 0
    best_model_state = None
    rollback_patience_threshold = 0
    lr_has_dropped = False  # Track if LR has dropped at least once
    rollback_count = 0  # Track cumulative number of rollbacks

    # LR warmup state — activated when resuming without optimizer state (H-1 fix)
    # Ramps LR from near-zero to target over _LR_WARMUP_STEPS steps to avoid Adam cold-start spike.
    _lr_warmup_active = False
    _lr_warmup_steps = 1000
    _lr_warmup_step = 0
    _lr_warmup_target = args.lr

    # Handle checkpoint resumption if specified
    if hasattr(args, "resume_from_checkpoint") and args.resume_from_checkpoint:
        print("=" * 60)
        print("RESUMING FROM CHECKPOINT")
        print("=" * 60)

        try:
            # Load checkpoint with comprehensive parameter merging
            checkpoint_results = load_checkpoint_for_training(
                args.resume_from_checkpoint, device, args, model=model
            )

            # Update starting epoch
            start_epoch = checkpoint_results["epoch"] + 1
            # Sanity-check the epoch value to catch corrupted checkpoints (L-2 fix).
            if not (1 <= start_epoch <= args.epochs):
                raise ValueError(
                    f"Checkpoint epoch {checkpoint_results['epoch']} is out of range "
                    f"[0, {args.epochs - 1}]; checkpoint may be corrupted."
                )
            print(f"Resuming training from epoch {start_epoch}")

            # Restore training metrics and state
            if checkpoint_results["metrics"]:
                best_val_loss = checkpoint_results["metrics"].get(
                    "best_val_loss", best_val_loss
                )
                best_val_accuracy = checkpoint_results["metrics"].get(
                    "best_val_accuracy", best_val_accuracy
                )
                best_epoch = checkpoint_results["metrics"].get("best_epoch", best_epoch)
                print(
                    f"Restored metrics - Best loss: {best_val_loss:.6f}, Best acc: {best_val_accuracy:.4f}, Best epoch: {best_epoch}"
                )
                best_val_loss = best_val_loss * 100
                best_val_accuracy = best_val_accuracy / 10
                print(
                    f"changed best_val_loss to {best_val_loss}, best_val_accuracy to {best_val_accuracy}",
                    flush=True,
                )
            lr_was_overridden = True
            if args.use_checkpoint_opt_and_sched:
                # Load optimizer and scheduler state with proper device handling
                opt_scheduler_status = load_optimizer_and_scheduler_state(
                    opt, scheduler, checkpoint_results, device
                )
            else:
                # Optimizer state intentionally skipped — Adam bias correction starts from step=0
                # with mature model weights, causing up to ~31x effective LR amplification in the
                # first ~1000 steps.  Ramp LR from near-zero to target to suppress this spike.
                print(
                    "WARNING: Resuming without optimizer state. "
                    "Activating LR warmup over 1000 steps to suppress Adam cold-start spike."
                )
                _lr_warmup_active = True
                _lr_warmup_target = args.lr
                for pg in opt.param_groups:
                    pg["lr"] = 0.0

            # Override learning rate if explicitly provided via command line

            # Get original checkpoint arguments (before merging)
            checkpoint_args_dict = {}
            if isinstance(checkpoint_results.get("args"), dict):
                checkpoint_args_dict = checkpoint_results["args"]
            elif hasattr(checkpoint_results.get("args"), "__dict__"):
                checkpoint_args_dict = vars(checkpoint_results["args"])

            # Also check other possible keys for checkpoint args
            for key in ["training_args", "hyperparams", "config"]:
                if key in checkpoint_results and checkpoint_results[key]:
                    if isinstance(checkpoint_results[key], dict):
                        checkpoint_args_dict = checkpoint_results[key]
                        break
                    elif hasattr(checkpoint_results[key], "__dict__"):
                        checkpoint_args_dict = vars(checkpoint_results[key])
                        break

            # Get the original command-line args (before any merging)
            original_cmd_args_dict = vars(args)

            # Print debug info about LR sources
            checkpoint_lr = checkpoint_args_dict.get("lr", "NOT_FOUND")
            command_line_lr = original_cmd_args_dict.get("lr", "NOT_FOUND")

            print(f"\n{'='*60}")
            print("LEARNING RATE OVERRIDE DEBUG")
            print(f"{'='*60}")
            print(f"Checkpoint LR: {checkpoint_lr}")
            print(f"Command-line LR: {command_line_lr}")
            if args.use_checkpoint_opt_and_sched:
                print(
                    f"Optimizer loaded: {opt_scheduler_status.get('optimizer_loaded', False)}"
                )

            # Check if learning rate was explicitly overridden from command line
            if args.use_checkpoint_opt_and_sched:
                lr_was_overridden = (
                    checkpoint_args_dict
                    and "lr" in checkpoint_args_dict
                    and "lr" in original_cmd_args_dict
                    and checkpoint_args_dict["lr"] != original_cmd_args_dict["lr"]
                    and opt_scheduler_status.get("optimizer_loaded", False)
                )

            print(f"LR override detected: {lr_was_overridden}")

            if lr_was_overridden:
                old_lr = checkpoint_args_dict["lr"]
                new_lr = original_cmd_args_dict["lr"]
                print(f"\n{'='*50}")
                print("APPLYING LEARNING RATE OVERRIDE")
                print(f"{'='*50}")
                print(f"Checkpoint learning rate: {old_lr}")
                print(f"Command-line learning rate: {new_lr}")

                # Check current optimizer state before override
                current_optimizer_lrs = [
                    param_group["lr"] for param_group in opt.param_groups
                ]
                print(f"Current optimizer LRs before override: {current_optimizer_lrs}")

                # Update optimizer learning rate in all parameter groups
                for i, param_group in enumerate(opt.param_groups):
                    param_group["lr"] = new_lr
                    print(f"  Updated param_group[{i}]['lr']: {param_group['lr']}")

                # Reset scheduler state to use the new learning rate
                # For ReduceLROnPlateau, we need to reset the internal best value and other state
                if hasattr(scheduler, "best"):
                    old_best = scheduler.best
                    scheduler.best = None  # Reset best metric so scheduler starts fresh
                    print(f"  Reset scheduler.best: {old_best} -> {scheduler.best}")
                if hasattr(scheduler, "cooldown_counter"):
                    old_cooldown = scheduler.cooldown_counter
                    scheduler.cooldown_counter = 0
                    print(
                        f"  Reset scheduler.cooldown_counter: {old_cooldown} -> {scheduler.cooldown_counter}"
                    )
                if hasattr(scheduler, "num_bad_epochs"):
                    old_bad_epochs = scheduler.num_bad_epochs
                    scheduler.num_bad_epochs = 0
                    print(
                        f"  Reset scheduler.num_bad_epochs: {old_bad_epochs} -> {scheduler.num_bad_epochs}"
                    )
                if hasattr(scheduler, "_last_lr"):
                    old_last_lr = scheduler._last_lr
                    scheduler._last_lr = [new_lr] * len(opt.param_groups)
                    print(
                        f"  Reset scheduler._last_lr: {old_last_lr} -> {scheduler._last_lr}"
                    )

                print(f"✓ Learning rate override completed: {old_lr} → {new_lr}")
                print(f"✓ Scheduler state reset to use new learning rate")
                print(f"{'='*50}\n")

                # Update current_lr for tracking
                current_lr = new_lr
            else:
                print("No learning rate override needed or possible")
                print(f"{'='*60}\n")
                # No LR override, use merged args value
                current_lr = checkpoint_results.get("merged_args", args).lr

            # Restore additional training state if available
            if checkpoint_results["training_state"]:
                training_state = checkpoint_results["training_state"]
                if not lr_was_overridden:
                    # Only use checkpointed LR if not overridden
                    current_lr = training_state.get("current_lr", args.lr)
                epochs_since_lr_reduction = training_state.get(
                    "epochs_since_lr_reduction", 0
                )
                epochs_since_best_checkpoint = training_state.get(
                    "epochs_since_best_checkpoint", 0
                )
                lr_has_dropped = training_state.get("lr_has_dropped", False)
                rollback_count = training_state.get("rollback_count", 0)
                patience_counter = training_state.get("patience_counter", 0)
                print(
                    f"Restored training state - LR: {current_lr}, patience: {patience_counter}"
                )

            # Use merged arguments for any parameter overrides
            if checkpoint_results["merged_args"] != args:
                print("Using merged arguments (checkpoint + command line overrides)")
                args = checkpoint_results["merged_args"]

            print("Checkpoint resumption completed successfully")

        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting fresh training instead")
            start_epoch = 1

    if args.enable_checkpoint_rollback:
        rollback_patience_threshold = int(
            scheduler.patience * args.rollback_patience_factor
        )
        print(
            f"Checkpoint rollback enabled: will rollback after {rollback_patience_threshold} epochs without improvement post-LR reduction"
        )
        # Restore persisted rollback state after a crash (M-4 fix).
        if hasattr(args, "resume_from_checkpoint") and args.resume_from_checkpoint:
            _rollback_path = os.path.join(output_base, "saved_models", "best_rollback.pt")
            if os.path.exists(_rollback_path):
                try:
                    best_model_state = torch.load(_rollback_path, map_location="cpu", weights_only=False)
                    print(f"Restored best_rollback.pt from {_rollback_path} — rollback safety net is active")
                except Exception as _e:
                    print(f"Warning: could not load best_rollback.pt: {_e}")

    # Check for sampling-only mode
    if hasattr(args, "sampling_only") and args.sampling_only:
        if not args.external_checkpoint:
            raise ValueError(
                "sampling_only mode requires --external_checkpoint to be specified"
            )

        print("=" * 60)
        print("SAMPLING-ONLY MODE DETECTED")
        print("=" * 60)
        print(
            f"Skipping training, using external checkpoint: {args.external_checkpoint}"
        )

        # Run sampling and evaluation directly
        from training.sample_utils import run_sampling_and_evaluation

        results = run_sampling_and_evaluation(
            checkpoint_path=args.external_checkpoint,
            args=args,
            device=device,
            output_base=output_base,
            model_name=model_name,
            job_timestamp=job_timestamp,
        )

        print("Sampling and evaluation completed successfully!")
        if results:
            print(f"Results saved to: {results.get('output_dir', 'N/A')}")

        return

    max_epochs = 3 if is_smoke_test else args.epochs
    if args.max_batches_per_epoch is not None:
        # Use user-specified batch limit
        max_batches_per_epoch = args.max_batches_per_epoch
        if is_main_process():
            print(
                f"Using user-specified batch limit: {max_batches_per_epoch} batches per epoch"
            )
    else:
        # Use the train_loader dataset length for max batches
        max_batches_per_epoch = (
            len(train_loader) if hasattr(train_loader, "__len__") else float("inf")
        )
        if is_main_process():
            print(f"Using full dataset: {max_batches_per_epoch} batches per epoch")

    # Create epoch progress bar (only on main process)
    if is_main_process():
        epoch_pbar = tqdm(
            range(start_epoch, max_epochs + 1),
            desc="Training",
            unit="epoch",
            disable=is_smoke_test,
            leave=True,
            ncols=100,
        )
    else:
        epoch_pbar = range(start_epoch, max_epochs + 1)

    # Define generators for dual validation noise schedules
    FIXED_SEED_FOR_VAL_FIXED_NOISE = 1
    # Fixed seed for reproducible validation
    val_fixed_gen = torch.Generator(device=device).manual_seed(
        FIXED_SEED_FOR_VAL_FIXED_NOISE
    )

    # Protein tracking now handled at dataset level

    # Full-generation validation setup (runs once, only on main process)
    gen_val_cache = None
    if getattr(args, 'validation_with_full_generation', False) and is_main_process():
        gen_val_cache = setup_generation_validation(args, graph_builder_kwargs, device)

    # AF2 ratio decay schedule — counts completed validation cycles
    val_cycle = 0

    try:
        for ep in epoch_pbar:
            # Record epoch start time
            epoch_start_time = time.time()

            # Disable epoch timeout (originally 70 minutes for hybrid training, infinite for PDB-only)
            # User requested no epoch timeout, so always use infinite timeout
            epoch_timeout = float("inf")

            # Set epoch for distributed sampler
            if hasattr(train_loader, "set_epoch"):
                # Hybrid dataset: set epoch on hybrid dataloader
                train_loader.set_epoch(ep)

            # Training phase
            model.train()
            total_train_loss = 0
            total_train_accuracy = 0
            total_train_elec_loss = 0.0
            total_train_elec_count = 0
            total_train_geom_loss = 0.0
            total_train_geom_count = 0

            # Track gradient norms across epoch
            total_gradient_norm = 0
            max_gradient_norm = 0
            gradient_count = 0

            # Track prediction diversity across entire epoch
            epoch_predicted_classes = (
                []
            )  # Collect all predictions for diversity analysis
            epoch_pred_entropies = []  # Collect prediction entropies

            # Reset epoch protein tracking for dataset-level coordination
            if args.verbose:
                from data.unified_dataset import reset_global_protein_epoch

                reset_global_protein_epoch()
                print(
                    f"[DEBUG] EPOCH {ep}: Reset dataset-level global protein tracker for {args.num_workers} workers"
                )

            # Debug: Print verbose status at epoch start
            if ep == 1:  # Only print once
                print(
                    f"[DEBUG] EPOCH START: args.verbose={getattr(args, 'verbose', 'NOT_SET')}, is_main_process()={is_main_process()}"
                )
                print(f"[DEBUG] Using dataset-level protein tracking")

                # Memory check
                import psutil

                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                print(f"[DEBUG] Current memory usage: {memory_mb:.1f} MB")

            # Create batch progress bar for first epoch only OR for hybrid debug mode
            show_batch_progress = (ep == 1) or (
                args.ratio_af2_pdb != 0 and max_batches_per_epoch < 5
            )
            if show_batch_progress:
                batch_pbar = tqdm(
                    enumerate(train_loader),
                    desc=f"Epoch {ep} batches",
                    total=min(len(train_loader), max_batches_per_epoch),
                    unit="batch",
                    leave=False,
                    ncols=120,
                )
                batch_iterator = batch_pbar
            else:
                batch_iterator = enumerate(train_loader)

            # Hybrid debug mode: Track per-batch statistics for AF2-containing training
            if (
                args.ratio_af2_pdb != 0
                and max_batches_per_epoch < 5
                and is_main_process()
            ):
                hybrid_batch_stats = []

            # Track OOM batch skipping statistics
            oom_skip_count = 0
            total_batches_attempted = 0

            for batch_idx, batch in batch_iterator:
                if batch_idx >= max_batches_per_epoch:
                    break

                total_batches_attempted += 1

                # Unpack batch data (now includes time values from dataset and optionally DSSP targets)
                if len(batch) == 5:
                    # DSSP format: (data, y, mask, time_values, dssp_targets)
                    data, y, mask, time_values, dssp_targets = batch
                    # Convert time values to tensor for batch processing
                    if time_values[0] is not None:
                        t = torch.tensor(
                            [tv for tv in time_values if tv is not None], device=device
                        )
                        B = t.shape[0]
                    else:
                        raise Exception(
                            "None observed in time values. Check get_item, the collator etc to make sure the time gets sampled and passed correctly."
                        )
                elif len(batch) == 4:
                    if args.lambda_dssp_loss is not None:
                        if args.lambda_dssp_loss > 0:
                            raise Exception(
                                "DSSP targets are required but not provided."
                            )
                    # Legacy format: (data, y, mask, time_values)
                    data, y, mask, time_values = batch
                    dssp_targets = None  # No DSSP targets available
                    # Convert time values to tensor for batch processing
                    if time_values[0] is not None:
                        t = torch.tensor(
                            [tv for tv in time_values if tv is not None], device=device
                        )
                        B = t.shape[0]
                    else:
                        raise Exception(
                            "None observed in time values. Check get_item, the collator etc to make sure the time gets sampled and passed correctly."
                        )
                else:
                    raise Exception(
                        "Time values are missing. They should've been generated inside __get_item__() and passed to the DataLoader."
                    )

                # Check epoch timeout (mainly for hybrid training with AF2 components)
                if epoch_timeout != float("inf"):
                    current_time = time.time()
                    elapsed_time = current_time - epoch_start_time
                    if elapsed_time > epoch_timeout:
                        if is_main_process():
                            print(
                                f"Epoch timeout reached after {elapsed_time/60:.1f} minutes (limit: {epoch_timeout/60:.1f} minutes)"
                            )
                            print(f"Processed {batch_idx} batches before timeout")
                        break

                # Hybrid debug mode: Collect batch statistics for AF2-containing training
                if (
                    args.ratio_af2_pdb != 0
                    and max_batches_per_epoch < 5
                    and is_main_process()
                ):
                    B, N, K = y.shape
                    num_nodes = data.num_nodes
                    avg_seq_len = num_nodes / B if B > 0 else 0
                    batch_stat = {
                        "batch": batch_idx + 1,
                        "proteins": B,
                        "avg_length": avg_seq_len,
                        "total_nodes": num_nodes,
                    }
                    hybrid_batch_stats.append(batch_stat)
                    print(
                        f"Hybrid Batch {batch_idx + 1}/{max_batches_per_epoch}: {B} proteins, avg length: {avg_seq_len:.1f}, total nodes: {num_nodes}",
                        flush=True,
                    )

                data, y = data.to(device), y.to(device)
                if mask is not None:
                    mask = mask.to(device)

                # Move DSSP targets to device if available
                if dssp_targets is not None:
                    # Handle batch of DSSP targets (some may be None)
                    dssp_targets_batch = []
                    for dssp_target in dssp_targets:
                        if dssp_target is not None:
                            dssp_targets_batch.append(dssp_target.to(device))
                        else:
                            dssp_targets_batch.append(None)
                    dssp_targets = dssp_targets_batch

                # Track unique proteins in verbose mode using dataset-level tracking
                if args.verbose and is_main_process():
                    # Get protein counts from dataset-level global tracker
                    from data.unified_dataset import get_global_protein_counts

                    shared_epoch_count, shared_cumulative_count = (
                        get_global_protein_counts()
                    )

                    # Debug logging for first few batches
                    if batch_idx < 10:
                        print(
                            f"[DEBUG] UNION ACROSS ALL {args.num_workers} WORKERS - Batch {batch_idx}: epoch_unique={shared_epoch_count}, cumulative={shared_cumulative_count}"
                        )

                    # Use dataset-level counts for logging (will be used later in wandb logging)
                    local_epoch_unique_count = shared_epoch_count
                    local_cumulative_unique_count = shared_cumulative_count
                else:
                    # No verbose logging - use default values
                    local_epoch_unique_count = 0
                    local_cumulative_unique_count = 0

                B, N, K = y.shape

                # Check probability constraints and fix if needed
                y_sums = y.sum(dim=-1)  # Shape: [B, N]
                all_close = torch.allclose(y_sums, torch.ones_like(y_sums), atol=1e-5)
                if not all_close:

                    # Normalize each sequence in the batch
                    for b in range(B):
                        seq_sums = y[b].sum(dim=-1)  # Shape: [N]

                        # Find positions that don't sum to 1
                        invalid_positions = ~torch.isclose(
                            seq_sums, torch.ones_like(seq_sums), atol=1e-5
                        )

                        if invalid_positions.any():
                            # For positions with sum = 0, set to uniform distribution
                            zero_sum_positions = seq_sums == 0
                            if zero_sum_positions.any():
                                y[b, zero_sum_positions] = 1.0 / K

                            # For other positions, normalize to sum to 1
                            non_zero_invalid = invalid_positions & ~zero_sum_positions
                            if non_zero_invalid.any():
                                y[b, non_zero_invalid] = y[
                                    b, non_zero_invalid
                                ] / seq_sums[non_zero_invalid].unsqueeze(-1)

                # Use time values from dataset if available, otherwise sample in training loop
                if t is not None:
                    # Time values already sampled per protein in dataset - use them directly
                    # Ensure t is on the correct device
                    t = t.to(device)
                    if args.verbose:
                        print(
                            f"Using dataset-sampled time values: mean={t.mean():.3f}, std={t.std():.3f}"
                        )
                else:
                    raise Exception(
                        "t is none. __get_item__ didn't create it correctly or it wasn't passed correctly."
                    )

                # Sample from Dirichlet distribution, optionally capturing max probability for overfit monitoring
                dirichlet_multiplier = getattr(
                    args, "dirichlet_multiplier_training", 1.0
                )
                if args.run_mode == "overfit_on_one":
                    x_t, max_train_dirichlet_prob = sample_forward(
                        y,
                        t,
                        generator=None,
                        return_max_prob=True,
                        dirichlet_multiplier=dirichlet_multiplier,
                    )
                    # Log the max probability to monitor noise variation in overfit mode
                    if should_log_to_wandb(args, is_smoke_test):
                        wandb.log(
                            {f"train/max_dirichlet_prob": max_train_dirichlet_prob}
                        )

                else:
                    x_t = sample_forward(
                        y, t, generator=None, dirichlet_multiplier=dirichlet_multiplier
                    )  # Use global RNG for training diversity

                # Apply inpainting-style training if enabled (p_unconditional < 1.0)
                # This clamps random positions to clean (ground-truth) probabilities
                # to teach the model to use clean neighbor sequence information
                inpaint_mask = None
                p_unconditional = getattr(args, 'p_unconditional', 1.0)
                if p_unconditional < 1.0:
                    x_t, inpaint_mask = apply_inpainting_noise(
                        x_t=x_t,
                        y=y,
                        p_unconditional=p_unconditional,
                        inpaint_ratio_min=getattr(args, 'inpaint_ratio_min', 0.1),
                        inpaint_ratio_max=getattr(args, 'inpaint_ratio_max', 0.3),
                    )

                    # Log inpainting statistics occasionally
                    if should_log_to_wandb(args, is_smoke_test) and batch_idx % 50 == 0:
                        if inpaint_mask is not None:
                            inpaint_ratio_actual = inpaint_mask.float().mean().item()
                            num_inpaint_samples = (inpaint_mask.sum(dim=1) > 0).sum().item()
                            wandb.log({
                                "train/inpaint_ratio_actual": inpaint_ratio_actual,
                                "train/inpaint_positions_per_batch": inpaint_mask.sum().item(),
                                "train/inpaint_samples_in_batch": num_inpaint_samples,
                                "train/unconditional_samples_in_batch": B - num_inpaint_samples,
                            })

                # Zero gradients
                opt.zero_grad()

                # Forward pass - handle single head, DSSP-only, and multi-aux-head modes
                electrostatic_logits = None
                geom_topology_logits = None
                try:
                    _any_new_aux = (
                        getattr(args, 'lambda_electrostatic_loss', 0.0) > 0.0
                        or getattr(args, 'lambda_geom_topology_loss', 0.0) > 0.0
                    )
                    if _any_new_aux:
                        # Multi-aux mode: model returns a dict of named logit tensors
                        model_result = model(data, t, x_t)
                        if isinstance(model_result, dict):
                            position_logits      = model_result['sequence']
                            dssp_logits          = model_result.get('dssp')
                            electrostatic_logits = model_result.get('electrostatic')
                            geom_topology_logits = model_result.get('geom_topology')
                        else:
                            # Fallback if model doesn't return dict
                            position_logits = model_result[0] if isinstance(model_result, tuple) else model_result
                            dssp_logits = None
                            electrostatic_logits = None
                            geom_topology_logits = None
                    elif args.lambda_dssp_loss > 0.0 and dssp_targets is not None:
                        # DSSP-only path (unchanged)
                        model_result = model(data, t, x_t, return_dssp=True)
                        if isinstance(model_result, tuple) and len(model_result) == 2:
                            position_logits, dssp_logits = model_result
                        else:
                            position_logits = model(data, t, x_t, return_dssp=False)
                            dssp_logits = None
                        electrostatic_logits = None
                        geom_topology_logits = None
                    else:
                        # Sequence only
                        position_logits = model(data, t, x_t, return_dssp=False)
                        dssp_logits = None
                        electrostatic_logits = None
                        geom_topology_logits = None

                except torch.cuda.OutOfMemoryError as e:
                    # Handle CUDA OOM errors during forward pass
                    oom_skip_count += 1

                    # Clear GPU cache to free up memory
                    torch.cuda.empty_cache()

                    # Log the OOM error and batch skip
                    if is_main_process():
                        print(
                            f"\n[WARNING] CUDA Out of Memory during forward pass on batch {batch_idx + 1}/{max_batches_per_epoch}"
                        )
                        print(f"[WARNING] Error: {str(e)}")
                        print(
                            f"[WARNING] Skipping this batch and continuing training..."
                        )
                        print(
                            f"[WARNING] Total OOM batches skipped so far: {oom_skip_count}/{total_batches_attempted}"
                        )

                        # Log memory stats for debugging
                        if torch.cuda.is_available():
                            allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
                            reserved = torch.cuda.memory_reserved() / (1024**3)  # GB
                            print(
                                f"[WARNING] GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB"
                            )

                    # Log to wandb if enabled
                    if should_log_to_wandb(args, is_smoke_test):
                        wandb.log(
                            {
                                "train/oom_batch_skipped": 1,
                                "train/total_oom_skips": oom_skip_count,
                                "train/oom_skip_rate": oom_skip_count
                                / total_batches_attempted,
                                "epoch": ep,
                                "batch": batch_idx,
                            }
                        )

                    # Continue to next batch
                    continue

                # Apply virtual node masking if enabled
                inpaint_mask_masked = None  # Will be set if inpainting is active
                if args.use_virtual_node:
                    # Create virtual node mask to exclude virtual nodes from loss
                    real_node_mask = mask_virtual_nodes_from_batch(data, B)

                    # Apply geometry masking to exclude nodes with missing coordinates
                    # This now handles both virtual nodes and geometry-missing nodes
                    # position_logits comes from model output and is already in graph space
                    masking_result = apply_geometry_masking(
                        position_logits,
                        y,
                        data,
                        B,
                        K,
                        device,
                        args.use_virtual_node,
                        v_pred_is_graph_space=True,
                        inpaint_mask=inpaint_mask,
                    )
                    # Unpack results - may have 2 or 3 elements depending on whether inpaint_mask was provided
                    if inpaint_mask is not None:
                        position_logits_masked, y_masked, inpaint_mask_masked = masking_result
                    else:
                        position_logits_masked, y_masked = masking_result

                    # Extract uncertainty weights if using flexible loss scaling
                    uncertainty_weights_masked = None
                    if args.flexible_loss_scaling:
                        # Extract uncertainty from node scalar features before masking
                        uncertainty_weights = data.x_s[:, 6].clone()  # [total_nodes]

                        # Set virtual node uncertainty to 0 if virtual node is used
                        if args.use_virtual_node:
                            uncertainty_weights[-1] = (
                                0.0  # Virtual node is the last node
                            )

                        # Apply the same masking to weights as was applied to predictions
                        uncertainty_weights_masked = apply_same_masking_to_weights(
                            uncertainty_weights,
                            data,
                            args.use_virtual_node,
                            y=y,
                            B=B,
                            K=K,
                            device=device,
                        )

                        # Debug printing for first batch of first epoch
                        if args.debug_mode and ep == 1 and batch_idx == 0:
                            print(
                                f"\nFLEXIBLE LOSS SCALING DEBUG (Epoch {ep}, Batch {batch_idx}):"
                            )
                            print(f"  Original uncertainty weights (x_s[:, 6]):")
                            print(f"    Shape: {uncertainty_weights.shape}")
                            print(f"    Mean: {uncertainty_weights.mean().item():.6f}")
                            print(f"    Std: {uncertainty_weights.std().item():.6f}")
                            print(f"    Min: {uncertainty_weights.min().item():.6f}")
                            print(f"    Max: {uncertainty_weights.max().item():.6f}")
                            print(
                                f"    Sample values (first 20): {uncertainty_weights[:20].cpu().numpy()}"
                            )

                            if uncertainty_weights_masked is not None:
                                print(f"  Masked uncertainty weights:")
                                print(f"    Shape: {uncertainty_weights_masked.shape}")
                                print(
                                    f"    Mean: {uncertainty_weights_masked.mean().item():.6f}"
                                )
                                print(
                                    f"    Std: {uncertainty_weights_masked.std().item():.6f}"
                                )
                                print(
                                    f"    Min: {uncertainty_weights_masked.min().item():.6f}"
                                )
                                print(
                                    f"    Max: {uncertainty_weights_masked.max().item():.6f}"
                                )
                                print(
                                    f"    Sample values (first 20): {uncertainty_weights_masked[:20].cpu().numpy()}"
                                )

                                # Analyze weight distribution
                                weights_np = uncertainty_weights_masked.cpu().numpy()
                                center_count = (
                                    (weights_np > 0.4) & (weights_np < 0.6)
                                ).sum()
                                total_count = len(weights_np)
                                center_pct = center_count / total_count * 100
                                print(f"  Weight distribution analysis:")
                                print(
                                    f"    Weights in [0.4, 0.6]: {center_count}/{total_count} ({center_pct:.1f}%)"
                                )
                                print(
                                    f"    Coefficient of variation: {uncertainty_weights_masked.std().item() / uncertainty_weights_masked.mean().item():.3f}"
                                )
                            print(f"  End flexible loss scaling debug\n")
                else:

                    # No virtual nodes: still apply geometry masking for nodes with missing coordinates
                    # position_logits comes from model output and is already in graph space
                    masking_result = apply_geometry_masking(
                        position_logits,
                        y,
                        data,
                        B,
                        K,
                        device,
                        use_virtual_node=False,
                        v_pred_is_graph_space=True,
                        inpaint_mask=inpaint_mask,
                    )
                    # Unpack results - may have 2 or 3 elements depending on whether inpaint_mask was provided
                    if inpaint_mask is not None:
                        position_logits_masked, y_masked, inpaint_mask_masked = masking_result
                    else:
                        position_logits_masked, y_masked = masking_result

                    # Extract uncertainty weights if using flexible loss scaling
                    uncertainty_weights_masked = None
                    if args.flexible_loss_scaling:
                        # Extract uncertainty from node scalar features before masking
                        uncertainty_weights = data.x_s[:, 6].clone()  # [total_nodes]

                        # Apply the same masking to weights as was applied to predictions
                        uncertainty_weights_masked = apply_same_masking_to_weights(
                            uncertainty_weights,
                            data,
                            use_virtual_node=False,
                            y=y,
                            B=B,
                            K=K,
                            device=device,
                        )

                        # Debug printing for first batch of first epoch
                        if args.debug_mode and ep == 1 and batch_idx == 0:
                            print(
                                f"\nFLEXIBLE LOSS SCALING DEBUG (Epoch {ep}, Batch {batch_idx}) - No Virtual Nodes:"
                            )
                            print(f"  Original uncertainty weights (x_s[:, 6]):")
                            print(f"    Shape: {uncertainty_weights.shape}")
                            print(f"    Mean: {uncertainty_weights.mean().item():.6f}")
                            print(f"    Std: {uncertainty_weights.std().item():.6f}")
                            print(f"    Min: {uncertainty_weights.min().item():.6f}")
                            print(f"    Max: {uncertainty_weights.max().item():.6f}")
                            print(
                                f"    Sample values (first 20): {uncertainty_weights[:20].cpu().numpy()}"
                            )

                            if uncertainty_weights_masked is not None:
                                print(f"  Masked uncertainty weights:")
                                print(f"    Shape: {uncertainty_weights_masked.shape}")
                                print(
                                    f"    Mean: {uncertainty_weights_masked.mean().item():.6f}"
                                )
                                print(
                                    f"    Std: {uncertainty_weights_masked.std().item():.6f}"
                                )
                                print(
                                    f"    Min: {uncertainty_weights_masked.min().item():.6f}"
                                )
                                print(
                                    f"    Max: {uncertainty_weights_masked.max().item():.6f}"
                                )
                                print(
                                    f"    Sample values (first 20): {uncertainty_weights_masked[:20].cpu().numpy()}"
                                )

                                # Analyze weight distribution
                                weights_np = uncertainty_weights_masked.cpu().numpy()
                                center_count = (
                                    (weights_np > 0.4) & (weights_np < 0.6)
                                ).sum()
                                total_count = len(weights_np)
                                center_pct = center_count / total_count * 100
                                print(f"  Weight distribution analysis:")
                                print(
                                    f"    Weights in [0.4, 0.6]: {center_count}/{total_count} ({center_pct:.1f}%)"
                                )
                                print(
                                    f"    Coefficient of variation: {uncertainty_weights_masked.std().item() / uncertainty_weights_masked.mean().item():.3f}"
                                )
                            print(f"  End flexible loss scaling debug\n")

                # Compute ONLY categorical cross-entropy loss between predicted and target positions
                # position_logits_masked: [num_valid_nodes, K] - raw logits
                # y_masked: [num_valid_nodes, K] - target probability distributions (hard or smoothed)

                # Apply inpainting mask to exclude clamped positions from loss
                # inpaint_mask_masked: [num_valid_nodes] where True = clamped to clean (exclude from loss)
                if inpaint_mask_masked is not None and inpaint_mask_masked.any():
                    # Create a mask for positions that should contribute to loss (NOT clamped)
                    loss_position_mask = ~inpaint_mask_masked  # True = include in loss

                    # Filter predictions and targets to only include non-clamped positions
                    position_logits_masked = position_logits_masked[loss_position_mask]
                    y_masked = y_masked[loss_position_mask]

                    # Also filter uncertainty weights if using flexible loss scaling
                    if uncertainty_weights_masked is not None:
                        uncertainty_weights_masked = uncertainty_weights_masked[loss_position_mask]

                    # Log how many positions were excluded
                    if should_log_to_wandb(args, is_smoke_test) and batch_idx % 50 == 0:
                        num_clamped = inpaint_mask_masked.sum().item()
                        num_total = len(inpaint_mask_masked)
                        wandb.log({
                            "train/inpaint_positions_excluded_from_loss": num_clamped,
                            "train/inpaint_loss_positions_remaining": num_total - num_clamped,
                        })

                if (
                    args.flexible_loss_scaling
                    and uncertainty_weights_masked is not None
                ):
                    # Use weighted cross-entropy loss with support for smoothed labels
                    if args.use_smoothed_labels:
                        # Similarity-matrix smoothed labels: use KL divergence approach weighted by uncertainty
                        log_probs = F.log_softmax(
                            position_logits_masked.view(-1, K), dim=-1
                        )  # [num_valid_nodes, K]
                        per_sample_loss = -(y_masked.view(-1, K) * log_probs).sum(
                            dim=-1
                        )  # [num_valid_nodes]
                        weighted_loss = per_sample_loss * uncertainty_weights_masked
                        total_weights = uncertainty_weights_masked.sum()
                        cce_loss = (
                            weighted_loss.sum() / total_weights
                            if total_weights > 0
                            else per_sample_loss.mean()
                        )
                    elif args.label_smoothing_factor > 0.0:
                        # Uniform label smoothing with flexible loss scaling
                        ls_factor = args.label_smoothing_factor
                        log_probs = F.log_softmax(
                            position_logits_masked.view(-1, K), dim=-1
                        )  # [num_valid_nodes, K]
                        targets = y_masked.argmax(dim=-1)  # [num_valid_nodes]

                        # NLL for true class
                        nll = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)  # [N]

                        # Mean log-prob over incorrect classes
                        sum_log_probs = log_probs.sum(dim=-1)
                        true_log_prob = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
                        mean_log_incorrect = (sum_log_probs - true_log_prob) / (K - 1)
                        smooth_loss = -mean_log_incorrect

                        # Combined per-sample loss
                        per_sample_loss = (1.0 - ls_factor) * nll + ls_factor * smooth_loss

                        weighted_loss = per_sample_loss * uncertainty_weights_masked
                        total_weights = uncertainty_weights_masked.sum()
                        cce_loss = (
                            weighted_loss.sum() / total_weights
                            if total_weights > 0
                            else per_sample_loss.mean()
                        )
                    else:
                        # Hard labels: use standard cross-entropy weighted by uncertainty
                        targets = y_masked.argmax(dim=-1)  # [num_valid_nodes]
                        per_sample_loss = F.cross_entropy(
                            position_logits_masked.view(-1, K),
                            targets,
                            reduction="none",
                        )
                        weighted_loss = per_sample_loss * uncertainty_weights_masked
                        total_weights = uncertainty_weights_masked.sum()
                        cce_loss = (
                            weighted_loss.sum() / total_weights
                            if total_weights > 0
                            else per_sample_loss.mean()
                        )

                    # Debug: Compare weighted vs unweighted loss
                    if args.debug_mode and ep == 1 and batch_idx == 0:
                        # Compute standard unweighted loss for comparison
                        if args.use_smoothed_labels or args.label_smoothing_factor > 0.0:
                            standard_loss = per_sample_loss.mean()
                        else:
                            standard_loss = F.cross_entropy(
                                position_logits_masked.view(-1, K),
                                targets,
                                reduction="mean",
                            )

                        print(f"\nLOSS COMPUTATION COMPARISON:")
                        print(
                            f"  Standard loss (unweighted): {standard_loss.item():.6f}"
                        )
                        print(f"  Flexible loss (weighted):   {cce_loss.item():.6f}")
                        print(
                            f"  Absolute difference:        {abs(standard_loss.item() - cce_loss.item()):.6f}"
                        )
                        print(
                            f"  Relative difference:        {abs(standard_loss.item() - cce_loss.item()) / standard_loss.item() * 100:.4f}%"
                        )
                        print(
                            f"  Total weight sum:           {total_weights.item():.6f}"
                        )
                        print(
                            f"  Average weight:             {uncertainty_weights_masked.mean().item():.6f}"
                        )

                        # Show per-sample loss statistics
                        print(f"  Per-sample loss stats:")
                        print(f"    Mean: {per_sample_loss.mean().item():.6f}")
                        print(f"    Std:  {per_sample_loss.std().item():.6f}")
                        print(f"    Min:  {per_sample_loss.min().item():.6f}")
                        print(f"    Max:  {per_sample_loss.max().item():.6f}")
                        print(f"  End loss comparison debug\n")

                else:
                    # Standard loss path (unweighted or AA-frequency-weighted)
                    cond_flow = (
                        model.module if hasattr(model, "module") else model
                    ).cond_flow
                    aa_class_weights = (
                        FIXED_AA_CLASS_WEIGHTS if getattr(args, 'weighted_cce_loss', False) else None
                    )
                    cce_loss = cond_flow.cross_entropy_loss(
                        position_logits_masked.view(
                            -1, K
                        ),  # [num_valid_nodes, K] - raw logits
                        y_masked.view(
                            -1, K
                        ),  # [num_valid_nodes, K] - target distributions
                        use_smoothed_labels=args.use_smoothed_labels,  # Use similarity-matrix smoothed labels if enabled
                        label_smoothing_factor=args.label_smoothing_factor,  # Use uniform label smoothing if > 0
                        class_weights=aa_class_weights,  # None unless --weighted_cce_loss
                    )

                # Sequence recovery loss
                sequence_loss = cce_loss

                # DSSP secondary structure prediction loss (if enabled and data available)
                dssp_loss = None
                dssp_accuracy = None

                if (
                    args.lambda_dssp_loss > 0.0
                    and dssp_targets is not None
                    and dssp_logits is not None
                ):
                    # DSSP predictions were obtained in the main forward pass above
                    model_obj = model.module if hasattr(model, "module") else model
                    if hasattr(model_obj, "dssp_final"):
                        # Create proper DSSP targets tensor from batch
                        # dssp_targets is a list of tensors, need to concatenate appropriately
                        dssp_targets_tensor = []
                        for i, target in enumerate(dssp_targets):
                            if target is not None:
                                dssp_targets_tensor.append(target)
                            else:
                                # If DSSP target is None, create dummy targets (will be masked out)
                                # Get actual length from the batch mask instead of estimating
                                if hasattr(mask, "__len__") and len(mask) > i:
                                    dummy_len = (
                                        mask[i].sum().item()
                                    )  # Use actual sequence length from mask
                                elif hasattr(data, "batch") and hasattr(data, "ptr"):
                                    # Get length from batch pointers if available
                                    if i < len(data.ptr) - 1:
                                        dummy_len = data.ptr[i + 1] - data.ptr[i]
                                    else:
                                        dummy_len = data.batch.shape[0] - data.ptr[i]
                                else:
                                    # Fallback: estimate from total position logits
                                    dummy_len = position_logits.shape[0] // len(
                                        dssp_targets
                                    )

                                # Create dummy targets including virtual node placeholder when needed
                                # The apply_dssp_masking function will handle proper alignment, but we need
                                # to create targets for real nodes only (virtual nodes are added by masking function)
                                dummy_targets = torch.zeros(
                                    dummy_len, dtype=torch.long, device=device
                                )
                                dssp_targets_tensor.append(dummy_targets)

                        # Concatenate all DSSP targets
                        if dssp_targets_tensor:
                            dssp_targets_concat = torch.cat(dssp_targets_tensor, dim=0)
                        else:
                            # No valid DSSP targets in this batch
                            dssp_targets_concat = None

                        if dssp_targets_concat is not None:
                            # Apply proper DSSP masking using dedicated function
                            dssp_logits_masked, dssp_targets_masked = (
                                apply_dssp_masking(
                                    dssp_logits,
                                    dssp_targets_concat,
                                    data,
                                    B,
                                    device,
                                    args.use_virtual_node,
                                    debug=args.debug_mode,
                                )
                            )

                            # Compute DSSP loss using the model's method
                            dssp_loss = model_obj.compute_dssp_loss(
                                dssp_logits_masked, dssp_targets_masked
                            )

                            # Compute DSSP accuracy separately
                            with torch.no_grad():
                                dssp_pred_classes = dssp_logits_masked.argmax(dim=-1)
                                dssp_correct = (
                                    dssp_pred_classes == dssp_targets_masked
                                ).float()
                                dssp_accuracy = dssp_correct.mean().item()

                            if args.debug_mode and ep == 1 and batch_idx == 0:
                                print(f"\nDSSP LOSS DEBUG:")
                                print(
                                    f"  DSSP targets in batch: {sum(1 for t in dssp_targets if t is not None)}/{len(dssp_targets)}"
                                )
                                print(f"  DSSP logits shape: {dssp_logits.shape}")
                                print(
                                    f"  DSSP targets concat shape: {dssp_targets_concat.shape}"
                                )
                                print(
                                    f"  DSSP logits masked shape: {dssp_logits_masked.shape}"
                                )
                                print(
                                    f"  DSSP targets masked shape: {dssp_targets_masked.shape}"
                                )
                                print(f"  DSSP loss: {dssp_loss.item():.6f}")
                                print(f"  DSSP accuracy: {dssp_accuracy:.4f}")
                                print(f"  Lambda DSSP: {args.lambda_dssp_loss}")
                                print(f"  End DSSP debug\n")
                        else:
                            # No valid DSSP targets after processing
                            dssp_loss = torch.tensor(0.0, device=device)
                            dssp_accuracy = 0.0
                    else:
                        print(
                            f"WARNING: DSSP loss enabled but model has no dssp_final layer"
                        )
                        dssp_loss = torch.tensor(0.0, device=device)
                        dssp_accuracy = 0.0
                else:
                    dssp_loss = torch.tensor(0.0, device=device)
                    dssp_accuracy = 0.0

                # --- Electrostatic and geometric topology auxiliary losses ---
                # get_valid_node_mask applies the same three-category exclusion as
                # apply_geometry_masking (virtual nodes, geom_missing, XXX residues),
                # giving elec/geom logits aligned with y_masked (pre-inpaint-filter).
                # We then apply the inpainting exclusion to stay consistent with sequence loss.
                _aux_valid_mask = None  # computed once, reused for both heads
                electrostatic_loss = torch.tensor(0.0, device=device)
                if (
                    getattr(args, 'lambda_electrostatic_loss', 0.0) > 0.0
                    and electrostatic_logits is not None
                ):
                    model_obj = model.module if hasattr(model, 'module') else model
                    if hasattr(model_obj, 'compute_electrostatic_loss'):
                        if _aux_valid_mask is None:
                            _aux_valid_mask = get_valid_node_mask(y, data, B, K, device, args.use_virtual_node)
                        elec_logits_masked = electrostatic_logits[_aux_valid_mask]
                        # Apply inpainting exclusion (same positions removed from sequence loss)
                        if inpaint_mask_masked is not None and inpaint_mask_masked.any():
                            elec_logits_masked = elec_logits_masked[~inpaint_mask_masked]
                        aa_targets_for_elec = y_masked.argmax(dim=-1)
                        electrostatic_loss = model_obj.compute_electrostatic_loss(
                            elec_logits_masked, aa_targets_for_elec
                        )

                geom_topology_loss = torch.tensor(0.0, device=device)
                if (
                    getattr(args, 'lambda_geom_topology_loss', 0.0) > 0.0
                    and geom_topology_logits is not None
                ):
                    model_obj = model.module if hasattr(model, 'module') else model
                    if hasattr(model_obj, 'compute_geom_topology_loss'):
                        if _aux_valid_mask is None:
                            _aux_valid_mask = get_valid_node_mask(y, data, B, K, device, args.use_virtual_node)
                        geom_logits_masked = geom_topology_logits[_aux_valid_mask]
                        if inpaint_mask_masked is not None and inpaint_mask_masked.any():
                            geom_logits_masked = geom_logits_masked[~inpaint_mask_masked]
                        aa_targets_for_geom = y_masked.argmax(dim=-1)
                        geom_topology_loss = model_obj.compute_geom_topology_loss(
                            geom_logits_masked, aa_targets_for_geom
                        )

                # Total multitask loss
                total_loss = (
                    sequence_loss
                    + args.lambda_dssp_loss * dssp_loss
                    + getattr(args, 'lambda_electrostatic_loss', 0.0) * electrostatic_loss
                    + getattr(args, 'lambda_geom_topology_loss', 0.0) * geom_topology_loss
                )

                # Check if any data remains after masking
                if position_logits_masked.numel() == 0:
                    print(
                        f"      ERROR: all data masked out! position_logits_masked is empty"
                    )
                    print(f"      Debug info:")
                    print(f"        position_logits shape: {position_logits.shape}")
                    print(f"        y shape: {y.shape}")
                    print(f"        batch size: {B}")
                    if args.use_virtual_node:
                        real_node_mask = mask_virtual_nodes_from_batch(data, B)
                        print(
                            f"        virtual nodes: {(~real_node_mask).sum().item()} out of {real_node_mask.size(0)}"
                        )
                    if hasattr(data, "geom_missing") and data.geom_missing is not None:
                        print(
                            f"        geometry missing nodes: {data.geom_missing.sum().item()}"
                        )

                    # This should not happen - there's a bug in masking logic
                    raise ValueError(
                        "All data was masked out - this indicates a bug in the masking logic"
                    )

                # Check for anomalous loss before backward pass
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    error_msg = f"Anomalous loss detected at epoch {ep}, batch {batch_idx}: {total_loss.item()}"
                    raise ValueError(error_msg)

                # Compute training accuracy (excluding virtual nodes)
                # Get predicted classes by taking argmax of position logits
                predicted_classes = position_logits_masked.argmax(
                    dim=-1
                )  # [total_real_nodes]
                # Convert one-hot targets to class indices for accuracy calculation
                target_classes = y_masked.argmax(dim=-1)  # [total_real_nodes]
                correct_predictions = (predicted_classes == target_classes).float()
                train_accuracy = (
                    correct_predictions.mean().item()
                )  # Average accuracy for this batch

                # Collect predictions for epoch-level diversity analysis
                epoch_predicted_classes.append(predicted_classes.cpu())
                if predicted_classes.numel() > 0:
                    pred_probs = torch.softmax(position_logits_masked, dim=-1)
                    pred_entropy = (
                        -(pred_probs * torch.log(pred_probs + 1e-8))
                        .sum(dim=-1)
                        .mean()
                        .item()
                    )
                    epoch_pred_entropies.append(pred_entropy)

                # Backward pass with OOM handling
                try:
                    total_loss.backward()
                except torch.cuda.OutOfMemoryError as e:
                    # Handle CUDA OOM errors during backward pass
                    oom_skip_count += 1

                    # Clear GPU cache to free up memory
                    torch.cuda.empty_cache()

                    # Log the OOM error and batch skip
                    if is_main_process():
                        print(
                            f"\n[WARNING] CUDA Out of Memory during backward pass on batch {batch_idx + 1}/{max_batches_per_epoch}"
                        )
                        print(f"[WARNING] Error: {str(e)}")
                        print(
                            f"[WARNING] Skipping this batch and continuing training..."
                        )
                        print(
                            f"[WARNING] Total OOM batches skipped so far: {oom_skip_count}/{total_batches_attempted}"
                        )

                        # Log memory stats for debugging
                        if torch.cuda.is_available():
                            allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
                            reserved = torch.cuda.memory_reserved() / (1024**3)  # GB
                            print(
                                f"[WARNING] GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB"
                            )

                    # Log to wandb if enabled
                    if should_log_to_wandb(args, is_smoke_test):
                        wandb.log(
                            {
                                "train/oom_batch_skipped": 1,
                                "train/total_oom_skips": oom_skip_count,
                                "train/oom_skip_rate": oom_skip_count
                                / total_batches_attempted,
                                "epoch": ep,
                                "batch": batch_idx,
                            }
                        )

                    # Continue to next batch
                    continue

                # Gradient clipping
                if hasattr(args, "grad_clip") and args.grad_clip > 0:
                    # This single call computes grad norm AND clips - much faster than manual calculation
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=args.grad_clip
                    )

                    if batch_idx < 3:
                        print(
                            f"    Gradient clipping applied: norm={grad_norm:.6f} (max_norm={args.grad_clip})"
                        )

                    # Warn about large gradients
                    if grad_norm > 100:
                        print(f"WARNING: Large gradient norm detected: {grad_norm:.2f}")
                else:
                    # No clipping - just compute norm for logging if needed
                    if (
                        should_log_to_wandb(args, is_smoke_test)
                        and args.run_mode == "overfit_on_one"
                    ):
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_norm=float("inf")
                        )
                    else:
                        grad_norm = 0.0  # Skip computation if not needed

                # Log gradient norm to wandb (only in overfit mode for debugging single protein)
                should_log_gradient = (
                    should_log_to_wandb(args, is_smoke_test)
                    and args.run_mode == "overfit_on_one"
                )
                if should_log_gradient:
                    print(f"    Logging gradients to wandb: norm={grad_norm:.6f}")
                    wandb.log(
                        {
                            "train/gradient_norm": grad_norm,
                            "epoch": ep,
                            "batch": batch_idx,
                        }
                    )

                # Accumulate gradient statistics for epoch logging
                total_gradient_norm += grad_norm
                max_gradient_norm = max(max_gradient_norm, grad_norm)
                gradient_count += 1

                # LR warmup (active only when resuming without optimizer state, H-1 fix)
                if _lr_warmup_active:
                    _lr_warmup_step += 1
                    warmup_lr = _lr_warmup_target * (_lr_warmup_step / _lr_warmup_steps)
                    for pg in opt.param_groups:
                        pg["lr"] = warmup_lr
                    if _lr_warmup_step >= _lr_warmup_steps:
                        _lr_warmup_active = False
                        print(f"LR warmup complete after {_lr_warmup_steps} steps, LR={_lr_warmup_target:.2e}")

                opt.step()

                total_train_loss += total_loss.item()
                total_train_accuracy += train_accuracy  # Accumulate training accuracy

                # Accumulate aux head losses for epoch-level logging
                if getattr(args, 'lambda_electrostatic_loss', 0.0) > 0.0:
                    total_train_elec_loss += electrostatic_loss.item()
                    total_train_elec_count += 1
                if getattr(args, 'lambda_geom_topology_loss', 0.0) > 0.0:
                    total_train_geom_loss += geom_topology_loss.item()
                    total_train_geom_count += 1

                # Update batch progress bar with metrics
                if show_batch_progress:
                    pbar_dict = {
                        "loss": f"{total_loss.item():.4f}",
                        "seq": f"{sequence_loss.item():.4f}",
                        "acc": f"{train_accuracy:.3f}",
                    }
                    if args.lambda_dssp_loss > 0.0 and dssp_loss is not None:
                        pbar_dict["dssp"] = f"{dssp_loss.item():.4f}"
                        pbar_dict["dssp_acc"] = f"{dssp_accuracy:.3f}"
                    if getattr(args, 'lambda_electrostatic_loss', 0.0) > 0.0:
                        pbar_dict["elec"] = f"{electrostatic_loss.item():.4f}"
                    if getattr(args, 'lambda_geom_topology_loss', 0.0) > 0.0:
                        pbar_dict["geom"] = f"{geom_topology_loss.item():.4f}"
                    batch_pbar.set_postfix(pbar_dict)

                # Log batch-level training loss (only in verbose mode)
                # Console logging
                if is_main_process() and args.verbose:
                    log_msg = (
                        f"Epoch {ep} Batch {batch_idx + 1}: loss={total_loss.item():.6f}, "
                        f"seq={sequence_loss.item():.6f}, acc={train_accuracy:.4f}"
                    )
                    if args.lambda_dssp_loss > 0.0 and dssp_loss is not None:
                        log_msg += f", dssp={dssp_loss.item():.6f}, dssp_acc={dssp_accuracy:.4f}"
                    print(log_msg)

                # Wandb logging (if enabled)
                if should_log_to_wandb(args, is_smoke_test):
                    wandb_batch_dict = {
                        "train/batch_loss": total_loss.item(),
                        "train/batch_sequence_loss": sequence_loss.item(),
                        "train/batch_accuracy": train_accuracy,
                        "train/learning_rate": opt.param_groups[0]["lr"],
                        "epoch": ep,
                        "batch": batch_idx,
                    }

                    # Add DSSP metrics if enabled
                    if args.lambda_dssp_loss > 0.0 and dssp_loss is not None:
                        wandb_batch_dict.update(
                            {
                                "train/batch_dssp_loss": dssp_loss.item(),
                                "train/batch_dssp_accuracy": dssp_accuracy,
                                "train/lambda_dssp_loss": args.lambda_dssp_loss,
                            }
                        )

                    # Add electrostatic metrics if enabled
                    if getattr(args, 'lambda_electrostatic_loss', 0.0) > 0.0:
                        wandb_batch_dict['train/batch_electrostatic_loss'] = electrostatic_loss.item()

                    # Add geometric topology metrics if enabled
                    if getattr(args, 'lambda_geom_topology_loss', 0.0) > 0.0:
                        wandb_batch_dict['train/batch_geom_topology_loss'] = geom_topology_loss.item()

                    # Add unique protein tracking in verbose mode
                    if args.verbose and is_main_process():
                        protein_count = (
                            B  # Use batch size since we track proteins at dataset level
                        )

                        # Use coordinated counts if available, otherwise use 0 (dataset-level tracking handles this)
                        if (
                            "local_epoch_unique_count" in locals()
                            and "local_cumulative_unique_count" in locals()
                        ):
                            epoch_count_to_log = local_epoch_unique_count
                            cumulative_count_to_log = local_cumulative_unique_count
                        else:
                            # Fallback - protein tracking happens at dataset level
                            epoch_count_to_log = 0
                            cumulative_count_to_log = 0

                        wandb_batch_dict.update(
                            {
                                "train/epoch_unique_proteins_union": epoch_count_to_log,
                                "train/cumulative_unique_proteins_union": cumulative_count_to_log,
                                "train/batch_protein_count": protein_count,
                            }
                        )

                        # Debug logging for first few batches
                        if batch_idx < 3:
                            mode = (
                                f"UNION-ACROSS-{args.num_workers}-WORKERS"
                                if args.num_workers > 1
                                else "SINGLE-WORKER"
                            )
                            print(
                                f"[DEBUG] {mode} WandB logging: epoch_unique={epoch_count_to_log}, "
                                f"cumulative={cumulative_count_to_log}, batch_count={protein_count}"
                            )
                            print(
                                f"[DEBUG] Using dataset-level protein tracking, batch_count=B={B}"
                            )
                            print(
                                f"[DEBUG] wandb_batch_dict keys: {list(wandb_batch_dict.keys())}"
                            )

                    wandb.log(wandb_batch_dict)

                # Additional dedicated logging for unique protein tracking (verbose mode)
                if (
                    args.verbose
                    and is_main_process()
                    and should_log_to_wandb(args, is_smoke_test)
                ):
                    protein_count = (
                        B  # Use batch size since we track proteins at dataset level
                    )

                    # Use coordinated counts if available, otherwise use 0 (dataset-level tracking handles this)
                    if (
                        "local_epoch_unique_count" in locals()
                        and "local_cumulative_unique_count" in locals()
                    ):
                        epoch_count_to_log = local_epoch_unique_count
                        cumulative_count_to_log = local_cumulative_unique_count
                    else:
                        # Fallback - protein tracking happens at dataset level
                        epoch_count_to_log = 0
                        cumulative_count_to_log = 0

                    # Debug for first few batches
                    if batch_idx < 3:
                        print(
                            f"[DEBUG] Logging unique proteins to WandB - args.verbose={args.verbose}, "
                            f"is_main_process={is_main_process()}, should_log={should_log_to_wandb(args, is_smoke_test)}"
                        )
                        print(
                            f"[DEBUG] wandb module available: {'wandb' in globals()}, "
                            f"wandb.run: {wandb.run if 'wandb' in globals() else 'not available'}"
                        )

                    try:
                        wandb.log(
                            {
                                "verbose/epoch_unique_proteins_union": epoch_count_to_log,
                                "verbose/cumulative_unique_proteins_union": cumulative_count_to_log,
                                "verbose/batch_protein_count": protein_count,
                                "epoch": ep,
                                "batch": batch_idx,
                            }
                        )

                        if batch_idx < 3:
                            print(
                                f"[DEBUG] Successfully logged to wandb: epoch_unique={epoch_count_to_log}, "
                                f"cumulative_unique={cumulative_count_to_log}, batch_count={protein_count}"
                            )
                    except Exception as e:
                        print(f"[ERROR] Failed to log to wandb: {e}")
                        import traceback

                        traceback.print_exc()

            # Close batch progress bar if it exists
            if show_batch_progress:
                batch_pbar.close()

            # Compute average training metrics for this epoch
            avg_train_loss = (
                total_train_loss / len(train_loader) if len(train_loader) > 0 else 0
            )
            avg_train_accuracy = (
                total_train_accuracy / len(train_loader) if len(train_loader) > 0 else 0
            )

            # Compute average gradient norms for the epoch
            avg_gradient_norm = (
                total_gradient_norm / gradient_count if gradient_count > 0 else 0
            )

            # Report OOM batch skip statistics for this epoch
            if is_main_process() and (
                oom_skip_count > 0 or total_batches_attempted > 0
            ):
                oom_skip_rate = (
                    oom_skip_count / total_batches_attempted
                    if total_batches_attempted > 0
                    else 0
                )
                print(f"\n[INFO] Epoch {ep} OOM Statistics:")
                print(f"[INFO] - Total batches attempted: {total_batches_attempted}")
                print(f"[INFO] - OOM batches skipped: {oom_skip_count}")
                print(f"[INFO] - OOM skip rate: {oom_skip_rate:.2%}")
                print(
                    f"[INFO] - Successfully processed batches: {total_batches_attempted - oom_skip_count}"
                )

            # No early exit for overfit mode - process entire validation set

            # Close batch progress bar if it was created
            if show_batch_progress:
                batch_pbar.close()

            # Calculate average training losses
            num_train_batches = min(len(train_loader), max_batches_per_epoch)

            avg_train_loss = total_train_loss / num_train_batches
            avg_train_accuracy = total_train_accuracy / num_train_batches
            avg_train_elec_loss = total_train_elec_loss / total_train_elec_count if total_train_elec_count > 0 else 0.0
            avg_train_geom_loss = total_train_geom_loss / total_train_geom_count if total_train_geom_count > 0 else 0.0

            # Calculate average gradient norms
            avg_gradient_norm = (
                total_gradient_norm / gradient_count if gradient_count > 0 else 0.0
            )

            # Analyze epoch-level prediction diversity
            if epoch_predicted_classes:
                all_predictions = torch.cat(epoch_predicted_classes, dim=0)
                epoch_unique_predictions = torch.unique(all_predictions).numel()
                epoch_pred_counts = torch.bincount(
                    all_predictions, minlength=21
                )  # 21 amino acids (20 standard + X)
                epoch_most_frequent_ratio = (
                    epoch_pred_counts.max().item() / all_predictions.numel()
                )
                epoch_avg_entropy = (
                    np.mean(epoch_pred_entropies) if epoch_pred_entropies else 0.0
                )

            else:
                raise ValueError(
                    "No predictions collected during training epoch - this indicates a bug in the training loop."
                )

            # Multiple Validation Phases - runs at every epoch
            model.eval()

            # Run validation at t=0 (noisy data) - used for LR scheduling and model selection
            tqdm.write(f"    Running val_t0 (t=0)")
            val_t0_metrics = run_validation_phase(
                model=model,
                val_loader=val_fixed_loader,
                val_generator=val_fixed_gen,
                args=args,
                device=device,
                max_batches_per_epoch=max_batches_per_epoch,
                K=K,
                val_name="val_t0",
                is_smoke_test=is_smoke_test,
                current_epoch=ep,
                fixed_time=0.0,
            )

            # Run validation at t=2 (low noise)
            tqdm.write(f"    Running val_t2 (t=2)")
            val_t2_metrics = run_validation_phase(
                model=model,
                val_loader=val_fixed_loader,
                val_generator=val_fixed_gen,
                args=args,
                device=device,
                max_batches_per_epoch=max_batches_per_epoch,
                K=K,
                val_name="val_t2",
                is_smoke_test=is_smoke_test,
                current_epoch=ep,
                fixed_time=2.0,
            )

            # Run validation at t=4 (medium noise)
            tqdm.write(f"    Running val_t4 (t=4)")
            val_t4_metrics = run_validation_phase(
                model=model,
                val_loader=val_fixed_loader,
                val_generator=val_fixed_gen,
                args=args,
                device=device,
                max_batches_per_epoch=max_batches_per_epoch,
                K=K,
                val_name="val_t4",
                is_smoke_test=is_smoke_test,
                current_epoch=ep,
                fixed_time=4.0,
            )

            # Run validation at t=6 (high noise)
            tqdm.write(f"    Running val_t6 (t=6)")
            val_t6_metrics = run_validation_phase(
                model=model,
                val_loader=val_fixed_loader,
                val_generator=val_fixed_gen,
                args=args,
                device=device,
                max_batches_per_epoch=max_batches_per_epoch,
                K=K,
                val_name="val_t6",
                is_smoke_test=is_smoke_test,
                current_epoch=ep,
                fixed_time=6.0,
            )

            # Run validation on unfixed noise schedule (used for regularization analysis)
            tqdm.write(f"    Running val_unfixed (variable t)")
            val_unfixed_metrics = run_validation_phase(
                model=model,
                val_loader=val_unfixed_loader,
                val_generator=None,  # Use None instead of fixed generator to get fresh random samples like training
                args=args,
                device=device,
                max_batches_per_epoch=max_batches_per_epoch,
                K=K,
                val_name="val_unfixed",
                is_smoke_test=is_smoke_test,
                current_epoch=ep,
            )

            # Full-generation validation (only main process, only when cache is ready)
            gen_val_metrics = None
            if gen_val_cache is not None and ep % args.val_gen_freq == 0:
                tqdm.write(f"    Running full-generation validation ({len(gen_val_cache)} proteins, {args.val_gen_steps} steps)")
                gen_val_metrics = run_generation_validation(
                    model=model,
                    gen_val_cache=gen_val_cache,
                    args=args,
                    device=device,
                )
                tqdm.write(
                    f"      GenVal - Mean recovery: {gen_val_metrics['mean_recovery']:.2f}%  "
                    f"Median: {gen_val_metrics['median_recovery']:.2f}%  "
                    f"Std: {gen_val_metrics['std_recovery']:.2f}%  "
                    f"(n={gen_val_metrics['n_proteins']})"
                )
                # Compute smoothed val_gen score for use as primary metric
                if use_val_gen_metric:
                    raw_score = gen_val_metrics['mean_recovery'] + gen_val_metrics['median_recovery']
                    k = getattr(args, 'num_val_cycles_to_smooth_metric_over', 5)
                    if len(val_gen_score_history) == 0:
                        smoothed_val_gen_score = raw_score
                    else:
                        history_window = val_gen_score_history[-(k - 1):]
                        smoothed_val_gen_score = 0.5 * raw_score + 0.5 * (sum(history_window) / len(history_window))
                    val_gen_score_history.append(raw_score)
                    tqdm.write(
                        f"      GenVal smoothed score: {smoothed_val_gen_score:.2f} "
                        f"(raw: {raw_score:.2f}, history n={len(val_gen_score_history)})"
                    )

            # Extract metrics for learning rate scheduling (use val_fixed at t=0)
            avg_val_fixed_loss = val_t0_metrics[
                "avg_loss"
            ]  # Use t=0 validation as primary fixed validation
            avg_val_fixed_accuracy = val_t0_metrics["avg_accuracy"]
            val_fixed_diversity_metrics = val_t0_metrics[
                "diversity_metrics"
            ]  # For backward compatibility
            val_t0_diversity_metrics = val_t0_metrics[
                "diversity_metrics"
            ]  # For console logging
            val_fixed_epoch_avg_metrics = val_t0_metrics.get("epoch_avg_metrics", {})

            # Extract DSSP metrics for t=0 (val_fixed)
            avg_val_fixed_dssp_loss = val_t0_metrics.get("avg_dssp_loss", 0.0)
            avg_val_fixed_dssp_accuracy = val_t0_metrics.get("avg_dssp_accuracy", 0.0)
            val_fixed_dssp_batch_count = val_t0_metrics.get("dssp_batch_count", 0)
            # Extract aux head metrics for val_fixed
            avg_val_fixed_elec_loss = val_t0_metrics.get("avg_electrostatic_loss", 0.0)
            avg_val_fixed_elec_accuracy = val_t0_metrics.get("avg_electrostatic_accuracy", 0.0)
            avg_val_fixed_geom_loss = val_t0_metrics.get("avg_geom_topology_loss", 0.0)
            avg_val_fixed_geom_accuracy = val_t0_metrics.get("avg_geom_topology_accuracy", 0.0)

            # Extract metrics for additional time points
            avg_val_t2_loss = val_t2_metrics["avg_loss"]
            avg_val_t2_accuracy = val_t2_metrics["avg_accuracy"]
            val_t2_diversity_metrics = val_t2_metrics["diversity_metrics"]

            # Extract DSSP metrics for t=2
            avg_val_t2_dssp_loss = val_t2_metrics.get("avg_dssp_loss", 0.0)
            avg_val_t2_dssp_accuracy = val_t2_metrics.get("avg_dssp_accuracy", 0.0)
            # Extract aux head metrics for val_t2
            avg_val_t2_elec_loss = val_t2_metrics.get("avg_electrostatic_loss", 0.0)
            avg_val_t2_elec_accuracy = val_t2_metrics.get("avg_electrostatic_accuracy", 0.0)
            avg_val_t2_geom_loss = val_t2_metrics.get("avg_geom_topology_loss", 0.0)
            avg_val_t2_geom_accuracy = val_t2_metrics.get("avg_geom_topology_accuracy", 0.0)

            avg_val_t4_loss = val_t4_metrics["avg_loss"]
            avg_val_t4_accuracy = val_t4_metrics["avg_accuracy"]
            val_t4_diversity_metrics = val_t4_metrics["diversity_metrics"]

            # Extract DSSP metrics for t=4
            avg_val_t4_dssp_loss = val_t4_metrics.get("avg_dssp_loss", 0.0)
            avg_val_t4_dssp_accuracy = val_t4_metrics.get("avg_dssp_accuracy", 0.0)
            # Extract aux head metrics for val_t4
            avg_val_t4_elec_loss = val_t4_metrics.get("avg_electrostatic_loss", 0.0)
            avg_val_t4_elec_accuracy = val_t4_metrics.get("avg_electrostatic_accuracy", 0.0)
            avg_val_t4_geom_loss = val_t4_metrics.get("avg_geom_topology_loss", 0.0)
            avg_val_t4_geom_accuracy = val_t4_metrics.get("avg_geom_topology_accuracy", 0.0)

            avg_val_t6_loss = val_t6_metrics["avg_loss"]
            avg_val_t6_accuracy = val_t6_metrics["avg_accuracy"]
            val_t6_diversity_metrics = val_t6_metrics["diversity_metrics"]

            # Extract DSSP metrics for t=6
            avg_val_t6_dssp_loss = val_t6_metrics.get("avg_dssp_loss", 0.0)
            avg_val_t6_dssp_accuracy = val_t6_metrics.get("avg_dssp_accuracy", 0.0)
            # Extract aux head metrics for val_t6
            avg_val_t6_elec_loss = val_t6_metrics.get("avg_electrostatic_loss", 0.0)
            avg_val_t6_elec_accuracy = val_t6_metrics.get("avg_electrostatic_accuracy", 0.0)
            avg_val_t6_geom_loss = val_t6_metrics.get("avg_geom_topology_loss", 0.0)
            avg_val_t6_geom_accuracy = val_t6_metrics.get("avg_geom_topology_accuracy", 0.0)

            # Extract unfixed metrics for comparison
            avg_val_unfixed_loss = val_unfixed_metrics["avg_loss"]
            avg_val_unfixed_accuracy = val_unfixed_metrics["avg_accuracy"]
            val_unfixed_diversity_metrics = val_unfixed_metrics["diversity_metrics"]
            val_unfixed_epoch_avg_metrics = val_unfixed_metrics.get(
                "epoch_avg_metrics", {}
            )

            # Extract DSSP metrics for val_unfixed
            avg_val_unfixed_dssp_loss = val_unfixed_metrics.get("avg_dssp_loss", 0.0)
            avg_val_unfixed_dssp_accuracy = val_unfixed_metrics.get(
                "avg_dssp_accuracy", 0.0
            )
            # Extract aux head metrics for val_unfixed
            avg_val_unfixed_elec_loss = val_unfixed_metrics.get("avg_electrostatic_loss", 0.0)
            avg_val_unfixed_elec_accuracy = val_unfixed_metrics.get("avg_electrostatic_accuracy", 0.0)
            avg_val_unfixed_geom_loss = val_unfixed_metrics.get("avg_geom_topology_loss", 0.0)
            avg_val_unfixed_geom_accuracy = val_unfixed_metrics.get("avg_geom_topology_accuracy", 0.0)

            # Calculate train/val discrepancy metrics for regularization analysis
            train_val_fixed_gap = abs(avg_train_loss - avg_val_fixed_loss)
            train_val_unfixed_gap = abs(avg_train_loss - avg_val_unfixed_loss)
            val_fixed_unfixed_gap = abs(avg_val_fixed_loss - avg_val_unfixed_loss)

            # Calculate combined validation metric (average of fixed and unfixed)
            avg_val_combined_loss = (avg_val_fixed_loss + avg_val_unfixed_loss) / 2.0
            avg_val_combined_accuracy = (
                avg_val_fixed_accuracy + avg_val_unfixed_accuracy
            ) / 2.0

            # Calculate comprehensive validation metric (average of all 5 validation losses)
            avg_val_all_loss = (
                avg_val_fixed_loss
                + avg_val_unfixed_loss
                + avg_val_t2_loss
                + avg_val_t4_loss
                + avg_val_t6_loss
            ) / 5.0
            avg_val_all_accuracy = (
                avg_val_fixed_accuracy
                + avg_val_unfixed_accuracy
                + avg_val_t2_accuracy
                + avg_val_t4_accuracy
                + avg_val_t6_accuracy
            ) / 5.0

            # Select validation metric based on configuration
            def get_validation_metric(
                args,
                train_loss,
                val_fixed_loss,
                val_unfixed_loss,
                val_combo_loss,
                val_all_loss,
            ):
                """Select the appropriate validation metric based on args.val_metric configuration."""
                if args.val_metric == "training":
                    return train_loss, "training"
                elif args.val_metric == "val_fixed":
                    return val_fixed_loss, "val_fixed"
                elif args.val_metric == "val_unfixed":
                    return val_unfixed_loss, "val_unfixed"
                elif args.val_metric == "val_combo":
                    return val_combo_loss, "val_combo"
                elif args.val_metric == "val_all":
                    return val_all_loss, "val_all"
                else:
                    # Fallback to val_fixed for safety
                    return val_fixed_loss, "val_fixed"

            # Get the primary validation metric for early stopping, LR scheduling, and rollback
            if use_val_gen_metric:
                # val_gen metric: use smoothed score if val_gen ran this epoch, else None (skip)
                if gen_val_metrics is not None:
                    primary_val_loss = smoothed_val_gen_score
                    primary_val_name = "val_gen_smoothed"
                else:
                    primary_val_loss = None  # sentinel: skip scheduler/rollback/early-stop this epoch
                    primary_val_name = "val_gen_smoothed"
            else:
                primary_val_loss, primary_val_name = get_validation_metric(
                    args,
                    avg_train_loss,
                    avg_val_fixed_loss,
                    avg_val_unfixed_loss,
                    avg_val_combined_loss,
                    avg_val_all_loss,
                )

            # Get the corresponding accuracy metric
            if args.val_metric == "training":
                primary_val_accuracy = avg_train_accuracy
            elif args.val_metric == "val_fixed":
                primary_val_accuracy = avg_val_fixed_accuracy
            elif args.val_metric == "val_unfixed":
                primary_val_accuracy = avg_val_unfixed_accuracy
            elif args.val_metric == "val_combo":
                primary_val_accuracy = (
                    avg_val_fixed_accuracy + avg_val_unfixed_accuracy
                ) / 2.0
            elif args.val_metric == "val_all":
                primary_val_accuracy = avg_val_all_accuracy
            elif args.val_metric == "val_gen":
                primary_val_accuracy = gen_val_metrics['mean_recovery'] / 100.0 if gen_val_metrics is not None else 0.0
            else:
                primary_val_accuracy = avg_val_fixed_accuracy

            # Print validation comparison
            should_print = is_main_process()

            if should_print:
                tqdm.write(f"    Validation Results:")
                tqdm.write(
                    f"      val_fixed   - Loss: {avg_val_fixed_loss:.6f}, Acc: {avg_val_fixed_accuracy:.4f}"
                )
                tqdm.write(
                    f"      val_unfixed - Loss: {avg_val_unfixed_loss:.6f}, Acc: {avg_val_unfixed_accuracy:.4f}"
                )
                tqdm.write(
                    f"      val_t2      - Loss: {avg_val_t2_loss:.6f}, Acc: {avg_val_t2_accuracy:.4f}"
                )
                tqdm.write(
                    f"      val_t4      - Loss: {avg_val_t4_loss:.6f}, Acc: {avg_val_t4_accuracy:.4f}"
                )
                tqdm.write(
                    f"      val_t6      - Loss: {avg_val_t6_loss:.6f}, Acc: {avg_val_t6_accuracy:.4f}"
                )
                tqdm.write(
                    f"      val_combined- Loss: {avg_val_combined_loss:.6f}, Acc: {avg_val_combined_accuracy:.4f}"
                )
                tqdm.write(
                    f"      val_all     - Loss: {avg_val_all_loss:.6f}, Acc: {avg_val_all_accuracy:.4f}"
                )
                tqdm.write(
                    f"      PRIMARY ({primary_val_name}): "
                    + (f"{primary_val_loss:.6f}" if primary_val_loss is not None else "N/A (skipped)")
                    + " [used for LR/early stopping]"
                )
                tqdm.write(f"      Train/val_fixed gap: {train_val_fixed_gap:.6f}")
                tqdm.write(f"      Train/val_unfixed gap: {train_val_unfixed_gap:.6f}")
                tqdm.write(
                    f"      val_fixed/val_unfixed gap: {val_fixed_unfixed_gap:.6f}"
                )
                # Warn if significant discrepancy suggests regularization issues
                if (
                    train_val_unfixed_gap > 1.5 * train_val_fixed_gap
                    and not args.run_mode == "overfit_on_one"
                ):
                    tqdm.write(
                        f"      WARNING: Large train/val_unfixed gap suggests potential regularization issues!"
                    )
                if (
                    val_fixed_unfixed_gap > 0.1
                    and not args.run_mode == "overfit_on_one"
                ):
                    tqdm.write(
                        f"      INFO: val_fixed vs val_unfixed gap indicates noise schedule sensitivity."
                    )

            # Track learning rate changes for checkpoint rollback
            # When val_metric=val_gen, primary_val_loss is None on epochs where val_gen didn't run
            # — skip all scheduler/rollback/early-stopping on those epochs.
            if primary_val_loss is not None:
                if args.enable_checkpoint_rollback:
                    old_lr = current_lr
                    # Use the configured validation metric for learning rate scheduling
                    scheduler.step(primary_val_loss)
                    current_lr = opt.param_groups[0]["lr"]

                    # Check if learning rate fell below minimum threshold
                    if current_lr < args.min_lr_stop_point:
                        print(
                            f"Learning rate dropped to {current_lr:.2e}, which is below minimum threshold of {args.min_lr_stop_point:.2e}."
                        )
                        print(
                            "Early stopping triggered due to minimum learning rate threshold."
                        )
                        break

                    # Detect if learning rate was reduced
                    if current_lr < old_lr:
                        epochs_since_lr_reduction = 0
                        lr_has_dropped = True  # Set flag when LR drops
                        print(
                            f"  Learning rate reduced from {old_lr:.2e} to {current_lr:.2e}"
                        )
                        print(
                            f"  Starting rollback monitoring (threshold: {rollback_patience_threshold} epochs)"
                        )
                    else:
                        epochs_since_lr_reduction += 1
                else:
                    # Use the configured validation metric for learning rate scheduling
                    scheduler.step(primary_val_loss)
                    current_lr = opt.param_groups[0]["lr"]

                    # Check if learning rate fell below minimum threshold
                    if current_lr < args.min_lr_stop_point:
                        print(
                            f"Learning rate dropped to {current_lr:.2e}, which is below minimum threshold of {args.min_lr_stop_point:.2e}."
                        )
                        print(
                            "Early stopping triggered due to minimum learning rate threshold."
                        )
                        break

            # Intermediate checkpoints are only saved when a new best model is found
            # This creates a complete history of best models without cluttering with every epoch

            # AF2 ratio decay schedule: update after each validation cycle
            val_cycle += 1
            af2_ratio_float, af2_ratio_int = None, None
            if args.ratio_af2_pdb > 0 and getattr(args, 'af2_ratio_decay', 1.0) != 1.0:
                af2_ratio_float, af2_ratio_int = train_loader.dataset.set_af2_ratio(
                    val_cycle=val_cycle,
                    initial_ratio=args.ratio_af2_pdb,
                    decay=args.af2_ratio_decay,
                )

            # Log epoch results to wandb - dual validation metrics logged every epoch
            should_log = should_log_to_wandb(args, is_smoke_test)
            if is_main_process():
                print(
                    f"DEBUG: Epoch {ep} - should_log_to_wandb={should_log}, args.use_wandb={args.use_wandb}, is_smoke_test={is_smoke_test}, run_mode={args.run_mode}"
                )

            if should_log:
                if is_main_process():
                    print(f"Logging epoch {ep} results to wandb")

                # Calculate epoch duration in minutes
                epoch_end_time = time.time()
                epoch_duration_minutes = (epoch_end_time - epoch_start_time) / 60.0

                wandb_log_dict = {
                    "train/epoch_loss": avg_train_loss,
                    "train/epoch_accuracy": avg_train_accuracy,
                    "train/epoch_electrostatic_loss": avg_train_elec_loss,
                    "train/epoch_geom_topology_loss": avg_train_geom_loss,
                    "train/epoch_gradient_norm": avg_gradient_norm,
                    "train/epoch_max_gradient_norm": max_gradient_norm,
                    "train/epoch_unique_predictions": epoch_unique_predictions,
                    "train/epoch_most_frequent_ratio": epoch_most_frequent_ratio,
                    "train/epoch_avg_entropy": epoch_avg_entropy,
                    "train/rollback_count": rollback_count,  # Cumulative rollback count
                    "timing/epoch_duration_minutes": epoch_duration_minutes,  # Epoch timing
                    # OOM batch skip statistics
                    "train/oom_batches_skipped": oom_skip_count,
                    "train/total_batches_attempted": total_batches_attempted,
                    "train/successful_batches": total_batches_attempted
                    - oom_skip_count,
                    "train/oom_skip_rate": (
                        oom_skip_count / total_batches_attempted
                        if total_batches_attempted > 0
                        else 0
                    ),
                    # val_fixed (t=0) metrics (for LR scheduling and model selection)
                    "val_fixed/epoch_loss": avg_val_fixed_loss,
                    "val_fixed/epoch_accuracy": avg_val_fixed_accuracy,
                    "val_fixed/epoch_unique_predictions": val_fixed_diversity_metrics[
                        "unique_predictions"
                    ],
                    "val_fixed/epoch_most_frequent_ratio": val_fixed_diversity_metrics[
                        "most_frequent_ratio"
                    ],
                    "val_fixed/epoch_avg_entropy": val_fixed_diversity_metrics[
                        "avg_entropy"
                    ],
                    # DSSP metrics for val_fixed (t=0)
                    "val_fixed/dssp_loss": avg_val_fixed_dssp_loss,
                    "val_fixed/dssp_accuracy": avg_val_fixed_dssp_accuracy,
                    "val_fixed/dssp_batch_count": val_fixed_dssp_batch_count,
                    # Aux head metrics for val_fixed
                    "val_fixed/electrostatic_loss": avg_val_fixed_elec_loss,
                    "val_fixed/electrostatic_accuracy": avg_val_fixed_elec_accuracy,
                    "val_fixed/geom_topology_loss": avg_val_fixed_geom_loss,
                    "val_fixed/geom_topology_accuracy": avg_val_fixed_geom_accuracy,
                    # val_t2 (t=2) metrics (low noise validation)
                    "val_t2/epoch_loss": avg_val_t2_loss,
                    "val_t2/epoch_accuracy": avg_val_t2_accuracy,
                    "val_t2/epoch_unique_predictions": val_t2_diversity_metrics[
                        "unique_predictions"
                    ],
                    "val_t2/epoch_most_frequent_ratio": val_t2_diversity_metrics[
                        "most_frequent_ratio"
                    ],
                    "val_t2/epoch_avg_entropy": val_t2_diversity_metrics["avg_entropy"],
                    # DSSP metrics for val_t2
                    "val_t2/dssp_loss": avg_val_t2_dssp_loss,
                    "val_t2/dssp_accuracy": avg_val_t2_dssp_accuracy,
                    # Aux head metrics for val_t2
                    "val_t2/electrostatic_loss": avg_val_t2_elec_loss,
                    "val_t2/electrostatic_accuracy": avg_val_t2_elec_accuracy,
                    "val_t2/geom_topology_loss": avg_val_t2_geom_loss,
                    "val_t2/geom_topology_accuracy": avg_val_t2_geom_accuracy,
                    # val_t4 (t=4) metrics (medium noise validation)
                    "val_t4/epoch_loss": avg_val_t4_loss,
                    "val_t4/epoch_accuracy": avg_val_t4_accuracy,
                    "val_t4/epoch_unique_predictions": val_t4_diversity_metrics[
                        "unique_predictions"
                    ],
                    "val_t4/epoch_most_frequent_ratio": val_t4_diversity_metrics[
                        "most_frequent_ratio"
                    ],
                    "val_t4/epoch_avg_entropy": val_t4_diversity_metrics["avg_entropy"],
                    # DSSP metrics for val_t4
                    "val_t4/dssp_loss": avg_val_t4_dssp_loss,
                    "val_t4/dssp_accuracy": avg_val_t4_dssp_accuracy,
                    # Aux head metrics for val_t4
                    "val_t4/electrostatic_loss": avg_val_t4_elec_loss,
                    "val_t4/electrostatic_accuracy": avg_val_t4_elec_accuracy,
                    "val_t4/geom_topology_loss": avg_val_t4_geom_loss,
                    "val_t4/geom_topology_accuracy": avg_val_t4_geom_accuracy,
                    # val_t6 (t=6) metrics (high noise validation)
                    "val_t6/epoch_loss": avg_val_t6_loss,
                    "val_t6/epoch_accuracy": avg_val_t6_accuracy,
                    "val_t6/epoch_unique_predictions": val_t6_diversity_metrics[
                        "unique_predictions"
                    ],
                    "val_t6/epoch_most_frequent_ratio": val_t6_diversity_metrics[
                        "most_frequent_ratio"
                    ],
                    "val_t6/epoch_avg_entropy": val_t6_diversity_metrics["avg_entropy"],
                    # DSSP metrics for val_t6
                    "val_t6/dssp_loss": avg_val_t6_dssp_loss,
                    "val_t6/dssp_accuracy": avg_val_t6_dssp_accuracy,
                    # Aux head metrics for val_t6
                    "val_t6/electrostatic_loss": avg_val_t6_elec_loss,
                    "val_t6/electrostatic_accuracy": avg_val_t6_elec_accuracy,
                    "val_t6/geom_topology_loss": avg_val_t6_geom_loss,
                    "val_t6/geom_topology_accuracy": avg_val_t6_geom_accuracy,
                    # val_unfixed metrics (for regularization analysis)
                    "val_unfixed/epoch_loss": avg_val_unfixed_loss,
                    "val_unfixed/epoch_accuracy": avg_val_unfixed_accuracy,
                    "val_unfixed/epoch_unique_predictions": val_unfixed_diversity_metrics[
                        "unique_predictions"
                    ],
                    "val_unfixed/epoch_most_frequent_ratio": val_unfixed_diversity_metrics[
                        "most_frequent_ratio"
                    ],
                    "val_unfixed/epoch_avg_entropy": val_unfixed_diversity_metrics[
                        "avg_entropy"
                    ],
                    # DSSP metrics for val_unfixed
                    "val_unfixed/dssp_loss": avg_val_unfixed_dssp_loss,
                    "val_unfixed/dssp_accuracy": avg_val_unfixed_dssp_accuracy,
                    # Aux head metrics for val_unfixed
                    "val_unfixed/electrostatic_loss": avg_val_unfixed_elec_loss,
                    "val_unfixed/electrostatic_accuracy": avg_val_unfixed_elec_accuracy,
                    "val_unfixed/geom_topology_loss": avg_val_unfixed_geom_loss,
                    "val_unfixed/geom_topology_accuracy": avg_val_unfixed_geom_accuracy,
                    # Multi-validation gap analysis
                    "multi_val/train_val_fixed_gap": train_val_fixed_gap,
                    "multi_val/train_val_unfixed_gap": train_val_unfixed_gap,
                    "multi_val/val_fixed_unfixed_gap": val_fixed_unfixed_gap,
                    # Combined validation metrics
                    "val_combo/epoch_loss": avg_val_combined_loss,
                    "val_combo/epoch_accuracy": avg_val_combined_accuracy,
                    # All validation metrics (val_all: average of all 5 validation losses)
                    "val_all/epoch_loss": avg_val_all_loss,
                    "val_all/epoch_accuracy": avg_val_all_accuracy,
                    # Primary validation metric (configurable via args.val_metric)
                    "primary_val/loss": primary_val_loss if primary_val_loss is not None else float('nan'),
                    "primary_val/accuracy": primary_val_accuracy,
                    "primary_val/metric_name": primary_val_name,
                    "val/learning_rate": opt.param_groups[0]["lr"],
                    "epoch": ep,
                }

                # Add epoch-level averaged validation metrics (collected across batches)
                if "avg_time_sampled" in val_fixed_epoch_avg_metrics:
                    wandb_log_dict["val_fixed/epoch_avg_time_sampled"] = (
                        val_fixed_epoch_avg_metrics["avg_time_sampled"]
                    )
                if "avg_max_dirichlet_prob" in val_fixed_epoch_avg_metrics:
                    wandb_log_dict["val_fixed/epoch_avg_max_dirichlet_prob"] = (
                        val_fixed_epoch_avg_metrics["avg_max_dirichlet_prob"]
                    )
                if "avg_max_dirichlet_prob" in val_unfixed_epoch_avg_metrics:
                    wandb_log_dict["val_unfixed/epoch_avg_max_dirichlet_prob"] = (
                        val_unfixed_epoch_avg_metrics["avg_max_dirichlet_prob"]
                    )

                # AF2 ratio schedule logging (only when decay is active)
                if af2_ratio_float is not None:
                    wandb_log_dict["train/af2_ratio_float"] = af2_ratio_float
                    wandb_log_dict["train/af2_ratio_int"] = af2_ratio_int

                # Add unique protein tracking metrics (verbose mode only)
                if args.verbose and is_main_process():
                    # Get current counts from dataset-level tracking
                    from data.unified_dataset import get_global_protein_counts

                    dataset_epoch_count, dataset_cumulative_count = (
                        get_global_protein_counts()
                    )
                    wandb_log_dict.update(
                        {
                            "train/epoch_unique_proteins_union_final": dataset_epoch_count,
                            "train/cumulative_unique_proteins_union_final": dataset_cumulative_count,
                        }
                    )

                # Full-generation validation metrics
                if gen_val_metrics is not None:
                    wandb_log_dict.update({
                        'val_gen/mean_recovery':   gen_val_metrics['mean_recovery'],
                        'val_gen/median_recovery': gen_val_metrics['median_recovery'],
                        'val_gen/std_recovery':    gen_val_metrics['std_recovery'],
                        'val_gen/n_proteins':      gen_val_metrics['n_proteins'],
                    })
                    if use_val_gen_metric:
                        wandb_log_dict['val_gen/smoothed_score'] = smoothed_val_gen_score
                        wandb_log_dict['val_gen/raw_score'] = gen_val_metrics['mean_recovery'] + gen_val_metrics['median_recovery']

                wandb.log(wandb_log_dict)
                # Force immediate logging to WandB
                if hasattr(wandb, "run") and wandb.run is not None:
                    try:
                        wandb.run._flush()
                    except:
                        pass  # _flush might not be available in all versions

            # Print epoch summary (only main process)
            if is_main_process():
                # Calculate epoch duration for console output too
                if "epoch_end_time" not in locals():
                    epoch_end_time = time.time()
                    epoch_duration_minutes = (epoch_end_time - epoch_start_time) / 60.0

                tqdm.write(
                    f"Epoch {ep}/{max_epochs} (Duration: {epoch_duration_minutes:.2f} min)"
                )
                tqdm.write(
                    f"  Train      - Total: {avg_train_loss:.4f}, Acc: {avg_train_accuracy:.4f}"
                )
                tqdm.write(
                    f"  Val t=0    - Total: {val_t0_metrics['avg_loss']:.4f}, Acc: {val_t0_metrics['avg_accuracy']:.4f}"
                )
                tqdm.write(
                    f"  Val t=2    - Total: {val_t2_metrics['avg_loss']:.4f}, Acc: {val_t2_metrics['avg_accuracy']:.4f}"
                )
                tqdm.write(
                    f"  Val t=4    - Total: {val_t4_metrics['avg_loss']:.4f}, Acc: {val_t4_metrics['avg_accuracy']:.4f}"
                )
                tqdm.write(
                    f"  Val t=6    - Total: {val_t6_metrics['avg_loss']:.4f}, Acc: {val_t6_metrics['avg_accuracy']:.4f}"
                )
                tqdm.write(
                    f"  Val Unfixed- Total: {avg_val_unfixed_loss:.4f}, Acc: {avg_val_unfixed_accuracy:.4f}"
                )
                tqdm.write(
                    f"  Train Diversity - Unique: {epoch_unique_predictions}/20, MostFreq: {epoch_most_frequent_ratio:.3f}, Entropy: {epoch_avg_entropy:.3f}"
                )

                # Add unique protein tracking in verbose mode
                if args.verbose:
                    # Use coordinated counts if available, otherwise get from dataset
                    if (
                        "local_epoch_unique_count" in locals()
                        and "local_cumulative_unique_count" in locals()
                    ):
                        epoch_count_to_display = local_epoch_unique_count
                        cumulative_count_to_display = local_cumulative_unique_count
                    else:
                        # Get from dataset-level tracking
                        from data.unified_dataset import get_global_protein_counts

                        epoch_count_to_display, cumulative_count_to_display = (
                            get_global_protein_counts()
                        )

                    tqdm.write(
                        f"  Protein Union (All {args.num_workers} Workers) - Epoch: {epoch_count_to_display} unique, Cumulative: {cumulative_count_to_display} unique"
                    )

                tqdm.write(
                    f"  Val t=0 Div     - Unique: {val_t0_diversity_metrics['unique_predictions']}/20, MostFreq: {val_t0_diversity_metrics['most_frequent_ratio']:.3f}, Entropy: {val_t0_diversity_metrics['avg_entropy']:.3f}"
                )
                tqdm.write(
                    f"  Val Unfixed Div - Unique: {val_unfixed_diversity_metrics['unique_predictions']}/20, MostFreq: {val_unfixed_diversity_metrics['most_frequent_ratio']:.3f}, Entropy: {val_unfixed_diversity_metrics['avg_entropy']:.3f}"
                )

                # Force output to be flushed immediately
                import sys

                sys.stdout.flush()

                # Print amino acid prediction analysis (only in regular mode, not smoke test)
                if (
                    "top_predicted_aa" in val_t0_diversity_metrics
                    and "bottom_predicted_aa" in val_t0_diversity_metrics
                ):
                    # Top 3 most predicted amino acids (val_t0)
                    top_aa_strs = [
                        f"{aa}({count}, {ratio:.1%})"
                        for aa, count, ratio in val_t0_diversity_metrics[
                            "top_predicted_aa"
                        ]
                    ]
                    print(f"  Val t=0 Top 3 AA    - {', '.join(top_aa_strs)}")

                    # Bottom 3 least predicted amino acids (val_t0)
                    if val_t0_diversity_metrics["bottom_predicted_aa"]:
                        bottom_aa_strs = [
                            f"{aa}({count}, {ratio:.1%})"
                            for aa, count, ratio in val_t0_diversity_metrics[
                                "bottom_predicted_aa"
                            ]
                        ]
                        print(f"  Val t=0 Bottom 3 AA - {', '.join(bottom_aa_strs)}")
                    else:
                        print(
                            f"  Val t=0 Bottom 3 AA - No predictions for least frequent amino acids"
                        )

            # Update epoch progress bar with current metrics (only main process)
            if is_main_process():
                epoch_pbar.set_postfix(
                    {
                        "train_loss": f"{avg_train_loss:.4f}",
                        "val_t0_loss": f'{val_t0_metrics["avg_loss"]:.4f}',
                        "val_t0_acc": f'{val_t0_metrics["avg_accuracy"]:.3f}',
                        "lr": f'{opt.param_groups[0]["lr"]:.2e}',
                    }
                )

            # In distributed setting, aggregate validation losses across ranks
            if is_distributed:
                # Aggregate all validation metrics across ranks for consistency
                val_t0_loss_tensor = torch.tensor(
                    val_t0_metrics["avg_loss"], device=device
                )
                val_t0_metrics["avg_loss"] = reduce_loss_across_processes(
                    val_t0_loss_tensor
                ).item()

                val_t2_loss_tensor = torch.tensor(
                    val_t2_metrics["avg_loss"], device=device
                )
                val_t2_metrics["avg_loss"] = reduce_loss_across_processes(
                    val_t2_loss_tensor
                ).item()

                val_t4_loss_tensor = torch.tensor(
                    val_t4_metrics["avg_loss"], device=device
                )
                val_t4_metrics["avg_loss"] = reduce_loss_across_processes(
                    val_t4_loss_tensor
                ).item()

                val_t6_loss_tensor = torch.tensor(
                    val_t6_metrics["avg_loss"], device=device
                )
                val_t6_metrics["avg_loss"] = reduce_loss_across_processes(
                    val_t6_loss_tensor
                ).item()

                val_unfixed_tensor = torch.tensor(avg_val_unfixed_loss, device=device)
                avg_val_unfixed_loss = reduce_loss_across_processes(
                    val_unfixed_tensor
                ).item()

                train_tensor = torch.tensor(avg_train_loss, device=device)
                avg_train_loss = reduce_loss_across_processes(train_tensor).item()

                # Recalculate the combined metric and primary metric with aggregated values
                # Using t=0 as the primary fixed validation metric
                avg_val_fixed_loss = val_t0_metrics[
                    "avg_loss"
                ]  # For backward compatibility
                avg_val_combined_loss = (
                    avg_val_fixed_loss + avg_val_unfixed_loss
                ) / 2.0
                # Recalculate val_all_loss with aggregated values
                avg_val_all_loss = (
                    avg_val_fixed_loss
                    + avg_val_unfixed_loss
                    + val_t2_metrics["avg_loss"]
                    + val_t4_metrics["avg_loss"]
                    + val_t6_metrics["avg_loss"]
                ) / 5.0
                if not use_val_gen_metric:
                    primary_val_loss, primary_val_name = get_validation_metric(
                        args,
                        avg_train_loss,
                        avg_val_fixed_loss,
                        avg_val_unfixed_loss,
                        avg_val_combined_loss,
                        avg_val_all_loss,
                    )
                # else: primary_val_loss already set from smoothed val_gen score above

            # Early stopping and best model selection based on configurable validation metric (skip for smoke test)
            # When val_metric=val_gen, primary_val_loss is None on non-val_gen epochs — skip entirely.
            _is_improvement = False
            if primary_val_loss is not None:
                _is_improvement = (
                    primary_val_loss > best_val_loss if use_val_gen_metric
                    else primary_val_loss < best_val_loss
                )

            if _is_improvement:
                best_val_loss = primary_val_loss
                best_epoch = ep
                patience_counter = 0
                epochs_since_best_checkpoint = 0

                # Save checkpoint state for potential rollback (on all ranks for distributed training)
                if args.enable_checkpoint_rollback:
                    best_model_state = {
                        k: v.cpu().clone() for k, v in model.state_dict().items()
                    }
                    # Persist to disk so rollback survives a crash (M-4 fix).
                    if is_main_process():
                        try:
                            _rollback_path = os.path.join(output_base, "saved_models", "best_rollback.pt")
                            os.makedirs(os.path.dirname(_rollback_path), exist_ok=True)
                            torch.save(best_model_state, _rollback_path)
                        except Exception as _e:
                            print(f"Warning: could not write best_rollback.pt: {_e}")

                # Only main process saves the actual model file
                if is_main_process():
                    # Save best model with metadata and timestamped name
                    current_metrics = {
                        "epoch": ep,
                        "train_loss": avg_train_loss,
                        "val_loss": primary_val_loss,  # Use configured primary validation metric
                        "val_accuracy": best_val_accuracy,
                        "train_accuracy": avg_train_accuracy,
                        "best_val_loss": best_val_loss,
                        "best_val_accuracy": best_val_accuracy,
                        "learning_rate": opt.param_groups[0]["lr"],
                    }

                    # Prepare comprehensive training state for checkpoint
                    current_training_state = {
                        "current_lr": current_lr,
                        "epochs_since_lr_reduction": epochs_since_lr_reduction,
                        "epochs_since_best_checkpoint": epochs_since_best_checkpoint,
                        "lr_has_dropped": lr_has_dropped,
                        "rollback_count": rollback_count,
                        "patience_counter": patience_counter,
                        "best_val_loss": best_val_loss,
                        "best_val_accuracy": best_val_accuracy,
                    }

                    _keep_all_best = getattr(args, 'keep_all_best_checkpoints', False)
                    try:
                        best_model_path = save_model_with_metadata(
                            model=model,
                            output_base=output_base,
                            model_name=model_name,
                            job_timestamp=job_timestamp,
                            metrics=current_metrics,
                            args=args,
                            is_best=True,
                            optimizer=opt,
                            scheduler=scheduler,
                            epoch=ep,
                            training_state=current_training_state,
                            keep_all_best=_keep_all_best,
                        )

                        # Copy to output-dir if specified (for pipeline support)
                        if hasattr(args, "output_dir") and args.output_dir:
                            try:
                                import shutil

                                output_dir_models = os.path.join(
                                    args.output_dir, "saved_models"
                                )
                                os.makedirs(output_dir_models, exist_ok=True)

                                # Copy the best model checkpoint
                                checkpoint_filename = os.path.basename(best_model_path)
                                output_checkpoint_path = os.path.join(
                                    output_dir_models, checkpoint_filename
                                )
                                shutil.copy2(best_model_path, output_checkpoint_path)
                                print(
                                    f"  Best model also saved to output-dir: {output_checkpoint_path}"
                                )
                            except Exception as copy_e:
                                print(
                                    f"  Warning: Failed to copy best model to output-dir: {copy_e}"
                                )

                        # Copy checkpoint to remote directory if specified
                        if args.checkpoint_copy_dir and is_main_process():
                            try:
                                import shutil

                                os.makedirs(args.checkpoint_copy_dir, exist_ok=True)

                                # Copy the best model checkpoint
                                checkpoint_filename = os.path.basename(best_model_path)
                                remote_checkpoint_path = os.path.join(
                                    args.checkpoint_copy_dir, checkpoint_filename
                                )
                                shutil.copy2(best_model_path, remote_checkpoint_path)
                                print(
                                    f"  Best model copied to remote location: {remote_checkpoint_path}"
                                )

                            except Exception as copy_e:
                                print(
                                    f"  Warning: Failed to copy best model to remote location: {copy_e}"
                                )

                        # Clean up old checkpoints (only when overwriting mode — unique files never need cleanup)
                        if args.save_intermediate_models and not _keep_all_best:
                            cleanup_old_checkpoints(
                                output_base, model_name, keep_last_n=5
                            )

                    except Exception as e:
                        print(f"Error saving model with metadata: {e}")
                        # Fallback to simple save with consistent naming.
                        # WARNING (L-1): This checkpoint contains ONLY model weights — no optimizer,
                        # scheduler, or metric state.  Resuming from it will trigger the Adam
                        # cold-start problem (H-1).  Use the last complete checkpoint instead.
                        print(
                            "WARNING: Fallback checkpoint is INCOMPLETE (weights only). "
                            "Resume from the last complete checkpoint to avoid training instability."
                        )
                        try:
                            if _keep_all_best:
                                fallback_filename = f"best_{model_name}_started_{job_timestamp}_epoch_{ep:04d}.pt"
                            else:
                                fallback_filename = f"best_so_far_{model_name}_started_{job_timestamp}.pt"
                            model_save_path = os.path.join(
                                output_base, "saved_models", fallback_filename
                            )
                            torch.save(model.state_dict(), model_save_path)
                        except Exception as e2:
                            print(f"Error with fallback save: {e2}")
                            if _keep_all_best:
                                torch.save(model.state_dict(), f"best_{model_name}_epoch_{ep:04d}.pt")
                            else:
                                torch.save(model.state_dict(), f"best_so_far_{model_name}.pt")

                    print(
                        f"  New best model saved ({primary_val_name}: {best_val_loss:.4f})"
                    )
                    if should_log_to_wandb(args, is_smoke_test):
                        wandb.log({
                            "checkpoint/saved": 1,
                            "checkpoint/epoch": ep,
                            "checkpoint/val_loss": best_val_loss,
                            "checkpoint/val_metric_name": primary_val_name,
                            "checkpoint/path": best_model_path or "",
                        })
            elif primary_val_loss is not None:
                patience_counter += 1
                epochs_since_best_checkpoint += 1
                if is_main_process():
                    print(f"  No improvement for {patience_counter} epochs")

                # Checkpoint rollback logic (synchronized across all ranks)
                if (
                    args.enable_checkpoint_rollback
                    and best_model_state is not None
                    and lr_has_dropped  # Only allow rollback after LR drop
                    and epochs_since_lr_reduction >= rollback_patience_threshold
                    and epochs_since_best_checkpoint >= rollback_patience_threshold
                ):

                    if is_main_process():
                        print(
                            f"   CHECKPOINT ROLLBACK: No improvement for {epochs_since_best_checkpoint} epochs since LR reduction"
                        )
                        print(
                            f"     Restoring model to best checkpoint (val_loss: {best_val_loss:.4f})"
                        )

                    # Restore model parameters on all ranks
                    model_device = next(model.parameters()).device
                    model.load_state_dict(
                        {k: v.to(model_device) for k, v in best_model_state.items()}
                    )
                    # load_state_dict does not change train/eval mode; model is currently in
                    # eval() from the validation pass — restore training mode explicitly (M-2 fix).
                    model.train()

                    # Selective optimizer reset: reset momentum but keep adaptive scaling
                    if is_main_process():
                        print(
                            f"     Resetting optimizer momentum while preserving adaptive scaling for LR={current_lr:.2e}"
                        )
                        print(f"     Ensuring device consistency on {model_device}")

                    # Use selective reset with device consistency to prevent device mismatch errors
                    reset_optimizer_momentum_selectively(
                        opt, target_device=model_device
                    )

                    # Verify device consistency after rollback
                    if is_main_process():
                        try:
                            # Quick verification that optimizer state tensors are on correct device
                            device_mismatches = []
                            for group in opt.param_groups:
                                for p in group["params"]:
                                    if p in opt.state:
                                        state = opt.state[p]
                                        for key, value in state.items():
                                            if (
                                                torch.is_tensor(value) and key != "step"
                                            ):  # step can be on CPU
                                                if value.device != model_device:
                                                    device_mismatches.append(
                                                        f"{key}: {value.device} != {model_device}"
                                                    )

                            if device_mismatches:
                                print(
                                    f"     WARNING: Device mismatches found after rollback: {device_mismatches}"
                                )
                            else:
                                print(
                                    f"     Device consistency verified: all optimizer tensors on {model_device}"
                                )
                        except Exception as e:
                            print(
                                f"     Device verification failed (non-critical): {e}"
                            )

                    # Reset scheduler state to be consistent with the restored checkpoint
                    # The scheduler should "forget" about the worse metrics that led to rollback
                    scheduler.best = best_val_loss  # Set scheduler's best to the restored checkpoint's performance
                    scheduler.cooldown_counter = 0  # Reset cooldown
                    scheduler.num_bad_epochs = 0  # Reset bad epochs counter

                    # Ensure learning rate is set to current (reduced) value on all ranks
                    for param_group in opt.param_groups:
                        param_group["lr"] = current_lr

                    # Reset counters to prevent back-to-back rollbacks and give model fresh chance
                    epochs_since_best_checkpoint = 0
                    # Do not reset epochs_since_lr_reduction - that tracks actual LR drops, not rollbacks
                    patience_counter = (
                        0  # Give the model another chance with the restored state
                    )

                    # Increment rollback counter
                    rollback_count += 1

                    if is_main_process():
                        print(
                            f"     Rollback complete, continuing with LR={current_lr:.2e}"
                        )
                        print(f"     Total rollbacks so far: {rollback_count}")
                        print(
                            f"     Next rollback possible after {rollback_patience_threshold} epochs without LR reduction"
                        )

            # Track best validation metrics using the configured validation metric
            if primary_val_accuracy > best_val_accuracy:
                best_val_accuracy = primary_val_accuracy

            # Early stopping check (applies regardless of smoke test mode)
            if patience_counter >= args.patience:
                if is_main_process():
                    print(f"Early stopping triggered after {ep} epochs")
                    return

        # Close epoch progress bar (only main process) - after loop completes
        if is_main_process() and hasattr(epoch_pbar, "close"):
            epoch_pbar.close()

    except KeyboardInterrupt:
        if is_main_process():
            print("\nTraining interrupted by user")
    finally:
        # Cleanup distributed resources
        cleanup_distributed()

        # Update config file with best metrics when training completes (only main process)
        if is_main_process():
            print("Training completed")

            # Prepare best metrics for config file
            best_metrics = {
                "best_val_loss": (
                    best_val_loss if best_val_loss != float("inf") else None
                ),
                "best_val_accuracy": best_val_accuracy,
                "best_epoch": best_epoch,
                "final_learning_rate": opt.param_groups[0]["lr"],
                "best_model_path": best_model_path,
            }

            update_config_with_best_metrics(
                config_dir, model_name, best_metrics, job_timestamp
            )

            # Print saved model path right after config file update
            if best_model_path and is_main_process():
                print(f"Best model saved to: {best_model_path}")

            # Finish wandb run properly
            if should_log_to_wandb(args, is_smoke_test):
                wandb.finish()
                print("WANDB run finished successfully")

            # Print final summary
            print(f"\nFinal Training Summary:")
            print(f"  Best validation loss: {best_val_loss:.4f} (epoch {best_epoch})")
            print(f"  Best validation accuracy: {best_val_accuracy:.4f}")
            print(f"  Final learning rate: {opt.param_groups[0]['lr']:.2e}")
        elif is_main_process():
            print("Smoke test completed")


if __name__ == "__main__":
    if _IFD_DEBUG: print("DEBUG: __main__ block reached, calling main()", flush=True)
    main()
    if _IFD_DEBUG: print("DEBUG: main() function completed", flush=True)
