
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
training_utils.py

Utility functions for training the inverse folding model.
These functions handle data processing, tensor health checks, virtual node masking, and model saving.
"""

import inspect
import json
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from flow.sampler import sample_forward

# Verbose chunk-loading and startup tracing, off unless asked for. These lines
# are genuinely useful when a training run misbehaves, but they should not be
# the default experience. Set IFD_DEBUG=1 to turn them back on.
_IFD_DEBUG = bool(os.environ.get("IFD_DEBUG"))


def _reapply_structure_noise_for_validation(data, fixed_time: float, val_loader, args=None):
    """
    Validation fix disabled - causes double-noising and reduces DSSP accuracy to 16%.

    The issue is complex:
    1. Validation dataset coordinates may already have noise
    2. Adding more noise on top causes double-noising
    3. Time-dependent DSSP variation may need to be handled differently

    For now, we disable this fix to maintain 90% DSSP accuracy.
    """
    print(f"[VAL_NOISE_DEBUG] Validation fix DISABLED - using original data (fixed_time={fixed_time})")
    return data


def _extract_graph_builder_params(args):
    """
    Extract all graph builder parameters from args.

    Args:
        args: Training arguments namespace

    Returns:
        dict: Dictionary of graph builder parameters
    """
    # Define all graph builder parameters based on GraphBuilder.__init__ signature
    graph_builder_param_names = {
        'k_neighbors': 'k',              # Map args name to GraphBuilder param name
        'k_farthest': 'k_farthest',
        'k_random': 'k_random',
        'max_edge_dist': 'max_edge_dist',
        'num_rbf_3d': 'num_rbf_3d',
        'num_rbf_seq': 'num_rbf_seq',
        # RBF distance range parameters (new)
        'rbf_3d_min': 'rbf_3d_min',
        'rbf_3d_max': 'rbf_3d_max',
        'rbf_3d_spacing': 'rbf_3d_spacing',
        'use_virtual_node': 'use_virtual_node',
        'no_source_indicator': 'no_source_indicator',
        'rigid_residue_noise': 'rigid_residue_noise',
        # Feature flags that affect node/edge dimensionality
        'include_node_degree': 'include_node_degree',
        'blur_uncertainty': 'blur_uncertainty',
        # Dataset parameters needed for consistent graph building
        'split_json': 'split_json',
        'map_pkl': 'map_pkl'
    }

    graph_params = {}
    for arg_name, param_name in graph_builder_param_names.items():
        if hasattr(args, arg_name):
            value = getattr(args, arg_name)
            graph_params[param_name] = value

    return graph_params


def _extract_model_architecture_params(args):
    """
    Extract model architecture parameters that affect model structure and sampling.

    Args:
        args: Training arguments namespace

    Returns:
        dict: Dictionary of model architecture parameters
    """
    # Define model architecture parameters that affect model behavior during sampling
    model_param_names = [
        'hidden_dim',
        'hidden_dim_v',
        'node_dim_s',
        'node_dim_v',
        'edge_dim_s',
        'edge_dim_v',
        'num_layers_gvp',
        'num_message_layers',
        'num_layers_prediction',
        'dropout',
        'use_qkv',
        'architecture',
        'flexible_loss_scaling',
        'use_entropy_features',
        'entropy_smoothing',
        'attention_skip_connection',
    ]

    model_params = {}
    for param_name in model_param_names:
        if hasattr(args, param_name):
            value = getattr(args, param_name)
            model_params[param_name] = value

    return model_params


def validate_checkpoint_completeness(args, graph_params, model_params):
    """
    Validate that we've captured all relevant parameters for reproducible sampling.
    Issues warnings for any critical parameters that might be missing.

    Args:
        args: Original training arguments
        graph_params: Extracted graph builder parameters
        model_params: Extracted model architecture parameters

    Returns:
        bool: True if validation passes, False if critical parameters are missing
    """
    validation_passed = True

    # Check critical graph builder parameters that should always be present
    critical_graph_params = {
        'k': 'k_neighbors',           # param_name: arg_name
        'k_farthest': 'k_farthest',
        'k_random': 'k_random',
        'use_virtual_node': 'use_virtual_node',
        'no_source_indicator': 'no_source_indicator'
    }

    missing_from_args = []
    missing_from_extraction = []

    for param_name, arg_name in critical_graph_params.items():
        if not hasattr(args, arg_name):
            missing_from_args.append(arg_name)
            validation_passed = False
        elif param_name not in graph_params:
            missing_from_extraction.append(param_name)
            validation_passed = False

    if missing_from_args:
        print(f"WARNING: Critical parameters missing from training args: {missing_from_args}")
        print("  This means the model was trained without specifying these important parameters.")

    if missing_from_extraction:
        print(f"WARNING: Critical parameters not extracted to checkpoint: {missing_from_extraction}")
        print("  This is a bug in the parameter extraction logic.")

    # Check important model architecture parameters (informational only)
    important_model_params = ['hidden_dim', 'architecture', 'use_qkv', 'num_layers_gvp']
    missing_model_params = []

    for param in important_model_params:
        if hasattr(args, param) and param not in model_params:
            missing_model_params.append(param)

    if missing_model_params:
        print(f"INFO: Model architecture parameters not captured (may use defaults): {missing_model_params}")
        # Note: These are informational only, not critical failures

    # Summary
    status = 'PASSED' if validation_passed else 'FAILED'
    print(f"\\nCheckpoint parameter validation: {status}")
    print(f"  Graph builder parameters captured: {len(graph_params)}")
    print(f"  Model architecture parameters captured: {len(model_params)}")

    if validation_passed:
        print("  [ok] All critical parameters captured for reproducible sampling")
    else:
        print("  [warn] Some critical parameters missing - sampling may not be fully reproducible")

    return validation_passed


def get_rank():
    """
    Get the current process rank for distributed training.
    This follows WANDB's recommended approach for distributed training.
    """
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    if "OMPI_COMM_WORLD_RANK" in os.environ:
        return int(os.environ["OMPI_COMM_WORLD_RANK"])
    return None


def should_log_to_wandb(args, is_smoke_test=False):
    """
    Determine if this process should log to wandb.
    Only rank 0 (or None for single-GPU) should log to wandb.
    """
    current_rank = get_rank()
    # Log to wandb in any non-smoke test or in overfit_on_one mode
    return args.use_wandb and (not is_smoke_test or args.run_mode == 'overfit_on_one') and current_rank in (0, None)


def update_config_with_best_metrics(config_dir, model_name, best_metrics, job_timestamp):
    """
    Update the config file with best performance metrics when training is complete.

    Args:
        config_dir: Directory containing the config file
        model_name: Model name for the config file
        best_metrics: Dictionary containing best performance metrics
        job_timestamp: Timestamp string for this job
    """
    config_path = os.path.join(config_dir, f'config_{job_timestamp}_{model_name}.json')

    try:
        # Read existing config
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Add best performance metrics (simplified for position prediction)
        config['training_results'] = {
            'best_val_loss': best_metrics.get('best_val_loss', float('inf')),
            'best_val_accuracy': best_metrics.get('best_val_accuracy', 0.0),
            'best_epoch': best_metrics.get('best_epoch', 0),
            'final_learning_rate': best_metrics.get('final_learning_rate', 0.0),
            'training_completed': True,
            'completion_timestamp': datetime.now().isoformat()
        }

        # Write updated config back
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"Updated config file with best training results: {config_path}")

    except Exception as e:
        print(f"Error updating config file with best metrics: {e}")


def run_validation_phase(model, val_loader, val_generator, args, device, max_batches_per_epoch, K,
                         val_name="validation", is_smoke_test=False, current_epoch=0, fixed_time=None):
    """
    Run validation on a single validation loader with a specific noise generator.

    Args:
        model: The model to evaluate
        val_loader: DataLoader for validation
        val_generator: Torch generator for reproducible noise sampling
        args: Training arguments
        device: Device to use
        max_batches_per_epoch: Maximum batches to process
        K: Number of amino acid classes
        val_name: Name for logging (e.g., "val_fixed", "val_unfixed")
        is_smoke_test: Whether in smoke test mode
        current_epoch: Current epoch number for epoch-based seeding

    Returns:
        Dict containing validation metrics
    """
    # Initialize metrics (simplified for position prediction)
    total_val_loss = 0
    val_node_count = 0
    val_loss_node_sum = 0.0
    val_acc_node_sum = 0.0
    val_batch_count = 0

    # DSSP metrics (new)
    val_dssp_loss_sum = 0.0
    val_dssp_acc_sum = 0.0
    val_dssp_node_count = 0
    dssp_batch_count = 0

    # Aux head metrics (electrostatic, geom_topology)
    val_elec_loss_sum = 0.0
    val_elec_acc_sum = 0.0
    val_elec_node_count = 0
    val_geom_loss_sum = 0.0
    val_geom_acc_sum = 0.0
    val_geom_node_count = 0

    val_epoch_predicted_classes = []
    val_epoch_pred_entropies = []

    # Collect per-batch metrics for epoch-level averaging
    batch_time_samples = []
    batch_max_dirichlet_probs = []

    with torch.no_grad():
        for batch_data in val_loader:
            # Unpack batch data (handle both legacy and DSSP formats)
            dssp_targets = None
            if len(batch_data) == 5:
                # DSSP format: (data, y, mask, time_value, dssp_targets)
                data, y, mask, time_value, dssp_targets = batch_data
            elif len(batch_data) == 4:
                # Legacy format: (data, y, mask, time_value)
                data, y, mask, time_value = batch_data
            else:
                raise Exception("Batch in validation loader should return time value as well.")
                data, y, mask = batch_data

            if val_batch_count >= max_batches_per_epoch:
                break

            data, y = data.to(device), y.to(device)
            if mask is not None:
                mask = mask.to(device)

            B, N, K = y.shape

            # Reseed generator with epoch-specific seed for reproducible but varying validation
            if val_generator is not None:
                # Combine base seed with epoch for deterministic but epoch-varying noise
                FIXED_SEED_FOR_VAL_FIXED_NOISE = 1  # Base seed
                epoch_seed = FIXED_SEED_FOR_VAL_FIXED_NOISE + (current_epoch * 1000) + val_batch_count
                val_generator.manual_seed(epoch_seed)


            # Use provided generator for reproducible validation noise
            if fixed_time is not None:
                # Fixed time validation phases (val_t0, val_t2, val_t4, val_t6)
                print(f"[VAL_NOISE_DEBUG] Fixed time validation: {val_name}, fixed_time={fixed_time}")
                t = torch.full((B,), fixed_time, device=device)  # Use specified fixed time
                actual_time = fixed_time

                # Reapply structure noise for validation time points
                data = _reapply_structure_noise_for_validation(data, actual_time, val_loader, args)

                # Collect time samples for epoch-level averaging instead of logging per batch
                batch_time_samples.append(t.mean().item())
            elif val_name == "val_fixed":
                # Legacy val_fixed case (fallback)
                t = torch.zeros(B, device=device)  # Default fixed time of 0 for val_fixed
                actual_time = 0.0

                # Reapply structure noise for validation time points
                data = _reapply_structure_noise_for_validation(data, actual_time, val_loader, args)

                # Collect time samples for epoch-level averaging instead of logging per batch
                batch_time_samples.append(t.mean().item())
            else:
                # Time sampling strategy for val_unfixed (should match training strategy)
                if args.time_sampling_strategy == 'uniform':
                    # Uniform sampling between t_min and t_max
                    t = torch.rand(B, device=device) * (args.t_max - args.t_min) + args.t_min
                elif args.time_sampling_strategy == 'exponential':
                    # Exponential sampling for val_unfixed (Dirichlet flow matching alignment)
                    exp_samples = torch.distributions.Exponential(rate=1.0).sample((B,)).to(device)
                    t = 0.0 + exp_samples * args.alpha_range
                    t = torch.clamp(t, min=0.0)
                else:
                    raise ValueError(f"Unknown time_sampling_strategy: {args.time_sampling_strategy}")

            # Sample from Dirichlet distribution, optionally capturing max probability for overfit monitoring
            dirichlet_multiplier = getattr(args, 'dirichlet_multiplier_training', 1.0)
            if args.run_mode == 'overfit_on_one':
                prob_t, max_dirichlet_prob = sample_forward(y, t, generator=val_generator, return_max_prob=True,
                                                          dirichlet_multiplier=dirichlet_multiplier)
                # Collect max probability for epoch-level averaging instead of logging per batch
                batch_max_dirichlet_probs.append(max_dirichlet_prob)

            else:
                prob_t = sample_forward(y, t, generator=val_generator, dirichlet_multiplier=dirichlet_multiplier)

            # Get model predictions for both sequence and DSSP
            # Handle dict return (multi-aux mode), tuple (DSSP-only), or tensor (single head)
            model_obj_val = model.module if hasattr(model, 'module') else model
            _any_new_aux_val = (
                getattr(model_obj_val, 'use_electrostatic_loss', False)
                or getattr(model_obj_val, 'use_geom_topology_loss', False)
            )
            if _any_new_aux_val:
                model_result_val = model(data, t, prob_t)
                if isinstance(model_result_val, dict):
                    pos_pred = model_result_val['sequence']
                    dssp_pred = model_result_val.get('dssp')
                    elec_pred = model_result_val.get('electrostatic')
                    geom_pred = model_result_val.get('geom_topology')
                else:
                    pos_pred = model_result_val
                    dssp_pred = None
                    elec_pred = None
                    geom_pred = None
            elif dssp_targets is not None:
                try:
                    pos_pred, dssp_pred = model(data, t, prob_t, return_dssp=True)
                except:
                    pos_pred = model(data, t, prob_t, return_dssp=False)
                    dssp_pred = None
                elec_pred = None
                geom_pred = None
            else:
                pos_pred = model(data, t, prob_t, return_dssp=False)
                dssp_pred = None
                elec_pred = None
                geom_pred = None

            # Apply virtual node masking if enabled
            if args.use_virtual_node:
                pos_pred_masked, y_masked = apply_geometry_masking(pos_pred, y, data, B, K, device, args.use_virtual_node, v_pred_is_graph_space=True)

                # Extract uncertainty weights if using flexible loss scaling (only for val_unfixed)
                uncertainty_weights_masked = None
                if hasattr(args, 'flexible_loss_scaling') and args.flexible_loss_scaling and val_name == "val_unfixed":
                    uncertainty_weights = data.x_s[:, 6].clone()  # [total_nodes]
                    if args.use_virtual_node:
                        uncertainty_weights[-1] = 0.0  # Virtual node is the last node
                    uncertainty_weights_masked = apply_same_masking_to_weights(
                        uncertainty_weights, data, args.use_virtual_node,
                        y=y, B=B, K=K, device=device
                    )
            else:
                # No virtual nodes: still apply geometry masking for nodes with missing coordinates
                pos_pred_masked, y_masked = apply_geometry_masking(pos_pred, y, data, B, K, device, use_virtual_node=False, v_pred_is_graph_space=True)

                # Extract uncertainty weights if using flexible loss scaling (only for val_unfixed)
                uncertainty_weights_masked = None
                if hasattr(args, 'flexible_loss_scaling') and args.flexible_loss_scaling and val_name == "val_unfixed":
                    uncertainty_weights = data.x_s[:, 6].clone()  # [total_nodes]
                    uncertainty_weights_masked = apply_same_masking_to_weights(
                        uncertainty_weights, data, use_virtual_node=False,
                        y=y, B=B, K=K, device=device
                    )

            # Compute loss with optional flexible loss scaling (only for val_unfixed)
            if hasattr(args, 'flexible_loss_scaling') and args.flexible_loss_scaling and val_name == "val_unfixed" and uncertainty_weights_masked is not None:
                # Use weighted cross-entropy loss for val_unfixed (match training settings for smoothed labels)
                if hasattr(args, 'use_smoothed_labels') and args.use_smoothed_labels:
                    # Smoothed labels: use KL divergence approach weighted by uncertainty (match training)
                    log_probs = F.log_softmax(pos_pred_masked.view(-1, K), dim=-1)  # [num_valid_nodes, K]
                    per_sample_loss = -(y_masked.view(-1, K) * log_probs).sum(dim=-1)  # [num_valid_nodes]
                    weighted_loss = per_sample_loss * uncertainty_weights_masked
                    total_weights = uncertainty_weights_masked.sum()
                    total_loss = weighted_loss.sum() / total_weights if total_weights > 0 else per_sample_loss.mean()
                else:
                    # Hard labels: use standard cross-entropy weighted by uncertainty
                    targets = y_masked.argmax(dim=-1)  # [num_valid_nodes]
                    per_sample_loss = F.cross_entropy(pos_pred_masked.view(-1, K), targets, reduction='none')
                    weighted_loss = per_sample_loss * uncertainty_weights_masked
                    total_weights = uncertainty_weights_masked.sum()
                    total_loss = weighted_loss.sum() / total_weights if total_weights > 0 else per_sample_loss.mean()
            else:
                # Standard unweighted loss (used for val_fixed always, and val_unfixed when flexible scaling is disabled)
                cond_flow = (model.module if hasattr(model, 'module') else model).cond_flow

                # For val_unfixed, match training smoothed label settings; for val_fixed, always use hard labels
                use_smoothed_for_this_phase = (val_name == "val_unfixed" and
                                             hasattr(args, 'use_smoothed_labels') and
                                             args.use_smoothed_labels)

                # Validation always uses hard labels (label_smoothing_factor=0.0)
                # This ensures fair comparison and consistent evaluation metrics
                total_loss = cond_flow.cross_entropy_loss(
                    pos_pred_masked.view(-1, K),     # [num_valid_nodes, K] - raw logits
                    y_masked.view(-1, K),          # [num_valid_nodes, K] - target distributions
                    use_smoothed_labels=use_smoothed_for_this_phase,
                    label_smoothing_factor=0.0  # Always use hard labels for validation
                )

            # Compute accuracy (excluding virtual nodes)
            # Get predicted classes by taking argmax of model predictions
            predicted_classes = pos_pred_masked.argmax(dim=-1)  # [total_real_nodes]
            # Convert one-hot targets to class indices for accuracy calculation
            target_classes = y_masked.argmax(dim=-1)  # [total_real_nodes]
            correct_predictions = (predicted_classes == target_classes).float()
            accuracy = correct_predictions.mean().item()  # Average accuracy for this batch

            # Compute DSSP loss and accuracy if DSSP targets are available
            dssp_loss = 0.0
            dssp_accuracy = 0.0
            dssp_nodes = 0
            if dssp_targets is not None and dssp_pred is not None:
                # Convert DSSP targets list to concatenated tensor if needed
                if isinstance(dssp_targets, (list, tuple)):
                    dssp_targets_concat = torch.cat([t.to(device) for t in dssp_targets if t is not None], dim=0)
                else:
                    dssp_targets_concat = dssp_targets.to(device)

                # Apply proper DSSP masking using dedicated function
                dssp_pred_masked, dssp_targets_masked = apply_dssp_masking(
                    dssp_pred, dssp_targets_concat, data, B, device, args.use_virtual_node, debug=False
                )

                if dssp_pred_masked.size(0) > 0:
                    # Compute DSSP loss using the model's method
                    dssp_loss = (model.module if hasattr(model, 'module') else model).compute_dssp_loss(
                        dssp_pred_masked, dssp_targets_masked
                    ).item()

                    # Compute DSSP accuracy
                    dssp_pred_classes = dssp_pred_masked.argmax(dim=-1)  # [num_valid_nodes]
                    dssp_correct = (dssp_pred_classes == dssp_targets_masked).float()
                    dssp_accuracy = dssp_correct.mean().item()
                    dssp_nodes = dssp_targets_masked.size(0)

                    # Accumulate DSSP metrics
                    val_dssp_loss_sum += dssp_loss * dssp_nodes
                    val_dssp_acc_sum += dssp_accuracy * dssp_nodes
                    val_dssp_node_count += dssp_nodes
                    dssp_batch_count += 1

            # Compute electrostatic and geom_topology aux losses if available.
            # get_valid_node_mask replicates apply_geometry_masking's exclusion logic
            # (virtual nodes, geom_missing, XXX residues) but returns a plain boolean mask
            # so it works with aux logits that have a different last dimension than K=21.
            aa_targets_val = y_masked.argmax(dim=-1)  # [num_valid_nodes], matches pos_pred_masked
            if (elec_pred is not None and getattr(model_obj_val, 'use_electrostatic_loss', False)) or \
               (geom_pred is not None and getattr(model_obj_val, 'use_geom_topology_loss', False)):
                valid_node_mask = get_valid_node_mask(y, data, B, K, device, args.use_virtual_node)

                if elec_pred is not None and getattr(model_obj_val, 'use_electrostatic_loss', False):
                    elec_logits_masked = elec_pred[valid_node_mask]  # [num_valid_nodes, num_elec_classes]
                    elec_label = model_obj_val._electrostatic_label[aa_targets_val]
                    valid_elec = elec_label >= 0
                    if valid_elec.any():
                        elec_logits_valid = elec_logits_masked[valid_elec]
                        elec_label_valid = elec_label[valid_elec]
                        elec_loss_val = F.cross_entropy(elec_logits_valid, elec_label_valid).item()
                        elec_acc_val = (elec_logits_valid.argmax(dim=-1) == elec_label_valid).float().mean().item()
                        n_elec = elec_label_valid.size(0)
                        val_elec_loss_sum += elec_loss_val * n_elec
                        val_elec_acc_sum += elec_acc_val * n_elec
                        val_elec_node_count += n_elec

                if geom_pred is not None and getattr(model_obj_val, 'use_geom_topology_loss', False):
                    geom_logits_masked = geom_pred[valid_node_mask]  # [num_valid_nodes, num_geom_classes]
                    geom_label = model_obj_val._geom_topology_label[aa_targets_val]
                    valid_geom = geom_label >= 0
                    if valid_geom.any():
                        geom_logits_valid = geom_logits_masked[valid_geom]
                        geom_label_valid = geom_label[valid_geom]
                        geom_loss_val = F.cross_entropy(geom_logits_valid, geom_label_valid).item()
                        geom_acc_val = (geom_logits_valid.argmax(dim=-1) == geom_label_valid).float().mean().item()
                        n_geom = geom_label_valid.size(0)
                        val_geom_loss_sum += geom_loss_val * n_geom
                        val_geom_acc_sum += geom_acc_val * n_geom
                        val_geom_node_count += n_geom

            # Collect validation predictions for epoch-level diversity analysis
            val_epoch_predicted_classes.append(predicted_classes.cpu())
            if predicted_classes.numel() > 0:
                pred_probs = torch.softmax(pos_pred_masked, dim=-1)
                pred_entropy = -(pred_probs * torch.log(pred_probs + 1e-8)).sum(dim=-1).mean().item()
                val_epoch_pred_entropies.append(pred_entropy)


            # Weight losses and accuracy by number of real nodes
            real_nodes = y_masked.size(0)
            val_loss_node_sum += total_loss.item() * real_nodes
            val_acc_node_sum += accuracy * real_nodes
            val_node_count += real_nodes
            val_batch_count += 1

    # Calculate average validation losses (simplified for position prediction)
    if val_node_count > 0:
        avg_val_loss = val_loss_node_sum / val_node_count
        avg_val_accuracy = val_acc_node_sum / val_node_count
    else:
        avg_val_loss = float('inf')
        avg_val_accuracy = 0.0

    # Calculate average DSSP metrics
    if val_dssp_node_count > 0:
        avg_val_dssp_loss = val_dssp_loss_sum / val_dssp_node_count
        avg_val_dssp_accuracy = val_dssp_acc_sum / val_dssp_node_count
    else:
        avg_val_dssp_loss = 0.0
        avg_val_dssp_accuracy = 0.0

    # Calculate average aux head metrics
    if val_elec_node_count > 0:
        avg_val_elec_loss = val_elec_loss_sum / val_elec_node_count
        avg_val_elec_accuracy = val_elec_acc_sum / val_elec_node_count
    else:
        avg_val_elec_loss = 0.0
        avg_val_elec_accuracy = 0.0

    if val_geom_node_count > 0:
        avg_val_geom_loss = val_geom_loss_sum / val_geom_node_count
        avg_val_geom_accuracy = val_geom_acc_sum / val_geom_node_count
    else:
        avg_val_geom_loss = 0.0
        avg_val_geom_accuracy = 0.0

    # Compute validation diversity metrics
    val_diversity_metrics = {}
    if val_epoch_predicted_classes:
        all_val_predictions = torch.cat(val_epoch_predicted_classes, dim=0)
        val_unique_predictions = len(torch.unique(all_val_predictions))
        val_diversity_metrics['unique_predictions'] = val_unique_predictions

        # Compute most frequent prediction ratio
        val_prediction_counts = torch.bincount(all_val_predictions, minlength=21)  # 21 amino acids (20 standard + X)
        val_most_frequent_count = val_prediction_counts.max().item()
        val_most_frequent_ratio = val_most_frequent_count / all_val_predictions.size(0)
        val_diversity_metrics['most_frequent_ratio'] = val_most_frequent_ratio

        # Store prediction counts for amino acid analysis (only main process prints)
        # Create amino acid mapping for interpretable output
        aa_names = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                   'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'X']

        # Convert counts to ratios for better interpretation
        total_predictions = all_val_predictions.size(0)
        val_prediction_ratios = val_prediction_counts.float() / total_predictions

        # Get top 3 most predicted amino acids
        top_values, top_indices = torch.topk(val_prediction_counts, k=min(3, (val_prediction_counts > 0).sum().item()))

        # Get bottom 3 least predicted amino acids (only among those with predictions > 0)
        nonzero_mask = val_prediction_counts > 0
        if nonzero_mask.sum() > 0:
            nonzero_counts = val_prediction_counts[nonzero_mask]
            nonzero_indices = torch.where(nonzero_mask)[0]
            bottom_values, bottom_rel_indices = torch.topk(nonzero_counts, k=min(3, nonzero_counts.size(0)), largest=False)
            bottom_indices = nonzero_indices[bottom_rel_indices]
        else:
            bottom_values = torch.tensor([])
            bottom_indices = torch.tensor([])

        val_diversity_metrics['top_predicted_aa'] = [(aa_names[idx.item()], count.item(), val_prediction_ratios[idx.item()].item())
                                                   for idx, count in zip(top_indices, top_values)]
        val_diversity_metrics['bottom_predicted_aa'] = [(aa_names[idx.item()], count.item(), val_prediction_ratios[idx.item()].item())
                                                       for idx, count in zip(bottom_indices, bottom_values)]
    else:
        val_diversity_metrics['unique_predictions'] = 0
        val_diversity_metrics['most_frequent_ratio'] = 0.0

    if val_epoch_pred_entropies:
        val_diversity_metrics['avg_entropy'] = np.mean(val_epoch_pred_entropies)
    else:
        val_diversity_metrics['avg_entropy'] = 0.0

    # Calculate epoch-level averaged metrics for logging
    epoch_avg_metrics = {}
    if batch_time_samples:
        epoch_avg_metrics['avg_time_sampled'] = np.mean(batch_time_samples)
    if batch_max_dirichlet_probs:
        epoch_avg_metrics['avg_max_dirichlet_prob'] = np.mean(batch_max_dirichlet_probs)

    return {
        'avg_loss': avg_val_loss,
        'avg_accuracy': avg_val_accuracy,
        'avg_dssp_loss': avg_val_dssp_loss,
        'avg_dssp_accuracy': avg_val_dssp_accuracy,
        'dssp_batch_count': dssp_batch_count,
        'avg_electrostatic_loss': avg_val_elec_loss,
        'avg_electrostatic_accuracy': avg_val_elec_accuracy,
        'avg_geom_topology_loss': avg_val_geom_loss,
        'avg_geom_topology_accuracy': avg_val_geom_accuracy,
        'diversity_metrics': val_diversity_metrics,
        'batch_count': val_batch_count,
        'epoch_avg_metrics': epoch_avg_metrics
    }


def apply_inpainting_noise(
    x_t: torch.Tensor,
    y: torch.Tensor,
    p_unconditional: float = 1.0,
    inpaint_ratio_min: float = 0.1,
    inpaint_ratio_max: float = 0.3,
) -> tuple:
    """
    Apply inpainting-style training by clamping random residues to clean probabilities.

    This teaches the model to use clean neighbor sequence information by exposing it
    to training scenarios where some neighbors have clean (ground-truth) probabilities
    while others have noisy probabilities from the diffusion process.

    Two training regimes:
    1. Unconditional (probability = p_unconditional): No clamping. Model learns natural dynamics.
    2. Inpainting (probability = 1 - p_unconditional): Random positions clamped to one-hot.
       The fraction clamped is sampled uniformly from [inpaint_ratio_min, inpaint_ratio_max].

    Args:
        x_t: Noisy sequence probabilities from Dirichlet sampling [B, N, K]
        y: Ground truth one-hot encoded sequences [B, N, K]
        p_unconditional: Probability of unconditional training (no clamping).
                         Default 1.0 means feature is disabled (always unconditional).
        inpaint_ratio_min: When inpainting, minimum fraction of positions to clamp (default: 0.1)
        inpaint_ratio_max: When inpainting, maximum fraction of positions to clamp (default: 0.3)

    Returns:
        x_t_inpainted: Modified x_t with some positions clamped to clean [B, N, K]
        inpaint_mask: Boolean mask where True = position was clamped to clean [B, N],
                      or None if p_unconditional >= 1.0
    """
    # If p_unconditional >= 1.0, feature is disabled
    if p_unconditional >= 1.0:
        return x_t, None

    B, N, K = x_t.shape
    device = x_t.device

    # For each sample in batch, decide: unconditional or inpainting?
    is_unconditional = torch.rand(B, device=device) < p_unconditional  # [B]

    # For inpainting samples, sample ratio uniformly from [ratio_min, ratio_max]
    # This is NOT time-dependent - simpler and matches inpainting inference
    uniform_samples = torch.rand(B, device=device)  # [B]
    inpaint_ratios = inpaint_ratio_min + (inpaint_ratio_max - inpaint_ratio_min) * uniform_samples  # [B]

    # Zero out ratio for unconditional samples
    inpaint_ratios = torch.where(is_unconditional, torch.zeros_like(inpaint_ratios), inpaint_ratios)  # [B]

    # Create per-position random values
    random_vals = torch.rand(B, N, device=device)  # [B, N]

    # Create inpainting mask: True where random < ratio (i.e., position is clamped)
    inpaint_mask = random_vals < inpaint_ratios.unsqueeze(1)  # [B, N]

    # Apply clamping: replace noisy positions with one-hot ground truth
    inpaint_mask_expanded = inpaint_mask.unsqueeze(-1).expand_as(x_t)  # [B, N, K]
    x_t_inpainted = torch.where(inpaint_mask_expanded, y, x_t)

    return x_t_inpainted, inpaint_mask


def mask_virtual_nodes_from_batch(data, B):
    """
    Create a mask to identify virtual nodes in a batched graph.
    Virtual nodes are typically added as the last node in each individual graph.

    Args:
        data: PyTorch Geometric Data object with batched graphs
        B: Batch size

    Returns:
        real_node_mask: Boolean tensor where True indicates real nodes, False indicates virtual nodes
    """
    batch_sizes = torch.bincount(data.batch)  # Number of nodes per graph in batch
    real_node_mask = torch.ones(data.batch.size(0), dtype=torch.bool, device=data.batch.device)

    offset = 0
    for b in range(B):
        num_nodes_in_graph = batch_sizes[b].item()
        # Mark the last node in each graph as virtual
        virtual_node_idx = offset + num_nodes_in_graph - 1
        real_node_mask[virtual_node_idx] = False
        offset += num_nodes_in_graph

    return real_node_mask




def apply_virtual_node_masking(v_pred, y, data, B, K, device, use_virtual_node=True):
    """
    Apply virtual node masking to exclude virtual nodes from loss computations.

    Args:
        v_pred: Model predictions [total_nodes, K]
        y: Ground truth [B, N, K]
        data: PyTorch Geometric data object with batch information
        B: Batch size
        K: Number of classes
        device: Device for tensors
        use_virtual_node: Whether virtual nodes are actually being used

    Returns:
        v_pred_masked: Predictions with virtual nodes excluded [total_real_nodes, K]
        y_masked: Ground truth with virtual nodes excluded [total_real_nodes, K]
    """
    # Get batch sizes (number of nodes per graph including virtual nodes)
    batch_sizes = torch.bincount(data.batch)  # [B]

    # Create lists to collect real node data
    v_pred_real_list = []
    y_real_list = []

    v_pred_offset = 0
    for b in range(B):
        num_total_nodes = batch_sizes[b].item()
        # Only subtract virtual node if we're actually using them
        num_real_nodes = num_total_nodes - (1 if use_virtual_node else 0)

        if num_real_nodes > 0:
            # Extract real node predictions
            v_pred_batch = v_pred[v_pred_offset:v_pred_offset+num_real_nodes]
            v_pred_real_list.append(v_pred_batch)

            # Extract corresponding ground truth (first num_real_nodes from y[b])
            y_batch = y[b, :num_real_nodes, :]
            y_real_list.append(y_batch)

        v_pred_offset += num_total_nodes

    # Concatenate all real entries
    if v_pred_real_list:
        v_pred_real = torch.cat(v_pred_real_list, dim=0)
        y_real = torch.cat(y_real_list, dim=0)
    else:
        v_pred_real = torch.empty(0, K, device=device)
        y_real = torch.empty(0, K, device=device)

    return v_pred_real, y_real


def compute_prediction_diversity_metrics(predicted_classes, logits_masked, K, is_smoke_test=False):
    """
    Compute prediction diversity metrics to detect potential shortcuts or mode collapse.

    Args:
        predicted_classes: Predicted class indices [N]
        logits_masked: Model logits [N, K]
        K: Number of classes
        is_smoke_test: Whether to print detailed metrics

    Returns:
        dict: Dictionary containing diversity metrics
    """
    if predicted_classes.numel() == 0:
        return {
            'entropy': 0.0,
            'unique_predictions': 0,
            'most_frequent_ratio': 0.0,
            'prediction_variance': 0.0
        }

    # 1. Prediction entropy - measures diversity of predicted distributions
    pred_probs = torch.softmax(logits_masked, dim=-1)  # [N, K]
    pred_entropy = -(pred_probs * torch.log(pred_probs + 1e-8)).sum(dim=-1).mean().item()

    # 2. Unique predictions count
    unique_predictions = torch.unique(predicted_classes).numel()

    # 3. Most frequent prediction ratio
    pred_counts = torch.bincount(predicted_classes, minlength=K)
    most_frequent_count = pred_counts.max().item()
    most_frequent_ratio = most_frequent_count / predicted_classes.numel()

    # 4. Prediction variance across positions
    pred_var = pred_probs.var(dim=0).mean().item()

    metrics = {
        'entropy': pred_entropy,
        'unique_predictions': unique_predictions,
        'most_frequent_ratio': most_frequent_ratio,
        'prediction_variance': pred_var
    }

    if is_smoke_test:
        print(f"    Prediction diversity metrics:")
        print(f"      Entropy: {pred_entropy:.4f} (higher = more diverse, max={np.log(K):.4f})")
        print(f"      Unique predictions: {unique_predictions}/{K} amino acids")
        print(f"      Most frequent AA ratio: {most_frequent_ratio:.4f} (lower = more diverse)")
        print(f"      Prediction variance: {pred_var:.6f} (higher = more diverse)")

        # Warning thresholds for potential shortcuts
        if pred_entropy < 1.0:  # Very low entropy
            print(f"      WARNING: Very low prediction entropy ({pred_entropy:.4f}) - possible shortcut!")
        if unique_predictions <= 3:  # Only predicting 3 or fewer amino acids
            print(f"      WARNING: Only predicting {unique_predictions} different amino acids - possible shortcut!")
        if most_frequent_ratio > 0.8:  # 80% of positions get same prediction
            print(f"      WARNING: {most_frequent_ratio:.1%} of positions get same prediction - possible shortcut!")

    return metrics



def validate_probability_constraints(y, K, is_smoke_test=False):
    """
    Validate and fix probability constraints in ground truth sequences.

    Args:
        y: Ground truth tensor [B, N, K]
        K: Number of classes
        is_smoke_test: Whether to print detailed information

    Returns:
        y: Fixed ground truth tensor
    """
    B, N, _ = y.shape

    # Check probability constraints and fix if needed
    y_sums = y.sum(dim=-1)  # Shape: [B, N]
    all_close = torch.allclose(y_sums, torch.ones_like(y_sums), atol=1e-5)

    if not all_close:
        if is_smoke_test:
            print(f"WARNING: Found sequences with invalid probability sums. Normalizing...")
            print(f"  y_sums range: [{y_sums.min():.6f}, {y_sums.max():.6f}]")

        # Fix each sequence in the batch
        for b in range(B):
            seq_sums = y[b].sum(dim=-1)  # Shape: [N]

            # Find positions that don't sum to 1
            invalid_positions = ~torch.isclose(seq_sums, torch.ones_like(seq_sums), atol=1e-5)

            if invalid_positions.any():
                # For positions with sum = 0, set to uniform distribution
                zero_sum_positions = (seq_sums == 0)
                if zero_sum_positions.any():
                    y[b, zero_sum_positions] = 1.0 / K

                # For other positions, normalize to sum to 1
                non_zero_invalid = invalid_positions & ~zero_sum_positions
                if non_zero_invalid.any():
                    y[b, non_zero_invalid] = y[b, non_zero_invalid] / seq_sums[non_zero_invalid].unsqueeze(-1)

        # Verify fix
        y_sums_fixed = y.sum(dim=-1)
        assert torch.allclose(y_sums_fixed, torch.ones_like(y_sums_fixed), atol=1e-5), \
            f"Failed to fix probability normalization: min={y_sums_fixed.min()}, max={y_sums_fixed.max()}"

    return y


def save_model_with_metadata(model, output_base, model_name, job_timestamp, metrics, args=None, is_best=False,
                             optimizer=None, scheduler=None, epoch=None, training_state=None,
                             keep_all_best=False):
    """
    Save model with comprehensive metadata and timestamped naming.

    Args:
        model: PyTorch model to save
        output_base: Base output directory
        model_name: Base name for the model
        job_timestamp: Timestamp string for this job (from job start)
        metrics: Dictionary of metrics to save
        args: Training arguments (optional)
        is_best: Whether this is the best model so far
        optimizer: Optimizer state to save (optional)
        scheduler: Learning rate scheduler state to save (optional)
        epoch: Current epoch number (optional)
        training_state: Additional training state dict (optional)

    Returns:
        str: Path where the model was saved
    """
    try:
        # Create filenames based on model type
        if is_best:
            if keep_all_best:
                # Unique file per best — never overwritten
                # Format: best_{model_name}_started_{timestamp}_epoch_{epoch:04d}.pt
                _epoch = epoch if epoch is not None else metrics.get('epoch', 'unknown')
                _epoch_str = f"{_epoch:04d}" if isinstance(_epoch, int) else str(_epoch)
                filename = f"best_{model_name}_started_{job_timestamp}_epoch_{_epoch_str}.pt"
            else:
                # Single best model file that gets overwritten when a better model is found
                # Format: best_so_far_{model_name}_started_{timestamp}.pt
                filename = f"best_so_far_{model_name}_started_{job_timestamp}.pt"
        else:
            # Intermediate models (only saved if --save_intermediate_models is set)
            epoch = metrics.get('epoch', 'unknown')
            filename = f"{job_timestamp}_{model_name}_epoch_{epoch}.pt"

        model_path = os.path.join(output_base, 'saved_models', filename)

        # Ensure the saved_models directory exists
        saved_models_dir = os.path.join(output_base, 'saved_models')
        os.makedirs(saved_models_dir, exist_ok=True)

        # Prepare metadata with error handling for each component
        metadata = {
            'model_state_dict': model.state_dict(),
            'metrics': metrics if metrics is not None else {},
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'is_best': is_best,
            'epoch': epoch if epoch is not None else metrics.get('epoch', 0) if metrics else 0
        }

        # Save optimizer state if provided with error handling
        if optimizer is not None:
            try:
                optimizer_state = optimizer.state_dict()
                if optimizer_state:  # Ensure it's not empty
                    metadata['optimizer_state_dict'] = optimizer_state
                    print(f"   Optimizer state included")
                else:
                    print(f"   Optimizer state is empty, skipping")
            except Exception as e:
                print(f"   Warning: Failed to save optimizer state: {e}")
                print(f"     Continuing without optimizer state")
        else:
            print(f"   No optimizer provided")

        # Save scheduler state if provided with error handling
        if scheduler is not None:
            try:
                scheduler_state = scheduler.state_dict()
                if scheduler_state:  # Ensure it's not empty
                    metadata['scheduler_state_dict'] = scheduler_state
                    print(f"   Scheduler state included")
                else:
                    print(f"   Scheduler state is empty, skipping")
            except Exception as e:
                print(f"   Warning: Failed to save scheduler state: {e}")
                print(f"     Continuing without scheduler state")
        else:
            print(f"   No scheduler provided")

        # Save additional training state if provided with error handling
        if training_state is not None:
            try:
                if isinstance(training_state, dict) and training_state:
                    metadata['training_state'] = training_state
                    print(f"   Training state included ({len(training_state)} items)")
                else:
                    print(f"   Training state is empty or invalid, skipping")
            except Exception as e:
                print(f"   Warning: Failed to save training state: {e}")
                print(f"     Continuing without training state")
        else:
            print(f"   No training state provided")

        # Save training arguments with error handling
        if args is not None:
            try:
                args_dict = vars(args) if hasattr(args, '__dict__') else args
                if args_dict:
                    metadata['args'] = args_dict
                    print(f"   Training arguments included")

                    # Extract and store parameters systematically to avoid omissions
                    try:
                        graph_params = _extract_graph_builder_params(args)
                        model_params = _extract_model_architecture_params(args)

                        # Validate completeness
                        validate_checkpoint_completeness(args, graph_params, model_params)

                        metadata['graph_builder_params'] = graph_params
                        metadata['model_architecture_params'] = model_params
                        print(f"   Graph and model parameters included")

                    except Exception as e:
                        print(f"   Warning: Failed to extract graph/model parameters: {e}")
                        print(f"     Basic args still saved, but parameter extraction failed")

                else:
                    print(f"   Training arguments are empty, skipping")
            except Exception as e:
                print(f"     Warning: Failed to save training arguments: {e}")
                print(f"     Continuing without training arguments")
        else:
            print(f"    No training arguments provided")

        # Save RNG states so a resumed run can reproduce the exact random trajectory (H-3 fix).
        try:
            import random
            import numpy as np
            rng_states = {
                'torch': torch.random.get_rng_state(),
                'numpy': np.random.get_state(),
                'python': random.getstate(),
            }
            if torch.cuda.is_available():
                rng_states['cuda'] = torch.cuda.get_rng_state_all()
            metadata['rng_states'] = rng_states
            print(f"   RNG states included")
        except Exception as e:
            print(f"   Warning: Failed to save RNG states: {e}")

        # Save the main model file
        torch.save(metadata, model_path)

        # Log what was saved
        if is_best:
            print(f"  Best model saved: {filename} (will be overwritten by future best models)")
        else:
            print(f"  Model saved: {filename}")

        print(f"[DEBUG] Full model path: {model_path}", flush=True)

        return model_path

    except Exception as e:
        print(f"Error saving model: {e}")
        fallback_path = f"{model_name}_fallback.pt"
        torch.save(model, fallback_path)
        print(f"  Fallback save: {fallback_path}")
        return fallback_path


def load_model_with_metadata(model_path):
    """
    Load model with metadata.

    Args:
        model_path: Path to the saved model

    Returns:
        tuple: (model_state_dict, metadata) or (model, None) for legacy saves
    """
    try:
        loaded = torch.load(model_path, map_location='cpu', weights_only=False)

        if isinstance(loaded, dict) and 'model_state_dict' in loaded:
            # New format with metadata
            return loaded['model_state_dict'], loaded
        else:
            # Legacy format - just the model
            return loaded, None

    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None, None


def validate_model_architecture_compatibility(checkpoint_params, current_args):
    """
    Validate that model architecture parameters from checkpoint are compatible with current arguments.
    Warns for any mismatches in parameters that affect model structure and architecture.

    Args:
        checkpoint_params: Dictionary containing parameters from checkpoint metadata
        current_args: Current command line arguments namespace

    Returns:
        bool: True if compatible, False if critical incompatibilities found
    """
    # Parameters that affect model architecture and must match
    architecture_critical_params = [
        'hidden_dim',           # Core hidden dimension
        'hidden_dim_v',         # Vector hidden dimension
        'node_dim_s',           # Scalar node dimension
        'node_dim_v',           # Vector node dimension
        'edge_dim_s',           # Scalar edge dimension
        'edge_dim_v',           # Vector edge dimension
        'num_layers_gvp',       # Number of GVP layers
        'num_message_layers',   # Number of message passing layers
        'num_layers_prediction', # Number of prediction layers
        'architecture',         # Model architecture type
        'use_qkv',             # QKV attention usage
        'head_hidden',         # Prediction head hidden dimension
    ]

    warnings_found = []
    critical_mismatches = []

    print("=== Model Architecture Compatibility Check ===")

    for param_name in architecture_critical_params:
        # Get values from both sources
        checkpoint_value = checkpoint_params.get(param_name)
        current_value = getattr(current_args, param_name, None)

        # Skip if parameter is not present in either source
        if checkpoint_value is None and current_value is None:
            continue

        # Check for mismatches
        if checkpoint_value is not None and current_value is not None:
            if checkpoint_value != current_value:
                warning_msg = f"  [warn] {param_name}: checkpoint={checkpoint_value} vs current={current_value}"
                warnings_found.append(warning_msg)

                # Determine if this is a critical mismatch that will cause model loading to fail
                if param_name in ['hidden_dim', 'hidden_dim_v', 'node_dim_s', 'node_dim_v',
                                 'edge_dim_s', 'edge_dim_v', 'num_layers_gvp', 'head_hidden']:
                    critical_mismatches.append(param_name)

        elif checkpoint_value is not None:
            info_msg = f"  [info] {param_name}: checkpoint={checkpoint_value}, current=None (will use checkpoint value)"
            print(info_msg)
        elif current_value is not None:
            info_msg = f"  [info] {param_name}: checkpoint=None, current={current_value} (checkpoint may use default)"
            print(info_msg)

    # Report findings
    if warnings_found:
        print("\n[warning] ARCHITECTURE PARAMETER MISMATCHES DETECTED:")
        for warning in warnings_found:
            print(warning)

        if critical_mismatches:
            print(f"\n[error] CRITICAL INCOMPATIBILITIES: {critical_mismatches}")
            print("These mismatches will likely cause model weight loading to fail!")
            print("Please ensure the checkpoint was trained with compatible architecture parameters.")
            return False
        else:
            print("\n[warn] Non-critical mismatches found - model may load but behavior could differ")
            return True
    else:
        print("[ok] No architecture parameter conflicts detected")
        return True


def load_checkpoint_for_training(checkpoint_path: str, device: torch.device, args, model=None):
    """
    Load a checkpoint for training resumption with comprehensive parameter extraction and override support.
    Based on the robust loading logic from sample_utils.py but extended for training use.

    Args:
        checkpoint_path: Path to the checkpoint file
        device: Target device to load tensors onto
        args: Current command line arguments (can override checkpoint parameters)
        model: Optional model to load state into (if None, returns parameters for model creation)

    Returns:
        dict: Comprehensive loading results containing:
            - model_state_dict: Model weights
            - optimizer_state_dict: Optimizer state (if available)
            - scheduler_state_dict: Scheduler state (if available)
            - epoch: Last epoch number (if available)
            - metrics: Training metrics (if available)
            - merged_args: Merged arguments (checkpoint + command line overrides)
            - training_state: Additional training state (if available)
    """
    print(f"Loading checkpoint for training resumption from: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Initialize results
    results = {
        'model_state_dict': None,
        'optimizer_state_dict': None,
        'scheduler_state_dict': None,
        'epoch': 0,
        'metrics': {},
        'merged_args': None,
        'training_state': {}
    }

    print("="*60)
    print("CHECKPOINT LOADING FOR TRAINING RESUMPTION")
    print("="*60)

    # Extract model state dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # New format with metadata
        results['model_state_dict'] = checkpoint['model_state_dict']

        # Gracefully handle optional components
        results['optimizer_state_dict'] = checkpoint.get('optimizer_state_dict')
        results['scheduler_state_dict'] = checkpoint.get('scheduler_state_dict')
        results['epoch'] = checkpoint.get('epoch', 0)
        results['training_state'] = checkpoint.get('training_state', {})

        # Handle metrics with fallback for different possible keys
        metrics_keys = ['metrics', 'best_metrics', 'training_metrics']
        results['metrics'] = {}
        for key in metrics_keys:
            if key in checkpoint and checkpoint[key] is not None:
                if isinstance(checkpoint[key], dict):
                    results['metrics'].update(checkpoint[key])
                    break

        print("Found modern checkpoint format with metadata")
        print(f"  Epoch: {results['epoch']}")
        print(f"  Has optimizer state: {results['optimizer_state_dict'] is not None}")
        print(f"  Has scheduler state: {results['scheduler_state_dict'] is not None}")
        print(f"  Available metrics: {list(results['metrics'].keys()) if results['metrics'] else 'None'}")

        # Report missing optional components
        missing_components = []
        if results['optimizer_state_dict'] is None:
            missing_components.append('optimizer_state_dict')
        if results['scheduler_state_dict'] is None:
            missing_components.append('scheduler_state_dict')
        if not results['metrics']:
            missing_components.append('metrics')

        if missing_components:
            print(f"  Missing optional components: {missing_components}")
            print("  → Will use fresh initialization for missing components")

        # Restore RNG states so the resumed run continues the same random trajectory (H-3 fix).
        if 'rng_states' in checkpoint:
            try:
                import random
                import numpy as np
                rng = checkpoint['rng_states']
                torch.random.set_rng_state(rng['torch'])
                np.random.set_state(rng['numpy'])
                random.setstate(rng['python'])
                if 'cuda' in rng and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(rng['cuda'])
                print("  RNG states restored")
            except Exception as e:
                print(f"  Warning: Failed to restore RNG states: {e}")
        else:
            print("  No RNG states in checkpoint (resumed run will diverge stochastically)")

    else:
        # Legacy format - just the model or direct state dict
        if hasattr(checkpoint, 'state_dict'):
            results['model_state_dict'] = checkpoint.state_dict()
        elif isinstance(checkpoint, dict):
            results['model_state_dict'] = checkpoint
        else:
            results['model_state_dict'] = checkpoint

        print("Found legacy checkpoint format (model weights only)")
        print("  → Optimizer, scheduler, and metrics will use fresh initialization")

    # Extract and merge training arguments with comprehensive fallback
    checkpoint_args = None
    args_sources = ['args', 'training_args', 'hyperparams', 'config']

    if isinstance(checkpoint, dict):
        for source in args_sources:
            if source in checkpoint and checkpoint[source] is not None:
                checkpoint_args = checkpoint[source]
                print(f"Found training arguments in checkpoint (key: '{source}')")
                break

    if checkpoint_args is None:
        print("No training arguments found in checkpoint")
        print("  → Using command line arguments only")

    # Create merged arguments with command line overrides
    if checkpoint_args is not None:
        try:
            # Convert to dict if needed
            if hasattr(checkpoint_args, '__dict__'):
                checkpoint_dict = vars(checkpoint_args)
            elif isinstance(checkpoint_args, dict):
                checkpoint_dict = checkpoint_args.copy()
            else:
                print(f"Warning: Unexpected checkpoint args format: {type(checkpoint_args)}")
                checkpoint_dict = {}

            # Start with checkpoint arguments
            merged_dict = checkpoint_dict.copy()

        except Exception as e:
            print(f"Warning: Failed to process checkpoint arguments: {e}")
            print("  → Using command line arguments only")
            checkpoint_dict = {}
            merged_dict = {}

        # Override with command line arguments (where provided)
        if hasattr(args, '__dict__'):
            try:
                args_dict = vars(args)
                overridden_params = []

                for key, value in args_dict.items():
                    # Only override if the command line value is explicitly set (not None/default)
                    if value is not None and key in checkpoint_dict:
                        if checkpoint_dict[key] != value:
                            overridden_params.append(f"{key}: {checkpoint_dict[key]} -> {value}")
                            merged_dict[key] = value
                    elif value is not None:
                        # New parameter not in checkpoint
                        merged_dict[key] = value

                if overridden_params:
                    print(f"\nCommand line overrides applied:")
                    for param in overridden_params:
                        print(f"  {param}")
                elif checkpoint_dict:
                    print("No parameter overrides needed (command line matches checkpoint)")

            except Exception as e:
                print(f"Warning: Failed to apply command line overrides: {e}")
                print("  → Using checkpoint arguments as-is")

        try:
            # Convert back to args-like object with error handling
            class MergedArgs:
                def __init__(self, **kwargs):
                    for key, value in kwargs.items():
                        setattr(self, key, value)

            results['merged_args'] = MergedArgs(**merged_dict)

        except Exception as e:
            print(f"Warning: Failed to create merged arguments: {e}")
            print("  → Falling back to command line arguments only")
            results['merged_args'] = args

    else:
        print("Using command line arguments only")
        results['merged_args'] = args

    # Validate model architecture compatibility before loading model weights
    print("\n" + "="*60)
    print("ARCHITECTURE COMPATIBILITY VALIDATION")
    print("="*60)

    if checkpoint_args is not None:
        # Use the merged arguments to get the most current values
        merged_args = results['merged_args']

        # Extract all checkpoint parameters (including architecture params)
        checkpoint_dict = {}
        if hasattr(checkpoint_args, '__dict__'):
            checkpoint_dict = vars(checkpoint_args)
        elif isinstance(checkpoint_args, dict):
            checkpoint_dict = checkpoint_args

        # Validate compatibility
        is_compatible = validate_model_architecture_compatibility(checkpoint_dict, merged_args)

        if not is_compatible:
            raise ValueError(
                "Critical model architecture incompatibilities detected! "
                "Cannot load model weights due to parameter mismatches. "
                "Please use a checkpoint trained with compatible architecture parameters."
            )
        print("="*60)
    else:
        print("No checkpoint architecture parameters to validate - using current arguments")
        print("="*60)

    # Load model state if model provided
    if model is not None and results['model_state_dict'] is not None:
        # Handle potential device mismatches first
        state_dict = results['model_state_dict']

        if not isinstance(state_dict, dict):
            raise ValueError(f"Invalid model state_dict format: {type(state_dict)}")

        # Handle DataParallel/DistributedDataParallel module prefix issue
        has_module_prefix = any(key.startswith('module.') for key in state_dict.keys())
        all_have_module_prefix = all(key.startswith('module.') for key in state_dict.keys())
        no_module_prefix = not has_module_prefix

        # Check if current model is wrapped in DistributedDataParallel
        is_ddp_model = hasattr(model, 'module')

        if all_have_module_prefix and not is_ddp_model:
            # Checkpoint has 'module.' prefix but model is not DDP wrapped - remove prefix
            print("Detected DataParallel checkpoint loading into non-distributed model - removing 'module.' prefix")
            state_dict = {key[7:]: value for key, value in state_dict.items()}
            print(f"✓ Converted {len(state_dict)} parameter keys")
        elif no_module_prefix and is_ddp_model:
            # Checkpoint has no 'module.' prefix but model is DDP wrapped - add prefix
            print("Detected non-distributed checkpoint loading into DistributedDataParallel model - adding 'module.' prefix")
            state_dict = {f'module.{key}': value for key, value in state_dict.items()}
            print(f"✓ Converted {len(state_dict)} parameter keys")

        # Move state dict to target device if needed
        if device is not None:
            try:
                state_dict = {k: v.to(device) if torch.is_tensor(v) else v
                             for k, v in state_dict.items()}
            except Exception as e:
                print(f"Warning: Failed to move some model tensors to {device}: {e}")
                print("  → Attempting to load with original device placement")

        # Load model weights. Use strict=True by default.
        # Exception: if only the new auxiliary head parameters are missing (when loading
        # a baseline checkpoint for fine-tuning with new heads), use strict=False.
        _AUX_HEAD_PREFIXES = (
            'electrostatic_final.',
            'geom_topology_final.',
            'burial_final.',
            '_electrostatic_label',
            '_geom_topology_label',
        )

        try:
            model.load_state_dict(state_dict, strict=True)
            print(f"✓ Model state loaded perfectly onto {device}")

        except RuntimeError as strict_error:
            # Check if only new aux head keys are missing (expected when fine-tuning
            # from a baseline checkpoint that predates the auxiliary heads)
            try:
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            except Exception:
                missing_keys, unexpected_keys = [], []

            only_new_aux_missing = (
                len(missing_keys) > 0
                and all(any(k.startswith(p) for p in _AUX_HEAD_PREFIXES) for k in missing_keys)
                and len(unexpected_keys) == 0
            )

            if only_new_aux_missing:
                model.load_state_dict(state_dict, strict=False)
                print(f"✓ Loaded checkpoint with strict=False (new aux head weights initialized fresh).")
                print(f"  Missing keys (new, expected): {missing_keys}")
            else:
                # Genuine architecture mismatch — provide detailed info then re-raise
                print(f"✗ CRITICAL ERROR: Model checkpoint is incompatible with current model architecture!")
                print(f"  Error details: {strict_error}")
                if missing_keys:
                    print(f"  Missing keys (model parameters not found in checkpoint):")
                    for key in missing_keys[:10]:  # Show first 10
                        print(f"    - {key}")
                    if len(missing_keys) > 10:
                        print(f"    ... and {len(missing_keys) - 10} more")

                if unexpected_keys:
                    print(f"  Unexpected keys (checkpoint parameters not in current model):")
                    for key in unexpected_keys[:10]:  # Show first 10
                        print(f"    - {key}")
                    if len(unexpected_keys) > 10:
                        print(f"    ... and {len(unexpected_keys) - 10} more")

                print(f"  → This likely indicates model architecture mismatch or corrupted checkpoint")
                print(f"  → Training cannot continue with missing model weights/biases")

                # Re-raise the original strict loading error to fail fast
                raise strict_error

    elif model is not None:
        print("✗ CRITICAL ERROR: No model state found in checkpoint")
        print("  → Cannot resume training without model weights")
        raise ValueError("Checkpoint must contain model weights ('model_state_dict') for training resumption")

    print("="*60)
    return results


def load_optimizer_and_scheduler_state(optimizer, scheduler, checkpoint_results, device):
    """
    Load optimizer and scheduler state from checkpoint results with comprehensive error handling.
    Gracefully handles missing or corrupted optimizer/scheduler states.

    Args:
        optimizer: PyTorch optimizer to load state into
        scheduler: Learning rate scheduler to load state into (can be None)
        checkpoint_results: Results dict from load_checkpoint_for_training
        device: Target device

    Returns:
        dict: Status information about what was loaded
    """
    status = {
        'optimizer_loaded': False,
        'scheduler_loaded': False,
        'warnings': [],
        'errors': []
    }

    print("="*50)
    print("LOADING OPTIMIZER AND SCHEDULER STATE")
    print("="*50)

    # Load optimizer state with comprehensive error handling
    if checkpoint_results.get('optimizer_state_dict') is not None:
        try:
            opt_state = checkpoint_results['optimizer_state_dict']

            # Validate optimizer state structure
            if not isinstance(opt_state, dict):
                raise ValueError(f"Invalid optimizer state format: {type(opt_state)}")

            if 'state' not in opt_state or 'param_groups' not in opt_state:
                raise ValueError("Optimizer state missing required keys ('state' or 'param_groups')")

            # Handle device placement for optimizer state tensors
            try:
                for param_id, state in opt_state['state'].items():
                    if isinstance(state, dict):
                        for key, value in state.items():
                            if torch.is_tensor(value):
                                state[key] = value.to(device)

                print(f"Moved optimizer state tensors to {device}")

            except Exception as e:
                # Device movement failed (e.g. OOM).  Loading tensors that are on the wrong device
                # would silently produce NaN parameter updates on the first opt.step().
                # Safest non-disruptive fix: zero out all state tensors in-place so the optimizer
                # loads a well-defined (fresh) state on the correct device rather than a mismatched
                # one — effectively equivalent to a fresh optimizer but without crashing (M-3 fix).
                status['warnings'].append(f"Failed to move optimizer tensors to device: {e}")
                print(f"Warning: {status['warnings'][-1]}")
                print("Zeroing optimizer state tensors to prevent device-mismatch NaNs; "
                      "optimizer will behave as if freshly initialized.")
                for param_id, state in opt_state['state'].items():
                    if isinstance(state, dict):
                        for key, value in list(state.items()):
                            if torch.is_tensor(value):
                                state[key] = torch.zeros_like(value, device=device)

            # Attempt to load optimizer state
            optimizer.load_state_dict(opt_state)
            status['optimizer_loaded'] = True
            print("✓ Optimizer state loaded successfully")

            # Report what was restored
            num_param_groups = len(opt_state.get('param_groups', []))
            num_states = len(opt_state.get('state', {}))
            print(f"  → Restored {num_param_groups} parameter groups and {num_states} parameter states")

        except Exception as e:
            error_msg = f"Failed to load optimizer state: {e}"
            status['errors'].append(error_msg)
            print(f"✗ Error: {error_msg}")
            print("  → Continuing with fresh optimizer state (momentum will be reset)")

    else:
        print("✗ No optimizer state found in checkpoint")
        print("  → Using fresh optimizer initialization")

    # Load scheduler state with comprehensive error handling
    if scheduler is not None:
        if checkpoint_results.get('scheduler_state_dict') is not None:
            try:
                sched_state = checkpoint_results['scheduler_state_dict']

                # Validate scheduler state structure
                if not isinstance(sched_state, dict):
                    raise ValueError(f"Invalid scheduler state format: {type(sched_state)}")

                # Attempt to load scheduler state
                scheduler.load_state_dict(sched_state)
                status['scheduler_loaded'] = True
                print("✓ Scheduler state loaded successfully")

                # Report what was restored
                if '_step_count' in sched_state:
                    print(f"  → Restored scheduler at step {sched_state['_step_count']}")
                if 'best' in sched_state:
                    print(f"  → Restored best metric: {sched_state['best']}")

            except Exception as e:
                error_msg = f"Failed to load scheduler state: {e}"
                status['errors'].append(error_msg)
                print(f"✗ Error: {error_msg}")
                print("  → Continuing with fresh scheduler state")

        else:
            print("✗ No scheduler state found in checkpoint")
            print("  → Using fresh scheduler initialization")
    else:
        print("○ No scheduler provided (scheduler=None)")

    # Summary
    print("="*50)
    components_loaded = []
    if status['optimizer_loaded']:
        components_loaded.append('optimizer')
    if status['scheduler_loaded']:
        components_loaded.append('scheduler')

    if components_loaded:
        print(f"✓ Successfully loaded: {', '.join(components_loaded)}")
    else:
        print("○ No optimizer or scheduler state loaded (will use fresh initialization)")

    if status['warnings']:
        print(f"[warn] Warnings: {len(status['warnings'])}")
    if status['errors']:
        print(f"✗ Errors: {len(status['errors'])}")

    return status


def cleanup_old_checkpoints(output_base, model_name, keep_last_n=5):
    """
    Clean up old checkpoint files, keeping only the most recent ones.

    Args:
        output_base: Base output directory
        model_name: Base model name to match
        keep_last_n: Number of recent checkpoints to keep
    """
    try:
        saved_models_dir = os.path.join(output_base, 'saved_models')
        if not os.path.exists(saved_models_dir):
            return

        # Find all checkpoint files for this model
        checkpoint_files = []
        for filename in os.listdir(saved_models_dir):
            if filename.startswith(model_name) and filename.endswith('.pt') and 'epoch_' in filename:
                file_path = os.path.join(saved_models_dir, filename)
                checkpoint_files.append((filename, file_path, os.path.getctime(file_path)))

        # Sort by creation time (newest first)
        checkpoint_files.sort(key=lambda x: x[2], reverse=True)

        # Remove old files
        for i, (filename, file_path, _) in enumerate(checkpoint_files):
            if i >= keep_last_n:
                try:
                    os.remove(file_path)
                    print(f"  Cleaned up old checkpoint: {filename}")
                except Exception as e:
                    print(f"  Failed to remove {filename}: {e}")

    except Exception as e:
        print(f"Error during checkpoint cleanup: {e}")




def _to_packed_nodes(tensor, batch_sizes, K):
    """
    Convert a [B, N, K] or [B, N] tensor to packed node structure [total_nodes, K] or [total_nodes].
    Handles variable N per batch using batch_sizes.
    """
    B = batch_sizes.size(0)
    packed = []
    for b in range(B):
        n = batch_sizes[b].item()
        if tensor.dim() == 3:
            packed.append(tensor[b, :n, :])
        elif tensor.dim() == 2:
            if K is None:
                # Boolean mask case - just take the first n elements
                packed.append(tensor[b, :n])
            else:
                packed.append(tensor[b, :n])
        else:
            raise ValueError(f"Unsupported tensor shape for packing: {tensor.shape}")
    return torch.cat(packed, dim=0)



def apply_same_masking_to_weights(uncertainty_weights, data, use_virtual_node=True, y=None, B=None, K=None, device=None):

    """
    Apply the same masking logic to uncertainty weights as apply_geometry_masking applies to predictions.
    This ensures the weights correspond exactly to the masked predictions.

    Args:
        uncertainty_weights: Per-node uncertainty weights [total_nodes]
        data: PyTorch Geometric data object
        use_virtual_node: Whether virtual nodes are being used
        y: Ground truth sequences [B, N, K] - needed for unknown residue detection
        B: Batch size - needed for unknown residue detection
        K: Number of classes - needed for unknown residue detection
        device: Device for tensors - needed for unknown residue detection

    Returns:
        uncertainty_weights_masked: Weights for valid nodes only [valid_nodes]
    """

    # Use the same batch size calculation as apply_geometry_masking
    batch_sizes = torch.bincount(data.batch)  # [B] - total nodes per batch including virtual
    total_nodes = data.batch.size(0)

    # Step 1: Create mask to exclude virtual nodes (same logic as apply_geometry_masking)
    real_node_mask = torch.ones(total_nodes, dtype=torch.bool, device=uncertainty_weights.device)
    if use_virtual_node:
        node_offset = 0
        for b in range(B):
            num_total_nodes = batch_sizes[b].item()
            virtual_node_idx = node_offset + num_total_nodes - 1
            real_node_mask[virtual_node_idx] = False
            node_offset += num_total_nodes

    # Step 2: Apply geometry masking to real nodes only
    uncertainty_weights_real = uncertainty_weights[real_node_mask]  # [total_real_nodes]

    # Create final mask starting with all real nodes
    final_mask = torch.ones(uncertainty_weights_real.size(0), dtype=torch.bool, device=uncertainty_weights.device)

    # Apply geometry missing mask
    if hasattr(data, 'geom_missing') and data.geom_missing is not None:
        if data.geom_missing.size(0) == uncertainty_weights_real.size(0):
            final_mask = final_mask & ~data.geom_missing
        else:
            print(f"WARNING: geom_missing size {data.geom_missing.size(0)} doesn't match real nodes {uncertainty_weights_real.size(0)}")

    # Step 3: Apply unknown residue masking (same logic as apply_geometry_masking)
    if y is not None and B is not None and K is not None and device is not None:
        # Calculate real batch sizes (excluding virtual nodes)
        real_batch_sizes = torch.zeros_like(batch_sizes)
        for b in range(B):
            num_total_nodes = batch_sizes[b].item()
            num_real_nodes = num_total_nodes - (1 if use_virtual_node else 0)
            real_batch_sizes[b] = num_real_nodes

        # Identify unknown residues ('X' or class 20) - exact same logic as apply_geometry_masking
        if y.dim() == 3:  # [B, N, K] format
            # Find positions where class 20 (unknown) has probability > 0.5
            unknown_mask_padded = (y[:, :, 20] > 0.5)  # [B, N]
            # Convert to packed format using real node counts
            unknown_mask_packed = _to_packed_nodes(unknown_mask_padded, real_batch_sizes, None)  # [total_real_nodes]
            unknown_mask_packed = unknown_mask_packed.to(device)
        else:
            # Already in packed format [total_real_nodes, K]
            unknown_mask_packed = (y[:, 20] > 0.5)  # [total_real_nodes]
            unknown_mask_packed = unknown_mask_packed.to(device)

        # Apply unknown residue mask
        if unknown_mask_packed.size(0) == uncertainty_weights_real.size(0):
            final_mask = final_mask & ~unknown_mask_packed
        else:
            print(f"WARNING: unknown_mask size {unknown_mask_packed.size(0)} doesn't match real nodes {uncertainty_weights_real.size(0)}")

    return uncertainty_weights_real[final_mask]

def apply_dssp_masking(dssp_logits, dssp_targets, data, B, device, use_virtual_node=True, debug=False):
    """
    Apply masking to DSSP predictions and targets, following the same logic as sequence masking
    but adapted for DSSP-specific data formats.

    Args:
        dssp_logits: DSSP predictions [total_nodes, 10]
        dssp_targets: DSSP target indices [total_nodes] (as tensor, not list)
        data: PyTorch Geometric data object
        B: Batch size
        device: Device for tensors
        use_virtual_node: Whether virtual nodes are being used

    Returns:
        Tuple of (dssp_logits_masked, dssp_targets_masked)
    """
    # Remove virtual nodes if enabled
    if use_virtual_node:
        # Virtual nodes are the last node in each graph in the batch
        batch_sizes = torch.bincount(data.batch)  # Number of nodes per graph
        virtual_indices = []

        for i in range(B):
            if i == 0:
                virtual_idx = batch_sizes[0] - 1
            else:
                virtual_idx = batch_sizes[:i+1].sum() - 1
            virtual_indices.append(virtual_idx.item())

        # Create mask to exclude virtual nodes
        real_node_mask = torch.ones(dssp_logits.size(0), dtype=torch.bool, device=device)
        for idx in virtual_indices:
            real_node_mask[idx] = False

        # DEBUG: Print shapes to diagnose the mismatch (only in debug mode)
        if debug:
            if _IFD_DEBUG: print(f"DEBUG DSSP MASKING:")
            print(f"  dssp_logits.shape: {dssp_logits.shape}")
            print(f"  dssp_targets.shape: {dssp_targets.shape}")
            print(f"  real_node_mask.shape: {real_node_mask.shape}")
            print(f"  virtual_indices: {virtual_indices}")
            print(f"  use_virtual_node: {use_virtual_node}")
            print(f"  batch_sizes: {batch_sizes}")

        # PROPER FIX: Ensure dssp_targets includes virtual node placeholders
        if real_node_mask.size(0) != dssp_targets.size(0):
            if debug:
                print(f"ERROR: Shape mismatch detected!")
                print(f"  real_node_mask size: {real_node_mask.size(0)}")
                print(f"  dssp_targets size: {dssp_targets.size(0)}")
                print(f"  Difference: {real_node_mask.size(0) - dssp_targets.size(0)}")

            # Import DSSP constants to use proper unknown class
            from data.dssp_constants import DSSP_TO_IDX

            # Instead of truncating dssp_logits, expand dssp_targets to include virtual node placeholders
            # Use DSSP 'X' class (index 9) as placeholder for virtual nodes (proper unknown/mask class)
            expanded_targets = torch.full((real_node_mask.size(0),), DSSP_TO_IDX['X'], dtype=dssp_targets.dtype, device=device)

            # Fill in the real protein positions (excluding virtual nodes)
            real_positions = 0
            for i in range(real_node_mask.size(0)):
                if real_node_mask[i]:  # This is a real protein node
                    if real_positions < dssp_targets.size(0):
                        expanded_targets[i] = dssp_targets[real_positions]
                        real_positions += 1
                # Virtual nodes keep their 'X' class values (index 9, will be masked out anyway)

            dssp_targets = expanded_targets
            if debug:
                print(f"  Expanded dssp_targets to size: {dssp_targets.size(0)} with DSSP 'X' class (index {DSSP_TO_IDX['X']}) for virtual nodes")

        dssp_logits_no_virtual = dssp_logits[real_node_mask]
        dssp_targets_no_virtual = dssp_targets[real_node_mask]
    else:
        dssp_logits_no_virtual = dssp_logits
        dssp_targets_no_virtual = dssp_targets

    # Apply geometry masking
    if hasattr(data, 'geom_missing') and data.geom_missing is not None:
        geom_missing = data.geom_missing

        # If virtual nodes were used, geom_missing needs to be adjusted
        if use_virtual_node:
            # geom_missing is for real nodes only, so it should match dssp_*_no_virtual
            geom_valid = ~geom_missing
        else:
            # No virtual nodes, geom_missing applies directly
            geom_valid = ~geom_missing

        # Apply geometry mask
        if geom_valid.size(0) == dssp_logits_no_virtual.size(0):
            dssp_logits_masked = dssp_logits_no_virtual[geom_valid]
            dssp_targets_masked = dssp_targets_no_virtual[geom_valid]
        else:
            print(f"WARNING: Geometry mask size {geom_valid.size(0)} doesn't match DSSP size {dssp_logits_no_virtual.size(0)}")
            dssp_logits_masked = dssp_logits_no_virtual
            dssp_targets_masked = dssp_targets_no_virtual
    else:
        # No geometry masking needed
        dssp_logits_masked = dssp_logits_no_virtual
        dssp_targets_masked = dssp_targets_no_virtual

    return dssp_logits_masked, dssp_targets_masked


def get_valid_node_mask(y, data, B, K, device, use_virtual_node=True):
    """
    Return a boolean mask [total_graph_nodes] selecting nodes that are valid for loss computation:
    excludes virtual nodes, geom_missing nodes, and unknown (XXX/class-20) residues.
    This is the same exclusion logic as apply_geometry_masking but without any shape assertion,
    so it can be used with aux logit tensors that have a different last dimension than K=21.
    """
    batch_sizes = torch.bincount(data.batch)  # [B]
    total_nodes = data.batch.size(0)

    # Step 1: exclude virtual nodes
    real_node_mask = torch.ones(total_nodes, dtype=torch.bool, device=device)
    if use_virtual_node:
        node_offset = 0
        for b in range(B):
            num_total_nodes = batch_sizes[b].item()
            real_node_mask[node_offset + num_total_nodes - 1] = False
            node_offset += num_total_nodes

    # Step 2: exclude unknown residues (class 20) using y [B, N, K]
    real_batch_sizes = torch.zeros_like(batch_sizes)
    for b in range(B):
        num_total_nodes = batch_sizes[b].item()
        real_batch_sizes[b] = num_total_nodes - (1 if use_virtual_node else 0)
    unknown_mask_packed = _to_packed_nodes((y[:, :, 20] > 0.5), real_batch_sizes, None).to(device)  # [total_real_nodes]

    # Step 3: exclude geom_missing nodes
    total_real_nodes = real_node_mask.sum().item()
    geom_missing_mask = torch.zeros(total_real_nodes, dtype=torch.bool, device=device)
    if hasattr(data, 'geom_missing') and data.geom_missing is not None:
        if data.geom_missing.size(0) == total_real_nodes:
            geom_missing_mask = data.geom_missing.bool().to(device)

    # Build final mask over real nodes only
    final_real_mask = torch.ones(total_real_nodes, dtype=torch.bool, device=device)
    if unknown_mask_packed.size(0) == total_real_nodes:
        final_real_mask = final_real_mask & ~unknown_mask_packed
    if geom_missing_mask.size(0) == total_real_nodes:
        final_real_mask = final_real_mask & ~geom_missing_mask

    # Embed back into full-node boolean mask [total_nodes]
    full_mask = torch.zeros(total_nodes, dtype=torch.bool, device=device)
    full_mask[real_node_mask] = final_real_mask
    return full_mask


def apply_geometry_masking(v_pred, y, data, B, K, device, use_virtual_node=True, v_pred_is_graph_space=True,
                           inpaint_mask=None):
    """
    Apply comprehensive masking to exclude virtual nodes, geometry-missing nodes,
    and 'X' (unknown) residue nodes from velocity loss computations.

    This function creates a proper alignment between graph nodes (v_pred) and
    sequence positions (y) by:
    1. Excluding virtual nodes from graph predictions
    2. Excluding nodes with missing geometry from both predictions and targets
    3. Excluding 'X' (unknown) residues from both predictions and targets
    4. Optionally: also process an inpainting mask to exclude clamped positions from loss

    Args:
        v_pred: Model velocity predictions [total_nodes, K] or [B, N, K]
        y: Ground truth sequences [B, N, K]
        data: PyTorch Geometric data object with:
            - batch: node-to-graph assignment
            - geom_missing: boolean mask for missing geometry [total_real_nodes]
            - filtered_seq: sequence string for each graph
        B: Batch size
        K: Number of amino acid classes (21 including 'X')
        device: Device for tensors
        use_virtual_node: Whether virtual nodes are being used
        v_pred_is_graph_space: Whether v_pred is already in graph space [total_nodes, K] vs sequence space [B, N, K]
        inpaint_mask: Optional boolean mask [B, N] where True = position was clamped to clean (should be excluded from loss)

    Returns:
        v_pred_masked: Velocity predictions for valid nodes [valid_nodes, K]
        y_masked: Ground truth for valid nodes [valid_nodes, K]
        inpaint_mask_masked: (Only if inpaint_mask provided) Boolean mask for valid nodes [valid_nodes]
                             where True = position was clamped to clean
    """

    batch_sizes = torch.bincount(data.batch)  # [B]
    total_nodes = data.batch.size(0)


    if not v_pred_is_graph_space and v_pred.dim() == 3 and v_pred.shape[0] == B:
        # Convert from sequence space [B, N, K] to graph space [total_nodes, K]
        v_pred = _to_packed_nodes(v_pred, batch_sizes, K)
        #print(f"[apply_geometry_masking] v_pred converted to packed: {v_pred.shape}")
    elif v_pred_is_graph_space and v_pred.shape[0] == total_nodes:
        # Already in correct graph space format
        #print(f"[apply_geometry_masking] v_pred already in graph space: {v_pred.shape}")
        pass
    elif v_pred.dim() > 2:
        # Handle unexpected multi-dimensional tensors more robustly
        print(f"[apply_geometry_masking] WARNING: v_pred has {v_pred.dim()} dimensions: {v_pred.shape}")

        # Try to identify the correct reshaping
        total_elements = v_pred.numel()
        expected_elements = total_nodes * K

        #print(f"[apply_geometry_masking] Total elements: {total_elements}, Expected: {expected_elements}")

        if total_elements == expected_elements:
            # We can safely reshape
            v_pred = v_pred.reshape(total_nodes, K)
            print(f"[apply_geometry_masking] Successfully reshaped to: {v_pred.shape}")
        elif total_elements > expected_elements and total_elements % K == 0:
            # Too many elements - try to take first total_nodes * K elements
            v_pred_flat = v_pred.reshape(-1, K)
            if v_pred_flat.size(0) >= total_nodes:
                v_pred = v_pred_flat[:total_nodes]
                #print(f"[apply_geometry_masking] Truncated to: {v_pred.shape}")
            else:
                #print(f"[apply_geometry_masking] ERROR: Cannot fix shape mismatch")
                raise ValueError(f"Cannot reshape v_pred {v_pred.shape} to match {total_nodes} nodes")
        else:
            #print(f"[apply_geometry_masking] ERROR: Incompatible tensor size")
            raise ValueError(f"v_pred shape {v_pred.shape} is incompatible with expected structure")

    # Final shape check
    if v_pred.shape[0] != total_nodes:
        #print(f"[apply_geometry_masking] ERROR: v_pred shape {v_pred.shape} does not match total_nodes {total_nodes}.")
        raise ValueError(f"v_pred shape {v_pred.shape} does not match total_nodes {total_nodes}")

    # Step 1: Create mask to exclude virtual nodes
    real_node_mask = torch.ones(total_nodes, dtype=torch.bool, device=device)
    if use_virtual_node:
        node_offset = 0
        for b in range(B):
            num_total_nodes = batch_sizes[b].item()
            virtual_node_idx = node_offset + num_total_nodes - 1
            real_node_mask[virtual_node_idx] = False
            node_offset += num_total_nodes

    v_pred_real = v_pred[real_node_mask]  # [total_real_nodes, K]

    # Step 2: Create additional masks for geometry-missing and unknown residues

    # Identify unknown residues ('X' or class 20)
    if y.dim() == 3:  # [B, N, K] format
        # Find positions where class 20 (unknown) has probability 1
        unknown_mask_padded = (y[:, :, 20] > 0.5)  # [B, N]
        # Debug prints commented out for performance
        # print(f"        y[:,:,20] max: {y[:, :, 20].max().item()}, min: {y[:, :, 20].min().item()}")
        # print(f"        Number of positions with class 20 > 0.5: {unknown_mask_padded.sum().item()}")

        # Convert to packed format using real node counts, not total node counts
        real_batch_sizes = torch.zeros_like(batch_sizes)
        for b in range(B):
            num_total_nodes = batch_sizes[b].item()
            num_real_nodes = num_total_nodes - (1 if use_virtual_node else 0)
            real_batch_sizes[b] = num_real_nodes
        unknown_mask_packed = _to_packed_nodes(unknown_mask_padded, real_batch_sizes, None)  # [total_real_nodes]
        # Ensure the mask is on the correct device
        unknown_mask_packed = unknown_mask_packed.to(device)
    else:
        # Already in packed format [total_real_nodes, K]
        unknown_mask_packed = (y[:, 20] > 0.5)  # [total_real_nodes]
        # Debug prints commented out for performance
        # print(f"        y[:,20] max: {y[:, 20].max().item()}, min: {y[:, 20].min().item()}")
        # print(f"        Number of positions with class 20 > 0.5: {unknown_mask_packed.sum().item()}")
        # Ensure the mask is on the correct device
        unknown_mask_packed = unknown_mask_packed.to(device)

    # Check for geometry missing mask in data
    geom_missing_mask = torch.zeros(v_pred_real.size(0), dtype=torch.bool, device=device)
    if hasattr(data, 'geom_missing') and data.geom_missing is not None:
        if data.geom_missing.size(0) == v_pred_real.size(0):
            geom_missing_mask = data.geom_missing.bool().to(device)  # Ensure on correct device
        else:
            print(f"    [apply_geometry_masking] WARNING: geom_missing size mismatch!")
            print(f"      data.geom_missing shape: {data.geom_missing.shape}")
            print(f"      v_pred_real size: {v_pred_real.size(0)}")
            print(f"      Using zero mask instead")
            geom_missing_mask = torch.zeros(v_pred_real.size(0), dtype=torch.bool, device=device)

    # Combine masks - exclude virtual nodes, unknown residues, and geometry-missing nodes
    # We want to keep unknown and geometry-missing nodes in the graph for message passing
    # but exclude them from the loss computation
    final_mask = torch.ones(v_pred_real.size(0), dtype=torch.bool, device=device)

    # Debug prints commented out for performance
    # print(f"      Initial final_mask: {final_mask.sum().item()} out of {final_mask.size(0)}")

    # Exclude unknown residues from loss
    if unknown_mask_packed.size(0) == final_mask.size(0):
        unknown_count = unknown_mask_packed.sum().item()
        final_mask = final_mask & ~unknown_mask_packed
        # print(f"      After excluding {unknown_count} unknown residues: {final_mask.sum().item()} remaining")
    else:
        print(f"      MASK SIZE MISMATCH - unknown_mask_packed: {unknown_mask_packed.size(0)} vs final_mask: {final_mask.size(0)}")

    # Exclude geometry-missing nodes from loss
    if geom_missing_mask.size(0) == final_mask.size(0):
        geom_missing_count = geom_missing_mask.sum().item()
        final_mask = final_mask & ~geom_missing_mask
        # print(f"      After excluding {geom_missing_count} geometry-missing nodes: {final_mask.sum().item()} remaining")
    else:
        # print(f"      MASK SIZE MISMATCH - geom_missing_mask: {geom_missing_mask.size(0)} vs final_mask: {final_mask.size(0)}")
        pass

    # print(f"      FINAL: {final_mask.sum().item()} valid nodes for loss out of {final_mask.size(0)} total real nodes")

    # If no nodes remain, print detailed debugging
    if final_mask.sum().item() == 0:
        print(f"    [apply_geometry_masking] ERROR: ALL NODES MASKED OUT!")
        print(f"      v_pred_real shape: {v_pred_real.shape}")
        print(f"      unknown_mask_packed: {unknown_mask_packed.sum().item()}/{unknown_mask_packed.size(0)} marked as unknown")
        print(f"      geom_missing_mask: {geom_missing_mask.sum().item()}/{geom_missing_mask.size(0)} marked as geometry missing")

        # Check what amino acids are in y_packed to understand unknown masking
        if y.dim() == 3:  # [B, N, K] format
            for b in range(min(B, 3)):  # Check first 3 batches
                batch_y = y[b]  # [N, K]
                aa_indices = batch_y.argmax(dim=-1)  # [N]
                unknown_positions = (aa_indices == 20).sum().item()  # Class 20 is 'X'
                print(f"        Batch {b}: {unknown_positions}/{aa_indices.size(0)} positions are unknown (class 20)")
                print(f"        Batch {b} AA distribution: {aa_indices.bincount(minlength=21).tolist()}")
        else:
            aa_indices = y.argmax(dim=-1) if y.dim() == 2 else None
            if aa_indices is not None:
                unknown_positions = (aa_indices == 20).sum().item()
                print(f"        {unknown_positions}/{aa_indices.size(0)} positions are unknown (class 20)")
                print(f"        AA distribution: {aa_indices.bincount(minlength=21).tolist()}")

    # Apply final masking
    v_pred_masked = v_pred_real[final_mask]  # [valid_nodes, K]

    # Step 3: Extract corresponding ground truth and apply the same masking

    # Convert y from padded format [B, N, K] to packed format [total_real_nodes, K]
    if y.dim() == 3:  # [B, N, K] format
        y_packed_list = []
        for b in range(B):
            num_total_nodes = batch_sizes[b].item()
            num_real_nodes = num_total_nodes - (1 if use_virtual_node else 0)
            if num_real_nodes > 0:
                # Take the first num_real_nodes from this batch's sequence
                y_batch = y[b, :num_real_nodes].to(device)  # [num_real_nodes, K] - ensure on correct device
                y_packed_list.append(y_batch)

        if y_packed_list:
            y_packed = torch.cat(y_packed_list, dim=0)  # [total_real_nodes, K]
        else:
            y_packed = torch.zeros(0, y.size(-1), device=device, dtype=y.dtype)
    else:
        # Already in packed format - ensure it's on the correct device
        y_packed = y.to(device)

    # Apply the same masking to ground truth
    y_masked = y_packed[final_mask]  # [valid_nodes, K]

    # Verify shapes match
    if v_pred_masked.shape != y_masked.shape:
        raise ValueError(f"Shape mismatch after masking: v_pred_masked {v_pred_masked.shape} != y_masked {y_masked.shape}")

    # print(f"[apply_geometry_masking] Final output shapes: v_pred_masked={v_pred_masked.shape}, y_masked={y_masked.shape}")

    # Handle inpainting mask if provided
    if inpaint_mask is not None:
        # Convert inpaint_mask from [B, N] to packed format [total_real_nodes]
        if inpaint_mask.dim() == 2:  # [B, N] format
            inpaint_mask_packed_list = []
            for b in range(B):
                num_total_nodes = batch_sizes[b].item()
                num_real_nodes = num_total_nodes - (1 if use_virtual_node else 0)
                if num_real_nodes > 0:
                    # Take the first num_real_nodes from this batch's inpaint mask
                    inpaint_batch = inpaint_mask[b, :num_real_nodes].to(device)
                    inpaint_mask_packed_list.append(inpaint_batch)

            if inpaint_mask_packed_list:
                inpaint_mask_packed = torch.cat(inpaint_mask_packed_list, dim=0)  # [total_real_nodes]
            else:
                inpaint_mask_packed = torch.zeros(0, dtype=torch.bool, device=device)
        else:
            # Already in packed format
            inpaint_mask_packed = inpaint_mask.to(device)

        # Apply the same final_mask to get inpaint_mask for valid nodes only
        inpaint_mask_masked = inpaint_mask_packed[final_mask]  # [valid_nodes]

        return v_pred_masked, y_masked, inpaint_mask_masked

    return v_pred_masked, y_masked


# ---------------------------------------------------------------------------
# Full-generation validation (--validation_with_full_generation)
# ---------------------------------------------------------------------------

def setup_generation_validation(args, graph_builder_kwargs, device):
    """
    One-time setup for full-generation validation.

    Loads the pre-filtered validation pkl, takes the first val_gen_k entries
    (in pkl order), applies flank filtering to each, builds a noise-free graph,
    and caches everything on the target device so no I/O or graph construction
    happens during the epoch loop.

    Args:
        args: Training arguments (needs val_gen_pkl, val_gen_k).
        graph_builder_kwargs: The same kwargs dict used for training graphs.
            structure_noise_mag_std will be overridden to 0.
        device: torch.device to place tensors on.

    Returns:
        List of dicts, one per protein:
            {
                'name': str,
                'batched_data': PyG Batch (single graph, on device),
                'gt_indices': LongTensor [N_real],   # argmax ground truth
                'n_real': int,                        # real residue count
            }
        Returns None if setup fails (backwards-compatible: caller checks for None).
    """
    import pickle
    from data.graph_builder import GraphBuilder, filter_missing_flanks
    from paths import default_val_gen_pkl
    from training.collate import collate_fn

    pkl_path = getattr(args, 'val_gen_pkl', None) or default_val_gen_pkl()
    k = getattr(args, 'val_gen_k', 100)

    print(f"[GenVal] Setting up full-generation validation: pkl={pkl_path}, k={k}", flush=True)

    # Load entries
    try:
        with open(pkl_path, 'rb') as f:
            all_entries = pickle.load(f)
    except Exception as e:
        print(f"[GenVal] ERROR loading pkl: {e}. Disabling generation validation.", flush=True)
        return None

    # pkl may be a dict {name: entry} or a list of entries — normalise to list
    if isinstance(all_entries, dict):
        all_entries = list(all_entries.values())

    entries = all_entries[:k]
    print(f"[GenVal] Loaded {len(entries)} entries (requested {k}, available {len(all_entries)})", flush=True)

    # Build a noise-free GraphBuilder using the same structural kwargs
    clean_gb_kwargs = {kk: vv for kk, vv in graph_builder_kwargs.items()}
    clean_gb_kwargs['structure_noise_mag_std'] = 0.0
    # verbose and smoke_test are fine to keep as-is
    try:
        graph_builder = GraphBuilder(**clean_gb_kwargs)
    except Exception as e:
        print(f"[GenVal] ERROR building GraphBuilder: {e}. Disabling generation validation.", flush=True)
        return None

    # AA single-letter to index mapping — canonical cath_dataset.py ordering:
    # ALA=0,CYS=1,ASP=2,GLU=3,PHE=4,GLY=5,HIS=6,ILE=7,LYS=8,LEU=9,
    # MET=10,ASN=11,PRO=12,GLN=13,ARG=14,SER=15,THR=16,VAL=17,TRP=18,TYR=19,XXX=20
    _ONE_TO_IDX = {
        'A':0,'C':1,'D':2,'E':3,'F':4,'G':5,'H':6,'I':7,'K':8,'L':9,
        'M':10,'N':11,'P':12,'Q':13,'R':14,'S':15,'T':16,'V':17,'W':18,'Y':19
    }

    dummy_y    = torch.zeros(1, 21)
    dummy_mask = torch.ones(1, dtype=torch.bool)
    dummy_time = torch.tensor(0.0)

    cache = []
    skipped = 0
    for entry in entries:
        name = entry.get('name', 'unknown')
        try:
            filtered_entry, _, _ = filter_missing_flanks(entry, verbose=False)
            graph_data = graph_builder.build_from_dict(
                filtered_entry, time_param=0.0, af_filter_mode=False)
            graph_data = graph_data.to(device)

            use_vn = getattr(graph_data, 'use_virtual_node', False)
            n_total = graph_data.num_nodes
            n_real  = n_total - 1 if use_vn else n_total

            # Ground-truth indices from the flank-filtered sequence
            seq = filtered_entry['seq'][:n_real]
            gt_indices = torch.zeros(n_real, dtype=torch.long, device=device)
            for pos, aa in enumerate(seq):
                gt_indices[pos] = _ONE_TO_IDX.get(aa, 20)

            # Pre-collate as a single-graph batch (done once, reused every epoch)
            batched_data, _, _, _ = collate_fn(
                [(graph_data, dummy_y, dummy_mask, dummy_time)])
            batched_data = batched_data.to(device)

            cache.append({
                'name':         name,
                'batched_data': batched_data,
                'gt_indices':   gt_indices,
                'n_real':       n_real,
                'use_vn':       use_vn,
            })
        except Exception as e:
            print(f"[GenVal] Skipping {name}: {e}", flush=True)
            skipped += 1

    print(f"[GenVal] Cache built: {len(cache)} proteins ready, {skipped} skipped.", flush=True)
    if len(cache) == 0:
        print("[GenVal] No proteins in cache — disabling generation validation.", flush=True)
        return None
    return cache


def run_generation_validation(model, gen_val_cache, args, device):
    """
    Run full Euler-integration sequence generation on the cached validation set
    and return sequence recovery statistics.

    Design:
    - Proteins are processed in mini-batches of val_gen_batch_size.
    - Within each mini-batch the PyG graphs are combined into a single flat
      batch (via Batch.from_data_list) so the model sees one forward pass per
      time step per mini-batch.
    - x (the evolving sequence state) is stored as a flat [total_real_nodes, K]
      tensor throughout — no padding in the integration tensors.
    - prob_t fed to the model is [B, max_N_in_minibatch, K] (unavoidable given
      the model interface) but the padding rows are never read back.
    - Velocity is computed fully vectorized on the flat tensor: no Python loops
      over positions or classes.
    - The static graph is reused every step; no graph rebuilding, no coord noise.

    Args:
        model: The DFM model (already on device).
        gen_val_cache: List of dicts returned by setup_generation_validation.
        args: Training arguments (val_gen_steps, t_min, t_max, flow_temp,
              val_gen_batch_size, dirichlet_concentration).
        device: torch.device.

    Returns:
        Dict with keys:
            'mean_recovery'   : float  (mean per-protein recovery, 0-100)
            'median_recovery' : float
            'std_recovery'    : float
            'n_proteins'      : int
    """
    from torch_geometric.data import Batch
    from training.sample_utils import simplex_proj

    K          = 21
    steps      = getattr(args, 'val_gen_steps', 8)
    T          = getattr(args, 't_max', 8.0)
    t_min      = getattr(args, 't_min', 0.0)
    flow_temp  = getattr(args, 'flow_temp', 1.0)
    conc       = 20.0  # Match actual sampler initialization (not dirichlet_concentration training arg)
    batch_size = getattr(args, 'val_gen_batch_size', 16)

    # steps+1 points so that iterating times[:-1] gives exactly `steps` model calls
    times = torch.linspace(t_min, T, steps + 1, device=device)
    dt    = (T - t_min) / steps

    # Fixed seed so the metric is comparable across epochs and runs
    rng = torch.Generator(device=device)
    rng.manual_seed(12345)

    # Initialise x for every protein with Dirichlet(conc) noise — flat layout
    alpha = conc * torch.ones(K, device=device)
    from torch.distributions import Dirichlet
    dirichlet = Dirichlet(alpha)

    # x_list[i]: [n_real_i, K]
    x_list = [dirichlet.sample((item['n_real'],)) for item in gen_val_cache]

    model.eval()
    n_proteins = len(gen_val_cache)

    with torch.no_grad():
        # ---- Euler integration ------------------------------------------------
        for step_idx, t_val in enumerate(times[:-1]):   # skip last step (no update)
            # Process proteins in mini-batches
            for mb_start in range(0, n_proteins, batch_size):
                mb_end   = min(mb_start + batch_size, n_proteins)
                mb_items = gen_val_cache[mb_start:mb_end]
                mb_x     = x_list[mb_start:mb_end]          # list of [N_i, K]
                B_mb     = len(mb_items)

                # --- Build batched PyG graph for this mini-batch ---------------
                graph_list = [item['batched_data'].to_data_list()[0]
                              for item in mb_items]
                batched_mb = Batch.from_data_list(graph_list).to(device)

                # --- Assemble prob_t [B_mb, max_N, K] -------------------------
                sizes   = [item['n_real'] for item in mb_items]
                max_N   = max(sizes)
                prob_t  = torch.full((B_mb, max_N, K), 1.0 / K, device=device)
                for b, (x_b, n) in enumerate(zip(mb_x, sizes)):
                    prob_t[b, :n] = x_b

                # --- Time tensor [B_mb] ----------------------------------------
                t_tensor = torch.full((B_mb,), t_val.item(), device=device)

                # --- Model forward ---------------------------------------------
                model_out = model(batched_mb, t_tensor, prob_t)
                if isinstance(model_out, dict):
                    logits = model_out['sequence']   # [total_nodes_mb, K]
                elif isinstance(model_out, tuple):
                    logits = model_out[0]            # [total_nodes_mb, K]
                else:
                    logits = model_out

                # --- Softmax → predicted probs (flat, real nodes only) ---------
                # Strip virtual nodes: for each protein, real nodes come first
                pred_flat_parts = []
                node_offset = 0
                for b, item in enumerate(mb_items):
                    n_total_b = (batched_mb.batch == b).sum().item()
                    n_real_b  = item['n_real']
                    pred_flat_parts.append(
                        torch.softmax(logits[node_offset:node_offset + n_real_b] / flow_temp, dim=-1)
                    )
                    node_offset += n_total_b
                pred_flat = torch.cat(pred_flat_parts, dim=0)  # [sum(n_real), K]

                # --- Vectorized velocity (c_factor=False, alternative_test=False) ---
                # Matches dirichlet_flow.py velocity() with use_c_factor=False:
                #   v_k = pred_k * (identity[k,k] - x_k) = pred_k * (1 - x_k)
                # Vectorized over all positions and classes simultaneously.
                x_flat = torch.cat(mb_x, dim=0)   # [sum(n_real), K]
                v_flat = pred_flat * (1.0 - x_flat)
                # Zero-sum correction (simplex tangent constraint), same as velocity()
                v_flat = v_flat - v_flat.sum(dim=-1, keepdim=True) / K

                # --- Euler step + simplex projection ---------------------------
                x_new_flat = simplex_proj(x_flat + dt * v_flat)

                # --- Write back into x_list per protein -----------------------
                offset = 0
                for b, n in enumerate(sizes):
                    x_list[mb_start + b] = x_new_flat[offset:offset + n]
                    offset += n

        # ---- Compute per-protein sequence recovery ---------------------------
        per_protein_recovery = []
        for item, x_final in zip(gen_val_cache, x_list):
            pred_indices = x_final.argmax(dim=-1)          # [N_real]
            gt           = item['gt_indices']
            # Exclude unknown (index 20) from ground-truth when scoring
            valid_mask   = gt != 20
            if valid_mask.sum() == 0:
                continue
            recovery = (pred_indices[valid_mask] == gt[valid_mask]).float().mean().item() * 100.0
            per_protein_recovery.append(recovery)

    if len(per_protein_recovery) == 0:
        return {'mean_recovery': 0.0, 'median_recovery': 0.0,
                'std_recovery': 0.0, 'n_proteins': 0}

    rec = torch.tensor(per_protein_recovery)
    return {
        'mean_recovery':   rec.mean().item(),
        'median_recovery': rec.median().item(),
        'std_recovery':    rec.std().item() if len(rec) > 1 else 0.0,
        'n_proteins':      len(per_protein_recovery),
    }

    return v_pred_masked, y_masked
