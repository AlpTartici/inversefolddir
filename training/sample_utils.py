# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Global cache for DSSP-AA initialization matrix
_DSSP_AA_INIT_CACHE = None


def load_dssp_aa_initialization(file_path='../datasets/dssp_aa_prob_matrix.npz'):
    """
    Load DSSP-conditioned amino acid initialization parameters.

    Returns:
        prob_matrix: np.array [num_dssp, 20] - relative alpha weights for each DSSP category
        dssp_to_row: dict mapping DSSP character to row index in prob_matrix
        aa_col_to_idx: list mapping prob_matrix column index to our AA_TO_IDX index
    """
    global _DSSP_AA_INIT_CACHE

    if _DSSP_AA_INIT_CACHE is not None:
        return _DSSP_AA_INIT_CACHE

    import numpy as np
    import os

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"DSSP-AA initialization file not found: {file_path}")

    # Load the npz file
    data = np.load(file_path)
    prob_matrix = data['prob_matrix']  # [num_dssp, 20]
    dssp_categories = data['dssp_categories']  # [num_dssp]
    amino_acids = data['amino_acids']  # [20]

    # Create mapping from DSSP char to row index in prob_matrix
    dssp_to_row = {str(dssp): i for i, dssp in enumerate(dssp_categories)}

    # Create mapping from prob_matrix column to our AA index
    # amino_acids are single-letter codes, need to convert to our three-letter indices
    aa_col_to_idx = []
    for aa_single in amino_acids:
        aa_triple = SINGLE_TO_TRIPLE.get(aa_single, 'XXX')
        aa_idx = AA_TO_IDX.get(aa_triple, 20)  # 20 is unknown
        aa_col_to_idx.append(aa_idx)

    _DSSP_AA_INIT_CACHE = (prob_matrix, dssp_to_row, aa_col_to_idx)
    return _DSSP_AA_INIT_CACHE


def create_dssp_conditioned_alphas(dssp_targets, N, K, device, dirichlet_concentration, verbose=False, first_pos_only=False):
    """
    Create position-specific Dirichlet alpha parameters based on DSSP targets.

    Args:
        dssp_targets: Tensor of DSSP indices [N] (using DSSP_TO_IDX mapping)
        N: Number of positions
        K: Number of amino acid classes (21)
        device: torch device
        dirichlet_concentration: Concentration parameter to scale alphas
        verbose: Print debug info
        first_pos_only: If True, only apply DSSP conditioning to first position, use uniform for rest

    Returns:
        alphas: Tensor [N, K] of position-specific alpha parameters
    """
    import torch
    import numpy as np
    from data.dssp_constants import IDX_TO_DSSP, DSSP_UNKNOWN_IDX

    # Load DSSP-AA initialization matrix
    prob_matrix, dssp_to_row, aa_col_to_idx = load_dssp_aa_initialization()

    # Initialize alphas tensor with uniform values
    alphas = torch.ones((N, K), device=device, dtype=torch.float32) * dirichlet_concentration

    # Convert dssp_targets to numpy for easier indexing
    if isinstance(dssp_targets, torch.Tensor):
        dssp_targets_np = dssp_targets.cpu().numpy()
    else:
        dssp_targets_np = np.array(dssp_targets)

    # Determine which positions to process
    positions_to_process = [0] if first_pos_only else range(N)

    for pos in positions_to_process:
        target_dssp_idx = int(dssp_targets_np[pos])
        target_dssp_char = IDX_TO_DSSP[target_dssp_idx]

        # Convert 'X' (unknown) to '-' (coil)
        if target_dssp_char == 'X':
            target_dssp_char = '-'

        # Get row from prob_matrix
        if target_dssp_char not in dssp_to_row:
            raise ValueError(f"DSSP category '{target_dssp_char}' not found in initialization matrix! "
                           f"Available categories: {list(dssp_to_row.keys())}")

        row_idx = dssp_to_row[target_dssp_char]
        row_alphas = prob_matrix[row_idx]  # [20] numpy array

        # Reset this position to zeros before setting DSSP-specific values
        alphas[pos, :] = 0.0

        # Map to our AA indices (K=21)
        for col_idx, aa_idx in enumerate(aa_col_to_idx):
            if aa_idx < K:
                alphas[pos, aa_idx] = float(row_alphas[col_idx])

        # Set unknown AA (index 20) to very small value (Dirichlet requires alpha > 0)
        alphas[pos, 20] = 1e-8

        # Scale by dirichlet_concentration (no renormalization)
        alphas[pos, :] = alphas[pos, :] * dirichlet_concentration

    if verbose:
        mode_str = "first position only" if first_pos_only else f"{N} positions"
        print(f"DSSP-conditioned initialization ({mode_str}):")
        print(f"  Created position-specific alphas for {mode_str}")
        print(f"  Alpha value range: [{alphas.min().item():.4f}, {alphas.max().item():.4f}]")
        print(f"  Mean alpha per position: {alphas.mean(dim=1).mean().item():.4f}")

    return alphas


def get_cached_dssp_aa_matrix_tensor(device):
    """
    Load DSSP→AA probability matrix as a cached tensor for vectorized operations.

    Returns:
        prob_matrix_tensor: [10, 21] tensor on specified device
        Maps DSSP index → AA probability distribution (20 standard + 1 unknown)
    """
    import torch

    global _DSSP_AA_TENSOR_CACHE

    if '_DSSP_AA_TENSOR_CACHE' not in globals():
        _DSSP_AA_TENSOR_CACHE = {}

    cache_key = str(device)
    if cache_key in _DSSP_AA_TENSOR_CACHE:
        return _DSSP_AA_TENSOR_CACHE[cache_key]

    # Load numpy matrix
    prob_matrix, dssp_to_row, aa_col_to_idx = load_dssp_aa_initialization()

    # Create [10, 21] tensor (10 DSSP classes, 21 AA classes)
    prob_tensor = torch.zeros(10, 21, dtype=torch.float32, device=device)

    # Map the 20 standard AAs from numpy matrix
    for col_idx, aa_idx in enumerate(aa_col_to_idx):
        if aa_idx < 20:  # Standard AA
            prob_tensor[:, aa_idx] = torch.tensor(prob_matrix[:, col_idx], dtype=torch.float32, device=device)

    # Set unknown AA (index 20) to small value
    prob_tensor[:, 20] = 1e-8

    # Normalize rows to sum to 1.0
    prob_tensor = prob_tensor / prob_tensor.sum(dim=1, keepdim=True)

    _DSSP_AA_TENSOR_CACHE[cache_key] = prob_tensor
    return prob_tensor


def blend_aa_distributions_with_dssp_guidance(
    model_aa_probs,      # torch.Tensor [N, 21] or [B, N, 21]
    model_dssp_probs,    # torch.Tensor [N, 10] or [B, N, 10]
    target_dssp_indices, # torch.Tensor [N] or [B, N]
    blending_method='geometric',
    verbose=False,
    current_step=None,       # Current sampling step (0, 1, 2, ..., steps-1)
    total_steps=None,        # Total number of sampling steps
    lambda_floor=0.2,        # Minimum lambda value (confidence floor)
    annealing_schedule='quadratic'  # 'linear', 'quadratic', or 'cubic'
):
    """
    Vectorized DSSP-guided blending of amino acid probability distributions with temporal annealing.

    Supports both single-protein [N, K] and batched [B, N, K] inputs.

    Algorithm:
    1. For each position:
       a. Compare argmax(model_dssp_probs) to target_dssp
       b. If match: use model_aa_probs unchanged
       c. If mismatch:
          - lambda = model_dssp_probs[target_dssp_idx] (confidence in target)
          - Apply confidence floor: lambda = max(lambda, lambda_floor)
          - Apply temporal annealing to increase lambda toward 1.0 as sampling progresses
          - Get p_aa_target from DSSP→AA matrix for target DSSP
          - Blend: geometric or arithmetic mean
          - Renormalize result

    Temporal Annealing:
        - Early steps: Use computed lambda (with floor), strong DSSP guidance when model is wrong
        - Late steps: Force lambda → 1.0, letting model make decisive predictions
        - This allows DSSP to guide initial exploration while enabling final commitment

    Args:
        model_aa_probs: AA probability distributions from model
        model_dssp_probs: DSSP probability distributions from model
        target_dssp_indices: Target DSSP class indices (use DSSP_TO_IDX mapping)
        blending_method: 'geometric' or 'arithmetic'
        verbose: Print blending statistics
        current_step: Current step in sampling (0-indexed). If None, no annealing applied.
        total_steps: Total number of steps. If None, no annealing applied.
        lambda_floor: Minimum lambda value to maintain model influence (default 0.2 = 20% model weight minimum)
        annealing_schedule: Shape of annealing curve ('linear', 'quadratic', 'cubic')

    Returns:
        Blended AA probability distributions (same shape as model_aa_probs)
    """
    import torch
    from data.dssp_constants import DSSP_UNKNOWN_IDX

    device = model_aa_probs.device
    is_batched = model_aa_probs.dim() == 3

    # Handle single-protein case by adding batch dimension
    if not is_batched:
        model_aa_probs = model_aa_probs.unsqueeze(0)      # [1, N, 21]
        model_dssp_probs = model_dssp_probs.unsqueeze(0)  # [1, N, 10]
        target_dssp_indices = target_dssp_indices.unsqueeze(0)  # [1, N]

    B, N, K = model_aa_probs.shape

    # Get DSSP→AA probability matrix [10, 21]
    dssp_aa_matrix = get_cached_dssp_aa_matrix_tensor(device)

    # Handle unknown DSSP targets: map DSSP_UNKNOWN_IDX (9) to '-' (0)
    target_dssp_indices = target_dssp_indices.clone()
    target_dssp_indices[target_dssp_indices == DSSP_UNKNOWN_IDX] = 0  # Coil

    # Get predicted DSSP classes [B, N]
    pred_dssp_classes = model_dssp_probs.argmax(dim=-1)

    # Identify positions where prediction != target [B, N]
    mismatch_mask = (pred_dssp_classes != target_dssp_indices)

    # Get target AA distributions from DSSP matrix
    # target_dssp_indices: [B, N] -> use as indices into dssp_aa_matrix [10, 21]
    target_aa_probs = dssp_aa_matrix[target_dssp_indices]  # [B, N, 21]

    # Get lambda (confidence in target DSSP) [B, N]
    # Use gather to extract model_dssp_probs at target_dssp_indices
    target_dssp_idx_expanded = target_dssp_indices.unsqueeze(-1)  # [B, N, 1]
    lambda_vals = model_dssp_probs.gather(dim=2, index=target_dssp_idx_expanded).squeeze(-1)  # [B, N]

    # Apply confidence floor to ensure model always has minimum influence
    lambda_vals = torch.clamp(lambda_vals, min=lambda_floor)

    # Apply temporal annealing if step information is provided
    if current_step is not None and total_steps is not None and total_steps > 1:
        # Compute progress through sampling: 0.0 at start → 1.0 at end
        progress = float(current_step) / float(total_steps - 1)

        # Apply annealing schedule to determine how much to increase lambda
        if annealing_schedule == 'linear':
            annealing_factor = progress
        elif annealing_schedule == 'quadratic':
            annealing_factor = progress ** 2
        elif annealing_schedule == 'cubic':
            annealing_factor = progress ** 3
        else:
            raise ValueError(f"Unknown annealing_schedule: {annealing_schedule}")

        # Interpolate lambda toward 1.0 based on progress
        # Early: annealing_factor ≈ 0, use original lambda_vals
        # Late: annealing_factor ≈ 1, lambda → 1.0
        lambda_vals = lambda_vals * (1.0 - annealing_factor) + annealing_factor

        if verbose:
            print(f"  [DSSP Annealing] Step {current_step}/{total_steps-1}, Progress: {progress:.3f}, Factor: {annealing_factor:.3f}")

    # Expand lambda for broadcasting [B, N, 1]
    lambda_expanded = lambda_vals.unsqueeze(-1)

    # Perform blending based on method
    if blending_method == 'geometric':
        # Geometric: (p_model^λ) × (p_target^(1-λ))
        # Use log-space to avoid numerical issues
        log_model = torch.log(model_aa_probs + 1e-10)
        log_target = torch.log(target_aa_probs + 1e-10)

        log_blended = lambda_expanded * log_model + (1 - lambda_expanded) * log_target
        blended_probs = torch.exp(log_blended)

    elif blending_method == 'arithmetic':
        # Arithmetic: λ×p_model + (1-λ)×p_target
        blended_probs = lambda_expanded * model_aa_probs + (1 - lambda_expanded) * target_aa_probs

    else:
        raise ValueError(f"Unknown blending method: {blending_method}")

    # Renormalize
    blended_probs = blended_probs / (blended_probs.sum(dim=-1, keepdim=True) + 1e-10)

    # Apply blending only where there's a mismatch
    # Keep original probabilities where prediction == target
    mismatch_mask_expanded = mismatch_mask.unsqueeze(-1)  # [B, N, 1]
    output_probs = torch.where(mismatch_mask_expanded, blended_probs, model_aa_probs)

    # Verbose statistics
    if verbose:
        total_positions = B * N
        num_mismatches = mismatch_mask.sum().item()
        avg_lambda = lambda_vals[mismatch_mask].mean().item() if num_mismatches > 0 else 0.0
        min_lambda = lambda_vals[mismatch_mask].min().item() if num_mismatches > 0 else 0.0
        max_lambda = lambda_vals[mismatch_mask].max().item() if num_mismatches > 0 else 0.0

        print(f"  [DSSP Guidance] Mismatches: {num_mismatches}/{total_positions} ({100*num_mismatches/total_positions:.1f}%)")
        print(f"  [DSSP Guidance] λ at mismatches - Avg: {avg_lambda:.3f}, Min: {min_lambda:.3f}, Max: {max_lambda:.3f}")
        print(f"  [DSSP Guidance] Blending: {blending_method}, Floor: {lambda_floor:.2f}")

    # Remove batch dimension if input was single-protein
    if not is_batched:
        output_probs = output_probs.squeeze(0)  # [N, 21]

    return output_probs


def create_structural_ensemble(input_spec, ensemble_size=1, structure_noise_mag_std=1.0,
                              uncertainty_struct_noise_scaling=False, device='cpu', args=None, dataset_params=None):
    """
    Create an ensemble of structures from a single input by adding uncertainty-scaled noise.
    Uses GraphBuilder's built-in noise functionality.
    """
    import numpy as np
    import torch
    from torch_geometric.data import Batch

    from data.graph_builder import GraphBuilder

    # Input spec is already a processed protein dictionary when called from ensemble mode
    entry = input_spec
    temp_files = []

    try:
        # Parameter precedence: CLI args first, then model checkpoint as fallback
        graph_builder_kwargs = {}

        # Helper function to resolve parameter with proper precedence
        def resolve_param(param_name, cli_value, checkpoint_value, default_value=None):
            if cli_value is not None:
                return cli_value, "CLI args"
            elif checkpoint_value is not None:
                return checkpoint_value, "checkpoint"
            else:
                return default_value, "default"

        # Resolve each parameter with CLI precedence
        if args:
            k_val, k_source = resolve_param('k', getattr(args, 'k_neighbors', None),
                                           dataset_params.get('k_neighbors') if dataset_params else None)
            k_farthest_val, k_farthest_source = resolve_param('k_farthest', getattr(args, 'k_farthest', None),
                                                             dataset_params.get('k_farthest') if dataset_params else None)
            k_random_val, k_random_source = resolve_param('k_random', getattr(args, 'k_random', None),
                                                         dataset_params.get('k_random') if dataset_params else None)
            max_edge_dist_val, max_edge_dist_source = resolve_param('max_edge_dist', getattr(args, 'max_edge_dist', None),
                                                                   dataset_params.get('max_edge_dist') if dataset_params else None)
            rbf_3d_min_val, rbf_3d_min_source = resolve_param('rbf_3d_min', getattr(args, 'rbf_3d_min', None),
                                                             dataset_params.get('rbf_3d_min') if dataset_params else None, 2.0)
            rbf_3d_max_val, rbf_3d_max_source = resolve_param('rbf_3d_max', getattr(args, 'rbf_3d_max', None),
                                                             dataset_params.get('rbf_3d_max') if dataset_params else None, 350.0)
            rbf_3d_spacing_val, rbf_3d_spacing_source = resolve_param('rbf_3d_spacing', getattr(args, 'rbf_3d_spacing', None),
                                                                     dataset_params.get('rbf_3d_spacing') if dataset_params else None, 'exponential')

            graph_builder_kwargs = {
                'k': k_val,
                'k_farthest': k_farthest_val,
                'k_random': k_random_val,
                'max_edge_dist': max_edge_dist_val,
                'num_rbf_3d': dataset_params.get('num_rbf_3d', 16) if dataset_params else 16,
                'num_rbf_seq': dataset_params.get('num_rbf_seq', 16) if dataset_params else 16,
                'use_virtual_node': dataset_params.get('use_virtual_node', True) if dataset_params else True,
                'no_source_indicator': dataset_params.get('no_source_indicator', False) if dataset_params else False,
                'rbf_3d_min': rbf_3d_min_val,
                'rbf_3d_max': rbf_3d_max_val,
                'rbf_3d_spacing': rbf_3d_spacing_val,
            }

            if getattr(args, 'verbose', False):
                print(f"[DEBUG] GraphBuilder parameter sources:")
                print(f"  k: {k_val} (source: {k_source})")
                print(f"  k_farthest: {k_farthest_val} (source: {k_farthest_source})")
                print(f"  k_random: {k_random_val} (source: {k_random_source})")
                print(f"  max_edge_dist: {max_edge_dist_val} (source: {max_edge_dist_source})")
                print(f"  rbf_3d_min: {rbf_3d_min_val} (source: {rbf_3d_min_source})")
                print(f"  rbf_3d_max: {rbf_3d_max_val} (source: {rbf_3d_max_source})")
        else:
            # Fallback to checkpoint only if no args
            if dataset_params:
                graph_builder_kwargs = {
                    'k': dataset_params.get('k_neighbors'),
                    'k_farthest': dataset_params.get('k_farthest'),
                    'k_random': dataset_params.get('k_random'),
                    'max_edge_dist': dataset_params.get('max_edge_dist'),
                    'num_rbf_3d': dataset_params.get('num_rbf_3d', 16),
                    'num_rbf_seq': dataset_params.get('num_rbf_seq', 16),
                    'use_virtual_node': dataset_params.get('use_virtual_node', True),
                    'no_source_indicator': dataset_params.get('no_source_indicator', False),
                    'rbf_3d_min': dataset_params.get('rbf_3d_min', 2.0),
                    'rbf_3d_max': dataset_params.get('rbf_3d_max', 350.0),
                    'rbf_3d_spacing': dataset_params.get('rbf_3d_spacing', 'exponential'),
                }

        # Add noise settings to graph builder (these are ensemble-specific)
        graph_builder_kwargs['structure_noise_mag_std'] = structure_noise_mag_std
        graph_builder_kwargs['uncertainty_struct_noise_scaling'] = uncertainty_struct_noise_scaling
        graph_builder_kwargs['time_based_struct_noise'] = 'fixed'  # No time-based scaling for ensemble

        print(f"Creating ensemble of {ensemble_size} replicas")
        if structure_noise_mag_std > 0:
            print(f"  Noise std: {structure_noise_mag_std} Å")
            if uncertainty_struct_noise_scaling:
                print(f"  Uncertainty-scaled noise enabled")

        if args and getattr(args, 'verbose', False):
            print(f"[DEBUG] Full GraphBuilder kwargs for ensemble:")
            for key, value in graph_builder_kwargs.items():
                print(f"  {key}: {value}")

        # Create graph builder with noise settings
        graph_builder = GraphBuilder(**graph_builder_kwargs)

        # Generate ensemble members
        ensemble_graphs = []
        previous_coords = None

        for replica_idx in range(ensemble_size):
            # Create a fresh GraphBuilder for each replica to prevent state pollution
            replica_graph_builder = GraphBuilder(**graph_builder_kwargs)

            # Build graph (GraphBuilder will apply noise internally)
            graph = replica_graph_builder.build_from_dict(entry.copy(), time_param=0.0, af_filter_mode=False)
            graph.ensemble_idx = replica_idx

            if args and getattr(args, 'verbose', False) and replica_idx == 0:
                print(f"[DEBUG] Replica {replica_idx} graph structure:")
                print(f"  num_nodes: {graph.num_nodes}")
                print(f"  num_edges: {graph.num_edges}")
                print(f"  pos.shape: {graph.pos.shape}")
                if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
                    print(f"  edge_attr.shape: {graph.edge_attr.shape}")
                else:
                    print(f"  edge_attr: None")

            # Verify diversity - check that noise was actually different
            if structure_noise_mag_std > 0 and replica_idx > 0:
                # Compare positions with first replica
                if replica_idx == 1:
                    # Store first replica's coordinates for comparison
                    previous_coords = ensemble_graphs[0].pos.clone()

                current_coords = graph.pos
                coord_diff = torch.norm(current_coords - previous_coords).item()

                if coord_diff < 1e-6:
                    raise RuntimeError(
                        f"CRITICAL ERROR: Replica {replica_idx} has identical coordinates to replica 0!\n"
                        f"Coordinate difference: {coord_diff:.2e} Å\n"
                        f"This indicates a random seed issue - all replicas are getting the same noise.\n"
                        f"Please check PyTorch random seed settings or file a bug report."
                    )

                if args and getattr(args, 'verbose', False):
                    print(f"  Replica {replica_idx} coordinate RMSD from replica 0: {coord_diff:.3f} Å")

            ensemble_graphs.append(graph)

        # Batch all graphs together
        batched_ensemble = Batch.from_data_list(ensemble_graphs)
        batched_ensemble.ensemble_size = ensemble_size

        return batched_ensemble.to(device)

    finally:
        # Cleanup temp files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass


def sample_with_ensemble_consensus(model, batched_ensemble, T=8.0, t_min=0.0, steps=50, K=21,
                                  consensus_strength=0.2, device='cpu', use_virtual_node=True, args=None):
    """
    Sample from an ensemble with state aggregation at each timestep.

    Key design: We aggregate STATES (current probabilities) not PREDICTIONS.
    This maintains diversity while reducing overconfidence.

    Args:
        model: Trained model
        batched_ensemble: Batched graph from create_structural_ensemble
        T: Maximum time
        t_min: Minimum time
        steps: Number of sampling steps
        K: Number of amino acid classes
        consensus_strength: How much to blend states (0=independent, 1=full consensus)
        device: PyTorch device
        args: Additional arguments

    Returns:
        Final consensus sequence as list of amino acid indices
    """
    import torch
    from torch.distributions import Dirichlet
    from torch_geometric.data import Batch
    from tqdm import tqdm

    model.eval()
    model = model.to(device)
    batched_ensemble = batched_ensemble.to(device)

    ensemble_size = batched_ensemble.ensemble_size

    # Get sequence length (should be same for all replicas)
    seq_lengths = []
    for i in range(ensemble_size):
        mask = (batched_ensemble.batch == i)
        num_nodes = mask.sum().item()
        # Handle virtual nodes - use the parameter passed to the function
        seq_len = num_nodes - 1 if use_virtual_node else num_nodes
        seq_lengths.append(seq_len)

    # Verify all replicas have same length
    if len(set(seq_lengths)) != 1:
        raise ValueError(f"Ensemble members have different lengths: {seq_lengths}")

    N = seq_lengths[0]

    print(f"\nSampling with {ensemble_size}-member ensemble, consensus_strength={consensus_strength}")

    # Initialize with different Dirichlet noise for each replica
    dirichlet_concentration = getattr(args, 'dirichlet_concentration', 20.0) if args else 20.0
    dirichlet_dist = Dirichlet(dirichlet_concentration * torch.ones(K, device=device))

    # Sample different initial states for each replica
    x_batch = dirichlet_dist.sample((ensemble_size, N))  # [ensemble_size, N, K]

    if args and getattr(args, 'verbose', False):
        print(f"[DEBUG] Initial Dirichlet states for ensemble:")
        print(f"  x_batch.shape: {x_batch.shape}")
        print(f"  dirichlet_concentration: {dirichlet_concentration}")
        for i in range(min(2, ensemble_size)):
            replica_entropy = -(x_batch[i] * torch.log(x_batch[i] + 1e-8)).sum(-1).mean()
            print(f"  Replica {i} initial entropy: {replica_entropy:.4f}")
            print(f"  Replica {i} first 3 pos max probs: {[x_batch[i, j].max().item() for j in range(3)]}")

        # Check if replicas are actually different
        if ensemble_size > 1:
            state_diff = torch.norm(x_batch[0] - x_batch[1]).item()
            print(f"  L2 difference between replica 0 and 1 initial states: {state_diff:.6f}")
            if state_diff < 1e-6:
                print(f"  WARNING: Initial states are nearly identical!")
            else:
                print(f"  ✓ Initial states are properly diversified")

    # Time integration
    times = torch.linspace(t_min, T, steps, device=device)
    dt = (T - t_min) / (steps - 1) if steps > 1 else 0

    with torch.no_grad():
        for step_idx, t_val in enumerate(tqdm(times[:-1], desc=f"Ensemble sampling")):

            # State consensus (aggregate current states before prediction)
            if consensus_strength > 0:
                # Get ensemble method from args, default to arithmetic
                ensemble_method = getattr(args, 'ensemble_method', 'arithmetic') if args else 'arithmetic'

                # Apply consensus blending using the specified method
                x_batch = compute_ensemble_consensus(x_batch, consensus_strength, ensemble_method, ensemble_size)

            # Get predictions from model using batched processing
            t_tensor = torch.full((ensemble_size,), t_val, device=device)

            if args and getattr(args, 'verbose', False) and step_idx == 0:
                print(f"[DEBUG] Computing predictions with BATCHED processing:")
                print(f"  batched_ensemble.num_graphs: {batched_ensemble.num_graphs}")
                print(f"  batched_ensemble.num_nodes: {batched_ensemble.num_nodes}")
                print(f"  x_batch.shape: {x_batch.shape}")
                print(f"  t_tensor.shape: {t_tensor.shape}")

            # Single batched forward pass for all replicas at once!
            model_output = model(batched_ensemble, t_tensor, x_batch)

            # Handle dict (multi-aux), tuple (DSSP-only), or tensor (single head)
            if isinstance(model_output, dict):
                logits = model_output['sequence']
            elif isinstance(model_output, tuple):
                logits = model_output[0]  # [total_nodes, K]
            else:
                logits = model_output  # [total_nodes, K]

            # Apply temperature
            flow_temp = getattr(args, 'flow_temp', 1.0) if args else 1.0
            pred_target = torch.softmax(logits / flow_temp, dim=-1)  # [total_nodes, K]

            if args and getattr(args, 'verbose', False) and step_idx == 0:
                print(f"[DEBUG] Batched predictions:")
                print(f"  logits.shape: {logits.shape}")
                print(f"  pred_target.shape: {pred_target.shape}")
                print(f"  logits stats: min={logits.min():.3f}, max={logits.max():.3f}, mean={logits.mean():.3f}")
                print(f"  pred_target stats: min={pred_target.min():.6f}, max={pred_target.max():.6f}")

            # Split predictions back into per-replica format [ensemble_size, N, K]
            pred_ensemble = torch.zeros((ensemble_size, N, K), dtype=torch.float32, device=device)

            # Extract predictions for each replica from batched output
            node_offset = 0
            for replica_idx in range(ensemble_size):
                # Each replica has N+1 nodes (N real + 1 virtual) if use_virtual_node
                nodes_per_replica = N + 1 if use_virtual_node else N

                # Extract this replica's predictions
                replica_predictions = pred_target[node_offset:node_offset + nodes_per_replica]  # [N+1, K] or [N, K]

                # Handle virtual nodes
                if use_virtual_node:
                    pred_real = replica_predictions[:N]  # Take only real nodes [N, K]
                else:
                    pred_real = replica_predictions  # [N, K]

                pred_ensemble[replica_idx] = pred_real
                node_offset += nodes_per_replica

                if args and getattr(args, 'verbose', False) and step_idx == 0 and replica_idx == 0:
                    print(f"[DEBUG] Replica {replica_idx} extracted predictions:")
                    print(f"  pred_real.shape: {pred_real.shape}")
                    print(f"  pred_real first 5 positions max probs: {[pred_real[i].max().item() for i in range(min(5, pred_real.shape[0]))]}")

            # Apply consensus if needed
            if consensus_strength > 0:
                # Apply consensus averaging
                consensus_weights = torch.softmax(consensus_strength * torch.ones(ensemble_size, device=device), dim=0)
                pred_probs = torch.sum(consensus_weights.view(-1, 1, 1) * pred_ensemble, dim=0, keepdim=True)
                pred_probs = pred_probs.expand(ensemble_size, -1, -1)  # Broadcast to all replicas
            else:
                # No consensus - use individual predictions
                pred_probs = pred_ensemble

            # Compute velocities using batched processing
            use_smoothed_targets = getattr(args, 'use_smoothed_targets', False) if args else False
            use_c_factor = getattr(args, 'use_c_factor', False) if args else False
            if step_idx == 0:  # Print only once at the start
                print(f"[SAMPLING] use_c_factor: {use_c_factor}", flush=True)

            if args and getattr(args, 'verbose', False) and step_idx == 0:
                print(f"[DEBUG] Computing velocities with batched processing:")
                print(f"  x_batch.shape: {x_batch.shape}")
                print(f"  pred_probs.shape: {pred_probs.shape}")
                print(f"  t_tensor.shape: {t_tensor.shape}")

            # Compute velocities for all replicas at once (much faster!)
            velocities = model.cond_flow.velocity(
                x_batch, pred_probs, t_tensor,
                use_virtual_node=use_virtual_node,
                use_smoothed_targets=use_smoothed_targets,
                use_c_factor=use_c_factor
            )

            if args and getattr(args, 'verbose', False) and step_idx == 0:
                print(f"[DEBUG] Batched velocities:")
                print(f"  velocities.shape: {velocities.shape}")
                print(f"  velocities stats: min={velocities.min():.6f}, max={velocities.max():.6f}, mean={velocities.mean():.6f}")

            # Update states with Euler step
            x_new = x_batch + dt * velocities
            x_batch = simplex_proj(x_new)

            # Debug: Check if replicas remain different after update
            if args and getattr(args, 'verbose', False) and ensemble_size > 1:
                state_diff = torch.norm(x_batch[0] - x_batch[1]).item()
                print(f"[DEBUG] After step {step_idx}: L2 diff between replicas = {state_diff:.6f}")
                if state_diff < 1e-6:
                    print(f"[DEBUG] WARNING: Replicas have become identical after update!")
                    print(f"[DEBUG] Replica 0 first 5 max probs: {[x_batch[0, i].max().item() for i in range(5)]}")
                    print(f"[DEBUG] Replica 1 first 5 max probs: {[x_batch[1, i].max().item() for i in range(5)]}")
                elif state_diff < 0.1:
                    print(f"[DEBUG] WARNING: Replicas are converging!")
                else:
                    print(f"[DEBUG] Replicas remain different")

    # Final sequence generation based on consensus strength
    if args and getattr(args, 'verbose', False):
        print(f"[DEBUG] Final consensus decision: consensus_strength={consensus_strength}")

    if consensus_strength > 0:
        # Consensus: average all replica probabilities using specified method
        ensemble_method = getattr(args, 'ensemble_method', 'arithmetic') if args else 'arithmetic'

        if args and getattr(args, 'verbose', False):
            print(f"[DEBUG] Applying final consensus averaging using {ensemble_method} method")

        # Apply the chosen consensus method directly for final sequence generation
        if ensemble_method == 'arithmetic':
            # Standard arithmetic mean in probability space
            final_consensus = x_batch.mean(dim=0)  # [N, K]
        elif ensemble_method == 'geometric':
            # Geometric mean in log space
            eps = 1e-8
            log_states = torch.log(x_batch + eps)  # [ensemble_size, N, K]
            log_mean = log_states.mean(dim=0)  # [N, K]
            final_consensus = torch.exp(log_mean)  # [N, K]
            # Normalize to ensure it's on the simplex
            final_consensus = final_consensus / final_consensus.sum(dim=-1, keepdim=True)
        else:
            # Fallback to arithmetic if unknown method
            final_consensus = x_batch.mean(dim=0)  # [N, K]

        final_sequence = final_consensus.argmax(-1).tolist()
    else:
        # No consensus: use first replica only (but let's debug all replicas)
        if args and getattr(args, 'verbose', False):
            print(f"[DEBUG] Using first replica only (no consensus)")
            print(f"[DEBUG] Debugging all replica sequences (first 20 positions):")
            for replica_idx in range(min(ensemble_size, 3)):  # Show up to 3 replicas
                replica_seq = x_batch[replica_idx].argmax(-1).tolist()
                print(f"  Replica {replica_idx}: {replica_seq[:20]}")

        final_sequence = x_batch[0].argmax(-1).tolist()

    return final_sequence


def adjust_batch_size_for_ensemble(batch_size, ensemble_size):
    """
    Adjust batch size to avoid splitting proteins across batches in ensemble mode.

    Args:
        batch_size: Desired batch size
        ensemble_size: Number of replicas per protein

    Returns:
        Adjusted batch size that's divisible by ensemble_size
    """
    if batch_size % ensemble_size == 0:
        return batch_size

    # Find the largest batch size <= original that's divisible by ensemble_size
    adjusted_batch_size = (batch_size // ensemble_size) * ensemble_size

    if adjusted_batch_size == 0:
        adjusted_batch_size = ensemble_size

    print(f"Adjusted batch size from {batch_size} to {adjusted_batch_size} to avoid splitting proteins across batches (ensemble_size={ensemble_size})")

    return adjusted_batch_size


def sample_multiple_proteins_with_ensemble(model, dataset, indices=None, steps=50, T=8.0, t_min=0.0, K=21,
                                         ensemble_size=5, consensus_strength=0.2,
                                         structure_noise_mag_std=0.0, uncertainty_struct_noise_scaling=False,
                                         batch_size=32, args=None):
    """
    Sample multiple proteins with ensemble consensus using mega-batching strategy.


    Args:
        model: Trained DFM model
        dataset: CathDataset instance
        indices: List of structure indices to sample (None = all)
        steps: Number of sampling steps
        T: Maximum time
        t_min: Minimum time
        K: Number of amino acid classes
        ensemble_size: Number of replicas per protein
        consensus_strength: How much to blend states (0=independent, 1=full consensus)
        structure_noise_mag_std: Standard deviation for structural noise
        uncertainty_struct_noise_scaling: Whether to scale noise by uncertainty
        batch_size: Desired batch size (will be adjusted to be divisible by ensemble_size)
        args: Arguments object

    Returns:
        List of result dictionaries with evaluation metrics
    """
    from torch.distributions import Dirichlet
    from torch_geometric.data import Batch
    from tqdm import tqdm

    if indices is None:
        indices = list(range(len(dataset)))

    # Adjust batch size to avoid splitting proteins across batches
    adjusted_batch_size = adjust_batch_size_for_ensemble(batch_size, ensemble_size)
    proteins_per_batch = adjusted_batch_size // ensemble_size

    print(f"\nEnsemble Multi-Protein Sampling Configuration:")
    print(f"  Total proteins: {len(indices)}")
    print(f"  Ensemble size: {ensemble_size}")
    print(f"  Consensus strength: {consensus_strength}")
    print(f"  Adjusted batch size: {adjusted_batch_size} (from {batch_size})")
    print(f"  Proteins per batch: {proteins_per_batch}")
    print(f"  Graphs per batch: {adjusted_batch_size} ({proteins_per_batch} proteins × {ensemble_size} replicas)")
    print(f"  Structure noise std: {structure_noise_mag_std} Å")

    # Calculate number of batches needed
    num_batches = (len(indices) + proteins_per_batch - 1) // proteins_per_batch
    print(f"  Total batches needed: {num_batches}")

    # Show batch breakdown for verification
    if getattr(args, 'verbose', False):
        print(f"  Batch breakdown:")
        for batch_idx in range(num_batches):
            batch_start = batch_idx * proteins_per_batch
            batch_end = min(batch_start + proteins_per_batch, len(indices))
            proteins_in_this_batch = batch_end - batch_start
            graphs_in_this_batch = proteins_in_this_batch * ensemble_size
            print(f"    Batch {batch_idx + 1}: {proteins_in_this_batch} proteins, {graphs_in_this_batch} graphs")

    model.eval()
    device = next(model.parameters()).device
    results = []

    # Extract dataset parameters for ensemble creation
    dataset_params = None
    if hasattr(dataset, 'graph_builder'):
        graph_builder = dataset.graph_builder
        dataset_params = {
            'k_neighbors': getattr(graph_builder, 'k', None),
            'k_farthest': getattr(graph_builder, 'k_farthest', None),
            'k_random': getattr(graph_builder, 'k_random', None),
            'max_edge_dist': getattr(graph_builder, 'max_edge_dist', None),
            'num_rbf_3d': getattr(graph_builder, 'num_rbf_3d', 16),
            'num_rbf_seq': getattr(graph_builder, 'num_rbf_seq', 16),
            'use_virtual_node': getattr(graph_builder, 'use_virtual_node', True),
            'no_source_indicator': getattr(graph_builder, 'no_source_indicator', False),
            'rbf_3d_min': getattr(graph_builder, 'rbf_3d_min', 2.0),
            'rbf_3d_max': getattr(graph_builder, 'rbf_3d_max', 350.0),
            'rbf_3d_spacing': getattr(graph_builder, 'rbf_3d_spacing', 'exponential'),
        }

    # Batch processing loop (only when total_proteins > proteins_per_batch)
    for batch_idx in range(num_batches):
        batch_start = batch_idx * proteins_per_batch
        batch_end = min(batch_start + proteins_per_batch, len(indices))
        batch_indices = indices[batch_start:batch_end]

        print(f"\nProcessing batch {batch_idx + 1}/{num_batches}: proteins {batch_start}-{batch_end-1}")

        try:
            # Create mega-batch with all protein ensembles
            print(f"  Creating ensemble mega-batch...")

            mega_batch_graphs = []
            protein_metadata = []

            for protein_pos, dataset_idx in enumerate(batch_indices):
                try:
                    # Get protein data from dataset
                    data, y_true, mask, time_value, dssp_targets = dataset[dataset_idx]
                    structure_name = getattr(data, 'name', f'structure_{dataset_idx}')

                    # Get the original entry from dataset for ensemble creation
                    if hasattr(dataset, 'entries'):
                        # CathDataset has entries list
                        original_entry = dataset.entries[dataset_idx]
                    elif hasattr(dataset, 'protein_entry'):
                        # SingleProteinDataset has single protein_entry
                        original_entry = dataset.protein_entry
                    else:
                        raise AttributeError(f"Cannot access original entry from dataset type: {type(dataset)}")

                    # Create ensemble for this protein using original entry
                    ensemble_batch = create_structural_ensemble(
                        original_entry,
                        ensemble_size=ensemble_size,
                        structure_noise_mag_std=structure_noise_mag_std,
                        uncertainty_struct_noise_scaling=uncertainty_struct_noise_scaling,
                        device=device,
                        args=args,
                        dataset_params=dataset_params
                    )

                    # Add ensemble graphs to mega batch
                    ensemble_graphs = ensemble_batch.to_data_list()
                    for replica_idx, graph in enumerate(ensemble_graphs):
                        graph.protein_idx = dataset_idx
                        graph.replica_idx = replica_idx
                        graph.protein_pos = protein_pos  # Position within this batch
                        mega_batch_graphs.append(graph)

                    # Get sequence length from original entry
                    seq_len = len(original_entry['seq'])

                    # Store metadata
                    protein_metadata.append({
                        'dataset_idx': dataset_idx,
                        'structure_name': structure_name,
                        'y_true': y_true,
                        'seq_len': seq_len,
                        'protein_pos': protein_pos
                    })

                except Exception as e:
                    print(f"    Error creating ensemble for protein {dataset_idx}: {e}")
                    # Add error result and continue
                    results.append({
                        'structure_idx': dataset_idx,
                        'structure_name': f'structure_{dataset_idx}',
                        'error': str(e),
                        'ensemble_size': ensemble_size
                    })
                    continue

            if not mega_batch_graphs:
                print(f"    No valid graphs in batch, skipping...")
                continue

            # Create mega-batch
            print(f"    Batching {len(mega_batch_graphs)} graphs ({len(protein_metadata)} proteins × {ensemble_size} replicas)...")
            mega_batch = Batch.from_data_list(mega_batch_graphs).to(device)

            # Initialize states for all proteins × replicas
            use_virtual_node = dataset_params.get('use_virtual_node', True) if dataset_params else True
            dirichlet_concentration = getattr(args, 'dirichlet_concentration', 20.0) if args else 20.0
            flow_temp = getattr(args, 'flow_temp', 1.0) if args else 1.0

            # Get max sequence length in this batch
            max_seq_len = max(meta['seq_len'] for meta in protein_metadata)
            total_graphs = len(mega_batch_graphs)

            # Initialize Dirichlet states: [total_graphs, max_seq_len, K]
            dirichlet_dist = Dirichlet(dirichlet_concentration * torch.ones(K, device=device))
            x_mega = torch.zeros(total_graphs, max_seq_len, K, device=device)

            graph_idx = 0
            for protein_meta in protein_metadata:
                seq_len = protein_meta['seq_len']
                for replica_idx in range(ensemble_size):
                    # Sample different initial state for each replica
                    x_init = dirichlet_dist.sample((seq_len,))  # [seq_len, K]
                    x_mega[graph_idx, :seq_len, :] = x_init
                    graph_idx += 1

            print(f"    Initialized {total_graphs} graph states with max_seq_len={max_seq_len}")

            # Time integration loop
            times = torch.linspace(t_min, T, steps, device=device)
            dt = (T - t_min) / (steps - 1) if steps > 1 else 0

            with torch.no_grad():
                for step_idx, t_val in enumerate(tqdm(times[:-1], desc=f"Batch {batch_idx + 1} ensemble sampling")):

                    # Apply consensus within each protein's ensemble (vectorized)
                    if consensus_strength > 0:
                        # Get ensemble method from args, default to arithmetic
                        ensemble_method = getattr(args, 'ensemble_method', 'arithmetic') if args else 'arithmetic'

                        protein_graph_idx = 0
                        for protein_meta in protein_metadata:
                            seq_len = protein_meta['seq_len']

                            # Extract this protein's replicas
                            protein_states = x_mega[protein_graph_idx:protein_graph_idx + ensemble_size, :seq_len, :]  # [ensemble_size, seq_len, K]

                            # Apply consensus blending using the specified method
                            blended_states = compute_ensemble_consensus(protein_states, consensus_strength, ensemble_method, ensemble_size)

                            # Write back to mega tensor
                            x_mega[protein_graph_idx:protein_graph_idx + ensemble_size, :seq_len, :] = blended_states
                            protein_graph_idx += ensemble_size

                    # Mega-batched model prediction (single forward pass for ALL graphs!)
                    t_tensor = torch.full((total_graphs,), t_val, device=device)
                    model_output = model(mega_batch, t_tensor, x_mega)

                    # Handle dict (multi-aux), tuple (DSSP-only), or tensor (single head)
                    if isinstance(model_output, dict):
                        logits = model_output['sequence']
                    elif isinstance(model_output, tuple):
                        logits = model_output[0]
                    else:
                        logits = model_output

                    pred_target = torch.softmax(logits / flow_temp, dim=-1)

                    # Split predictions back to [total_graphs, max_seq_len, K] format
                    pred_mega = torch.zeros(total_graphs, max_seq_len, K, device=device)

                    node_offset = 0
                    for graph_idx in range(total_graphs):
                        # Count nodes for this graph
                        mask = (mega_batch.batch == graph_idx)
                        num_nodes = mask.sum().item()

                        # Extract predictions for this graph
                        graph_predictions = pred_target[node_offset:node_offset + num_nodes]

                        # Handle virtual nodes
                        if use_virtual_node and num_nodes > 0:
                            real_nodes = num_nodes - 1
                            seq_len = min(real_nodes, max_seq_len)
                            pred_mega[graph_idx, :seq_len, :] = graph_predictions[:seq_len]
                        else:
                            seq_len = min(num_nodes, max_seq_len)
                            pred_mega[graph_idx, :seq_len, :] = graph_predictions[:seq_len]

                        node_offset += num_nodes

                    # Mega-batched velocity computation (single call for ALL graphs!)
                    use_smoothed_targets = getattr(args, 'use_smoothed_targets', False) if args else False
                    use_c_factor = getattr(args, 'use_c_factor', False) if args else False
                    if step_idx == 0:  # Print only once at the start
                        print(f"[SAMPLING] use_c_factor: {use_c_factor}", flush=True)

                    velocities = model.cond_flow.velocity(
                        x_mega, pred_mega, t_tensor,
                        use_virtual_node=use_virtual_node,
                        use_smoothed_targets=getattr(args, 'use_smoothed_targets', False) if args else False,
                        use_c_factor=getattr(args, 'use_c_factor', False) if args else False
                    )

                    # Update states with Euler step
                    x_new = x_mega + dt * velocities
                    x_mega = simplex_proj(x_new)

            # Extract results per protein
            print(f"    Generating final sequences...")

            protein_graph_idx = 0
            for protein_meta in protein_metadata:
                try:
                    seq_len = protein_meta['seq_len']

                    # Extract this protein's replica states
                    protein_states = x_mega[protein_graph_idx:protein_graph_idx + ensemble_size, :seq_len, :]  # [ensemble_size, seq_len, K]

                    # Generate consensus sequence for this protein
                    if consensus_strength > 0:
                        # Get ensemble method from args, default to arithmetic
                        ensemble_method = getattr(args, 'ensemble_method', 'arithmetic') if args else 'arithmetic'

                        # Apply the chosen consensus method directly for final sequence generation
                        if ensemble_method == 'arithmetic':
                            # Standard arithmetic mean in probability space
                            final_consensus = protein_states.mean(dim=0)  # [seq_len, K]
                        elif ensemble_method == 'geometric':
                            # Geometric mean in log space
                            eps = 1e-8
                            log_states = torch.log(protein_states + eps)  # [ensemble_size, seq_len, K]
                            log_mean = log_states.mean(dim=0)  # [seq_len, K]
                            final_consensus = torch.exp(log_mean)  # [seq_len, K]
                            # Normalize to ensure it's on the simplex
                            final_consensus = final_consensus / final_consensus.sum(dim=-1, keepdim=True)
                        else:
                            # Fallback to arithmetic if unknown method
                            final_consensus = protein_states.mean(dim=0)  # [seq_len, K]

                        final_sequence = final_consensus.argmax(-1).tolist()
                    else:
                        final_sequence = protein_states[0].argmax(-1).tolist()  # Use first replica

                    # Calculate accuracy if ground truth available
                    accuracy = None
                    true_seq = None
                    if protein_meta['y_true'] is not None:
                        y_true = protein_meta['y_true']
                        if use_virtual_node and y_true.shape[0] > seq_len:
                            y_true = y_true[:seq_len]  # Remove virtual node

                        true_seq = y_true.argmax(-1).tolist()
                        correct = sum(p == t for p, t in zip(final_sequence, true_seq))
                        accuracy = correct / len(final_sequence) * 100

                    # Convert to amino acid names
                    predicted_aa = []
                    for idx_val in final_sequence:
                        if 0 <= idx_val < len(IDX_TO_AA):
                            predicted_aa.append(IDX_TO_AA[idx_val])
                        else:
                            predicted_aa.append('XXX')

                    result = {
                        'structure_idx': protein_meta['dataset_idx'],
                        'structure_name': protein_meta['structure_name'],
                        'ensemble_size': ensemble_size,
                        'consensus_strength': consensus_strength,
                        'length': len(final_sequence),
                        'predicted_indices': final_sequence,
                        'predicted_aa': predicted_aa,
                        'predicted_sequence': ''.join([THREE_TO_ONE.get(aa, 'X') for aa in predicted_aa]),
                        'true_indices': true_seq,
                        'accuracy': accuracy,
                        'final_probabilities': (final_consensus.cpu().numpy() if consensus_strength > 0
                                              else protein_states[0].cpu().numpy())
                    }

                    results.append(result)
                    protein_graph_idx += ensemble_size

                except Exception as e:
                    print(f"      Error generating sequence for protein {protein_meta['dataset_idx']}: {e}")
                    results.append({
                        'structure_idx': protein_meta['dataset_idx'],
                        'structure_name': protein_meta['structure_name'],
                        'error': str(e),
                        'ensemble_size': ensemble_size
                    })
                    protein_graph_idx += ensemble_size

        except Exception as e:
            print(f"    Error processing batch {batch_idx}: {e}")
            # Add error results for all proteins in this batch
            for idx in batch_indices:
                results.append({
                    'structure_idx': idx,
                    'structure_name': f'structure_{idx}',
                    'error': str(e),
                    'ensemble_size': ensemble_size
                })

    print(f"\nCompleted ensemble sampling: {len(results)} proteins with ensemble_size={ensemble_size}")
    return results


"""
Utility functions for protein sequence sampling.

This module contains helper functions, classes, and utilities that support
the core sampling functionality in sample.py.
"""

import argparse
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.distributions import Dirichlet
from torch.utils.data import Dataset
from tqdm import tqdm

# Amino acid mappings (constants used across sampling utilities)
THREE_TO_ONE = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    'XXX': 'X'
}

AA_TO_IDX = {
    'ALA': 0, 'CYS': 1, 'ASP': 2, 'GLU': 3, 'PHE': 4, 'GLY': 5, 'HIS': 6, 'ILE': 7,
    'LYS': 8, 'LEU': 9, 'MET': 10, 'ASN': 11, 'PRO': 12, 'GLN': 13, 'ARG': 14, 'SER': 15,
    'THR': 16, 'VAL': 17, 'TRP': 18, 'TYR': 19, 'XXX': 20
}

IDX_TO_AA = ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE',
             'LYS', 'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER',
             'THR', 'VAL', 'TRP', 'TYR', 'XXX']

SINGLE_TO_TRIPLE = {
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
    'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
    'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
    'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL',
    'X': 'XXX'
}


# Input processing functions for PDB download and chain selection
def download_pdb(pdb_id: str, save_path: str = None) -> str:
    """Download PDB file from RCSB PDB database."""
    import tempfile

    import requests

    if save_path is None:
        temp_dir = tempfile.mkdtemp()
        save_path = os.path.join(temp_dir, f"{pdb_id.lower()}.pdb")

    url = f"https://files.rcsb.org/download/{pdb_id.lower()}.pdb"
    print(f"Downloading {pdb_id} from RCSB PDB...")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        with open(save_path, 'w') as f:
            f.write(response.text)

        print(f"Downloaded to: {save_path}")
        return save_path
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to download PDB {pdb_id}: {e}")


def extract_chain_from_pdb(pdb_path: str, chain_id: str) -> str:
    """Extract specific chain from PDB file using BioPython."""
    try:
        from Bio import PDB
        output_path = f"{os.path.splitext(pdb_path)[0]}_chain{chain_id.upper()}.pdb"

        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_path)

        # Find the specified chain
        target_chain = None
        for model in structure:
            if chain_id.upper() in [chain.id for chain in model]:
                target_chain = model[chain_id.upper()]
                break

        if target_chain is None:
            available_chains = []
            for model in structure:
                available_chains.extend([chain.id for chain in model])
            raise ValueError(f"Chain {chain_id} not found. Available chains: {list(set(available_chains))}")

        # Save only the specified chain
        io = PDB.PDBIO()
        io.set_structure(target_chain)
        io.save(output_path)

        print(f"Extracted chain {chain_id} to: {output_path}")
        return output_path
    except ImportError:
        raise ImportError("BioPython is required for chain extraction. Install with: pip install biopython")
    except Exception as e:
        raise RuntimeError(f"Failed to extract chain {chain_id}: {e}")


def compute_dssp_for_entry(pdb_path, chain_id=None, verbose=False):
    """
    Compute DSSP secondary structure for a PDB file.

    Args:
        pdb_path: Path to PDB file
        chain_id: Specific chain ID (optional, defaults to longest chain)
        verbose: Print debug info

    Returns:
        List of DSSP characters (e.g., ['H', 'E', '-', ...]) or None if computation fails
    """
    try:
        # Import here to avoid circular dependencies
        from eval.generate_dssp_from_pdbs import compute_dssp_for_pdb
        from Bio.PDB import PDBParser

        # If chain_id not specified, automatically select the longest chain
        if chain_id is None:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("temp", pdb_path)
            model = structure[0]

            # Find the longest chain by counting protein residues
            longest_chain_id = None
            max_residues = 0

            for chain in model:
                # Count protein residues (excluding heteroatoms)
                protein_residues = [r for r in chain.get_residues() if r.get_id()[0] == " "]
                num_residues = len(protein_residues)

                if num_residues > max_residues:
                    max_residues = num_residues
                    longest_chain_id = chain.id

            if longest_chain_id is not None:
                chain_id = longest_chain_id
                if verbose:
                    print(f"[DSSP] Auto-selected longest chain: {chain_id} ({max_residues} residues)")
            else:
                if verbose:
                    print(f"[DSSP] Warning: No protein chains found in {pdb_path}")
                return None

        sequence, dssp_8_array, dssp_3_array = compute_dssp_for_pdb(
            pdb_path,
            model_index=0,
            chain_id=chain_id
        )

        if verbose:
            print(f"[DSSP] Computed DSSP for {pdb_path}")
            print(f"[DSSP] Chain: {chain_id}")
            print(f"[DSSP] Sequence length: {len(sequence)}")
            print(f"[DSSP] DSSP length: {len(dssp_8_array)}")

        return dssp_8_array

    except Exception as e:
        if verbose:
            print(f"[DSSP] Warning: Failed to compute DSSP for {pdb_path}: {e}")
        # Return None if computation fails
        return None


def should_compute_dssp_for_sampling(args):
    """
    Determine if DSSP should be computed based on sampling arguments.

    Args:
        args: Argument namespace with sampling parameters

    Returns:
        Boolean indicating if DSSP computation is needed
    """
    # Check ensemble size
    if getattr(args, 'ensemble_size', 1) < 2:
        # Check DSSP-related flags
        if getattr(args, 'dssp_initialization', False):
            return True
        if getattr(args, 'dssp_guidance', False):
            return True
        if getattr(args, 'filter_out_missing_flanks', False):
            return True
        # Check multi-sample selection flags
        if getattr(args, 'pick_based_on_dssp', False):
            return True

    return False


def pdb_to_dict_format(pdb_path: str, structure_name: str, compute_dssp=False,
                       verbose=False, keep_chains: list = None) -> dict:
    """
    Convert PDB file to dictionary format compatible with GraphBuilder.build_from_dict().
    Uses existing PDBProcessor to extract coordinates and B-factors.

    By default, selects only the longest protein chain from multi-chain PDB files.
    When keep_chains is provided, all listed chains are read and concatenated in the
    order given, and the entry dict will include a 'chain_ids' key mapping each
    residue position to its originating chain letter.

    Args:
        pdb_path: Path to PDB file
        structure_name: Name for the structure
        compute_dssp: If True, compute DSSP and add to entry dict
        verbose: Print processing details
        keep_chains: Optional list of chain IDs to keep and concatenate, e.g. ['B', 'G'].
                     Order matters — residues are concatenated in the order given.
                     When None (default), behaves as before: longest chain only.

    Returns:
        Entry dictionary with structure data. When keep_chains is used the dict
        includes 'chain_ids': list[str] of length L giving the chain letter for
        each residue, and 'chain_offsets': dict mapping chain_id -> 0-based start
        index in the concatenated sequence.
    """
    from pathlib import Path
    import tempfile
    from Bio.PDB import PDBParser, PDBIO, Select

    from data.pdb_processor import PDBProcessor

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("temp", pdb_path)
    model = structure[0]

    # -------------------------------------------------------------------------
    # Multi-chain concatenation mode
    # -------------------------------------------------------------------------
    if keep_chains is not None:
        if verbose:
            print(f"[PDB] Multi-chain mode: keeping chains {keep_chains} from {pdb_path}")

        all_seq       = []
        all_N         = []
        all_CA        = []
        all_C         = []
        all_O         = []
        all_b_factors = []
        all_chain_ids = []   # per-residue chain letter
        chain_offsets = {}   # chain_id -> start index in concatenated sequence

        for chain_id in keep_chains:
            # Write this chain to a temp file so PDBProcessor can parse it cleanly
            class ChainSelect(Select):
                def __init__(self, cid): self.cid = cid
                def accept_chain(self, c): return c.id == self.cid

            io = PDBIO()
            io.set_structure(structure)
            tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False)
            tmp_path = tmp.name
            tmp.close()
            io.save(tmp_path, ChainSelect(chain_id))

            try:
                pdb_dir   = os.path.dirname(tmp_path)
                processor = PDBProcessor(pdb_directory=pdb_dir, verbose=False)
                seq, coords_dict = processor._extract_sequence_and_coords_from_file(Path(tmp_path))
                if seq is None or coords_dict is None:
                    raise ValueError(f"Failed to extract chain {chain_id} from {pdb_path}")

                b_fac = processor._extract_b_factors_from_file(Path(tmp_path))
                if b_fac is None:
                    b_fac = [50.0] * len(seq)
                    if verbose:
                        print(f"[PDB] Warning: using default B-factors for chain {chain_id}")
            finally:
                os.unlink(tmp_path)

            chain_offsets[chain_id] = len(all_seq)
            all_seq.extend(list(seq))
            all_chain_ids.extend([chain_id] * len(seq))
            all_N.append(coords_dict['N'])
            all_CA.append(coords_dict['CA'])
            all_C.append(coords_dict['C'])
            all_O.append(coords_dict['O'])
            b_arr = np.array([float(x) for x in b_fac], dtype=np.float32)
            all_b_factors.append(b_arr)

            if verbose:
                print(f"[PDB]   chain {chain_id}: {len(seq)} residues (offset {chain_offsets[chain_id]})")

        concat_seq      = ''.join(all_seq)
        concat_N        = np.concatenate(all_N,        axis=0)
        concat_CA       = np.concatenate(all_CA,       axis=0)
        concat_C        = np.concatenate(all_C,        axis=0)
        concat_O        = np.concatenate(all_O,        axis=0)
        concat_b        = np.concatenate(all_b_factors, axis=0)

        entry = {
            'name':          structure_name,
            'seq':           concat_seq,
            'coords': {
                'N':  concat_N,
                'CA': concat_CA,
                'C':  concat_C,
                'O':  concat_O,
            },
            'b_factors':     concat_b,
            'source':        'pdb',
            'chain_ids':     all_chain_ids,   # list[str], length L
            'chain_offsets': chain_offsets,   # dict chain_id -> int
        }

        if compute_dssp and verbose:
            print(f"[DSSP] Warning: DSSP not computed for multi-chain concatenated input")

        if verbose:
            print(f"[PDB] Concatenated structure: {len(concat_seq)} residues total")

        return entry

    # -------------------------------------------------------------------------
    # Original single-chain mode (unchanged)
    # -------------------------------------------------------------------------

    # Find the longest chain
    longest_chain_id = None
    max_residues = 0
    for chain in model:
        protein_residues = [r for r in chain.get_residues() if r.get_id()[0] == " "]
        num_residues = len(protein_residues)
        if num_residues > max_residues:
            max_residues = num_residues
            longest_chain_id = chain.id

    if longest_chain_id is None:
        raise ValueError(f"No protein chains found in {pdb_path}")

    num_chains = len([c for c in model if len([r for r in c.get_residues() if r.get_id()[0] == " "]) > 0])
    temp_pdb_path = None

    # If multiple protein chains exist, extract the longest one to a temp file
    if num_chains > 1:
        if verbose:
            print(f"[PDB] Multi-chain PDB detected, extracting longest chain: {longest_chain_id} ({max_residues} residues)")

        class ChainSelect(Select):
            def __init__(self, chain_id):
                self.chain_id = chain_id
            def accept_chain(self, chain):
                return chain.id == self.chain_id

        io = PDBIO()
        io.set_structure(structure)
        temp_pdb = tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False)
        temp_pdb_path = temp_pdb.name
        temp_pdb.close()
        io.save(temp_pdb_path, ChainSelect(longest_chain_id))
        pdb_to_process = temp_pdb_path
        selected_chain_id = longest_chain_id
    else:
        pdb_to_process = pdb_path
        selected_chain_id = longest_chain_id

    try:
        pdb_dir = os.path.dirname(pdb_to_process)
        processor = PDBProcessor(pdb_directory=pdb_dir, verbose=False)

        sequence, coords_dict = processor._extract_sequence_and_coords_from_file(Path(pdb_to_process))
        if sequence is None or coords_dict is None:
            raise ValueError(f"Failed to extract sequence/coordinates from {pdb_path}")

        b_factors = processor._extract_b_factors_from_file(Path(pdb_to_process))
        if b_factors is None:
            b_factors = [50.0] * len(sequence)
            if verbose:
                print(f"Warning: Using default B-factors for {structure_name}")

        if not isinstance(b_factors, np.ndarray):
            b_factors = np.array([float(x) for x in b_factors], dtype=np.float32)
        else:
            b_factors = b_factors.astype(np.float32)

        entry = {
            'name': structure_name,
            'seq': sequence,
            'coords': {
                'N': coords_dict['N'],
                'CA': coords_dict['CA'],
                'C': coords_dict['C'],
                'O': coords_dict['O']
            },
            'b_factors': b_factors,
            'source': 'pdb'
        }

        if compute_dssp:
            dssp_array = compute_dssp_for_entry(pdb_path, chain_id=selected_chain_id, verbose=verbose)
            if dssp_array is not None:
                entry['dssp'] = dssp_array
                if verbose:
                    print(f"[DSSP] Added DSSP to entry: {len(dssp_array)} positions")
            else:
                if verbose:
                    print(f"[DSSP] Warning: DSSP computation failed, entry will not have DSSP")

    finally:
        if temp_pdb_path is not None and os.path.exists(temp_pdb_path):
            os.unlink(temp_pdb_path)

    return entry


def cif_to_dict_format(cif_path: str, structure_name: str, compute_dssp=False, verbose=False) -> dict:
    """
    Convert CIF file to dictionary format using existing CIF parser.

    Args:
        cif_path: Path to CIF file
        structure_name: Name for the structure
        compute_dssp: If True, compute DSSP and add to entry dict
        verbose: Print processing details

    Returns:
        Entry dictionary with structure data
    """
    from data.cif_parser import parse_cif_backbone_auto

    coords, scores, residue_types, source = parse_cif_backbone_auto(cif_path)

    # Convert to dict format
    coords_dict = {
        'N': coords[:, 0, :].numpy(),
        'CA': coords[:, 1, :].numpy(),
        'C': coords[:, 2, :].numpy(),
        'O': coords[:, 3, :].numpy()
    }

    sequence = ''.join([THREE_TO_ONE.get(aa, 'X') for aa in residue_types])

    entry = {
        'name': structure_name,
        'seq': sequence,
        'coords': coords_dict,
        'b_factors': scores.numpy(),
        'source': source
    }

    # Add DSSP if requested
    # Note: For CIF files, we would need to convert to PDB first or use mmCIF-compatible DSSP
    # For now, we skip DSSP computation for CIF files
    if compute_dssp and verbose:
        print(f"[DSSP] Warning: DSSP computation from CIF files not yet implemented, skipping")

    return entry


def process_input_specification(input_spec: str, compute_dssp=False, verbose=False,
                                keep_chains: list = None) -> tuple:
    """
    Process input specification and return (structure_dict, temp_files_to_cleanup).

    Supported formats:
    - Local PDB files: '/path/to/file.pdb' or 'protein.pdb'
    - Local CIF files: '/path/to/file.cif' or 'protein.cif'
    - PDB IDs: '1abc'
    - PDB ID with chain: '1abc.C'

    Args:
        input_spec: Input specification string
        compute_dssp: If True, compute DSSP and add to entry dict
        verbose: Print processing details
        keep_chains: Optional list of chain IDs to keep and concatenate (PDB files only).
                     When provided, all listed chains are concatenated in order and the
                     returned entry dict includes 'chain_ids' and 'chain_offsets'.
                     When None (default), only the longest chain is kept.

    Returns:
        tuple: (structure_dict, temp_files_list)
    """
    import tempfile

    temp_files = []
    input_type = None

    try:
        # 1. Check if it's a local file path (absolute or relative)
        if os.path.exists(input_spec):
            input_type = "local_file"
            print(f"Detected local file: {input_spec}")

            structure_name = os.path.splitext(os.path.basename(input_spec))[0]

            if input_spec.lower().endswith('.pdb'):
                print("Processing PDB file...")
                entry = pdb_to_dict_format(input_spec, structure_name, compute_dssp=compute_dssp,
                                           verbose=verbose, keep_chains=keep_chains)
                return entry, temp_files
            elif input_spec.lower().endswith('.cif'):
                print("Processing CIF file...")
                entry = cif_to_dict_format(input_spec, structure_name, compute_dssp=compute_dssp, verbose=verbose)
                return entry, temp_files
            else:
                raise ValueError(f"Unsupported file format: {input_spec}. Supported: .pdb, .cif")

        # 2. Parse as PDB ID (with optional chain)
        if '.' in input_spec:
            pdb_id, chain_id = input_spec.split('.', 1)
            input_type = "pdb_id_with_chain"
            if verbose:
                print(f"Detected PDB ID with chain: {pdb_id}.{chain_id}")
        else:
            pdb_id = input_spec
            chain_id = None
            input_type = "pdb_id"
            if verbose:
                print(f"Detected PDB ID: {pdb_id}")

        # 3. Validate PDB ID format (4 characters, alphanumeric)
        if not (len(pdb_id) == 4 and pdb_id.isalnum()):
            raise ValueError(
                f"Invalid PDB ID format: '{pdb_id}'. Expected 4 alphanumeric characters.\n"
                f"Examples: '1abc', '2XYZ', '1fcd.C' (with chain)"
            )

        # 4. Download PDB file
        temp_dir = tempfile.mkdtemp()
        if verbose:
            print(f"Creating temporary directory: {temp_dir}")

        pdb_path = download_pdb(pdb_id, os.path.join(temp_dir, f"{pdb_id.lower()}.pdb"))
        temp_files.append(pdb_path)

        # 5. Extract specific chain if requested
        if chain_id:
            print(f"Extracting chain {chain_id}...")
            pdb_path = extract_chain_from_pdb(pdb_path, chain_id)
            temp_files.append(pdb_path)
            structure_name = f"{pdb_id.upper()}.{chain_id.upper()}"
        else:
            structure_name = pdb_id.upper()

        # 6. Convert to dict format
        if verbose:
            print("Converting to internal dictionary format...")
        entry = pdb_to_dict_format(pdb_path, structure_name, compute_dssp=compute_dssp, verbose=verbose)

        print(f"Successfully processed {input_type}: {structure_name}")
        return entry, temp_files

    except Exception as e:
        # Cleanup temp files on error
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass

        # Provide helpful error message based on input type
        if input_type == "local_file":
            raise ValueError(f"Failed to process local file '{input_spec}': {e}")
        elif input_type in ["pdb_id", "pdb_id_with_chain"]:
            raise ValueError(f"Failed to process PDB ID '{input_spec}': {e}\n"
                           f"Make sure the PDB ID exists and is accessible from RCSB PDB database.")
        else:
            raise ValueError(f"Failed to process input '{input_spec}': {e}\n"
                           f"Supported formats:\n"
                           f"  - Local files: '/path/to/file.pdb', 'protein.cif'\n"
                           f"  - PDB IDs: '1abc', '2XYZ'\n"
                           f"  - PDB ID + chain: '1fcd.C', '2abc.A'")



class CustomInputDataset:
    """Minimal dataset for custom input compatibility with existing sampling functions."""
    def __init__(self, graph_data, entry, graph_builder=None):
        self.graph_data = graph_data
        self.entry = entry
        self.graph_builder = graph_builder

    def __getitem__(self, idx):
        if idx != 0:
            raise IndexError("Only one entry")
        return self.graph_data, None, None, 0.0, None  # data, y_true, mask, time, dssp

    def __len__(self):
        return 1


def compute_ensemble_consensus(protein_states, consensus_strength, ensemble_method='arithmetic', ensemble_size=None):
    """
    Compute ensemble consensus using either arithmetic or geometric averaging.

    Args:
        protein_states: [ensemble_size, seq_len, K] tensor of probability states
        consensus_strength: Float between 0-1, strength of consensus blending
        ensemble_method: 'arithmetic' or 'geometric'
        ensemble_size: Size of ensemble (auto-detected if None)

    Returns:
        Blended protein states with same shape as input, normalized on simplex
    """
    if ensemble_size is None:
        ensemble_size = protein_states.shape[0]

    if ensemble_method == 'arithmetic':
        # Standard arithmetic mean in probability space
        protein_mean = protein_states.mean(dim=0, keepdim=True)  # [1, seq_len, K]

        # Apply consensus blending
        blended_states = (1 - consensus_strength) * protein_states + \
                        consensus_strength * protein_mean.expand(ensemble_size, -1, -1)

    elif ensemble_method == 'geometric':
        # Geometric mean in log space
        eps = 1e-8  # Small epsilon to avoid log(0)

        # Convert to log space (add epsilon to avoid log(0))
        log_states = torch.log(protein_states + eps)  # [ensemble_size, seq_len, K]

        # Compute mean in log space
        log_mean = log_states.mean(dim=0, keepdim=True)  # [1, seq_len, K]

        # Apply consensus blending in log space
        blended_log_states = (1 - consensus_strength) * log_states + \
                            consensus_strength * log_mean.expand(ensemble_size, -1, -1)

        # Convert back to probability space
        blended_states = torch.exp(blended_log_states)

    else:
        raise ValueError(f"Unknown ensemble_method: {ensemble_method}. Choose 'arithmetic' or 'geometric'")

    # Ensure states remain on simplex (normalize to sum to 1)
    return simplex_proj(blended_states)


def compute_sampling_metrics(predicted_probabilities, ground_truth_onehot, data, model, args, device, use_virtual_node, K):
    """
    Compute comprehensive evaluation metrics for sampling results.

    Args:
        predicted_probabilities: [N, K] tensor of predicted probabilities
        ground_truth_onehot: [N, K] tensor of ground truth one-hot vectors
        data: Graph data object
        model: Trained model
        args: Arguments object
        device: Device to run computations on
        use_virtual_node: Whether virtual nodes are used
        K: Number of amino acid classes

    Returns:
        Dict containing evaluation metrics
    """
    with torch.no_grad():
        # Ensure tensors are on the same device
        if predicted_probabilities.device != device:
            predicted_probabilities = predicted_probabilities.to(device)
        if ground_truth_onehot.device != device:
            ground_truth_onehot = ground_truth_onehot.to(device)

        # Basic accuracy
        predicted_classes = predicted_probabilities.argmax(-1)
        true_classes = ground_truth_onehot.argmax(-1)

        # Create mask to exclude unknown residues (X = index 20) from accuracy calculation
        valid_mask = (true_classes != 20)  # Exclude positions where ground truth is X/unknown

        if valid_mask.sum() > 0:
            # Only calculate accuracy for non-unknown positions
            correct = (predicted_classes == true_classes).float()
            accuracy = correct[valid_mask].mean().item()

            # Cross-entropy loss with hard labels (only for valid positions)
            cce_loss_hard = F.cross_entropy(predicted_probabilities[valid_mask], true_classes[valid_mask]).item()

            # Cross-entropy loss with soft labels (only for valid positions)
            eps = 1e-8
            log_probs = torch.log(predicted_probabilities + eps)
            cce_loss_smooth = -(ground_truth_onehot[valid_mask] * log_probs[valid_mask]).sum(-1).mean().item()

            # Per-position confidence (only for valid positions)
            confidence = predicted_probabilities[valid_mask].max(-1)[0].mean().item()

            # Entropy of predictions (only for valid positions)
            entropy = -(predicted_probabilities[valid_mask] * torch.log(predicted_probabilities[valid_mask] + eps)).sum(-1).mean().item()

            # Top-k accuracy (k=3, only for valid positions)
            top3_predictions = predicted_probabilities[valid_mask].topk(3, dim=-1)[1]
            top3_correct = (top3_predictions == true_classes[valid_mask].unsqueeze(-1)).any(-1).float()
            top3_accuracy = top3_correct.mean().item()
        else:
            # If all positions are unknown, set default values
            accuracy = 0.0
            cce_loss_hard = float('inf')
            cce_loss_smooth = float('inf')
            confidence = 0.0
            entropy = 0.0
            top3_accuracy = 0.0

        # Per-class accuracy (excluding unknown residues)
        per_class_correct = torch.zeros(K, device=device)
        per_class_total = torch.zeros(K, device=device)

        if valid_mask.sum() > 0:
            correct = (predicted_classes == true_classes).float()
            for k in range(K):
                if k == 20:  # Skip unknown class (X/XXX)
                    continue
                mask = (true_classes == k) & valid_mask  # Only count valid positions
                if mask.sum() > 0:
                    per_class_correct[k] = correct[mask].sum()
                    per_class_total[k] = mask.sum()

        eps = 1e-8
        per_class_accuracy = per_class_correct / (per_class_total + eps)

        metrics = {
            'accuracy': accuracy,
            'cce_loss_hard': cce_loss_hard,
            'cce_loss_smooth': cce_loss_smooth,
            'confidence': confidence,
            'entropy': entropy,
            'top3_accuracy': top3_accuracy,
            'per_class_accuracy': per_class_accuracy.cpu().numpy(),
            'total_positions': len(true_classes),
            'valid_positions': valid_mask.sum().item(),
            'unknown_positions': (true_classes == 20).sum().item(),
            'fraction_valid': valid_mask.sum().item() / len(true_classes) if len(true_classes) > 0 else 0.0
        }

        return metrics


def simplex_proj(seq):
    """
    Project sequences onto the probability simplex using Euclidean projection.

    This ensures that each position's probability distribution sums to 1 and
    all probabilities are non-negative, which maintains the proper simplex constraint.

    Args:
        seq: Tensor of shape [B, N, K] representing probability distributions

    Returns:
        Projected tensor of the same shape
    """
    # Handle negative values first by clamping to 0
    seq_pos = torch.clamp(seq, min=0.0)

    # Normalize to ensure sum equals 1 for each position
    seq_sum = seq_pos.sum(-1, keepdim=True)

    # Add small epsilon to avoid division by zero
    eps = 1e-8
    seq_normalized = seq_pos / (seq_sum + eps)

    return seq_normalized


def compute_sequence_perplexity(final_probabilities):
    """
    Compute sequence-level perplexity from final probability distributions.

    Perplexity = exp(-1/N * sum(log(p_i))) where p_i is the probability of the
    predicted amino acid at position i. Lower perplexity indicates higher confidence.

    Args:
        final_probabilities: Tensor of shape [N, K] with amino acid probabilities

    Returns:
        Perplexity value (float)
    """
    import torch

    eps = 1e-10
    # Get the probability of the argmax (predicted) amino acid at each position
    max_probs = final_probabilities.max(dim=-1)[0]  # [N]
    # Compute negative log-likelihood
    nll = -torch.log(max_probs + eps).mean()  # Average over positions
    # Perplexity is exp of NLL
    perplexity = torch.exp(nll).item()
    return perplexity


def select_best_sample(replicate_results, primary_metric='dssp'):
    """
    Select the best-performing sample based on primary metric with intelligent tie-breaking.

    Selection strategy:
    1. Primary metric (DSSP accuracy or perplexity)
    2. If tied: Check if any tied sample has first residue != 'M'
       - Prefer non-'M' starting sequences (more flexible N-terminus)
    3. If still tied: Use secondary metric
       - If primary='dssp', secondary is perplexity (lower is better)
       - If primary='perplexity', secondary is DSSP accuracy (higher is better)
    4. If still tied: Take first occurrence

    Args:
        replicate_results: List of result dicts from sample_chain_with_replicates()
                          Each must contain 'final_probabilities', 'predicted_sequence',
                          and optionally 'final_dssp_accuracy' and 'sequence_perplexity'
        primary_metric: Primary selection criterion ('dssp' or 'perplexity')

    Returns:
        Best result dictionary with added fields:
            'selected_reason': str explaining selection
            'all_primary_metrics': list of all primary metric values
    """
    import torch
    import numpy as np

    if not replicate_results:
        raise ValueError("Empty replicate_results list.")

    # Ensure all results have perplexity computed
    for r in replicate_results:
        if 'sequence_perplexity' not in r:
            # Compute perplexity if not present
            if r.get('final_probabilities') is not None:
                probs = r['final_probabilities']
                if isinstance(probs, np.ndarray):
                    probs = torch.tensor(probs)
                r['sequence_perplexity'] = compute_sequence_perplexity(probs)
            else:
                raise ValueError("Missing 'final_probabilities' for perplexity computation.")

    # Extract metrics based on primary criterion
    if primary_metric == 'dssp':
        if 'final_dssp_accuracy' not in replicate_results[0]:
            raise ValueError("DSSP accuracy not available. Ensure DSSP targets are provided.")

        primary_values = [r['final_dssp_accuracy'] for r in replicate_results]
        best_primary = max(primary_values)  # Higher DSSP accuracy is better
        best_indices = [i for i, v in enumerate(primary_values) if v == best_primary]

    elif primary_metric == 'perplexity':
        primary_values = [r['sequence_perplexity'] for r in replicate_results]
        best_primary = min(primary_values)  # Lower perplexity is better
        best_indices = [i for i, v in enumerate(primary_values) if v == best_primary]

    else:
        raise ValueError(f"Invalid primary_metric: {primary_metric}. Must be 'dssp' or 'perplexity'.")

    # Stage 1: Primary metric selection
    if len(best_indices) == 1:
        best_idx = best_indices[0]
        if primary_metric == 'dssp':
            reason = f"Highest DSSP accuracy: {best_primary:.4f}"
        else:
            reason = f"Lowest perplexity: {best_primary:.4f}"
    else:
        # Stage 2: Tie-breaking by first residue
        non_M_indices = [i for i in best_indices
                        if len(replicate_results[i]['predicted_sequence']) > 0
                        and replicate_results[i]['predicted_sequence'][0] != 'M']

        if len(non_M_indices) > 0 and len(non_M_indices) < len(best_indices):
            # Some start with non-M, prefer those
            best_indices = non_M_indices
            if primary_metric == 'dssp':
                reason = f"Tied DSSP accuracy: {best_primary:.4f}, selected non-M start"
            else:
                reason = f"Tied perplexity: {best_primary:.4f}, selected non-M start"

        # Stage 3: Secondary metric tie-breaking
        if len(best_indices) > 1:
            if primary_metric == 'dssp':
                # Use perplexity as secondary (lower is better)
                secondary_values = [replicate_results[i]['sequence_perplexity'] for i in best_indices]
                best_secondary_idx = best_indices[np.argmin(secondary_values)]
                best_secondary_value = min(secondary_values)
                best_idx = best_secondary_idx
                reason = f"Tied DSSP accuracy: {best_primary:.4f}, selected by perplexity: {best_secondary_value:.4f}"

            else:  # primary_metric == 'perplexity'
                # Use DSSP as secondary (higher is better)
                if 'final_dssp_accuracy' in replicate_results[0]:
                    secondary_values = [replicate_results[i]['final_dssp_accuracy'] for i in best_indices]
                    best_secondary_idx = best_indices[np.argmax(secondary_values)]
                    best_secondary_value = max(secondary_values)
                    best_idx = best_secondary_idx
                    reason = f"Tied perplexity: {best_primary:.4f}, selected by DSSP accuracy: {best_secondary_value:.4f}"
                else:
                    # DSSP not available, take first
                    best_idx = best_indices[0]
                    reason = f"Tied perplexity: {best_primary:.4f}, selected first"
        else:
            best_idx = best_indices[0]

    # Prepare result
    best_result = replicate_results[best_idx].copy()
    best_result['selected_reason'] = reason
    best_result['all_primary_metrics'] = primary_values

    # Add secondary metrics for reference
    if primary_metric == 'dssp':
        best_result['all_perplexities'] = [r['sequence_perplexity'] for r in replicate_results]
    else:
        if 'final_dssp_accuracy' in replicate_results[0]:
            best_result['all_dssp_accuracies'] = [r['final_dssp_accuracy'] for r in replicate_results]

    return best_result


def apply_recycle_clipping(x_T, args, device, N, K, model=None, batched_data=None,
                           ever_flipped=None, T=8.0):
    """
    Apply selective position clipping between recycling round 1 and round 2.

    For each replicate, computes a fix_mask [N] bool tensor identifying confident
    positions. Confident positions are clamped to one_hot(argmax(x_T[rep])).
    Uncertain positions are re-initialized with fresh Dirichlet noise.

    Args:
        x_T: [num_replicates, N, K] final simplex state after round 1
        args: Parsed args namespace with recycle_clip_criterion and related flags
        device: torch.device
        N: Number of real residue positions (excluding virtual node)
        K: Number of AA classes (21)
        model: DFM model — required only for aux_consistency criterion
        batched_data: Batched graph data — required only for aux_consistency criterion
        ever_flipped: [num_replicates, N] bool tensor — required for flip_ever criterion.
                      True at position i if argmax(p_hat) ever changed during round 1.
        T: Time value used for aux_consistency forward pass (should match args.T)

    Returns:
        x_clipped: [num_replicates_out, N, K] initial state for round 2.
                   For replicate_consensus: [1, N, K] (single run from consensus).
                   For all other criteria: same num_replicates as input.
        n_fixed_list: list of int, number of fixed positions per output replicate
    """
    import torch
    import torch.nn.functional as F
    from torch.distributions import Dirichlet
    from data.aa_constants import ELECTROSTATIC_LABEL, GEOM_TOPOLOGY_LABEL

    criterion = getattr(args, 'recycle_clip_criterion', None)
    fix_fraction = getattr(args, 'recycle_fix_fraction', 0.80)
    fix_threshold = getattr(args, 'recycle_fix_threshold', None)
    dirichlet_concentration = getattr(args, 'dirichlet_concentration', 20.0) or 20.0

    num_replicates = x_T.shape[0]
    dirichlet_dist = Dirichlet(dirichlet_concentration * torch.ones(K, device=device))

    def make_hybrid(fix_mask, x_T_rep):
        """Build [N, K] hybrid: fixed positions -> one_hot(argmax(x_T)), others -> Dirichlet."""
        x_T_argmax = x_T_rep.argmax(dim=-1)  # [N] int
        x_new = dirichlet_dist.sample((N,))   # [N, K] fresh noise
        fixed_indices = fix_mask.nonzero(as_tuple=True)[0]
        if fixed_indices.numel() > 0:
            x_new[fixed_indices] = F.one_hot(
                x_T_argmax[fixed_indices], num_classes=K
            ).float()
        return x_new

    # ------------------------------------------------------------------
    # CRITERION: state_max_prob_percentile
    # ------------------------------------------------------------------
    if criterion == 'state_max_prob_percentile':
        x_clipped_list = []
        n_fixed_list = []
        for rep_idx in range(num_replicates):
            max_probs = x_T[rep_idx].max(dim=-1).values  # [N]
            k = int(N * fix_fraction)
            k = max(0, min(k, N))
            topk_indices = max_probs.topk(k).indices      # top-k by max prob
            fix_mask = torch.zeros(N, dtype=torch.bool, device=device)
            fix_mask[topk_indices] = True
            x_clipped_list.append(make_hybrid(fix_mask, x_T[rep_idx]))
            n_fixed_list.append(int(fix_mask.sum().item()))
        return torch.stack(x_clipped_list, dim=0), n_fixed_list

    # ------------------------------------------------------------------
    # CRITERION: state_max_prob_absolute
    # ------------------------------------------------------------------
    elif criterion == 'state_max_prob_absolute':
        x_clipped_list = []
        n_fixed_list = []
        for rep_idx in range(num_replicates):
            max_probs = x_T[rep_idx].max(dim=-1).values  # [N]
            fix_mask = max_probs > fix_threshold           # [N] bool
            x_clipped_list.append(make_hybrid(fix_mask, x_T[rep_idx]))
            n_fixed_list.append(int(fix_mask.sum().item()))
        return torch.stack(x_clipped_list, dim=0), n_fixed_list

    # ------------------------------------------------------------------
    # CRITERION: flip_ever
    # ever_flipped[rep, i] is True if argmax(p_hat) changed at any step
    # ------------------------------------------------------------------
    elif criterion == 'flip_ever':
        if ever_flipped is None:
            raise ValueError(
                "apply_recycle_clipping: flip_ever criterion requires ever_flipped tensor "
                "to be passed. It must have been accumulated during the round-1 step loop."
            )
        x_clipped_list = []
        n_fixed_list = []
        for rep_idx in range(num_replicates):
            never_flipped = ~ever_flipped[rep_idx]  # [N] bool
            k_max = int(N * fix_fraction)
            k_max = max(0, min(k_max, N))
            n_never = int(never_flipped.sum().item())

            if n_never <= k_max:
                # Fix all never-flipped positions (may be less than fix_fraction)
                fix_mask = never_flipped
            else:
                # Too many never-flipped: cap at fix_fraction, break ties by max(x_T)
                max_probs = x_T[rep_idx].max(dim=-1).values.clone()
                max_probs[ever_flipped[rep_idx]] = -1.0  # exclude flipped from selection
                topk_indices = max_probs.topk(k_max).indices
                fix_mask = torch.zeros(N, dtype=torch.bool, device=device)
                fix_mask[topk_indices] = True

            x_clipped_list.append(make_hybrid(fix_mask, x_T[rep_idx]))
            n_fixed_list.append(int(fix_mask.sum().item()))
        return torch.stack(x_clipped_list, dim=0), n_fixed_list

    # ------------------------------------------------------------------
    # CRITERION: aux_consistency
    # Binary: argmax(elec_head) == ELECTROSTATIC_LABEL[argmax(x_T)]
    #     AND argmax(geom_head) == GEOM_TOPOLOGY_LABEL[argmax(x_T)]
    # Both heads must agree. X residues (label == -1) always re-randomized.
    #
    # AA index ordering (canonical, from data/cath_dataset.py AA_TO_IDX):
    #   ALA=0, CYS=1, ASP=2, GLU=3, PHE=4, GLY=5, HIS=6, ILE=7,
    #   LYS=8, LEU=9, MET=10, ASN=11, PRO=12, GLN=13, ARG=14, SER=15,
    #   THR=16, VAL=17, TRP=18, TYR=19, XXX=20
    # ELECTROSTATIC_LABEL and GEOM_TOPOLOGY_LABEL in data/aa_constants.py
    # are indexed by this same order (see aa_constants.py header comment).
    # model._electrostatic_label and model._geom_topology_label are the
    # same tensors registered as buffers. argmax(x_T[i]) returns an index
    # in [0, 20] in this same ordering -> direct lookup is correct.
    # ------------------------------------------------------------------
    elif criterion == 'aux_consistency':
        if model is None or batched_data is None:
            raise ValueError(
                "apply_recycle_clipping: aux_consistency criterion requires model and "
                "batched_data to be passed."
            )
        if not (getattr(model, 'use_electrostatic_loss', False) and
                getattr(model, 'use_geom_topology_loss', False)):
            raise ValueError(
                "apply_recycle_clipping: aux_consistency criterion requires the model to "
                "have been trained with --use_electrostatic_loss and --use_geom_topology_loss."
            )

        # Label lookup tensors (indexed by AA index 0-20, same as x_T argmax)
        elec_label = ELECTROSTATIC_LABEL.to(device)  # [21], values in {0..4, -1}
        geom_label = GEOM_TOPOLOGY_LABEL.to(device)  # [21], values in {0..4, -1}

        t_final = torch.full((1,), T, device=device)

        x_clipped_list = []
        n_fixed_list = []
        model.eval()
        with torch.no_grad():
            for rep_idx in range(num_replicates):
                x_rep = x_T[rep_idx:rep_idx + 1]  # [1, N, K]
                model_output = model(batched_data, t_final, x_rep)

                if not isinstance(model_output, dict):
                    raise ValueError(
                        "apply_recycle_clipping: aux_consistency requires model to return "
                        "dict output with 'electrostatic' and 'geom_topology' keys."
                    )

                elec_logits = model_output.get('electrostatic')  # [total_nodes, 5]
                geom_logits = model_output.get('geom_topology')  # [total_nodes, 5]

                if elec_logits is None or geom_logits is None:
                    raise ValueError(
                        "apply_recycle_clipping: model output missing 'electrostatic' or "
                        "'geom_topology' keys."
                    )

                # Slice to real nodes only (no virtual node handling needed per design)
                elec_logits_real = elec_logits[:N]  # [N, 5]
                geom_logits_real = geom_logits[:N]  # [N, 5]

                # Get provisional AA from current state argmax
                x_T_argmax = x_T[rep_idx].argmax(dim=-1)  # [N] int in [0, 20]

                # Look up deterministic classes for the committed AA
                elec_true = elec_label[x_T_argmax]  # [N] int in {0..4, -1}
                geom_true = geom_label[x_T_argmax]  # [N] int in {0..4, -1}

                # Predicted classes from aux heads
                elec_pred = elec_logits_real.argmax(dim=-1)  # [N] int in {0..4}
                geom_pred = geom_logits_real.argmax(dim=-1)  # [N] int in {0..4}

                # Binary agreement: both heads must agree AND AA must not be X (label != -1)
                is_X = (x_T_argmax == 20)  # [N] bool — X residues always re-randomized
                elec_agrees = (elec_pred == elec_true)   # [N] bool
                geom_agrees = (geom_pred == geom_true)   # [N] bool
                fix_mask = elec_agrees & geom_agrees & ~is_X  # [N] bool

                x_clipped_list.append(make_hybrid(fix_mask, x_T[rep_idx]))
                n_fixed_list.append(int(fix_mask.sum().item()))

        return torch.stack(x_clipped_list, dim=0), n_fixed_list

    # ------------------------------------------------------------------
    # CRITERION: replicate_consensus
    # All replicates must agree on argmax(x_T) for a position to be fixed.
    # Returns a single [1, N, K] state for round 2.
    # ------------------------------------------------------------------
    elif criterion == 'replicate_consensus':
        if num_replicates < 2:
            raise ValueError(
                "apply_recycle_clipping: replicate_consensus requires at least 2 replicates."
            )

        # Decode each replicate's committed sequence
        decoded = x_T.argmax(dim=-1)  # [num_replicates, N] int

        # All-agree mask: max == min across replicate dimension at each position
        all_agree = (decoded.max(dim=0).values == decoded.min(dim=0).values)  # [N] bool

        # Consensus AA: use rep 0 (all agree so any rep works)
        consensus_aa = decoded[0]  # [N] int

        # Build single round-2 starting state
        x_new = dirichlet_dist.sample((N,))   # [N, K] fresh noise everywhere
        fixed_indices = all_agree.nonzero(as_tuple=True)[0]
        if fixed_indices.numel() > 0:
            x_new[fixed_indices] = F.one_hot(
                consensus_aa[fixed_indices], num_classes=K
            ).float()

        n_fixed = int(all_agree.sum().item())
        # Return [1, N, K] — single round-2 trajectory from consensus state
        return x_new.unsqueeze(0), [n_fixed]

    else:
        raise ValueError(f"apply_recycle_clipping: unknown criterion '{criterion}'.")


def select_best_sample_aux_heads(replicate_results, model, data, args, device, T=8.0):
    """
    Select the best replicate by auxiliary head self-consistency scoring.

    For each replicate, runs one forward pass at t=T using the replicate's final x_T
    and computes how well the sampled sequence agrees with the electrostatic and
    geometry/topology head predictions from structure. Lower score = better agreement.

    Ranking score = w_elec * CE(elec_logits, elec_label) + w_geom * CE(geom_logits, geom_label)
    where labels are the deterministic lookup of the sampled amino acid identity.

    Args:
        replicate_results: List of result dicts from sample_chain_with_replicates().
                           Each must contain 'final_probabilities' [N, K] and
                           'predicted_indices' list of int.
        model: The trained DFM model (must have been trained with electrostatic and
               geometry/topology heads; raises ValueError otherwise).
        data: The graph data object for this protein (un-batched).
        args: Arguments object. Must have aux_ranking_weight_electrostatic and
              aux_ranking_weight_geometry set.
        device: torch.device
        T: Maximum time value (used for the final forward pass at t=T).

    Returns:
        Best result dictionary (the element of replicate_results with the lowest
        aux-head score), with added fields:
            'selected_reason': str
            'all_aux_scores': list of float, one per replicate
    """
    import torch
    import torch.nn.functional as F
    from training.collate import collate_fn

    # Check model has the required heads
    if not (getattr(model, 'use_electrostatic_loss', False) and
            getattr(model, 'use_geom_topology_loss', False)):
        raise ValueError(
            "--aux_head_ranking requested but the model was not trained with both "
            "--use_electrostatic_loss and --use_geom_topology_loss. "
            "Check your model checkpoint."
        )

    # Silently pass through if only one replicate
    if len(replicate_results) == 1:
        best = replicate_results[0].copy()
        best['selected_reason'] = "Single replicate, no ranking performed"
        best['all_aux_scores'] = [0.0]
        return best

    w_elec = args.aux_ranking_weight_electrostatic
    w_geom = args.aux_ranking_weight_geometry

    # Retrieve deterministic label lookup tables from model buffers
    elec_label = model._electrostatic_label   # [K] int tensor
    geom_label = model._geom_topology_label   # [K] int tensor

    # Prepare batched graph structure (shared across all replicates)
    K = replicate_results[0]['final_probabilities'].shape[-1]
    dummy_y = torch.zeros(1, K)
    dummy_mask = torch.ones(1, dtype=torch.bool)
    dummy_time = torch.tensor(0.0)
    batched_data, _, _, _ = collate_fn([(data, dummy_y, dummy_mask, dummy_time)])
    batched_data = batched_data.to(device)

    t_final = torch.full((1,), T, device=device)

    use_virtual_node = getattr(data, 'use_virtual_node', False)
    total_nodes = data.num_nodes if hasattr(data, 'num_nodes') else data.x_s.size(0)
    N = (total_nodes - 1) if use_virtual_node else total_nodes

    scores = []
    model.eval()
    with torch.no_grad():
        for r in replicate_results:
            # Reconstruct x_T for this replicate as a [1, N, K] tensor
            probs = r['final_probabilities']  # numpy [N, K] or tensor
            if not isinstance(probs, torch.Tensor):
                probs = torch.tensor(probs, dtype=torch.float32)
            probs = probs.to(device)
            x_rep = probs.unsqueeze(0)  # [1, N, K]

            # Forward pass at t=T
            model_output = model(batched_data, t_final, x_rep)

            if not isinstance(model_output, dict):
                raise ValueError(
                    "Model did not return a dict output. Cannot extract auxiliary heads. "
                    "Ensure the model was loaded with aux heads enabled."
                )

            elec_logits = model_output.get('electrostatic')  # [total_nodes, 5]
            geom_logits = model_output.get('geom_topology')  # [total_nodes, 5]

            if elec_logits is None or geom_logits is None:
                raise ValueError(
                    "--aux_head_ranking: model output dict missing 'electrostatic' or 'geom_topology' keys. "
                    "Model may not have been trained with those heads."
                )

            # Slice to real nodes only (exclude virtual node)
            elec_logits_real = elec_logits[:N]  # [N, 5]
            geom_logits_real = geom_logits[:N]  # [N, 5]

            # Get the sampled amino acid indices for this replicate
            pred_indices = torch.tensor(r['predicted_indices'], dtype=torch.long, device=device)  # [N]

            # Look up deterministic property labels for the sampled amino acids
            elec_targets = elec_label[pred_indices]  # [N], values in {0..4, -1}
            geom_targets = geom_label[pred_indices]  # [N], values in {0..4, -1}

            # Mask out unknown/virtual residues (label == -1, e.g. X residues)
            elec_valid = elec_targets >= 0
            geom_valid = geom_targets >= 0

            # Compute cross-entropy losses (same as training)
            ce_elec = 0.0
            if elec_valid.any():
                ce_elec = F.cross_entropy(
                    elec_logits_real[elec_valid],
                    elec_targets[elec_valid],
                    reduction='mean'
                ).item()

            ce_geom = 0.0
            if geom_valid.any():
                ce_geom = F.cross_entropy(
                    geom_logits_real[geom_valid],
                    geom_targets[geom_valid],
                    reduction='mean'
                ).item()

            score = w_elec * ce_elec + w_geom * ce_geom
            scores.append(score)

    # Pick the replicate with the lowest (best) score
    best_idx = int(torch.tensor(scores).argmin().item())
    best_result = replicate_results[best_idx].copy()
    best_result['selected_reason'] = (
        f"Aux head ranking: score={scores[best_idx]:.4f} "
        f"(w_elec={w_elec}, w_geom={w_geom})"
    )
    best_result['all_aux_scores'] = scores

    return best_result


class DistributedSamplingDataset(Dataset):
    """
    Dataset wrapper for distributed sampling that provides index-based access
    to structures for sampling across multiple GPUs.
    """

    def __init__(self, base_dataset, indices=None):
        """
        Initialize distributed sampling dataset.

        Args:
            base_dataset: Base dataset (e.g., CathDataset)
            indices: List of indices to include (None for all)
        """
        self.base_dataset = base_dataset
        self.indices = indices if indices is not None else list(range(len(base_dataset)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        return self.base_dataset[actual_idx], actual_idx


def ensure_pdb_format(structure_path, verbose=False):
    """
    Return a path to a PDB-format copy of ``structure_path``.

    Structures from the PDB, AlphaFold DB, and most modern tooling are
    distributed as mmCIF, while the downstream parsing and DSSP steps expect
    PDB. Converting is mechanical, so do it here rather than asking the user
    to run a separate command and get the argument order right.

    Files that are already PDB are returned unchanged. Converted files are
    written next to the original when that directory is writable, and to a
    temporary directory otherwise, so repeat runs do not re-convert.

    Returns:
        Path to a PDB file (str).
    """
    import tempfile
    from pathlib import Path

    source = Path(structure_path)
    if source.suffix.lower() not in ('.cif', '.mmcif'):
        return str(source)

    if not source.exists():
        raise FileNotFoundError(
            f"Structure file not found: {source}\n"
            f"Check the path and try again."
        )

    converted = source.with_suffix('.pdb')
    if not os.access(source.parent, os.W_OK):
        converted = Path(tempfile.gettempdir()) / f"{source.stem}.pdb"

    # Reuse an earlier conversion if it is newer than the source.
    if converted.exists() and converted.stat().st_mtime >= source.stat().st_mtime:
        if verbose:
            print(f"[cif] Using existing conversion: {converted}")
        return str(converted)

    if verbose:
        print(f"[cif] Converting mmCIF to PDB: {source.name} -> {converted.name}")

    try:
        from Bio.PDB import MMCIFParser, PDBIO, Select

        class _ProteinOnly(Select):
            """Drop waters, ligands, and other heteroatoms."""

            def accept_residue(self, residue):
                return residue.get_id()[0] == ' '

        parser = MMCIFParser(QUIET=not verbose)
        structure = parser.get_structure(source.stem, str(source))

        writer = PDBIO()
        writer.set_structure(structure)
        writer.save(str(converted), select=_ProteinOnly())
    except Exception as exc:
        raise RuntimeError(
            f"Could not convert {source.name} from mmCIF to PDB: {exc}\n\n"
            f"This usually means the file is malformed or is not actually "
            f"mmCIF. You can convert it manually with:\n"
            f"  python helpers/convert_cif_to_pdb.py {source} output.pdb\n"
            f"and then pass output.pdb instead."
        )

    if not converted.exists() or converted.stat().st_size == 0:
        raise RuntimeError(
            f"Conversion of {source.name} produced an empty file. "
            f"The mmCIF may contain no standard protein residues."
        )

    if verbose:
        n_atoms = sum(1 for line in open(converted) if line.startswith('ATOM'))
        print(f"[cif] Converted successfully ({n_atoms} atom records)")

    return str(converted)


class _GraphBuilderOnlyDataset:
    """
    Minimal stand-in for CathDataset when the CATH reference files are absent.

    Designing from a user-supplied structure (--pdb_input) needs only the
    GraphBuilder, which is configured entirely from checkpoint parameters.
    The reference split/map files exist to look structures up by UniProt or
    PDB ID, so they are required only for those input modes. This class
    provides the builder and raises a clear, actionable error if anything
    tries to use the lookup tables.
    """

    def __init__(self, graph_builder_kwargs, missing_split_json=None,
                 missing_map_pkl=None):
        from data.graph_builder import GraphBuilder

        self.graph_builder = GraphBuilder(**graph_builder_kwargs)
        self._missing_split_json = missing_split_json
        self._missing_map_pkl = missing_map_pkl
        self.map_data = None
        self.file_paths = []

    def _unavailable(self):
        return RuntimeError(
            "This input mode needs the CATH reference dataset, which was not "
            "found.\n"
            f"  expected splits : {self._missing_split_json}\n"
            f"  expected map    : {self._missing_map_pkl}\n\n"
            "Either:\n"
            "  - design from a structure file instead, with "
            "--pdb_input yourfile.pdb (no reference data needed), or\n"
            "  - download the CATH files and pass --split_json / --map_pkl.\n"
            "See docs/GETTING_STARTED.md."
        )

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        raise self._unavailable()


class SamplingCoordinator:
    """
    Coordinates distributed sampling across multiple GPUs with proper cleanup.
    """

    def __init__(self, model_path: str, dataset_path: str, split: str = 'validation'):
        """
        Initialize sampling coordinator.

        Args:
            model_path: Path to trained model
            dataset_path: Path to dataset
            split: Dataset split to use
        """
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.split = split
        self.is_distributed = False
        self.model = None
        self.dataset = None

    def setup_distributed(self, rank: int, world_size: int, master_port: int = 29500):
        """Setup distributed training environment."""
        import torch.distributed as dist

        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(master_port)

        dist.init_process_group(
            backend='nccl',
            rank=rank,
            world_size=world_size
        )

        torch.cuda.set_device(rank)
        self.is_distributed = True

    def cleanup_distributed(self):
        """Cleanup distributed environment."""
        if self.is_distributed:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()
            self.is_distributed = False

    def load_model_and_dataset(self, device: torch.device, args=None):
        """Load model and dataset on specified device."""
        # Load model and extract dataset parameters
        self.model, dataset_params = load_model_distributed(self.model_path, device, args)

        # Store dataset_params for access by other methods
        self.dataset_params = dataset_params

        # Use command-line overrides if provided, otherwise use dataset parameters from checkpoint
        split_json = (args.split_json if args and args.split_json else dataset_params.get('split_json')) or '../datasets/cath-4.2/chain_set_splits.json'
        map_pkl = (args.map_pkl if args and args.map_pkl else dataset_params.get('map_pkl')) or '../datasets/cath-4.2/chain_set_map_with_b_factors_dssp.pkl'
        use_virtual_node = dataset_params.get('use_virtual_node', True)

        # Extract RBF parameters from dataset_params (should always be available now)
        rbf_3d_min = dataset_params.get('rbf_3d_min')
        rbf_3d_max = dataset_params.get('rbf_3d_max')
        rbf_3d_spacing = dataset_params.get('rbf_3d_spacing')

        if rbf_3d_min is None or rbf_3d_max is None or rbf_3d_spacing is None:
            raise RuntimeError(
                f"Internal error: RBF parameters should have been resolved in load_model_distributed.\n"
                f"Got: rbf_3d_min={rbf_3d_min}, rbf_3d_max={rbf_3d_max}, rbf_3d_spacing={rbf_3d_spacing}"
            )

        print(f"Using dataset parameters:")
        print(f"  split_json: {split_json}" + (" (command-line override)" if args and args.split_json else " (from checkpoint)" if dataset_params.get('split_json') else " (default)"))
        print(f"  map_pkl: {map_pkl}" + (" (command-line override)" if args and args.map_pkl else " (from checkpoint)" if dataset_params.get('map_pkl') else " (default)"))
        print(f"  use_virtual_node: {use_virtual_node}")
        print(f"  rbf_3d_min: {rbf_3d_min}")
        print(f"  rbf_3d_max: {rbf_3d_max}")
        print(f"  rbf_3d_spacing: {rbf_3d_spacing}")

        # Prepare graph builder parameters
        graph_builder_kwargs = {
            'k': dataset_params.get('k_neighbors'),
            'k_farthest': dataset_params.get('k_farthest'),
            'k_random': dataset_params.get('k_random'),
            'max_edge_dist': dataset_params.get('max_edge_dist'),
            'num_rbf_3d': dataset_params.get('num_rbf_3d'),
            'num_rbf_seq': dataset_params.get('num_rbf_seq'),
            'no_source_indicator': dataset_params.get('no_source_indicator'),
            'rbf_3d_min': rbf_3d_min,
            'rbf_3d_max': rbf_3d_max,
            'rbf_3d_spacing': rbf_3d_spacing,
            'include_node_degree': dataset_params.get('include_node_degree', False),
            'blur_uncertainty': dataset_params.get('blur_uncertainty', False),
        }

        # Special handling for GraphBuilder parameter validation:
        # When max_edge_dist is specified, k/k_farthest/k_random should be explicitly None
        # When max_edge_dist is None, k/k_farthest/k_random must be provided
        if dataset_params.get('max_edge_dist') is not None:
            # Distance-based mode: ensure k parameters are explicitly None
            graph_builder_kwargs.update({
                'k': None,
                'k_farthest': None,
                'k_random': None,
            })

        # Remove None values to use GraphBuilder defaults (except for distance-based mode)
        if dataset_params.get('max_edge_dist') is not None:
            # In distance-based mode, keep None values for k parameters
            graph_builder_kwargs = {k: v for k, v in graph_builder_kwargs.items()
                                  if v is not None or k in ['k', 'k_farthest', 'k_random']}
        else:
            # In k-neighbor mode, remove None values
            graph_builder_kwargs = {k: v for k, v in graph_builder_kwargs.items() if v is not None}

        # Load dataset.
        #
        # When the user supplies their own structure (--pdb_input), the only
        # thing needed from the dataset is its GraphBuilder, which is built
        # purely from checkpoint parameters. The CATH reference files are a
        # ~270 MB download used to look structures up by UniProt/PDB ID, so
        # requiring them for a local file would be a large and pointless
        # barrier. Fall back to a builder-only stand-in when they are absent.
        from data.cath_dataset import CathDataset

        # Check usability, not just presence. A Git LFS pointer is a small text
        # file that exists but is not a pickle, and a truncated download looks
        # the same, so size-check the map before trying to load 270 MB.
        reference_data_available = False
        if split_json and map_pkl:
            try:
                reference_data_available = (
                    os.path.exists(split_json)
                    and os.path.exists(map_pkl)
                    and os.path.getsize(map_pkl) > 1_000_000
                )
            except OSError:
                reference_data_available = False

        if reference_data_available:
            self.dataset = CathDataset(
                split_json=split_json,
                map_pkl=map_pkl,
                split=self.split,
                graph_builder_kwargs=graph_builder_kwargs,
                # Add required time sampling parameters with defaults for sampling
                time_sampling_strategy='uniform',
                t_min=0.0,
                t_max=8.0,
                alpha_range=1.0
            )
        else:
            self.dataset = _GraphBuilderOnlyDataset(
                graph_builder_kwargs,
                missing_split_json=split_json,
                missing_map_pkl=map_pkl,
            )

        return self.model, self.dataset

    def sample_structures(self, indices: List[int], device: torch.device, args=None) -> List[Dict]:
        """Sample sequences for given structure indices."""
        if self.model is None or self.dataset is None:
            raise RuntimeError("Model and dataset must be loaded first")

        from .sample import sample_multiple_proteins

        return sample_multiple_proteins(
            self.model,
            self.dataset,
            indices=indices,
            args=args
        )


def setup_distributed_sampling(device='auto'):
    """
    Setup distributed sampling environment.

    Args:
        device: Device specification ('auto', 'cuda', 'cpu', or specific device)

    Returns:
        Tuple of (device, world_size, rank)
    """
    import torch.distributed as dist

    # Determine device
    if device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    elif isinstance(device, str):
        device = torch.device(device)

    # Check if distributed is available and initialized
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        if not dist.is_available():
            print("Warning: Distributed training not available, falling back to single GPU")
            return device, 1, 0

        if not dist.is_initialized():
            print("Warning: Distributed not initialized, falling back to single GPU")
            return device, 1, 0

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        print(f"Distributed sampling: rank {rank}/{world_size} on device {device}")
        return device, world_size, rank
    else:
        print(f"Single device sampling on {device}")
        return device, 1, 0


def cleanup_distributed_sampling():
    """Cleanup distributed sampling environment."""
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _download_checkpoint_from_url(url: str, cache_dir: str = None) -> str:
    """
    Download a checkpoint from a URL with caching support.
    Uses urllib (built-in), falling back to requests.

    Args:
        url: The HTTP(S) URL to download from
        cache_dir: Directory to cache downloaded checkpoints

    Returns:
        str: Path to the downloaded checkpoint file
    """
    # tempfile.gettempdir() rather than a hardcoded /tmp: this project
    # supports Windows, where /tmp does not exist.
    if cache_dir is None:
        import tempfile
        cache_dir = os.path.join(tempfile.gettempdir(), "inversefolddir_checkpoints")
    import hashlib
    import os
    import subprocess
    from pathlib import Path
    from urllib.parse import urlparse

    # Create cache directory
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Generate cache filename from URL hash
    url_hash = hashlib.md5(url.encode()).hexdigest()
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path) or "checkpoint.pt"
    cache_filename = f"{url_hash}_{filename}"
    cached_file_path = cache_path / cache_filename

    # Return cached file if it exists
    if cached_file_path.exists():
        print(f"Using cached checkpoint: {cached_file_path}")
        return str(cached_file_path)

    print(f"Downloading checkpoint from: {url}")
    print(f"Caching to: {cached_file_path}")

    # Method 1: urllib (built-in Python, no extra dependencies)
    try:
        import urllib.error
        import urllib.request

        print("Attempting download with urllib (built-in)...")

        # Download with progress reporting for large files
        def download_progress(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, (block_num * block_size * 100) / total_size)
                if block_num % 1000 == 0:  # Print every ~8MB for typical block sizes
                    print(f"Download progress: {percent:.1f}%")

        urllib.request.urlretrieve(url, cached_file_path, reporthook=download_progress)
        print(f"Successfully downloaded checkpoint using urllib: {cached_file_path}")
        return str(cached_file_path)

    except urllib.error.URLError as e:
        print(f"urllib download failed: {e}")
    except Exception as e:
        print(f"urllib download failed with unexpected error: {e}")

    # Method 2: requests as final fallback (if available)
    try:
        import requests
        print("Attempting download with requests (fallback)...")

        # Download with streaming to handle large files
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        # Get file size if available
        file_size = int(response.headers.get('content-length', 0))
        if file_size > 0:
            print(f"Download size: {file_size / (1024*1024):.1f} MB")

        # Download and save
        with open(cached_file_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if file_size > 0 and downloaded % (1024*1024*10) == 0:  # Progress every 10MB
                        progress = (downloaded / file_size) * 100
                        print(f"Download progress: {progress:.1f}%")

        print(f"Successfully downloaded checkpoint using requests: {cached_file_path}")
        return str(cached_file_path)

    except ImportError:
        print("requests library not available")
    except Exception as e:
        print(f"requests download failed: {e}")

    # Clean up partial download and raise error
    if cached_file_path.exists():
        cached_file_path.unlink()

    raise RuntimeError(f"Failed to download checkpoint from {url} using all available methods (urllib, requests)")


def _auto_discover_best_model(original_path: str) -> str:
    """
    Auto-discover local 'best' model files in the working directory.

    Searches in multiple locations for files containing 'best' and ending with '.pt':
    1. Current working directory
    2. Parent directory of the original path
    3. Common model directories (./saved_models, ./output/saved_models, etc.)

    Args:
        original_path: The original path that wasn't found

    Returns:
        str: Path to the discovered model file, or None if not found
    """
    import glob
    import os
    from pathlib import Path

    print(f"Auto-discovering local 'best' model files (original path not found: {original_path})")

    # Get search directories
    search_dirs = []

    # 1. Current working directory
    search_dirs.append(os.getcwd())

    # 2. Parent directory of original path (if it has one)
    if original_path and os.path.dirname(original_path):
        search_dirs.append(os.path.dirname(original_path))

    # 3. Common model directory patterns
    common_dirs = [
        './saved_models',
        './output/saved_models',
        '../output/saved_models',
        './models',
        './checkpoints',
        '.'  # Current directory as fallback
    ]
    search_dirs.extend(common_dirs)

    # Remove duplicates and ensure directories exist
    search_dirs = [d for d in set(search_dirs) if os.path.isdir(d)]

    print(f"Searching in directories: {search_dirs}")

    best_models = []

    # Search for files containing 'best' and ending with '.pt'
    for search_dir in search_dirs:
        pattern = os.path.join(search_dir, '*best*.pt')
        matches = glob.glob(pattern)
        if matches:
            best_models.extend(matches)
            print(f"Found {len(matches)} 'best' models in {search_dir}: {[os.path.basename(m) for m in matches]}")

    if not best_models:
        print("No local 'best' model files found")
        return None

    # Remove duplicates and sort by modification time (newest first)
    best_models = list(set(best_models))
    best_models.sort(key=lambda x: os.path.getmtime(x), reverse=True)

    selected_model = best_models[0]
    print(f"Auto-selected newest 'best' model: {selected_model}")

    if len(best_models) > 1:
        print(f"Note: Found {len(best_models)} 'best' models, using newest: {os.path.basename(selected_model)}")
        print("Other candidates:")
        for model in best_models[1:]:
            mtime = os.path.getmtime(model)
            print(f"  - {os.path.basename(model)} (modified: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))})")

    return selected_model


# Refuse to build a c_factor table larger than this many alpha rows. Each row is
# a 1000-point incomplete-beta evaluation, so the build is linear in row count;
# 200k rows takes on the order of a minute. Anything past that is a sign of an
# alpha_spacing/horizon combination that will not finish, and a clear error beats
# a job that appears to hang at model load.
MAX_ALPHA_GRID_ROWS = 200_000


def sampling_horizon(args, default=None):
    """
    Return the flow horizon a sampling run will actually integrate to.

    The two entry points spell this differently -- ``sample.py`` takes ``--T``
    and ``inpainting.py`` takes ``--t_max`` -- and neither is stored in the
    checkpoint, so this is the one place that reconciles them.
    """
    for name in ('T', 't_max'):
        value = getattr(args, name, None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return default


def _alpha_max_for_horizon(dfm_kwargs, args, trained_t_max=None):
    """
    Widen the checkpoint's ``alpha_max`` so the c_factor grid covers the horizon.

    Returns the alpha_max to build the grid with. Unchanged when the checkpoint
    already covers the requested horizon, which is the common case.
    """
    stored_alpha_max = float(dfm_kwargs['alpha_max'])
    horizon = sampling_horizon(args)
    if horizon is None:
        return stored_alpha_max

    # alpha = 1 + t, plus one spacing of headroom for float accumulation in t.
    spacing = float(dfm_kwargs['alpha_spacing'])
    needed = 1.0 + horizon + spacing
    if needed <= stored_alpha_max:
        return stored_alpha_max

    rows = int((needed - float(dfm_kwargs['alpha_min'])) / spacing) + 1
    if rows > MAX_ALPHA_GRID_ROWS:
        raise ValueError(
            f"Sampling at horizon T={horizon:g} needs a Dirichlet c_factor grid of "
            f"~{rows:,} rows (alpha up to {needed:.2f} at spacing {spacing:g}), past the "
            f"{MAX_ALPHA_GRID_ROWS:,}-row limit. Building it would take longer than the "
            "sampling run.\nLower the horizon, or sample without the c_factor "
            "(omit --use_c_factor), which does not use this grid."
        )

    print(
        f"\nExtending the Dirichlet c_factor grid: alpha_max "
        f"{stored_alpha_max:g} -> {needed:.2f} to cover the requested horizon "
        f"T={horizon:g} (alpha = 1 + t). {rows:,} rows; this adds a few seconds "
        "to model load."
    )
    if trained_t_max is not None and horizon > float(trained_t_max):
        print(
            f"  NOTE: this checkpoint was trained at t_max={float(trained_t_max):g}. "
            f"Sampling at T={horizon:g} is extrapolation beyond the trained horizon; "
            "it is supported, but validate the output. See docs/GETTING_STARTED.md."
        )
    return needed


def load_model_distributed(model_path: str, device: torch.device, args):
    """
    Load a trained model for distributed sampling and extract dataset parameters.
    Supports both local file paths and HTTP(S) URLs.

    Args:
        model_path: Path to the model checkpoint (local path or HTTP(S) URL)
        device: Device to load the model on
        args: Arguments object with model configuration

    Returns:
        Tuple of (model, dataset_params) where dataset_params contains
        the dataset configuration extracted from the checkpoint
    """
    # Dynamic import to avoid circular dependencies
    import os
    import sys

    # Checkpoint introspection is diagnostic, not something a user designing a
    # sequence needs to read. It ran to ~400 lines per invocation. Keep it, but
    # behind --verbose.
    verbose = getattr(args, 'verbose', False)

    def vprint(*a, **kw):
        if verbose:
            print(*a, **kw)

    # Add the parent directory to the Python path if not already there
    parent_dir = os.path.join(os.path.dirname(__file__), '..')
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)

    from models.dfm_model import DFMNodeClassifier

    # Prove the device computes before anything is loaded onto it. A GPU whose
    # architecture the installed PyTorch has no kernels for can return zeros
    # rather than raising, which surfaces much later as a nonsense sequence or an
    # unrelated-looking Dirichlet error. See device_check.py.
    if getattr(device, 'type', str(device)) == 'cuda':
        from device_check import assert_cuda_device_usable
        assert_cuda_device_usable(getattr(device, 'index', None) or 0,
                                  verbose=getattr(args, 'verbose', False))

    print(f"Loading model from: {model_path}")

    # Handle remote URLs
    local_model_path = model_path
    if model_path.startswith('http://') or model_path.startswith('https://'):
        local_model_path = _download_checkpoint_from_url(model_path)
    elif not os.path.exists(model_path):
        # Try to auto-discover local 'best' model files
        auto_discovered_path = _auto_discover_best_model(model_path)
        if auto_discovered_path:
            print(f"Auto-discovered local best model: {auto_discovered_path}")
            local_model_path = auto_discovered_path
        else:
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    # Load checkpoint.
    #
    # PyTorch 2.6 flipped torch.load's `weights_only` default to True, which
    # rejects checkpoints containing pickled numpy objects and argparse
    # Namespaces -- exactly what these checkpoints store alongside the weights.
    # Without this, every user on a current PyTorch would fail at the first
    # design. Pass weights_only=False where the argument exists, and fall back
    # for PyTorch versions predating it.
    try:
        checkpoint = torch.load(local_model_path, map_location=device,
                                weights_only=False)
    except TypeError:
        # PyTorch < 1.13 has no weights_only argument.
        checkpoint = torch.load(local_model_path, map_location=device)

    # Extract model arguments
    if 'args' in checkpoint:
        model_args = checkpoint['args']
        vprint("="*60)
        vprint("CHECKPOINT PARAMETER EXTRACTION")
        vprint("="*60)
        vprint("Found 'args' in checkpoint - extracting model configuration")

        # Determine if model_args is a dict or an object
        is_dict = isinstance(model_args, dict)
        vprint(f"Args type: {type(model_args).__name__}")

        # Helper function to get values from either dict or object
        def get_arg_value(key, default=None):
            if is_dict:
                return model_args.get(key, default)
            else:
                return getattr(model_args, key, default)

        # Show all available parameters
        vprint("\nAll available parameters in checkpoint args:")
        if is_dict:
            for key in sorted(model_args.keys()):
                value = model_args[key]
                vprint(f"  {key}: {value}")
        elif hasattr(model_args, '__dict__'):
            for attr_name in sorted(model_args.__dict__.keys()):
                attr_value = getattr(model_args, attr_name)
                vprint(f"  {attr_name}: {attr_value}")
        else:
            print("  Warning: model_args has no accessible attributes")

        vprint("\nKey model architecture parameters:")
        vprint(f"  num_layers_gvp: {get_arg_value('num_layers_gvp', get_arg_value('num_layers', 'NOT FOUND'))}")  # Fallback for old checkpoints
        vprint(f"  num_layers_prediction: {get_arg_value('num_layers_prediction', 'NOT FOUND')}")
        vprint(f"  hidden_dim: {get_arg_value('hidden_dim', 'NOT FOUND')}")
        vprint(f"  use_qkv: {get_arg_value('use_qkv', 'NOT FOUND')}")
        vprint(f"  time_dim: {get_arg_value('time_dim', 'NOT FOUND')}")
        vprint(f"  use_virtual_node: {get_arg_value('use_virtual_node', 'NOT FOUND')}")
        vprint(f"  node_dims: {(get_arg_value('node_dim_s'), get_arg_value('node_dim_v'))}")
        vprint(f"  edge_dims: {(get_arg_value('edge_dim_s'), get_arg_value('edge_dim_v'))}")
        vprint(f"  hidden_dims: {(get_arg_value('hidden_dim'), get_arg_value('hidden_dim_v'))}")
        vprint(f"  dropout: {get_arg_value('dropout', 'NOT FOUND')}")

        vprint("\nDataset-related parameters:")

        # Check for dedicated graph_builder_params section first (new format)
        graph_builder_params = checkpoint.get('graph_builder_params', {})
        if graph_builder_params:
            vprint("Using dataset parameters from dedicated graph_builder_params section")
            split_json_cp = graph_builder_params.get('split_json') or get_arg_value('split_json')
            map_pkl_cp = graph_builder_params.get('map_pkl') or get_arg_value('map_pkl')
            use_virtual_cp = graph_builder_params.get('use_virtual_node')
            if use_virtual_cp is None:
                use_virtual_cp = get_arg_value('use_virtual_node')
        else:
            vprint("Using dataset parameters from args section")
            split_json_cp = get_arg_value('split_json')
            map_pkl_cp = get_arg_value('map_pkl')
            use_virtual_cp = get_arg_value('use_virtual_node')

        max_length_cp = get_arg_value('max_length')
        use_graph_builder_cp = get_arg_value('use_graph_builder')

        vprint(f"  split_json: {split_json_cp}")
        vprint(f"  map_pkl: {map_pkl_cp}")
        vprint(f"  use_virtual_node: {use_virtual_cp}")
        vprint(f"  max_length: {max_length_cp}")
        vprint(f"  use_graph_builder: {use_graph_builder_cp}")

        vprint("\nGraph building parameters:")

        # Check for dedicated graph_builder_params section first (new format)
        graph_builder_params = checkpoint.get('graph_builder_params', {})
        if graph_builder_params:
            vprint("Found dedicated graph_builder_params section in checkpoint")
            # Handle parameter name mapping (GraphBuilder uses 'k' but training uses 'k_neighbors')
            k_neighbors_cp = graph_builder_params.get('k') or graph_builder_params.get('k_neighbors')
            k_farthest_cp = graph_builder_params.get('k_farthest')
            k_random_cp = graph_builder_params.get('k_random')
            max_edge_dist_cp = graph_builder_params.get('max_edge_dist')
            num_rbf_3d_cp = graph_builder_params.get('num_rbf_3d')
            num_rbf_seq_cp = graph_builder_params.get('num_rbf_seq')
            no_source_indicator_cp = graph_builder_params.get('no_source_indicator')
            # RBF distance range parameters (new)
            rbf_3d_min_cp = graph_builder_params.get('rbf_3d_min')
            rbf_3d_max_cp = graph_builder_params.get('rbf_3d_max')
            rbf_3d_spacing_cp = graph_builder_params.get('rbf_3d_spacing')
            # Feature flags that affect node dimensionality
            include_node_degree_cp = graph_builder_params.get('include_node_degree', False)
            blur_uncertainty_cp = graph_builder_params.get('blur_uncertainty', False)
        else:
            vprint("No dedicated graph_builder_params section found, extracting from args")
            k_neighbors_cp = get_arg_value('k_neighbors')
            k_farthest_cp = get_arg_value('k_farthest')
            k_random_cp = get_arg_value('k_random')
            max_edge_dist_cp = get_arg_value('max_edge_dist')
            num_rbf_3d_cp = get_arg_value('num_rbf_3d')
            num_rbf_seq_cp = get_arg_value('num_rbf_seq')
            no_source_indicator_cp = get_arg_value('no_source_indicator')
            # RBF distance range parameters (fallback to defaults if not in args)
            rbf_3d_min_cp = get_arg_value('rbf_3d_min')
            rbf_3d_max_cp = get_arg_value('rbf_3d_max')
            rbf_3d_spacing_cp = get_arg_value('rbf_3d_spacing')
            # Feature flags (default False for backwards compatibility with old checkpoints)
            include_node_degree_cp = get_arg_value('include_node_degree') or False
            blur_uncertainty_cp = get_arg_value('blur_uncertainty') or False

        # Check for model architecture parameters (new systematic format)
        model_architecture_params = checkpoint.get('model_architecture_params', {})
        if model_architecture_params:
            vprint("Found dedicated model_architecture_params section in checkpoint")
            vprint("Model architecture parameters from checkpoint:")
            for param_name, param_value in model_architecture_params.items():
                vprint(f"  {param_name}: {param_value}")
        else:
            vprint("No dedicated model_architecture_params section found")

        vprint(f"  k_neighbors: {k_neighbors_cp}")
        vprint(f"  k_farthest: {k_farthest_cp}")
        vprint(f"  k_random: {k_random_cp}")
        vprint(f"  max_edge_dist: {max_edge_dist_cp}")
        vprint(f"  num_rbf_3d: {num_rbf_3d_cp}")
        vprint(f"  num_rbf_seq: {num_rbf_seq_cp}")
        vprint(f"  no_source_indicator: {no_source_indicator_cp}")
        vprint(f"  rbf_3d_min: {rbf_3d_min_cp} (default: 2.0 if None)")
        vprint(f"  rbf_3d_max: {rbf_3d_max_cp} (default: 350.0 if None)")
        vprint(f"  rbf_3d_spacing: {rbf_3d_spacing_cp} (default: exponential if None)")
        vprint(f"  include_node_degree: {include_node_degree_cp}")
        vprint(f"  blur_uncertainty: {blur_uncertainty_cp}")

        vprint("\nTraining parameters:")
        vprint(f"  learning_rate (lr): {get_arg_value('lr', 'NOT FOUND')}")
        vprint(f"  batch_size (batch): {get_arg_value('batch', 'NOT FOUND')}")
        vprint(f"  epochs: {get_arg_value('epochs', 'NOT FOUND')}")
        vprint(f"  alpha_min: {get_arg_value('alpha_min', 'NOT FOUND')}")
        vprint(f"  alpha_max: {get_arg_value('alpha_max', 'NOT FOUND')}")

        vprint("\nTime parameters:")
        t_max_cp = get_arg_value('t_max')
        t_min_cp = get_arg_value('t_min')
        vprint(f"  t_max: {t_max_cp} (default: 8.0 if None)")
        vprint(f"  t_min: {t_min_cp} (default: 0.0 if None)")

        # Extract dataset parameters with detailed fallback logic
        vprint("\n" + "="*60)
        vprint("DATASET PARAMETER RESOLUTION")
        vprint("="*60)

        # Always use command line arguments if available, then fall back to checkpoint, then default
        # For split_json
        if hasattr(args, 'split_json') and args.split_json:
            final_split_json = args.split_json
            split_json_source = "command line args"
        elif split_json_cp:
            final_split_json = split_json_cp
            split_json_source = "checkpoint"
        else:
            final_split_json = '../datasets/cath-4.2/chain_set_splits.json'
            split_json_source = "default"

        # For map_pkl
        if hasattr(args, 'map_pkl') and args.map_pkl:
            final_map_pkl = args.map_pkl
            map_pkl_source = "command line args"
        elif map_pkl_cp:
            final_map_pkl = map_pkl_cp
            map_pkl_source = "checkpoint"
        else:
            final_map_pkl = '../datasets/cath-4.2/chain_set_map_with_b_factors_dssp.pkl'
            map_pkl_source = "default"

        # For use_virtual_node
        if use_virtual_cp is not None:
            final_use_virtual = use_virtual_cp
            virtual_source = "checkpoint"
        else:
            final_use_virtual = None  # Will be inferred later
            virtual_source = "will be inferred from filename"

        dataset_params = {
            'split_json': final_split_json,
            'map_pkl': final_map_pkl,
            'use_virtual_node': final_use_virtual,
            'max_length': max_length_cp,
            'use_graph_builder': use_graph_builder_cp if use_graph_builder_cp is not None else True,
            # Graph builder parameters from checkpoint
            'k_neighbors': k_neighbors_cp,
            'k_farthest': k_farthest_cp,
            'k_random': k_random_cp,
            'max_edge_dist': max_edge_dist_cp,
            'num_rbf_3d': num_rbf_3d_cp,
            'num_rbf_seq': num_rbf_seq_cp,
            'no_source_indicator': no_source_indicator_cp,
            # Feature flags that affect node dimensionality
            'include_node_degree': include_node_degree_cp,
            'blur_uncertainty': blur_uncertainty_cp,
        }

        # Handle RBF distance range parameters with backwards compatibility
        # Priority: checkpoint > command line args > error (no defaults for backwards compatibility)
        rbf_3d_min_final = None
        rbf_3d_max_final = None
        rbf_3d_spacing_final = None
        rbf_source = None

        if rbf_3d_min_cp is not None and rbf_3d_max_cp is not None and rbf_3d_spacing_cp is not None:
            # All RBF parameters found in checkpoint
            rbf_3d_min_final = rbf_3d_min_cp
            rbf_3d_max_final = rbf_3d_max_cp
            rbf_3d_spacing_final = rbf_3d_spacing_cp
            rbf_source = "checkpoint"
        elif args and hasattr(args, 'rbf_3d_min') and hasattr(args, 'rbf_3d_max') and hasattr(args, 'rbf_3d_spacing'):
            # Check if command line args are provided
            if args.rbf_3d_min is not None and args.rbf_3d_max is not None and args.rbf_3d_spacing is not None:
                rbf_3d_min_final = args.rbf_3d_min
                rbf_3d_max_final = args.rbf_3d_max
                rbf_3d_spacing_final = args.rbf_3d_spacing
                rbf_source = "command line arguments"
            else:
                # Some command line args are None
                missing_args = []
                if args.rbf_3d_min is None:
                    missing_args.append('--rbf_3d_min')
                if args.rbf_3d_max is None:
                    missing_args.append('--rbf_3d_max')
                if args.rbf_3d_spacing is None:
                    missing_args.append('--rbf_3d_spacing')

                raise ValueError(
                    f"RBF parameters not found in checkpoint and missing from command line.\n"
                    f"For backwards compatibility with older checkpoints, please provide:\n"
                    f"  {' '.join(missing_args)}\n"
                    f"Example: --rbf_3d_min 2.0 --rbf_3d_max 350.0 --rbf_3d_spacing exponential\n"
                    f"Checkpoint RBF params: rbf_3d_min={rbf_3d_min_cp}, rbf_3d_max={rbf_3d_max_cp}, rbf_3d_spacing={rbf_3d_spacing_cp}"
                )
        else:
            # No args object or missing attributes
            raise ValueError(
                f"RBF parameters not found in checkpoint and no command line arguments available.\n"
                f"For backwards compatibility with older checkpoints, please provide:\n"
                f"  --rbf_3d_min <value> --rbf_3d_max <value> --rbf_3d_spacing <spacing_type>\n"
                f"Example: --rbf_3d_min 2.0 --rbf_3d_max 350.0 --rbf_3d_spacing exponential\n"
                f"Checkpoint RBF params: rbf_3d_min={rbf_3d_min_cp}, rbf_3d_max={rbf_3d_max_cp}, rbf_3d_spacing={rbf_3d_spacing_cp}"
            )

        # Add RBF parameters to dataset_params
        dataset_params['rbf_3d_min'] = rbf_3d_min_final
        dataset_params['rbf_3d_max'] = rbf_3d_max_final
        dataset_params['rbf_3d_spacing'] = rbf_3d_spacing_final

        # Add model architecture parameters from checkpoint
        dataset_params['model_architecture_params'] = model_architecture_params

        # Add time parameters to dataset_params
        dataset_params['t_max'] = t_max_cp or 8.0  # Default to 8.0 if not in checkpoint
        dataset_params['t_min'] = t_min_cp or 0.0  # Default to 0.0 if not in checkpoint

        vprint(f"split_json: {final_split_json} (source: {split_json_source})")
        vprint(f"map_pkl: {final_map_pkl} (source: {map_pkl_source})")
        vprint(f"use_virtual_node: {final_use_virtual} (source: {virtual_source})")
        vprint(f"max_length: {max_length_cp} (source: {'checkpoint' if max_length_cp else 'not specified'})")
        vprint(f"use_graph_builder: {dataset_params['use_graph_builder']} (source: {'checkpoint' if use_graph_builder_cp else 'default'})")

        vprint(f"\nGraph builder parameter resolution:")
        vprint(f"k_neighbors: {k_neighbors_cp} (source: {'checkpoint' if k_neighbors_cp is not None else 'not in checkpoint'})")
        vprint(f"k_farthest: {k_farthest_cp} (source: {'checkpoint' if k_farthest_cp is not None else 'not in checkpoint'})")
        vprint(f"k_random: {k_random_cp} (source: {'checkpoint' if k_random_cp is not None else 'not in checkpoint'})")
        vprint(f"max_edge_dist: {max_edge_dist_cp} (source: {'checkpoint' if max_edge_dist_cp is not None else 'not in checkpoint'})")
        vprint(f"num_rbf_3d: {num_rbf_3d_cp} (source: {'checkpoint' if num_rbf_3d_cp is not None else 'not in checkpoint'})")
        vprint(f"num_rbf_seq: {num_rbf_seq_cp} (source: {'checkpoint' if num_rbf_seq_cp is not None else 'not in checkpoint'})")

        vprint(f"\nRBF distance range parameters:")
        vprint(f"rbf_3d_min: {rbf_3d_min_final} (source: {rbf_source})")
        vprint(f"rbf_3d_max: {rbf_3d_max_final} (source: {rbf_source})")
        vprint(f"rbf_3d_spacing: {rbf_3d_spacing_final} (source: {rbf_source})")

        vprint(f"\nTime parameter resolution:")
        vprint(f"t_max: {dataset_params['t_max']} (source: {'checkpoint' if t_max_cp else 'default'})")
        vprint(f"t_min: {dataset_params['t_min']} (source: {'checkpoint' if t_min_cp else 'default'})")

    else:
        raise Exception(f'Checkpoint arguments are not found for model in {model_path}')


    # Infer architecture from state_dict keys if args are missing
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    vprint("\n" + "="*60)
    vprint("MODEL ARCHITECTURE INFERENCE")
    vprint("="*60)

    # Find maximum layer number from state_dict keys
    max_layer = 0
    layer_keys = []
    interleaved_layer_keys = []

    # Check for embed layers
    for key in state_dict.keys():
        if 'gnn.gnn.embed.' in key:
            try:
                layer_num = int(key.split('gnn.gnn.embed.')[1].split('.')[0])
                max_layer = max(max_layer, layer_num)
                layer_keys.append((layer_num, key))
            except:
                pass

    # Check for interleaved layers
    for key in state_dict.keys():
        if 'gnn.gnn.interleaved_layers.' in key:
            try:
                layer_num = int(key.split('gnn.gnn.interleaved_layers.')[1].split('.')[0])
                max_layer = max(max_layer, layer_num)
                interleaved_layer_keys.append((layer_num, key))
            except:
                pass

    # Infer num_layers from state dict
    inferred_num_layers = max_layer + 1  # layers are 0-indexed

    vprint(f"State dict analysis:")
    vprint(f"  Total keys in state_dict: {len(state_dict.keys())}")
    vprint(f"  Embed layer keys found: {len(layer_keys)}")
    vprint(f"  Interleaved layer keys found: {len(interleaved_layer_keys)}")
    if layer_keys:
        vprint(f"  Embed layer range: 0 to {max([ln for ln, _ in layer_keys])}")
        vprint(f"  Example embed layer keys: {[key for _, key in sorted(layer_keys)[:3]]}")
    if interleaved_layer_keys:
        vprint(f"  Interleaved layer range: 0 to {max([ln for ln, _ in interleaved_layer_keys])}")
        vprint(f"  Example interleaved layer keys: {[key for _, key in sorted(interleaved_layer_keys)[:3]]}")
    if layer_keys and len(layer_keys) > 3:
        vprint(f"    ... and {len(layer_keys) - 3} more embed keys")
    if interleaved_layer_keys and len(interleaved_layer_keys) > 3:
        vprint(f"    ... and {len(interleaved_layer_keys) - 3} more interleaved keys")
        if len(layer_keys) > 3:
            vprint(f"    ... and {len(layer_keys) - 3} more")
    vprint(f"  Inferred num_layers: {inferred_num_layers}")

    # Check for virtual node usage from model filename and dataset params
    model_filename = os.path.basename(model_path)
    use_virtual_inferred = 'noVirtual' not in model_filename and 'Novirtual' not in model_filename

    vprint(f"\nVirtual node inference:")
    vprint(f"  Model filename: {model_filename}")
    vprint(f"  Contains 'noVirtual': {'noVirtual' in model_filename}")
    vprint(f"  Contains 'Novirtual': {'Novirtual' in model_filename}")
    vprint(f"  Inferred use_virtual_node: {use_virtual_inferred}")

    # Use dataset parameter if available, otherwise use inference from filename
    if dataset_params['use_virtual_node'] is not None:
        use_virtual_final = dataset_params['use_virtual_node']
        vprint(f"  Final use_virtual_node: {use_virtual_final} (from checkpoint)")
    else:
        use_virtual_final = use_virtual_inferred
        vprint(f"  Final use_virtual_node: {use_virtual_final} (inferred from filename)")
        # Update dataset_params with inferred value
        dataset_params['use_virtual_node'] = use_virtual_final

    # Create model with proper parameter structure, using inferred values when needed
    vprint("\n" + "="*60)
    vprint("FINAL MODEL CREATION PARAMETERS")
    vprint("="*60)

    # Detect architecture details from processed checkpoint keys
    checkpoint_keys = set(state_dict.keys())  # Use processed state_dict (without module. prefix)
    has_old_layers = any('gnn.gnn.layers.' in key for key in checkpoint_keys)
    has_new_message_layers = any('gnn.gnn.message_layers.' in key for key in checkpoint_keys)
    has_new_embed = any('gnn.gnn.embed.' in key for key in checkpoint_keys)
    has_interleaved = any('gnn.gnn.interleaved_layers.' in key for key in checkpoint_keys)

    vprint(f"Architecture detection:")
    vprint(f"  Checkpoint has old 'layers': {has_old_layers}")
    vprint(f"  Checkpoint has new 'message_layers': {has_new_message_layers}")
    vprint(f"  Checkpoint has new 'embed': {has_new_embed}")
    vprint(f"  Checkpoint has 'interleaved_layers': {has_interleaved}")

    # Use the EXACT architecture from the checkpoint args first
    checkpoint_architecture = get_arg_value('architecture', 'blocked')
    vprint(f"  Checkpoint specifies architecture: '{checkpoint_architecture}'")

    # Auto-detect architecture parameters from checkpoint structure
    if has_interleaved:
        # Interleaved architecture detected - count actual layers from state dict
        interleaved_layer_indices = set()
        for key in checkpoint_keys:
            if 'gnn.gnn.interleaved_layers.' in key:
                try:
                    layer_idx = int(key.split('gnn.gnn.interleaved_layers.')[1].split('.')[0])
                    interleaved_layer_indices.add(layer_idx)
                except:
                    pass

        total_interleaved_layers = len(interleaved_layer_indices)
        max_interleaved_idx = max(interleaved_layer_indices) if interleaved_layer_indices else 0
        vprint(f"Detected interleaved architecture with {total_interleaved_layers} total layers (indices: {sorted(interleaved_layer_indices)})")

        # For interleaved, extract the actual embed/message layer counts from checkpoint args
        # But ensure they match the actual structure in the state dict
        checkpoint_embed_layers = get_arg_value('num_layers_gvp', get_arg_value('num_layers', 5))
        checkpoint_message_layers = get_arg_value('num_message_layers', 2)

        # Validate against state dict structure - the total should match
        expected_total = checkpoint_embed_layers + checkpoint_message_layers
        if total_interleaved_layers != expected_total:
            print(f"  Warning: Checkpoint args indicate {expected_total} layers ({checkpoint_embed_layers} embed + {checkpoint_message_layers} message)")
            vprint(f"           but state dict has {total_interleaved_layers} interleaved layers")
            vprint(f"  Using actual state dict structure for model creation")

            # Infer layer distribution from the actual structure
            # Typically embed layers come first, then message layers
            # But we need to create a model that matches the checkpoint exactly
            inferred_num_embed_layers = checkpoint_embed_layers
            inferred_num_message_layers = checkpoint_message_layers
        else:
            inferred_num_embed_layers = checkpoint_embed_layers
            inferred_num_message_layers = checkpoint_message_layers

        inferred_architecture = 'interleaved'
        use_legacy_naming = False

        vprint(f"Using interleaved architecture from checkpoint:")
        vprint(f"  num_embed_layers: {inferred_num_embed_layers}")
        vprint(f"  num_message_layers: {inferred_num_message_layers}")
        vprint(f"  architecture: {inferred_architecture}")

    elif has_old_layers and not has_new_message_layers:
        # Old architecture - need to determine layer split
        old_layer_indices = set()
        for key in checkpoint_keys:
            if key.startswith('gnn.gnn.layers.'):
                try:
                    layer_idx = int(key.split('gnn.gnn.layers.')[1].split('.')[0])
                    old_layer_indices.add(layer_idx)
                except:
                    pass

        total_old_layers = len(old_layer_indices)
        vprint(f"Detected old architecture with {total_old_layers} total layers")

        # For full compatibility, create a model that uses the old 'layers' naming
        # by setting num_message_layers=0 so ResidueGNN uses the old architecture
        inferred_num_embed_layers = total_old_layers
        inferred_num_message_layers = 0  # Force old architecture
        inferred_architecture = 'blocked'
        use_legacy_naming = True

        vprint(f"Using legacy-compatible architecture:")
        vprint(f"  num_embed_layers: {inferred_num_embed_layers}")
        vprint(f"  num_message_layers: {inferred_num_message_layers} (forces old 'layers' naming)")
        vprint(f"  architecture: {inferred_architecture}")

    else:
        # New blocked architecture or mixed - use checkpoint parameters exactly
        inferred_num_embed_layers = get_arg_value('num_layers_gvp', get_arg_value('num_layers', 4))
        inferred_num_message_layers = get_arg_value('num_message_layers', 1)
        inferred_architecture = checkpoint_architecture  # Use exact architecture from checkpoint
        use_legacy_naming = False

        vprint(f"Using modern blocked architecture:")
        vprint(f"  num_embed_layers: {inferred_num_embed_layers}")
        vprint(f"  num_message_layers: {inferred_num_message_layers}")
        vprint(f"  architecture: {inferred_architecture}")

    gvp_kwargs = {
        'node_dims': (get_arg_value('node_dim_s', 10), get_arg_value('node_dim_v', 3)),
        'edge_dims': (get_arg_value('edge_dim_s', 32), get_arg_value('edge_dim_v', 1)),
        'hidden_dims': (get_arg_value('hidden_dim', 256), get_arg_value('hidden_dim_v', 64)),
        'num_layers': inferred_num_embed_layers,  # For backward compatibility
        'num_embed_layers': inferred_num_embed_layers,
        'num_message_layers': inferred_num_message_layers,
        'architecture': inferred_architecture,
        'use_qkv': get_arg_value('use_qkv', True),
        'dropout': get_arg_value('dropout', 0.1),
        'update_edge_feats': get_arg_value('update_edge_feats', True),
        'attention_skip_connection': get_arg_value('attention_skip_connection', False),
        # use_virtual_node is handled at the data level, not ResidueGNN parameter
    }

    use_smoothed_labels = get_arg_value('use_smoothed_labels', False)
    label_similarity_csv = get_arg_value('label_similarity_csv', None) if use_smoothed_labels else None

    dfm_kwargs = {
        'K': 21,
        'alpha_min': get_arg_value('alpha_min', 1.0),
        'alpha_max': get_arg_value('alpha_max', 10.0),
        'alpha_spacing': get_arg_value('alpha_spacing', 0.01),
        'label_similarity_csv': label_similarity_csv,
    }

    vprint("GVP kwargs (for GNN architecture):")
    for key, value in gvp_kwargs.items():
        if key in ['node_dims', 'edge_dims', 'hidden_dims']:
            source = "checkpoint" if all(get_arg_value(f"{key.split('_')[0]}_{dim}") is not None for dim in ['s', 'v']) else "default/inferred"
        else:
            source = "checkpoint" if get_arg_value(key) is not None else "default/inferred"
        vprint(f"  {key}: {value} (source: {source})")

    vprint("\nDFM kwargs (for flow matching):")
    for key, value in dfm_kwargs.items():
        source = "checkpoint" if get_arg_value(key) is not None else "default"
        vprint(f"  {key}: {value} (source: {source})")

    # The c_factor lookup grid spans alpha in [alpha_min, alpha_max], and alpha
    # = 1 + t during sampling. Those bounds come from the checkpoint, so a run at
    # a horizon longer than the one the model was trained at would ask for alpha
    # past the end of the table. Grow the table to cover the requested horizon
    # instead, so the c_factor stays correct rather than saturating at the last
    # row. Costs nothing at the default horizon, where the grid already covers it.
    dfm_kwargs['alpha_max'] = _alpha_max_for_horizon(
        dfm_kwargs, args, trained_t_max=get_arg_value('t_max'))

    vprint(f"\nAdditional model parameters:")
    time_dim = get_arg_value('time_dim', 64)
    time_scale = get_arg_value('time_scale', 1.0)
    head_hidden = get_arg_value('head_hidden', 256)
    head_dropout = get_arg_value('head_dropout', 0.1)
    head_depth = get_arg_value('num_layers_prediction', 4)
    recycle_steps = get_arg_value('recycle_steps', 1)
    time_integration = get_arg_value('time_integration', 'film')
    use_time_conditioning = not get_arg_value('disable_time_conditioning', False)

    vprint(f"  time_dim: {time_dim} (source: {'checkpoint' if get_arg_value('time_dim') is not None else 'default'})")
    vprint(f"  time_scale: {time_scale} (source: {'checkpoint' if get_arg_value('time_scale') is not None else 'default'})")
    vprint(f"  head_hidden: {head_hidden} (source: {'checkpoint' if get_arg_value('head_hidden') is not None else 'default'})")
    vprint(f"  head_dropout: {head_dropout} (source: {'checkpoint' if get_arg_value('head_dropout') is not None else 'default'})")
    vprint(f"  head_depth: {head_depth} (source: {'checkpoint' if get_arg_value('num_layers_prediction') is not None else 'default'})")
    vprint(f"  recycle_steps: {recycle_steps} (source: {'checkpoint' if get_arg_value('recycle_steps') is not None else 'default'})")
    vprint(f"  time_integration: {time_integration} (source: {'checkpoint' if get_arg_value('time_integration') is not None else 'default'})")
    vprint(f"  use_time_conditioning: {use_time_conditioning} (source: {'checkpoint' if get_arg_value('disable_time_conditioning') is not None else 'default'})")

    # Extract DSSP loss parameter
    lambda_dssp_loss = get_arg_value('lambda_dssp_loss')

    # Extract auxiliary loss parameters (needed to reconstruct head architecture)
    lambda_electrostatic_loss = get_arg_value('lambda_electrostatic_loss')
    lambda_geom_topology_loss = get_arg_value('lambda_geom_topology_loss')

    # Detect head architecture directly from checkpoint state dict keys as a reliable fallback.
    # The checkpoint state dict is not yet extracted here, so peek at it directly.
    _raw_sd = checkpoint.get('model_state_dict', checkpoint)
    _has_module_prefix = any(k.startswith('module.') for k in _raw_sd.keys()) if hasattr(_raw_sd, 'keys') else False
    _ckpt_keys = {(k[7:] if _has_module_prefix else k) for k in _raw_sd.keys()} if hasattr(_raw_sd, 'keys') else set()
    _checkpoint_uses_shared_head = any('shared_head.' in k for k in _ckpt_keys)
    _checkpoint_has_geom = any('geom_topology_final' in k for k in _ckpt_keys)
    _checkpoint_has_elec = any('electrostatic_final' in k for k in _ckpt_keys)
    _checkpoint_has_dssp = any('dssp_final' in k for k in _ckpt_keys)

    if _checkpoint_uses_shared_head:
        vprint("Head architecture detected from checkpoint state dict: shared_head (aux loss mode)")
        # Override lambda values to be non-zero so DFMNodeClassifier builds shared_head.
        # Use a sentinel value of 1.0; the actual lambda values don't matter for inference.
        if not (lambda_dssp_loss is not None and lambda_dssp_loss > 0) and _checkpoint_has_dssp:
            lambda_dssp_loss = 1.0
            vprint("  Overriding lambda_dssp_loss=1.0 (detected dssp_final in checkpoint)")
        if not (lambda_electrostatic_loss is not None and lambda_electrostatic_loss > 0) and _checkpoint_has_elec:
            lambda_electrostatic_loss = 1.0
            vprint("  Overriding lambda_electrostatic_loss=1.0 (detected electrostatic_final in checkpoint)")
        if not (lambda_geom_topology_loss is not None and lambda_geom_topology_loss > 0) and _checkpoint_has_geom:
            lambda_geom_topology_loss = 1.0
            vprint("  Overriding lambda_geom_topology_loss=1.0 (detected geom_topology_final in checkpoint)")
        if not (_checkpoint_has_dssp or _checkpoint_has_elec or _checkpoint_has_geom):
            # shared_head present but no specific aux finals — at minimum need dssp or geom to trigger shared path
            lambda_geom_topology_loss = 1.0
            vprint("  Overriding lambda_geom_topology_loss=1.0 (shared_head detected, no specific aux finals found)")
    else:
        vprint("Head architecture detected from checkpoint state dict: head (single-head mode)")

    # Extract entropy feature parameters
    use_entropy_features = get_arg_value('use_entropy_features', False)
    entropy_smoothing = get_arg_value('entropy_smoothing', 0.1)

    # Adjust edge_dim_s if entropy features are enabled
    if use_entropy_features:
        current_edge_dim_s = gvp_kwargs['edge_dims'][0]
        gvp_kwargs['edge_dims'] = (current_edge_dim_s + 2, gvp_kwargs['edge_dims'][1])
        vprint(f"Entropy edge features enabled: edge_dim_s adjusted to {gvp_kwargs['edge_dims'][0]}")

    vprint(f"\nCreating DFMNodeClassifier with {inferred_num_embed_layers} embed layers, {inferred_num_message_layers} message layers, {head_depth} prediction head layers")
    vprint(f"Architecture: {inferred_architecture}, virtual_node={use_virtual_final} (stored in dataset_params)")
    if lambda_dssp_loss is not None and lambda_dssp_loss > 0:
        vprint(f"DSSP multitask learning enabled: lambda_dssp_loss={lambda_dssp_loss}")
    else:
        vprint("DSSP multitask learning disabled")
    if lambda_electrostatic_loss is not None and lambda_electrostatic_loss > 0:
        vprint(f"Electrostatic aux head enabled: lambda_electrostatic_loss={lambda_electrostatic_loss}")
    if lambda_geom_topology_loss is not None and lambda_geom_topology_loss > 0:
        vprint(f"Geom topology aux head enabled: lambda_geom_topology_loss={lambda_geom_topology_loss}")

    model = DFMNodeClassifier(
        gvp_kwargs=gvp_kwargs,
        dfm_kwargs=dfm_kwargs,
        time_dim=time_dim,
        time_scale=time_scale,
        head_hidden=head_hidden,
        head_dropout=head_dropout,
        head_depth=head_depth,
        recycle_steps=recycle_steps,
        time_integration=time_integration,
        use_time_conditioning=use_time_conditioning,
        lambda_dssp_loss=lambda_dssp_loss,
        lambda_electrostatic_loss=lambda_electrostatic_loss,
        lambda_geom_topology_loss=lambda_geom_topology_loss,
        use_entropy_features=use_entropy_features,
        entropy_smoothing=entropy_smoothing,
    )

    # Load state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Preprocess state dict to handle DataParallel/DistributedDataParallel keys
    # Remove 'module.' prefix if present (from DataParallel/DistributedDataParallel)
    processed_state_dict = {}
    has_module_prefix = any(key.startswith('module.') for key in state_dict.keys())

    if has_module_prefix:
        vprint("Detected DataParallel/DistributedDataParallel checkpoint - removing 'module.' prefix")
        for key, value in state_dict.items():
            if key.startswith('module.'):
                new_key = key[7:]  # Remove 'module.' prefix
                processed_state_dict[new_key] = value
            else:
                processed_state_dict[key] = value
        state_dict = processed_state_dict
        vprint(f"Preprocessed state dict: {len(state_dict)} keys after prefix removal")

    # Architecture detection and key adaptation
    vprint("\n" + "="*60)
    vprint("ARCHITECTURE DETECTION AND KEY ADAPTATION")
    vprint("="*60)

    # Detect architecture from checkpoint keys
    checkpoint_keys = set(state_dict.keys())
    model_keys = set(model.state_dict().keys())

    # Detect if checkpoint uses old 'layers' naming vs new 'message_layers' + 'embed' naming
    has_old_layers = any('gnn.gnn.layers.' in key for key in checkpoint_keys)
    has_new_message_layers = any('gnn.gnn.message_layers.' in key for key in checkpoint_keys)
    has_new_embed = any('gnn.gnn.embed.' in key for key in checkpoint_keys)
    has_interleaved_layers = any('gnn.gnn.interleaved_layers.' in key for key in checkpoint_keys)

    # Detect architecture type from model keys
    model_has_message_layers = any('gnn.gnn.message_layers.' in key for key in model_keys)
    model_has_embed = any('gnn.gnn.embed.' in key for key in model_keys)
    model_has_interleaved = any('gnn.gnn.interleaved_layers.' in key for key in model_keys)

    vprint(f"Checkpoint architecture detection:")
    vprint(f"  Has old 'layers' naming: {has_old_layers}")
    vprint(f"  Has new 'message_layers': {has_new_message_layers}")
    vprint(f"  Has new 'embed': {has_new_embed}")
    vprint(f"  Has 'interleaved_layers': {has_interleaved_layers}")

    vprint(f"Model architecture detection:")
    vprint(f"  Has 'message_layers': {model_has_message_layers}")
    vprint(f"  Has 'embed': {model_has_embed}")
    vprint(f"  Has 'interleaved_layers': {model_has_interleaved}")

    # Check for fundamental architecture mismatches that cannot be resolved by key mapping
    if has_interleaved_layers and not model_has_interleaved:
        print("\nCRITICAL ERROR: Architecture Mismatch!")
        vprint("  Checkpoint uses 'interleaved_layers' architecture")
        vprint("  Model was created with 'blocked' architecture")
        vprint("  This requires recreating the model with architecture='interleaved'")
        vprint("\nThe model must be recreated with the EXACT same architecture as the checkpoint.")
        raise RuntimeError("Architecture mismatch: checkpoint is 'interleaved' but model is 'blocked'. Cannot load.")

    if not has_interleaved_layers and model_has_interleaved:
        print("\nCRITICAL ERROR: Architecture Mismatch!")
        vprint("  Checkpoint uses 'blocked' architecture")
        vprint("  Model was created with 'interleaved' architecture")
        vprint("  This requires recreating the model with architecture='blocked'")
        vprint("\nThe model must be recreated with the EXACT same architecture as the checkpoint.")
        raise RuntimeError("Architecture mismatch: checkpoint is 'blocked' but model is 'interleaved'. Cannot load.")

    # Determine if we need key mapping
    needs_key_mapping = False
    mapping_strategy = None

    if has_old_layers and (model_has_message_layers or model_has_embed or model_has_interleaved):
        needs_key_mapping = True
        if model_has_interleaved:
            mapping_strategy = "old_to_interleaved"
        elif model_has_message_layers and model_has_embed:
            mapping_strategy = "old_to_blocked"
        else:
            mapping_strategy = "old_to_unknown"
    elif (has_new_message_layers or has_new_embed) and model_has_interleaved:
        needs_key_mapping = True
        mapping_strategy = "blocked_to_interleaved"
    elif (has_new_message_layers or has_new_embed) and (model_has_message_layers and model_has_embed):
        needs_key_mapping = False  # Same architecture
        mapping_strategy = "no_mapping_needed"

    vprint(f"Key mapping needed: {needs_key_mapping}")
    vprint(f"Mapping strategy: {mapping_strategy}")

    if needs_key_mapping:
        vprint("\nApplying flexible key mapping...")
        adapted_state_dict = {}

        if mapping_strategy == "old_to_blocked":
            # Map old 'layers' to new 'embed' structure only - ignore message_layers for now
            # This creates a simpler mapping that should work better

            # First, copy all non-GNN keys as-is
            for key, value in state_dict.items():
                if not key.startswith('gnn.gnn.layers.'):
                    adapted_state_dict[key] = value

            # Extract layer information from old format
            old_layer_keys = [key for key in checkpoint_keys if key.startswith('gnn.gnn.layers.')]
            old_layer_indices = set()
            for key in old_layer_keys:
                try:
                    layer_idx = int(key.split('gnn.gnn.layers.')[1].split('.')[0])
                    old_layer_indices.add(layer_idx)
                except:
                    pass

            old_layer_indices = sorted(old_layer_indices)
            vprint(f"  Found {len(old_layer_indices)} old layers: {old_layer_indices}")

            # Simple direct mapping: old layers -> embed layers with same indices
            vprint(f"  Direct mapping: old layers -> embed layers")
            mapped_count = 0
            for old_idx in old_layer_indices:
                for key in old_layer_keys:
                    if key.startswith(f'gnn.gnn.layers.{old_idx}.'):
                        suffix = key[len(f'gnn.gnn.layers.{old_idx}.'):]
                        new_key = f'gnn.gnn.embed.{old_idx}.{suffix}'
                        adapted_state_dict[new_key] = state_dict[key]
                        mapped_count += 1
                        if mapped_count <= 5:  # Show first few mappings
                            vprint(f"    Mapped: {key} -> {new_key}")

            if mapped_count > 5:
                vprint(f"    ... and {mapped_count - 5} more mappings")

            vprint(f"  Successfully mapped {mapped_count} parameter keys")

        elif mapping_strategy == "old_to_interleaved":
            # Map old 'layers' to new 'interleaved_layers' structure
            vprint("  Mapping old layers to interleaved architecture...")

            # Copy all non-GNN keys as-is
            for key, value in state_dict.items():
                if not key.startswith('gnn.gnn.layers.'):
                    adapted_state_dict[key] = value

            # Map old layers to interleaved layers sequentially
            old_layer_keys = [key for key in checkpoint_keys if key.startswith('gnn.gnn.layers.')]
            old_layer_indices = set()
            for key in old_layer_keys:
                try:
                    layer_idx = int(key.split('gnn.gnn.layers.')[1].split('.')[0])
                    old_layer_indices.add(layer_idx)
                except:
                    pass

            old_layer_indices = sorted(old_layer_indices)

            # Get interleaved layer count from model
            model_interleaved_indices = set()
            for key in model_keys:
                if key.startswith('gnn.gnn.interleaved_layers.'):
                    try:
                        layer_idx = int(key.split('gnn.gnn.interleaved_layers.')[1].split('.')[0])
                        model_interleaved_indices.add(layer_idx)
                    except:
                        pass

            model_interleaved_indices = sorted(model_interleaved_indices)
            vprint(f"  Model interleaved layers: {model_interleaved_indices}")

            # Map old layers to interleaved layers sequentially
            for i, old_idx in enumerate(old_layer_indices):
                if i < len(model_interleaved_indices):
                    new_idx = model_interleaved_indices[i]
                    for key in old_layer_keys:
                        if key.startswith(f'gnn.gnn.layers.{old_idx}.'):
                            suffix = key[len(f'gnn.gnn.layers.{old_idx}.'):]
                            new_key = f'gnn.gnn.interleaved_layers.{new_idx}.{suffix}'
                            adapted_state_dict[new_key] = state_dict[key]
                            vprint(f"    Mapped: {key} -> {new_key}")

        else:
            print(f"  Warning: Mapping strategy '{mapping_strategy}' not implemented, attempting partial mapping...")
            # Copy everything we can and skip what we can't
            for key, value in state_dict.items():
                if key in model_keys:
                    adapted_state_dict[key] = value

        # Use adapted state dict
        final_state_dict = adapted_state_dict
        vprint(f"Successfully created adapted state dict with {len(final_state_dict)} keys")

    else:
        vprint("No key mapping needed - architectures are compatible")
        final_state_dict = state_dict

    # Final validation
    model_keys = set(model.state_dict().keys())
    final_checkpoint_keys = set(final_state_dict.keys())

    missing_keys = model_keys - final_checkpoint_keys
    unexpected_keys = final_checkpoint_keys - model_keys

    vprint(f"\nFinal validation after key adaptation:")
    vprint(f"  Missing keys: {len(missing_keys)}")
    vprint(f"  Unexpected keys: {len(unexpected_keys)}")

    if missing_keys:
        vprint(f"  Missing keys (first 5): {list(missing_keys)[:5]}")
    if unexpected_keys:
        vprint(f"  Unexpected keys (first 5): {list(unexpected_keys)[:5]}")

    # Only fail if we have significant missing keys (not just a few parameters)
    critical_missing_threshold = 0.1  # Allow up to 10% missing keys
    critical_missing_ratio = len(missing_keys) / len(model_keys) if model_keys else 0

    if critical_missing_ratio > critical_missing_threshold:
        print(f"\nFATAL ERROR: Too many missing keys after adaptation!")
        vprint(f"Missing {len(missing_keys)}/{len(model_keys)} keys ({critical_missing_ratio:.1%} > {critical_missing_threshold:.1%} threshold)")
        vprint(f"This suggests fundamental architecture incompatibility that cannot be resolved by key mapping.")

        raise RuntimeError(f"Model loading failed: {critical_missing_ratio:.1%} of model keys are missing from checkpoint")

    # Load with strict=False to handle minor remaining key differences
    load_result = model.load_state_dict(final_state_dict, strict=False)

    if load_result.missing_keys:
        print(f"\nWarning: {len(load_result.missing_keys)} parameters could not be loaded from checkpoint:")
        for key in load_result.missing_keys[:5]:
            vprint(f"  - {key}")
        if len(load_result.missing_keys) > 5:
            vprint(f"  ... and {len(load_result.missing_keys) - 5} more")
        vprint("These parameters will be randomly initialized.")

    if load_result.unexpected_keys:
        vprint(f"\nInfo: {len(load_result.unexpected_keys)} parameters in checkpoint were not used:")
        for key in load_result.unexpected_keys[:5]:
            vprint(f"  - {key}")
        if len(load_result.unexpected_keys) > 5:
            vprint(f"  ... and {len(load_result.unexpected_keys) - 5} more")
        vprint("These parameters were ignored.")

    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()

    vprint(f"\nModel loaded successfully on {device}")

    # One concise line by default, so the user knows what was loaded without
    # reading the whole checkpoint configuration.
    if not verbose:
        _n = sum(p.numel() for p in model.parameters())
        _name = checkpoint.get('model_name', os.path.basename(local_model_path))
        _epoch = checkpoint.get('epoch')
        _epoch = f", epoch {_epoch}" if _epoch is not None else ""
        print(f"Loaded {_name}{_epoch}: {_n / 1e6:.1f}M parameters on {device}. "
              "Use --verbose for the full checkpoint configuration.")

    vprint("\n" + "="*60)
    vprint("FINAL DATASET PARAMETERS TO BE USED")
    vprint("="*60)
    vprint("These parameters will be used for dataset creation:")
    for key, value in dataset_params.items():
        vprint(f"  {key}: {value}")
    vprint("="*60)

    return model, dataset_params


def run_sampling_and_evaluation(checkpoint_path: str, args, device, output_base: str, model_name: str, job_timestamp: str):
    """
    Run sampling and evaluation using an external checkpoint.
    This function is called when running in sampling-only mode.

    Args:
        checkpoint_path: Path or URL to the checkpoint
        args: Training arguments object
        device: PyTorch device
        output_base: Base output directory
        model_name: Name for the model
        job_timestamp: Timestamp for this job

    Returns:
        dict: Results information including output directory
    """
    import json
    import os
    from pathlib import Path

    from data.cath_dataset import CathDataset

    print(f"\nLoading model from checkpoint: {checkpoint_path}")

    # Load model and dataset parameters from checkpoint
    model, dataset_params = load_model_distributed(checkpoint_path, device, args)

    # Create dataset for sampling using parameters from checkpoint
    print(f"\nCreating dataset with parameters from checkpoint:")
    for key, value in dataset_params.items():
        if value is not None:
            print(f"  {key}: {value}")

    # Prepare graph builder parameters
    graph_builder_kwargs = {
        'k': dataset_params.get('k_neighbors'),
        'k_farthest': dataset_params.get('k_farthest'),
        'k_random': dataset_params.get('k_random'),
        'num_rbf_3d': dataset_params.get('num_rbf_3d'),
        'num_rbf_seq': dataset_params.get('num_rbf_seq'),
        'no_source_indicator': dataset_params.get('no_source_indicator'),
        # RBF distance range parameters (should always be available now)
        'rbf_3d_min': dataset_params.get('rbf_3d_min'),
        'rbf_3d_max': dataset_params.get('rbf_3d_max'),
        'rbf_3d_spacing': dataset_params.get('rbf_3d_spacing')
    }

    # Validate RBF parameters
    if (graph_builder_kwargs['rbf_3d_min'] is None or
        graph_builder_kwargs['rbf_3d_max'] is None or
        graph_builder_kwargs['rbf_3d_spacing'] is None):
        raise RuntimeError(
            f"Internal error: RBF parameters should have been resolved in load_model_distributed.\n"
            f"Got: rbf_3d_min={graph_builder_kwargs['rbf_3d_min']}, "
            f"rbf_3d_max={graph_builder_kwargs['rbf_3d_max']}, "
            f"rbf_3d_spacing={graph_builder_kwargs['rbf_3d_spacing']}"
        )

    # Remove None values to use GraphBuilder defaults
    graph_builder_kwargs = {k: v for k, v in graph_builder_kwargs.items() if v is not None}

    # Create CATH dataset for sampling
    dataset = CathDataset(
        split_json=dataset_params['split_json'],
        map_pkl=dataset_params['map_pkl'],
        split='validation',  # Default to validation set for sampling
        max_len=dataset_params.get('max_length'),
        graph_builder_kwargs=graph_builder_kwargs,
        # Required time sampling parameters for sampling
        time_sampling_strategy='uniform',
        t_min=0.0,
        t_max=8.0,
        alpha_range=1.0
    )

    print(f"Dataset created with {len(dataset)} structures")

    # Set up output directories
    output_dir = os.path.join(output_base, 'sampling_results', job_timestamp)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Output directory: {output_dir}")

    # Run sampling with parameters from config or defaults
    sampling_config = getattr(args, 'sampling', {}) if hasattr(args, 'sampling') else {}

    # Extract sampling parameters (with defaults)
    num_samples = sampling_config.get('num_samples', 100)
    steps = sampling_config.get('steps', 20)
    T = sampling_config.get('T', 8.0)
    t_min = sampling_config.get('t_min', 0.0)
    temperature = sampling_config.get('flow_temp', 1.0)
    integration_method = sampling_config.get('integration_method', 'euler')
    save_probabilities = sampling_config.get('save_probabilities', True)

    # Get indices to sample (first num_samples from validation set)
    indices = list(range(min(num_samples, len(dataset))))

    print(f"\nStarting sampling with parameters:")
    print(f"  Number of samples: {len(indices)}")
    print(f"  Steps: {steps}")
    print(f"  Temperature: {temperature}")
    print(f"  Integration method: {integration_method}")
    print(f"  T: {T}, t_min: {t_min}")

    # Run sampling with batching for efficiency
    batch_size = getattr(args, 'batch_size', 4)  # Default batch size of 4
    results = sample_multiple_proteins_batched(
        model=model,
        dataset=dataset,
        indices=indices,
        steps=steps,
        T=T,
        t_min=t_min,
        save_probabilities=save_probabilities,
        integration_method=integration_method,
        batch_size=batch_size,
        args=args
    )

    # Save results
    output_prefix = f"{model_name}_{job_timestamp}"
    save_results_to_files(
        results=results,
        output_prefix=output_prefix,
        output_dir=output_dir,
        model_name=model_name,
        split='validation',
        steps=steps,
        T=T,
        save_probabilities=save_probabilities
    )

    print(f"\nSampling completed! Results saved to: {output_dir}")

    # Optional: Run structure prediction and evaluation if configured
    # This would integrate with your evaluation pipeline

    return {
        'output_dir': output_dir,
        'num_samples': len(indices),
        'results': results
    }


def align_sequence_to_structure(full_sequence, structure_sequence, verbose=False):
    """
    Efficiently align a full sequence (e.g., from UniProt) to a structure sequence.

    Args:
        full_sequence: Full sequence string
        structure_sequence: Structure sequence string (what's in the dataset)
        verbose: Print alignment details

    Returns:
        Dict with alignment information including start position and mapping
    """
    if verbose:
        print(f"Aligning sequences:")
        print(f"  Full sequence length: {len(full_sequence)}")
        print(f"  Structure sequence length: {len(structure_sequence)}")

    if not full_sequence or not structure_sequence:
        raise ValueError("Both full_sequence and structure_sequence must be non-empty")

    if len(structure_sequence) > len(full_sequence):
        raise ValueError(f"Structure sequence ({len(structure_sequence)}) cannot be longer than full sequence ({len(full_sequence)})")

    # Simple substring search for exact matches - O(n*m) but very fast for exact matches
    start_pos = full_sequence.find(structure_sequence)

    if start_pos != -1:
        # Perfect match
        if verbose:
            print(f"  Perfect match found at position {start_pos}")

        # Create position mapping: structure_pos -> full_seq_pos
        mapping = {i: start_pos + i for i in range(len(structure_sequence))}

        return {
            'mapping': mapping,
            'start_pos': start_pos,
            'alignment_score': 1.0
        }

    # Fuzzy matching with early termination
    best_score = 0
    best_start = 0
    min_required_score = 0.8
    struct_len = len(structure_sequence)
    min_matches_required = int(struct_len * min_required_score)

    # Sliding window approach with optimizations
    search_end = len(full_sequence) - struct_len + 1
    for start in range(search_end):
        matches = 0
        max_possible_remaining = struct_len

        # Count matches with early termination
        for i in range(struct_len):
            if structure_sequence[i] == full_sequence[start + i]:
                matches += 1

            max_possible_remaining -= 1
            # Early termination: if we can't possibly beat the best score, skip
            if matches + max_possible_remaining <= best_score * struct_len:
                break

            # Early success: if we already have enough matches, we can stop counting
            if matches >= min_matches_required:
                matches = matches  # Continue counting for exact score
                for j in range(i + 1, struct_len):
                    if structure_sequence[j] == full_sequence[start + j]:
                        matches += 1
                break

        score = matches / struct_len
        if score > best_score:
            best_score = score
            best_start = start

            # Early termination: if we found a very good match, stop searching
            if score >= 0.95:
                break

    if best_score >= min_required_score:
        best_mapping = {i: best_start + i for i in range(struct_len)}

        if verbose:
            print(f"  Best fuzzy match: {best_score:.2%} at position {best_start}")

        return {
            'mapping': best_mapping,
            'start_pos': best_start,
            'alignment_score': best_score
        }
    else:
        if verbose:
            print(f"  No good alignment found (best score: {best_score:.2%})")

        return {
            'mapping': {},
            'start_pos': -1,
            'alignment_score': best_score
        }


def create_inpainting_mask_with_alignment(full_sequence, structure_sequence,
                                         mask_positions=None, known_sequence=None,
                                         mask_ratio=0.3, verbose=False, device='cpu'):
    """
    Create an inpainting mask using alignment between full and structure sequences.

    Args:
        full_sequence: Full sequence (e.g., from UniProt)
        structure_sequence: Structure sequence (from dataset)
        mask_positions: Positions in FULL sequence to mask
        known_sequence: Known template with 'X' for positions to predict
        mask_ratio: Random masking ratio if no specific positions
        verbose: Print detailed information
        device: Device for PyTorch tensor

    Returns:
        Dict with mask information and alignment details (mask as PyTorch tensor)
    """
    # First align the sequences
    alignment_info = align_sequence_to_structure(full_sequence, structure_sequence, verbose)

    if alignment_info['alignment_score'] < 0.8:
        raise ValueError(f"Poor sequence alignment (score: {alignment_info['alignment_score']:.3f} < 0.8). "
                        f"Cannot reliably map positions between sequences. "
                        f"Full sequence length: {len(full_sequence)}, "
                        f"Structure sequence length: {len(structure_sequence)}")

    structure_len = len(structure_sequence)
    mask = torch.zeros(structure_len, dtype=torch.bool, device=device)

    if mask_positions is not None:
        # Convert full sequence positions to structure positions
        if isinstance(mask_positions, str):
            positions = [int(x.strip()) for x in mask_positions.split(',')]
        else:
            positions = mask_positions

        mapping = alignment_info['mapping']
        reverse_mapping = {v: k for k, v in mapping.items()}

        masked_count = 0
        for full_pos in positions:
            if full_pos in reverse_mapping:
                struct_pos = reverse_mapping[full_pos]
                mask[struct_pos] = True
                masked_count += 1
                if verbose:
                    print(f"  Full position {full_pos} -> Structure position {struct_pos}")

        if verbose:
            print(f"Mapped {masked_count}/{len(positions)} positions from full to structure sequence")

    elif known_sequence is not None:
        # Use known sequence template
        if len(known_sequence) != len(full_sequence):
            raise ValueError(f"Known sequence length ({len(known_sequence)}) doesn't match full sequence length ({len(full_sequence)})")

        mapping = alignment_info['mapping']
        masked_count = 0

        for struct_pos, full_pos in mapping.items():
            if full_pos < len(known_sequence) and known_sequence[full_pos].upper() == 'X':
                mask[struct_pos] = True
                masked_count += 1

        if verbose:
            print(f"Mapped {masked_count} masked positions from known sequence template")

    else:
        # Random masking
        num_to_mask = int(structure_len * mask_ratio)
        if num_to_mask > 0:
            random_indices = torch.randperm(structure_len, device=device)[:num_to_mask]
            mask[random_indices] = True

        if verbose:
            print(f"Random masking: {num_to_mask} positions")

    return {
        'mask': mask,
        'alignment_info': alignment_info,
        'num_masked': mask.sum().item()
    }


def create_simple_inpainting_mask(structure_sequence, mask_positions=None,
                                known_sequence=None, mask_ratio=0.3, device='cpu'):
    """
    Create a simple inpainting mask without sequence alignment.

    Args:
        structure_sequence: Structure sequence string
        mask_positions: Specific positions to mask (0-indexed)
        known_sequence: Template with 'X' for positions to predict
        mask_ratio: Random masking ratio
        device: Device for PyTorch tensor

    Returns:
        Dict with mask information (mask as PyTorch tensor)
    """
    structure_len = len(structure_sequence)
    mask = torch.zeros(structure_len, dtype=torch.bool, device=device)

    if mask_positions is not None:
        if isinstance(mask_positions, str):
            positions = [int(x.strip()) for x in mask_positions.split(',')]
        else:
            positions = mask_positions

        for pos in positions:
            if 0 <= pos < structure_len:
                mask[pos] = True

    elif known_sequence is not None:
        if len(known_sequence) != structure_len:
            raise ValueError(f"Known sequence length ({len(known_sequence)}) doesn't match structure sequence length ({structure_len})")

        for i, aa in enumerate(known_sequence):
            if aa.upper() == 'X':
                mask[i] = True

    else:
        # Random masking
        num_to_mask = int(structure_len * mask_ratio)
        if num_to_mask > 0:
            random_indices = torch.randperm(structure_len, device=device)[:num_to_mask]
            mask[random_indices] = True

    return {
        'mask': mask,
        'alignment_info': {'mapping': None, 'start_pos': None, 'alignment_score': 0.0},
        'num_masked': mask.sum().item()
    }


def create_mask_from_fixed_positions(N, fixed_positions, device='cpu'):
    """
    Create inpainting mask from list of fixed positions.

    Args:
        N: Total sequence length
        fixed_positions: List of 0-indexed positions to keep FIXED
        device: PyTorch device

    Returns:
        Boolean tensor [N] where:
            - False = position is FIXED (not sampled)
            - True = position is MASKED (will be sampled/inpainted)

    Example:
        N=10, fixed_positions=[1, 5, 8]
        Returns: [True, False, True, True, True, False, True, True, False, True]
        Meaning: Sample positions 0,2,3,4,6,7,9; keep 1,5,8 fixed
    """
    mask = torch.ones(N, dtype=torch.bool, device=device)  # All True = all masked

    if fixed_positions:
        fixed_tensor = torch.tensor(fixed_positions, dtype=torch.long, device=device)
        mask[fixed_tensor] = False  # Mark fixed positions as False

    return mask


def generate_detailed_json_output(results, structure_names, output_dir, output_prefix, K=21):
    """
    Generate detailed JSON output with time-step information for each protein.

    Args:
        results: List of result dictionaries containing trajectory data
        structure_names: List of structure names/PDB IDs
        output_dir: Output directory
        output_prefix: Prefix for output filename

    Returns:
        str: Path to the generated JSON file
    """
    import numpy as np

    # Build output data structure
    detailed_output = {}

    for i, (result, structure_name) in enumerate(zip(results, structure_names)):
        if 'trajectory_data' not in result:
            print(f"Warning: No trajectory data for structure {structure_name}")
            continue

        trajectory = result['trajectory_data']
        pdb_id = structure_name if structure_name != 'unknown' else f"structure_{i}"

        # Initialize structure data
        detailed_output[pdb_id] = {}

        # Add flank filtering metadata if present
        if 'flank_filtering' in result:
            detailed_output[pdb_id]['flank_filtering'] = result['flank_filtering']

        # Process each position
        for pos, pos_data in trajectory['positions'].items():
            # Get trajectory for this position
            time_points = pos_data['time_points']
            most_likely_aas = pos_data['most_likely_aa']
            probabilities = pos_data['probabilities']
            detailed_breakdowns = pos_data.get('detailed_breakdown', [])

            # Create trajectory entries with detailed amino acid breakdown
            trajectory_entries = []
            for idx, (t, aa_idx, prob) in enumerate(zip(time_points, most_likely_aas, probabilities)):
                # Bounds checking for most likely AA
                if 0 <= aa_idx < len(IDX_TO_AA):
                    aa_name = IDX_TO_AA[aa_idx]
                    aa_single = THREE_TO_ONE.get(aa_name, 'X')
                else:
                    aa_name = 'XXX'
                    aa_single = 'X'

                trajectory_entry = {
                    'time_point': round(float(t), 4),
                    'most_likely_amino_acid': aa_single,
                    'amino_acid_name': aa_name,
                    'current_probability': round(float(prob), 6)
                }

                # Add detailed amino acid breakdown if available
                if idx < len(detailed_breakdowns):
                    trajectory_entry['amino_acid_breakdown'] = detailed_breakdowns[idx]

                trajectory_entries.append(trajectory_entry)

            # Get ground truth information for this position if available
            ground_truth_aa = None
            ground_truth_name = None

            if 'true_indices' in result and result['true_indices'] is not None:
                true_indices = result['true_indices']
                if pos < len(true_indices):
                    true_idx = true_indices[pos]
                    if 0 <= true_idx < len(IDX_TO_AA):
                        ground_truth_name = IDX_TO_AA[true_idx]
                        ground_truth_aa = THREE_TO_ONE.get(ground_truth_name, 'X')
                    else:
                        ground_truth_name = 'XXX'
                        ground_truth_aa = 'X'

            # Store position data
            position_info = {
                'trajectory': trajectory_entries,
                'final_prediction': trajectory_entries[-1]['most_likely_amino_acid'] if trajectory_entries else 'X',
                'final_probability': trajectory_entries[-1]['current_probability'] if trajectory_entries else 0.0,
                'ground_truth': ground_truth_aa,
                'ground_truth_name': ground_truth_name
            }

            detailed_output[pdb_id][str(pos)] = position_info

    # Generate output filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_filename = f"{timestamp}_{output_prefix}_detailed_predictions.json"
    json_filepath = os.path.join(output_dir, json_filename)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save JSON with pretty formatting
    with open(json_filepath, 'w') as f:
        json.dump(detailed_output, f, indent=2, sort_keys=True)

    print(f"Detailed JSON output saved to: {json_filepath}")

    # Generate trajectory analysis file for get_prediction_accuracy function
    trajectory_analysis_data = {}

    for i, (result, structure_name) in enumerate(zip(results, structure_names)):
        if 'trajectory_data' not in result:
            continue

        trajectory = result['trajectory_data']
        pdb_id = structure_name if structure_name != 'unknown' else f"structure_{i}"

        # Extract model predictions at each timestep
        time_points = trajectory['time_points']
        positions = trajectory['positions']

        if not positions:
            continue

        # Get sequence length from first position
        first_pos_key = list(positions.keys())[0]
        first_pos_data = positions[first_pos_key]
        num_timesteps = len(first_pos_data['time_points'])

        # Initialize arrays for this protein
        sequence_length = len(positions)
        model_predictions = []
        current_states = []

        # Extract data for each timestep
        for timestep in range(num_timesteps):
            timestep_predictions = np.zeros((sequence_length, K))
            timestep_states = np.zeros((sequence_length, K))

            for pos_idx, (pos_key, pos_data) in enumerate(positions.items()):
                if timestep < len(pos_data.get('detailed_breakdown', [])):
                    # Get the full probability distribution for this timestep
                    breakdown = pos_data['detailed_breakdown'][timestep]
                    for aa_single_letter, aa_data in breakdown.items():
                        # Convert single-letter AA to 3-letter, then to index
                        if aa_single_letter in SINGLE_TO_TRIPLE:
                            aa_three_letter = SINGLE_TO_TRIPLE[aa_single_letter]
                            if aa_three_letter in AA_TO_IDX:
                                aa_idx = AA_TO_IDX[aa_three_letter]
                                # Use the predicted_prob for model predictions, current_prob for current states
                                if isinstance(aa_data, dict):
                                    pred_prob = aa_data.get('predicted_prob', 0.0)
                                    curr_prob = aa_data.get('current_prob', 0.0)
                                    timestep_predictions[pos_idx, aa_idx] = pred_prob
                                    timestep_states[pos_idx, aa_idx] = curr_prob
                                else:
                                    # Fallback if aa_data is not a dict (shouldn't happen with new format)
                                    timestep_predictions[pos_idx, aa_idx] = float(aa_data)
                                    timestep_states[pos_idx, aa_idx] = float(aa_data)

            model_predictions.append(timestep_predictions)
            current_states.append(timestep_states)

        # Store predictions and states
        trajectory_analysis_data[f'{pdb_id}_model_predictions'] = model_predictions
        trajectory_analysis_data[f'{pdb_id}_current_states'] = current_states

        # Store ground truth
        if 'true_indices' in result and result['true_indices'] is not None:
            ground_truth_onehot = np.zeros((sequence_length, K))
            for pos_idx, true_idx in enumerate(result['true_indices']):
                if 0 <= true_idx < K:
                    ground_truth_onehot[pos_idx, true_idx] = 1.0
            trajectory_analysis_data[f'{pdb_id}_ground_truth'] = ground_truth_onehot

    # Save trajectory analysis file
    if trajectory_analysis_data:
        trajectory_filename = f"{timestamp}_{output_prefix}_trajectory_analysis.npz"
        trajectory_filepath = os.path.join(output_dir, trajectory_filename)

        np.savez_compressed(trajectory_filepath, **trajectory_analysis_data)
        print(f"Trajectory analysis data saved to: {trajectory_filepath}")
        print(f"Use get_prediction_accuracy('{trajectory_filename}', timestep, pdb_id) to analyze results")

    return json_filepath


def sample_multiple_proteins_inpainting_with_trajectory(model, dataset, indices=None, steps=50, T=8.0, t_min=0.0, K=21,
                                                     mask_positions_list=None, known_sequences_list=None, mask_ratio=0.3,
                                                     integration_method='euler', rtol=1e-5, atol=1e-8, max_structures=None,
                                                     output_dir=None, output_prefix=None, args=None):
    """
    Sample multiple proteins with inpainting while tracking trajectories for detailed JSON output.

    Args:
        model: Trained model
        dataset: Dataset
        indices: List of structure indices to sample
        steps: Number of sampling steps
        T: Maximum time
        t_min: Minimum time
        K: Number of amino acid classes
        mask_positions_list: List of mask positions for each protein (or None for random masking)
        known_sequences_list: List of known sequences for each protein (or None)
        mask_ratio: Random masking ratio if no specific positions
        integration_method: 'euler' or 'rk45'
        rtol: Relative tolerance for RK45
        atol: Absolute tolerance for RK45
        max_structures: Maximum number of structures to process
        output_dir: Output directory for trajectory analysis (optional)
        output_prefix: Output prefix for trajectory analysis (optional)
        args: Arguments object (optional, for global access)

    Returns:
        tuple: (results, structure_names)
    """
    # Import here to avoid circular imports
    from training.inpainting import sample_chain_inpainting_with_trajectory

    if indices is None:
        indices = list(range(len(dataset)))

    if max_structures is not None:
        indices = indices[:max_structures]

    results = []
    structure_names = []

    print(f"Sampling {len(indices)} structures with inpainting trajectory tracking...")

    for i, idx in enumerate(indices):
        print(f"Processing structure {i+1}/{len(indices)} (index {idx})...")

        try:
            data, y_true, mask, time_value, dssp_targets = dataset[idx]  # Unpack 5 values (includes DSSP)
            structure_name = getattr(data, 'name', f'structure_{idx}')
            structure_names.append(structure_name)

            # Get structure sequence
            structure_sequence = getattr(data, 'filtered_seq', None)
            if structure_sequence is None:
                raise ValueError(f"No filtered_seq found for structure {idx}")

            # Get mask positions and known sequence for this protein
            mask_positions = mask_positions_list[i] if mask_positions_list and i < len(mask_positions_list) else None
            known_sequence = known_sequences_list[i] if known_sequences_list and i < len(known_sequences_list) else None

            # Create temporary args object for compatibility
            class TempArgs:
                def __init__(self, dirichlet_concentration, flow_temp=1.0):
                    self.dirichlet_concentration = dirichlet_concentration
                    self.flow_temp = flow_temp
                    self.use_smoothed_targets = getattr(args, 'use_smoothed_targets', False) if args else False
                    self.use_c_factor = getattr(args, 'use_c_factor', False) if args else False

            temp_args = TempArgs(args.dirichlet_concentration if args else 20.0)

            # Sample with inpainting trajectory
            if integration_method == 'rk45':
                raise NotImplementedError("RK45 integration was moved to legacy file.")
            else:
                final_probs, pred_seq, mask_tensor, alignment_info, eval_metrics, trajectory_data = sample_chain_inpainting_with_trajectory(
                    model, data, T=T, t_min=t_min, steps=steps, K=K,
                    full_sequence=structure_sequence,  # Use structure sequence as full sequence
                    structure_sequence=structure_sequence,
                    mask_positions=mask_positions,
                    known_sequence=known_sequence,
                    mask_ratio=mask_ratio,
                    args=temp_args
                )

            # Calculate legacy accuracy if ground truth available
            accuracy = None
            true_seq = None

            # Try to get ground truth from y_true first, then from data.filtered_seq
            if y_true is not None:
                true_seq = y_true.argmax(-1).tolist()
            elif hasattr(data, 'filtered_seq') and data.filtered_seq is not None:
                # Convert filtered_seq to indices using the same mapping as the dataset
                true_seq = []
                for aa_char in data.filtered_seq:
                    if aa_char in SINGLE_TO_TRIPLE:
                        aa3 = SINGLE_TO_TRIPLE[aa_char]
                        if aa3 in AA_TO_IDX:
                            true_seq.append(AA_TO_IDX[aa3])
                        else:
                            true_seq.append(20)  # Unknown
                    else:
                        true_seq.append(20)  # Unknown

            # Calculate accuracy if we have ground truth
            if true_seq is not None:
                correct = sum(p == t for p, t in zip(pred_seq, true_seq))
                accuracy = correct / len(pred_seq) * 100

            result = {
                'structure_idx': idx,
                'structure_name': structure_name,
                'length': len(pred_seq),
                'predicted_sequence': pred_seq,
                'true_indices': true_seq,
                'accuracy': accuracy,
                'trajectory_data': trajectory_data,
                'eval_metrics': eval_metrics,  # Detailed evaluation metrics
                'inpainting_mask': mask_tensor.cpu().numpy(),
                'alignment_info': alignment_info,
                'mask_positions': mask_positions,
                'known_sequence': known_sequence,
                'mask_ratio': mask_ratio
            }

            results.append(result)

        except Exception as e:
            print(f"Error processing structure {idx}: {e}")
            structure_names.append(f'structure_{idx}')
            results.append({
                'structure_idx': idx,
                'error': str(e)
            })

    # Generate trajectory analysis files if output directory is specified
    if output_dir is not None:
        # Use default output_prefix if not provided
        output_prefix_final = output_prefix if output_prefix is not None else "inpainting_trajectory"
        print(f"\nGenerating inpainting trajectory analysis files for {len(indices)} proteins...")
        try:
            from training.inpainting import generate_inpainting_trajectory_json
            json_filepath = generate_inpainting_trajectory_json(results, structure_names, output_dir, output_prefix_final)
            print(f"Inpainting trajectory JSON saved to: {json_filepath}")
        except Exception as e:
            print(f"Warning: Could not save trajectory analysis files: {e}")

    return results, structure_names


def sample_multiple_proteins_with_trajectory(model, dataset, indices=None, steps=50, T=8.0, t_min=0.0, K=21,
                                           integration_method='euler', rtol=1e-5, atol=1e-8, max_structures=None,
                                           output_dir=None, output_prefix=None, args=None):
    """
    Sample multiple proteins while tracking trajectories for detailed JSON output.

    Args:
        model: Trained model
        dataset: Dataset
        indices: List of structure indices to sample
        steps: Number of sampling steps
        T: Maximum time
        K: Number of amino acid classes
        integration_method: 'euler' or 'rk45'
        rtol: Relative tolerance for RK45
        atol: Absolute tolerance for RK45
        max_structures: Maximum number of structures to process
        output_dir: Output directory for trajectory analysis (optional)
        output_prefix: Output prefix for trajectory analysis (optional)
        args: Arguments object (optional, for global access)

    Returns:
        tuple: (results, structure_names)
    """
    # Import here to avoid circular imports
    from .sample import sample_chain_with_trajectory

    if indices is None:
        indices = list(range(len(dataset)))

    if max_structures is not None:
        indices = indices[:max_structures]

    results = []
    structure_names = []

    print(f"Sampling {len(indices)} structures with trajectory tracking...")

    for i, idx in enumerate(indices):
        print(f"Processing structure {i+1}/{len(indices)} (index {idx})...")

        try:
            data, y_true, mask, time_value, dssp_targets = dataset[idx]  # Unpack 5 values (includes DSSP)
            structure_name = getattr(data, 'name', f'structure_{idx}')
            structure_names.append(structure_name)

            # Sample with trajectory
            # Create temporary args object for dirichlet_concentration
            class TempArgs:
                def __init__(self, dirichlet_concentration):
                    self.dirichlet_concentration = dirichlet_concentration
                    self.flow_temp = 1.0  # Default temperature

            temp_args = TempArgs(args.dirichlet_concentration if args else 20.0)

            if integration_method == 'rk45':
                raise NotImplementedError("RK45 integration was moved to legacy file.")
            else:
                final_probs, pred_seq, trajectory, eval_metrics = sample_chain_with_trajectory(
                    model, data, T=T, t_min=t_min, steps=steps, K=K, args=temp_args
                )

            # Calculate legacy accuracy if ground truth available (for backwards compatibility)
            accuracy = None
            true_seq = None

            # Try to get ground truth from y_true first, then from data.filtered_seq
            if y_true is not None:
                true_seq = y_true.argmax(-1).tolist()
            elif hasattr(data, 'filtered_seq') and data.filtered_seq is not None:
                # Convert filtered_seq to indices using the same mapping as the dataset
                true_seq = []
                for aa_char in data.filtered_seq:
                    if aa_char in SINGLE_TO_TRIPLE:
                        aa3 = SINGLE_TO_TRIPLE[aa_char]
                        if aa3 in AA_TO_IDX:
                            true_seq.append(AA_TO_IDX[aa3])
                        else:
                            true_seq.append(20)  # Unknown
                    else:
                        true_seq.append(20)  # Unknown

            # Calculate accuracy if we have ground truth
            if true_seq is not None:
                correct = sum(p == t for p, t in zip(pred_seq, true_seq))
                accuracy = correct / len(pred_seq) * 100

            result = {
                'structure_idx': idx,
                'structure_name': structure_name,
                'length': len(pred_seq),
                'predicted_sequence': pred_seq,
                'true_indices': true_seq,
                'accuracy': accuracy,
                'trajectory_data': trajectory,
                'eval_metrics': eval_metrics  # Detailed evaluation metrics
            }

            results.append(result)

        except Exception as e:
            print(f"Error processing structure {idx}: {e}")
            structure_names.append(f'structure_{idx}')
            results.append({
                'structure_idx': idx,
                'error': str(e)
            })

    # Generate trajectory analysis NPZ file if we have < 4 proteins and output directory is specified
    if len(indices) < 4 and output_dir is not None:
        # Use default output_prefix if not provided
        output_prefix_final = output_prefix if output_prefix is not None else "protein_sampling_trajectory"
        if output_prefix is None:
            print(f"\nAutomatically generating trajectory analysis NPZ file for {len(indices)} proteins (< 4 proteins)...")
        else:
            print(f"\nGenerating trajectory analysis NPZ file for {len(indices)} proteins...")
        try:
            from training.trajectory_saver import save_trajectory_analysis_npz
            npz_filepath = save_trajectory_analysis_npz(results, structure_names, output_dir, output_prefix_final)
            print(f"Trajectory analysis saved to: {npz_filepath}")
        except Exception as e:
            print(f"Warning: Could not save trajectory analysis NPZ file: {e}")

    return results, structure_names





def sample_multiple_proteins(model, dataset, indices=None, steps=50, T=8.0, K=21, save_probabilities=True,
                            integration_method='euler', rtol=1e-5, atol=1e-8, args=None):
    """
    Sample sequences for multiple protein structures.

    Args:
        model: Trained DFM model
        dataset: CathDataset instance
        indices: List of structure indices to sample (None = all)
        steps: Number of sampling steps
        T: Maximum time
        K: Number of amino acid classes
        save_probabilities: Whether to save probability distributions
        integration_method: 'euler' or 'rk45'
        rtol: Relative tolerance for RK45
        atol: Absolute tolerance for RK45
        args: Arguments object (optional)

    Returns:
        List of result dictionaries with evaluation metrics
    """
    # Import here to avoid circular imports
    from .sample import sample_chain, sample_chain_with_replicates, select_best_sample_aux_heads

    if indices is None:
        indices = list(range(len(dataset)))

    # Get number of samples per protein (default to 1 for backward compatibility)
    num_sample_per_protein = getattr(args, 'num_sample_per_protein', 1) if args else 1

    use_recycling = args and getattr(args, 'recycled_sampling', False)
    use_aux_ranking = args and getattr(args, 'aux_head_ranking', False)
    pick_based_on_dssp = args and getattr(args, 'pick_based_on_dssp', False)
    pick_based_on_perplexity = args and getattr(args, 'pick_based_on_perplexity', False)
    use_replicate_path = num_sample_per_protein > 1 and (
        use_recycling or use_aux_ranking or pick_based_on_dssp or pick_based_on_perplexity
    )

    device = next(model.parameters()).device

    results = []

    total_samples = len(indices) * num_sample_per_protein
    print(f"Sampling {len(indices)} structures with {num_sample_per_protein} samples per structure ({total_samples} total samples) using {integration_method} integration...")
    if use_replicate_path:
        print(f"  Using replicate path (recycling={use_recycling}, aux_ranking={use_aux_ranking}, pick_dssp={pick_based_on_dssp}, pick_perplexity={pick_based_on_perplexity})")

    for i, idx in enumerate(indices):
        print(f"Processing structure {i+1}/{len(indices)} (index {idx}) with {num_sample_per_protein} samples...")

        try:
            data, y_true, mask, time_value, dssp_targets = dataset[idx]  # Unpack 5 values (includes DSSP)
            structure_name = getattr(data, 'name', f'structure_{idx}')

            # Create temporary args object for dirichlet_concentration if needed
            if args is None:
                class TempArgs:
                    def __init__(self):
                        self.dirichlet_concentration = 20.0
                        self.flow_temp = 1.0
                temp_args = TempArgs()
            else:
                temp_args = args

            if use_replicate_path:
                # Route through sample_chain_with_replicates for recycling and/or aux ranking
                replicate_results = sample_chain_with_replicates(
                    model, data, dataset,
                    structure_idx=idx,
                    T=T,
                    t_min=0.0,
                    steps=steps,
                    K=K,
                    num_replicates=num_sample_per_protein,
                    dssp_targets=dssp_targets,
                    track_dssp_accuracy=(pick_based_on_dssp or (pick_based_on_perplexity and dssp_targets is not None)),
                    verbose=False,
                    args=temp_args
                )

                # Select best replicate
                if use_aux_ranking:
                    best_result = select_best_sample_aux_heads(
                        replicate_results, model, data, temp_args, device, T=T
                    )
                elif pick_based_on_dssp:
                    best_result = select_best_sample(replicate_results, primary_metric='dssp')
                elif pick_based_on_perplexity:
                    best_result = select_best_sample(replicate_results, primary_metric='perplexity')
                else:
                    best_result = replicate_results[0]
                    best_result['selected_reason'] = 'recycled_sampling with no selection criterion, took replicate 0'

                final_probs = torch.tensor(best_result['final_probabilities']) if not isinstance(best_result['final_probabilities'], torch.Tensor) else best_result['final_probabilities']
                pred_seq = best_result['predicted_indices']
                eval_metrics = best_result.get('evaluation_metrics', {})
                accuracy = None
                true_seq = None
                if y_true is not None:
                    true_seq = y_true.argmax(-1).tolist()
                    correct = sum(p == t for p, t in zip(pred_seq, true_seq))
                    accuracy = correct / len(pred_seq) * 100

                predicted_aa = [IDX_TO_AA[i] if 0 <= i < len(IDX_TO_AA) else 'XXX' for i in pred_seq]
                result = {
                    'structure_idx': idx,
                    'structure_name': structure_name,
                    'sample_idx': 0,
                    'length': len(pred_seq),
                    'predicted_indices': pred_seq,
                    'predicted_aa': predicted_aa,
                    'predicted_sequence': ''.join([THREE_TO_ONE[aa] for aa in predicted_aa]),
                    'true_indices': true_seq,
                    'accuracy': accuracy,
                    'eval_metrics': eval_metrics,
                    'final_probabilities': final_probs.cpu().numpy(),
                    'selection_metadata': {k: v for k, v in best_result.items() if k in ('selected_reason', 'all_aux_scores', 'sequence_perplexity', 'final_dssp_accuracy')},
                }
                results.append(result)
                continue

            # Sample multiple sequences for the same structure (original loop path)
            for sample_idx in range(num_sample_per_protein):
                if num_sample_per_protein > 1:
                    print(f"  Sample {sample_idx+1}/{num_sample_per_protein}...")

                # Sample sequence with evaluation metrics
                # Each call to sample_chain will use different random noise due to PyTorch's RNG
                if integration_method == 'rk45':
                    raise NotImplementedError("RK45 integration was moved to legacy file.")
                else:
                    # `dataset`, `idx` and `dssp_targets` are all in scope here and
                    # the recycled-sampling branch above already passes them. This
                    # call omitted `dataset` (a required positional) and unpacked
                    # three values from a four-value return, so every structure in a
                    # --sample_all run failed with "sample_chain() missing 1 required
                    # positional argument".
                    final_probs, pred_seq, eval_metrics, _dssp_logits = sample_chain(
                        model, data, dataset, structure_idx=idx,
                        T=T, t_min=0.0, steps=steps, K=K,
                        verbose=getattr(args, 'verbose', False),
                        args=temp_args, dssp_targets=dssp_targets
                    )

                # Calculate legacy accuracy if ground truth available (for backwards compatibility)
                accuracy = None
                true_seq = None
                if y_true is not None:
                    true_seq = y_true.argmax(-1).tolist()
                    correct = sum(p == t for p, t in zip(pred_seq, true_seq))
                    accuracy = correct / len(pred_seq) * 100

                # Convert to amino acid names
                predicted_aa = []
                for idx_val in pred_seq:
                    if 0 <= idx_val < len(IDX_TO_AA):
                        predicted_aa.append(IDX_TO_AA[idx_val])
                    else:
                        predicted_aa.append('XXX')

                result = {
                    'structure_idx': idx,
                    'structure_name': structure_name,
                    'sample_idx': sample_idx,
                    'length': len(pred_seq),
                    'predicted_indices': pred_seq,
                    'predicted_aa': predicted_aa,
                    'predicted_sequence': ''.join([THREE_TO_ONE[aa] for aa in predicted_aa]),
                    'true_indices': true_seq,
                    'accuracy': accuracy,
                    'eval_metrics': eval_metrics,
                    'final_probabilities': final_probs.cpu().numpy()
                }

                # Don't aggregate statistics - each sample will be saved separately
                results.append(result)

        except Exception as e:
            print(f"Error processing structure {idx}: {e}")
            results.append({
                'structure_idx': idx,
                'error': str(e)
            })

    # Organize results by sample index for separate file saving
    if num_sample_per_protein > 1:
        print(f"\nCompleted sampling: {len(indices)} structures × {num_sample_per_protein} samples = {len(results)} total samples")
        print("Results will be saved in separate files for each sample index.")

    return results

def parse_protein_list_from_file(file_path):
    """
    Parse protein names from a text file.

    Args:
        file_path: Path to file containing protein names (one per line)

    Returns:
        List of protein names
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Protein list file not found: {file_path}")

    protein_names = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):  # Skip empty lines and comments
                protein_names.append(line)

    print(f"Loaded {len(protein_names)} protein names from {file_path}")
    return protein_names


def parse_protein_indices_from_string(indices_str):
    """
    Parse protein indices from comma-separated string.

    Args:
        indices_str: Comma-separated string of indices (e.g., "0,5,10,15")

    Returns:
        List of integers
    """
    try:
        indices = [int(idx.strip()) for idx in indices_str.split(',')]
        print(f"Parsed {len(indices)} protein indices: {indices}")
        return indices
    except ValueError as e:
        raise ValueError(f"Invalid protein indices format: {indices_str}. Expected comma-separated integers.") from e


def parse_protein_names_from_string(names_str):
    """
    Parse protein names from comma-separated string.

    Args:
        names_str: Comma-separated string of names (e.g., "1a0o.A,1abc.B,1def.C")

    Returns:
        List of protein names
    """
    names = [name.strip() for name in names_str.split(',')]
    print(f"Parsed {len(names)} protein names: {names}")
    return names


def get_indices_for_protein_names(dataset, protein_names, verbose=False):
    """
    Get dataset indices for specific protein names.

    Args:
        dataset: CathDataset instance
        protein_names: List of protein names to find
        verbose: Whether to print detailed matching information

    Returns:
        Dict mapping protein names to their dataset indices
    """
    if verbose:
        print(f"Looking for {len(protein_names)} proteins in dataset of {len(dataset)} structures...")

    name_to_index = {}
    missing_proteins = []

    # Create a mapping from protein names to indices
    for idx in range(len(dataset)):
        try:
            data, _, _, _ = dataset[idx]
            protein_name = getattr(data, 'name', None)
            if protein_name:
                name_to_index[protein_name] = idx
                if verbose and protein_name in protein_names:
                    print(f"  Found {protein_name} at index {idx}")
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not load structure at index {idx}: {e}")
            continue

    # Find indices for requested proteins
    found_indices = {}
    for protein_name in protein_names:
        if protein_name in name_to_index:
            found_indices[protein_name] = name_to_index[protein_name]
        else:
            missing_proteins.append(protein_name)

    if missing_proteins:
        print(f"Warning: Could not find {len(missing_proteins)} proteins in dataset:")
        for protein in missing_proteins:
            print(f"  - {protein}")

        # Suggest similar names if available
        available_names = list(name_to_index.keys())
        if available_names:
            print(f"\nAvailable protein names (first 10): {available_names[:10]}")

    print(f"Successfully found {len(found_indices)} out of {len(protein_names)} requested proteins")
    return found_indices


def resolve_protein_sampling_mode(args, dataset):
    """
    Resolve which proteins to sample based on command line arguments.

    Args:
        args: Parsed command line arguments
        dataset: CathDataset instance

    Returns:
        Tuple of (indices_to_sample, sampling_mode_description)
    """
    # Priority order: protein_list > protein_names > protein_indices > protein_name > sample_all > structure_idx

    if args.protein_list:
        print(f"Using protein list from file: {args.protein_list}")
        protein_names = parse_protein_list_from_file(args.protein_list)
        name_to_index = get_indices_for_protein_names(dataset, protein_names, verbose=args.verbose)
        indices = list(name_to_index.values())
        return indices, f"protein list from {args.protein_list} ({len(indices)} proteins)"

    elif args.protein_names:
        print(f"Using protein names from command line")
        protein_names = parse_protein_names_from_string(args.protein_names)
        name_to_index = get_indices_for_protein_names(dataset, protein_names, verbose=args.verbose)
        indices = list(name_to_index.values())
        return indices, f"protein names from command line ({len(indices)} proteins)"

    elif args.protein_indices:
        print(f"Using protein indices from command line")
        indices = parse_protein_indices_from_string(args.protein_indices)
        # Validate indices
        max_idx = len(dataset) - 1
        invalid_indices = [idx for idx in indices if idx < 0 or idx > max_idx]
        if invalid_indices:
            raise ValueError(f"Invalid indices {invalid_indices}. Dataset has indices 0-{max_idx}")
        return indices, f"protein indices from command line ({len(indices)} proteins)"

    elif args.protein_name:
        print(f"Using single protein name: {args.protein_name}")
        name_to_index = get_indices_for_protein_names(dataset, [args.protein_name], verbose=args.verbose)
        if not name_to_index:
            raise ValueError(f"Protein '{args.protein_name}' not found in dataset")
        indices = list(name_to_index.values())
        return indices, f"single protein '{args.protein_name}'"

    elif args.sample_all:
        if args.max_structures:
            indices = list(range(min(args.max_structures, len(dataset))))
        else:
            indices = list(range(len(dataset)))
        return indices, f"all proteins in {args.split} split ({len(indices)} proteins)"

    else:
        # Default to single structure by index
        if args.structure_idx >= len(dataset):
            raise ValueError(f"Structure index {args.structure_idx} out of range (max: {len(dataset)-1})")
        return [args.structure_idx], f"single protein at index {args.structure_idx}"

def save_results_to_files(results, output_prefix, output_dir, model_name=None, split=None,
                          steps=None, T=None, save_probabilities=True):
    """
    Save sampling results to comprehensive output files.
    If multiple samples per protein are present, saves separate files for each sample index.

    Args:
        results: List of result dictionaries
        output_prefix: Prefix for output filenames
        output_dir: Output directory
        model_name: Name of the model used
        split: Dataset split used
        steps: Number of sampling steps
        T: Maximum time value
        save_probabilities: Write the per-residue probability NPZ. Turning this
            off (``--no_probabilities``) only skips the file; the probabilities
            are still computed and carried in ``results``, because sample
            selection and perplexity need them.

    Returns:
        Dictionary with file paths and metadata
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Check if we have multiple samples per protein
    sample_indices = set()
    for result in results:
        if 'sample_idx' in result:
            sample_indices.add(result['sample_idx'])

    has_multiple_samples = len(sample_indices) > 1 or (len(sample_indices) == 1 and 0 not in sample_indices)

    if has_multiple_samples:
        # Group results by sample index
        results_by_sample = {}
        for result in results:
            sample_idx = result.get('sample_idx', 0)
            if sample_idx not in results_by_sample:
                results_by_sample[sample_idx] = []
            results_by_sample[sample_idx].append(result)

        print(f"Multiple samples detected. Saving {len(results_by_sample)} separate sets of files...")

        # Save separate files for each sample index
        file_info = {
            'timestamp': timestamp,
            'sample_files': {}
        }

        for sample_idx in sorted(results_by_sample.keys()):
            sample_results = results_by_sample[sample_idx]
            sample_suffix = f"_sample{sample_idx+1}"

            sample_file_info = _save_single_sample_files(
                sample_results, output_prefix + sample_suffix, output_dir,
                timestamp, model_name, split, steps, T,
                save_probabilities=save_probabilities
            )

            file_info['sample_files'][sample_idx] = sample_file_info

            print(f"  Sample {sample_idx+1}: {len(sample_results)} structures")
            print(f"    Sequences: {sample_file_info['sequences_file']}")
            print(f"    Probabilities: {sample_file_info['probabilities_file']}")
            print(f"    Metadata: {sample_file_info['metadata_file']}")

        return file_info

    else:
        # Single sample per protein - use original logic
        return _save_single_sample_files(
            results, output_prefix, output_dir, timestamp, model_name, split, steps, T,
            save_probabilities=save_probabilities
        )


def get_blosum62_similarity(predicted_sequence: str, true_sequence: str) -> dict:
    """
    Compute BLOSUM62 sequence similarity metrics.  Thin wrapper around
    eval/structure_comparator.compute_blosum62_similarity that handles the
    one-time path setup so callers don't need to repeat it.

    Returns dict with keys seq_sim_blosum_frac and seq_sim_blosum_mean
    (both None if sequences are empty or Biopython is unavailable).
    """
    import sys, os as _os
    _eval_dir = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'eval')
    if _eval_dir not in sys.path:
        sys.path.insert(0, _eval_dir)
    try:
        from sequence_metrics import compute_blosum62_similarity
        return compute_blosum62_similarity(predicted_sequence, true_sequence)
    except Exception:
        return {'seq_sim_blosum_frac': None, 'seq_sim_blosum_mean': None}


def _save_single_sample_files(results, output_prefix, output_dir, timestamp, model_name=None,
                              split=None, steps=None, T=None, save_probabilities=True):
    """
    Save sampling results for a single sample set to comprehensive output files.

    Args:
        results: List of result dictionaries for a single sample
        output_prefix: Prefix for output filenames
        output_dir: Output directory
        timestamp: Timestamp string
        model_name: Name of the model used
        split: Dataset split used
        steps: Number of sampling steps
        T: Maximum time value

    Returns:
        Dictionary with file paths and metadata
    """
    # Generate filenames
    #sequences_file = os.path.join(output_dir, f"{timestamp}_{output_prefix}_sequences.csv")
    sequences_file = os.path.join(output_dir, f"{output_prefix}_sequences.csv")
    probabilities_file = os.path.join(output_dir, f"{timestamp}_{output_prefix}_probabilities.npz")
    metadata_file = os.path.join(output_dir, f"{timestamp}_{output_prefix}_metadata.txt")

    # 1. Save sequences as CSV
    print(f"  Saving sequences to: {sequences_file}")
    sequences_data = []
    for result in results:
        if 'error' not in result:
            # Convert predicted_sequence properly - check if it's indices or amino acids
            pred_seq = result.get('predicted_sequence', result.get('predicted_indices', []))

            # Convert numpy arrays to lists first
            if isinstance(pred_seq, np.ndarray):
                pred_seq = pred_seq.tolist()

            # If pred_seq is a list of indices, convert to amino acid string
            if isinstance(pred_seq, list) and pred_seq and isinstance(pred_seq[0], (int, np.integer)):
                # It's indices, convert to amino acid string
                aa_string = ''
                for idx in pred_seq:
                    idx = int(idx)  # Ensure it's a Python int, not numpy int
                    if 0 <= idx < len(IDX_TO_AA):
                        aa_string += THREE_TO_ONE[IDX_TO_AA[idx]]
                    else:
                        aa_string += 'X'
                pred_seq_final = aa_string
            else:
                # It's already a string
                pred_seq_final = str(pred_seq)

            # Handle true_sequence properly
            true_seq = result.get('true_sequence', '')
            if result.get('true_indices') and not true_seq:
                # Convert true_indices to amino acid string
                true_indices = result['true_indices']

                # Convert numpy arrays to lists
                if isinstance(true_indices, np.ndarray):
                    true_indices = true_indices.tolist()

                if true_indices:
                    true_aa_string = ''
                    for idx in true_indices:
                        idx = int(idx)  # Ensure it's a Python int, not numpy int
                        if 0 <= idx < len(IDX_TO_AA):
                            true_aa_string += THREE_TO_ONE[IDX_TO_AA[idx]]
                        else:
                            true_aa_string += 'X'
                    true_seq = true_aa_string

            # Clean structure_idx and length to ensure they're Python native types
            structure_idx = result['structure_idx']
            if isinstance(structure_idx, (np.integer, np.floating)):
                structure_idx = structure_idx.item()

            length = result['length']
            if isinstance(length, (np.integer, np.floating)):
                length = length.item()

            accuracy = result.get('accuracy', None)
            if accuracy is not None and isinstance(accuracy, (np.integer, np.floating)):
                accuracy = accuracy.item()

            # Compute BLOSUM62 sequence similarity
            _blosum = get_blosum62_similarity(pred_seq_final, true_seq)

            row = {
                'structure_idx': structure_idx,
                'structure_name': result.get('structure_name', f"structure_{structure_idx}"),
                'length': length,
                'predicted_sequence': pred_seq_final,
                'true_sequence': true_seq,
                'accuracy': accuracy,
                'seq_sim_blosum_frac': _blosum['seq_sim_blosum_frac'],
                'seq_sim_blosum_mean': _blosum['seq_sim_blosum_mean'],
            }

            # Add selection metadata if available
            if 'sequence_perplexity' in result:
                perplexity = result['sequence_perplexity']
                if isinstance(perplexity, (np.integer, np.floating)):
                    perplexity = perplexity.item()
                row['sequence_perplexity'] = perplexity

            if 'final_dssp_accuracy' in result:
                dssp_acc = result['final_dssp_accuracy']
                if isinstance(dssp_acc, (np.integer, np.floating)):
                    dssp_acc = dssp_acc.item()
                row['final_dssp_accuracy'] = dssp_acc

            if 'selected_reason' in result:
                row['selected_reason'] = result['selected_reason']

            sequences_data.append(row)

    if sequences_data:
        # Convert numpy types to native Python types to avoid pandas compatibility issues
        def clean_value(val):
            """Recursively clean numpy types from nested structures."""
            if val is None:
                return None
            elif isinstance(val, np.ndarray):
                return val.tolist()
            elif isinstance(val, (np.integer, np.floating)):
                return val.item()
            elif hasattr(val, 'item') and not isinstance(val, str):  # numpy scalar but not string
                try:
                    return val.item()
                except (AttributeError, ValueError):
                    return val
            elif isinstance(val, dict):
                # Convert dict to JSON string for CSV compatibility
                return str({str(k): clean_value(v) for k, v in val.items()})
            elif isinstance(val, (list, tuple)):
                # Check if it's a list of simple types or complex types
                if val and isinstance(val[0], (dict, np.ndarray)):
                    # Complex nested structure - convert to string
                    return str([clean_value(item) for item in val])
                else:
                    # Simple list - keep as list but clean elements
                    return [clean_value(item) for item in val]
            else:
                return val

        clean_data = []
        for item in sequences_data:
            clean_item = {}
            for key, value in item.items():
                clean_item[str(key)] = clean_value(value)
            clean_data.append(clean_item)

        # Write CSV manually to avoid pandas/numpy incompatibility issues
        # (pandas 2.2.2 + numpy 1.26.4 has a known bug)
        try:
            import csv
            with open(sequences_file, 'w', newline='') as f:
                if clean_data:
                    writer = csv.DictWriter(f, fieldnames=clean_data[0].keys())
                    writer.writeheader()
                    writer.writerows(clean_data)
        except Exception as e:
            print(f"ERROR writing CSV file. Debug info:")
            print(f"Exception: {type(e).__name__}: {e}")
            for i, item in enumerate(clean_data):
                print(f"\nItem {i}:")
                for key, value in item.items():
                    value_type = type(value)
                    value_repr = repr(value) if not isinstance(value, str) or len(str(value)) < 100 else repr(value)[:100] + '...'
                    print(f"  {key}: {value_type} = {value_repr}")
            raise

    # 2. Save probabilities as NPZ
    #
    # Select on the value, not on key presence: the sampling paths always carry
    # this key, and a result that failed part-way can carry it as None. Selecting
    # on presence used to hand None to np.argmax further down, which killed the
    # run *after* all the sampling work was done.
    successful_results = [r for r in results
                          if 'error' not in r and r.get('final_probabilities') is not None]
    if successful_results and save_probabilities:
        print(f"  Saving probabilities to: {probabilities_file}")
        prob_data = {}

        # Create amino acid index mapping
        prob_data['aa_index_to_name'] = np.array(IDX_TO_AA)

        # Save individual structure probabilities
        for result in successful_results:
            struct_name = result.get('structure_name', f"structure_{result['structure_idx']}")
            prob_data[f'probs_{struct_name}'] = result['final_probabilities']

        prob_data['structure_indices'] = np.array([r['structure_idx'] for r in successful_results])

        # LEGACY FORMAT: Add support for the old analysis format with struct_{index} keys
        for i, result in enumerate(successful_results):
            struct_idx = result['structure_idx']

            # Probabilities for this structure
            prob_data[f'struct_{struct_idx}_probabilities'] = result['final_probabilities']

            # True indices (ground truth as indices)
            if 'true_indices' in result and result['true_indices'] is not None:
                prob_data[f'struct_{struct_idx}_true_indices'] = np.array(result['true_indices'])
            elif 'true_sequence' in result and result['true_sequence']:
                # Convert true sequence string to indices
                true_seq = result['true_sequence']
                true_indices = []
                for aa_char in true_seq:
                    # Convert single letter to 3-letter, then to index
                    three_letter = SINGLE_TO_TRIPLE.get(aa_char.upper(), 'XXX')
                    idx = AA_TO_IDX.get(three_letter, 20)  # Default to 'XXX' (index 20)
                    true_indices.append(idx)
                prob_data[f'struct_{struct_idx}_true_indices'] = np.array(true_indices)
            else:
                # Default to all unknown if no ground truth available
                seq_len = result['final_probabilities'].shape[0]
                prob_data[f'struct_{struct_idx}_true_indices'] = np.full(seq_len, 20, dtype=int)  # All 'XXX'

            # Predicted indices (argmax of probabilities)
            predicted_indices = np.argmax(result['final_probabilities'], axis=1)
            prob_data[f'struct_{struct_idx}_predicted_indices'] = predicted_indices

        np.savez_compressed(probabilities_file, **prob_data)
    elif not save_probabilities:
        print("  Skipping the probabilities NPZ (--no_probabilities)")
        probabilities_file = None
    else:
        print("  No successful results with probabilities to save")
        probabilities_file = None

    # 3. Save metadata
    print(f"  Saving metadata to: {metadata_file}")
    with open(metadata_file, 'w') as f:
        f.write("PROTEIN SEQUENCE SAMPLING METADATA\n")
        f.write("="*50 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("SAMPLING PARAMETERS:\n")
        f.write(f"  Model: {model_name or 'Unknown'}\n")
        f.write(f"  Dataset split: {split or 'Unknown'}\n")
        f.write(f"  Sampling steps: {steps or 'Unknown'}\n")
        f.write(f"  Max time (T): {T or 'Unknown'}\n\n")

        f.write("RESULTS SUMMARY:\n")
        total_structures = len(results)
        successful = len([r for r in results if 'error' not in r])
        failed = total_structures - successful

        f.write(f"  Total structures: {total_structures}\n")
        f.write(f"  Successful: {successful}\n")
        f.write(f"  Failed: {failed}\n")

        if successful > 0:
            accuracies = [r.get('accuracy') for r in results if 'error' not in r and r.get('accuracy') is not None]
            if accuracies:
                avg_acc = np.mean(accuracies)
                f.write(f"  Average accuracy: {avg_acc:.2f}%\n")
                std_acc = np.std(accuracies)
                f.write(f"  STD accuracy: {std_acc:.2f}%\n")
                max_acc = np.max(accuracies)
                f.write(f"  Max accuracy: {max_acc:.2f}%\n")
                min_acc = np.min(accuracies)
                f.write(f"  Min accuracy: {min_acc:.2f}%\n")

            # BLOSUM62 sequence similarity summary
            blosum_fracs = [r.get('seq_sim_blosum_frac') for r in results if 'error' not in r and r.get('seq_sim_blosum_frac') is not None]
            blosum_means = [r.get('seq_sim_blosum_mean') for r in results if 'error' not in r and r.get('seq_sim_blosum_mean') is not None]
            if blosum_fracs:
                f.write(f"  Average seq_sim_blosum_frac: {np.mean(blosum_fracs):.4f}\n")
                f.write(f"  Average seq_sim_blosum_mean: {np.mean(blosum_means):.4f}\n")

    return {
        'sequences_file': sequences_file,
        'probabilities_file': probabilities_file,
        'metadata_file': metadata_file,
        'timestamp': timestamp
    }


def get_prediction_accuracy(file_name, timestep, pdb_id, return_pred=False, check='pred', output_dir='../output/prediction/'):
    """
    Analyze prediction accuracy from trajectory analysis files.

    Args:
        file_name: Name of the .npz file containing trajectory analysis
        timestep: Which timestep to analyze
        pdb_id: ID of the protein structure
        return_pred: Whether to return the predictions
        check: 'pred' for model predictions or 'state' for current states
        output_dir: Directory containing the prediction files

    Returns:
        accuracy, cross_entropy_loss[, predictions]
    """
    import os

    import numpy as np

    file_path = os.path.join(output_dir, file_name)
    if not os.path.exists(file_path):
        raise ValueError(f"File not found: {file_path}")

    probabilities = np.load(file_path, allow_pickle=True)

    if check == 'pred':
        key = f'{pdb_id}_model_predictions'
    elif check == 'state':
        key = f'{pdb_id}_current_states'
    else:
        raise ValueError("check must be either 'pred' or 'state'")

    try:
        model_predictions = probabilities[key][timestep]
    except KeyError:
        raise ValueError(f"Index {pdb_id} not found in the probabilities file. Available keys: {list(probabilities.keys())}")

    ground_truth = probabilities[f'{pdb_id}_ground_truth']

    # Calculate accuracy and categorical cross entropy
    # For accuracy, get the max index for each row in model_predictions
    model_predictions_index = np.argmax(model_predictions, axis=1)
    ground_truth_index = np.argmax(ground_truth, axis=1)

    # Calculate accuracy
    accuracy = np.mean(np.array(model_predictions_index) == np.array(ground_truth_index))

    # Manually calculate categorical cross entropy
    ce_loss = -(ground_truth * np.log(model_predictions + 1e-10)).mean(axis=1)

    if return_pred:
        return accuracy, ce_loss.mean(), model_predictions

    return accuracy, ce_loss.mean()


def parse_protein_list_from_file(file_path):
    """
    Parse protein names from a text file.

    Args:
        file_path: Path to file containing protein names (one per line)

    Returns:
        List of protein names
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Protein list file not found: {file_path}")

    protein_names = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):  # Skip empty lines and comments
                protein_names.append(line)

    print(f"Loaded {len(protein_names)} protein names from {file_path}")
    return protein_names


def parse_protein_indices_from_string(indices_str):
    """
    Parse protein indices from comma-separated string.

    Args:
        indices_str: Comma-separated string of indices (e.g., "0,5,10,15")

    Returns:
        List of integers
    """
    try:
        indices = [int(idx.strip()) for idx in indices_str.split(',')]
        print(f"Parsed {len(indices)} protein indices: {indices}")
        return indices
    except ValueError as e:
        raise ValueError(f"Invalid protein indices format: {indices_str}. Expected comma-separated integers.") from e


def parse_protein_names_from_string(names_str):
    """
    Parse protein names from comma-separated string.

    Args:
        names_str: Comma-separated string of names (e.g., "1a0o.A,1abc.B,1def.C")

    Returns:
        List of protein names
    """
    names = [name.strip() for name in names_str.split(',')]
    print(f"Parsed {len(names)} protein names: {names}")
    return names


def get_indices_for_protein_names(dataset, protein_names, verbose=False):
    """
    Get dataset indices for specific protein names.

    Args:
        dataset: CathDataset instance
        protein_names: List of protein names to find
        verbose: Whether to print detailed matching information

    Returns:
        Dict mapping protein names to their dataset indices
    """
    if verbose:
        print(f"Looking for {len(protein_names)} proteins in dataset of {len(dataset)} structures...")

    name_to_index = {}
    missing_proteins = []

    # Create a mapping from protein names to indices
    for idx in range(len(dataset)):
        try:
            data, _, _, _ = dataset[idx]
            protein_name = getattr(data, 'name', None)
            if protein_name:
                name_to_index[protein_name] = idx
                if verbose and protein_name in protein_names:
                    print(f"  Found {protein_name} at index {idx}")
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not load structure at index {idx}: {e}")
            continue

    # Find indices for requested proteins
    found_indices = {}
    for protein_name in protein_names:
        if protein_name in name_to_index:
            found_indices[protein_name] = name_to_index[protein_name]
        else:
            missing_proteins.append(protein_name)

    if missing_proteins:
        print(f"Warning: Could not find {len(missing_proteins)} proteins in dataset:")
        for protein in missing_proteins:
            print(f"  - {protein}")

        # Suggest similar names if available
        available_names = list(name_to_index.keys())
        if available_names:
            print(f"\nAvailable protein names (first 10): {available_names[:10]}")

    print(f"Successfully found {len(found_indices)} out of {len(protein_names)} requested proteins")
    return found_indices


def resolve_protein_sampling_mode(args, dataset):
    """
    Resolve which proteins to sample based on command line arguments.

    Args:
        args: Parsed command line arguments
        dataset: CathDataset instance

    Returns:
        Tuple of (indices_to_sample, sampling_mode_description)
    """
    # Priority order: protein_list > protein_names > protein_indices > protein_name > sample_all > structure_idx

    if args.protein_list:
        print(f"Using protein list from file: {args.protein_list}")
        protein_names = parse_protein_list_from_file(args.protein_list)
        name_to_index = get_indices_for_protein_names(dataset, protein_names, verbose=args.verbose)
        indices = list(name_to_index.values())
        return indices, f"protein list from {args.protein_list} ({len(indices)} proteins)"

    elif args.protein_names:
        print(f"Using protein names from command line")
        protein_names = parse_protein_names_from_string(args.protein_names)
        name_to_index = get_indices_for_protein_names(dataset, protein_names, verbose=args.verbose)
        indices = list(name_to_index.values())
        return indices, f"protein names from command line ({len(indices)} proteins)"

    elif args.protein_indices:
        print(f"Using protein indices from command line")
        indices = parse_protein_indices_from_string(args.protein_indices)
        # Validate indices
        max_idx = len(dataset) - 1
        invalid_indices = [idx for idx in indices if idx < 0 or idx > max_idx]
        if invalid_indices:
            raise ValueError(f"Invalid indices {invalid_indices}. Dataset has indices 0-{max_idx}")
        return indices, f"protein indices from command line ({len(indices)} proteins)"

    elif args.protein_name:
        print(f"Using single protein name: {args.protein_name}")
        name_to_index = get_indices_for_protein_names(dataset, [args.protein_name], verbose=args.verbose)
        if not name_to_index:
            raise ValueError(f"Protein '{args.protein_name}' not found in dataset")
        indices = list(name_to_index.values())
        return indices, f"single protein '{args.protein_name}'"

    elif args.sample_all:
        if args.max_structures:
            indices = list(range(min(args.max_structures, len(dataset))))
        else:
            indices = list(range(len(dataset)))
        return indices, f"all proteins in {args.split} split ({len(indices)} proteins)"

    else:
        # Default to single structure by index
        if args.structure_idx >= len(dataset):
            raise ValueError(f"Structure index {args.structure_idx} out of range (max: {len(dataset)-1})")
        return [args.structure_idx], f"single protein at index {args.structure_idx}"


def load_config_from_json(config_path):
    """
    Load configuration parameters from a JSON file.

    Args:
        config_path: Path to JSON configuration file

    Returns:
        Dictionary of configuration parameters
    """
    import json

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Normalize config keys to match argument names (remove spaces, convert to lowercase)
    normalized_config = {}
    for key, value in config.items():
        # Remove comments in parentheses and normalize key name
        clean_key = key.split('(')[0].strip().replace(' ', '_').lower()

        # Convert string booleans to actual booleans
        if isinstance(value, str):
            value_stripped = value.strip().lower()
            if value_stripped in ['true', 'false']:
                value = value_stripped == 'true'

        normalized_config[clean_key] = value

    return normalized_config


def apply_config_to_args(args, config):
    """
    Apply config file parameters to parsed arguments.
    Only updates arguments that were not explicitly provided on command line.

    Args:
        args: Parsed arguments from argparse
        config: Dictionary of config parameters from JSON file

    Returns:
        Modified args with config values applied
    """
    import sys

    # Get which arguments were explicitly provided on command line
    cli_args = set()
    for i, arg in enumerate(sys.argv):
        if arg.startswith('--'):
            cli_args.add(arg[2:].replace('-', '_'))

    # Apply config values only for arguments not provided on CLI
    print("setting the followings from the config:")
    for key, value in config.items():
        if key not in cli_args and hasattr(args, key):
            setattr(args, key, value)
            print(f"key {key} {value}")

    return args


def create_argument_parser():
    """Create and return the argument parser for protein sequence sampling."""
    parser = argparse.ArgumentParser(description="Generate protein sequences using trained DFM model")

    # Config file parameter (highest priority for loading defaults)
    parser.add_argument('--config_file', type=str, default=None,
                       help="Path to JSON config file containing default parameters. "
                            "Command-line arguments override config file values.")

    parser.add_argument('--model_path', type=str,
                       default='../ckpts/inverse_folddir_model.pt',
                       help="Path to the trained model checkpoint")

    # Direct PDB/structure input (bypasses dataset)
    parser.add_argument('--pdb_input', type=str, default=None,
                       help="Direct PDB input: local file path, PDB ID (e.g., '1abc'), or PDB ID with chain (e.g., '1fcd.C'). "
                            "Supports: '/path/to/file.pdb', '/path/to/file.cif', '1abc', '1fcd.C'. "
                            "When used, bypasses dataset loading and directly processes the specified structure.")

    parser.add_argument('--split_json', type=str,
                       default='../datasets/cath-4.2/chain_set_splits.json',
                       help="Path to dataset splits")
    parser.add_argument('--map_pkl', type=str,
                       default='../datasets/cath-4.2/chain_set_map_with_b_factors_dssp.pkl',
                       help="Path to dataset mapping")
    parser.add_argument('--structure_idx', type=int, default=0,
                       help="Index of structure to use for sampling (ignored if other protein selection options are used)")
    parser.add_argument('--protein_name', type=str, default=None,
                       help="Name of specific protein to sample (e.g., '1a0o.A'). If provided, overrides --structure_idx")
    parser.add_argument('--protein_list', type=str, default=None,
                       help="Path to a file containing list of protein names (one per line) to sample")
    parser.add_argument('--protein_names', type=str, default=None,
                       help="Comma-separated list of protein names to sample (e.g., '1a0o.A,1abc.B,1def.C')")
    parser.add_argument('--protein_indices', type=str, default=None,
                       help="Comma-separated list of dataset indices to sample (e.g., '0,5,10,15')")
    parser.add_argument('--sample_all', action='store_true',
                       help="Sample sequences for all structures in the dataset split")
    parser.add_argument('--max_structures', type=int, default=None,
                       help="Maximum number of structures to sample (if --sample_all is used)")
    parser.add_argument('--output_prefix', type=str, default=None,
                       help="Prefix for output files (defaults to 'protein_sampling_trajectory' for trajectory analysis when <4 proteins)")
    parser.add_argument('--output_dir', type=str, default='output/prediction/',
                       help="Directory to save output files - trajectory analysis NPZ automatically generated for <4 proteins (default: ../output/prediction/)")
    parser.add_argument('--save_probabilities', action='store_true', default=True,
                       help="Save raw probability distributions (default: True)")
    parser.add_argument('--no_probabilities', action='store_true',
                       help="Don't save raw probabilities (faster, less disk space)")
    parser.add_argument('--detailed_json', action='store_true',
                       help="Generate detailed JSON output with time-step information (automatically enabled for <4 proteins)")
    parser.add_argument('--split', type=str, default='validation',
                       choices=['train', 'validation', 'test'],
                       help="Which dataset split to use")
    parser.add_argument('--flow_temp', type=float, default=1.0,
                       help="Temperature for flow sampling (default: 1.0). When --recycled_sampling is used, this is the round-2 (refinement) temperature.")
    parser.add_argument('--time_as_temperature', action='store_true',
                       help="Use time-dependent temperature: flow_temp * (1 + time_temp_scale * (T - t) / T). Starts at flow_temp*(1+time_temp_scale) and decreases to flow_temp at t=T. Requires --time_temp_scale.")
    parser.add_argument('--time_temp_scale', type=float, default=None,
                       help="Scale factor for time-dependent temperature schedule. Required when --time_as_temperature is set. "
                            "flow_temp_t = flow_temp * (1 + time_temp_scale * (T - t) / T). "
                            "E.g., time_temp_scale=1.0 doubles temperature at t=0; time_temp_scale=9.0 gives 10x at t=0.")
    parser.add_argument('--return_last_pred', action='store_true',
                       help="Decode the final sequence from the last model forward pass prediction rather than the final simplex state")
    parser.add_argument('--recycled_sampling', action='store_true',
                       help="Enable two-round warm-start recycling: round 1 uses --flow_temp_round1 (warm/exploratory), "
                            "round 2 initializes from round-1 x_T and uses --flow_temp (sharp/refinement).")
    parser.add_argument('--flow_temp_round1', type=float, default=None,
                       help="Temperature for round-1 of recycled sampling (warm/exploratory). Required when --recycled_sampling is set.")
    parser.add_argument('--steps', type=int, default=20,
                       help="Number of sampling steps (only used for Euler integration)")
    parser.add_argument('--T', type=float, default=8.0,
                       help="Maximum time (noise level)")
    parser.add_argument('--t_min', type=float, default=0.0,
                       help="Minimum time (initial noise level, default: 0.0)")

    # Integration method selection
    parser.add_argument('--integration_method', type=str, default='euler',
                       choices=['euler'],
                       help="Integration method: 'euler' for fixed-step Euler (default: euler)")
    parser.add_argument('--rtol', type=float, default=1e-5,
                       help="Relative tolerance for RK45 integration (default: 1e-5)")
    parser.add_argument('--atol', type=float, default=1e-8,
                       help="Absolute tolerance for RK45 integration (default: 1e-8)")

    # Sampling parameters
    parser.add_argument('--dirichlet_concentration', type=float, default=20.0,
                       help="Concentration parameter for the initial Dirichlet distribution (default: 20.0). "
                            "This was previously declared as 1.0 while the help text, every internal "
                            "fallback, and inpainting.py all used 20.0 -- so a bare sample.py run started "
                            "from a different noise distribution than every other entry point.")
    parser.add_argument('--dssp_initialization', action='store_true',
                       help="Initialize Dirichlet noise conditioned on target DSSP (only when ensemble_size=1)")
    parser.add_argument('--dssp_initialization_first_pos_only', action='store_true',
                       help="Apply DSSP initialization only to the first residue position (requires --dssp_initialization)")
    parser.add_argument('--dssp_guidance', action='store_true',
                       help="Enable DSSP-guided AA distribution blending during sampling. "
                            "When predicted DSSP doesn't match target DSSP, blends model AA distribution "
                            "with target DSSP-conditioned AA distribution. Only works when ensemble_size=1. "
                            "(default: False)")
    parser.add_argument('--dssp_blending', type=str, default='geometric',
                       choices=['geometric', 'arithmetic'],
                       help="Blending method for DSSP guidance: 'geometric' uses (p_model^lambda * p_target^(1-lambda)), "
                            "'arithmetic' uses (lambda * p_model + (1-lambda) * p_target). "
                            "Only used when --dssp_guidance is enabled. (default: geometric)")
    parser.add_argument('--dssp_lambda_floor', type=float, default=0.2,
                       help="Minimum lambda value (confidence floor) for DSSP guidance. Ensures model always has at least "
                            "this fraction of influence even when DSSP prediction is very wrong. "
                            "Range: [0.0, 1.0]. Higher values = more model influence. "
                            "Only used when --dssp_guidance is enabled. (default: 0.2)")
    parser.add_argument('--dssp_annealing_schedule', type=str, default='quadratic',
                       choices=['linear', 'quadratic', 'cubic'],
                       help="Temporal annealing schedule for DSSP guidance. Controls how lambda increases toward 1.0 "
                            "over sampling steps. 'linear' = steady increase, 'quadratic' = slow start then rapid, "
                            "'cubic' = very slow start then very rapid. "
                            "Only used when --dssp_guidance is enabled. (default: quadratic)")
    parser.add_argument('--num_sample_per_protein', type=int, default=1,
                       help="Number of sequences to sample per protein structure with different noise realizations (default: 1)")
    parser.add_argument('--pick_based_on_dssp', action='store_true',
                       help="When num_sample_per_protein > 1, select the best sample based on final DSSP accuracy "
                            "(with perplexity tie-breaker). Requires DSSP targets. Mutually exclusive with --pick_based_on_perplexity.")
    parser.add_argument('--pick_based_on_perplexity', action='store_true',
                       help="When num_sample_per_protein > 1, select the best sample based on sequence perplexity "
                            "(lower is better, with DSSP tie-breaker if available). Mutually exclusive with --pick_based_on_dssp.")
    parser.add_argument('--filter_out_missing_flanks', action='store_true',
                       help="Remove residues with missing coordinates from start and end before sampling. "
                            "Preserves residues in the middle with missing coordinates. Output includes offset metadata "
                            "for mapping predictions back to original numbering. Only affects sampling, not training.")
    parser.add_argument('--remove_missing_positions', action='store_true',
                       help="After sampling, remove positions from the generated sequence where the input residue "
                            "coordinates were missing (data.geom_missing == True). The resulting sequence will be "
                            "shorter. The output includes a list of retained position indices for mapping back to "
                            "the original numbering.")
    parser.add_argument('--auto_config', action='store_true',
                       help="Automatically extract model configuration from checkpoint")
    parser.add_argument('--verbose', action='store_true',
                       help="Enable verbose debugging output")

    # Graph building parameters - Auto-extracted from model checkpoint when possible
    parser.add_argument('--k_neighbors', type=int, default=None,
                       help="Number of nearest neighbors per node in graph construction (auto-extracted from model if not provided)")
    parser.add_argument('--k_farthest', type=int, default=None,
                       help="Number of farthest neighbors per node in graph construction (auto-extracted from model if not provided)")
    parser.add_argument('--k_random', type=int, default=None,
                       help="Number of random neighbors per node in graph construction (auto-extracted from model if not provided)")
    parser.add_argument('--max_edge_dist', type=float, default=None,
                       help="Maximum distance cutoff (Angstroms) for edge creation. Overrides k_neighbors, k_farthest, k_random. Max 80 neighbors per node. (auto-extracted from model if not provided)")
    parser.add_argument('--num_rbf_3d', type=int, default=None,
                       help="Number of RBF features for 3D distances in graph construction (auto-extracted from model if not provided)")
    parser.add_argument('--num_rbf_seq', type=int, default=None,
                       help="Number of RBF features for sequence distances in graph construction (auto-extracted from model if not provided)")
    parser.add_argument('--use_virtual_node', action='store_true',
                       help="Enable virtual node connectivity (auto-extracted from model if not provided)")
    parser.add_argument('--no_virtual_node', action='store_true',
                       help="Disable virtual node connectivity (overrides model setting)")

    # RBF distance range parameters
    parser.add_argument('--rbf_3d_min', type=float, default=None,
                       help="Minimum distance for 3D RBF features (auto-extracted from model if not provided, default: 2.0)")
    parser.add_argument('--rbf_3d_max', type=float, default=None,
                       help="Maximum distance for 3D RBF features (auto-extracted from model if not provided, default: 350.0)")
    parser.add_argument('--rbf_3d_spacing', type=str, default=None,
                       choices=['exponential', 'linear', 'log'],
                       help="Spacing method for 3D RBF features (auto-extracted from model if not provided, default: exponential)")

    # Velocity function control parameters
    parser.add_argument('--use_c_factor', action='store_true',
                       help="Enable c_factor calculation in velocity function (default: False, sets c_factor=1.0)")
    parser.add_argument('--use_smoothed_targets', action='store_true', default=False,
                       help="Use smoothed targets in velocity computation (default: False). Automatically set to True if --use_smoothed_labels is present.")

    # Structure noise parameters for sampling
    parser.add_argument('--structure_noise_mag_std', type=float, default=None,
                       help="Standard deviation for Gaussian noise added to atom coordinates during sampling. If not provided, uses checkpoint value. Set to 0.0 to disable noise.")
    parser.add_argument('--time_based_struct_noise', type=str, default=None,
                       choices=['increasing', 'decreasing', 'fixed'],
                       help="Time-based structure noise scaling during sampling: 'increasing', 'decreasing', or 'fixed'. If not provided, uses checkpoint value.")
    parser.add_argument('--uncertainty_struct_noise_scaling', action='store_true', default=False,
                       help="Scale structure noise based on uncertainty: more flexible parts get more noise. If not provided, uses checkpoint value.")

    # Auxiliary head re-ranking parameters
    parser.add_argument('--aux_head_ranking', action='store_true',
                       help="Rank replicates by auxiliary head self-consistency score: how well the sampled sequence "
                            "agrees with what the electrostatic and geometry heads predict from the structure. "
                            "Requires the model to have been trained with --use_electrostatic_loss and --use_geom_topology_loss. "
                            "Silently does nothing when num_sample_per_protein=1. Raises an error if heads are absent.")
    parser.add_argument('--aux_ranking_weight_electrostatic', type=float, default=None,
                       help="Weight for electrostatic head cross-entropy in the auxiliary ranking score. "
                            "Required when --aux_head_ranking is set. Mirrors training weight (e.g. 0.4).")
    parser.add_argument('--aux_ranking_weight_geometry', type=float, default=None,
                       help="Weight for geometry/topology head cross-entropy in the auxiliary ranking score. "
                            "Required when --aux_head_ranking is set. Mirrors training weight (e.g. 0.1).")

    # Recycled sampling with selective position clipping
    parser.add_argument('--recycle_clip_criterion', type=str, default=None,
                       choices=['state_max_prob_percentile', 'state_max_prob_absolute',
                                'flip_ever', 'aux_consistency', 'replicate_consensus'],
                       help="Confidence criterion for recycled sampling with position clipping. "
                            "Requires --recycled_sampling. Confident positions are fixed to one-hot "
                            "(argmax of final x_T); uncertain positions are re-initialized with fresh "
                            "Dirichlet noise for round 2. "
                            "Choices: state_max_prob_percentile (top fix_fraction by max(x_T)), "
                            "state_max_prob_absolute (max(x_T) > fix_threshold), "
                            "flip_ever (prediction argmax never changed during trajectory), "
                            "aux_consistency (aux head argmax agrees with x_T argmax AA class), "
                            "replicate_consensus (all replicates agree on argmax(x_T); requires num_sample_per_protein > 1).")
    parser.add_argument('--recycle_fix_fraction', type=float, default=0.80,
                       help="Fraction of positions to fix when using recycle_clip_criterion "
                            "[state_max_prob_percentile | flip_ever]. For flip_ever, acts as an "
                            "upper bound: if fewer positions qualify, only those are fixed. "
                            "Not used by aux_consistency or replicate_consensus. Range: [0.0, 1.0]. "
                            "Default: 0.80.")
    parser.add_argument('--recycle_fix_threshold', type=float, default=None,
                       help="Absolute max-probability threshold for recycle_clip_criterion="
                            "state_max_prob_absolute. Positions where max(x_T) > threshold are fixed. "
                            "Only valid with state_max_prob_absolute criterion.")

    # Ensemble sampling parameters
    parser.add_argument('--ensemble_size', type=int, default=1,
                       help="Number of structurally noised replicas to create (default: 1)")
    parser.add_argument('--ensemble_consensus_strength', type=float, default=0.2,
                       help="State consensus strength: 0=independent, 1=full consensus (default: 0.2)")
    parser.add_argument('--ensemble_method', type=str, default='geometric', choices=['arithmetic', 'geometric'],
                       help="Ensemble consensus method: arithmetic=mean in probability space, geometric=mean in log space (default: arithmetic)")

    # Missing distributed arguments that orchestrator tries to pass
    parser.add_argument('--distributed', action='store_true',
                       help="Enable distributed sampling")
    parser.add_argument('--batch_size', type=int, default=None,
                       help="Batch size for sampling")
    parser.add_argument('--force_batch', action='store_true',
                       help="Force batched sampling for multiple proteins (skip individual sampling phase)")
    parser.add_argument('--threads_per_gpu', type=int, default=2,
                       help="Number of threads per GPU for distributed sampling")

    # Trajectory recording (compact, analysis-oriented)
    parser.add_argument('--save_trajectory_npz', action='store_true',
                       help="Record a compact per-step denoising trajectory for every sampled protein "
                            "(model-prediction and simplex-state argmax, max probability, top1-top2 gap "
                            "and entropy) and write it to NPZ. Only supported by batched sampling "
                            "(--batch_size > 1 or --force_batch). Does not alter the sampling math.")
    parser.add_argument('--trajectory_dir', type=str, default=None,
                       help="Directory for --save_trajectory_npz output (defaults to --output_dir).")
    parser.add_argument('--seed', type=int, default=None,
                       help="Random seed for the initial Dirichlet noise (and any other sampling "
                            "randomness). Fixing it makes runs with different sampling settings share "
                            "identical initial conditions, so they can be compared as paired samples.")

    return parser


def process_sampling_args(args):
    """
    Post-process parsed arguments to handle interdependent logic.

    Args:
        args: Parsed arguments from create_argument_parser()

    Returns:
        args: Modified arguments with resolved dependencies
    """
    # Seed every RNG that sampling touches. With a fixed seed the protein order and batch
    # composition are deterministic, so the Dirichlet initialisations drawn for run A and
    # run B are identical position-for-position and the two runs are paired.
    seed = getattr(args, 'seed', None)
    if seed is not None:
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"Random seed set to {seed}")

    # If use_smoothed_labels is True, automatically set use_smoothed_targets to True
    if getattr(args, 'use_smoothed_labels', False):
        args.use_smoothed_targets = True
        print("Setting use_smoothed_targets=True because --use_smoothed_labels was provided")
    elif not hasattr(args, 'use_smoothed_targets'):
        args.use_smoothed_targets = False

    # Validate time_as_temperature requires time_temp_scale
    if getattr(args, 'time_as_temperature', False):
        if getattr(args, 'time_temp_scale', None) is None:
            raise ValueError(
                "--time_as_temperature requires --time_temp_scale to be set. "
                "Example: --time_temp_scale 1.0 doubles temperature at t=0, "
                "--time_temp_scale 9.0 gives 10x temperature at t=0."
            )

    # Validate recycled_sampling requires flow_temp_round1
    if getattr(args, 'recycled_sampling', False):
        if getattr(args, 'flow_temp_round1', None) is None:
            raise ValueError(
                "--recycled_sampling requires --flow_temp_round1 to be set. "
                "This is the (warm) temperature for round 1. --flow_temp is used for round 2 (refinement)."
            )

    # Validate recycle_clip_criterion dependencies
    clip_criterion = getattr(args, 'recycle_clip_criterion', None)
    if clip_criterion is not None:
        if not getattr(args, 'recycled_sampling', False):
            raise ValueError(
                "--recycle_clip_criterion requires --recycled_sampling to be set."
            )
        if clip_criterion == 'state_max_prob_absolute':
            if getattr(args, 'recycle_fix_threshold', None) is None:
                raise ValueError(
                    "--recycle_clip_criterion=state_max_prob_absolute requires "
                    "--recycle_fix_threshold to be set (e.g. --recycle_fix_threshold 0.27)."
                )
        if clip_criterion == 'replicate_consensus':
            if getattr(args, 'num_sample_per_protein', 1) <= 1:
                raise ValueError(
                    "--recycle_clip_criterion=replicate_consensus requires "
                    "--num_sample_per_protein > 1 to have multiple replicates for consensus."
                )
        if clip_criterion in ('aux_consistency', 'replicate_consensus'):
            if getattr(args, 'recycle_fix_fraction', None) is not None and getattr(args, 'recycle_fix_fraction', 0.80) != 0.80:
                import warnings
                warnings.warn(
                    f"--recycle_fix_fraction has no effect with --recycle_clip_criterion={clip_criterion}. "
                    f"It will be ignored.",
                    UserWarning
                )
        if getattr(args, 'recycle_fix_threshold', None) is not None and clip_criterion != 'state_max_prob_absolute':
            raise ValueError(
                "--recycle_fix_threshold is only valid with "
                "--recycle_clip_criterion=state_max_prob_absolute."
            )

    # Validate aux_head_ranking requires both weights
    if getattr(args, 'aux_head_ranking', False):
        if getattr(args, 'aux_ranking_weight_electrostatic', None) is None:
            raise ValueError(
                "--aux_head_ranking requires --aux_ranking_weight_electrostatic to be set. "
                "Recommended value: 0.4 (mirrors training weight)."
            )
        if getattr(args, 'aux_ranking_weight_geometry', None) is None:
            raise ValueError(
                "--aux_head_ranking requires --aux_ranking_weight_geometry to be set. "
                "Recommended value: 0.1 (mirrors training weight)."
            )

    return args


def _summarize_simplex_step(probs):
    """
    Reduce a [N, K] probability matrix to the four per-position scalars the trajectory
    analysis needs, so full [T, N, K] tensors never have to be stored.

    Args:
        probs: [N, K] tensor of probabilities over amino acid classes (rows sum to 1).

    Returns:
        dict with numpy arrays:
            'argmax'  [N] int8    - index of the most likely amino acid
            'maxprob' [N] float16 - probability of that amino acid
            'top2gap' [N] float16 - most likely minus second most likely probability
            'entropy' [N] float16 - Shannon entropy of the distribution (nats)
    """
    top2 = probs.topk(2, dim=-1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-10))).sum(-1)
    return {
        'argmax': top2.indices[:, 0].to(torch.int8).cpu().numpy(),
        'maxprob': top2.values[:, 0].to(torch.float16).cpu().numpy(),
        'top2gap': (top2.values[:, 0] - top2.values[:, 1]).to(torch.float16).cpu().numpy(),
        'entropy': entropy.to(torch.float16).cpu().numpy(),
    }


def sample_multiple_proteins_batched(model, dataset, indices=None, steps=50, T=8.0, t_min=0.0, K=21,
                                   save_probabilities=True, integration_method='euler', batch_size=4, args=None):
    """
    Sample sequences for multiple protein structures using proper batched processing.

    This function uses the same exact sampling logic as sample_chain but processes
    multiple proteins simultaneously for improved GPU utilization.

    Args:
        model: Trained DFM model
        dataset: CathDataset instance
        indices: List of structure indices to sample (None = all)
        steps: Number of sampling steps
        T: Maximum time
        t_min: Minimum time
        K: Number of amino acid classes
        save_probabilities: Whether to save probability distributions
        integration_method: 'euler' or 'rk45'
        batch_size: Number of proteins to process simultaneously
        args: Arguments object (optional)

    Returns:
        List of result dictionaries with evaluation metrics

    When args.save_trajectory_npz is set, each result also carries a 'trajectory' entry
    holding compact per-step arrays (see _summarize_simplex_step). Recording is read-only
    with respect to the sampling state: it never writes back into x_batch or predicted_batch.
    """
    import torch.nn.functional as F
    from torch.distributions import Dirichlet

    from training.collate import collate_fn

    if indices is None:
        indices = list(range(len(dataset)))

    device = next(model.parameters()).device
    model.eval()
    results = []

    record_traj = getattr(args, 'save_trajectory_npz', False) if args else False

    print(f"Batched sampling of {len(indices)} structures using {integration_method} integration...")
    print(f"Batch size: {batch_size}, Steps: {steps}, T: {T}")

    # Process proteins in batches
    for batch_start in range(0, len(indices), batch_size):
        batch_end = min(batch_start + batch_size, len(indices))
        batch_indices = indices[batch_start:batch_end]
        current_batch_size = len(batch_indices)

        print(f"Processing batch {batch_start//batch_size + 1}/{(len(indices) + batch_size - 1)//batch_size}: "
              f"structures {batch_start+1}-{batch_end} (indices {batch_indices})")

        try:
            # Load batch of proteins
            batch_data = []
            batch_y_true = []
            batch_info = []
            batch_use_virtual = []
            batch_dssp_targets = []

            # Check if flank filtering is enabled
            filter_flanks = getattr(args, 'filter_out_missing_flanks', False) if args else False

            for idx in batch_indices:
                data, y_true, mask, time_value, dssp_targets = dataset[idx]  # Unpack 5 values (includes DSSP)
                structure_name = getattr(data, 'name', f'structure_{idx}')
                use_virtual_node = getattr(data, 'use_virtual_node', False)

                # Apply flank filtering if requested
                flank_metadata = None
                if filter_flanks:
                    # Get original entry from dataset
                    if hasattr(dataset, 'entries'):
                        entry = dataset.entries[idx]
                    elif hasattr(dataset, 'protein_entry'):
                        entry = dataset.protein_entry
                    else:
                        print(f"Warning: Cannot filter flanks for {structure_name} - unknown dataset type")
                        entry = None

                    if entry is not None:
                        try:
                            from data.graph_builder import filter_missing_flanks
                            from data.shared_processing import create_target_tensor_and_mask

                            # Filter flanking residues with missing coordinates
                            filtered_entry, start_offset, end_offset = filter_missing_flanks(
                                entry, verbose=False
                            )

                            # Check if filtering actually removed residues
                            if start_offset > 0 or end_offset < len(entry['seq']) - 1:
                                # Rebuild graph with filtered entry
                                data_new = dataset.graph_builder.build_from_dict(
                                    filtered_entry, time_param=time_value, af_filter_mode=False
                                )

                                # Preserve important attributes from original data
                                data_new.name = structure_name
                                data_new.source = getattr(data, 'source', 'unknown')

                                data = data_new

                                # Rebuild targets with filtered entry
                                y_true, mask, dssp_targets = create_target_tensor_and_mask(
                                    data, filtered_entry, strict_validation=True
                                )

                                # Store metadata for output mapping
                                flank_metadata = {
                                    'applied': True,
                                    'original_length': len(entry['seq']),
                                    'filtered_length': len(filtered_entry['seq']),
                                    'start_offset': start_offset,
                                    'end_offset': end_offset
                                }

                                if args and getattr(args, 'verbose', False):
                                    print(f"  [{structure_name}] Filtered: {len(entry['seq'])} → {len(filtered_entry['seq'])} residues")

                        except Exception as e:
                            print(f"Warning: Flank filtering failed for {structure_name}: {e}")

                # Create dummy targets for collate_fn (same as sample_chain)
                dummy_y = torch.zeros(1, K)
                dummy_mask = torch.ones(1, dtype=torch.bool)
                dummy_time = torch.tensor(0.0)

                batch_data.append((data, dummy_y, dummy_mask, dummy_time))
                batch_y_true.append(y_true)
                batch_info.append({'idx': idx, 'name': structure_name, 'flank_metadata': flank_metadata})
                batch_use_virtual.append(use_virtual_node)
                batch_dssp_targets.append(dssp_targets)

            # Collate batch
            batched_data, y_pad, mask_pad, time_batch = collate_fn(batch_data)
            batched_data = batched_data.to(device)

            # Handle virtual nodes and get protein sizes (like sample_chain)
            protein_sizes = []
            protein_starts = []
            node_offset = 0

            for i in range(current_batch_size):
                # Find where this protein's nodes are in the batch
                protein_mask = (batched_data.batch == i)
                protein_node_count = protein_mask.sum().item()

                # Handle virtual nodes like in sample_chain
                use_virtual_node = batch_use_virtual[i]
                if use_virtual_node:
                    actual_size = protein_node_count - 1  # Exclude virtual node
                else:
                    actual_size = protein_node_count

                protein_sizes.append(actual_size)
                protein_starts.append(node_offset)
                node_offset += actual_size

            # Initialize sequences using Dirichlet noise
            dirichlet_concentration = args.dirichlet_concentration if args else 1
            ensemble_size = getattr(args, 'ensemble_size', 1) if args else 1
            use_dssp_init = getattr(args, 'dssp_initialization', False) if args else False

            # Check if we should use DSSP-conditioned initialization
            # Only when ensemble_size==1, flag is set, and DSSP targets are available
            if use_dssp_init and ensemble_size == 1:
                from data.dssp_constants import DSSP_UNKNOWN_IDX

                # Create initial sequences with DSSP conditioning for each protein
                x_parts = []
                for b in range(current_batch_size):
                    size = protein_sizes[b]
                    dssp_tgt = batch_dssp_targets[b]

                    if dssp_tgt is not None:
                        # Prepare DSSP targets for this protein
                        dssp_tensor = dssp_tgt.to(device) if isinstance(dssp_tgt, torch.Tensor) else torch.tensor(dssp_tgt, device=device)

                        # Handle size mismatch
                        if len(dssp_tensor) > size:
                            dssp_tensor = dssp_tensor[:size]
                        elif len(dssp_tensor) < size:
                            padding = torch.full((size - len(dssp_tensor),), DSSP_UNKNOWN_IDX, device=device, dtype=dssp_tensor.dtype)
                            dssp_tensor = torch.cat([dssp_tensor, padding])

                        # Create position-specific alphas based on DSSP targets
                        alphas = create_dssp_conditioned_alphas(
                            dssp_tensor, size, K, device, dirichlet_concentration, verbose=False
                        )

                        # Sample from position-specific Dirichlet distributions
                        x_protein = torch.zeros((1, size, K), device=device, dtype=torch.float32)
                        for pos in range(size):
                            dirichlet_dist = Dirichlet(alphas[pos])
                            x_protein[0, pos] = dirichlet_dist.sample()

                        x_parts.append(x_protein)
                    else:
                        # No DSSP targets for this protein, use uniform initialization
                        print(f"Warning: DSSP initialization requested but protein {b} has no DSSP targets. Using uniform initialization.")
                        dirichlet_dist = Dirichlet(dirichlet_concentration * torch.ones(K, device=device))
                        x_protein = dirichlet_dist.sample((1, size))
                        x_parts.append(x_protein)

                print(f"[DSSP INIT] Initialized {current_batch_size} proteins with DSSP-conditioned Dirichlet noise")
            else:
                # Standard uniform Dirichlet initialization
                if use_dssp_init and ensemble_size != 1:
                    print(f"Warning: --dssp_initialization requested but ensemble_size={ensemble_size} != 1. Using uniform initialization.")

                dirichlet_dist = Dirichlet(dirichlet_concentration * torch.ones(K, device=device))

                # Create initial sequences for all real nodes
                x_parts = []
                for size in protein_sizes:
                    x_protein = dirichlet_dist.sample((1, size))  # [1, N_protein, K]
                    x_parts.append(x_protein)

            # Concatenate to create batch tensor [batch_size, max_N, K] with padding
            max_size = max(protein_sizes)
            x_batch = torch.zeros(current_batch_size, max_size, K, device=device)

            for i, x_protein in enumerate(x_parts):
                size = protein_sizes[i]
                x_batch[i, :size, :] = x_protein[0, :size, :]

            # Time integration (same as sample_chain)
            times = torch.linspace(t_min, T, steps, device=device)
            dt = (T - t_min) / (steps - 1) if steps > 1 else 0

            # Print c_factor status at the start
            use_c_factor = getattr(args, 'use_c_factor', False) if args else False
            print(f"[SAMPLING] use_c_factor: {use_c_factor}", flush=True)

            # Per-protein trajectory buffers: traj_steps[b] is a list of per-step dicts.
            # The step loop breaks before the final time point, so the prediction lists end
            # up with (steps - 1) entries; the state lists get one extra entry appended
            # after the loop for the final x_T, giving `steps` entries.
            if record_traj:
                traj_pred = [[] for _ in range(current_batch_size)]
                traj_state = [[] for _ in range(current_batch_size)]

            # Check if we should track DSSP accuracy (only if ensemble_size == 1 and dssp_targets provided)
            track_dssp = getattr(args, 'ensemble_size', 1) == 1 and any(dt is not None for dt in batch_dssp_targets)
            if track_dssp:
                # Import DSSP constants
                from data.dssp_constants import DSSP_TO_IDX, IDX_TO_DSSP, NUM_DSSP_CLASSES, DSSP_UNKNOWN_IDX

                # Prepare DSSP targets for each protein in batch
                batch_dssp_tensors = []
                batch_dssp_masks = []
                for b in range(current_batch_size):
                    dssp_tgt = batch_dssp_targets[b]
                    size = protein_sizes[b]

                    if dssp_tgt is not None:
                        # Convert to tensor
                        dssp_tensor = dssp_tgt.to(device) if isinstance(dssp_tgt, torch.Tensor) else torch.tensor(dssp_tgt, device=device)

                        # Handle size mismatch
                        if len(dssp_tensor) > size:
                            dssp_tensor = dssp_tensor[:size]
                        elif len(dssp_tensor) < size:
                            padding = torch.full((size - len(dssp_tensor),), DSSP_UNKNOWN_IDX, device=device, dtype=dssp_tensor.dtype)
                            dssp_tensor = torch.cat([dssp_tensor, padding])

                        # Create mask for valid positions
                        valid_mask = dssp_tensor != DSSP_UNKNOWN_IDX
                    else:
                        # No DSSP data for this protein
                        dssp_tensor = torch.full((size,), DSSP_UNKNOWN_IDX, device=device, dtype=torch.long)
                        valid_mask = torch.zeros(size, device=device, dtype=torch.bool)

                    batch_dssp_tensors.append(dssp_tensor)
                    batch_dssp_masks.append(valid_mask)

            with torch.no_grad():
                time_steps = tqdm(enumerate(times), total=len(times), desc=f"Batched sampling ({current_batch_size} proteins)")
                for i, t_val in time_steps:
                    if i == len(times) - 1:  # Skip last step
                        break

                    t = torch.full((current_batch_size,), t_val, device=device)

                    # Prepare sequence input exactly like sample_chain
                    # The model expects only real node probabilities with shape [batch_size, max_real_nodes, K]
                    # The model will handle virtual nodes internally

                    # Create padded tensor for real nodes only
                    max_real_size = max(protein_sizes)
                    x_model_input = torch.zeros(current_batch_size, max_real_size, K, device=device)

                    for b in range(current_batch_size):
                        real_size = protein_sizes[b]
                        x_model_input[b, :real_size, :] = x_batch[b, :real_size, :]

                    # Get position predictions from model (same as sample_chain)
                    model_output = model(batched_data, t, x_model_input)

                    # Handle dict (multi-aux), tuple (DSSP-only), or tensor (single head)
                    dssp_logits = None
                    if isinstance(model_output, dict):
                        position_logits = model_output['sequence']
                        dssp_logits = model_output.get('dssp')
                    elif isinstance(model_output, tuple):
                        position_logits = model_output[0]  # Use only sequence logits for sampling
                        dssp_logits = model_output[1] if len(model_output) > 1 else None
                    else:
                        position_logits = model_output

                    # Compute DSSP accuracy at this step if tracking is enabled
                    if track_dssp and dssp_logits is not None:
                        # Compute average DSSP accuracy across all proteins in the batch
                        total_correct = 0
                        total_valid = 0

                        pred_offset = 0
                        for b in range(current_batch_size):
                            use_virtual = batch_use_virtual[b]
                            protein_total_nodes = (batched_data.batch == b).sum().item()
                            size = protein_sizes[b]

                            # Extract DSSP predictions for this protein
                            if use_virtual:
                                real_nodes = protein_total_nodes - 1
                                dssp_pred_protein = dssp_logits[pred_offset:pred_offset + real_nodes]
                            else:
                                dssp_pred_protein = dssp_logits[pred_offset:pred_offset + protein_total_nodes]

                            pred_offset += protein_total_nodes

                            # Get predicted classes
                            dssp_pred_classes = dssp_pred_protein.argmax(dim=-1)[:size]

                            # Compare with ground truth for valid positions
                            valid_mask = batch_dssp_masks[b]
                            if valid_mask.sum() > 0:
                                correct = (dssp_pred_classes[valid_mask] == batch_dssp_tensors[b][valid_mask]).sum().item()
                                total_correct += correct
                                total_valid += valid_mask.sum().item()

                        # Update progress bar with average DSSP accuracy
                        if total_valid > 0:
                            dssp_accuracy = (total_correct / total_valid) * 100.0
                            time_steps.set_postfix({
                                't': f'{t_val:.3f}',
                                'batch': f'{current_batch_size}',
                                'dssp_acc': f'{dssp_accuracy:.1f}%'
                            })
                        else:
                            time_steps.set_postfix({'t': f'{t_val:.3f}', 'batch': f'{current_batch_size}'})
                    else:
                        # Update progress bar with current step info (no DSSP tracking)
                        time_steps.set_postfix({'t': f'{t_val:.3f}', 'batch': f'{current_batch_size}'})

                    # Apply temperature scaling (same as sample_chain)
                    # Apply time-dependent temperature if requested
                    if args and getattr(args, 'time_as_temperature', False):
                        # Temperature starts high (at t_min) and decreases as we approach t_max
                        flow_temp = T - t_val + 0.1
                    else:
                        flow_temp = args.flow_temp if args and hasattr(args, 'flow_temp') else 1.0
                    predicted_target = torch.softmax(position_logits / flow_temp, dim=-1)

                    # ========== DSSP GUIDANCE ==========
                    # Apply DSSP-guided blending if enabled
                    use_dssp_guidance = getattr(args, 'dssp_guidance', False) if args else False
                    ensemble_size_for_guidance = getattr(args, 'ensemble_size', 1) if args else 1

                    # Only apply guidance if ensemble_size == 1 and we have DSSP targets
                    if use_dssp_guidance and ensemble_size_for_guidance == 1 and dssp_logits is not None:
                        # Check if any proteins have DSSP targets
                        has_dssp = any(dt is not None for dt in batch_dssp_targets)

                        if has_dssp:
                            # Get DSSP probabilities
                            dssp_probs = torch.softmax(dssp_logits / flow_temp, dim=-1)  # [total_nodes, 10]

                            # Extract per-protein predictions and apply guidance
                            pred_offset = 0
                            guided_predictions = []

                            for b in range(current_batch_size):
                                use_virtual = batch_use_virtual[b]
                                protein_total_nodes = (batched_data.batch == b).sum().item()
                                size = protein_sizes[b]

                                # Extract this protein's predictions
                                if use_virtual:
                                    real_nodes = protein_total_nodes - 1
                                    aa_pred = predicted_target[pred_offset:pred_offset + real_nodes][:size]
                                    dssp_pred = dssp_probs[pred_offset:pred_offset + real_nodes][:size]
                                else:
                                    aa_pred = predicted_target[pred_offset:pred_offset + protein_total_nodes][:size]
                                    dssp_pred = dssp_probs[pred_offset:pred_offset + protein_total_nodes][:size]

                                pred_offset += protein_total_nodes

                                # Apply blending if this protein has DSSP targets
                                dssp_tgt = batch_dssp_targets[b]
                                if dssp_tgt is not None:
                                    # Prepare DSSP targets
                                    from data.dssp_constants import DSSP_UNKNOWN_IDX
                                    dssp_tensor = dssp_tgt.to(device) if isinstance(dssp_tgt, torch.Tensor) else torch.tensor(dssp_tgt, device=device)

                                    # Handle size mismatch
                                    if len(dssp_tensor) > size:
                                        dssp_tensor = dssp_tensor[:size]
                                    elif len(dssp_tensor) < size:
                                        padding = torch.full((size - len(dssp_tensor),), DSSP_UNKNOWN_IDX, device=device, dtype=dssp_tensor.dtype)
                                        dssp_tensor = torch.cat([dssp_tensor, padding])

                                    # Apply vectorized blending
                                    blending_method = getattr(args, 'dssp_blending', 'geometric') if args else 'geometric'
                                    aa_blended = blend_aa_distributions_with_dssp_guidance(
                                        model_aa_probs=aa_pred,       # [size, 21]
                                        model_dssp_probs=dssp_pred,   # [size, 10]
                                        target_dssp_indices=dssp_tensor,  # [size]
                                        blending_method=blending_method,
                                        verbose=False
                                    )
                                    guided_predictions.append(aa_blended)
                                else:
                                    # No DSSP targets, keep original
                                    guided_predictions.append(aa_pred)

                            # Rebuild predicted_target with guided predictions
                            # This will be used in the unbatching step that follows
                            new_predicted_target = torch.zeros_like(predicted_target)
                            pred_offset = 0
                            for b in range(current_batch_size):
                                use_virtual = batch_use_virtual[b]
                                protein_total_nodes = (batched_data.batch == b).sum().item()
                                size = protein_sizes[b]

                                if use_virtual:
                                    real_nodes = protein_total_nodes - 1
                                    new_predicted_target[pred_offset:pred_offset + real_nodes][:size] = guided_predictions[b]
                                else:
                                    new_predicted_target[pred_offset:pred_offset + protein_total_nodes][:size] = guided_predictions[b]

                                pred_offset += protein_total_nodes

                            predicted_target = new_predicted_target
                    # ========== END DSSP GUIDANCE ==========

                    # Handle virtual nodes in predictions (same as sample_chain)
                    predicted_target_real = []
                    pred_offset = 0
                    for b in range(current_batch_size):
                        use_virtual = batch_use_virtual[b]
                        protein_total_nodes = (batched_data.batch == b).sum().item()

                        if use_virtual:
                            # Take only real nodes (exclude last virtual node)
                            real_nodes = protein_total_nodes - 1
                            pred_real = predicted_target[pred_offset:pred_offset + real_nodes]
                        else:
                            pred_real = predicted_target[pred_offset:pred_offset + protein_total_nodes]

                        predicted_target_real.append(pred_real)
                        pred_offset += protein_total_nodes

                    # Reshape predictions back to batch format
                    predicted_batch = torch.zeros_like(x_batch)
                    for b, pred_real in enumerate(predicted_target_real):
                        size = protein_sizes[b]
                        predicted_batch[b, :size, :] = pred_real

                    # Record the model prediction and the pre-update simplex state for this
                    # step. Read-only: nothing below depends on these buffers.
                    if record_traj:
                        for b in range(current_batch_size):
                            size = protein_sizes[b]
                            traj_pred[b].append(_summarize_simplex_step(predicted_batch[b, :size, :]))
                            traj_state[b].append(_summarize_simplex_step(x_batch[b, :size, :]))

                    # Compute velocity using conditional flow (same logic as sample_chain)
                    v_batch = torch.zeros_like(x_batch)

                    for b in range(current_batch_size):
                        size = protein_sizes[b]
                        x_protein = x_batch[b:b+1, :size, :]  # [1, N, K]
                        pred_protein = predicted_batch[b:b+1, :size, :]  # [1, N, K]

                        # Try to use conditional flow if available (same as sample_chain)
                        try:
                            cond_flow = model.cond_flow
                            use_virtual = batch_use_virtual[b]
                            use_smoothed_targets_batch = getattr(args, 'use_smoothed_targets', False) if args else False
                            v_protein = cond_flow.velocity(
                                x_protein,
                                pred_protein,
                                t[b:b+1],
                                use_virtual_node=use_virtual,
                                use_smoothed_targets=getattr(args, 'use_smoothed_targets', False),
                                use_c_factor=getattr(args, 'use_c_factor', False)
                            )
                        except:
                            # Fallback to simple velocity if cond_flow not available
                            v_protein = pred_protein - x_protein

                        v_batch[b, :size, :] = v_protein[0, :size, :]

                    # Euler step with simplex projection (same as sample_chain)
                    x_new = x_batch + dt * v_batch

                    # Apply simplex projection to each protein
                    for b in range(current_batch_size):
                        size = protein_sizes[b]
                        x_new[b, :size, :] = simplex_proj(x_new[b:b+1, :size, :])[0]

                    x_batch = x_new

            # Final simplex state (t = T). This is the state the sequence is decoded from,
            # so the state trajectory gets one more entry than the prediction trajectory.
            if record_traj:
                for b in range(current_batch_size):
                    size = protein_sizes[b]
                    traj_state[b].append(_summarize_simplex_step(x_batch[b, :size, :]))

            # Extract final results for each protein
            for b in range(current_batch_size):
                size = protein_sizes[b]
                final_probabilities = x_batch[b, :size, :].cpu()  # [N, K]
                predicted_sequence = final_probabilities.argmax(-1).tolist()

                # Calculate evaluation metrics
                y_true = batch_y_true[b]
                eval_metrics = {}
                accuracy = None
                true_seq = None

                if y_true is not None:
                    true_seq = y_true.argmax(-1).tolist() if y_true.dim() > 1 else y_true.tolist()
                    if len(true_seq) == len(predicted_sequence):
                        correct = sum(p == t for p, t in zip(predicted_sequence, true_seq))
                        accuracy = correct / len(predicted_sequence) * 100

                # Convert to amino acid names
                predicted_aa = []
                for idx_val in predicted_sequence:
                    if 0 <= idx_val < len(IDX_TO_AA):
                        predicted_aa.append(IDX_TO_AA[idx_val])
                    else:
                        predicted_aa.append('XXX')

                result = {
                    'structure_idx': batch_info[b]['idx'],
                    'structure_name': batch_info[b]['name'],
                    'length': len(predicted_sequence),
                    'predicted_indices': predicted_sequence,
                    'predicted_aa': predicted_aa,
                    'predicted_sequence': ''.join([THREE_TO_ONE[aa] for aa in predicted_aa]),
                    'true_indices': true_seq,
                    'accuracy': accuracy,
                    'eval_metrics': eval_metrics,
                    'final_probabilities': final_probabilities.numpy()
                }

                # Add flank filtering metadata if it was applied
                flank_meta = batch_info[b].get('flank_metadata')
                if flank_meta is not None:
                    result['flank_filtering'] = flank_meta

                # Stack the per-step summaries into [T, N] arrays for this protein.
                if record_traj:
                    dssp_tgt = batch_dssp_targets[b]
                    if dssp_tgt is not None:
                        dssp_np = (dssp_tgt.detach().cpu().numpy() if isinstance(dssp_tgt, torch.Tensor)
                                   else np.asarray(dssp_tgt))
                        dssp_np = dssp_np.astype(np.int8)[:size]
                    else:
                        dssp_np = None

                    result['trajectory'] = {
                        **{f'pred_{k}': np.stack([s[k] for s in traj_pred[b]])
                           for k in ('argmax', 'maxprob', 'top2gap', 'entropy')},
                        **{f'state_{k}': np.stack([s[k] for s in traj_state[b]])
                           for k in ('argmax', 'maxprob', 'top2gap', 'entropy')},
                        'true_indices': (np.asarray(true_seq, dtype=np.int8)
                                         if true_seq is not None else None),
                        'dssp_targets': dssp_np,
                        'length': size,
                    }

                results.append(result)

        except Exception as e:
            print(f"Error processing batch {batch_indices}: {e}")
            import traceback
            traceback.print_exc()
            # Add error results for each protein in the failed batch
            for idx in batch_indices:
                results.append({
                    'structure_idx': idx,
                    'error': str(e)
                })

    return results


def parse_protein_list_from_file(file_path):
    """
    Parse protein names from a text file.

    Args:
        file_path: Path to file containing protein names (one per line)

    Returns:
        List of protein names
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Protein list file not found: {file_path}")

    protein_names = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):  # Skip empty lines and comments
                protein_names.append(line)

    print(f"Loaded {len(protein_names)} protein names from {file_path}")
    return protein_names


def get_indices_for_protein_names(dataset, protein_names, verbose=False):
    """
    Get dataset indices for specific protein names.

    Args:
        dataset: CathDataset instance
        protein_names: List of protein names to find
        verbose: Whether to print detailed matching information

    Returns:
        Dict mapping protein names to their dataset indices
    """
    if verbose:
        print(f"Looking for {len(protein_names)} proteins in dataset of {len(dataset)} structures...")

    name_to_index = {}
    missing_proteins = []

    # Create a mapping from protein names to indices
    for idx in range(len(dataset)):
        try:
            data, _, _, _, _ = dataset[idx]  # Unpack 5 values (includes DSSP)
            protein_name = getattr(data, 'name', None)
            if protein_name:
                name_to_index[protein_name] = idx
                if verbose and protein_name in protein_names:
                    print(f"  Found {protein_name} at index {idx}")
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not load structure at index {idx}: {e}")
            continue

    # Find indices for requested proteins
    found_indices = {}
    for protein_name in protein_names:
        if protein_name in name_to_index:
            found_indices[protein_name] = name_to_index[protein_name]
        else:
            missing_proteins.append(protein_name)

    if missing_proteins:
        print(f"Warning: Could not find {len(missing_proteins)} proteins in dataset:")
        for protein in missing_proteins:
            print(f"  - {protein}")

        # Suggest similar names if available
        available_names = list(name_to_index.keys())
        if available_names:
            print(f"\nAvailable protein names (first 10): {available_names[:10]}")

    print(f"Successfully found {len(found_indices)} out of {len(protein_names)} requested proteins")
    return found_indices


def resolve_protein_sampling_mode(args, dataset):
    """
    Resolve which proteins to sample based on command line arguments.

    Args:
        args: Parsed command line arguments
        dataset: CathDataset instance

    Returns:
        Tuple of (indices_to_sample, sampling_mode_description)
    """
    # Priority order: protein_list > protein_name > sample_all > structure_idx

    if args.protein_list:
        print(f"Using protein list from file: {args.protein_list}")
        protein_names = parse_protein_list_from_file(args.protein_list)
        name_to_index = get_indices_for_protein_names(dataset, protein_names, verbose=args.verbose)
        indices = list(name_to_index.values())
        return indices, f"protein list from {args.protein_list} ({len(indices)} proteins)"

    elif args.protein_name:
        print(f"Using specific protein: {args.protein_name}")
        name_to_index = get_indices_for_protein_names(dataset, [args.protein_name], verbose=args.verbose)
        if args.protein_name in name_to_index:
            return [name_to_index[args.protein_name]], f"specific protein '{args.protein_name}'"
        else:
            raise ValueError(f"Protein '{args.protein_name}' not found in dataset")

    elif args.sample_all:
        print("Using sample_all mode - sampling all structures in the dataset")
        indices = list(range(len(dataset)))
        if args.max_structures:
            indices = indices[:args.max_structures]
            return indices, f"all structures (limited to first {args.max_structures})"
        return indices, f"all structures ({len(indices)} total)"

    else:
        # Use structure_idx (default mode)
        print(f"Using structure index: {args.structure_idx}")
        if args.structure_idx >= len(dataset):
            raise ValueError(f"Structure index {args.structure_idx} is out of range (dataset has {len(dataset)} structures)")
        return [args.structure_idx], f"single structure at index {args.structure_idx}"
