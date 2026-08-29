# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import sys

import torch
import torch.nn.functional as F
from torch import nn

sys.path.append("..")

from data.dssp_constants import DSSP_TO_IDX, FIXED_DSSP_CLASS_WEIGHTS, NUM_DSSP_CLASSES
from data.aa_constants import (
    ELECTROSTATIC_LABEL, NUM_ELECTROSTATIC_CLASSES,
    GEOM_TOPOLOGY_LABEL, NUM_GEOM_TOPOLOGY_CLASSES,
)
from features.time_embed import GaussianFourierProjection
from flow.dirichlet_flow import DirichletConditionalFlow
from models.gnn import InverseFoldGVP


class DFMNodeClassifier(nn.Module):
    """
    The main model for inverse folding using Dirichlet Flow Matching.

    This class orchestrates the entire process:
    1. It takes a protein graph, a noisy sequence `x_t`, and a noise level `t`.
    2. It injects the `x_t` and `t` information into the graph's features.
    3. It processes the feature-rich graph using a GVP-GNN (`InverseFoldGVP`).
    4. It predicts position logits for each residue in the sequence.
    """

    def __init__(
        self,
        time_dim: int = 64,
        time_scale: float = 30.0,
        head_hidden: int = 128,
        gvp_kwargs: dict = {},
        dfm_kwargs: dict = {},
        head_dropout: float = -1,
        head_depth: int = -1,
        recycle_steps: int = -1,
        use_time_conditioning: bool = True,
        time_integration: str = "film",
        lambda_dssp_loss: float = None,
        lambda_electrostatic_loss: float = None,
        lambda_geom_topology_loss: float = None,
        use_entropy_features: bool = False,
        entropy_smoothing: float = 0.1,
    ):
        """
        Initializes all components of the model.

        Args:
            gvp_kwargs (dict): Arguments for the `InverseFoldGVP` GNN, including
                               node/edge/hidden dimensions.
            dfm_kwargs (dict): Arguments for the `DirichletConditionalFlow` helper,
                               including K (number of classes) and alpha parameters.
            time_dim (int): The embedding dimension for the time projection.
            head_hidden (int): The hidden dimension for the final prediction head.
            head_dropout (float): The dropout rate for the final MLP head.
            head_depth (int): Number of layers in the prediction head (default: 2).
            recycle_steps (int): Number of recycling iterations through the GNN.
            use_time_conditioning (bool): Whether to enable time conditioning in GVP layers.
            time_integration (str): Time integration method ('film' or 'add').
            lambda_dssp_loss (float, optional): Weight for DSSP loss. If None, DSSP loss is disabled.
            use_entropy_features (bool): Whether to add entropy as edge features.
            entropy_smoothing (float): Smoothing constant for entropy features.
        """
        super().__init__()

        # Store configuration
        self.recycle_steps = recycle_steps
        self.head_depth = head_depth
        self.head_hidden = head_hidden
        self.original_node_dim = gvp_kwargs["node_dims"][0]
        self.use_time_conditioning = use_time_conditioning
        self.time_integration = time_integration

        # DSSP configuration
        self.lambda_dssp_loss = lambda_dssp_loss
        self.use_dssp_loss = lambda_dssp_loss is not None and lambda_dssp_loss > 0

        # New auxiliary head flags
        self.use_electrostatic_loss = lambda_electrostatic_loss is not None and lambda_electrostatic_loss > 0
        self.use_geom_topology_loss = lambda_geom_topology_loss is not None and lambda_geom_topology_loss > 0
        self.use_any_aux_loss = (self.use_dssp_loss or self.use_electrostatic_loss
                                 or self.use_geom_topology_loss)

        # Entropy edge features configuration
        self.use_entropy_features = use_entropy_features
        self.entropy_smoothing = entropy_smoothing

        # The GNN's hidden scalar and vector dimensions
        hidden_s, hidden_v = gvp_kwargs["hidden_dims"]

        # Configure GNN with time conditioning parameters
        gnn_kwargs = gvp_kwargs.copy()
        gnn_kwargs["use_predictions"] = True
        if use_time_conditioning:
            # Use hybrid conditioning (predictions via concat, time via film/add)
            gnn_kwargs["use_time_conditioning"] = True
            gnn_kwargs["time_dim"] = time_dim
            gnn_kwargs["time_integration"] = time_integration
        else:
            # Legacy mode: only prediction integration
            gnn_kwargs["prediction_integration"] = "concat"

        # Pass through SE(3)-equivariant edge update flag
        gnn_kwargs["update_edge_feats"] = gvp_kwargs.get("update_edge_feats", True)

        # Pass through attention skip connection flag
        gnn_kwargs["attention_skip_connection"] = gvp_kwargs.get("attention_skip_connection", False)

        # Instantiate the core GNN
        self.gnn = InverseFoldGVP(**gnn_kwargs)

        # Time embedding (always needed for time injection)
        self.time_emb = GaussianFourierProjection(embed_dim=time_dim, scale=time_scale)

        # Time conditioning for final head
        if use_time_conditioning:
            if time_integration == "film":
                # FiLM conditioning for prediction head
                self.head_time_scale = nn.Linear(time_dim, head_hidden)
                self.head_time_shift = nn.Linear(time_dim, head_hidden)
            elif time_integration == "add":
                # Additive conditioning for prediction head
                self.head_time_proj = nn.Linear(time_dim, head_hidden)
        else:
            # Legacy time injection
            self.time_lin = nn.Linear(time_dim, hidden_s)

        # DSSP class weights (fixed, no EMA)
        if self.use_dssp_loss:
            # Register fixed class weights as a buffer
            if FIXED_DSSP_CLASS_WEIGHTS is not None:
                self.register_buffer(
                    "dssp_class_weights", FIXED_DSSP_CLASS_WEIGHTS.clone()
                )
            else:
                # Fallback to uniform weights if torch not available during import
                uniform_weights = torch.ones(NUM_DSSP_CLASSES, dtype=torch.float32)
                uniform_weights[DSSP_TO_IDX["X"]] = 0.0  # X class gets 0 weight
                self.register_buffer("dssp_class_weights", uniform_weights)

        # Register AA→class lookup tables as buffers (move to GPU with model)
        if self.use_electrostatic_loss:
            self.register_buffer('_electrostatic_label', ELECTROSTATIC_LABEL.clone())
        if self.use_geom_topology_loss:
            self.register_buffer('_geom_topology_label', GEOM_TOPOLOGY_LABEL.clone())

        # The final MLP head - input size needs to account for enhanced geometric features
        # Calculate the exact dimension based on the GNN's enhanced geometric feature extraction
        combined_hidden_dim = self._calculate_combined_feature_dim(hidden_s, hidden_v)

        # Build prediction heads (sequence is always present, DSSP is optional)
        sequence_output_dim = dfm_kwargs.get("K", 21)

        if self.use_any_aux_loss:
            # Build dual heads: shared layers + separate final layers
            self.shared_head, self.skip_weights = self._build_prediction_head(
                combined_hidden_dim,
                head_hidden,
                head_dropout,
                head_depth,
                head_hidden,  # shared output = head_hidden
            )

            # Separate final layers for sequence and each enabled aux head
            self.sequence_final = nn.Linear(head_hidden, sequence_output_dim)
            if self.use_dssp_loss:
                self.dssp_final = nn.Linear(head_hidden, NUM_DSSP_CLASSES)
            if self.use_electrostatic_loss:
                self.electrostatic_final = nn.Linear(head_hidden, NUM_ELECTROSTATIC_CLASSES)
            if self.use_geom_topology_loss:
                self.geom_topology_final = nn.Linear(head_hidden, NUM_GEOM_TOPOLOGY_CLASSES)
            self.head = None  # Not used in dual mode
        else:
            # Single head for sequence only
            self.head, self.skip_weights = self._build_prediction_head(
                combined_hidden_dim,
                head_hidden,
                head_dropout,
                head_depth,
                sequence_output_dim,
            )
            self.shared_head = None
            self.sequence_final = None
            self.dssp_final = None
            self.electrostatic_final = None
            self.geom_topology_final = None

        # The helper class for calculating cross-entropy loss and other utilities.
        self.cond_flow = DirichletConditionalFlow(**dfm_kwargs)

    def _calculate_combined_feature_dim(self, hidden_s: int, hidden_v: int) -> int:
        """
        Calculate the exact dimension of combined features from the enhanced GNN output.

        Enhanced geometric features include:
        - Scalar features: hidden_s
        - Vector norms: hidden_v
        - Vector variance: hidden_v
        - Pairwise dot products: hidden_v * (hidden_v - 1) / 2 (if hidden_v > 1)
        - Vector coherence: hidden_v (if hidden_v > 1)
        - Cross product norms: hidden_v * (hidden_v - 1) / 2 (if hidden_v >= 2)

        Args:
            hidden_s: Scalar hidden dimension
            hidden_v: Vector hidden dimension

        Returns:
            Total combined feature dimension
        """
        # Basic scalar features
        total_dim = hidden_s

        # Vector norms (one per vector channel)
        total_dim += hidden_v

        # Vector variance (one per vector channel)
        total_dim += hidden_v

        # Pairwise dot products (if multiple vector channels)
        if hidden_v > 1:
            dot_pairs = hidden_v * (hidden_v - 1) // 2
            total_dim += dot_pairs

        # Vector coherence features (if multiple vector channels)
        if hidden_v > 1:
            total_dim += hidden_v

        # Cross product norms (if at least 2 vector channels)
        if hidden_v >= 2:
            cross_pairs = hidden_v * (hidden_v - 1) // 2
            total_dim += cross_pairs

        return total_dim

    def _build_prediction_head(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
        depth: int,
        output_dim: int,
    ):
        """
        Build a configurable prediction head with the specified depth and multi-skip connections.

        Args:
            input_dim: Input dimension from GNN
            hidden_dim: Hidden dimension for intermediate layers
            dropout: Dropout rate
            depth: Number of layers (minimum 1)
            output_dim: Output dimension (K classes)

        Returns:
            tuple: (nn.ModuleList of layers, nn.Parameter for skip weights or None)
        """
        layers = nn.ModuleList()
        skip_weights = None

        if depth == 1:
            # Direct projection: input_dim -> output_dim
            layers.append(nn.Linear(input_dim, output_dim))

        elif depth == 2:
            # Two layers: input_dim -> hidden_dim -> output_dim
            layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.LeakyReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, output_dim),
                ]
            )

        else:
            # Multiple layers with skip connections every 3 layers
            # First layer: input_dim -> hidden_dim
            layers.extend(
                [nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(), nn.Dropout(dropout)]
            )

            # Intermediate layers (depth - 2 times): hidden_dim -> hidden_dim
            for _ in range(depth - 2):
                layers.extend(
                    [
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.LeakyReLU(),
                        nn.Dropout(dropout),
                    ]
                )

            # Final layer: hidden_dim -> output_dim
            layers.append(nn.Linear(hidden_dim, output_dim))

            # Calculate skip connection points (every 3 layers, starting from layer 1)
            # We want to skip from the output of LeakyReLU layers (after activation)
            skip_points = []

            # For simplicity, we'll collect skip connections from every 3rd layer group
            # Layer groups:
            # - Group 0: layers[0,1,2] = Linear, LeakyReLU, Dropout (skip from layer 1)
            # - Group 1: layers[3,4,5] = Linear, LeakyReLU, Dropout (skip from layer 4)
            # - Group 2: layers[6,7,8] = Linear, LeakyReLU, Dropout (skip from layer 7)
            # etc.

            for group_idx in range(
                0, depth - 1, 3
            ):  # Every 3 groups, excluding final layer
                if group_idx == 0:
                    # First group: skip from LeakyReLU at index 1
                    skip_points.append(1)
                else:
                    # Subsequent groups: skip from LeakyReLU at index 3*group_idx + 1
                    leaky_relu_idx = 3 * group_idx + 1
                    if leaky_relu_idx < len(layers) - 1:  # Don't skip from final layer
                        skip_points.append(leaky_relu_idx)

            # Only create skip weights if we have skip connections (depth >= 4)
            if len(skip_points) > 0 and depth >= 4:
                # Initialize skip weights to 1.0 (equal contribution initially)
                # Add 1 for the direct connection from previous layer
                num_skip_connections = len(skip_points) + 1  # +1 for direct connection
                skip_weights = nn.Parameter(torch.ones(num_skip_connections))

                # Store skip points for use in forward pass
                self.skip_points = skip_points
                self.has_skip_connections = True
            else:
                self.skip_points = []
                self.has_skip_connections = False

        return layers, skip_weights

    def _apply_head_with_time_conditioning(
        self, hs, node_time_emb, t_emb, batch_mask, return_both_heads=False
    ):
        """
        Apply the prediction head with time conditioning and skip connections.
        Time conditioning is always applied after the first Linear+LeakyReLU, regardless of head depth.

        Args:
            hs: GNN output features
            node_time_emb: Time embeddings per node (for new time conditioning)
            t_emb: Time embeddings per batch (for legacy time conditioning)
            batch_mask: Batch assignment for nodes
            return_both_heads: If True and dual heads exist, return (sequence_logits, dssp_logits)

        Returns:
            If return_both_heads=False: sequence prediction logits
            If return_both_heads=True: tuple of (sequence_logits, dssp_logits) or just sequence_logits
        """
        # Determine which head structure to use
        head_layers = self.shared_head if self.use_any_aux_loss else self.head

        if self.use_time_conditioning:
            if self.head_depth == 1:
                # Direct projection - no time conditioning possible
                shared_features = head_layers[0](hs)
            else:
                # Apply first layer (Linear)
                h = head_layers[0](hs)

                # Apply activation (LeakyReLU) - this is always at index 1 for depth >= 2
                h = head_layers[1](h)

                # Apply time conditioning after first Linear+LeakyReLU
                if self.time_integration == "film":
                    scale = F.relu(self.head_time_scale(node_time_emb))
                    shift = F.relu(self.head_time_shift(node_time_emb))
                    h = h * (1.0 + scale) + shift
                elif self.time_integration == "add":
                    time_proj = F.relu(self.head_time_proj(node_time_emb))
                    h = h + time_proj

                # Apply remaining layers with skip connections
                skip_features = []

                # Process through layers, collecting skip features at the right points
                layer_idx = 2  # Start after dropout following time conditioning

                # Collect skip from first layer (after time conditioning)
                if hasattr(self, "skip_points") and 1 in self.skip_points:
                    skip_features.append(h.clone())

                # Apply dropout after time conditioning
                if layer_idx < len(head_layers) and isinstance(
                    head_layers[layer_idx], nn.Dropout
                ):
                    h = head_layers[layer_idx](h)
                    layer_idx += 1

                # Process intermediate layers
                while layer_idx < len(head_layers) - 1:  # Exclude final layer
                    # Apply linear layer
                    if isinstance(head_layers[layer_idx], nn.Linear):
                        h = head_layers[layer_idx](h)
                        layer_idx += 1

                        # Apply activation
                        if layer_idx < len(head_layers) and isinstance(
                            head_layers[layer_idx], nn.LeakyReLU
                        ):
                            h = head_layers[layer_idx](h)
                            layer_idx += 1

                            # Check if we should collect skip feature after activation
                            if (
                                hasattr(self, "skip_points")
                                and (layer_idx - 1) in self.skip_points
                            ):
                                skip_features.append(h.clone())

                            # Apply dropout
                            if layer_idx < len(head_layers) and isinstance(
                                head_layers[layer_idx], nn.Dropout
                            ):
                                h = head_layers[layer_idx](h)
                                layer_idx += 1
                    else:
                        layer_idx += 1

                # Apply final shared layer with skip connections
                if (
                    hasattr(self, "has_skip_connections")
                    and self.has_skip_connections
                    and len(skip_features) > 0
                ):
                    # All skip features should have the same dimension (hidden_dim)
                    # Weighted combination of skip features and current feature
                    all_features = skip_features + [h]  # Add current features

                    # Apply weights (normalized via softmax for stability)
                    weights = F.softmax(self.skip_weights, dim=0)

                    # Weighted combination - all features should have same dimension
                    weighted_features = torch.stack(
                        [w * feat for w, feat in zip(weights, all_features)], dim=0
                    )
                    h = torch.sum(weighted_features, dim=0)

                # Apply final layer
                if self.use_any_aux_loss:
                    # For dual heads, the final shared layer
                    if layer_idx < len(head_layers):
                        shared_features = head_layers[-1](h)
                    else:
                        shared_features = h
                else:
                    # For single head, apply the final output layer
                    shared_features = head_layers[-1](h)
        else:
            # Legacy time injection - add time embedding to GNN output
            th = self.time_lin(t_emb)  # Shape: [B, hidden_s]
            hs_with_time = hs + th[batch_mask]

            # Apply head layers with skip connections if enabled
            if self.head_depth == 1:
                shared_features = head_layers[0](hs_with_time)
            else:
                h = hs_with_time
                skip_features = []

                # Process through all layers except the last, collecting skip features
                layer_idx = 0
                while layer_idx < len(head_layers) - 1:
                    h = head_layers[layer_idx](h)

                    # Check if this is a skip point (after LeakyReLU layers)
                    if (
                        hasattr(self, "skip_points")
                        and layer_idx in self.skip_points
                        and isinstance(head_layers[layer_idx], nn.LeakyReLU)
                    ):
                        skip_features.append(h.clone())

                    layer_idx += 1

                # Apply final layer with skip connections
                if (
                    hasattr(self, "has_skip_connections")
                    and self.has_skip_connections
                    and len(skip_features) > 0
                ):
                    # Weighted combination of skip features and current feature
                    all_features = skip_features + [h]  # Add current features

                    # Apply weights (normalized via softmax for stability)
                    weights = F.softmax(self.skip_weights, dim=0)

                    # Weighted combination
                    weighted_features = torch.stack(
                        [w * feat for w, feat in zip(weights, all_features)], dim=0
                    )
                    h = torch.sum(weighted_features, dim=0)

                # Apply final layer
                shared_features = head_layers[-1](h)

        # Determine if any NEW auxiliary head (beyond DSSP) is active
        any_new_aux = self.use_electrostatic_loss or self.use_geom_topology_loss

        if any_new_aux:
            # Dict return: caller gets named tensors for each enabled head
            result = {'sequence': self.sequence_final(shared_features)}
            if self.use_dssp_loss:
                result['dssp'] = self.dssp_final(shared_features)
            if self.use_electrostatic_loss:
                result['electrostatic'] = self.electrostatic_final(shared_features)
            if self.use_geom_topology_loss:
                result['geom_topology'] = self.geom_topology_final(shared_features)
            return result
        elif self.use_dssp_loss and return_both_heads:
            # PRESERVED: existing DSSP-only tuple path for backward compatibility
            sequence_logits = self.sequence_final(shared_features)
            dssp_logits = self.dssp_final(shared_features)
            return sequence_logits, dssp_logits
        elif self.use_dssp_loss:
            return self.sequence_final(shared_features)
        else:
            # Single head case
            return shared_features

    def _compute_entropy_edge_features(self, node_probs, edge_index):
        """
        Compute entropy-based edge features from node probability distributions.

        For each edge (src -> dst), computes:
        - Source entropy: H(p_src) - how uncertain the source node's prediction is
        - Target entropy: H(p_dst) - how uncertain the target node's prediction is

        Both entropies are normalized to [0, 1] and smoothed to prevent spiky noise beacons.

        Args:
            node_probs: Node probability distributions [total_nodes, K]
            edge_index: Edge connectivity [2, E]

        Returns:
            entropy_features: [E, 2] tensor with (smoothed_src_entropy, smoothed_dst_entropy)
        """
        # Compute per-node entropy: H(p) = -sum(p * log(p))
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        log_probs = torch.log(node_probs + eps)
        node_entropy = -torch.sum(node_probs * log_probs, dim=-1)  # [total_nodes]

        # Normalize entropy to [0, 1] by dividing by max entropy (log(K))
        K = node_probs.size(-1)
        max_entropy = torch.log(torch.tensor(K, dtype=node_probs.dtype, device=node_probs.device))
        node_entropy_normalized = node_entropy / max_entropy  # [total_nodes]

        # Apply smoothing: blend toward 0.5 (neutral) to prevent spiky noise beacons
        # smoothed = (1 - alpha) * original + alpha * 0.5
        alpha = self.entropy_smoothing
        node_entropy_smoothed = (1.0 - alpha) * node_entropy_normalized + alpha * 0.5

        # Gather source and target entropies for each edge
        src_nodes = edge_index[0]  # [E]
        dst_nodes = edge_index[1]  # [E]

        src_entropy = node_entropy_smoothed[src_nodes]  # [E]
        dst_entropy = node_entropy_smoothed[dst_nodes]  # [E]

        # Stack as [E, 2] feature tensor
        entropy_features = torch.stack([src_entropy, dst_entropy], dim=-1)

        return entropy_features

    def forward(self, data, t, prob_t, return_dssp=None):
        """
        Performs the full forward pass to predict the probability logits for each residue.

        Args:
            data (torch_geometric.data.Data): The batched graph data.
            t (torch.Tensor): The noise levels for the batch, shape [B].
            prob_t (torch.Tensor): The noisy sequence probabilities for the batch, shape [B, N, K].
                                  Note: prob_t contains probability distributions from Dirichlet sampling.
            return_dssp (bool, optional): Whether to return DSSP predictions. If None, auto-determined by model config.

        Returns:
            torch.Tensor or tuple:
                - If single head or return_dssp=False: sequence logits [total_nodes, K]
                - If dual head and return_dssp=True: (sequence_logits, dssp_logits)
        """
        B = prob_t.size(0)

        # Handle both batched and single structure cases
        if hasattr(data, "batch") and data.batch is not None:
            batch_mask = data.batch
        else:
            # Single structure case: create a dummy batch mask
            total_nodes = data.x_s.size(0)
            batch_mask = torch.zeros(
                total_nodes, dtype=torch.long, device=data.x_s.device
            )

        # Step 1: Prepare sequence information as prediction probabilities
        # Convert prob_t to per-node prediction probabilities for the GNN
        # Note: prob_t contains probability distributions from the diffusion process, not logits
        batch_sizes = torch.bincount(batch_mask)

        # Map sequence prob_t probabilities to node-level prediction probabilities
        node_prediction_probs = []

        for i in range(B):
            num_nodes_in_graph = batch_sizes[i].item()

            if num_nodes_in_graph > 0:
                n_real = num_nodes_in_graph - 1  # Exclude virtual node

                # Take the sequence probabilities for this graph
                seq_probs = prob_t[
                    i
                ]  # [N, K] - these are probabilities from Dirichlet sampling

                # Map sequence positions to real nodes
                seq_len = seq_probs.size(0)
                actual_len = min(n_real, seq_len)

                if actual_len > 0:
                    # Use the first actual_len probabilities for real nodes
                    real_probs = seq_probs[:actual_len]  # [actual_len, K]

                    # If we need more probabilities, pad with uniform distribution
                    assert (
                        n_real >= actual_len
                    ), f"Expected at least {n_real} real nodes, but got {actual_len}"
                    if n_real > actual_len:
                        uniform_probs = torch.full(
                            (n_real - actual_len, seq_probs.size(-1)),
                            1.0 / seq_probs.size(-1),
                            device=seq_probs.device,
                            dtype=seq_probs.dtype,
                        )
                        real_probs = torch.cat([real_probs, uniform_probs], dim=0)

                    # Virtual node probabilities (uniform distribution)
                    virtual_probs = torch.full(
                        (1, seq_probs.size(-1)),
                        1.0 / seq_probs.size(-1),
                        device=seq_probs.device,
                        dtype=seq_probs.dtype,
                    )

                    # Combine real and virtual probabilities
                    graph_probs = torch.cat(
                        [real_probs, virtual_probs], dim=0
                    )  # [num_nodes_in_graph, K]
                else:
                    # All nodes get uniform probabilities
                    graph_probs = torch.full(
                        (num_nodes_in_graph, seq_probs.size(-1)),
                        1.0 / seq_probs.size(-1),
                        device=seq_probs.device,
                        dtype=seq_probs.dtype,
                    )

                node_prediction_probs.append(graph_probs)

        # Concatenate all node prediction probabilities
        if node_prediction_probs:
            initial_prediction_probs = torch.cat(
                node_prediction_probs, dim=0
            )  # [total_nodes, K]
        else:
            raise Exception("Input probabilities should exist.")
            # Fallback: uniform probabilities for all nodes
            initial_prediction_probs = torch.full(
                (data.x_s.size(0), 21),
                1.0 / 21.0,
                device=data.x_s.device,
                dtype=data.x_s.dtype,
            )

        # Convert probabilities to logits for internal processing
        # Add small epsilon to avoid log(0)
        # eps = 1e-8
        # initial_prediction_logits = torch.log(initial_prediction_probs + eps)

        # Step 2: Process the graph with time conditioning
        # Generate time embeddings (uses t+1 as implemented)
        t_emb = self.time_emb(t)  # [B, time_dim] - Gaussian Fourier projection

        # Broadcast time embeddings to all nodes
        if self.use_time_conditioning:
            node_time_emb = t_emb[batch_mask]  # [total_nodes, time_dim]
        else:
            node_time_emb = None

        # Iterative processing with time conditioning
        hs = None
        current_prediction_probs = initial_prediction_probs

        for recycle_iter in range(self.recycle_steps):
            # Compute entropy-based edge features if enabled
            extra_edge_s = None
            if self.use_entropy_features:
                extra_edge_s = self._compute_entropy_edge_features(
                    current_prediction_probs, data.edge_index
                )

            # Pass predictions and time embeddings through the GNN
            hs = self.gnn(
                data, predictions=current_prediction_probs, time_emb=node_time_emb,
                extra_edge_s=extra_edge_s
            )

            # For recycling iterations before the last, generate intermediate predictions
            if recycle_iter < self.recycle_steps - 1:
                # Only use sequence predictions for recycling (DSSP doesn't affect sequence during recycling)
                recycle_output = self._apply_head_with_time_conditioning(
                    hs, node_time_emb, t_emb, batch_mask
                )
                if isinstance(recycle_output, dict):
                    current_prediction_logits = recycle_output['sequence']
                else:
                    current_prediction_logits = recycle_output
                current_prediction_probs = torch.softmax(
                    current_prediction_logits, dim=-1
                )

        # Step 3: Final GNN pass + prediction head
        # If recycle_steps <= 0, hs was never set by the loop above — run GNN once now.
        if hs is None:
            extra_edge_s = None
            if self.use_entropy_features:
                extra_edge_s = self._compute_entropy_edge_features(
                    current_prediction_probs, data.edge_index
                )
            hs = self.gnn(
                data, predictions=current_prediction_probs, time_emb=node_time_emb,
                extra_edge_s=extra_edge_s
            )

        # Check if new aux heads are active (determines return format)
        any_new_aux = self.use_electrostatic_loss or self.use_geom_topology_loss

        if any_new_aux:
            # Multi-aux mode: returns dict from _apply_head_with_time_conditioning
            head_output = self._apply_head_with_time_conditioning(
                hs, node_time_emb, t_emb, batch_mask, return_both_heads=True
            )
            # head_output is already a dict
            return head_output

        # Determine if we should return DSSP predictions (legacy path)
        should_return_dssp = (
            return_dssp if return_dssp is not None else self.use_dssp_loss
        )

        if should_return_dssp and self.use_dssp_loss:
            # Get both sequence and DSSP predictions
            sequence_logits, dssp_logits = self._apply_head_with_time_conditioning(
                hs, node_time_emb, t_emb, batch_mask, return_both_heads=True
            )

            # Ensure correct shapes
            if sequence_logits.dim() != 2 or sequence_logits.size(-1) != 21:
                raise ValueError(
                    f"Sequence output has wrong shape: {sequence_logits.shape}, expected [total_nodes, 21]"
                )
            if dssp_logits.dim() != 2 or dssp_logits.size(-1) != NUM_DSSP_CLASSES:
                raise ValueError(
                    f"DSSP output has wrong shape: {dssp_logits.shape}, expected [total_nodes, {NUM_DSSP_CLASSES}]"
                )

            return sequence_logits, dssp_logits
        else:
            # Get only sequence predictions
            sequence_logits = self._apply_head_with_time_conditioning(
                hs, node_time_emb, t_emb, batch_mask
            )

            # Ensure correct shape
            if sequence_logits.dim() != 2 or sequence_logits.size(-1) != 21:
                raise ValueError(
                    f"Model output has wrong shape: {sequence_logits.shape}, expected [total_nodes, 21]"
                )

            return sequence_logits  # Shape: [total_num_nodes, K] - raw logits

    def compute_dssp_loss(self, dssp_logits, dssp_targets, node_mask=None):
        """
        Compute DSSP classification loss with fixed class frequency balancing.

        Args:
            dssp_logits (torch.Tensor): DSSP predictions [N, num_dssp_classes]
            dssp_targets (torch.Tensor): DSSP target indices [N] or one-hot [N, num_dssp_classes]
            node_mask (torch.Tensor, optional): Binary mask for valid nodes [N]

        Returns:
            torch.Tensor: Weighted cross-entropy loss for DSSP prediction
        """
        if not self.use_dssp_loss:
            return torch.tensor(0.0, device=dssp_logits.device)

        # Convert one-hot targets to indices if needed
        if dssp_targets.dim() == 2:
            dssp_target_indices = dssp_targets.argmax(dim=-1)
        else:
            dssp_target_indices = dssp_targets

        # Create mask for valid predictions (exclude virtual nodes marked as 'X')
        valid_mask = dssp_target_indices != DSSP_TO_IDX["X"]  # X is virtual/unknown
        if node_mask is not None:
            valid_mask = valid_mask & node_mask.bool()

        if not valid_mask.any():
            return torch.tensor(0.0, device=dssp_logits.device)

        # Filter to valid predictions only
        valid_logits = dssp_logits[valid_mask]
        valid_targets = dssp_target_indices[valid_mask]

        # Use fixed class weights (no EMA updates)
        class_weights = self.dssp_class_weights

        # Compute weighted cross-entropy loss
        loss = F.cross_entropy(
            valid_logits, valid_targets, weight=class_weights, reduction="mean"
        )

        return loss

    def compute_electrostatic_loss(self, electrostatic_logits, aa_targets):
        """
        Compute electrostatic/polarity auxiliary loss.
        Targets derived on-the-fly from aa_targets using self._electrostatic_label.

        Virtual node (AA index 20=XXX) maps to -1 and is automatically excluded.

        Args:
            electrostatic_logits: [N, 5] logits from electrostatic_final
            aa_targets: [N] integer AA class indices

        Returns:
            scalar loss tensor
        """
        elec_targets = self._electrostatic_label[aa_targets]  # [N], values in {0..4, -1}
        valid_mask = elec_targets >= 0  # excludes X residues and virtual node
        if not valid_mask.any():
            return torch.tensor(0.0, device=electrostatic_logits.device, requires_grad=True)
        return F.cross_entropy(
            electrostatic_logits[valid_mask],
            elec_targets[valid_mask],
            reduction='mean',
        )

    def compute_geom_topology_loss(self, geom_topology_logits, aa_targets):
        """
        Compute geometric topology auxiliary loss.
        Targets derived on-the-fly from aa_targets using self._geom_topology_label.
        5 classes: AROMATIC(F/Y/W), CB_BRANCHED(V/I), EXTENDED(L/M), SMALL(G/A), OTHER.

        Note: T is OTHER (polar confusion, not hydrophobic). P is OTHER (84.9% recall already).
        Virtual node (AA index 20=XXX) maps to -1 and is automatically excluded.

        Args:
            geom_topology_logits: [N, 5] logits from geom_topology_final
            aa_targets: [N] integer AA class indices

        Returns:
            scalar loss tensor
        """
        geom_targets = self._geom_topology_label[aa_targets]  # [N], values in {0..4, -1}
        valid_mask = geom_targets >= 0
        if not valid_mask.any():
            return torch.tensor(0.0, device=geom_topology_logits.device, requires_grad=True)
        return F.cross_entropy(
            geom_topology_logits[valid_mask],
            geom_targets[valid_mask],
            reduction='mean',
        )

    def compute_loss(
        self,
        data,
        prob_t,
        y,
        t,
        dssp_targets=None,
        use_flexible_loss_scaling=False,
        precomputed_logits=None,
    ):
        """
        A helper method to compute the cross-entropy loss for a given batch.
        It runs the forward pass and then uses the flow helper to get the loss.

        Args:
            data: PyTorch Geometric data object containing graph
            prob_t: Noisy sequence probabilities
            y: Target one-hot encoded sequence
            t: Time step
            dssp_targets: DSSP target labels [N] or one-hot [N, num_dssp_classes] (optional)
            use_flexible_loss_scaling: Whether to apply uncertainty-based loss weighting
            precomputed_logits: Optional pre-computed logits to avoid duplicate forward pass

        Returns:
            torch.Tensor: Combined loss (sequence + weighted DSSP if enabled)
        """
        # Use pre-computed logits if provided, otherwise run forward pass
        if precomputed_logits is not None:
            if self.use_dssp_loss and isinstance(precomputed_logits, tuple):
                sequence_logits, dssp_logits = precomputed_logits
            else:
                sequence_logits = precomputed_logits
                dssp_logits = None
        else:
            forward_output = self.forward(
                data, t, prob_t, return_dssp=self.use_dssp_loss
            )
            if self.use_dssp_loss and isinstance(forward_output, tuple):
                sequence_logits, dssp_logits = forward_output
            else:
                sequence_logits = forward_output
                dssp_logits = None

        # Compute sequence loss
        if use_flexible_loss_scaling:
            # Extract uncertainty weights from node scalar features
            # Uncertainty is stored in data.x_s[:, 6] - higher values = higher confidence = higher weight
            uncertainty_weights = data.x_s[:, 6].clone()  # [num_nodes]

            # Set virtual node uncertainty to 0 if virtual node is used
            if hasattr(data, "use_virtual_node") and data.use_virtual_node:
                uncertainty_weights[-1] = 0.0  # Virtual node is the last node

            # Apply weighted cross-entropy loss
            sequence_loss = self._weighted_cross_entropy_loss(
                sequence_logits, y, uncertainty_weights
            )
        else:
            # Standard unweighted loss
            sequence_loss = self.cond_flow.cross_entropy_loss(sequence_logits, y)

        # Compute DSSP loss if enabled and targets provided
        if self.use_dssp_loss and dssp_targets is not None and dssp_logits is not None:
            dssp_loss = self.compute_dssp_loss(dssp_logits, dssp_targets)
            # Combine losses with lambda weighting
            total_loss = sequence_loss + self.lambda_dssp_loss * dssp_loss
            return total_loss
        else:
            # Return only sequence loss
            return sequence_loss

    def _weighted_cross_entropy_loss(self, logits, y_onehot, weights):
        """
        Compute weighted cross-entropy loss based on uncertainty values.

        Args:
            logits: Model predictions [N, K]
            y_onehot: One-hot encoded targets [N, K]
            weights: Per-node weights [N] (higher = more important)

        Returns:
            Weighted cross-entropy loss
        """
        # Get true class indices
        targets = y_onehot.argmax(dim=-1)  # [N]

        # Compute per-sample cross-entropy loss (without reduction)
        per_sample_loss = F.cross_entropy(logits, targets, reduction="none")  # [N]

        # Apply weights and compute weighted average
        weighted_loss = per_sample_loss * weights  # [N]

        # Sum weighted losses and normalize by sum of weights
        total_weighted_loss = weighted_loss.sum()
        total_weights = weights.sum()

        # Avoid division by zero
        if total_weights > 0:
            return total_weighted_loss / total_weights
        else:
            # Fallback to unweighted loss if all weights are zero
            return per_sample_loss.mean()
