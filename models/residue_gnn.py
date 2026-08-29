
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
residue_gnn.py

This script defines the core Graph Neural Network (GNN) architecture used for
processing the protein graphs. It is a stack of Geometric Vector Perceptron (GVP)
convolutional layers.
"""

import sys

import torch
import torch.nn.functional as F
from torch import nn

sys.path.append('..')
from gvp import GVP, LayerNorm
from models.attention_layer import AttentionLayer
from models.edge_update import SE3EdgeUpdate
from models.equivariant_time_gvp import (
    HybridConditionedGVP,
    HybridConditionedGVPWithResidual,
    HybridConditionedLayerNorm,
)


class GVPWithResidual(nn.Module):
    """
    GVP layer with residual connection for deep feature processing.
    This allows stacking many GVP layers without vanishing gradients.
    """
    def __init__(self, node_dims, hidden_dims, dropout=0.1):
        super().__init__()
        self.gvp = GVP(node_dims, hidden_dims, activations=(F.relu, None))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x is a tuple (scalar_features, vector_features)
        identity_s, identity_v = x
        out_s, out_v = self.gvp(x)

        # Apply residual connection
        out_s = identity_s + self.dropout(out_s)
        out_v = identity_v + out_v  # No dropout on vectors to preserve direction

        return out_s, out_v

class ResidueGNN(nn.Module):
    """
    A GNN for processing protein residue graphs with both scalar and vector features.

    The network consists of deep embedding layers to project the input features
    into richer representations, followed by a smaller number of AttentionLayers that
    perform message passing on the graph. This design separates feature learning
    from graph communication for better efficiency and expressiveness.
    """
    def __init__(self,
                 node_dims=(7,3),
                 edge_dims=(48,1),
                 hidden_dims=(32,16),
                 num_layers=None,
                 num_embed_layers=None,
                 num_message_layers=None,
                 dropout=0.1,
                 use_qkv=False,
                 use_predictions=True,
                 prediction_integration='concat',
                 K=21,
                 # New time conditioning parameters
                 use_time_conditioning=True,
                 time_dim=64,
                 time_integration='film',
                 # Architecture configuration
                 architecture='interleaved',
                 # Edge feature updates
                 update_edge_feats=True,
                 # Attention skip connection
                 attention_skip_connection=False):
        """
        Args:
            node_dims (tuple): A tuple (s, v) for the dimensions of scalar and
                               vector node features.
            edge_dims (tuple): A tuple (s, v) for the dimensions of scalar and
                               vector edge features. Note that the vector part is
                               not used by the AttentionLayer.
            hidden_dims (tuple): A tuple (s, v) for the dimensions of the hidden
                                 scalar and vector representations.
            num_layers (int): The number of message passing layers (kept for backward compatibility).
            num_embed_layers (int): The number of deep embedding layers (defaults to num_layers-1).
            num_message_layers (int): The number of message passing layers (defaults to 2).
            dropout (float): The dropout rate.
            use_qkv (bool): Whether to use QKV attention instead of simple attention.
            use_predictions (bool): Whether to use current predictions in processing (default: True).
            prediction_integration (str): How to integrate predictions ('concat', 'add', 'gated').
            K (int): Number of amino acid classes.
            use_time_conditioning (bool): Whether to enable time conditioning (default: False).
            time_dim (int): Dimension of time embeddings.
            time_integration (str): Time integration method ('film' or 'add').
            architecture (str): Architecture type - 'interleaved'.
            attention_skip_connection (bool): If True, add residual skip connection around
                each AttentionLayer: hs = hs + attn_out. Default False.
        """
        super().__init__()

        # Handle backward compatibility for layer counts
        if num_embed_layers is None:
            if num_layers is not None:
                num_embed_layers = num_layers
            else:
                num_embed_layers = 4  # Default value
        if num_message_layers is None:
            num_message_layers = 1  # Default value

        # Store configuration
        self.architecture = architecture
        self.use_predictions = use_predictions
        self.prediction_integration = prediction_integration
        self.use_time_conditioning = use_time_conditioning
        self.time_dim = time_dim
        self.time_integration = time_integration
        self.K = K
        self.hidden_dims = hidden_dims
        self.edge_dims = edge_dims
        self.dropout_rate = dropout
        self.use_qkv = use_qkv
        self.update_edge_feats = update_edge_feats
        self.attention_skip_connection = attention_skip_connection

        # Validate architecture type
        if architecture not in ['interleaved']:
            raise ValueError(f"Unknown architecture '{architecture}'. Must be 'interleaved'.")

        # Calculate input dimensions based on prediction integration
        # Note: For hybrid conditioning, predictions are handled within the layers
        if use_time_conditioning:
            # With hybrid conditioning, we don't augment dimensions here
            # since predictions are handled within HybridConditionedGVP
            augmented_node_dims = node_dims
        else:
            # Legacy prediction integration for backward compatibility
            if use_predictions and prediction_integration == 'concat':
                # Add prediction dimension to scalar node features
                prediction_proj_dim = hidden_dims[0] // 2  # Use half of hidden dim for predictions
                augmented_node_dims = (node_dims[0] + prediction_proj_dim, node_dims[1])
                self.pred_proj = nn.Linear(K, prediction_proj_dim)
            elif use_predictions and prediction_integration == 'gated':
                # Use gating mechanism to combine predictions with features
                augmented_node_dims = node_dims
                self.pred_proj = nn.Linear(K, node_dims[0])
                self.pred_gate = nn.Sequential(
                    nn.Linear(K + node_dims[0], node_dims[0]),
                    nn.Sigmoid()
                )
            else:
                # No augmentation or additive integration
                augmented_node_dims = node_dims
                if use_predictions:
                    self.pred_proj = nn.Linear(K, node_dims[0])

        self._build_interleaved_architecture(
                augmented_node_dims, num_embed_layers, num_message_layers
        )

        # Create SE(3)-equivariant edge update layers if enabled
        if update_edge_feats:
            # One edge update layer per GVP+LN block (before attention)
            num_edge_updates = num_embed_layers  # Same as GVP layers
            self.edge_update_layers = nn.ModuleList([
                SE3EdgeUpdate(
                    node_s_dim=hidden_dims[0],
                    node_v_dim=hidden_dims[1],
                    edge_s_dim=edge_dims[0],
                    edge_v_dim=edge_dims[1],
                    hidden_dim=hidden_dims[0] * 2,  # Auto-derive
                    dropout=dropout
                )
                for _ in range(num_edge_updates)
            ])
            print(f"Created {num_edge_updates} SE(3)-equivariant edge update layers")
        else:
            self.edge_update_layers = None

        self.dropout = nn.Dropout(dropout)


    def _build_interleaved_architecture(self, augmented_node_dims, num_embed_layers, num_message_layers):
        """
        Build the interleaved architecture: GVP and Attention layers interleaved.

        Algorithm:
        1. Space attention layers one GVP+LayerNorm apart
        2. Reserve early positions for GVP blocks when num_gvp - num_attention > 1
        3. Always end with GVP+LayerNorm

        Examples:
        - 4 GVP + 2 Attention: GVP₁+LN → GVP₂+LN → Attn₁ → GVP₃+LN → Attn₂ → GVP₄+LN
        - 4 GVP + 3 Attention: GVP₁+LN → Attn₁ → GVP₂+LN → Attn₂ → GVP₃+LN → Attn₃ → GVP₄+LN
        """
        if num_embed_layers < 1:
            raise ValueError("Need at least 1 GVP layer for interleaved architecture")
        if num_message_layers >= num_embed_layers:
            raise ValueError(f"Too many attention layers ({num_message_layers}) for GVP layers ({num_embed_layers}). Need at least one more GVP than attention layers.")

        # Calculate number of early GVP blocks to reserve
        num_early_gvp = max(0, num_embed_layers - num_message_layers - 1)

        # Build interleaved sequence
        interleaved_layers = []
        gvp_count = 0
        attention_count = 0

        # Add early GVP blocks first
        for i in range(num_early_gvp):
            gvp_layer, layer_norm = self._create_gvp_block(
                augmented_node_dims if gvp_count == 0 else self.hidden_dims,
                self.hidden_dims,
                is_first_layer=(gvp_count == 0)
            )
            interleaved_layers.extend([gvp_layer, layer_norm])
            gvp_count += 1

        # Interleave remaining GVP and attention layers
        remaining_gvp = num_embed_layers - gvp_count
        remaining_attention = num_message_layers

        while remaining_gvp > 0 or remaining_attention > 0:
            # Add GVP if we have remaining
            if remaining_gvp > 0:
                gvp_layer, layer_norm = self._create_gvp_block(
                    augmented_node_dims if gvp_count == 0 else self.hidden_dims,
                    self.hidden_dims,
                    is_first_layer=(gvp_count == 0)
                )
                interleaved_layers.extend([gvp_layer, layer_norm])
                gvp_count += 1
                remaining_gvp -= 1

            # Add attention if we have remaining and it won't be the last layer
            if remaining_attention > 0 and remaining_gvp > 0:
                attention_layer = AttentionLayer(
                    s_in=self.hidden_dims[0], s_out=self.hidden_dims[0],
                    v_in=self.hidden_dims[1], v_out=self.hidden_dims[1],
                    edge_s=self.edge_dims[0], dropout=self.dropout_rate,
                    use_qkv=self.use_qkv, K=self.K, edge_v=self.edge_dims[1])
                interleaved_layers.append(attention_layer)
                attention_count += 1
                remaining_attention -= 1

        self.interleaved_layers = nn.ModuleList(interleaved_layers)

        print(f"Built interleaved architecture: {gvp_count} GVP layers, {attention_count} attention layers")

    def _create_gvp_block(self, in_dims, out_dims, is_first_layer=False):
        """Create a GVP + LayerNorm block with appropriate conditioning."""
        if self.use_time_conditioning:
            if is_first_layer:
                gvp_layer = HybridConditionedGVP(
                    in_dims=in_dims,
                    out_dims=out_dims,
                    time_dim=self.time_dim,
                    prediction_dim=self.K,
                    activations=(None, None),
                    time_integration=self.time_integration
                )
            else:
                gvp_layer = HybridConditionedGVPWithResidual(
                    node_dims=in_dims,  # Should be same as out_dims for residual
                    time_dim=self.time_dim,
                    prediction_dim=self.K,
                    dropout=self.dropout_rate,
                    time_integration=self.time_integration
                )
            layer_norm = HybridConditionedLayerNorm(out_dims)
        else:
            # For non-time-conditioned layers, just use the dimensions passed in
            # The caller should handle augmentation if needed
            if is_first_layer:
                gvp_layer = GVP(in_dims, out_dims, activations=(None, None))
            else:
                gvp_layer = GVPWithResidual(in_dims, out_dims, dropout=self.dropout_rate)
            layer_norm = LayerNorm(out_dims)

        return gvp_layer, layer_norm


    def _integrate_predictions(self, node_s, predictions):
        """
        Integrate prediction probabilities with node scalar features.

        Args:
            node_s: Original scalar node features [N, node_dim_s]
            predictions: Current prediction probabilities [N, K]

        Returns:
            Enhanced node features
        """
        if not self.use_predictions:
            return node_s

        # Predictions should never be None when use_predictions=True
        # In the actual training/inference pipeline, predictions are always provided
        # either from ground truth or initialized as uniform probabilities
        if predictions is None:
            raise ValueError(
                "predictions cannot be None when use_predictions=True. "
                "In the actual training pipeline, predictions are always provided. "
                "If you're testing, either set use_predictions=False or provide actual predictions."
            )

        # Use prediction probabilities directly (no conversion needed)
        if self.prediction_integration == 'concat':
            # Project probabilities and concatenate
            pred_features = self.pred_proj(predictions)  # [N, proj_dim]
            return torch.cat([node_s, pred_features], dim=-1)

        elif self.prediction_integration == 'add':
            # Project probabilities to same dimension and add
            pred_features = self.pred_proj(predictions)  # [N, node_dim_s]
            return node_s + pred_features

        elif self.prediction_integration == 'gated':
            # Use gating mechanism to selectively combine
            pred_features = self.pred_proj(predictions)  # [N, node_dim_s]
            gate_input = torch.cat([predictions, node_s], dim=-1)
            gate = self.pred_gate(gate_input)  # [N, node_dim_s]
            return gate * pred_features + (1 - gate) * node_s

        else:
            return node_s

    def forward(self, node_s, node_v, edge_index, edge_s, edge_v,
                predictions=None, time_emb=None):
        """
        Performs the forward pass through the GNN with prediction and time conditioning.

        Args:
            node_s: Scalar node features, shape [N, node_dims[0]].
            node_v: Vector node features, shape [N, node_dims[1], 3].
            edge_index: Graph connectivity, shape [2, E].
            edge_s: Scalar edge features, shape [E, edge_dims[0]].
            edge_v: Vector edge features, shape [E, edge_dims[1], 3].
                    (Note: this is unused by the AttentionLayer).
            predictions: Current prediction probabilities, shape [N, K] (optional).
                        These are used directly without conversion.
            time_emb: Time embeddings, shape [N, time_dim] (optional).

        Returns:
            A tuple (hs, hv) of the final scalar and vector node embeddings.
        """
        if self.use_time_conditioning and time_emb is None:
            raise ValueError("time_emb is required when use_time_conditioning=True")

        return self._forward_interleaved(node_s, node_v, edge_index, edge_s, edge_v, predictions, time_emb)


    def _forward_interleaved(self, node_s, node_v, edge_index, edge_s, edge_v, predictions, time_emb):
        """
        Forward pass for interleaved architecture with GVP-to-GVP residuals.

        GVP layers form the main pathway with residual connections.
        Attention layers are inserted between GVP layers to add structural context.
        Edge features are updated after LayerNorm blocks (before Attention) if enabled.
        """
        hs, hv = node_s, node_v

        # Track GVP residuals - attention layers don't participate in residuals
        gvp_residual_s, gvp_residual_v = None, None

        # Track edge update index
        edge_update_idx = 0

        for layer in self.interleaved_layers:
            if isinstance(layer, (GVP, HybridConditionedGVP)):
                # First GVP layer or non-residual GVP
                if self.use_time_conditioning and isinstance(layer, HybridConditionedGVP):
                    hs, hv = layer((hs, hv), predictions=predictions, time_emb=time_emb)
                else:
                    if not self.use_time_conditioning:
                        # Handle prediction integration for first layer
                        if gvp_residual_s is None:  # First GVP layer
                            enhanced_hs = self._integrate_predictions(hs, predictions)
                            hs, hv = layer((enhanced_hs, hv))
                        else:
                            hs, hv = layer((hs, hv))
                    else:
                        hs, hv = layer((hs, hv))

                # Store for next GVP residual
                gvp_residual_s, gvp_residual_v = hs, hv

            elif isinstance(layer, (GVPWithResidual, HybridConditionedGVPWithResidual)):
                # GVP with residual connection
                if gvp_residual_s is not None:
                    # Apply GVP with residual from previous GVP
                    if self.use_time_conditioning:
                        out_s, out_v = layer((hs, hv), predictions=predictions, time_emb=time_emb)
                    else:
                        out_s, out_v = layer((hs, hv))

                    # Add residual connection from previous GVP
                    hs = out_s + gvp_residual_s
                    hv = out_v + gvp_residual_v
                else:
                    # No previous GVP to connect to
                    if self.use_time_conditioning:
                        hs, hv = layer((hs, hv), predictions=predictions, time_emb=time_emb)
                    else:
                        hs, hv = layer((hs, hv))

                # Update residual for next GVP
                gvp_residual_s, gvp_residual_v = hs, hv

            elif isinstance(layer, (LayerNorm, HybridConditionedLayerNorm)):
                # Layer normalization
                if isinstance(layer, HybridConditionedLayerNorm):
                    hs, hv = layer((hs, hv))
                else:
                    hs, hv = layer((hs, hv))

                # Edge update after LayerNorm, before Attention
                if self.update_edge_feats and self.edge_update_layers is not None:
                    if edge_update_idx < len(self.edge_update_layers):
                        edge_s, edge_v = self.edge_update_layers[edge_update_idx](
                            edge_index, hs, hv, edge_s, edge_v
                        )
                        edge_update_idx += 1

            elif isinstance(layer, AttentionLayer):
                # Attention layer
                h_s_out, h_v_out = layer(x_s=hs,
                                         x_v=hv,
                                         edge_index=edge_index,
                                         edge_attr_s=edge_s,
                                         edge_attr_v=edge_v)  # NOW PASSING EDGE VECTORS!

                # Optionally add residual skip connection around attention
                if self.attention_skip_connection:
                    hs = hs + h_s_out
                    hv = hv + h_v_out
                else:
                    hs = h_s_out
                    hv = h_v_out

                # Apply activation and dropout after attention
                hs = F.leaky_relu(hs)
                hs = self.dropout(hs)

                # Note: gvp_residual stays the same - attention doesn't update the GVP pathway

        return hs, hv
