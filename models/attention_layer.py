
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
attention_layer.py
Simple attention-based MessagePassing with optional QKV attention.

Borrowed-from concept in VN-eGNN/src/models/egnn.py (but reimplemented).
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class AttentionLayer(MessagePassing):
    def __init__(self, s_in, s_out, v_in, v_out, edge_s, dropout, use_qkv=False, K=21, edge_v=1):
        super().__init__(aggr='add')  # Sum messages from neighbors

        # Linear projections for scalar node features
        self.lin_s = nn.Linear(s_in, s_out)

        # Linear layer for vector features - treats each vector component independently
        # Input: [N, v_in, 3] -> Output: [N, v_out, 3]
        self.lin_v = nn.Linear(v_in, v_out, bias=False)

        # Linear projection for scalar edge features
        self.lin_edge = nn.Linear(edge_s, s_out)

        # Edge vector projection (if edge vectors exist)
        self.edge_v_dim = edge_v
        if edge_v > 0:
            self.lin_edge_v = nn.Linear(edge_v, v_out, bias=False)
        else:
            self.lin_edge_v = None

        # Geometric attention network - extracts invariants from vector features
        # This learns what geometric relationships matter for attention
        self.geom_attn_net = nn.Sequential(
            nn.Linear(v_in + v_in + edge_v, s_out // 2),  # Compact geometric features
            nn.SiLU(),
            nn.Linear(s_out // 2, 1)
        )


        self.dropout = nn.Dropout(dropout)

        # Separate MLPs for node's own features and neighbor messages
        self.node_mlp = nn.Sequential(
            nn.Linear(s_out, s_out),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(s_out, s_out)
        )

        self.neighbor_mlp = nn.Sequential(
            nn.Linear(s_out, s_out),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(s_out, s_out)
        )

        # Attention mechanism (predictions are already in node embeddings, no need for separate handling)
        self.use_qkv = use_qkv

        if use_qkv:
            # QKV attention: separate projections for Query, Key, Value
            self.q_proj = nn.Linear(s_out, s_out)
            self.k_proj = nn.Linear(s_out, s_out)
            self.v_proj = nn.Linear(s_out, s_out)
            self.scale = s_out ** -0.5  # Scaling factor for attention
        else:
            # Simple attention MLP: takes concatenated scalar features of the target node,
            # source node, and edge features
            self.attn_fc = nn.Sequential(
                nn.Linear(3 * s_out, s_out),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(s_out, 1)
            )

    def forward(self, x_s, x_v, edge_index, edge_attr_s, edge_attr_v=None):
        # Project scalar and vector node features
        x_s_proj = self.lin_s(x_s)        # [N, s_out]

        # The vector features x_v have shape [N, v_in, 3]
        # Apply linear transformation to each vector channel independently
        x_v_proj = self.lin_v(x_v.transpose(-1, -2)).transpose(-1, -2)  # [N, v_out, 3]

        # Project scalar edge features
        edge_attr_s_proj = self.lin_edge(edge_attr_s)  # [E, s_out]

        # Project edge vector features if they exist
        edge_attr_v_proj = None
        if edge_attr_v is not None and self.lin_edge_v is not None:
            edge_attr_v_proj = self.lin_edge_v(edge_attr_v.transpose(-1, -2)).transpose(-1, -2)  # [E, v_out, 3]

        # CRITICAL FIX: Don't pass vector features directly to propagate
        # PyG expects only 2D tensors. We'll handle vectors manually in message/aggregate
        # Store original vectors for later use
        self._temp_x_v = x_v_proj
        self._temp_edge_v = edge_attr_v_proj

        # Pass only scalar features to propagate (PyG compatible)
        aggr_s = self.propagate(edge_index,
                              x_s=x_s_proj,
                              edge_attr_s=edge_attr_s_proj,
                              size=None)

        # Manually handle vector message passing
        aggr_v = self._manual_vector_propagate(edge_index, x_v_proj, edge_attr_v_proj)

        # Clean up temporary storage
        self._temp_x_v = None
        self._temp_edge_v = None

        # Update with aggregated features
        out_s = self.update(aggr_s, x_s_proj)
        out_v = x_v_proj + aggr_v

        return out_s, out_v

    def _manual_vector_propagate(self, edge_index, x_v, edge_attr_v):
        """
        Manually handle vector message passing since PyG doesn't support 3D tensors well.
        Simplified version that just ensures vectors are aggregated.
        """
        src, dst = edge_index

        # Get source vectors
        x_v_j = x_v[src]  # Source vectors [E, v_out, 3]

        # Simple aggregation - just sum neighboring vectors
        # This ensures vectors participate in message passing
        aggr_v = torch.zeros_like(x_v)
        aggr_v.index_add_(0, dst, x_v_j)

        return aggr_v

    def message(self, x_s_i, x_s_j, edge_attr_s, index=None, size_i=None):
        """
        Simplified message function that only handles scalar features.
        Vector features are handled separately in _manual_vector_propagate.
        """
        # --- Scalar attention (existing logic) ---
        if self.use_qkv:
            # QKV attention mechanism
            query_context = x_s_i
            key_context = x_s_j + edge_attr_s
            value_context = x_s_j + edge_attr_s

            q = self.q_proj(query_context)  # [E, s_out]
            k = self.k_proj(key_context)    # [E, s_out]
            v = self.v_proj(value_context)  # [E, s_out]

            # Compute attention scores
            scalar_attn = torch.sum(q * k, dim=-1) * self.scale  # [E]
            msg_s = v  # Will be weighted by softmax below

        else:
            # Simple attention
            attention_input = torch.cat([x_s_i, x_s_j, edge_attr_s], dim=-1)
            scalar_attn = self.attn_fc(attention_input).squeeze(-1)  # [E]
            msg_s = x_s_j  # Will be weighted by softmax below

        # Apply attention weights
        alpha = softmax(scalar_attn, index, num_nodes=size_i)
        msg_s = msg_s * alpha.unsqueeze(-1)
        msg_s = self.dropout(msg_s)

        return msg_s

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        """
        Aggregate scalar messages only (vectors handled manually).
        """
        # Standard scalar aggregation
        if dim_size is None:
            dim_size = index.max().item() + 1

        # Manual aggregation using torch operations
        aggr_s = torch.zeros(dim_size, inputs.shape[1], device=inputs.device, dtype=inputs.dtype)
        aggr_s.index_add_(0, index, inputs)

        return aggr_s

    def update(self, aggr_out, x_s):
        """
        Update function for scalar features only (used by PyG).
        Vector updates are handled separately in the main forward method.
        """
        # Update scalar features with MLPs
        node_features = self.node_mlp(x_s)  # [N, s_out]
        neighbor_features = self.neighbor_mlp(aggr_out)  # [N, s_out]
        out_s = node_features + neighbor_features  # [N, s_out]

        return out_s
