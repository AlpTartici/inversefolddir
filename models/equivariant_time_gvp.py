
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
equivariant_time_gvp.py

Hybrid conditioning GVP layers:
- Predictions: Concatenation (rich information preservation)
- Time: FiLM or Add (configurable global context modulation)

Preserves SE(3) equivariance by only conditioning scalar features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from gvp import GVP, LayerNorm


class HybridConditionedGVP(nn.Module):
    """
    GVP with hybrid conditioning approach preserving SE(3) equivariance.

    - Predictions: Always concatenated (rich information preservation)
    - Time: FiLM or additive conditioning (configurable)

    Equivariance is preserved by only conditioning scalar features.
    """
    def __init__(self, in_dims, out_dims, time_dim, prediction_dim=21,
                 h_dim=None, activations=(F.relu, None), vector_gate=False,
                 time_integration='film'):  # 'film' or 'add'
        super().__init__()

        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.time_dim = time_dim
        self.prediction_dim = prediction_dim
        self.time_integration = time_integration

        # Prediction integration: concatenation with projection
        # Use 1/4 of output dimension for prediction features to avoid dimension explosion
        self.pred_proj_dim = max(self.so // 4, 8)  # At least 8 dims for predictions
        self.pred_proj = nn.Linear(prediction_dim, self.pred_proj_dim)

        # Expand input dimensions to include projected predictions
        expanded_in_dims = (self.si + self.pred_proj_dim, self.vi)

        # Base GVP layer with expanded input for predictions
        self.gvp = GVP(expanded_in_dims, out_dims, h_dim, activations, vector_gate)

        # Time conditioning setup
        self.time_norm = nn.LayerNorm(time_dim)

        if time_integration == 'film':
            # FiLM conditioning: scale and shift parameters
            self.time_to_scale = nn.Linear(time_dim, self.so)
            self.time_to_shift = nn.Linear(time_dim, self.so)

            # Initialize to identity (scale=0, shift=0)
            nn.init.zeros_(self.time_to_scale.weight)
            nn.init.zeros_(self.time_to_scale.bias)
            nn.init.zeros_(self.time_to_shift.weight)
            nn.init.zeros_(self.time_to_shift.bias)

        elif time_integration == 'add':
            # Additive conditioning: project time to output scalar dimension
            self.time_proj = nn.Linear(time_dim, self.so)

            # Initialize to zero (identity)
            nn.init.zeros_(self.time_proj.weight)
            nn.init.zeros_(self.time_proj.bias)
        else:
            raise ValueError(f"Unknown time_integration: {time_integration}")

    def forward(self, x, predictions=None, time_emb=None):
        """
        Hybrid conditioning forward pass preserving equivariance.

        Args:
            x: (scalar_features, vector_features) tuple or scalar_features tensor
            predictions: [N, prediction_dim] prediction probabilities
            time_emb: [N, time_dim] time embeddings

        Returns:
            Time and prediction conditioned output
        """
        # Step 1: Handle input format
        if isinstance(x, tuple):
            s, v = x
        else:
            s, v = x, None

        # Step 2: Integrate predictions via concatenation
        if predictions is not None:
            pred_features = self.pred_proj(predictions)  # [N, pred_proj_dim]
        else:
            raise Exception("Predictions not provided to the GVP.")
            # Use zero predictions if not provided
            pred_features = torch.zeros(s.shape[0], self.pred_proj_dim,
                                      device=s.device, dtype=s.dtype)

        # Concatenate predictions to scalar features
        s_with_pred = torch.cat([s, pred_features], dim=-1)

        # Step 3: Apply base GVP with prediction-enhanced features
        if v is not None:
            gvp_out = self.gvp((s_with_pred, v))
        else:
            raise Exception("the vector features are missing. They should be in GVP input.")
            gvp_out = self.gvp(s_with_pred)

        # Step 4: Apply time conditioning (only to scalar features - preserves equivariance)
        if time_emb is not None:
            time_emb_norm = self.time_norm(time_emb)

            if isinstance(gvp_out, tuple):
                s_out, v_out = gvp_out

                # Apply time conditioning to scalars only
                if self.time_integration == 'film':
                    # Linear projection then ReLU before FiLM conditioning
                    scale = F.relu(self.time_to_scale(time_emb_norm))  # [N, so]
                    shift = F.relu(self.time_to_shift(time_emb_norm))  # [N, so]
                    s_conditioned = s_out * (1.0 + scale) + shift

                elif self.time_integration == 'add':
                    # Linear projection then ReLU before additive conditioning
                    time_proj = F.relu(self.time_proj(time_emb_norm))  # [N, so]
                    s_conditioned = s_out + time_proj
                else:
                    # No time conditioning, use original output
                    s_conditioned = s_out

                # Vector features remain unchanged - preserves equivariance!
                return s_conditioned, v_out

            else:
                # Scalar-only output
                if self.time_integration == 'film':
                    # Linear projection then ReLU before FiLM conditioning
                    scale = F.relu(self.time_to_scale(time_emb_norm))
                    shift = F.relu(self.time_to_shift(time_emb_norm))
                    return gvp_out * (1.0 + scale) + shift

                elif self.time_integration == 'add':
                    # Linear projection then ReLU before additive conditioning
                    time_proj = F.relu(self.time_proj(time_emb_norm))
                    return gvp_out + time_proj
                else:
                    # No time conditioning, use original output
                    return gvp_out
        else:
            return gvp_out

class HybridConditionedGVPWithResidual(nn.Module):
    """
    Hybrid conditioned GVP with residual connections.
    """
    def __init__(self, node_dims, time_dim, prediction_dim=21,
                 dropout=0.1, time_integration='film'):
        super().__init__()

        self.hybrid_gvp = HybridConditionedGVP(
            node_dims, node_dims, time_dim, prediction_dim,
            activations=(F.relu, None),
            time_integration=time_integration
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, predictions=None, time_emb=None):
        """
        Forward with residual connection.
        """
        identity_s, identity_v = x

        # Hybrid conditioning
        out_s, out_v = self.hybrid_gvp(x, predictions, time_emb)

        # Residual connections (preserve equivariance)
        out_s = identity_s + self.dropout(out_s)
        out_v = identity_v + out_v  # No dropout on vectors

        return out_s, out_v

class HybridConditionedLayerNorm(nn.Module):
    """
    Layer normalization compatible with hybrid conditioning interface.
    """
    def __init__(self, dims):
        super().__init__()
        self.layer_norm = LayerNorm(dims)

    def forward(self, x, predictions=None, time_emb=None):
        """
        Layer norm - ignores conditioning inputs for interface consistency.
        """
        return self.layer_norm(x)
