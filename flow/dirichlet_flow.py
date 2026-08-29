
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
dirichlet_flow.py

This script implements the core logic for Conditional Flow Matching (CFM) with a
Dirichlet distribution. It is responsible for calculating the target "velocity"
or "drift" field that the neural network will be trained to predict.

The key idea is to define a "noising" process that transforms a clean one-hot
encoded amino acid sequence into a noisy sample from a Dirichlet distribution.
This class then calculates the exact vector field of this process, providing a
ground truth target for the GNN model to match.

To make this efficient, it pre-computes a grid of values for the derivative of
the Beta CDF, which is central to the velocity calculation.
"""
import numpy as np
import scipy.special
import torch
import torch.nn.functional as F


# Set once the velocity projection has been reported, so a routine numerical
# detail does not produce one warning per sampling step.
_WARNED_VELOCITY_PROJECTION = False


class DirichletConditionalFlow:
    """
    Reference implementation based on the Stark et al. Dirichlet Flow Matching paper.

    This class manages the mathematics of the Dirichlet flow matching process
    using the exact same approach as in the reference implementation.
    """

    def __init__(self, K=21, alpha_min=1, alpha_max=100, alpha_spacing=0.0001, label_similarity_csv=None):
        """
        Initialize the Dirichlet Conditional Flow exactly as in reference.

        Args:
            K: Number of classes (21 amino acids including X)
            alpha_min: Minimum concentration parameter
            alpha_max: Maximum concentration parameter
            alpha_spacing: Step size for alpha grid
            label_similarity_csv: Path to CSV file containing amino acid similarity matrix
        """
        self.K = K
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.alpha_spacing = float(alpha_spacing)

        # Create alpha grid exactly as in reference
        self.alphas = np.arange(alpha_min, alpha_max + alpha_spacing, alpha_spacing)

        # Create simplex value grid
        num_bs = 1000
        self.bs = np.linspace(0.0, 1.0, num_bs)
        self.beta_cdfs = []
        for alph in self.alphas:
            self.beta_cdfs.append(scipy.special.betainc(alph, K-1, self.bs))

        self.beta_cdfs = np.array(self.beta_cdfs)

        # Pre-compute derivatives
        self.beta_cdfs_derivative = np.diff(self.beta_cdfs, axis=0) / alpha_spacing

        # Load amino acid similarity matrix for smoothed labels
        self.label_similarity_matrix = None
        if label_similarity_csv is not None:
            self._load_similarity_matrix(label_similarity_csv)

    def _load_similarity_matrix(self, csv_path: str):
        """
        Load and process the amino acid similarity matrix from CSV.

        Args:
            csv_path: Path to the CSV file containing the similarity matrix
        """
        try:
            import pandas as pd

            # Load the CSV file
            try:
                df = pd.read_csv(csv_path, index_col=0)
            except Exception as e:
                df = pd.read_csv("df_combined_for_one_hot.csv", index_col=0)

            # Verify it's a 20x20 matrix for the 20 standard amino acids
            if df.shape != (20, 20):
                raise ValueError(f"Expected 20x20 similarity matrix, got {df.shape}")

            # Convert to numpy array and then to torch tensor
            similarity_20x20 = df.values.astype(np.float32)

            # Normalize rows to ensure they sum to 1.0
            row_sums = np.sum(similarity_20x20, axis=1)
            if not np.allclose(row_sums, 1.0, atol=1e-3):
                print(f"Info: Normalizing similarity matrix rows. Original row sums: {row_sums[:5]}...")
                similarity_20x20 = similarity_20x20 / row_sums[:, np.newaxis]
                print(f"Info: After normalization, row sums: {np.sum(similarity_20x20, axis=1)[:5]}...")

            # Expand to 21x21 by adding row/column for 'X' (unknown amino acid)
            # Special handling for 'X' class:
            # - When true label is 'X': distribute evenly across 20 standard AAs
            # - When predicting 'X': apply penalty (low probability)
            similarity_21x21 = np.zeros((21, 21), dtype=np.float32)
            similarity_21x21[:20, :20] = similarity_20x20

            # Row 20 (true label is 'X'): distribute evenly across 20 standard amino acids
            similarity_21x21[20, :20] = 1.0 / 20.0  # Each standard AA gets equal probability
            similarity_21x21[20, 20] = 0.0  # Don't predict 'X' when true label is 'X'

            # Column 20 (predicting 'X'): apply penalty by setting low probabilities
            # This discourages predicting 'X' regardless of true label
            similarity_21x21[:20, 20] = 0.0  # Never encourage predicting 'X'
            # similarity_21x21[20, 20] already set to 0.0 above

            # Convert to torch tensor and store
            self.label_similarity_matrix = torch.from_numpy(similarity_21x21)

            print(f"Successfully loaded {csv_path} as {similarity_21x21.shape} similarity matrix")
            print(f"Sample diagonal values: {np.diag(similarity_21x21)[:5]}")

        except Exception as e:
            print(f"Warning: Failed to load similarity matrix from {csv_path}: {e}")
            print("Falling back to standard cross-entropy loss.")
            self.label_similarity_matrix = None

    def _assert_alpha_in_grid(self, alpha_val):
        """
        Refuse to evaluate the c_factor outside the precomputed alpha grid.

        The lookup below picks the nearest grid row. For an alpha past the end of
        the grid that silently returns the *last* row instead of failing, which
        makes every step past the grid edge integrate a frozen, saturated
        c_factor while still producing a plausible-looking sequence. That is the
        worst kind of failure in a released tool, so it is an error here.

        The grid extent comes from the checkpoint (`alpha_min` / `alpha_max` /
        `alpha_spacing`), and alpha = 1 + t during sampling, so a run at horizon
        T needs alpha_max >= 1 + T. `load_model_distributed` grows the grid to
        cover the requested horizon before sampling starts, so reaching this
        error means something bypassed that path.
        """
        # One spacing of slack: alpha == alpha_max lands on the final row, and
        # floating-point accumulation of t can overshoot it by a hair.
        if self.alpha_min - self.alpha_spacing <= alpha_val <= self.alpha_max + self.alpha_spacing:
            return

        raise ValueError(
            f"Dirichlet flow c_factor requested at alpha={alpha_val:.4f}, outside the "
            f"precomputed grid [{self.alpha_min:g}, {self.alpha_max:g}].\n"
            "The grid is built from the checkpoint's alpha_min/alpha_max/alpha_spacing, "
            "and alpha = 1 + t, so a sampling horizon T needs alpha_max >= 1 + T.\n"
            "Fix by one of:\n"
            f"  - lower the sampling horizon to T <= {self.alpha_max - 1:g} "
            "(--T for sample.py, --t_max for inpainting.py), or\n"
            "  - sample without the c_factor (omit --use_c_factor), which does not "
            "use this grid at all."
        )

    def c_factor(self, bs, alpha, use_original=False):
        """
        Exact reference implementation of c_factor computation from Stark et al.

        This uses the precomputed beta CDFs and their derivatives exactly as in their codebase,
        but computes power operations in log space for numerical stability.
        """
        # Handle the case where bs and alpha might be tensors
        if isinstance(bs, torch.Tensor):
            bs = bs.detach().cpu().numpy()
        if isinstance(alpha, torch.Tensor):
            alpha = alpha.detach().cpu().numpy()

        if use_original:
            return self.c_factor_original(bs, alpha)

        # Ensure bs is a numpy array
        bs = np.asarray(bs)

        # Handle scalar alpha
        if np.isscalar(alpha):
            alpha_val = float(alpha)
        else:
            alpha_val = float(alpha.item())

        self._assert_alpha_in_grid(alpha_val)

        # Numerical stability constants
        EPS = 1e-10
        MAX_LOG = 20

        # Clamp bs to prevent log(0) and extreme values
        bs_safe = np.clip(bs, EPS, 1.0 - EPS)

        # Reference implementation logic - using log space for numerical stability
        # Step 1: Compute the log beta function term
        try:
            log_out1 = scipy.special.betaln(alpha_val, self.K - 1)
        except (OverflowError, ValueError):
            log_out1 = -MAX_LOG  # Very small value

        # Step 2: Compute the scaling factor in log space
        # Original: out1 / ((1 - bs) ** (self.K - 1))
        # Log space: exp(log_out1 - (self.K - 1) * log(1 - bs))
        one_minus_bs = 1 - bs_safe
        one_minus_bs_safe = np.clip(one_minus_bs, EPS, 1.0)  # Prevent log(0)
        log_denominator = (self.K - 1) * np.log(one_minus_bs_safe)
        log_out2 = log_out1 - log_denominator

        # Clip to prevent overflow
        log_out2 = np.clip(log_out2, -MAX_LOG, MAX_LOG)

        # Step 3: Apply the power scaling in log space
        # Original: out2 / (bs ** (alpha_val - 1))
        # Log space: exp(log_out2 - (alpha_val - 1) * log(bs))
        if alpha_val - 1 != 0:
            log_numerator = (alpha_val - 1) * np.log(bs_safe)
            log_out = log_out2 - log_numerator
        else:
            log_out = log_out2

        # Clip to prevent overflow
        log_out = np.clip(log_out, -MAX_LOG, MAX_LOG)

        # Convert back from log space with safety checks
        out = np.where(
            bs_safe < (1.0 - EPS),
            np.exp(log_out),
            0
        )

        # Additional safety for extreme values - cap the c_factor more aggressively
        MAX_C_FACTOR = 100.0  # Maximum reasonable c_factor value
        out = np.where(np.isfinite(out), out, 0)
        out = np.clip(out, 0, MAX_C_FACTOR)

        # Step 4: Find closest alpha in precomputed grid for interpolation
        alpha_idx = np.argmin(np.abs(alpha_val - self.alphas))

        # Ensure we don't go out of bounds for the derivative array
        alpha_idx = min(alpha_idx, len(self.beta_cdfs_derivative) - 1)

        # Step 5: Get the precomputed derivative function
        I_func = self.beta_cdfs_derivative[alpha_idx]

        # Step 6: Interpolate the derivative values
        interp = -np.interp(bs_safe.ravel(), self.bs, I_func)

        # Step 7: Final c_factor calculation combining all terms
        final = interp * out.ravel()

        # Final safety check for NaN or infinite values
        final = np.where(np.isfinite(final), final, 0.0)

        return final.reshape(bs.shape)

    def c_factor_original(self, bs, alpha):
        # Add numerical stability constant
        EPS = 1e-10

        self._assert_alpha_in_grid(float(np.asarray(alpha).reshape(-1)[0]))

        out1 = scipy.special.beta(alpha, self.K - 1)

        # Ensure bs doesn't get too close to 1 to prevent divide by zero
        bs_safe = np.clip(bs, 0, 1 - EPS)

        # Compute denominator with numerical safety
        denominator = (1 - bs_safe) ** (self.K - 1)
        out2 = np.where(denominator > EPS, out1 / denominator, 0)

        # Compute bs power with numerical safety
        bs_power = bs_safe ** (alpha - 1)
        out = np.where(bs_power > EPS, out2 / bs_power, 0)

        # Clamp alpha to valid range to prevent index out of bounds
        alpha_clamped = np.clip(alpha, self.alphas[0], self.alphas[-1])
        alpha_idx = np.argmin(np.abs(alpha_clamped - self.alphas))

        # Additional safety check for bounds
        alpha_idx = np.clip(alpha_idx, 0, len(self.beta_cdfs_derivative) - 1)

        I_func = self.beta_cdfs_derivative[alpha_idx]
        interp = -np.interp(bs_safe, self.bs, I_func)
        final = interp * out

        # Final safety check for NaN or infinite values
        final = np.where(np.isfinite(final), final, 0.0)

        return final

    def velocity(self, xt, predicted_probs, times, use_virtual_node=False, use_smoothed_targets=False, velocity_zero_sum=False, use_c_factor=False, alternative_test=False):
        """
        Exact reference implementation of velocity computation from Stark et al.

        In Dirichlet flow matching, the velocity must be tangent to the simplex and naturally
        maintains the constraint that velocities sum to zero. The proper formula includes
        probability weighting to ensure simplex constraint.

        Args:
            xt: Current noisy positions [B, N, K] or [N, K]
            predicted_probs: predicted_probs distribution [B, N, K] or [N, K]
            times: Noise levels [B] or scalar
            use_virtual_node: Whether virtual nodes are used
            use_smoothed_targets: Whether to use smoothed targets
            velocity_zero_sum: Whether to enforce zero-sum constraint on velocity components

        Returns:
            velocity: The velocity field with same shape as xt
        """
        alphas = times + 1.0  # Convert noise levels to concentration parameters
        # Determine target distribution: smooth or hard
        target_distribution = torch.eye(self.K, device=predicted_probs.device, dtype=predicted_probs.dtype)

        # Convert inputs to numpy for c_factor computation
        xt_np = xt.detach().cpu().numpy() if isinstance(xt, torch.Tensor) else xt
        alphas_np = alphas.detach().cpu().numpy() if isinstance(alphas, torch.Tensor) else alphas

        # Handle batched vs non-batched inputs
        if len(xt.shape) == 3:  # Batched: [B, N, K]
            B, N, K = xt.shape
            velocity = torch.zeros_like(xt)

            for b in range(B):
                alpha_val = alphas_np[b] if len(alphas_np.shape) > 0 else alphas_np

                for n in range(N):
                    # Get current simplex coordinates and target
                    x_pos = xt_np[b, n, :]  # [K] - current probabilities
                    # Use identity matrix as target, scaled by predicted probabilities
                    target_pos = target_distribution.detach().cpu().numpy()  # [K, K] identity matrix
                    predicted_probs_np = predicted_probs[b, n, :].detach().cpu().numpy()  # [K] - predicted probabilities

                    # Ensure we don't have zero probabilities (numerical stability)
                    x_pos_safe = np.maximum(x_pos, 1e-8)

                    # Compute velocity using proper Dirichlet flow formula
                    # The correct formula includes probability weighting to naturally satisfy simplex constraint
                    velocity_components = []
                    for k in range(K):
                        # Get c_factor for this component

                        # Velocity component: v_k = c_k * predicted_prob_k * (identity_matrix[k] - x_k)

                        if use_c_factor:
                            c_k = self.c_factor(np.array([x_pos_safe[k]]), alpha_val)[0]
                        else:
                            c_k = 1.0

                        if alternative_test:
                            # Alternative test: v_k = c_k * (predicted_prob_k - x_k)
                            v_k = c_k * (predicted_probs_np[k] - x_pos[k])
                        else:
                            # Original formula: v_k = c_k * predicted_prob_k * (identity_matrix[k] - x_k)
                            v_k = c_k * predicted_probs_np[k] * (target_pos[k, k] - x_pos[k])

                        velocity_components.append(v_k)

                    # Convert to tensor
                    velocity_tensor = torch.tensor(velocity_components, dtype=xt.dtype, device=xt.device)

                    # Optionally enforce simplex constraint (sum = 0)
                    if velocity_zero_sum:
                        velocity_sum = velocity_tensor.sum()
                        velocity[b, n, :] = velocity_tensor - velocity_sum / K
                    else:
                        velocity[b, n, :] = velocity_tensor

        else:  # Non-batched: [N, K]
            N, K = xt.shape
            velocity = torch.zeros_like(xt)

            alpha_val = alphas_np.item() if hasattr(alphas_np, 'item') else float(alphas_np)

            for n in range(N):
                # Get current simplex coordinates and target
                x_pos = xt_np[n, :]  # [K]
                # Use identity matrix as target, scaled by predicted probabilities
                target_pos = target_distribution.detach().cpu().numpy()  # [K, K] identity matrix
                predicted_probs_np = predicted_probs[n, :].detach().cpu().numpy()  # [K] - predicted probabilities

                # Ensure we don't have zero probabilities (numerical stability)
                x_pos_safe = np.maximum(x_pos, 1e-8)

                # Compute velocity using proper Dirichlet flow formula
                velocity_components = []
                for k in range(K):
                    # Get c_factor for this component
                    if use_c_factor:
                        c_k = self.c_factor(np.array([x_pos_safe[k]]), alpha_val)[0]
                    else:
                        c_k = 1.0  # Default to 1 if not using c_factor


                    if alternative_test:
                        # Alternative test: v_k = c_k * (predicted_prob_k - x_k)
                        v_k = c_k * (predicted_probs_np[k] - x_pos[k])
                    else:
                        # Original formula: v_k = c_k * predicted_prob_k * (identity_matrix[k] - x_k)
                        # Velocity component: v_k = c_k * predicted_prob_k * (identity_matrix[k] - x_k)
                        v_k = c_k * predicted_probs_np[k] * (target_pos[k, k] - x_pos[k])
                    velocity_components.append(v_k)

                # Convert to tensor
                velocity_tensor = torch.tensor(velocity_components, dtype=xt.dtype, device=xt.device)

                # Optionally enforce simplex constraint (sum = 0)
                if velocity_zero_sum:
                    velocity_sum = velocity_tensor.sum()
                    velocity[n, :] = velocity_tensor - velocity_sum / K
                else:
                    velocity[n, :] = velocity_tensor

        # Enforce velocity sums to zero (simplex constraint)
        velocity_sum = velocity.sum(dim=-1, keepdim=True)

        # Check if velocities are close to zero-sum; if not, enforce it
        if not torch.allclose(velocity_sum, torch.zeros_like(velocity_sum), atol=1e-1):
            # This fires on essentially every step of a normal run: the per-residue
            # velocity is assembled component-wise and only projected onto the
            # simplex tangent space here. The projection below is the fix, not a
            # fallback, so this is expected behaviour rather than a fault.
            #
            # It used to warn on every occurrence -- 13 warnings for a single
            # 20-step design -- which trained users to ignore warnings from this
            # module. Say it once per process, and say it accurately.
            global _WARNED_VELOCITY_PROJECTION
            if not _WARNED_VELOCITY_PROJECTION and velocity_sum.abs().max() > 0.5:
                _WARNED_VELOCITY_PROJECTION = True
                import warnings
                warnings.warn(
                    "Velocity is being projected onto the simplex tangent space "
                    f"(max deviation from zero-sum: {velocity_sum.abs().max().item():.3f}). "
                    "This is expected and is corrected in place; it is reported once "
                    "per process, not per step.",
                    UserWarning,
                )

            # Enforce zero-sum constraint by subtracting mean
            K = velocity.shape[-1]
            velocity = velocity - velocity_sum / K

        return velocity



    def _get_smoothed_targets(self, y_onehot):
        """
        Convert one-hot targets to smoothed targets using the similarity matrix.

        Args:
            y_onehot: One-hot encoded targets [..., K]

        Returns:
            Smoothed target distributions [..., K]
        """
        # Move similarity matrix to the same device as y_onehot
        similarity_matrix = self.label_similarity_matrix.to(y_onehot.device)

        # Get true class indices from one-hot encoding
        true_classes = y_onehot.argmax(dim=-1)  # [...]

        # Get smoothed target distributions by indexing into similarity matrix
        # similarity_matrix[i] gives the smoothed distribution for class i
        smoothed_targets = similarity_matrix[true_classes]  # [..., K]

        return smoothed_targets



    def cross_entropy_loss(self, logits, y_onehot, use_smoothed_labels=False,
                           label_smoothing_factor=0.0, class_weights=None):
        """
        Cross-entropy loss with optional label smoothing and class weighting support.

        Supports two smoothing approaches (mutually exclusive):
        1. Similarity-matrix smoothing (use_smoothed_labels=True): Uses amino acid similarity matrix
        2. Uniform smoothing (label_smoothing_factor > 0): Distributes factor uniformly to non-true classes

        Args:
            logits: Model predictions [N, K]
            y_onehot: One-hot encoded targets [N, K]
            use_smoothed_labels: Whether to use similarity-matrix smoothed labels (only for training)
            label_smoothing_factor: Uniform label smoothing strength (0.0 to disable)
            class_weights: Optional [K] tensor of per-class weights for frequency-weighted loss.
                           Applied only to the standard hard-label path. X class (index 20)
                           should have weight 0.0 to remain masked. None = unweighted.
        """
        if use_smoothed_labels and self.label_similarity_matrix is not None:
            # Use similarity-matrix smoothed labels for training
            return self._smoothed_cross_entropy_loss(logits, y_onehot)
        elif label_smoothing_factor > 0.0:
            # Use uniform label smoothing
            return self._uniform_label_smoothing_loss(logits, y_onehot, label_smoothing_factor)
        else:
            # Standard hard labels, optionally frequency-weighted
            targets = y_onehot.argmax(dim=-1)
            weight = class_weights.to(logits.device) if class_weights is not None else None
            return F.cross_entropy(logits, targets, weight=weight)

    def _uniform_label_smoothing_loss(self, logits, y_onehot, factor):
        """
        Compute cross-entropy loss with uniform label smoothing.

        For each sample:
        - True class probability = 1 - factor
        - Each incorrect class probability = factor / (K - 1)

        Args:
            logits: Model predictions [N, K]
            y_onehot: One-hot encoded targets [N, K]
            factor: Smoothing strength (0.0 to 1.0)

        Returns:
            Smoothed cross-entropy loss (scalar)
        """
        # Basic shape validation
        if logits.shape != y_onehot.shape:
            raise ValueError(f"Shape mismatch: logits {logits.shape} != y_onehot {y_onehot.shape}")

        N, K = logits.shape

        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)  # [N, K]

        # Get true class indices
        targets = y_onehot.argmax(dim=-1)  # [N]

        # Negative log likelihood for true class
        nll = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)  # [N]

        # Compute mean log-prob over incorrect classes only
        # sum_log_probs - true_log_prob gives sum over incorrect classes
        sum_log_probs = log_probs.sum(dim=-1)  # [N]
        true_log_prob = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)  # [N]
        mean_log_incorrect = (sum_log_probs - true_log_prob) / (K - 1)  # [N]

        # Smooth loss = -mean_log_incorrect (we want to minimize this)
        smooth_loss = -mean_log_incorrect  # [N]

        # Combined loss: (1 - factor) * NLL + factor * smooth_loss
        loss = (1.0 - factor) * nll + factor * smooth_loss

        return loss.mean()

    def _smoothed_cross_entropy_loss(self, logits, y_onehot):
        """
        Compute cross-entropy loss with smoothed labels using the similarity matrix.

        Args:
            logits: Model predictions [N, K]
            y_onehot: One-hot encoded targets [N, K]

        Returns:
            Smoothed cross-entropy loss
        """
        # Basic shape validation
        if logits.shape != y_onehot.shape:
            raise ValueError(f"Shape mismatch: logits {logits.shape} != y_onehot {y_onehot.shape}")

        # Move similarity matrix to the same device as logits
        similarity_matrix = self.label_similarity_matrix.to(logits.device)

        # Get true class indices
        true_classes = y_onehot.argmax(dim=-1)  # [N]

        # Get smoothed target distributions by indexing into similarity matrix
        # similarity_matrix[i] gives the smoothed distribution for class i
        smoothed_targets = similarity_matrix[true_classes]  # [N, K]

        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)  # [N, K]

        # Compute negative log-likelihood under smoothed targets
        # This is equivalent to KL divergence: KL(smoothed_targets || predicted_probs)
        loss = -(smoothed_targets * log_probs).sum(dim=-1).mean()

        return loss
