# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
from torch.distributions import Dirichlet


def sample_forward(
    y_onehot: torch.Tensor,
    t: torch.Tensor,
    generator: torch.Generator = None,
    return_max_prob: bool = False,
    dirichlet_multiplier: float = 1.0,
) -> torch.Tensor:
    """
    Samples a noisy simplex point `x_t` from the forward diffusion process.

    This function takes a batch of one-hot encoded ground truth sequences (`y_onehot`)
    and a batch of noise levels (`t`) and generates a corresponding batch of
    noisy samples `x_t`.

    The noising process is defined by sampling from a Dirichlet distribution where
    the concentration parameters `alpha` are determined by the noise level `t` and
    the ground truth class. Specifically, for a given class `i`, the concentration
    is `α_i = (1 + t) * multiplier` if `i` is the true class, and `α_j = 1 * multiplier`
    for all other classes `j`.

    Args:
        y_onehot (torch.Tensor): The ground truth labels, shape [B, N, K].
        t (torch.Tensor): The noise levels for each sample in the batch, shape [B].
        generator (torch.Generator, optional): Random number generator for reproducible sampling.
                                             If None, uses global RNG state.
        return_max_prob (bool, optional): If True, returns (x_t, max_prob) where max_prob
                                        is the highest probability value in the batch.
        dirichlet_multiplier (float, optional): Multiplier for all Dirichlet concentration
                                               parameters. Higher values increase confidence
                                               (less noise). Default: 1.0 (no change).

    Returns:
        torch.Tensor or tuple: The noisy samples `x_t`, shape [B, N, K].
                              If return_max_prob=True, returns (x_t, max_prob_value).
    """
    B, N, K = y_onehot.shape

    # The concentration for the true class is (1 + t) * multiplier.
    # We reshape `t` to be broadcastable with `y_onehot`.
    alpha_true = (t.view(B, 1, 1) + 1.0) * dirichlet_multiplier  # Shape: [B, 1, 1]

    # Construct the full alpha tensor.
    # For the true class (where y_onehot is 1), alpha = 1*multiplier + (1 * (alpha_true - 1*multiplier)) = alpha_true.
    # For other classes (where y_onehot is 0), alpha = 1*multiplier + (0 * (alpha_true - 1*multiplier)) = 1*multiplier.
    alpha = dirichlet_multiplier + (y_onehot * (alpha_true - dirichlet_multiplier))

    # The Dirichlet distribution in PyTorch expects a 2D input, so we reshape.
    alpha_flat = alpha.view(-1, K)

    # Sample from the Dirichlet distribution.
    # Since PyTorch's Dirichlet.sample() doesn't support generator parameter directly,
    # but the validation system already re-seeds generators per epoch, we can
    # temporarily seed the global RNG and let the existing seeding handle reproducibility
    if generator is not None:
        # Save current global random state
        original_rng_state = torch.get_rng_state()
        original_cuda_rng_state = None
        if torch.cuda.is_available() and alpha_flat.is_cuda:
            original_cuda_rng_state = torch.cuda.get_rng_state(device=alpha_flat.device)

        try:
            # Use a simple hash of the generator's current state to get a seed
            # This avoids device compatibility issues
            gen_state_tensor = generator.get_state()
            # Simple hash: sum of first few bytes converted to int
            state_bytes = gen_state_tensor.cpu().numpy().tobytes()[:8]  # First 8 bytes
            seed = hash(state_bytes) % (2**31 - 1)  # Convert to positive int

            # Set global RNG to this seed temporarily
            torch.manual_seed(seed)
            if torch.cuda.is_available() and alpha_flat.is_cuda:
                torch.cuda.manual_seed(seed)

            # Sample using the temporarily seeded global RNG
            x = Dirichlet(alpha_flat).sample()

            # Advance the generator state manually for next call
            # We'll create a new generator with a different seed derived from current state
            # This ensures the generator state changes for the next call
            current_seed = hash(state_bytes) % (2**31 - 1)
            new_seed = (current_seed + 1) % (2**31 - 1)  # Simple increment
            generator.manual_seed(new_seed)

        finally:
            # Restore original global RNG state
            torch.set_rng_state(original_rng_state)
            if original_cuda_rng_state is not None:
                torch.cuda.set_rng_state(
                    original_cuda_rng_state, device=alpha_flat.device
                )
    else:
        # Use global RNG directly
        x = Dirichlet(alpha_flat).sample()

    # Reshape the output back to the original batch dimensions.
    x_reshaped = x.view(B, N, K)

    if return_max_prob:
        # Calculate the maximum probability value in this batch
        max_prob = x_reshaped.max().item()
        return x_reshaped, max_prob
    else:
        return x_reshaped
