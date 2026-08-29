# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
dssp_constants.py

Constants and mappings for DSSP secondary structure classification.
Contains mappings, class information, and utility functions for DSSP processing.
"""
from typing import List, Optional

# Try to import torch, but make it optional for constants-only usage
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None

# DSSP secondary structure mappings
# Based on DSSP 8-class classification plus coil ('-') and unknown ('X')
DSSP_TO_IDX = {
    '-': 0,  # Coil/Loop (most common)
    'H': 1,  # Alpha helix
    'E': 2,  # Beta strand
    'P': 3,  # Polyproline II helix / Turn
    'S': 4,  # Bend
    'T': 5,  # Turn
    'B': 6,  # Beta bridge
    'G': 7,  # 3-10 helix
    'I': 8,  # Pi helix (rare)
    'X': 9   # Unknown (for masking)
}

# Reverse mapping
IDX_TO_DSSP = {idx: dssp for dssp, idx in DSSP_TO_IDX.items()}

# Number of DSSP classes
NUM_DSSP_CLASSES = len(DSSP_TO_IDX)  # 10 classes total
NUM_DSSP_VALID_CLASSES = NUM_DSSP_CLASSES - 1  # 9 classes (excluding 'X')

# Unknown/mask index
DSSP_UNKNOWN_IDX = DSSP_TO_IDX['X']

# DSSP class frequencies from real data (excluding X)
# Based on: {'X': 353376, '-': 855054, 'H': 1570902, 'E': 1017152, 'P': 79686,
#           'S': 406106, 'T': 511675, 'B': 49751, 'G': 165686, 'I': 24274}
DSSP_CLASS_COUNTS = {
    '-': 855054,   # Coil/Loop
    'H': 1570902,  # Alpha helix (most frequent)
    'E': 1017152,  # Beta strand
    'P': 79686,    # 3-10 helix
    'S': 406106,   # Bend
    'T': 511675,   # Turn
    'B': 49751,    # Beta bridge (very rare)
    'G': 165686,   # 3-10 helix
    'I': 24274,    # Pi helix (rarest)
    # 'X' excluded - will get weight 0
}

def compute_fixed_dssp_class_weights():
    """
    Compute fixed DSSP class weights based on real data statistics.
    Uses square root of inverse frequency for balancing.

    Returns:
        torch.Tensor: Class weights [NUM_DSSP_CLASSES] with X class = 0
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for weight computation")

    # Get total count (excluding X)
    total_count = sum(DSSP_CLASS_COUNTS.values())

    # Compute frequencies for valid classes
    frequencies = []
    for dssp_char in ['-', 'H', 'E', 'P', 'S', 'T', 'B', 'G', 'I']:  # Exclude 'X'
        if dssp_char in DSSP_CLASS_COUNTS:
            freq = DSSP_CLASS_COUNTS[dssp_char] / total_count
            frequencies.append(freq)
        else:
            frequencies.append(1e-8)  # Minimal frequency for missing classes

    frequencies = torch.tensor(frequencies, dtype=torch.float32)

    # Square root of inverse frequency weighting
    weights = torch.sqrt(1.0 / torch.clamp(frequencies, min=1e-8))

    # Normalize so mean = 1.0
    weights = weights / weights.mean()

    # Create full weight tensor (including X class = 0)
    full_weights = torch.zeros(NUM_DSSP_CLASSES, dtype=torch.float32)
    full_weights[:-1] = weights  # All classes except X
    full_weights[-1] = 0.0       # X class gets 0 weight

    return full_weights

# Precompute the fixed weights
FIXED_DSSP_CLASS_WEIGHTS = None
if _TORCH_AVAILABLE:
    FIXED_DSSP_CLASS_WEIGHTS = compute_fixed_dssp_class_weights()

# Default class frequencies (uniform for non-X classes) - kept for compatibility
# Will be converted to tensor when needed
DEFAULT_DSSP_CLASS_FREQUENCIES = [1.0/NUM_DSSP_VALID_CLASSES] * NUM_DSSP_VALID_CLASSES

# DSSP class names for logging/visualization
DSSP_CLASS_NAMES = {
    0: 'Coil',
    1: 'Alpha Helix',
    2: 'Beta Strand',
    3: 'Turn/PPII',
    4: 'Bend',
    5: 'Turn',
    6: 'Beta Bridge',
    7: '3-10 Helix',
    8: 'Pi Helix',
    9: 'Unknown'
}


def dssp_string_to_indices(dssp_string: List[str]):
    """
    Convert DSSP string list to tensor of indices.

    Args:
        dssp_string: List of DSSP characters (e.g., ['H', 'H', 'E', '-', 'X'])

    Returns:
        torch.Tensor: Tensor of DSSP indices [N]
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("torch is required for dssp_string_to_indices")

    indices = [DSSP_TO_IDX[char] for char in dssp_string]
    return torch.tensor(indices, dtype=torch.long)


def dssp_indices_to_string(dssp_indices) -> List[str]:
    """
    Convert tensor of DSSP indices back to string list.

    Args:
        dssp_indices: Tensor of DSSP indices [N] or list of indices

    Returns:
        List[str]: List of DSSP characters
    """
    if _TORCH_AVAILABLE and isinstance(dssp_indices, torch.Tensor):
        indices = dssp_indices.tolist()
    else:
        indices = dssp_indices
    return [IDX_TO_DSSP[idx] for idx in indices]


def create_dssp_mask(dssp_indices):
    """
    Create mask for valid DSSP positions (not 'X').

    Args:
        dssp_indices: Tensor of DSSP indices [N]

    Returns:
        torch.Tensor: Boolean mask [N] where True = valid position, False = 'X'
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("torch is required for create_dssp_mask")

    return dssp_indices != DSSP_UNKNOWN_IDX


def dssp_onehot_encoding(dssp_indices):
    """
    Convert DSSP indices to one-hot encoding.

    Args:
        dssp_indices: Tensor of DSSP indices [N]

    Returns:
        torch.Tensor: One-hot encoded tensor [N, NUM_DSSP_CLASSES]
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("torch is required for dssp_onehot_encoding")

    return torch.nn.functional.one_hot(dssp_indices, num_classes=NUM_DSSP_CLASSES).float()


def compute_dssp_class_weights(class_frequencies):
    """
    Compute class weights for balanced loss using square root of inverse frequency.

    Args:
        class_frequencies: Tensor of class frequencies [NUM_DSSP_VALID_CLASSES] (excluding X)

    Returns:
        torch.Tensor: Class weights [NUM_DSSP_CLASSES] (including 0 weight for X)
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("torch is required for compute_dssp_class_weights")

    # Compute square root of inverse frequency for valid classes
    eps = 1e-8  # Prevent division by zero
    valid_weights = 1.0 / torch.sqrt(class_frequencies + eps)

    # Create full weight tensor (including X class with weight 0)
    full_weights = torch.zeros(NUM_DSSP_CLASSES)
    full_weights[:NUM_DSSP_VALID_CLASSES] = valid_weights
    full_weights[DSSP_UNKNOWN_IDX] = 0.0  # Zero weight for unknown class

    return full_weights


def update_class_frequencies_ema(current_frequencies,
                                batch_dssp_indices,
                                decay: float = 0.99,
                                device: Optional = None):
    """
    Update class frequencies using exponential moving average.

    Args:
        current_frequencies: Current class frequencies [NUM_DSSP_VALID_CLASSES]
        batch_dssp_indices: DSSP indices from current batch [N]
        decay: EMA decay factor (0.99 = slow update)
        device: Device to place tensors on

    Returns:
        torch.Tensor: Updated class frequencies [NUM_DSSP_VALID_CLASSES]
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("torch is required for update_class_frequencies_ema")

    if device is None:
        device = batch_dssp_indices.device

    current_frequencies = current_frequencies.to(device)

    # Create mask for valid classes (not X)
    valid_mask = batch_dssp_indices != DSSP_UNKNOWN_IDX
    valid_indices = batch_dssp_indices[valid_mask]

    if len(valid_indices) == 0:
        # No valid samples in batch, return current frequencies
        return current_frequencies

    # Count occurrences of each valid class in batch
    batch_counts = torch.bincount(valid_indices, minlength=NUM_DSSP_CLASSES)[:NUM_DSSP_VALID_CLASSES]
    batch_frequencies = batch_counts.float() / valid_indices.size(0)

    # Update using EMA: new_freq = decay * current_freq + (1-decay) * batch_freq
    updated_frequencies = decay * current_frequencies + (1 - decay) * batch_frequencies.to(device)

    return updated_frequencies


# Example DSSP distribution from CATH (for reference)
CATH_DSSP_DISTRIBUTION = {
    'X': 353376,   # Unknown (excluded from balancing)
    '-': 855054,   # Coil
    'H': 1570902,  # Alpha helix
    'E': 1017152,  # Beta strand
    'P': 79686,    # Turn/PPII
    'S': 406106,   # Bend
    'T': 511675,   # Turn
    'B': 49751,    # Beta bridge
    'G': 165686,   # 3-10 helix
    'I': 24274     # Pi helix
}
