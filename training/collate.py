# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
from torch_geometric.data import Batch


def collate_fn(batch):
    """
    A custom collate function for the PyTorch DataLoader.

    This function takes a list of (data, y, mask, time) or (data, y, mask, time, dssp_targets) tuples
    and processes them into a batched format that the model can consume. It handles:
    1. Uses `torch_geometric.data.Batch.from_data_list` to combine multiple
       individual graph `Data` objects into a single, large graph object.
    2. Pads the ground truth sequence tensors (`y`) and the corresponding
       mask tensors so that all sequences in the batch have the same length.
    3. Handles optional DSSP targets for multitask learning.

    Args:
        batch (list): A list of tuples, where each tuple contains a
                      `torch_geometric.data.Data` object, a `torch.Tensor` for the sequence,
                      a `torch.Tensor` for the mask, a time value, and optionally DSSP targets.

    Returns:
        A tuple containing:
        - batched (Batch): The single, combined graph object for the whole batch.
        - Ypad (torch.Tensor): A tensor of padded ground truth sequences, shape [B, N_max, K].
        - mask_pad (torch.Tensor): A tensor of padded masks, shape [B, N_max].
        - time_batch (torch.Tensor): A tensor of time values, shape [B].
        - dssp_targets (list): List of DSSP target tensors (if available).
    """
    # Determine batch format by checking the first item
    batch_size = len(batch[0])
    has_dssp = batch_size == 5
    has_time = batch_size >= 4

    # Separate the components based on format
    try:
        if has_dssp:
            # DSSP format: (data, y, mask, time, dssp_targets)
            datas, ys, masks, times, dssp_targets = zip(*batch)
        elif has_time:
            # Legacy format: (data, y, mask, time)
            datas, ys, masks, times = zip(*batch)
            dssp_targets = None
        else:
            raise Exception("Unexpected batch format. Time should be included.")
    except Exception as e:
        print("Batch sample:", batch[0])
        print(f"Batch format: {len(batch[0])} items per batch element")
        raise Exception(e)

    # Create a single batched graph
    batched = Batch.from_data_list(datas)

    # Get batch size, max sequence length, and number of classes
    B = len(ys)
    Nmax = max(y.size(0) for y in ys)
    K = ys[0].size(1)

    # Create zero tensors for the padded sequences and the mask
    Ypad = torch.zeros(B, Nmax, K)
    mask_pad = torch.zeros(B, Nmax, dtype=torch.bool)

    # Copy the data from each sequence and mask into the padded tensors
    for i, (y, m) in enumerate(zip(ys, masks)):
        n = y.size(0)
        Ypad[i, :n] = y
        mask_pad[i, :n] = m  # Use the provided mask

    if has_time:
        # Convert time values to tensor
        time_batch = torch.tensor(times, dtype=torch.float32)
        if has_dssp:
            return batched, Ypad, mask_pad, time_batch, dssp_targets
        else:
            return batched, Ypad, mask_pad, time_batch
    else:
        raise Exception("Unexpected batch format. Time should be included.")
        return batched, Ypad, mask_pad
