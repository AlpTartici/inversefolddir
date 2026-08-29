# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
unified_dataset.py

A unified dataset that combines AF2 and PDB data into a single entries list,
using the same simple and elegant approach as the original CATH dataset.

This eliminates the need for separate AF2/PDB processing pipelines and
returns to the clean original architecture.
"""

import json
import multiprocessing as mp
import os
import pickle
import random
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .graph_builder import GraphBuilder

# Verbose chunk-loading and startup tracing, off unless asked for. These lines
# are genuinely useful when a training run misbehaves, but they should not be
# the default experience. Set IFD_DEBUG=1 to turn them back on.
_IFD_DEBUG = bool(os.environ.get("IFD_DEBUG"))

# Fix for numpy 1.x/2.x compatibility when unpickling data created with different numpy versions
# This ensures that pickled data created with numpy 2.x can be loaded in numpy 1.x environments
try:
    # Try to import numpy._core.numeric (numpy 2.x)
    import numpy._core.numeric
except ImportError:
    # If it doesn't exist, create a compatibility alias for numpy 1.x environments
    # This allows unpickling of data created with numpy 2.x in numpy 1.x environments
    if hasattr(np, 'core') and hasattr(np.core, 'numeric'):
        sys.modules['numpy._core.numeric'] = np.core.numeric
    elif hasattr(np, 'numeric'):
        sys.modules['numpy._core.numeric'] = np.numeric
    else:
        # Fallback: create a minimal compatibility module
        class NumericCompat:
            pass
        for attr in ['normalize_axis_tuple', 'normalize_axis_index']:
            if hasattr(np, attr):
                setattr(NumericCompat, attr, getattr(np, attr))
        sys.modules['numpy._core.numeric'] = NumericCompat()

# Additional numpy compatibility for other _core modules
try:
    import numpy._core.multiarray
except ImportError:
    if hasattr(np, 'core') and hasattr(np.core, 'multiarray'):
        sys.modules['numpy._core.multiarray'] = np.core.multiarray
    elif hasattr(np, 'multiarray'):
        sys.modules['numpy._core.multiarray'] = np.multiarray

try:
    import numpy._core.umath
except ImportError:
    if hasattr(np, 'core') and hasattr(np.core, 'umath'):
        sys.modules['numpy._core.umath'] = np.core.umath
    elif hasattr(np, 'umath'):
        sys.modules['numpy._core.umath'] = np.umath


class SharedProteinTracker:
    """
    Multi-worker protein tracking system that correctly handles set unions across DataLoader workers.
    Uses multiprocessing Manager to share protein sets between workers and main process.
    """
    def __init__(self, num_workers=0):
        self.num_workers = num_workers
        self.manager = None
        self.shared_epoch_proteins = None
        self.shared_cumulative_proteins = None
        self.worker_locks = None

        if num_workers > 0:
            # Only create shared state for multi-worker scenarios
            self.manager = mp.Manager()
            # Use Manager().list() to store protein names, convert to set when needed
            self.shared_epoch_proteins = self.manager.list()
            self.shared_cumulative_proteins = self.manager.list()
            self.worker_locks = self.manager.Lock()
            # Note: Only print debug info if needed, to avoid slowing down non-verbose training

    def add_proteins(self, protein_names):
        """Add proteins from a batch, handling both single and multi-worker cases"""
        if self.num_workers <= 1:
            # Single worker: no coordination needed, use local sets
            return protein_names

        # Multi-worker: add to shared storage with locking
        with self.worker_locks:
            # Convert current shared lists to sets for deduplication
            current_epoch = set(self.shared_epoch_proteins)
            current_cumulative = set(self.shared_cumulative_proteins)

            # Debug: Log what we're adding
            new_proteins = set(protein_names) - current_epoch
            if len(new_proteins) > 0:
                print(f"[DEBUG TRACKER ADD] Adding {len(new_proteins)} new proteins: {list(new_proteins)[:3]}... (total epoch before: {len(current_epoch)})")

            # Add new proteins
            new_epoch_proteins = current_epoch.union(set(protein_names))
            new_cumulative_proteins = current_cumulative.union(set(protein_names))

            # Update shared storage (clear and repopulate to avoid reference issues)
            self.shared_epoch_proteins[:] = list(new_epoch_proteins)
            self.shared_cumulative_proteins[:] = list(new_cumulative_proteins)

            # Debug: Log new totals
            if len(new_proteins) > 0:
                print(f"[DEBUG TRACKER ADD] Total epoch proteins now: {len(new_epoch_proteins)}")

        return protein_names

    def get_counts(self):
        """
        Get current protein counts across all workers.

        Returns:
            tuple: (epoch_unique_across_all_workers, cumulative_unique_across_all_workers)

        Note: These counts represent the actual union set of all unique proteins
        seen across all workers combined, not per-worker counts.
        """
        if self.num_workers <= 1:
            # Single worker: return placeholder counts (actual tracking happens locally)
            return 0, 0

        # Multi-worker: return counts from shared storage
        # These represent the actual union of all proteins seen across all workers
        with self.worker_locks:
            # Convert to sets to get unique counts across all workers
            epoch_unique_proteins = set(self.shared_epoch_proteins)
            cumulative_unique_proteins = set(self.shared_cumulative_proteins)

            epoch_count = len(epoch_unique_proteins)
            cumulative_count = len(cumulative_unique_proteins)

            return epoch_count, cumulative_count

    def reset_epoch(self):
        """Reset epoch protein tracking (keep cumulative)"""
        if self.num_workers > 1:
            with self.worker_locks:
                self.shared_epoch_proteins[:] = []

    def cleanup(self):
        """Clean up shared resources"""
        if self.manager:
            self.manager.shutdown()
            self.manager = None


# Global shared protein tracker instance (shared across all dataset instances)
_global_protein_tracker = None


def initialize_global_protein_tracker(num_workers, verbose=False):
    """Initialize the global protein tracker that will be shared across all dataset workers"""
    global _global_protein_tracker
    if _global_protein_tracker is None and verbose:
        _global_protein_tracker = SharedProteinTracker(num_workers)
        print(f"[DEBUG] Global protein tracker initialized for {num_workers} workers")
    return _global_protein_tracker


def get_global_protein_tracker():
    """Get the global protein tracker instance"""
    global _global_protein_tracker
    return _global_protein_tracker


class UnifiedDataset(Dataset):
    """
    Unified dataset that combines AF2 and PDB data using the original simple approach.

    All proteins (regardless of source) are loaded into a single entries list and
    processed through the same graph building pipeline. The only difference is
    in the source indicators and uncertainty normalization handled by GraphBuilder.
    """

    def __init__(self,
                 # PDB parameters
                 split_json: Optional[str] = None,
                 map_pkl: Optional[str] = None,
                 split: str = 'train',

                 # AF2 parameters
                 af2_chunk_dir: Optional[str] = None,
                 af2_chunk_limit: Optional[int] = None,
                 # NOTE: AF2 data is ALWAYS loaded lazily - no upfront loading option

                 # Mixing parameters
                 ratio_af2_pdb: int = 0,  # 0=PDB only, -1=AF2 only, X=X AF2 per 1 PDB
                 heterogeneous_batches: bool = True,  # True=mixed AF2+PDB batches (default), False=homogeneous batches

                 # Common parameters
                 max_len: Optional[int] = None,
                 graph_builder_kwargs: Optional[Dict] = None,
                 verbose: bool = False,  # Control debug output and protein counting

                 # Time sampling parameters
                 time_sampling_strategy: str = None,
                 t_min: float = 0.0,
                 t_max: float = 8.0,
                 alpha_range: float = None,

                 # Iteration control
                 deterministic: bool = True,  # If True, cycle through data deterministically

                 # Distributed training
                 rank: int = 0,
                 world_size: int = 1,
                 seed: int = 42,

                 # Optional protein list filter for efficient subsampling
                 protein_list_filter: Optional[List[str]] = None,

                 # AF2 pLDDT-based segment chopping
                 af2_plddt_threshold: float = 0.0,
                 af2_min_segment_len: int = 60,

                 # Batch size (used for probability clamping)
                 batch_size: int = 128):
        """
        Initialize unified dataset.

        Args:
            split_json: Path to CATH split JSON file (for PDB data)
            map_pkl: Path to CATH mapping pickle file (for PDB data)
            split: Data split ('train', 'validation', 'test')
            af2_chunk_dir: Directory containing AF2 pickle chunks
            af2_chunk_limit: Maximum number of AF2 chunks to load (None = all)
            ratio_af2_pdb: Mixing ratio (0=PDB only, -1=AF2 only, X=X AF2 per 1 PDB)
            heterogeneous_batches: If True, create mixed AF2+PDB batches; if False, homogeneous batches
            max_len: Maximum sequence length filter
            graph_builder_kwargs: Arguments for GraphBuilder
            deterministic: If True, cycle through data instead of random sampling
            rank: Process rank for distributed training
            world_size: Total number of processes
            seed: Random seed
            protein_list_filter: Optional list of protein names to include (for efficient subsampling)
            af2_plddt_threshold: pLDDT threshold below which residues are removed (0.0 = disabled)
            af2_min_segment_len: Minimum contiguous segment length to keep after chopping
            batch_size: Expected DataLoader batch size; used to clamp AF2/PDB probabilities
                        so neither source is excluded from a full batch in expectation

        Note: AF2 data is ALWAYS loaded lazily for memory efficiency.
        """
        self.split = split
        self.max_len = max_len
        self.af2_plddt_threshold = af2_plddt_threshold
        self.af2_min_segment_len = af2_min_segment_len
        self.batch_size = batch_size
        self.ratio_af2_pdb = ratio_af2_pdb
        self.heterogeneous_batches = heterogeneous_batches
        self.verbose = verbose
        self.deterministic = deterministic
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.protein_list_filter = set(protein_list_filter) if protein_list_filter else None

        # Print filter info if provided
        if self.protein_list_filter:
            print(f"Using protein list filter: {len(self.protein_list_filter)} proteins specified")

        # Time sampling parameters
        self.time_sampling_strategy = time_sampling_strategy
        self.t_min = t_min
        self.t_max = t_max
        self.alpha_range = alpha_range

        # Initialize AF2 lazy loading state (legacy variables for compatibility)
        # MIGHT DELETE
        self.af2_chunk_dir = af2_chunk_dir
        self.af2_chunk_limit = af2_chunk_limit

        # Simple AF2 chunk state (local to this worker)
        self.af2_state = {
            'group_chunk_map': {},
            'available_groups': [],
            'available_chunks': [],
            'current_group_idx': 0,
            'current_chunk_idx': 0,
            'current_chunk_data': [],
            'chunk_offset': 0,
            'total_proteins_seen': 0,
            'chunks_loaded_count': 0,
            'initialized': False
        }

        # Legacy variables for backward compatibility
        self.af2_group_chunk_map = self.af2_state['group_chunk_map']
        self.af2_available_groups = self.af2_state['available_groups']
        self.af2_available_chunks = self.af2_state['available_chunks']
        self.af2_current_group_idx = self.af2_state['current_group_idx']
        self.af2_current_chunk_idx = self.af2_state['current_chunk_idx']
        self.af2_current_chunk_data = self.af2_state['current_chunk_data']
        self.af2_chunk_offset = self.af2_state['chunk_offset']

        # Initialize graph builder
        if graph_builder_kwargs is None:
            graph_builder_kwargs = {}
        self.graph_builder = GraphBuilder(**graph_builder_kwargs)

        # Initialize unified entries list
        self.entries = []
        self.pdb_entries = []
        self.af2_entries = []

        # Load data based on ratio - ALWAYS USE LAZY LOADING FOR AF2
        if ratio_af2_pdb == 0:
            # PDB-only mode
            self._load_pdb_data(split_json, map_pkl, split)
            self.entries = self.pdb_entries
            print(f"PDB-only mode: Loaded {len(self.entries)} PDB structures")

        elif ratio_af2_pdb == -1:
            # All AF2 mode - lazy loading only
            self._init_af2_lazy(af2_chunk_dir)
        else:
            # Mixed mode: Use lazy AF2 loading + PDB cycling
            self._load_pdb_data(split_json, map_pkl, split)
            self._init_af2_lazy_for_mixed(af2_chunk_dir, af2_chunk_limit)  # Use mixed mode lazy loading
            self._init_mixed_mode_tracking(ratio_af2_pdb)
            print(f"Mixed mode: Loaded {len(self.pdb_entries)} PDB + AF2 (lazy) → dynamic mixing with {ratio_af2_pdb}:1 ratio")

        # Set up deterministic iteration if requested
        if deterministic:
            self.current_index = rank  # Start each rank at different offset
        else:
            self.rng = random.Random(seed + rank)

        # Initialize protein tracking for verbose mode
        # This happens in each worker process, so they all share the same global tracker
        if verbose:
            self.protein_tracker = get_global_protein_tracker()
            if self.protein_tracker is None:
                print(f"[DEBUG] Worker {rank}: Global protein tracker not initialized, skipping dataset-level tracking")
        else:
            self.protein_tracker = None

    def _chop_low_plddt_segments(self, protein_data: dict) -> List[dict]:
        """
        Split an AF2 entry into contiguous segments where pLDDT >= threshold.

        If af2_plddt_threshold == 0.0 (default), returns the entry unchanged as a
        single-element list (fast path, no pLDDT data required).

        Each qualifying segment is returned as an independent entry dict with sliced
        coords, seq, and plddt arrays.  Segments shorter than af2_min_segment_len are
        discarded.

        Args:
            protein_data: dict with keys 'seq', 'coords', optionally 'plddt'

        Returns:
            List of entry dicts (possibly empty if no segment passes the length filter).
        """
        if self.af2_plddt_threshold <= 0.0:
            return [protein_data]

        plddt = protein_data.get('plddt', None)
        if plddt is None:
            # No pLDDT data available – return as-is
            return [protein_data]

        plddt = np.asarray(plddt, dtype=np.float32)
        L = len(plddt)
        if L == 0:
            return []

        high_conf = plddt >= self.af2_plddt_threshold  # boolean mask [L]

        # Find contiguous runs of True
        segments = []
        in_seg = False
        seg_start = 0
        for i in range(L):
            if high_conf[i] and not in_seg:
                seg_start = i
                in_seg = True
            elif not high_conf[i] and in_seg:
                segments.append((seg_start, i))
                in_seg = False
        if in_seg:
            segments.append((seg_start, L))

        results = []
        base_name = protein_data.get('name', 'unknown')
        coords = protein_data['coords']
        seq = protein_data['seq']

        for seg_idx, (start, end) in enumerate(segments):
            seg_len = end - start
            if seg_len < self.af2_min_segment_len:
                continue

            entry = protein_data.copy()
            entry['seq'] = seq[start:end]
            entry['plddt'] = plddt[start:end]

            # Slice each backbone atom array in coords dict
            sliced_coords = {}
            for atom_key, arr in coords.items():
                arr = np.asarray(arr)
                sliced_coords[atom_key] = arr[start:end]
            entry['coords'] = sliced_coords

            # Give segment a unique name so downstream logging is informative
            if len(segments) > 1:
                entry['name'] = f"{base_name}_seg{seg_idx}"
            results.append(entry)

        return results

    def _load_pdb_data(self, split_json: str, map_pkl: str, split: str):
        """Load PDB data from CATH files."""
        if not split_json or not map_pkl:
            print("Skipping PDB data loading: split_json or map_pkl not provided")
            return

        # Load split information
        with open(split_json, 'r') as f:
            split_data = json.load(f)

        # Load file path mappings
        with open(map_pkl, 'rb') as f:
            map_data = pickle.load(f)

        # Load PDB entries (same as original CATH dataset)
        filtered_count = 0
        for cath_id in split_data[split]:
            if cath_id in map_data:
                # Apply protein list filter if provided
                if self.protein_list_filter and cath_id not in self.protein_list_filter:
                    filtered_count += 1
                    continue

                entry = map_data[cath_id]
                if isinstance(entry, dict) and 'seq' in entry and 'coords' in entry:
                    if self.max_len is None or len(entry['seq']) <= self.max_len:
                        # Mark as PDB source and preserve ID
                        entry = entry.copy()
                        entry['name'] = cath_id
                        entry['source'] = 'pdb'  # Explicit source marking
                        self.pdb_entries.append(entry)

        if self.protein_list_filter:
            print(f"Filtered out {filtered_count} PDB proteins not in protein list")
            print(f"Loaded {len(self.pdb_entries)} PDB entries from CATH (filtered from protein list)")
        else:
            print(f"Loaded {len(self.pdb_entries)} PDB entries from CATH")

    def _clamp_mix_probabilities(self, batch_size: int):
        """
        Clamp pdb_probability so that neither source is effectively absent from
        a full batch.

        With batch_size B, the expected number of PDB items is
        B * pdb_probability.  By keeping pdb_probability in [1/B, (B-1)/B] we
        ensure the expected count of each source is ≥ 1.

        The probability of a batch of size 128 having zero PDB items when
        pdb_probability = 1/128 ≈ 0.78% is (127/128)^128 ≈ 0.37% — already
        negligibly rare.  This is far simpler and more robust than any
        per-item slot assignment, which breaks with multiple DataLoader workers.
        """
        if not hasattr(self, 'pdb_probability'):
            return
        bs = max(batch_size, 2)  # guard against batch_size=1
        min_p = 1.0 / bs
        max_p = (bs - 1.0) / bs
        self.pdb_probability = max(min_p, min(max_p, self.pdb_probability))
        self.af2_probability = 1.0 - self.pdb_probability

    def _init_mixed_mode_tracking(self, ratio_af2_pdb: int):
        """Initialize tracking for dynamic mixed mode with lazy AF2 loading."""
        self.ratio_af2_pdb = ratio_af2_pdb
        self.pdb_cycle_index = 0  # Track position in PDB list for cycling

        # Calculate probability of serving PDB vs AF2 for heterogeneous batches
        # For ratio 15:1, probability of PDB = 1/(15+1) = 1/16 = 6.25%
        total_ratio = ratio_af2_pdb + 1
        self.pdb_probability = 1.0 / total_ratio
        self.af2_probability = ratio_af2_pdb / total_ratio

        # Clamp so neither source is excluded from a full batch in expectation
        bs = getattr(self, 'batch_size', 128)
        self._clamp_mix_probabilities(bs)

        # Initialize random generator for probabilistic serving
        import random
        self.mixed_rng = random.Random(self.seed + self.rank)

        print(f"Mixed mode tracking initialized: {ratio_af2_pdb}:1 ratio → PDB probability: {self.pdb_probability:.2%}, AF2 probability: {self.af2_probability:.2%}")

    def set_af2_ratio(self, val_cycle: int, initial_ratio: int, decay: float):
        """
        Update the AF2:PDB serving probabilities according to the decay schedule.

        Called once after each validation cycle. ratio_float can decay below 1,
        allowing more PDB than AF2 late in training. Counts are computed as
        integers that sum to batch_size, with each clamped to at least 1.

        Args:
            val_cycle:     Number of validation cycles completed so far (0-indexed
                           on first call, so ratio_af2_pdb is unchanged until after
                           the first validation).
            initial_ratio: The initial AF2:PDB ratio (args.ratio_af2_pdb).
            decay:         Geometric decay factor per cycle (args.af2_ratio_decay).

        Returns:
            (ratio_float, num_af2) for logging.
        """
        if not hasattr(self, 'pdb_probability'):
            # Not in mixed mode — nothing to update
            return None, None

        bs = max(getattr(self, 'batch_size', 128), 2)
        ratio_float = initial_ratio * (decay ** val_cycle)

        # Compute integer counts that sum to batch_size, each at least 1
        num_af2 = round(bs * ratio_float / (ratio_float + 1))
        num_af2 = max(1, min(bs - 1, num_af2))
        num_pdb = bs - num_af2

        self.pdb_probability = num_pdb / bs
        self.af2_probability = num_af2 / bs

        print(f"[AF2 ratio schedule] cycle={val_cycle}: ratio_float={ratio_float:.3f} → "
              f"num_pdb={num_pdb}, num_af2={num_af2} (batch_size={bs}) "
              f"(PDB prob={self.pdb_probability:.2%}, AF2 prob={self.af2_probability:.2%})")

        return ratio_float, num_af2

    def _get_next_af2_entry(self):
        """Get next AF2 entry from lazy chunk loading."""
        # Debug: Log chunk state on first few getitem calls
        if not hasattr(self, '_getitem_call_count'):
            self._getitem_call_count = 0
        self._getitem_call_count += 1

        if self._getitem_call_count <= 5 or self._getitem_call_count % 50 == 0:
            if self.verbose:
                current_group_num = self.af2_state['available_groups'][self.af2_state['current_group_idx']] if self.af2_state['current_group_idx'] < len(self.af2_state['available_groups']) else "INVALID"
                current_chunk_num = self.af2_state['available_chunks'][self.af2_state['current_chunk_idx']] if self.af2_state['current_chunk_idx'] < len(self.af2_state['available_chunks']) else "INVALID"
                print(f"[AF2 #{self._getitem_call_count}] chunk_state: group_{current_group_num}/chunk_{current_chunk_num:03d}, offset={self.af2_state['chunk_offset']}/{len(self.af2_state['current_chunk_data'])}, TOTAL: {self.af2_state['total_proteins_seen']} proteins seen")

        # Check if we need to load next chunk
        if self.af2_state['chunk_offset'] >= len(self.af2_state['current_chunk_data']):
            if not self._load_next_af2_chunk():
                # Reset to beginning and try again (should not happen with cyclic loading)
                print(f"[WARNING] AF2 chunk loading failed, resetting to beginning!")
                self.af2_state['current_group_idx'] = 0
                self.af2_state['current_chunk_idx'] = 0
                self._load_next_af2_chunk()
                self._load_next_af2_chunk()

        # Get entry from current chunk
        if self.af2_state['current_chunk_data']:
            entry = self.af2_state['current_chunk_data'][self.af2_state['chunk_offset']]
            self.af2_state['chunk_offset'] += 1
            # Update local reference
            self.af2_chunk_offset = self.af2_state['chunk_offset']
            return entry
        else:
            raise IndexError("No AF2 data available")

    def _get_next_pdb_entry(self):
        """Get next PDB entry cycling through the PDB list."""
        if not self.pdb_entries:
            raise IndexError("No PDB data available")

        # Get current PDB entry and advance index
        entry = self.pdb_entries[self.pdb_cycle_index]
        self.pdb_cycle_index = (self.pdb_cycle_index + 1) % len(self.pdb_entries)

        return entry

    def _load_af2_data(self, af2_chunk_dir: str, af2_chunk_limit: Optional[int] = None):
        """Load AF2 data from chunk files."""
        if not af2_chunk_dir:
            print("Skipping AF2 data loading: af2_chunk_dir not provided")
            return

        # Find all AF2 chunk files - handle both group_* subdirectory and direct chunk file structures
        chunk_files = []

        # Check if using group_* subdirectory structure
        has_groups = any(d.startswith('group_') for d in os.listdir(af2_chunk_dir) if os.path.isdir(os.path.join(af2_chunk_dir, d)))

        if has_groups:
            # Original group_* structure
            for group_dir in sorted(os.listdir(af2_chunk_dir)):
                group_path = os.path.join(af2_chunk_dir, group_dir)
                if os.path.isdir(group_path) and group_dir.startswith('group_'):
                    for chunk_file in sorted(os.listdir(group_path)):
                        if chunk_file.endswith('.pkl'):
                            chunk_path = os.path.join(group_path, chunk_file)
                            chunk_files.append(chunk_path)
        else:
            # Direct chunk files in the directory
            for chunk_file in sorted(os.listdir(af2_chunk_dir)):
                if chunk_file.endswith('.pkl') and chunk_file.startswith('af2_chunk_'):
                    chunk_path = os.path.join(af2_chunk_dir, chunk_file)
                    chunk_files.append(chunk_path)

        # Limit chunks if requested
        if af2_chunk_limit is not None:
            chunk_files = chunk_files[:af2_chunk_limit]
            print(f"Limited to first {af2_chunk_limit} AF2 chunks")

        # Load AF2 entries from chunks
        total_chunks_processed = 0
        total_proteins_found = 0
        total_proteins_filtered = 0

        for chunk_path in chunk_files:
            try:
                with open(chunk_path, 'rb') as f:
                    chunk_data = pickle.load(f)

                total_chunks_processed += 1
                chunk_proteins_found = 0
                chunk_proteins_filtered = 0

                # Debug: Check chunk data structure
                if total_chunks_processed <= 3:  # Only for first few chunks
                    if _IFD_DEBUG: print(f"DEBUG: Chunk {os.path.basename(chunk_path)} - type: {type(chunk_data)}, length: {len(chunk_data) if hasattr(chunk_data, '__len__') else 'unknown'}")
                    if hasattr(chunk_data, '__len__') and len(chunk_data) > 0:
                        sample_entry = chunk_data[0] if isinstance(chunk_data, list) else list(chunk_data.values())[0] if isinstance(chunk_data, dict) else None
                        if sample_entry:
                            if _IFD_DEBUG: print(f"DEBUG: Sample entry type: {type(sample_entry)}, keys: {list(sample_entry.keys()) if isinstance(sample_entry, dict) else 'not dict'}")

                for protein_data in chunk_data:
                    chunk_proteins_found += 1
                    total_proteins_found += 1

                    if isinstance(protein_data, dict) and 'seq' in protein_data and 'coords' in protein_data:
                        if self.max_len is None or len(protein_data['seq']) <= self.max_len:
                            # Mark as AF2 source and create name from chunk
                            entry = protein_data.copy()
                            entry['source'] = 'af2'  # Explicit source marking

                            # Create unique name from chunk path and index
                            chunk_name = os.path.basename(chunk_path).replace('.pkl', '')
                            entry['name'] = f"{chunk_name}_{len(self.af2_entries)}"

                            self.af2_entries.append(entry)
                        else:
                            chunk_proteins_filtered += 1
                            total_proteins_filtered += 1
                            if total_chunks_processed <= 3:  # Debug for first few chunks
                                if _IFD_DEBUG: print(f"DEBUG: Protein filtered due to length: {len(protein_data['seq'])} > {self.max_len}")
                    else:
                        chunk_proteins_filtered += 1
                        total_proteins_filtered += 1
                        if total_chunks_processed <= 3:  # Debug for first few chunks
                            missing_keys = []
                            if not isinstance(protein_data, dict):
                                if _IFD_DEBUG: print(f"DEBUG: Protein not dict: {type(protein_data)}")
                            else:
                                if 'seq' not in protein_data:
                                    missing_keys.append('seq')
                                if 'coords' not in protein_data:
                                    missing_keys.append('coords')
                                if missing_keys:
                                    if _IFD_DEBUG: print(f"DEBUG: Protein missing keys: {missing_keys}, available: {list(protein_data.keys())}")

                # Print chunk summary
                if total_chunks_processed <= 10 or total_chunks_processed % 100 == 0:
                    print(f"Processed chunk {total_chunks_processed}/{len(chunk_files)}: {chunk_proteins_found} found, {len(self.af2_entries)} loaded so far, {chunk_proteins_filtered} filtered")

            except Exception as e:
                print(f"Error loading AF2 chunk {chunk_path}: {e}")

        print(f"AF2 loading summary: {total_chunks_processed} chunks processed, {total_proteins_found} proteins found, {len(self.af2_entries)} loaded, {total_proteins_filtered} filtered")

        print(f"Loaded {len(self.af2_entries)} AF2 entries from {len(chunk_files)} chunks")

        # FAIL-FAST: If AF2 data was requested but none was loaded, fail immediately
        if self.ratio_af2_pdb > 0 and len(self.af2_entries) == 0:
            error_msg = f"FATAL ERROR: AF2 data requested (ratio_af2_pdb={self.ratio_af2_pdb}) but 0 AF2 entries were loaded!\n"
            error_msg += f"  - AF2 chunk directory: {af2_chunk_dir}\n"
            error_msg += f"  - Chunks found: {len(chunk_files)}\n"
            error_msg += f"  - Chunks processed: {total_chunks_processed}\n"
            error_msg += f"  - Proteins found in chunks: {total_proteins_found}\n"
            error_msg += f"  - Proteins filtered out: {total_proteins_filtered}\n"

            if total_proteins_found == 0:
                error_msg += f"  - ISSUE: No proteins found in any chunks - check chunk file format or corruption\n"
            elif total_proteins_filtered == total_proteins_found:
                error_msg += f"  - ISSUE: All proteins were filtered out - check max_len parameter ({self.max_len}) or data keys\n"
            else:
                error_msg += f"  - ISSUE: Unknown filtering issue - check debug output above\n"

            error_msg += f"Fix the AF2 data loading or set ratio_af2_pdb=0 for PDB-only training."
            raise RuntimeError(error_msg)

    def _create_mixed_entries(self, ratio_af2_pdb: int):
        """Create mixed entries list according to ratio."""
        if not self.pdb_entries or not self.af2_entries:
            error_msg = f"FATAL ERROR: Mixed mode requested (ratio_af2_pdb={ratio_af2_pdb}) but missing required data!\n"
            error_msg += f"  - PDB entries available: {len(self.pdb_entries) if self.pdb_entries else 0}\n"
            error_msg += f"  - AF2 entries available: {len(self.af2_entries) if self.af2_entries else 0}\n"

            if not self.pdb_entries:
                error_msg += f"  - ISSUE: No PDB entries loaded - check PDB data path\n"
            if not self.af2_entries:
                error_msg += f"  - ISSUE: No AF2 entries loaded - check AF2 chunk loading above\n"

            error_msg += f"Mixed mode requires both PDB and AF2 data. Fix data loading or use single-mode training."
            raise RuntimeError(error_msg)

        # Check if heterogeneous batches are requested
        if getattr(self, 'heterogeneous_batches', False):
            # Heterogeneous mode: Create a balanced mix for random sampling
            self._create_heterogeneous_entries(ratio_af2_pdb)
        else:
            # Homogeneous mode: Create pattern for sequential homogeneous batches
            self._create_homogeneous_entries(ratio_af2_pdb)

    def _create_homogeneous_entries(self, ratio_af2_pdb: int):
        """Create entries pattern for homogeneous batches (current behavior)."""
        # Create pattern: X AF2 entries per 1 PDB entry
        self.entries = []
        pdb_idx = 0
        af2_idx = 0

        # Cycle through creating pattern
        while pdb_idx < len(self.pdb_entries) and af2_idx < len(self.af2_entries):
            # Add ratio_af2_pdb AF2 entries
            for _ in range(ratio_af2_pdb):
                if af2_idx < len(self.af2_entries):
                    self.entries.append(self.af2_entries[af2_idx])
                    af2_idx += 1

            # Add 1 PDB entry
            if pdb_idx < len(self.pdb_entries):
                self.entries.append(self.pdb_entries[pdb_idx])
                pdb_idx += 1

        # Add any remaining entries
        while af2_idx < len(self.af2_entries):
            self.entries.append(self.af2_entries[af2_idx])
            af2_idx += 1

        while pdb_idx < len(self.pdb_entries):
            self.entries.append(self.pdb_entries[pdb_idx])
            pdb_idx += 1

        print(f"Homogeneous batches: Created {len(self.entries)} entries in {ratio_af2_pdb}:1 pattern")

    def _create_heterogeneous_entries(self, ratio_af2_pdb: int):
        """Create entries list optimized for heterogeneous batches with ratio maintained per batch."""
        # Calculate target proportions within each batch
        total_ratio = ratio_af2_pdb + 1
        af2_proportion = ratio_af2_pdb / total_ratio
        pdb_proportion = 1.0 / total_ratio

        # Determine balanced amounts to ensure we can fill batches properly
        min_complete_cycles = min(
            len(self.af2_entries) // ratio_af2_pdb,
            len(self.pdb_entries) // 1
        )

        # Use balanced amounts to maintain exact ratio
        num_af2 = min_complete_cycles * ratio_af2_pdb
        num_pdb = min_complete_cycles * 1

        # Create the mixed list by repeating the ratio pattern
        self.entries = []
        af2_idx = 0
        pdb_idx = 0

        # Create entries in ratio chunks, then shuffle for heterogeneous batches
        for cycle in range(min_complete_cycles):
            # Add ratio_af2_pdb AF2 entries
            cycle_entries = []
            for _ in range(ratio_af2_pdb):
                if af2_idx < len(self.af2_entries):
                    cycle_entries.append(self.af2_entries[af2_idx])
                    af2_idx += 1

            # Add 1 PDB entry
            if pdb_idx < len(self.pdb_entries):
                cycle_entries.append(self.pdb_entries[pdb_idx])
                pdb_idx += 1

            # Shuffle this cycle's entries for heterogeneous mixing
            if not self.deterministic:
                import random
                rng = random.Random(self.seed + self.rank + cycle)
                rng.shuffle(cycle_entries)

            # Add to main entries list
            self.entries.extend(cycle_entries)

        print(f"Heterogeneous batches: Created {len(self.entries)} entries ({num_af2} AF2 + {num_pdb} PDB) in {ratio_af2_pdb}:1 ratio cycles")

    def _init_af2_lazy(self, af2_chunk_dir):
        """Initialize AF2 data with lazy loading for pure AF2 mode - sets up full af2_state"""
        if not af2_chunk_dir or not os.path.exists(af2_chunk_dir):
            error_msg = f"AF2 chunk directory not found or not provided for AF2-only mode!\n"
            error_msg += f"  - Provided path: {af2_chunk_dir}\n"
            error_msg += f"  - Path exists: {os.path.exists(af2_chunk_dir) if af2_chunk_dir else False}\n"
            error_msg += f"AF2-only mode requires valid AF2 chunk directory. Check path or use PDB-only mode (ratio_af2_pdb=0)."
            raise RuntimeError(error_msg)

        # Use the same initialization as mixed mode to ensure consistency
        self._init_af2_lazy_for_mixed(af2_chunk_dir, None)

        # For AF2-only mode, try to load first chunk and set entries
        if self.af2_state['available_groups'] and self.af2_state['available_chunks']:
            try:
                # Load first chunk to initialize entries
                success = self._load_next_af2_chunk()
                if success and hasattr(self, 'af2_current_chunk_data'):
                    self.entries = self.af2_current_chunk_data.copy()
                    print(f"AF2-only mode (lazy): Initialized with {len(self.af2_state['available_groups'])} groups × {len(self.af2_state['available_chunks'])} chunks, first chunk has {len(self.af2_current_chunk_data)} entries")
                else:
                    print("AF2-only mode (lazy): Could not load first chunk, using empty entries")
                    self.entries = []
            except Exception as e:
                print(f"Warning: Failed to load first AF2 chunk: {e}")
                self.entries = []
        else:
            print("AF2-only mode (lazy): No groups or chunks available, using empty entries")
            self.entries = []

    def _init_af2_lazy_for_mixed(self, af2_chunk_dir, af2_chunk_limit):
        """Initialize AF2 data for mixed mode with lazy loading - NO upfront loading"""
        if not af2_chunk_dir or not os.path.exists(af2_chunk_dir):
            error_msg = f"AF2 chunk directory not found or not provided for mixed mode!\n"
            error_msg += f"  - Provided path: {af2_chunk_dir}\n"
            error_msg += f"  - Path exists: {os.path.exists(af2_chunk_dir) if af2_chunk_dir else False}\n"
            error_msg += f"Mixed mode requires valid AF2 chunk directory. Check path or use PDB-only mode (ratio_af2_pdb=0)."
            raise RuntimeError(error_msg)

        # Organize chunk files by group and chunk number for cyclic loading
        # Pattern: group_1/chunk_001.pkl → group_2/chunk_001.pkl → ... → group_8/chunk_001.pkl →
        #          group_1/chunk_002.pkl → group_2/chunk_002.pkl → ... → group_8/chunk_002.pkl → ...

        group_chunk_map = {}  # {group_num: {chunk_num: full_path}}
        available_groups = []
        available_chunks = set()

        # Look for group_* subdirectories and organize by group/chunk numbers
        if os.path.exists(af2_chunk_dir):
            for item in os.listdir(af2_chunk_dir):
                item_path = os.path.join(af2_chunk_dir, item)
                if os.path.isdir(item_path) and item.startswith('group_'):
                    try:
                        group_num = int(item.split('_')[1])  # Extract number from group_X
                        available_groups.append(group_num)
                        group_chunk_map[group_num] = {}

                        # Find .pkl files in this group directory
                        group_pkl_files = [f for f in os.listdir(item_path) if f.endswith('.pkl')]
                        for pkl_file in group_pkl_files:
                            # Extract chunk number from filename (af2_chunk_XXXXXX.pkl format)
                            try:
                                if pkl_file.startswith('af2_chunk_') and pkl_file.endswith('.pkl'):
                                    # Extract XXXXXX from af2_chunk_XXXXXX.pkl
                                    chunk_num_str = pkl_file[10:-4]  # Remove 'af2_chunk_' prefix (10 chars) and '.pkl' suffix
                                    chunk_num = int(chunk_num_str)
                                    available_chunks.add(chunk_num)
                                    group_chunk_map[group_num][chunk_num] = os.path.join(item_path, pkl_file)
                                else:
                                    print(f"Warning: File {pkl_file} doesn't match af2_chunk_XXXXXX.pkl pattern, skipping")
                            except (IndexError, ValueError) as e:
                                print(f"Warning: Could not parse chunk number from {pkl_file}: {e}, skipping")
                                continue
                    except (IndexError, ValueError):
                        print(f"Warning: Could not parse group number from {item}, skipping")
                        continue

        available_groups.sort()  # Ensure consistent group ordering
        available_chunks = sorted(list(available_chunks))  # Convert to sorted list

        if not available_groups or not available_chunks:
            # Enhanced error message with subdirectory information
            error_msg = f"No AF2 chunk files found for mixed mode!\n"
            error_msg += f"  - AF2 chunk directory: {af2_chunk_dir}\n"

            if os.path.exists(af2_chunk_dir):
                dir_contents = os.listdir(af2_chunk_dir)
                error_msg += f"  - Items in directory: {len(dir_contents)}\n"
                error_msg += f"  - Directory contents: {dir_contents}\n"

                # Check for group_* subdirectories
                group_dirs = [item for item in dir_contents if os.path.isdir(os.path.join(af2_chunk_dir, item)) and item.startswith('group_')]
                error_msg += f"  - group_* subdirectories found: {len(group_dirs)}\n"
                if group_dirs:
                    error_msg += f"  - group_* dirs: {group_dirs}\n"
                    # Check what's in the first group directory
                    first_group_path = os.path.join(af2_chunk_dir, group_dirs[0])
                    first_group_contents = os.listdir(first_group_path)
                    pkl_in_first_group = [f for f in first_group_contents if f.endswith('.pkl')]
                    error_msg += f"  - Contents of {group_dirs[0]}: {len(first_group_contents)} items, {len(pkl_in_first_group)} .pkl files\n"
            else:
                error_msg += f"  - Directory does not exist\n"

            error_msg += f"  - Available groups: {available_groups}\n"
            error_msg += f"  - Available chunks: {available_chunks}\n"
            error_msg += f"Mixed mode requires AF2 chunk files in group_*/af2_chunk_*.pkl format. Check directory structure or use PDB-only mode."
            raise RuntimeError(error_msg)

        # Apply chunk limit if specified (limit total chunks loaded across all groups)
        if af2_chunk_limit:
            total_chunks_loaded = 0
            limited_available_chunks = []
            for chunk_num in available_chunks:
                if total_chunks_loaded >= af2_chunk_limit:
                    break
                # Check if this chunk exists in any group
                chunk_exists_in_groups = sum(1 for g in available_groups if chunk_num in group_chunk_map[g])
                if chunk_exists_in_groups > 0:
                    limited_available_chunks.append(chunk_num)
                    total_chunks_loaded += chunk_exists_in_groups
            available_chunks = limited_available_chunks

        # Initialize cyclic loading state
        if not self.af2_state['initialized']:
            self.af2_state['group_chunk_map'] = group_chunk_map
            self.af2_state['available_groups'] = available_groups
            self.af2_state['available_chunks'] = available_chunks
            self.af2_state['current_group_idx'] = 0
            self.af2_state['current_chunk_idx'] = 0
            self.af2_state['current_chunk_data'] = []
            self.af2_state['chunk_offset'] = 0
            self.af2_state['global_state_consulted'] = False  # guards one-time global resume
            self.af2_state['initialized'] = True

            print(f"AF2 STATE: Initialized cyclic loading")
            print(f"  → {len(available_groups)} groups, {len(available_chunks)} chunks per group")
            print(f"  → Total unique chunk combinations: {len(available_groups) * len(available_chunks)}")
        else:
            print(f"AF2 STATE: Reusing existing state")
            print(f"  → Current position: group_idx={self.af2_state['current_group_idx']}, chunk_idx={self.af2_state['current_chunk_idx']}")
            print(f"  → Progress: {self.af2_state['chunks_loaded_count']} chunks loaded, {self.af2_state['total_proteins_seen']} proteins seen")

        # Update local references to point to persistent state
        self.af2_group_chunk_map = self.af2_state['group_chunk_map']
        self.af2_available_groups = self.af2_state['available_groups']
        self.af2_available_chunks = self.af2_state['available_chunks']

        if self.verbose:
            print(f"Initialized cyclic AF2 loading: {len(available_groups)} groups, {len(available_chunks)} chunks per group")
            print(f"Groups: {available_groups}")
            print(f"Chunks: {available_chunks[:20]}...{available_chunks[-10:] if len(available_chunks) > 30 else available_chunks}")
            print(f"Total available chunks: {len(available_chunks)} (first 10: {available_chunks[:10]})")
            print(f"af2_chunk_limit was: {af2_chunk_limit}")
            print(f"Estimated chunks to cycle through before repeat: {len(available_groups) * len(available_chunks)}")
            print(f"With 300 batches/epoch, 256 proteins/batch = ~{(300 * 256) // 1976} chunks per epoch (assuming ~1976 proteins/chunk)")

        # For mixed mode, create placeholder entries to enable ratio calculations
        # We'll load chunks on-demand during __getitem__

        # For mixed mode, create placeholder AF2 entries to enable ratio calculations
        # We'll load chunks on-demand during __getitem__

        # Estimate AF2 count by sampling first available chunk without loading all
        estimated_total_af2 = 0
        estimated_proteins_per_chunk = 0

        if available_groups and available_chunks:
            try:
                # Get first chunk from first group for estimation
                first_group = available_groups[0]
                first_chunk = available_chunks[0]
                if first_chunk in group_chunk_map[first_group]:
                    first_chunk_path = group_chunk_map[first_group][first_chunk]
                    with open(first_chunk_path, 'rb') as f:
                        first_chunk_data = pickle.load(f)

                    estimated_proteins_per_chunk = len(first_chunk_data) if isinstance(first_chunk_data, list) else len(first_chunk_data.keys()) if isinstance(first_chunk_data, dict) else 0
                    total_chunk_files = sum(len(chunks) for chunks in group_chunk_map.values())
                    estimated_total_af2 = estimated_proteins_per_chunk * total_chunk_files

                    print(f"AF2 estimation: ~{estimated_proteins_per_chunk} proteins/chunk × {total_chunk_files} chunks = ~{estimated_total_af2} total AF2 proteins")

            except Exception as e:
                print(f"Warning: Could not estimate AF2 count: {e}")
                estimated_total_af2 = 1000  # Fallback estimate
                estimated_proteins_per_chunk = 100  # Fallback estimate

        # Create placeholder entries for ratio calculation (will be filled lazily)
        self.af2_entries = [{'placeholder': True, 'group_chunk_idx': i // estimated_proteins_per_chunk, 'protein_index': i % estimated_proteins_per_chunk}
                           for i in range(estimated_total_af2)]

        print(f"AF2 lazy mode (mixed): Initialized {len(available_groups)} groups × {len(available_chunks)} chunks, estimated {estimated_total_af2} AF2 entries")

    def _get_global_chunk_state(self):
        """Get global chunk state using multiple persistence methods."""
        try:
            # Method 1: Environment variables (primary)
            training_run_id = f"af2_global_state_{hash(self.af2_chunk_dir or 'none')}"
            group_key = f"{training_run_id}_group_idx"
            chunk_key = f"{training_run_id}_chunk_idx"
            protein_key = f"{training_run_id}_protein_count"

            group_idx = int(os.environ.get(group_key, '0'))
            chunk_idx = int(os.environ.get(chunk_key, '0'))
            protein_count = int(os.environ.get(protein_key, '0'))

            # Method 2: Shared memory fallback (if available)
            if group_idx == 0 and chunk_idx == 0 and protein_count == 0:
                try:
                    import tempfile
                    shared_file = os.path.join(tempfile.gettempdir(), f"{training_run_id}_shared.state")
                    if os.path.exists(shared_file):
                        with open(shared_file, 'r') as f:
                            line = f.read().strip()
                            if line:
                                parts = line.split(',')
                                if len(parts) >= 3:
                                    group_idx = int(parts[0])
                                    chunk_idx = int(parts[1])
                                    protein_count = int(parts[2])
                                    print(f"FALLBACK: Loaded state from shared file: group={group_idx}, chunk={chunk_idx}, proteins={protein_count}")
                except Exception:
                    pass

            return group_idx, chunk_idx, protein_count
        except (ValueError, KeyError):
            return 0, 0, 0

    def _set_global_chunk_state(self, group_idx, chunk_idx, protein_count):
        """Set global chunk state using multiple persistence methods."""
        try:
            # Get current worker info to avoid worker conflicts
            import torch
            worker_info = torch.utils.data.get_worker_info()

            # Only worker 0 (or main process) should update global state
            # This prevents workers from overwriting each other's progress
            if worker_info is None or worker_info.id == 0:
                # Method 1: Environment variables (primary)
                training_run_id = f"af2_global_state_{hash(self.af2_chunk_dir or 'none')}"
                group_key = f"{training_run_id}_group_idx"
                chunk_key = f"{training_run_id}_chunk_idx"
                protein_key = f"{training_run_id}_protein_count"

                os.environ[group_key] = str(group_idx)
                os.environ[chunk_key] = str(chunk_idx)
                os.environ[protein_key] = str(protein_count)

                # Method 2: Shared memory backup
                try:
                    import tempfile
                    shared_file = os.path.join(tempfile.gettempdir(), f"{training_run_id}_shared.state")
                    temp_file = shared_file + '.tmp'
                    with open(temp_file, 'w') as f:
                        f.write(f"{group_idx},{chunk_idx},{protein_count}")
                    os.rename(temp_file, shared_file)
                except Exception:
                    pass

                print(f"GLOBAL STATE (WORKER-0 ONLY): Saved group_idx={group_idx}, chunk_idx={chunk_idx}, proteins={protein_count}")
            else:
                # Other workers don't update global state to prevent conflicts
                print(f"WORKER-{worker_info.id}: Skipping global state update to prevent conflicts")
        except Exception as e:
            print(f"Warning: Could not save global state: {e}")

    def _load_next_af2_chunk(self):
        """
        Load the next AF2 chunk using cyclic group-then-chunk progression.
        Pattern: group_1/chunk_001.pkl → group_2/chunk_001.pkl → ... → group_8/chunk_001.pkl →
                 group_1/chunk_002.pkl → group_2/chunk_002.pkl → ... → group_8/chunk_002.pkl → ...
        """
        if not self.af2_state['available_groups'] or not self.af2_state['available_chunks']:
            print("No AF2 groups or chunks available")
            return False

        # On the very first chunk load (before worker_init_fn has run, i.e. num_workers=0 /
        # main-process mode), check global state to resume across restarts.
        # worker_init_fn sets current_group_idx/chunk_idx directly and marks
        # global_state_consulted=True so this block is skipped in worker processes.
        if not self.af2_state.get('global_state_consulted', False):
            self.af2_state['global_state_consulted'] = True

            # Only meaningful in single-worker (num_workers=0) runs; with persistent workers
            # the position is already set by worker_init_fn and we should not override it.
            import torch
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is None:
                global_group_idx, global_chunk_idx, global_protein_count = self._get_global_chunk_state()
                print(f"GLOBAL STATE CHECK: First load - global_state: group_idx={global_group_idx}, chunk_idx={global_chunk_idx}, proteins={global_protein_count}")
                if global_protein_count > 0:
                    if global_group_idx < len(self.af2_state['available_groups']):
                        self.af2_state['current_group_idx'] = global_group_idx
                    if global_chunk_idx < len(self.af2_state['available_chunks']):
                        self.af2_state['current_chunk_idx'] = global_chunk_idx
                    self.af2_state['total_proteins_seen'] = global_protein_count
                    print(f"GLOBAL STATE: Resuming from group_idx={global_group_idx}, chunk_idx={global_chunk_idx}, proteins={global_protein_count}")
                else:
                    print(f"GLOBAL STATE: No resume state, starting from beginning")
            else:
                print(f"WORKER STATE: Using worker-init position - group_idx={self.af2_state['current_group_idx']}, chunk_idx={self.af2_state['current_chunk_idx']}")
        else:
            print(f"WORKER STATE: Using current worker position - group_idx={self.af2_state['current_group_idx']}, chunk_idx={self.af2_state['current_chunk_idx']}, proteins={self.af2_state['total_proteins_seen']}")

        # Get current group and chunk numbers from (possibly updated) state
        current_group_num = self.af2_state['available_groups'][self.af2_state['current_group_idx']]
        current_chunk_num = self.af2_state['available_chunks'][self.af2_state['current_chunk_idx']]

        print(f"LOADING: group_{current_group_num}/chunk_{current_chunk_num:03d} (group_idx={self.af2_state['current_group_idx']}, chunk_idx={self.af2_state['current_chunk_idx']}) [TOTAL: {self.af2_state['total_proteins_seen']} proteins]")

        # Check if current chunk exists in current group
        if current_chunk_num not in self.af2_group_chunk_map[current_group_num]:
            available_in_group = list(self.af2_group_chunk_map[current_group_num].keys())
            print(f"Chunk {current_chunk_num} not found in group {current_group_num}")
            print(f"   Available chunks in group {current_group_num}: {available_in_group[:10]}...{available_in_group[-5:] if len(available_in_group) > 15 else ''}")
            print(f"   Advancing to next group...")

            # CRITICAL FIX: Add recursion protection to prevent infinite loops
            if not hasattr(self, '_recursion_count'):
                self._recursion_count = 0
            self._recursion_count += 1

            if self._recursion_count > 50:  # Prevent infinite recursion
                print(f"ERROR: Too many recursive calls ({self._recursion_count}), breaking recursion")
                print(f"Available groups: {self.af2_state['available_groups']}")
                print(f"Available chunks: {self.af2_state['available_chunks']}")
                raise RuntimeError("Chunk loading recursion limit exceeded - possibly corrupted chunk directory structure")

            self._advance_to_next_group_or_chunk()
            result = self._load_next_af2_chunk()  # Try again with next group/chunk
            self._recursion_count -= 1  # Decrement on successful return
            return result

        chunk_path = self.af2_group_chunk_map[current_group_num][current_chunk_num]

        try:
            # Robust pickle loading with numpy module-path repair. Chunks pickled
            # under one numpy major version can fail to unpickle under another.
            try:
                with open(chunk_path, 'rb') as f:
                    chunk_data = pickle.load(f)
            except (ModuleNotFoundError, ImportError) as e:
                if 'numpy._core' in str(e):
                    # Repair numpy._core module aliasing inside worker processes
                    import sys

                    import numpy as np

                    # Try to fix the specific module that's missing
                    missing_module = str(e).split("'")[1] if "'" in str(e) else ""
                    if missing_module.startswith('numpy._core.'):
                        base_name = missing_module.replace('numpy._core.', '')
                        # Try different numpy module paths
                        for base in [f'numpy.{base_name}', f'numpy.core.{base_name}']:
                            try:
                                base_mod = __import__(base, fromlist=[''])
                                sys.modules[missing_module] = base_mod
                                break
                            except ImportError:
                                continue

                    # Retry pickle loading after fixing modules
                    with open(chunk_path, 'rb') as f:
                        chunk_data = pickle.load(f)
                else:
                    raise  # Re-raise if it's not a numpy._core issue

            # Handle different chunk data formats
            processed_data = []
            total_entries = 0

            # Check if chunk_data is a dict (protein_id -> protein_data) or list
            if isinstance(chunk_data, dict):
                print(f"CHUNK FORMAT: Dictionary with {len(chunk_data)} protein entries")
                total_entries = len(chunk_data)

                # Convert dict format to list format expected by processing logic
                protein_list = []
                for protein_id, protein_data in chunk_data.items():
                    if isinstance(protein_data, dict):
                        # Add the protein_id as the name if not already present
                        if 'name' not in protein_data:
                            protein_data['name'] = protein_id
                        protein_list.append(protein_data)
                    else:
                        if _IFD_DEBUG: print(f"DEBUG: Skipping protein {protein_id} - data is not a dict: {type(protein_data)}")

                chunk_data = protein_list  # Replace with list format

            elif isinstance(chunk_data, list):
                print(f"CHUNK FORMAT: List with {len(chunk_data)} entries")
                total_entries = len(chunk_data)
            else:
                print(f"CHUNK FORMAT ERROR: Unexpected type {type(chunk_data)}")
                print(f"  Content preview: {str(chunk_data)[:200]}...")
                self._advance_to_next_group_or_chunk()
                return self._load_next_af2_chunk()  # Skip problematic chunk

            if _IFD_DEBUG: print(f"DEBUG: Processing chunk with {len(chunk_data)} entries after format conversion")

            filtered_no_dict = 0
            filtered_no_seq = 0
            filtered_no_coords = 0
            filtered_too_long = 0

            for i, protein_data in enumerate(chunk_data):
                if not isinstance(protein_data, dict):
                    filtered_no_dict += 1
                    if i < 3:  # Show first few for debugging
                        if _IFD_DEBUG: print(f"DEBUG: Entry {i} not a dict: {type(protein_data)}")
                        print(f"DEBUG: Entry {i} content: {repr(protein_data)[:200]}...")  # Show first 200 chars
                    continue

                if 'seq' not in protein_data:
                    filtered_no_seq += 1
                    if i < 3:  # Show first few for debugging
                        if _IFD_DEBUG: print(f"DEBUG: Entry {i} missing 'seq' key. Keys: {list(protein_data.keys())}")
                    continue

                if 'coords' not in protein_data:
                    filtered_no_coords += 1
                    if i < 3:  # Show first few for debugging
                        if _IFD_DEBUG: print(f"DEBUG: Entry {i} missing 'coords' key. Keys: {list(protein_data.keys())}")
                    continue

                if self.max_len is not None and len(protein_data['seq']) > self.max_len:
                    filtered_too_long += 1
                    if i < 3:  # Show first few for debugging
                        if _IFD_DEBUG: print(f"DEBUG: Entry {i} too long: {len(protein_data['seq'])} > {self.max_len}")
                    continue

                # Entry passed all filters
                entry = protein_data.copy()
                entry['source'] = 'af2'

                # Create unique name with group and chunk info if not already present
                if 'name' not in entry or not entry['name']:
                    entry['name'] = f"group_{current_group_num}_chunk_{current_chunk_num:03d}_{len(processed_data)}"

                # Apply pLDDT-based segment chopping (no-op when threshold == 0)
                segments = self._chop_low_plddt_segments(entry)
                processed_data.extend(segments)

            # Report filtering results - ALWAYS log for debugging data loss
            chunk_basename = os.path.basename(chunk_path)
            efficiency = (len(processed_data) / total_entries * 100) if total_entries > 0 else 0
            print(f"CHUNK ANALYSIS: {chunk_basename} (group_{current_group_num}/chunk_{current_chunk_num:03d})")
            print(f"  → Total entries: {total_entries}")
            print(f"  → Filtered (not dict): {filtered_no_dict}")
            print(f"  → Filtered (no 'seq'): {filtered_no_seq}")
            print(f"  → Filtered (no 'coords'): {filtered_no_coords}")
            print(f"  → Filtered (too long): {filtered_too_long}")
            print(f"  → Valid entries: {len(processed_data)} ({efficiency:.1f}% efficiency)")

            # Flag problematic chunks
            if total_entries < 100:
                print(f"    TINY CHUNK WARNING: Only {total_entries} entries!")
            elif efficiency < 50:
                print(f"    LOW EFFICIENCY WARNING: {efficiency:.1f}% pass rate!")
            elif efficiency > 95:
                print(f"   HIGH EFFICIENCY: {efficiency:.1f}% pass rate")

            # Update persistent state with new chunk data
            self.af2_state['current_chunk_data'] = processed_data
            self.af2_state['chunk_offset'] = 0
            self.af2_state['chunks_loaded_count'] += 1
            self.af2_state['total_proteins_seen'] += len(processed_data)

            # Track chunk size statistics for analysis
            if 'chunk_sizes' not in self.af2_state:
                self.af2_state['chunk_sizes'] = []
            self.af2_state['chunk_sizes'].append(total_entries)

            # Log summary statistics every 10 chunks
            if self.af2_state['chunks_loaded_count'] % 10 == 0:
                sizes = self.af2_state['chunk_sizes']
                avg_size = sum(sizes) / len(sizes) if sizes else 0
                min_size = min(sizes) if sizes else 0
                max_size = max(sizes) if sizes else 0
                tiny_chunks = sum(1 for s in sizes if s < 100)
                print(f"\n CHUNK STATISTICS (after {self.af2_state['chunks_loaded_count']} chunks):")
                print(f"   Total unique proteins discovered: {len(self.af2_state.get('seen_proteins', set()))}")
                print(f"   Average chunk size: {avg_size:.1f} proteins")
                print(f"   Size range: {min_size} - {max_size} proteins")
                print(f"   Tiny chunks (<100 proteins): {tiny_chunks}/{len(sizes)} ({tiny_chunks/len(sizes)*100:.1f}%)")
                print(f"   Data utilization: {self.af2_state['total_proteins_seen']/self.af2_state['chunks_loaded_count']:.1f} proteins/chunk average\n")

            # Update local references
            self.af2_current_chunk_data = self.af2_state['current_chunk_data']
            self.af2_chunk_offset = self.af2_state['chunk_offset']

            # Extract protein names for diversity tracking
            protein_names_in_chunk = []
            for entry in processed_data:
                if isinstance(entry, dict) and 'name' in entry:
                    protein_names_in_chunk.append(entry['name'])

            print(f"Loaded AF2 group_{current_group_num}/chunk_{current_chunk_num:03d} with {len(processed_data)} entries")
            print(f"  → Protein diversity sample: {protein_names_in_chunk[:3]}...{protein_names_in_chunk[-2:] if len(protein_names_in_chunk) > 5 else ''}")
            print(f"Progress: group {self.af2_state['current_group_idx'] + 1}/{len(self.af2_state['available_groups'])}, chunk {self.af2_state['current_chunk_idx'] + 1}/{len(self.af2_state['available_chunks'])}")
            print(f"  → AF2 STATE: {self.af2_state['chunks_loaded_count']} chunks loaded, {self.af2_state['total_proteins_seen']} total proteins seen")

            # Advance to next group (or next chunk if we've cycled through all groups)
            self._advance_to_next_group_or_chunk()

            # Save global state for cross-epoch continuity
            self._set_global_chunk_state(
                self.af2_state['current_group_idx'],
                self.af2_state['current_chunk_idx'],
                self.af2_state['total_proteins_seen']
            )

            return True

        except Exception as e:
            print(f"ERROR loading chunk {chunk_path}: {e}")
            print(f"   Advancing to next chunk...")
            # Skip this group/chunk combination and try the next one
            self._advance_to_next_group_or_chunk()
            return self._load_next_af2_chunk()  # Try next group/chunk

    def _advance_to_next_group_or_chunk(self):
        """
        Advance to the next group within the same chunk, or to the next chunk if we've
        cycled through all groups.

        FIXED: Worker-specific advancement to prevent conflicts and ensure full coverage.
        Each worker advances through its allocated chunk space only.
        """
        # Get current worker info to enable worker-specific logic
        import torch
        worker_info = torch.utils.data.get_worker_info()
        worker_rank = worker_info.id if worker_info else 0

        # Debug: Show current state before advancement
        current_group_num = self.af2_state['available_groups'][self.af2_state['current_group_idx']] if self.af2_state['current_group_idx'] < len(self.af2_state['available_groups']) else "INVALID"
        current_chunk_num = self.af2_state['available_chunks'][self.af2_state['current_chunk_idx']] if self.af2_state['current_chunk_idx'] < len(self.af2_state['available_chunks']) else "INVALID"
        print(f"Worker-{worker_rank} BEFORE advance: group_idx={self.af2_state['current_group_idx']}/{len(self.af2_state['available_groups'])} (group_{current_group_num}), chunk_idx={self.af2_state['current_chunk_idx']}/{len(self.af2_state['available_chunks'])} (chunk_{current_chunk_num}) [TOTAL: {self.af2_state['total_proteins_seen']} proteins]")

        # Check if worker has allocated chunk space (from worker_init_fn)
        if ('worker_chunk_base' in self.af2_state and
            'worker_chunk_limit' in self.af2_state and
            'worker_rank' in self.af2_state):

            # Worker-specific advancement within allocated slice.
            # Linear space encoding: pos = chunk_idx * num_groups + group_idx
            # (groups are the fast-cycling dimension, chunks are the slow dimension)
            worker_chunk_base = self.af2_state['worker_chunk_base']
            worker_chunk_limit = self.af2_state['worker_chunk_limit']
            available_groups = self.af2_state['available_groups']
            available_chunks = self.af2_state['available_chunks']
            num_groups = len(available_groups)
            num_chunks = len(available_chunks)

            # Move to next group first (groups cycle fast)
            self.af2_state['current_group_idx'] += 1

            # If we've gone through all groups for this chunk, advance to next chunk
            if self.af2_state['current_group_idx'] >= num_groups:
                self.af2_state['current_group_idx'] = 0  # Reset to first group
                next_chunk_idx = self.af2_state['current_chunk_idx'] + 1

                # Compute next linear position (chunk_idx * num_groups + 0, group already reset)
                next_linear_pos = next_chunk_idx * num_groups

                # Check if we've reached the worker's chunk limit
                if next_linear_pos >= worker_chunk_limit:
                    # Wrap around to beginning of worker's allocated space
                    next_linear_pos = worker_chunk_base
                    print(f"Worker-{worker_rank}: Reached chunk limit {worker_chunk_limit}, wrapping to base {worker_chunk_base}")

                # Decode: pos = chunk_idx * num_groups + group_idx
                new_chunk_idx = (next_linear_pos // num_groups) % num_chunks
                new_group_idx = next_linear_pos % num_groups

                self.af2_state['current_group_idx'] = new_group_idx
                self.af2_state['current_chunk_idx'] = new_chunk_idx

                print(f"Worker-{worker_rank}: Advanced to linear pos {next_linear_pos} → group_{available_groups[new_group_idx]}/chunk_{available_chunks[new_chunk_idx]:03d}")
                print(f"Worker-{worker_rank}: Allocated space: {worker_chunk_base} to {worker_chunk_limit-1}")

        else:
            # Fallback: Original advancement logic for backwards compatibility
            print(f"Worker-{worker_rank}: No allocated chunk space, using legacy advancement")

            # Move to next group
            self.af2_state['current_group_idx'] += 1

            # If we've gone through all groups, move to next chunk and reset to first group
            if self.af2_state['current_group_idx'] >= len(self.af2_state['available_groups']):
                print(f"Completed all groups for chunk {current_chunk_num}, advancing to next chunk...")
                self.af2_state['current_group_idx'] = 0  # Reset to first group

                # Normal progression to next chunk
                self.af2_state['current_chunk_idx'] += 1

                # If we've gone through all available chunks, wrap around to first chunk
                if self.af2_state['current_chunk_idx'] >= len(self.af2_state['available_chunks']):
                    self.af2_state['current_chunk_idx'] = 0
                    print("Completed full cycle through all available chunks, wrapping around to beginning")
                else:
                    next_chunk_num = self.af2_state['available_chunks'][self.af2_state['current_chunk_idx']]
                    print(f"Advanced to chunk index {self.af2_state['current_chunk_idx']} (chunk {next_chunk_num})")

        # Update local references to match global state
        self.af2_current_group_idx = self.af2_state['current_group_idx']
        self.af2_current_chunk_idx = self.af2_state['current_chunk_idx']

        # Debug: Show state after advancement
        new_group_num = self.af2_state['available_groups'][self.af2_state['current_group_idx']] if self.af2_state['current_group_idx'] < len(self.af2_state['available_groups']) else "INVALID"
        new_chunk_num = self.af2_state['available_chunks'][self.af2_state['current_chunk_idx']] if self.af2_state['current_chunk_idx'] < len(self.af2_state['available_chunks']) else "INVALID"
        print(f"Worker-{worker_rank} AFTER advance: group_idx={self.af2_state['current_group_idx']}/{len(self.af2_state['available_groups'])} (group_{new_group_num}), chunk_idx={self.af2_state['current_chunk_idx']}/{len(self.af2_state['available_chunks'])} (chunk_{new_chunk_num}) [TOTAL: {self.af2_state['total_proteins_seen']} proteins]")
        print(f"Worker-{worker_rank} Next target: group_{new_group_num}/chunk_{new_chunk_num:03d}")

    def __len__(self):
        """Return the number of samples in the dataset."""
        # Debug prints only in verbose mode to avoid slowing down non-verbose training
        if getattr(self, 'verbose', False):
            if _IFD_DEBUG: print(f"DEBUG: __len__ called - af2_state exists: {hasattr(self, 'af2_state')}")
            if hasattr(self, 'af2_state'):
                if _IFD_DEBUG: print(f"DEBUG: af2_state.group_chunk_map exists: {self.af2_state.get('group_chunk_map') is not None}")

        # For AF2 lazy loading (both pure AF2 and mixed mode), return a large number
        # so PyTorch doesn't limit workers
        if hasattr(self, 'af2_state') and self.af2_state.get('group_chunk_map'):
            # Total AF2 proteins: ~10 million across 8 groups
            total_groups = len(self.af2_state.get('available_groups', []))
            total_chunks = len(self.af2_state.get('available_chunks', []))

            if total_groups > 0 and total_chunks > 0:
                # Estimate: 10M proteins / 8 groups / 690 chunks ≈ 1,815 proteins/chunk
                estimated_proteins_per_chunk = 10_000_000 // (total_groups * total_chunks)
                estimated_total = total_groups * total_chunks * estimated_proteins_per_chunk

                # For dynamic mixed mode, return large number to ensure continuous AF2 loading
                # This ensures workers get enough work to keep loading new chunks
                if hasattr(self, 'ratio_af2_pdb') and self.ratio_af2_pdb > 0:
                    # Dynamic mixed mode: return large number to ensure continuous chunk loading
                    if getattr(self, 'verbose', False):
                        if _IFD_DEBUG: print(f"DEBUG: Dynamic mixed mode __len__ returning {max(estimated_total, 10_000_000)} to enable continuous AF2 chunk loading")
                    return max(estimated_total, 10_000_000)
                else:
                    # Pure AF2 mode
                    if getattr(self, 'verbose', False):
                        if _IFD_DEBUG: print(f"DEBUG: Pure AF2 mode __len__ returning {max(estimated_total, 10_000_000)}")
                    return max(estimated_total, 10_000_000)
            else:
                if getattr(self, 'verbose', False):
                    if _IFD_DEBUG: print(f"DEBUG: AF2 state incomplete - returning fallback 10M")
                return 10_000_000  # Conservative fallback
        else:
            # Standard mode: return actual entries count
            if getattr(self, 'verbose', False):
                if _IFD_DEBUG: print(f"DEBUG: Standard mode __len__ returning {len(self.entries)}")
            return len(self.entries)

    def __getitem__(self, idx):
        """
        Get a single protein structure as a graph.

        For mixed mode: Dynamically serves AF2 or PDB based on ratio pattern while continuously
        loading new AF2 chunks (no fixed entry list).

        For AF2-only mode: Uses lazy loading from chunks.
        For PDB-only mode: Uses standard entry list.

        Args:
            idx: Index of the protein to load

        Returns:
            data: PyTorch Geometric Data object
            y: one-hot encoded sequence tensor [L, 20]
            mask: boolean tensor [L]
        """
        # Debug: Track unique indices being requested
        if self.verbose and idx < 10:
            import torch
            worker_info = torch.utils.data.get_worker_info()
            worker_id = worker_info.id if worker_info else "main"
            print(f"[DEBUG DATASET] Worker-{worker_id} requesting idx={idx} (dataset_len={len(self)})")

        # Initialize actual_idx for all paths (needed for logging)
        actual_idx = idx  # Default fallback for all modes

        # Mixed mode: Probabilistic AF2/PDB serving based on ratio
        if (hasattr(self, 'af2_state') and self.af2_state['group_chunk_map'] and
            hasattr(self, 'ratio_af2_pdb') and self.ratio_af2_pdb > 0):

            # Safety check: ensure mixed_rng is available
            if not hasattr(self, 'mixed_rng'):
                print("WARNING: mixed_rng not initialized, re-initializing...")
                import random
                self.mixed_rng = random.Random(self.seed + self.rank)

            # Probabilistic AF2/PDB decision.
            # pdb_probability is clamped in _clamp_mix_probabilities so it is
            # never so extreme that one source is effectively excluded from a
            # full batch (see _init_mixed_mode_tracking / set_af2_ratio).
            if self.mixed_rng.random() < self.pdb_probability:
                entry = self._get_next_pdb_entry()
            else:
                entry = self._get_next_af2_entry()

        # Pure AF2 mode: Lazy loading from chunks
        elif (hasattr(self, 'af2_state') and self.af2_state['group_chunk_map']):
            entry = self._get_next_af2_entry()

        # PDB-only mode or standard mode: Use entry list
        else:
            # Handle deterministic cycling
            if self.deterministic:
                # For multi-worker DataLoaders, each worker needs different starting points
                import torch
                worker_info = torch.utils.data.get_worker_info()
                if worker_info is not None:
                    # Multi-worker case: distribute indices across workers
                    worker_id = worker_info.id
                    num_workers = worker_info.num_workers
                    # Each worker gets every num_workers-th index starting from worker_id
                    actual_idx = (idx * num_workers + worker_id) % len(self.entries)

                    if self.verbose and idx < 5:
                        print(f"[DEBUG WORKER] Worker-{worker_id}/{num_workers} mapping idx={idx} -> actual_idx={actual_idx}")
                else:
                    # Single-worker case: use distributed training logic
                    actual_idx = (self.current_index) % len(self.entries)
                    self.current_index += self.world_size  # Skip ahead by world_size for next access

                entry = self.entries[actual_idx]
            else:
                # Random access (original behavior)
                entry = self.entries[idx]

        try:
            # Use shared processing function to ensure identical behavior to CATH dataset
            from .shared_processing import process_protein_entry

            # Determine source type for this entry
            source = entry.get('source', 'unknown')

            # Use shared processing with relaxed validation for AF2 (allows unknown residues)
            # but strict validation for PDB to match original CATH behavior
            strict_validation = (source == 'pdb')

            # FIXED: Sample time value FIRST (before graph building) to enable time-dependent structure noise
            if self.time_sampling_strategy == 'uniform':
                import torch
                time_value = torch.rand(1).item() * (self.t_max - self.t_min) + self.t_min
            elif self.time_sampling_strategy == 'exponential':
                import torch
                exp_sample = torch.distributions.Exponential(rate=1.0).sample((1,)).item()
                time_value = self.t_min + exp_sample * self.alpha_range
                time_value = max(self.t_min, min(self.t_max, time_value))  # Clamp to [t_min, t_max]
            else:
                # Default to 0.0 if no strategy is specified
                raise Exception("time sampling strategy must be specified.")

            # Log time coefficient for structure noise debugging (verbose mode only)
            if self.verbose and idx < 5:  # Only log for first few samples to avoid spam
                noise_enabled = getattr(self.graph_builder, 'structure_noise_mag_std', 0) > 0
                noise_type = getattr(self.graph_builder, 'time_based_struct_noise', 'fixed')
                print(f"[UNIFIED DATASET] Entry {idx}: time_value={time_value:.3f}, noise_enabled={noise_enabled}, noise_type={noise_type}")

            # FIXED: Pass time_param to enable time-dependent structure noise
            data, y, mask, dssp_targets = process_protein_entry(
                self.graph_builder,
                entry,
                source,
                strict_validation=strict_validation,
                time_param=time_value  # CRITICAL FIX: Now structure noise can be time-dependent!
            )

            # Track protein for multi-worker coordination in verbose mode
            if self.protein_tracker and self.verbose:
                protein_name = entry.get('name', f'unknown_{actual_idx}')
                self.protein_tracker.add_proteins([protein_name])

                # Debug: Log first few protein additions to verify tracking is working
                if idx < 5:
                    import multiprocessing as mp
                    worker_info = None
                    if hasattr(torch.utils.data, 'get_worker_info'):
                        worker_info = torch.utils.data.get_worker_info()
                    worker_id = worker_info.id if worker_info else "main"
                    print(f"[DEBUG TRACKER] Worker-{worker_id} added protein: {protein_name} (idx={idx}, actual_idx={actual_idx})")
            elif self.verbose:
                # Debug: Check why tracking isn't happening
                protein_name = entry.get('name', f'unknown_{idx}')
                has_tracker = self.protein_tracker is not None
                if idx < 5:
                    print(f"[DEBUG TRACKER SKIP] idx={idx}, protein={protein_name}, has_tracker={has_tracker}, verbose={self.verbose}")

            return data, y, mask, time_value, dssp_targets

        except Exception as e:
            print(f"Error processing entry {entry.get('name', 'unknown')}: {e}")
            raise

    def __iter__(self):
        """
        Custom iterator that preserves AF2 chunk state across epoch boundaries.

        With persistent_workers=True (multi-worker), each worker process retains its own
        af2_state between epochs and advances independently — no intervention needed here.

        In single-worker (num_workers=0) mode, this runs in the main process and we restore
        the saved global state so that restarts resume from the right position.
        """
        import torch
        worker_info = torch.utils.data.get_worker_info()

        # Only restore global state in the main (non-worker) process for single-worker runs.
        # In multi-worker mode workers manage their own state via worker_init_fn.
        if worker_info is None and hasattr(self, 'af2_state') and self.af2_state.get('group_chunk_map'):
            global_group_idx, global_chunk_idx, global_protein_count = self._get_global_chunk_state()

            if global_protein_count > 0:
                print(f"[ITERATOR] Restoring AF2 state from global: group={global_group_idx}, chunk={global_chunk_idx}, proteins={global_protein_count}")

                available_groups = self.af2_state.get('available_groups', [])
                available_chunks = self.af2_state.get('available_chunks', [])

                if available_groups and global_group_idx < len(available_groups):
                    self.af2_state['current_group_idx'] = global_group_idx
                if available_chunks and global_chunk_idx < len(available_chunks):
                    self.af2_state['current_chunk_idx'] = global_chunk_idx

                # Clear chunk data to force reload from the restored position
                self.af2_state['current_chunk_data'] = []
                self.af2_state['chunk_offset'] = 0
                self.af2_state['total_proteins_seen'] = global_protein_count
                # Reset the flag so the global state isn't consulted again mid-epoch
                self.af2_state['global_state_consulted'] = True

        # Use default PyTorch Dataset iterator
        return super().__iter__()

    def reset_iteration(self):
        """Reset deterministic iteration counter."""
        if self.deterministic:
            self.current_index = self.rank


def create_unified_dataloader(
    # PDB parameters
    split_json: Optional[str] = None,
    map_pkl: Optional[str] = None,
    split: str = 'train',

    # AF2 parameters
    af2_chunk_dir: Optional[str] = None,
    af2_chunk_limit: Optional[int] = None,
    # NOTE: AF2 data is ALWAYS loaded lazily - no upfront loading option

    # Mixing parameters
    ratio_af2_pdb: int = 0,
    heterogeneous_batches: bool = True,  # Default to heterogeneous (mixed AF2+PDB per batch)

    # DataLoader parameters
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = True,
    shuffle: bool = True,

    # Common parameters
    max_len: Optional[int] = None,
    graph_builder_kwargs: Optional[Dict] = None,
    verbose: bool = False,  # Control debug output and protein counting

    # Time sampling parameters
    time_sampling_strategy: str = None,
    t_min: float = 0.0,
    t_max: float = 8.0,
    alpha_range: float = None,

    # Iteration control
    deterministic: bool = True,

    # Distributed training
    rank: int = 0,
    world_size: int = 1,
    seed: int = 42,

    # AF2 pLDDT-based segment chopping
    af2_plddt_threshold: float = 0.0,
    af2_min_segment_len: int = 60):
    """
    Create unified DataLoader using the same simple approach as original CATH dataset.

    Returns:
        DataLoader configured for unified AF2+PDB training
    """
    from torch.utils.data import DataLoader

    from training.collate import collate_fn

    # Initialize global protein tracker for multi-worker coordination in verbose mode
    print(f"[DEBUG] create_unified_dataloader: verbose={verbose}, num_workers={num_workers}")
    if verbose and num_workers > 0:
        print(f"[DEBUG] About to initialize global protein tracker...")
        initialize_global_protein_tracker(num_workers, verbose=True)
        print(f"[DEBUG] Global protein tracker initialization completed")

    # Create unified dataset
    dataset = UnifiedDataset(
        split_json=split_json,
        map_pkl=map_pkl,
        split=split,
        af2_chunk_dir=af2_chunk_dir,
        af2_chunk_limit=af2_chunk_limit,
        ratio_af2_pdb=ratio_af2_pdb,
        heterogeneous_batches=heterogeneous_batches,
        max_len=max_len,
        graph_builder_kwargs=graph_builder_kwargs,
        verbose=verbose,
        time_sampling_strategy=time_sampling_strategy,
        t_min=t_min,
        t_max=t_max,
        alpha_range=alpha_range,
        deterministic=deterministic,
        rank=rank,
        world_size=world_size,
        seed=seed,
        af2_plddt_threshold=af2_plddt_threshold,
        af2_min_segment_len=af2_min_segment_len,
        batch_size=batch_size,
    )

    # Create sampler for distributed training
    sampler = None
    if world_size > 1:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            seed=seed,
            shuffle=shuffle and not deterministic  # Don't shuffle if deterministic
        )
        shuffle = False  # Disable shuffle when using DistributedSampler
    elif deterministic:
        shuffle = False  # Don't shuffle if deterministic

    # Define worker initialization function for multi-worker coordination
    def worker_init_fn(worker_id):
        """
        Initialize each DataLoader worker to start from a different AF2 chunk position.
        This ensures that with 8 workers, they don't all process the same proteins.
        FIXED: Each worker gets completely isolated chunk progression to prevent conflicts.

        Also repairs worker processes that do not properly
        inherit the conda environment or have numpy module path issues.
        """
        import os
        import sys

        import torch

        # Ensure numpy's module aliases are present in each DataLoader worker.
        # Spawned workers do not always inherit them, which breaks unpickling.
        try:
            # Force reload of critical modules in worker process
            import importlib

            # Ensure numpy is properly available
            import numpy as np

            # Test and repair numpy._core module availability
            core_modules = ['numpy._core.numeric', 'numpy._core.multiarray', 'numpy._core.umath']
            for module_name in core_modules:
                try:
                    __import__(module_name)
                except ImportError:
                    # Create the numpy 1.x/2.x compatibility mapping
                    base_module = module_name.replace('numpy._core.', 'numpy.')
                    try:
                        base_mod = __import__(base_module, fromlist=[''])
                        sys.modules[module_name] = base_mod
                    except ImportError:
                        # Try numpy.core instead of numpy._core
                        core_module = module_name.replace('numpy._core.', 'numpy.core.')
                        try:
                            core_mod = __import__(core_module, fromlist=[''])
                            sys.modules[module_name] = core_mod
                        except ImportError:
                            pass  # Skip if neither works

            # Ensure pickle module is fresh in worker
            importlib.reload(__import__('pickle'))

        except Exception as e:
            print(f"Worker {worker_id} environment setup warning: {e}")

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Get the dataset instance for this worker
            dataset = worker_info.dataset

            # If this dataset has AF2 state (lazy loading), coordinate workers
            if hasattr(dataset, 'af2_state') and dataset.af2_state.get('group_chunk_map'):
                # FIXED: Get global state for reference, but don't let workers interfere
                global_group_idx, global_chunk_idx, global_protein_count = dataset._get_global_chunk_state()

                print(f"Worker {worker_info.id}: Global state reference - group={global_group_idx}, chunk={global_chunk_idx}, proteins={global_protein_count}")

                # Calculate deterministic, isolated starting position for this worker
                total_workers = worker_info.num_workers
                worker_rank = worker_info.id

                # Get available chunk space
                available_groups = dataset.af2_state.get('available_groups', [])
                available_chunks = dataset.af2_state.get('available_chunks', [])

                if available_groups and available_chunks:
                    # Each worker gets 1/N of the total chunk space, non-overlapping.
                    # Linear encoding: pos = chunk_idx * num_groups + group_idx
                    # (matches _advance_to_next_group_or_chunk)
                    num_groups = len(available_groups)
                    num_chunks = len(available_chunks)
                    total_chunk_combinations = num_groups * num_chunks
                    chunks_per_worker = total_chunk_combinations // total_workers
                    worker_start_offset = worker_rank * chunks_per_worker

                    # Decode starting position: pos = chunk_idx * num_groups + group_idx
                    worker_start_chunk_idx = (worker_start_offset // num_groups) % num_chunks
                    worker_start_group_idx = worker_start_offset % num_groups

                    dataset.af2_state['current_group_idx'] = worker_start_group_idx
                    dataset.af2_state['current_chunk_idx'] = worker_start_chunk_idx

                    print(f"Worker {worker_rank}: ISOLATED ASSIGNMENT")
                    print(f"  → Total combinations: {total_chunk_combinations}")
                    print(f"  → Chunks per worker: {chunks_per_worker}")
                    print(f"  → Worker offset: {worker_start_offset}")
                    print(f"  → Assigned: group_{available_groups[worker_start_group_idx]}/chunk_{available_chunks[worker_start_chunk_idx]:03d}")
                    print(f"  → Worker chunk range: {worker_start_offset} to {worker_start_offset + chunks_per_worker - 1}")

                    # Store worker allocation bounds for _advance_to_next_group_or_chunk
                    dataset.af2_state['worker_chunk_base'] = worker_start_offset
                    dataset.af2_state['worker_chunk_limit'] = worker_start_offset + chunks_per_worker
                    dataset.af2_state['worker_rank'] = worker_rank

                    # Debug: Show all worker assignments for verification
                    if worker_rank == 0:
                        print(f"  → DEBUG: Worker chunk space allocation:")
                        for w in range(total_workers):
                            w_start_offset = w * chunks_per_worker
                            w_start_chunk_idx = (w_start_offset // num_groups) % num_chunks
                            w_start_group_idx = w_start_offset % num_groups
                            w_end_offset = w_start_offset + chunks_per_worker - 1
                            print(f"     Worker {w}: offset {w_start_offset}-{w_end_offset}, starts at group_{available_groups[w_start_group_idx]}/chunk_{available_chunks[w_start_chunk_idx]:03d}")

                elif available_groups:
                    # Fallback: group-only distribution
                    target_group_idx = worker_rank % len(available_groups)
                    dataset.af2_state['current_group_idx'] = target_group_idx
                    dataset.af2_state['current_chunk_idx'] = 0
                    print(f"Worker {worker_rank}: FALLBACK - Assigned to group {available_groups[target_group_idx]}")
                else:
                    print(f"Worker {worker_rank}: No AF2 groups/chunks available for coordination")

                # Mark global state as already consulted so _load_next_af2_chunk won't
                # override the position we just set above.
                dataset.af2_state['global_state_consulted'] = True

                # Each worker tracks its own protein count independently
                dataset.af2_state['total_proteins_seen'] = 0

                # Clear any existing chunk data to force reload from new position
                dataset.af2_state['current_chunk_data'] = []
                dataset.af2_state['chunk_offset'] = 0

                print(f"Worker {worker_rank}: Initialization complete - isolated chunk progression enabled")

                # Update local references
                if hasattr(dataset, 'af2_current_chunk_data'):
                    dataset.af2_current_chunk_data = []
                if hasattr(dataset, 'af2_chunk_offset'):
                    dataset.af2_chunk_offset = 0

                print(f"Worker {worker_rank}: Initialized with ISOLATED AF2 coordination")

    if _IFD_DEBUG: print(f"DEBUG: num_workers = {num_workers}")

    # Create dataloader with worker coordination
    if _IFD_DEBUG: print(f"DEBUG: About to create DataLoader with dataset length: {len(dataset)}")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,  # CRITICAL: Keep workers alive across epochs
        collate_fn=collate_fn,  # Use original collate function!
        worker_init_fn=worker_init_fn if num_workers > 0 else None  # Only use for multi-worker
    )
    if _IFD_DEBUG: print(f"DEBUG: DataLoader created with {num_workers} workers, persistent_workers={num_workers > 0}")

    return dataloader


def get_global_protein_counts():
    """Get protein counts from the global tracker"""
    global _global_protein_tracker
    if _global_protein_tracker is not None:
        return _global_protein_tracker.get_counts()
    return 0, 0


def reset_global_protein_epoch():
    """Reset epoch protein tracking in the global tracker"""
    global _global_protein_tracker
    if _global_protein_tracker is not None:
        _global_protein_tracker.reset_epoch()
