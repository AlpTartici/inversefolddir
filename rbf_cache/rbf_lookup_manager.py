#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025-2026 Alp Tartici, Stanford University.
# Licensed under the MIT License.

"""
RBF Lookup Table Manager

Handles creation, saving, and loading of RBF lookup tables for fast edge feature computation.

The RBFLookupManager automatically detects the project root directory using multiple strategies:
1. Environment variable INVERSE_FOLDING_ROOT (highest priority)
2. Walking up from current file to find directories with project markers
3. Looking for 'inverse-folding' in the current path
5. Checking current working directory for project markers
6. Searching parent directories for project markers
7. Fallback to relative path from current file location

This makes the cache directory robust across different deployment contexts including:
- Local development environments
- CI/CD pipelines
- Distributed training environments
- System-wide installations
"""

import os
import math
import numpy as np
import torch
import json
from pathlib import Path
import fcntl
import time

# Global process-level cache to avoid repeated loading in DataLoader workers
# This persists across multiple RBFLookupManager instances within the same process
_GLOBAL_RBF_CACHE = {
    '3d': {},    # device -> tensor
    'seq': {},   # device -> tensor
    'loaded': {  # track what's been loaded to avoid redundant prints
        '3d': set(),
        'seq': set()
    }
}

class RBFLookupManager:
    """Manages RBF lookup tables for 3D distances and sequence distances."""

    def __init__(self, cache_dir=None, verbose=True, rbf_3d_min=2.0, rbf_3d_max=350.0, rbf_3d_spacing='exponential'):
        """Initialize RBF lookup manager.

        Args:
            cache_dir: Directory to store lookup tables. If None, auto-detects based on project structure.
            verbose: Whether to print loading/creation messages. Set to False for quiet operation during training.
            rbf_3d_min: Minimum distance (Angstroms) for 3D RBF centers (default: 2.0)
            rbf_3d_max: Maximum distance (Angstroms) for 3D RBF centers (default: 350.0)
            rbf_3d_spacing: Spacing type for 3D RBF centers - 'linear' or 'exponential' (default: exponential)
        """
        self.verbose = verbose

        if cache_dir is None:
            cache_dir = self._find_project_cache_dir()

        self.cache_dir = Path(cache_dir)

        # Try to create cache directory, but handle read-only filesystems gracefully
        try:
            self.cache_dir.mkdir(exist_ok=True)
        except (PermissionError, OSError) as e:
            # In read-only containers, try alternative locations
            # tempfile.gettempdir() rather than a hardcoded /tmp, so this also
            # works on Windows.
            import tempfile
            alternative_locations = [
                Path(tempfile.gettempdir()) / "rbf_cache",
                Path.home() / ".cache" / "rbf_cache",
            ]

            cache_created = False
            for alt_cache in alternative_locations:
                try:
                    alt_cache.mkdir(parents=True, exist_ok=True)
                    self.cache_dir = alt_cache
                    cache_created = True
                    print(f"[warn] Original cache dir not writable, using: {alt_cache}")
                    break
                except (PermissionError, OSError):
                    continue

            if not cache_created:
                raise RuntimeError(f"Cannot create cache directory. Tried: {cache_dir}, {alternative_locations}")

        # Initialize RBF parameters
        self._initialize_parameters(rbf_3d_min, rbf_3d_max, rbf_3d_spacing)

        # Add verbose-aware print helper
        self._initial_setup_complete = False

    def _vprint(self, message, force=False):
        """Print message only if verbose mode is enabled or force is True."""
        if self.verbose or force:
            print(message)

    def _initialize_parameters(self, rbf_3d_min, rbf_3d_max, rbf_3d_spacing):
        """Initialize RBF parameters."""
        # 3D RBF parameters (configurable via constructor)
        self.rbf_3d_params = {
            'D_min': rbf_3d_min,
            'D_max': rbf_3d_max,
            'D_count': 16,
            'spacing': rbf_3d_spacing,
            'resolution': 0.001  # 0.001Å resolution
        }

        # Sequence embedding parameters (matching current graph_builder.py)
        self.rbf_seq_params = {
            'num_embeddings': 16,
            'period_range': [2, 1000],
            'max_seq_dist': 2000,  # Maximum absolute sequence distance
            'min_seq_dist': -2000  # Minimum sequence distance (negative)
        }

        # Generate file names
        self.rbf_3d_filename = self._generate_3d_filename()
        self.rbf_seq_filename = self._generate_seq_filename()

        # DEBUG: Print RBF configuration when verbose is True
        if self.verbose:
            print(f"RBF Manager initialized with 3D range: {rbf_3d_min:.1f}-{rbf_3d_max:.1f}Å")
            print(f"RBF 3D filename: {self.rbf_3d_filename}")
            print(f"RBF cache directory: {self.cache_dir}")

        # Note: Using global process-level cache instead of instance-level cache
        # This allows sharing between multiple RBFLookupManager instances in the same process
        # (e.g., across different DataLoader workers)

    def _find_project_cache_dir(self):
        """Auto-detect project root and return rbf_cache directory path."""
        # Method 1: Check environment variable
        if 'INVERSE_FOLDING_ROOT' in os.environ:
            project_root = Path(os.environ['INVERSE_FOLDING_ROOT'])
            if project_root.exists():
                cache_dir = project_root / "rbf_cache"
                self._vprint(f"Using project root from INVERSE_FOLDING_ROOT: {project_root}")
                self._vprint(f"Cache directory: {cache_dir}")
                return str(cache_dir)

        # Method 2: Start from current file location and walk up to find project root
        current_path = Path(__file__).resolve()

        # Look for common project markers (adjust as needed for your project)
        project_markers = [
            'training',      # training directory
            'datasets',      # datasets directory
            'configs'        # config directory
        ]

        # Walk up directory tree to find project root
        for parent in [current_path.parent] + list(current_path.parents):
            # Check if this directory contains project markers
            markers_found = sum(1 for marker in project_markers if (parent / marker).exists())

            # If we find multiple markers, this is likely the project root
            if markers_found >= 2:
                cache_dir = parent / "rbf_cache"
                self._vprint(f"Auto-detected project root: {parent}")
                self._vprint(f"Using cache directory: {cache_dir}")
                return str(cache_dir)

        # Method 3: Check for common deployment patterns
        path_parts = current_path.parts

        # Look for inverse-folding anywhere in the path
        if 'inverse-folding' in path_parts:
            inverse_folding_idx = path_parts.index('inverse-folding')
            project_root = Path(*path_parts[:inverse_folding_idx + 1])
            if project_root.exists():
                cache_dir = project_root / "rbf_cache"
                self._vprint(f"Found inverse-folding in path: {project_root}")
                self._vprint(f"Using cache directory: {cache_dir}")
                return str(cache_dir)

        # Method 5: Check current working directory
        cwd = Path.cwd()
        markers_found = sum(1 for marker in project_markers if (cwd / marker).exists())
        if markers_found >= 2:
            cache_dir = cwd / "rbf_cache"
            self._vprint(f"Found project in current working directory: {cwd}")
            self._vprint(f"Using cache directory: {cache_dir}")
            return str(cache_dir)

        # Method 6: Look for any directory containing multiple project markers
        # Search common parent directories
        search_paths = [
            current_path.parent,
            current_path.parent.parent,
            Path.cwd(),
            Path.cwd().parent,
        ]

        for search_path in search_paths:
            if search_path.exists():
                markers_found = sum(1 for marker in project_markers if (search_path / marker).exists())
                if markers_found >= 2:
                    cache_dir = search_path / "rbf_cache"
                    self._vprint(f"Found project markers in: {search_path}")
                    self._vprint(f"Using cache directory: {cache_dir}")
                    return str(cache_dir)

        # Fallback: use a cache directory relative to this file
        fallback_cache = current_path.parent.parent / "rbf_cache"
        self._vprint(f"Could not auto-detect project root, using fallback: {fallback_cache}")
        if self.verbose:
            print(f"You can set INVERSE_FOLDING_ROOT environment variable to specify the project root")
        return str(fallback_cache)

    def get_detection_info(self):
        """Get information about how the cache directory was detected.

        Returns:
            dict: Information about cache directory detection
        """
        return {
            'cache_dir': str(self.cache_dir),
            'cache_dir_exists': self.cache_dir.exists(),
            'detection_methods': [
                'Environment variable INVERSE_FOLDING_ROOT',
                'Project markers in parent directories',
                'inverse-folding in path',
                'Current working directory',
                'Parent directory search',
                'Fallback relative path'
            ],
            'env_var_set': 'INVERSE_FOLDING_ROOT' in os.environ,
            'env_var_value': os.environ.get('INVERSE_FOLDING_ROOT'),
            'current_file': str(Path(__file__).resolve()),
            'current_cwd': str(Path.cwd())
        }

    def _generate_3d_filename(self):
        """Generate filename for 3D RBF lookup table."""
        params = self.rbf_3d_params
        # Include both the RBF center range and the lookup table extent in filename
        name = (f"rbf_3d_features{params['D_count']}_"
                f"centers{params['D_min']}to{params['D_max']}_"
                f"table{2*params['D_max']}_"
                f"res{params['resolution']}_{params['spacing']}.npy")
        return name

    def _generate_seq_filename(self):
        """Generate filename for sequence RBF lookup table."""
        params = self.rbf_seq_params
        name = (f"rbf_seq_features{params['num_embeddings']}_"
                f"min{params['period_range'][0]}_max{params['period_range'][1]}_"
                f"seqrange{params['min_seq_dist']}to{params['max_seq_dist']}.npy")
        return name

    def _compute_3d_rbf_table(self):
        """Compute 3D distance RBF lookup table."""
        params = self.rbf_3d_params

        # Distance bins - extend lookup table to 2x D_max
        max_dist_centers = params['D_max']  # RBF centers go up to this distance
        max_dist_table = 2 * params['D_max']  # Lookup table extends to 2x D_max
        resolution = params['resolution']
        num_bins = int(max_dist_table / resolution) + 1
        distances = torch.arange(0, num_bins, dtype=torch.float32) * resolution

        # RBF centers (matching graph_builder.py logic) - centers stay within D_min to D_max
        D_min, D_max, D_count = params['D_min'], params['D_max'], params['D_count']
        if params['spacing'] == 'exponential':
            log_min = math.log(D_min + 1e-3)
            log_max = math.log(D_max + 1e-3)
            centers = torch.exp(torch.linspace(log_min, log_max, D_count))
        else:
            centers = torch.linspace(D_min, D_max, D_count)

        width = (D_max - D_min) / D_count

        # Compute RBF values for all distances (now extends to 2x D_max)
        if self.verbose:
            print(f"Computing 3D RBF table: {num_bins:,} distances × {D_count} features...")
            print(f"RBF centers: {D_min:.1f}Å to {D_max:.1f}Å, Lookup table: 0.0Å to {max_dist_table:.1f}Å")
        distances_expanded = distances.unsqueeze(-1)  # [num_bins, 1]
        centers_expanded = centers.unsqueeze(0)       # [1, D_count]

        diff = distances_expanded - centers_expanded  # [num_bins, D_count]
        rbf_table = torch.exp(-(diff ** 2) / (2 * width ** 2))

        return rbf_table.numpy()

    def _compute_seq_rbf_table(self):
        """Compute sequence distance positional embedding lookup table."""
        params = self.rbf_seq_params

        min_seq_dist = params['min_seq_dist']
        max_seq_dist = params['max_seq_dist']
        num_embeddings = params['num_embeddings']

        if self.verbose:
            print(f"Computing sequence RBF table: {max_seq_dist - min_seq_dist + 1} distances × {num_embeddings} features...")

        # All possible sequence distances (including negative)
        seq_distances = torch.arange(min_seq_dist, max_seq_dist + 1, dtype=torch.float32)

        # Frequency computation (matching graph_builder.py)
        frequency = torch.exp(
            torch.arange(0, num_embeddings, 2, dtype=torch.float32)
            * -(np.log(10000.0) / num_embeddings)
        )

        angles = seq_distances.unsqueeze(-1) * frequency.unsqueeze(0)
        embeddings = torch.cat([torch.cos(angles), torch.sin(angles)], -1)

        return embeddings.numpy()

    def _save_table_atomic(self, table, filepath):
        """Save table atomically with file locking."""
        # Create temp files with .tmp extension before final extension
        temp_filepath = filepath.parent / (filepath.stem + '.tmp.npy')

        # Save metadata alongside
        metadata = {
            'rbf_3d_params': self.rbf_3d_params,
            'rbf_seq_params': self.rbf_seq_params,
            'created_at': time.time(),
            'shape': table.shape,
            'dtype': str(table.dtype)
        }

        metadata_filepath = filepath.with_suffix('.json')
        temp_metadata_filepath = filepath.parent / (filepath.stem + '.tmp.json')

        try:
            # Save table and metadata to temp files
            # Use .npy format without extension (np.save adds .npy automatically)
            temp_path_no_ext = str(temp_filepath)[:-4]  # Remove .npy for np.save
            np.save(temp_path_no_ext, table)

            with open(temp_metadata_filepath, 'w') as f:
                json.dump(metadata, f, indent=2)

            # Atomic rename
            temp_filepath.rename(filepath)
            temp_metadata_filepath.rename(metadata_filepath)

            if self.verbose:
                print(f"Saved lookup table: {filepath}")
                print(f"Table shape: {table.shape}, memory: {table.nbytes / 1024 / 1024:.1f} MB")

        except Exception as e:
            # Cleanup on failure
            if temp_filepath.exists():
                temp_filepath.unlink()
            if temp_metadata_filepath.exists():
                temp_metadata_filepath.unlink()
            raise RuntimeError(f"Failed to save lookup table {filepath}: {e}")

    def ensure_3d_rbf_table(self):
        """Ensure 3D RBF lookup table exists, create if missing."""
        filepath = self.cache_dir / self.rbf_3d_filename

        # Double-check pattern: first check without lock for performance
        if filepath.exists():
            self._vprint(f"Found existing 3D RBF table: {filepath}")
            return

        # Use file lock to prevent race condition between multiple workers
        lock_filepath = filepath.with_suffix('.lock')
        try:
            # Create lock file and acquire exclusive lock
            lock_file = open(lock_filepath, 'w', encoding='utf-8')
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

            # Double-check: file might have been created by another process while waiting for lock
            if filepath.exists():
                self._vprint(f"Found existing 3D RBF table (after lock): {filepath}")
                return

            self._vprint(f"Creating 3D RBF lookup table: {filepath}")
            table = self._compute_3d_rbf_table()
            self._save_table_atomic(table, filepath)

        finally:
            # Always release lock and cleanup
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()
                if lock_filepath.exists():
                    lock_filepath.unlink()
            except (OSError, IOError):
                pass  # Ignore cleanup errors

    def ensure_seq_rbf_table(self):
        """Ensure sequence RBF lookup table exists, create if missing."""
        filepath = self.cache_dir / self.rbf_seq_filename

        # Double-check pattern: first check without lock for performance
        if filepath.exists():
            self._vprint(f"Found existing sequence RBF table: {filepath}")
            return

        # Use file lock to prevent race condition between multiple workers
        lock_filepath = filepath.with_suffix('.lock')
        try:
            # Create lock file and acquire exclusive lock
            lock_file = open(lock_filepath, 'w', encoding='utf-8')
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

            # Double-check: file might have been created by another process while waiting for lock
            if filepath.exists():
                self._vprint(f"Found existing sequence RBF table (after lock): {filepath}")
                return

            self._vprint(f"Creating sequence RBF lookup table: {filepath}")
            table = self._compute_seq_rbf_table()
            self._save_table_atomic(table, filepath)

        finally:
            # Always release lock and cleanup
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()
                if lock_filepath.exists():
                    lock_filepath.unlink()
            except (OSError, IOError):
                pass  # Ignore cleanup errors

    def load_3d_rbf_table(self, device='cpu'):
        """Load 3D RBF lookup table and cache in memory using global process-level cache."""
        global _GLOBAL_RBF_CACHE
        target_device = torch.device(device)
        device_str = str(target_device)

        # Check global cache first (shared across all RBFLookupManager instances in this process)
        if device_str in _GLOBAL_RBF_CACHE['3d']:
            return _GLOBAL_RBF_CACHE['3d'][device_str]

        # Ensure table exists before trying to load
        self.ensure_3d_rbf_table()

        filepath = self.cache_dir / self.rbf_3d_filename
        if not filepath.exists():
            raise FileNotFoundError(f"3D RBF lookup table not found: {filepath}. "
                                  f"ensure_3d_rbf_table() failed to create it.")

        try:
            # Only print loading message once per process (not per manager instance)
            if device_str not in _GLOBAL_RBF_CACHE['loaded']['3d']:
                self._vprint(f"Loading 3D RBF table from disk into {device} memory...")
                _GLOBAL_RBF_CACHE['loaded']['3d'].add(device_str)

            table_np = np.load(filepath)
            table_tensor = torch.from_numpy(table_np).to(target_device)

            # Cache in global cache (shared across all instances in this process)
            _GLOBAL_RBF_CACHE['3d'][device_str] = table_tensor

            # Only print cache info once per process
            if self.verbose:
                self._vprint(f"Cached 3D RBF table: {table_np.shape}, {table_np.nbytes / 1024 / 1024:.1f} MB on {target_device}")

            return table_tensor

            # Only print cache info once per session
            if self.verbose:
                self._vprint(f"Cached 3D RBF table: {table_np.shape}, {table_np.nbytes / 1024 / 1024:.1f} MB on {target_device}")

            return table_tensor
        except Exception as e:
            raise RuntimeError(f"Failed to load 3D RBF table {filepath}: {e}")

    def load_seq_rbf_table(self, device='cpu'):
        """Load sequence RBF lookup table and cache in memory using global process-level cache."""
        global _GLOBAL_RBF_CACHE
        target_device = torch.device(device)
        device_str = str(target_device)

        # Check global cache first (shared across all RBFLookupManager instances in this process)
        if device_str in _GLOBAL_RBF_CACHE['seq']:
            return _GLOBAL_RBF_CACHE['seq'][device_str]

        # Ensure table exists before trying to load
        self.ensure_seq_rbf_table()

        filepath = self.cache_dir / self.rbf_seq_filename
        if not filepath.exists():
            raise FileNotFoundError(f"Sequence RBF lookup table not found: {filepath}. "
                                  f"ensure_seq_rbf_table() failed to create it.")

        try:
            # Only print loading message once per process (not per manager instance)
            if device_str not in _GLOBAL_RBF_CACHE['loaded']['seq']:
                self._vprint(f"Loading sequence RBF table from disk into {device} memory...")
                _GLOBAL_RBF_CACHE['loaded']['seq'].add(device_str)

            table_np = np.load(filepath)
            table_tensor = torch.from_numpy(table_np).to(target_device)

            # Cache in global cache (shared across all instances in this process)
            _GLOBAL_RBF_CACHE['seq'][device_str] = table_tensor

            # Only print cache info once per process
            if self.verbose:
                self._vprint(f"Cached sequence RBF table: {table_np.shape}, {table_np.nbytes / 1024 / 1024:.1f} MB on {target_device}")

            return table_tensor

            # Only print cache info once per session
            if self.verbose:
                self._vprint(f"Cached sequence RBF table: {table_np.shape}, {table_np.nbytes / 1024 / 1024:.1f} MB on {target_device}")

            return table_tensor
        except Exception as e:
            raise RuntimeError(f"Failed to load sequence RBF table {filepath}: {e}")

    def lookup_3d_rbf(self, distances, device='cpu'):
        """Fast 3D RBF lookup.

        Args:
            distances: Distance tensor [...]
            device: Device for computation

        Returns:
            RBF features [..., D_count]
        """
        table = self.load_3d_rbf_table(device)
        resolution = self.rbf_3d_params['resolution']
        max_dist_table = 2 * self.rbf_3d_params['D_max']  # Table extends to 2x D_max

        # Quantize distances to lookup indices
        indices = torch.round(distances / resolution).long()
        indices = torch.clamp(indices, 0, len(table) - 1)

        # Handle out-of-range distances (beyond 2x D_max)
        out_of_range = distances > max_dist_table

        # Lookup
        rbf_features = table[indices]

        # Zero out out-of-range features (beyond 2x D_max)
        if out_of_range.any():
            rbf_features = rbf_features * (~out_of_range).unsqueeze(-1).float()

        return rbf_features

    def lookup_seq_rbf(self, seq_distances, device='cpu'):
        """Fast sequence RBF lookup.

        Args:
            seq_distances: Sequence distance tensor [...] (integers)
            device: Device for computation

        Returns:
            Positional embeddings [..., num_embeddings]
        """
        table = self.load_seq_rbf_table(device)
        min_seq_dist = self.rbf_seq_params['min_seq_dist']
        max_seq_dist = self.rbf_seq_params['max_seq_dist']

        # Convert sequence distances to table indices
        # Table starts at min_seq_dist, so index = seq_dist - min_seq_dist
        indices = seq_distances.long() - min_seq_dist

        # Clamp to valid range
        indices = torch.clamp(indices, 0, len(table) - 1)

        # Lookup
        embeddings = table[indices]

        return embeddings

def preload_global_rbf_cache(device='cpu', cache_dir=None, verbose=True):
    """
    Global function to pre-load RBF tables into the process-level cache.

    This is useful to call once per worker process to avoid loading messages
    during the first graph construction in each worker.

    Args:
        device: Device to load tables to ('cpu' or 'cuda:0', etc.)
        cache_dir: Cache directory (auto-detected if None)
        verbose: Whether to print loading messages
    """
    # Create a temporary manager just for loading
    temp_manager = RBFLookupManager(cache_dir=cache_dir, verbose=verbose)
    temp_manager.load_3d_rbf_table(device)
    temp_manager.load_seq_rbf_table(device)

    if verbose:
        print(f"Pre-loaded RBF tables for device {device} into global cache")
