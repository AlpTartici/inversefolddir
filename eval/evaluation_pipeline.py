
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Evaluation Pipeline for Protein Structure Prediction

This module provides a complete pipeline for evaluating protein structure predictions
by coordinating structure prediction and comparison steps.
"""

import argparse
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from structure_comparator import BatchStructureComparator, StructureComparator
from structure_predictor import BatchStructurePredictor

# Validate critical dependencies early (fail-fast)
try:
    import tmtools
except ImportError as e:
    raise ImportError(f"tmtools package is required for structure comparison: {e}")

try:
    import pandas as pd
except ImportError as e:
    raise ImportError(f"pandas package is required for data handling: {e}")

# Distributed imports (with fallback)
try:
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader, Dataset
    from torch.utils.data.distributed import DistributedSampler
    DISTRIBUTED_AVAILABLE = True
except ImportError:
    DISTRIBUTED_AVAILABLE = False
    print("Warning: Distributed processing not available")

# Configure logging (will be reconfigured based on verbosity)
logger = logging.getLogger(__name__)

# ====================================================================
# DISTRIBUTED EVALUATION UTILITIES
# ====================================================================

def setup_distributed_evaluation(device='auto'):
    """
    Setup distributed evaluation environment.
    """
    if not DISTRIBUTED_AVAILABLE:
        return False, 0, 1, 0, torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Check if we're in a distributed environment
    if 'WORLD_SIZE' in os.environ:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])

        # Initialize distributed training
        dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')

        # Set device
        if device == 'auto':
            if torch.cuda.is_available():
                device = torch.device(f'cuda:{local_rank}')
                torch.cuda.set_device(local_rank)
            else:
                device = torch.device('cpu')
        else:
            device = torch.device(device)

        return True, rank, world_size, local_rank, device
    else:
        # Single process mode
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return False, 0, 1, 0, device


def cleanup_distributed_evaluation():
    """
    Cleanup distributed evaluation environment.
    """
    if DISTRIBUTED_AVAILABLE and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if this is the main process."""
    return not dist.is_initialized() or dist.get_rank() == 0


@dataclass
class EvaluationTask:
    """Container for evaluation task information."""
    structure_name: str
    sequence: str
    index: int
    predicted_structure_path: str = None
    reference_structure_path: str = None
    success: bool = False
    error: str = None


class DistributedEvaluationDataset(Dataset):
    """Dataset wrapper for distributed evaluation tasks."""

    def __init__(self, csv_path: str, reference_dir: str, predicted_dir: str):
        self.csv_path = csv_path
        self.reference_dir = Path(reference_dir)
        self.predicted_dir = Path(predicted_dir)

        # Load CSV data
        import pandas as pd
        self.df = pd.read_csv(csv_path)

        # Create evaluation tasks
        self.tasks = []
        for idx, row in self.df.iterrows():
            task = EvaluationTask(
                structure_name=row['structure_name'],
                sequence=row['predicted_sequence'],
                index=idx
            )
            self.tasks.append(task)

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        return self.tasks[idx]


class DistributedStructurePredictor:
    """Distributed version of structure prediction with efficient model loading."""

    def __init__(self, predicted_dir: str, reference_dir: str, device: str = "auto"):
        self.predicted_dir = Path(predicted_dir)
        self.reference_dir = Path(reference_dir)
        self.device = device

        # Create output directory
        self.predicted_dir.mkdir(parents=True, exist_ok=True)

        # Model loading optimization - load once per rank
        self._model_loaded = False
        self._predictor = None
        self._model_lock = threading.Lock()

        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        # Results storage
        self.local_results = []
        self.global_results = []

        # Performance tracking
        self.predictions_made = 0
        self.start_time = None

    def _get_predictor(self):
        """Get ESMFold predictor with singleton pattern to avoid multiple loads."""
        if not self._model_loaded:
            with self._model_lock:
                if not self._model_loaded:  # Double-check locking
                    logger.info(f"Rank {self.rank}: Loading ESMFold model (this may take a few minutes)...")
                    from structure_predictor import ESMFoldPredictor
                    self._predictor = ESMFoldPredictor(device=self.device)
                    self._model_loaded = True
                    logger.info(f"Rank {self.rank}: ESMFold model loaded successfully")
        return self._predictor

    def predict_batch_structures(self, tasks: List[EvaluationTask]) -> List[Dict[str, Any]]:
        """
        Predict structures for multiple tasks efficiently using batching.

        Args:
            tasks: List of evaluation tasks

        Returns:
            List of results
        """
        if not tasks:
            return []

        # Get predictor (loads model only once per rank)
        predictor = self._get_predictor()

        results = []
        sequences_to_predict = []
        task_map = {}

        # Prepare batch of sequences that need prediction
        for task in tasks:
            pred_path = self.predicted_dir / f"{task.structure_name}.pdb"
            ref_path = self.reference_dir / f"{task.structure_name}.pdb"

            # Check if prediction already exists
            if pred_path.exists():
                logger.debug(f"Rank {self.rank}: Skipping {task.structure_name} (already exists)")
                results.append({
                    'task': task,
                    'predicted_path': str(pred_path),
                    'reference_path': str(ref_path) if ref_path.exists() else None,
                    'success': True,
                    'skipped': True,
                    'rank': self.rank
                })
            else:
                sequences_to_predict.append(task.sequence)
                task_map[len(sequences_to_predict) - 1] = task

        # Batch predict structures if any sequences need prediction
        if sequences_to_predict:
            logger.info(f"Rank {self.rank}: Predicting {len(sequences_to_predict)} structures in batch")

            try:
                # Use batch prediction if available, otherwise predict individually
                if hasattr(predictor, 'predict_batch_structures'):
                    # If the predictor supports batch processing
                    pdb_strings = predictor.predict_batch_structures(sequences_to_predict)
                else:
                    # Fallback to individual predictions - use the model directly for PDB strings
                    pdb_strings = []
                    for seq in sequences_to_predict:
                        try:
                            # Use the model's infer_pdb method directly to get PDB string
                            with torch.no_grad():
                                pdb_string = predictor.model.infer_pdb(seq)
                                pdb_strings.append(pdb_string)
                        except Exception as e:
                            logger.error(f"Failed to predict structure for sequence: {e}")
                            # Add a placeholder or skip this sequence
                            pdb_strings.append(None)

                # Save predicted structures
                for i, pdb_string in enumerate(pdb_strings):
                    task = task_map[i]
                    pred_path = self.predicted_dir / f"{task.structure_name}.pdb"
                    ref_path = self.reference_dir / f"{task.structure_name}.pdb"

                    if pdb_string is not None:
                        # Save PDB file
                        with open(pred_path, 'w') as f:
                            f.write(pdb_string)

                        results.append({
                            'task': task,
                            'predicted_path': str(pred_path),
                            'reference_path': str(ref_path) if ref_path.exists() else None,
                            'success': True,
                            'skipped': False,
                            'rank': self.rank
                        })

                        self.predictions_made += 1
                    else:
                        # Handle failed prediction
                        results.append({
                            'task': task,
                            'predicted_path': None,
                            'reference_path': str(ref_path) if ref_path.exists() else None,
                            'success': False,
                            'skipped': False,
                            'error': 'Prediction failed',
                            'rank': self.rank
                        })

            except Exception as e:
                logger.error(f"Rank {self.rank}: Batch prediction failed: {e}")
                # Add error results for all failed predictions
                for i in range(len(sequences_to_predict)):
                    task = task_map[i]
                    results.append({
                        'task': task,
                        'predicted_path': None,
                        'reference_path': None,
                        'success': False,
                        'error': str(e),
                        'rank': self.rank
                    })

        return results

    def run_distributed_prediction(self, dataset: DistributedEvaluationDataset,
                                  batch_size: int = 8, max_workers: int = 1):
        """
        Run distributed structure prediction with efficient batching.

        Args:
            dataset: Evaluation dataset
            batch_size: Number of structures to predict in each batch
            max_workers: Number of worker threads (set to 1 to avoid model conflicts)
        """
        self.start_time = time.time()

        # Create distributed sampler
        sampler = DistributedSampler(dataset, shuffle=False)

        # Custom collate function to handle EvaluationTask objects
        def evaluation_collate_fn(batch):
            """Custom collate function that simply returns the batch as-is for EvaluationTask objects."""
            return batch

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues with model loading
            pin_memory=False,
            collate_fn=evaluation_collate_fn  # Use custom collate function
        )

        logger.info(f"Rank {self.rank}: Processing {len(dataloader)} batches of size {batch_size}")

        # Process batches
        for batch_idx, batch in enumerate(dataloader):
            # Convert batch to list of tasks
            tasks = batch if isinstance(batch, list) else [batch]

            # Process batch efficiently
            batch_results = self.predict_batch_structures(tasks)
            self.local_results.extend(batch_results)

            # Progress update
            if (batch_idx + 1) % 5 == 0:
                elapsed = time.time() - self.start_time
                rate = self.predictions_made / elapsed if elapsed > 0 else 0
                logger.info(f"Rank {self.rank}: Processed {batch_idx + 1}/{len(dataloader)} batches, {self.predictions_made} predictions, {rate:.2f} pred/sec")

        # Synchronize all processes
        if dist.is_initialized():
            dist.barrier()

        # Gather results from all processes
        self.gather_results()

        elapsed = time.time() - self.start_time
        logger.info(f"Rank {self.rank}: Completed {self.predictions_made} predictions in {elapsed:.2f}s")

    def gather_results(self):
        """Gather results from all distributed processes."""
        if not dist.is_initialized():
            self.global_results = self.local_results
            return

        # Serialize local results with proper handling of task objects
        serializable_results = []
        for result in self.local_results:
            serializable_result = result.copy()
            if 'task' in serializable_result and hasattr(serializable_result['task'], 'index'):
                # Convert task object to dictionary to preserve structure
                task = serializable_result['task']
                serializable_result['task'] = {
                    'structure_name': getattr(task, 'structure_name', ''),
                    'sequence': getattr(task, 'sequence', ''),
                    'index': getattr(task, 'index', 0),
                    'predicted_structure_path': getattr(task, 'predicted_structure_path', None),
                    'reference_structure_path': getattr(task, 'reference_structure_path', None),
                    'success': getattr(task, 'success', False),
                    'error': getattr(task, 'error', None)
                }
            serializable_results.append(serializable_result)

        local_results_serialized = json.dumps(serializable_results)

        # Gather all results to rank 0
        if is_main_process():
            all_results = [None] * self.world_size
            dist.gather_object(local_results_serialized, all_results, dst=0)

            # Deserialize and combine
            for rank_results in all_results:
                if rank_results is not None:
                    try:
                        rank_data = json.loads(rank_results)
                        self.global_results.extend(rank_data)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to deserialize results from a rank")

            # Sort by task index - handle both dict and string cases defensively
            def get_task_index(x):
                task = x.get('task', {})
                if isinstance(task, dict):
                    return task.get('index', 0)
                elif hasattr(task, 'index'):
                    return task.index
                else:
                    # Fallback for string representations
                    return 0

            self.global_results.sort(key=get_task_index)
        else:
            dist.gather_object(local_results_serialized, None, dst=0)


class DistributedStructureComparator:
    """Distributed version of structure comparison with efficient batching."""

    def __init__(self, predicted_dir: str, reference_dir: str, csv_path: str = None):
        self.predicted_dir = Path(predicted_dir)
        self.reference_dir = Path(reference_dir)
        self.csv_path = csv_path

        # Load CSV if provided
        if csv_path:
            import pandas as pd
            self.df = pd.read_csv(csv_path)
        else:
            self.df = None

        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        # Results storage
        self.local_results = []
        self.global_results = []

        # Performance tracking
        self.comparisons_made = 0
        self.start_time = None

    def compare_structures_batch(self, tasks: List[Tuple[str, str, Dict]],
                                max_workers: int = 4) -> List[Dict[str, Any]]:
        """
        Compare structures for a batch of tasks using parallel processing.

        Args:
            tasks: List of (pred_path, ref_path, info) tuples
            max_workers: Maximum number of worker threads

        Returns:
            List of comparison results
        """
        if not tasks:
            return []

        results = []

        # Use ThreadPoolExecutor for parallel TM-score calculations
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self._compare_single_structure, pred_path, ref_path, info): (pred_path, ref_path)
                for pred_path, ref_path, info in tasks
            }

            # Collect results as they complete
            for future in as_completed(future_to_task):
                result = future.result()
                results.append(result)

                # Progress update
                self.comparisons_made += 1
                if self.comparisons_made % 20 == 0:
                    elapsed = time.time() - self.start_time if self.start_time else 0
                    rate = self.comparisons_made / elapsed if elapsed > 0 else 0
                    logger.info(f"Rank {self.rank}: {self.comparisons_made} comparisons made, {rate:.2f} comp/sec")

        return results

    def _compare_single_structure(self, pred_path: str, ref_path: str,
                                 structure_info: Dict = None) -> Dict[str, Any]:
        """
        Compare two structures using TM-score.

        Args:
            pred_path: Path to predicted structure
            ref_path: Path to reference structure
            structure_info: Additional structure information

        Returns:
            Comparison result dictionary
        """
        try:
            # Create comparator (lightweight operation)
            comparator = StructureComparator()

            # Perform comparison
            result = comparator.calculate_tm_score(pred_path, ref_path)

            # Handle comparison failure
            if result is None:
                return {
                    'success': False,
                    'error': 'TM-score calculation failed',
                    'rank': self.rank,
                    'predicted_structure_path': pred_path,
                    'reference_structure_path': ref_path
                }

            # Add additional info
            if structure_info:
                result.update(structure_info)

            result['success'] = True
            result['rank'] = self.rank

            return result

        except Exception as e:
            logger.error(f"Rank {self.rank}: Failed to compare structures: {e}")
            return {
                'success': False,
                'error': str(e),
                'rank': self.rank,
                'predicted_structure_path': pred_path,
                'reference_structure_path': ref_path
            }

    def create_comparison_tasks(self) -> List[Tuple[str, str, Dict]]:
        """Create list of comparison tasks."""
        tasks = []

        # Find all predicted structures
        pred_files = list(self.predicted_dir.glob("*.pdb"))

        for pred_file in pred_files:
            structure_name = pred_file.stem
            ref_file = self.reference_dir / f"{structure_name}.pdb"

            if ref_file.exists():
                # Get additional info from CSV if available
                structure_info = {'structure_name': structure_name}
                if self.df is not None:
                    mask = self.df['structure_name'] == structure_name
                    if mask.any():
                        row = self.df[mask].iloc[0]
                        structure_info.update({
                            'predicted_sequence': row.get('predicted_sequence', ''),
                            'sequence_length': len(row.get('predicted_sequence', '')),
                            'sequence_accuracy': row.get('sequence_accuracy', None)
                        })

                tasks.append((str(pred_file), str(ref_file), structure_info))

        return tasks

    def run_distributed_comparison(self, max_workers: int = 4):
        """
        Run distributed structure comparison with efficient batching.

        Args:
            max_workers: Worker threads per GPU for parallel TM-score calculations
        """
        self.start_time = time.time()

        # Create comparison tasks
        all_tasks = self.create_comparison_tasks()

        if not all_tasks:
            logger.warning(f"Rank {self.rank}: No comparison tasks found")
            return

        # Distribute tasks across ranks
        tasks_per_rank = len(all_tasks) // self.world_size
        remainder = len(all_tasks) % self.world_size

        start_idx = self.rank * tasks_per_rank + min(self.rank, remainder)
        end_idx = start_idx + tasks_per_rank + (1 if self.rank < remainder else 0)

        my_tasks = all_tasks[start_idx:end_idx]

        logger.info(f"Rank {self.rank}: Processing {len(my_tasks)} comparison tasks")

        # Process tasks in batches for better efficiency
        batch_size = 32  # Process 32 comparisons at a time
        for i in range(0, len(my_tasks), batch_size):
            batch_tasks = my_tasks[i:i + batch_size]
            batch_results = self.compare_structures_batch(batch_tasks, max_workers)
            self.local_results.extend(batch_results)

            logger.info(f"Rank {self.rank}: Processed {i + len(batch_tasks)}/{len(my_tasks)} tasks")

        # Synchronize all processes
        if dist.is_initialized():
            dist.barrier()

        # Gather results from all processes
        self.gather_results()

        elapsed = time.time() - self.start_time
        logger.info(f"Rank {self.rank}: Completed {self.comparisons_made} comparisons in {elapsed:.2f}s")

    def gather_results(self):
        """Gather results from all distributed processes."""
        if not dist.is_initialized():
            self.global_results = self.local_results
            return

        # Serialize local results without default=str to avoid string conversion issues
        local_results_serialized = json.dumps(self.local_results)

        # Gather all results to rank 0
        if is_main_process():
            all_results = [None] * self.world_size
            dist.gather_object(local_results_serialized, all_results, dst=0)

            # Deserialize and combine
            for rank_results in all_results:
                if rank_results is not None:
                    try:
                        rank_data = json.loads(rank_results)
                        self.global_results.extend(rank_data)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to deserialize results from a rank")

            # Sort by structure name - handle missing keys defensively
            def get_structure_name(x):
                if isinstance(x, dict):
                    return x.get('structure_name', '')
                else:
                    # Fallback for non-dict objects
                    return str(x)

            self.global_results.sort(key=get_structure_name)
        else:
            dist.gather_object(local_results_serialized, None, dst=0)


# ====================================================================
# END DISTRIBUTED EVALUATION UTILITIES
# ====================================================================

def setup_logging(verbose: bool = False):
    """Setup logging configuration based on verbosity level."""
    level = logging.DEBUG if verbose else logging.INFO
    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s' if verbose else '%(asctime)s - %(levelname)s - %(message)s'

    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Configure root logger
    logging.basicConfig(
        level=level,
        format=format_str,
        force=True
    )

    # Set our logger level
    logger.setLevel(level)

    if verbose:
        logger.debug("Verbose logging enabled")


class EvaluationPipeline:
    """
    Complete evaluation pipeline for protein structure predictions.
    """

    def __init__(self, csv_path: str, reference_dir: str, output_base_dir: str = None,
                 verbose: bool = False, protein_subset_path: str = None,
                 protein_timeout_minutes: float = 3.0):
        """
        Initialize the evaluation pipeline.

        Args:
            csv_path: Path to CSV file with sequence predictions
            reference_dir: Directory containing reference ESMFold predictions
            output_base_dir: Base directory for outputs (default: auto-generated)
            verbose: Enable verbose logging for detailed progress tracking
            protein_subset_path: Path to .npy file with specific proteins to evaluate (optional)
            protein_timeout_minutes: Timeout per protein in minutes (default: 3.0)
        """
        self.verbose = verbose
        setup_logging(verbose)

        logger.debug("Initializing EvaluationPipeline...")

        self.csv_path = Path(csv_path)
        self.reference_dir = Path(reference_dir)
        self.protein_subset_path = protein_subset_path
        self.protein_timeout_minutes = protein_timeout_minutes
        self.protein_subset = None

        logger.debug(f"Input validation:")
        logger.debug(f"  CSV path exists: {self.csv_path.exists()}")
        logger.debug(f"  Reference dir exists: {self.reference_dir.exists()}")

        # Create output directory with timestamp
        if output_base_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_name = self.csv_path.stem
            output_base_dir = f"../output/evaluation/{timestamp}_{csv_name}"
            logger.debug(f"Auto-generated output directory: {output_base_dir}")

        self.output_dir = Path(output_base_dir)
        self.predicted_structures_dir = self.output_dir / "predicted_structures"
        self.results_dir = self.output_dir / "results"

        # Create directories
        logger.debug("Creating output directories...")
        self.predicted_structures_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Directories created: {self.output_dir}")

        # Load and validate protein subset if provided
        if self.protein_subset_path:
            self._load_and_validate_protein_subset()

        logger.info(f"Evaluation pipeline initialized:")
        logger.info(f"  Input CSV: {self.csv_path}")
        logger.info(f"  Reference dir: {self.reference_dir}")
        logger.info(f"  Output dir: {self.output_dir}")
        logger.info(f"  Verbose mode: {self.verbose}")
        if self.protein_subset_path:
            logger.info(f"  Protein subset: {len(self.protein_subset)} proteins from {self.protein_subset_path}")
            logger.info(f"  Protein timeout: {self.protein_timeout_minutes} minutes")

    def _load_and_validate_protein_subset(self):
        """Load and validate the protein subset from numpy file."""
        import numpy as np
        import pandas as pd

        try:
            # Load the protein subset
            self.protein_subset = pd.read_csv(self.protein_subset_path)['Protein_ID'].tolist() #np.load(self.protein_subset_path)
            logger.info(f"Loaded {len(self.protein_subset)} proteins from subset file")

            # Validate format (should be strings like '1exk.A')
            for i, protein in enumerate(self.protein_subset[:5]):  # Check first 5
                if not isinstance(protein, (str, np.str_)) or '.' not in protein:
                    raise ValueError(f"Invalid protein format at index {i}: '{protein}'. Expected format: 'pdbid.chain'")

            # Convert to list of strings for easier handling
            self.protein_subset = [str(p) for p in self.protein_subset]

            # Load and check against CSV data
            logger.info("Validating protein subset against CSV data...")
            df = pd.read_csv(self.csv_path)

            if 'structure_name' not in df.columns:
                raise ValueError("CSV file must contain 'structure_name' column")

            available_proteins = set(df['structure_name'].tolist())
            missing_proteins = []

            for protein in self.protein_subset:
                if protein not in available_proteins:
                    missing_proteins.append(protein)

            if missing_proteins:
                logger.error(f"The following proteins from subset are missing in CSV data:")
                for protein in missing_proteins[:10]:  # Show first 10
                    logger.error(f"  - {protein}")
                if len(missing_proteins) > 10:
                    logger.error(f"  ... and {len(missing_proteins) - 10} more")
                print(f"{len(missing_proteins)} proteins from subset not found in CSV data. Missing {missing_proteins}", flush=True)
                # raise ValueError(f"{len(missing_proteins)} proteins from subset not found in CSV data")

            # Check against reference structures
            logger.info("Validating protein subset against reference structures...")
            ref_files = list(self.reference_dir.glob("*.pdb"))
            available_refs = set()
            for ref_file in ref_files:
                # Reference files are named directly as structure_name.pdb
                structure_name = ref_file.stem
                available_refs.add(structure_name)

            missing_refs = []
            for protein in self.protein_subset:
                if protein not in available_refs:
                    missing_refs.append(protein)

            if missing_refs:
                logger.error(f"The following proteins from subset are missing reference structures:")
                for protein in missing_refs[:10]:  # Show first 10
                    logger.error(f"  - {protein}")
                if len(missing_refs) > 10:
                    logger.error(f"  ... and {len(missing_refs) - 10} more")
                raise ValueError(f"{len(missing_refs)} proteins from subset missing reference structures")

            logger.info(f"✓ All {len(self.protein_subset)} proteins validated successfully")

        except Exception as e:
            logger.error(f"Failed to load or validate protein subset: {e}")
            raise

    def run_structure_prediction(self, overwrite: bool = False) -> Dict[str, Any]:
        """
        Run structure prediction step.

        Args:
            overwrite: Whether to overwrite existing predictions

        Returns:
            Dictionary with prediction statistics
        """
        logger.info("="*50)
        logger.info("STEP 1: STRUCTURE PREDICTION")
        logger.info("="*50)

        logger.debug("Setting up structure predictor...")

        try:
            predictor = BatchStructurePredictor(
                csv_path=str(self.csv_path),
                output_dir=str(self.predicted_structures_dir),
                reference_dir=str(self.reference_dir),
                verbose=self.verbose,
                protein_subset=self.protein_subset,
                protein_timeout_minutes=self.protein_timeout_minutes
            )

            logger.debug("Starting batch prediction...")
            if self.verbose:
                logger.debug(f"Overwrite mode: {overwrite}")
                logger.debug(f"Output directory: {self.predicted_structures_dir}")

            stats = predictor.predict_batch(overwrite=overwrite)

            logger.info("Structure prediction completed. Statistics:")
            for key, value in stats.items():
                logger.info(f"  {key}: {value}")

            # Save prediction log
            log_path = self.results_dir / "prediction_log.txt"
            logger.debug(f"Saving prediction log to: {log_path}")

            with open(log_path, 'w') as f:
                f.write("Structure Prediction Log\n")
                f.write("=" * 30 + "\n")
                f.write(f"Timestamp: {datetime.now()}\n")
                f.write(f"Input CSV: {self.csv_path}\n")
                f.write(f"Output directory: {self.predicted_structures_dir}\n")
                f.write(f"Reference directory: {self.reference_dir}\n")
                f.write(f"Verbose mode: {self.verbose}\n")
                f.write(f"Overwrite: {overwrite}\n\n")

                f.write("Statistics:\n")
                for key, value in stats.items():
                    f.write(f"  {key}: {value}\n")

            logger.info(f"Prediction log saved to {log_path}")
            logger.debug("Structure prediction step completed successfully")
            return stats

        except Exception as e:
            logger.error(f"Structure prediction failed: {e}")
            if self.verbose:
                logger.debug("Stack trace:", exc_info=True)
            raise

    def run_structure_comparison(self) -> str:
        """
        Run structure comparison step.

        Returns:
            Path to comparison results CSV
        """
        logger.info("="*50)
        logger.info("STEP 2: STRUCTURE COMPARISON")
        logger.info("="*50)

        logger.debug("Setting up structure comparator...")

        try:
            comparator = BatchStructureComparator(
                predicted_dir=str(self.predicted_structures_dir),
                reference_dir=str(self.reference_dir),
                csv_path=str(self.csv_path),
                verbose=self.verbose,
                protein_subset=self.protein_subset,
                protein_timeout_minutes=self.protein_timeout_minutes
            )

            results_path = self.results_dir / "structure_comparison_results.csv"
            logger.debug(f"Comparison results will be saved to: {results_path}")
            logger.info("Starting structure comparison (this may take several minutes)...")

            if self.verbose:
                logger.debug("Structure comparison running in verbose mode - detailed progress will be shown")

            results_df = comparator.compare_structures(str(results_path))

            # Check if any comparisons were successful
            if len(results_df) == 0:
                logger.error("No successful structure comparisons were performed!")
                logger.error("This could be due to:")
                logger.error("  - No matching structure pairs found")
                logger.error("  - All TM-score calculations failed")
                logger.error("  - Missing reference structures")

                # Create an empty CSV file so the pipeline doesn't crash
                import pandas as pd
                empty_df = pd.DataFrame(columns=[
                    'structure_name', 'sequence_length', 'predicted_sequence', 'true_sequence',
                    'sequence_accuracy', 'pred_structure_length', 'ref_structure_length',
                    'tm_score', 'tm_score_ref', 'rmsd', 'aligned_length', 'seq_id_aligned',
                    'seq_sim_blosum_frac', 'seq_sim_blosum_mean',
                    'plddt_pearson', 'plddt_spearman',
                    'predicted_structure_path', 'reference_structure_path'
                ])
                empty_df.to_csv(results_path, index=False)
                logger.warning(f"Created empty results file: {results_path}")

                return str(results_path)

            logger.info(f"Structure comparison completed!")
            logger.info(f"Processed {len(results_df)} structure pairs")
            logger.info(f"Comparison results saved to {results_path}")
            logger.debug("Structure comparison step completed successfully")

            return str(results_path)

        except Exception as e:
            logger.error(f"Structure comparison failed: {e}")
            if self.verbose:
                logger.debug("Stack trace:", exc_info=True)

            # Check if this is due to missing structures
            pred_files = list(self.predicted_structures_dir.glob("*.pdb"))
            ref_files = list(self.reference_dir.glob("*.pdb"))

            logger.error(f"Debugging information:")
            logger.error(f"  Predicted structures found: {len(pred_files)}")
            logger.error(f"  Reference structures found: {len(ref_files)}")

            if len(pred_files) == 0:
                logger.error("  No predicted structures found! Check structure prediction step.")
            if len(ref_files) == 0:
                logger.error(f"  No reference structures found in {self.reference_dir}")
                logger.error("  Check the reference directory path.")

            raise

    def generate_summary_report(self, comparison_results_path: str):
        """
        Generate a comprehensive summary report.

        Args:
            comparison_results_path: Path to comparison results CSV
        """
        logger.info("="*50)
        logger.info("STEP 3: SUMMARY REPORT")
        logger.info("="*50)

        logger.debug("Loading comparison results for summary generation...")

        try:
            import pandas as pd

            # Load comparison results
            logger.debug(f"Reading comparison results from: {comparison_results_path}")

            # Check if file exists
            if not Path(comparison_results_path).exists():
                logger.error(f"Comparison results file not found: {comparison_results_path}")
                raise FileNotFoundError(f"Comparison results file not found: {comparison_results_path}")

            df = pd.read_csv(comparison_results_path)
            logger.debug(f"Loaded {len(df)} comparison results")

            # Check if DataFrame is empty
            if len(df) == 0:
                logger.warning("No comparison results found - generating empty summary report")

                # Generate empty summary report
                report_path = self.results_dir / "evaluation_summary.txt"
                with open(report_path, 'w') as f:
                    f.write("PROTEIN STRUCTURE EVALUATION SUMMARY\n")
                    f.write("=" * 50 + "\n")
                    f.write(f"Generated: {datetime.now()}\n")
                    f.write(f"Input CSV: {self.csv_path}\n")
                    f.write(f"Total structures evaluated: 0\n")
                    f.write(f"Verbose mode: {self.verbose}\n\n")

                    f.write("STATUS: NO SUCCESSFUL COMPARISONS\n")
                    f.write("-" * 40 + "\n")
                    f.write("No structure comparisons were successful.\n")
                    f.write("This could be due to:\n")
                    f.write("  - Missing reference structures\n")
                    f.write("  - Failed structure predictions\n")
                    f.write("  - TM-score calculation errors\n")
                    f.write("  - File path issues\n\n")
                    f.write("Please check the prediction log and error messages above.\n")

                logger.info(f"Empty summary report saved to {report_path}")

                # Print console message
                print("\n" + "="*60)
                print("EVALUATION COMPLETED WITH ISSUES")
                print("="*60)
                print("No successful structure comparisons were performed.")
                print("Please check the logs for more details.")
                print(f"Results directory: {self.output_dir}")
                print("="*60)

                return

            # Continue with normal summary generation if we have data
            # Generate summary report
            report_path = self.results_dir / "evaluation_summary.txt"
            logger.debug(f"Generating summary report: {report_path}")

            if self.verbose:
                logger.debug("Calculating statistics...")
                logger.debug(f"  TM-score range: {df['tm_score'].min():.4f} - {df['tm_score'].max():.4f}")
                logger.debug(f"  RMSD range: {df['rmsd'].min():.2f} - {df['rmsd'].max():.2f} Å")

            with open(report_path, 'w') as f:
                f.write("PROTEIN STRUCTURE EVALUATION SUMMARY\n")
                f.write("=" * 50 + "\n")
                f.write(f"Generated: {datetime.now()}\n")
                f.write(f"Input CSV: {self.csv_path}\n")
                f.write(f"Total structures evaluated: {len(df)}\n")
                f.write(f"Verbose mode: {self.verbose}\n\n")

                # Overall statistics
                f.write("OVERALL PERFORMANCE\n")
                f.write("-" * 20 + "\n")
                f.write(f"Average TM-score: {df['tm_score'].mean():.4f} ± {df['tm_score'].std():.4f}\n")
                f.write(f"Average RMSD: {df['rmsd'].mean():.2f} ± {df['rmsd'].std():.2f} Å\n")
                f.write(f"Average sequence accuracy: {df['sequence_accuracy'].mean():.2f}%\n\n")

                # TM-score distribution
                f.write("TM-SCORE DISTRIBUTION\n")
                f.write("-" * 20 + "\n")
                thresholds = [0.5, 0.4, 0.3, 0.2]
                for threshold in thresholds:
                    count = (df['tm_score'] > threshold).sum()
                    percentage = (df['tm_score'] > threshold).mean() * 100
                    f.write(f"TM-score > {threshold}: {count} structures ({percentage:.1f}%)\n")
                    if self.verbose:
                        logger.debug(f"TM-score > {threshold}: {count}/{len(df)} ({percentage:.1f}%)")
                f.write("\n")

                # Correlation analysis
                corr_tm_seq = df['tm_score'].corr(df['sequence_accuracy'])
                f.write("CORRELATION ANALYSIS\n")
                f.write("-" * 20 + "\n")
                f.write(f"TM-score vs Sequence Accuracy: {corr_tm_seq:.3f}\n\n")
                if self.verbose:
                    logger.debug(f"Correlation (TM-score vs Seq Accuracy): {corr_tm_seq:.3f}")

                # Top and worst performers
                f.write("TOP 10 STRUCTURES (by TM-score)\n")
                f.write("-" * 35 + "\n")
                top10 = df.nlargest(10, 'tm_score')
                for _, row in top10.iterrows():
                    f.write(f"{row['structure_name']:12} TM={row['tm_score']:.4f} RMSD={row['rmsd']:.2f}Å SeqAcc={row['sequence_accuracy']:.1f}%\n")

                f.write("\nWORST 10 STRUCTURES (by TM-score)\n")
                f.write("-" * 37 + "\n")
                worst10 = df.nsmallest(10, 'tm_score')
                for _, row in worst10.iterrows():
                    f.write(f"{row['structure_name']:12} TM={row['tm_score']:.4f} RMSD={row['rmsd']:.2f}Å SeqAcc={row['sequence_accuracy']:.1f}%\n")

                # Length-based analysis
                f.write("\nPERFORMANCE BY SEQUENCE LENGTH\n")
                f.write("-" * 30 + "\n")
                length_bins = [(0, 50), (50, 100), (100, 150), (150, 200), (200, float('inf'))]
                for min_len, max_len in length_bins:
                    if max_len == float('inf'):
                        mask = df['sequence_length'] >= min_len
                        label = f"{min_len}+"
                    else:
                        mask = (df['sequence_length'] >= min_len) & (df['sequence_length'] < max_len)
                        label = f"{min_len}-{max_len}"

                    if mask.sum() > 0:
                        subset = df[mask]
                        f.write(f"Length {label:8}: {len(subset):3d} structures, avg TM={subset['tm_score'].mean():.4f}\n")
                        if self.verbose:
                            logger.debug(f"Length bin {label}: {len(subset)} structures, avg TM={subset['tm_score'].mean():.4f}")

            logger.info(f"Summary report saved to {report_path}")
            logger.debug("Summary generation completed")

            # Generate detailed CSV with individual protein results
            csv_output_path = self.results_dir / "detailed_results.csv"
            logger.info(f"Generating detailed CSV with individual protein results...")

            # Select and rename columns for the output CSV
            output_columns = {
                'structure_name': 'Protein_ID',
                'tm_score': 'TM_Score',
                'rmsd': 'RMSD',
                'sequence_accuracy': 'Sequence_Accuracy_Percent',
                'sequence_length': 'Sequence_Length',
                'seq_sim_blosum_frac': 'Seq_Sim_BLOSUM62_Frac',
                'seq_sim_blosum_mean': 'Seq_Sim_BLOSUM62_Mean',
                'plddt_pearson': 'pLDDT_Pearson',
                'plddt_spearman': 'pLDDT_Spearman',
            }

            # Create output dataframe with selected columns (only those present in df)
            available_columns = [col for col in output_columns.keys() if col in df.columns]
            output_df = df[available_columns].copy()
            output_df = output_df.rename(columns={col: output_columns[col] for col in available_columns})

            # Round numerical values for cleaner output (convert to numeric first)
            output_df['TM_Score'] = pd.to_numeric(output_df['TM_Score'], errors='coerce').round(4)
            output_df['RMSD'] = pd.to_numeric(output_df['RMSD'], errors='coerce').round(2)
            if 'Sequence_Accuracy_Percent' in output_df.columns:
                output_df['Sequence_Accuracy_Percent'] = pd.to_numeric(output_df['Sequence_Accuracy_Percent'], errors='coerce').round(2)

            # Sort by TM-score descending
            output_df = output_df.sort_values('TM_Score', ascending=False)

            # Save to CSV
            output_df.to_csv(csv_output_path, index=False)
            logger.info(f"Detailed results CSV saved to {csv_output_path}")

            # Also print key metrics to console
            print("\n" + "="*60)
            print("EVALUATION COMPLETED")
            print("="*60)
            print(f"Structures evaluated: {len(df)}")
            print(f"Average TM-score: {df['tm_score'].mean():.4f}")
            print(f"STD TM-score: {df['tm_score'].std():.4f}")
            print(f"Median TM-score: {df['tm_score'].median():.4f}")
            print(f"Minimum TM-score: {df['tm_score'].min():.4f}")
            print(f"Average RMSD: {df['rmsd'].mean():.2f} Å")
            print(f"Structures with TM-score > 0.5: {(df['tm_score'] > 0.5).sum()} ({(df['tm_score'] > 0.5).mean()*100:.1f}%)")
            print(f"Results directory: {self.output_dir}")
            print(f"Detailed results CSV: {csv_output_path}")
            print("="*60)

        except Exception as e:
            logger.error(f"Failed to generate summary report: {e}")
            if self.verbose:
                logger.debug("Stack trace:", exc_info=True)
            raise

    def run_complete_evaluation(self, overwrite_predictions: bool = False):
        """
        Run the complete evaluation pipeline: prediction + comparison + summary.

        Args:
            overwrite_predictions: Whether to overwrite existing predicted structures
        """
        logger.info("Starting complete evaluation pipeline")
        logger.debug(f"Overwrite predictions: {overwrite_predictions}")

        try:
            # Step 1: Structure prediction
            logger.info("Step 1/3: Running structure prediction...")
            prediction_results = self.run_structure_prediction(overwrite=overwrite_predictions)

            # Step 2: Structure comparison
            logger.info("Step 2/3: Running structure comparison...")
            comparison_results_path = self.run_structure_comparison()

            # Step 3: Generate summary report
            logger.info("Step 3/3: Generating summary report...")
            self.generate_summary_report(comparison_results_path)

            logger.info("Complete evaluation pipeline finished successfully")

        except Exception as e:
            logger.error(f"Complete evaluation pipeline failed: {e}")
            if self.verbose:
                logger.debug("Stack trace:", exc_info=True)
            raise


def run_distributed_evaluation(args):
    """
    Run distributed evaluation pipeline across multiple GPUs.

    Args:
        args: Command line arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        # Setup distributed environment
        is_distributed, rank, world_size, local_rank, device = setup_distributed_evaluation(args.device)

        # Create output directory
        output_dir = args.output_dir
        if not output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"evaluation_output_{timestamp}"

        # Only rank 0 creates directory
        if is_main_process():
            os.makedirs(output_dir, exist_ok=True)

        # Wait for directory creation
        if is_distributed:
            dist.barrier()

        predicted_dir = os.path.join(output_dir, "predicted_structures")
        results_file = Path(output_dir) / "comparison_results.csv"

        # Create distributed dataset
        dataset = DistributedEvaluationDataset(
            csv_path=args.csv_path,
            reference_dir=args.reference_dir,
            predicted_dir=predicted_dir
        )

        # Initialize distributed predictor
        predictor = DistributedStructurePredictor(
            predicted_dir=predicted_dir,
            reference_dir=args.reference_dir,
            device=device
        )

        # Run distributed structure prediction
        if not args.compare_only:
            logger.info(f"Rank {rank}: Starting distributed structure prediction...")
            predictor.run_distributed_prediction(
                dataset=dataset,
                batch_size=args.batch_size,
                max_workers=1  # Keep as 1 to avoid model loading conflicts
            )

        # Run distributed structure comparison
        if not args.predict_only:
            logger.info(f"Rank {rank}: Starting distributed structure comparison...")
            comparator = DistributedStructureComparator(
                predicted_dir=predicted_dir,
                reference_dir=args.reference_dir,
                csv_path=args.csv_path
            )

            comparator.run_distributed_comparison(max_workers=args.max_workers)

            # Save results (only rank 0)
            if is_main_process():
                logger.info("Saving comparison results...")

                # Convert results to DataFrame
                if comparator.global_results:
                    # Use try/except to handle pandas import in case it's not available
                    try:
                        import pandas as pd
                        df = pd.DataFrame(comparator.global_results)
                        df.to_csv(results_file, index=False)
                        logger.info(f"Results saved to {results_file}")

                        # Generate detailed CSV with individual protein results
                        detailed_csv_path = results_file.parent / "detailed_results.csv"
                        logger.info(f"Generating detailed CSV with individual protein results...")

                        # Select and rename columns for the output CSV
                        output_columns = {
                            'structure_name': 'Protein_ID',
                            'tm_score': 'TM_Score',
                            'rmsd': 'RMSD',
                            'sequence_accuracy': 'Sequence_Accuracy_Percent',
                            'sequence_length': 'Sequence_Length',
                            'seq_sim_blosum_frac': 'Seq_Sim_BLOSUM62_Frac',
                            'seq_sim_blosum_mean': 'Seq_Sim_BLOSUM62_Mean',
                            'plddt_pearson': 'pLDDT_Pearson',
                            'plddt_spearman': 'pLDDT_Spearman',
                        }

                        # Create output dataframe with selected columns
                        available_columns = [col for col in output_columns.keys() if col in df.columns]
                        output_df = df[available_columns].copy()
                        output_df = output_df.rename(columns={col: output_columns[col] for col in available_columns})

                        # Round numerical values for cleaner output
                        if 'TM_Score' in output_df.columns:
                            output_df['TM_Score'] = pd.to_numeric(output_df['TM_Score'], errors='coerce').round(4)
                        if 'RMSD' in output_df.columns:
                            output_df['RMSD'] = pd.to_numeric(output_df['RMSD'], errors='coerce').round(2)
                        if 'Sequence_Accuracy_Percent' in output_df.columns:
                            output_df['Sequence_Accuracy_Percent'] = pd.to_numeric(output_df['Sequence_Accuracy_Percent'], errors='coerce').round(2)

                        # Sort by TM-score descending
                        if 'TM_Score' in output_df.columns:
                            output_df = output_df.sort_values('TM_Score', ascending=False)

                        # Save to CSV
                        output_df.to_csv(detailed_csv_path, index=False)
                        logger.info(f"Detailed results CSV saved to {detailed_csv_path}")

                        # Print summary
                        print("\n" + "="*60)
                        print("DISTRIBUTED EVALUATION COMPLETED")
                        print("="*60)
                        print(f"Total structures: {len(df)}")

                        # Check if we have successful comparisons
                        if 'tm_score' in df.columns and len(df) > 0:
                            print(f"Average TM-score: {df['tm_score'].mean():.4f}")
                            if 'rmsd' in df.columns:
                                print(f"Average RMSD: {df['rmsd'].mean():.2f} Å")
                            print(f"High-quality structures (TM > 0.5): {(df['tm_score'] > 0.5).sum()}")
                        else:
                            print("No successful structure comparisons were completed.")

                        print(f"Results saved to: {results_file}")
                        print(f"Detailed results CSV: {detailed_csv_path}")
                        print("="*60)

                    except ImportError:
                        # Fallback to JSON if pandas not available
                        import json
                        with open(results_file.replace('.csv', '.json'), 'w') as f:
                            json.dump(comparator.global_results, f, indent=2)
                        logger.info(f"Results saved to {results_file.replace('.csv', '.json')} (pandas not available)")
                else:
                    logger.warning("No comparison results to save")

        # Cleanup
        cleanup_distributed_evaluation()

        return 0

    except Exception as e:
        logger.error(f"Distributed evaluation failed: {e}")
        if args.verbose:
            logger.debug("Full error details:", exc_info=True)

        # Cleanup on failure
        try:
            cleanup_distributed_evaluation()
        except:
            pass

        return 1


def main():
    """Command-line interface for the evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description="Complete evaluation pipeline for protein structure predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete evaluation
  python evaluation_pipeline.py --csv_path predictions.csv --reference_dir /path/to/reference/structures

  # Run with verbose output for detailed progress tracking
  python evaluation_pipeline.py \
    --csv_path ../output/prediction/20250709_233849_batch_validation_sampling_virtual_sequences.csv \
    --reference_dir ../datasets/esmfold_predictions/esmfold_predictions_on_ref_valid \
    --verbose

  # Specify custom output directory
  python evaluation_pipeline.py --csv_path predictions.csv --reference_dir /path/to/reference/structures --output_dir /path/to/output

  # Overwrite existing predictions
  python evaluation_pipeline.py --csv_path predictions.csv --reference_dir /path/to/reference/structures --overwrite
        """
    )

    parser.add_argument("--csv_path", required=True, help="Path to CSV file with sequence predictions")
    parser.add_argument("--reference_dir", required=True, help="Directory containing reference ESMFold predictions")
    parser.add_argument("--output_dir", help="Output directory (default: auto-generated)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output for detailed progress tracking")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing structure predictions")
    parser.add_argument("--predict-only", action="store_true",
                       help="Only run structure prediction step")
    parser.add_argument("--compare-only", action="store_true",
                       help="Only run comparison step (requires existing predictions)")
    parser.add_argument("--protein-subset",
                       help="Path to .npy file containing specific proteins to evaluate (speeds up evaluation)")
    parser.add_argument("--protein-timeout", type=float, default=3.0,
                       help="Timeout per protein in minutes (default: 3.0)")

    # Distributed evaluation arguments
    parser.add_argument("--distributed", action="store_true",
                       help="Run distributed evaluation across multiple GPUs")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size for distributed prediction (default: 8)")
    parser.add_argument("--max-workers", type=int, default=4,
                       help="Maximum worker threads for parallel processing (default: 4)")
    parser.add_argument("--device", default="auto",
                       help="Device to use (auto, cuda, cpu) (default: auto)")

    args = parser.parse_args()

    # Setup logging based on verbosity before creating pipeline
    setup_logging(args.verbose)

    try:
        if args.verbose:
            logger.info("Verbose mode enabled - detailed progress will be shown")

        # Check if distributed evaluation is requested
        if args.distributed:
            # Run distributed evaluation
            return run_distributed_evaluation(args)
        else:
            # Run standard evaluation
            pipeline = EvaluationPipeline(
                csv_path=args.csv_path,
                reference_dir=args.reference_dir,
                output_base_dir=args.output_dir,
                verbose=args.verbose,
                protein_subset_path=args.protein_subset,
                protein_timeout_minutes=args.protein_timeout
            )

            if args.predict_only:
                logger.info("Running prediction-only mode")
                pipeline.run_structure_prediction(overwrite=args.overwrite)
            elif args.compare_only:
                logger.info("Running comparison-only mode")
                comparison_results = pipeline.run_structure_comparison()
                pipeline.generate_summary_report(comparison_results)
            else:
                logger.info("Running complete evaluation pipeline")
                pipeline.run_complete_evaluation(overwrite_predictions=args.overwrite)

        return 0

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        if args.verbose:
            logger.debug("Full error details:", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
