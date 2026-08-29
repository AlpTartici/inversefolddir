# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
ESMFold Structure Prediction Module

This module handles protein structure prediction using ESMFold for sequences
from prediction CSV files.
"""

import logging
import os
import signal
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from transformers import AutoTokenizer, EsmForProteinFolding

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ESMFoldPredictor:
    """
    A class to handle ESMFold structure predictions for protein sequences.
    """

    def __init__(self, device: str = "auto"):
        """
        Initialize the ESMFold predictor.

        Args:
            device: Device to use ("cuda", "cpu", or "auto")
        """
        self.device = self._get_device(device)
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _get_device(self, device: str) -> str:
        """Determine the appropriate device to use."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _load_model(self):
        """Load ESMFold model and tokenizer."""
        try:
            logger.info("Loading ESMFold model...")
            self.model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")

            if self.device == "cuda":
                self.model = self.model.cuda()

            self.model.eval()

            logger.info(f"ESMFold model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load ESMFold model: {e}")
            raise

    def predict_structure(self, sequence: str, output_path: str) -> bool:
        """
        Predict structure for a given sequence.

        Args:
            sequence: Protein sequence
            output_path: Path to save the predicted structure

        Returns:
            bool: True if prediction successful, False otherwise
        """
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with torch.no_grad():
                pdb_string = self.model.infer_pdb(sequence)

            with open(output_path, "w") as f:
                f.write(pdb_string)

            logger.debug(f"Structure predicted and saved to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to predict structure for sequence: {e}")
            return False


class BatchStructurePredictor:
    """
    Handles batch structure prediction from CSV files containing sequence predictions.
    """

    def __init__(
        self,
        csv_path: str,
        output_dir: str,
        reference_dir: str,
        verbose: bool = False,
        protein_subset: List[str] = None,
        protein_timeout_minutes: float = 3.0,
    ):
        """
        Initialize batch predictor.

        Args:
            csv_path: Path to CSV file with predictions
            output_dir: Directory to save predicted structures
            reference_dir: Directory containing reference ESMFold predictions
            verbose: Enable verbose logging
            protein_subset: List of specific proteins to predict (optional)
            protein_timeout_minutes: Timeout per protein in minutes
        """
        self.csv_path = Path(csv_path)
        self.output_dir = Path(output_dir)
        self.reference_dir = Path(reference_dir)
        self.predictor = ESMFoldPredictor()
        self.verbose = verbose
        self.protein_subset = protein_subset
        self.protein_timeout_minutes = protein_timeout_minutes

        # Validate inputs
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized batch predictor:")
        logger.info(f"  CSV: {self.csv_path}")
        logger.info(f"  Output: {self.output_dir}")
        logger.info(f"  Reference: {self.reference_dir}")

        if verbose:
            logger.debug(f"Verbose mode enabled for structure prediction")
            logger.debug(f"  Predictor device: {self.predictor.device}")
            logger.debug(f"  Output directory exists: {self.output_dir.exists()}")
            logger.debug(f"  Reference directory exists: {self.reference_dir.exists()}")

            # Count reference files
            if self.reference_dir.exists():
                ref_files = list(self.reference_dir.glob("*.pdb"))
                logger.debug(f"  Reference files found: {len(ref_files)}")
                if len(ref_files) > 0:
                    logger.debug(
                        f"  Sample reference files: {[f.name for f in ref_files[:3]]}"
                    )
            else:
                logger.warning(
                    f"Reference directory does not exist: {self.reference_dir}"
                )
                logger.warning("All structures will be marked as missing_reference")

    def load_predictions(self) -> pd.DataFrame:
        """Load and validate the predictions CSV."""
        try:
            df = pd.read_csv(self.csv_path)

            # Validate required columns
            required_cols = ["structure_name", "predicted_sequence"]
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Convert predicted_sequence from indices to amino acids if needed
            def convert_sequence_format(row):
                pred_seq = row["predicted_sequence"]

                # Check if the sequence is in index format (string representation of list)
                if (
                    isinstance(pred_seq, str)
                    and pred_seq.startswith("[")
                    and pred_seq.endswith("]")
                ):
                    try:
                        # Parse the string representation of indices
                        import ast

                        indices = ast.literal_eval(pred_seq)

                        # Convert indices to amino acids
                        idx_to_aa = [
                            "ALA",
                            "CYS",
                            "ASP",
                            "GLU",
                            "PHE",
                            "GLY",
                            "HIS",
                            "ILE",
                            "LYS",
                            "LEU",
                            "MET",
                            "ASN",
                            "PRO",
                            "GLN",
                            "ARG",
                            "SER",
                            "THR",
                            "VAL",
                            "TRP",
                            "TYR",
                            "XXX",
                        ]

                        # Three-letter to one-letter conversion
                        THREE_TO_ONE = {
                            "ALA": "A",
                            "CYS": "C",
                            "ASP": "D",
                            "GLU": "E",
                            "PHE": "F",
                            "GLY": "G",
                            "HIS": "H",
                            "ILE": "I",
                            "LYS": "K",
                            "LEU": "L",
                            "MET": "M",
                            "ASN": "N",
                            "PRO": "P",
                            "GLN": "Q",
                            "ARG": "R",
                            "SER": "S",
                            "THR": "T",
                            "VAL": "V",
                            "TRP": "W",
                            "TYR": "Y",
                            "XXX": "X",
                        }

                        # Convert indices to amino acid sequence
                        aa_sequence = ""
                        for idx in indices:
                            if 0 <= idx < len(idx_to_aa):
                                aa_sequence += THREE_TO_ONE[idx_to_aa[idx]]
                            else:
                                aa_sequence += "X"  # Unknown

                        logger.info(
                            f"Converted sequence for {row['structure_name']}: {len(indices)} indices -> {len(aa_sequence)} amino acids"
                        )
                        return aa_sequence

                    except (ValueError, SyntaxError) as e:
                        logger.warning(
                            f"Failed to parse indices for {row['structure_name']}: {e}"
                        )
                        return pred_seq

                # Already in amino acid format
                return pred_seq

            # Convert sequences from indices to amino acids if needed
            df["predicted_sequence"] = df.apply(convert_sequence_format, axis=1)

            # Validate sequence lengths (but don't require true_sequence)
            def check_length_consistency(row):
                pred_seq = row["predicted_sequence"]
                pred_len = len(pred_seq)

                # Check if length column exists and is consistent
                if "length" in df.columns:
                    stated_len = row["length"]
                    if pred_len != stated_len:
                        logger.warning(
                            f"Length mismatch for {row['structure_name']}: "
                            f"predicted={pred_len}, stated={stated_len} (using predicted length)"
                        )

                # Check for true_sequence if available (optional)
                if "true_sequence" in df.columns:
                    true_seq = row["true_sequence"]
                    if pd.isna(true_seq) or true_seq is None or true_seq == "":
                        logger.debug(
                            f"No true_sequence available for {row['structure_name']} (evaluation will proceed)"
                        )
                    else:
                        true_len = len(true_seq)
                        if pred_len != true_len:
                            logger.warning(
                                f"Sequence length mismatch for {row['structure_name']}: "
                                f"predicted={pred_len}, true={true_len}"
                            )

                # Always return True - we can evaluate without true sequence
                return True

            df["length_consistent"] = df.apply(check_length_consistency, axis=1)

            # Update length column to match actual predicted sequence length
            df["length"] = df["predicted_sequence"].apply(len)

            logger.info(f"Loaded {len(df)} predictions from CSV")
            logger.info(
                f"Sequence length range: {df['length'].min()}-{df['length'].max()} amino acids"
            )

            return df

        except Exception as e:
            logger.error(f"Failed to load predictions CSV: {e}")
            raise

    def predict_batch(self, overwrite: bool = False) -> Dict[str, Any]:
        """
        Predict structures for all sequences in the CSV.

        Args:
            overwrite: Whether to overwrite existing predictions

        Returns:
            Dict with prediction statistics
        """
        df = self.load_predictions()

        # Filter by protein subset if provided
        if self.protein_subset:
            initial_count = len(df)
            df = df[df["structure_name"].isin(self.protein_subset)]
            logger.info(
                f"Filtered dataset: {len(df)} proteins from subset (was {initial_count} total)"
            )

            if len(df) == 0:
                raise ValueError("No proteins from subset found in CSV data")

        stats = {
            "total": len(df),
            "successful": 0,
            "failed": 0,
            "skipped": 0,
            "missing_reference": 0,
            "timeout": 0,
        }

        logger.info(f"Starting batch prediction for {stats['total']} structures...")

        if self.verbose:
            logger.debug(f"Prediction settings:")
            logger.debug(f"  Overwrite existing: {overwrite}")
            logger.debug(f"  Total sequences to process: {stats['total']}")
            if self.protein_subset:
                logger.debug(
                    f"  Using protein subset: {len(self.protein_subset)} proteins"
                )
                logger.debug(
                    f"  Timeout per protein: {self.protein_timeout_minutes} minutes"
                )

        for idx, row in df.iterrows():
            structure_name = row["structure_name"]
            predicted_sequence = row["predicted_sequence"]

            if self.verbose:
                seq_length = len(predicted_sequence)
                logger.info(
                    f"Processing {structure_name} ({idx+1}/{stats['total']}) - sequence length: {seq_length}"
                )
            else:
                logger.info(f"Processing {structure_name} ({idx+1}/{stats['total']})")

            # Skip if lengths are inconsistent
            if not row["length_consistent"]:
                logger.warning(f"Skipping {structure_name} due to length inconsistency")
                stats["skipped"] += 1
                continue

            # Define output path — use a sequence hash so that rows sharing the same
            # structure_name (e.g. expval sweeps with one reference protein) each get
            # a unique file.  The hash is deterministic, so re-runs skip already-folded
            # sequences correctly.
            import hashlib as _hl
            seq_hash = _hl.md5(predicted_sequence.encode()).hexdigest()[:12]
            pred_output_path = self.output_dir / f"pred_for_{seq_hash}.pdb"

            # Check if prediction already exists
            if pred_output_path.exists() and not overwrite:
                if self.verbose:
                    logger.debug(f"Prediction exists for {structure_name}, skipping")
                else:
                    logger.info(f"Prediction exists for {structure_name}, skipping")
                stats["skipped"] += 1
                continue

            # Check if reference exists
            ref_path = self.reference_dir / f"{structure_name}.pdb"
            if not ref_path.exists():
                logger.warning(f"Reference structure not found for {structure_name}")
                stats["missing_reference"] += 1
                if self.verbose:
                    logger.debug(f"  Expected reference at: {ref_path}")

            # Predict structure with timeout
            try:
                success = self._predict_with_timeout(
                    predicted_sequence, str(pred_output_path), structure_name
                )

                if success:
                    stats["successful"] += 1
                    logger.info(f"✓ Predicted structure for {structure_name}")
                    if self.verbose:
                        file_size = pred_output_path.stat().st_size
                        logger.debug(f"  Output file size: {file_size} bytes")
                else:
                    stats["failed"] += 1
                    logger.error(f"✗ Failed to predict structure for {structure_name}")

            except TimeoutError:
                stats["timeout"] += 1
                logger.warning(
                    f"⏰ Timeout ({self.protein_timeout_minutes} min) for {structure_name}"
                )

        logger.info("Batch prediction completed:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")

        if self.verbose:
            success_rate = (
                (stats["successful"] / stats["total"]) * 100
                if stats["total"] > 0
                else 0
            )
            logger.debug(f"Success rate: {success_rate:.1f}%")

        return stats

    def _predict_with_timeout(
        self, sequence: str, output_path: str, structure_name: str
    ) -> bool:
        """
        Predict structure with timeout handling.

        Args:
            sequence: Protein sequence
            output_path: Path to save prediction
            structure_name: Name of the structure (for logging)

        Returns:
            bool: True if successful, False if failed

        Raises:
            TimeoutError: If prediction exceeds timeout
        """

        def timeout_handler(signum, frame):
            raise TimeoutError(f"Prediction timeout for {structure_name}")

        # Set up timeout
        timeout_seconds = int(self.protein_timeout_minutes * 60)
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)

        try:
            if self.verbose:
                logger.debug(
                    f"Starting ESMFold prediction for {structure_name} with {timeout_seconds}s timeout..."
                )

            success = self.predictor.predict_structure(sequence, output_path)
            signal.alarm(0)  # Cancel timeout
            return success

        except TimeoutError:
            signal.alarm(0)  # Cancel timeout
            logger.warning(
                f"Prediction timed out for {structure_name} after {self.protein_timeout_minutes} minutes"
            )
            raise

        except Exception as e:
            signal.alarm(0)  # Cancel timeout
            logger.error(f"Prediction failed for {structure_name}: {e}")
            return False

        finally:
            signal.signal(signal.SIGALRM, old_handler)  # Restore old handler


def main():
    """Example usage of the structure predictor."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Predict protein structures using ESMFold"
    )
    parser.add_argument("--csv_path", help="Path to CSV file with predictions")
    parser.add_argument("--output_dir", help="Directory to save predicted structures")
    parser.add_argument(
        "--reference_dir", help="Directory with reference ESMFold predictions"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing predictions"
    )

    args = parser.parse_args()

    try:
        predictor = BatchStructurePredictor(
            args.csv_path, args.output_dir, args.reference_dir
        )

        stats = predictor.predict_batch(overwrite=args.overwrite)

        print("\n" + "=" * 50)
        print("PREDICTION SUMMARY")
        print("=" * 50)
        for key, value in stats.items():
            print(f"{key:20}: {value}")
        print("=" * 50)

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
