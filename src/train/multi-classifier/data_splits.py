"""Data splitting utilities for consistent train/val/test splits."""

import json
import pickle
import random
from pathlib import Path

import numpy as np

try:
    from .logging_config import get_logger
except ImportError:
    from logging_config import get_logger


class DataSplitter:
    """Handle consistent data splitting across datasets."""

    def __init__(self, data_dir: str, seed: int = 42):
        """
        Initialize data splitter.

        Args:
            data_dir: Base data directory
            seed: Random seed for reproducible splits
        """
        self.data_dir = Path(data_dir)
        self.seed = seed
        self.logger = get_logger()

        # Set random seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)

        # Split ratios
        self.train_ratio = 0.8
        self.val_ratio = 0.1
        self.test_ratio = 0.1

        # Path for saving split information
        self.splits_dir = Path(__file__).parent / "splits"
        self.splits_dir.mkdir(parents=True, exist_ok=True)

    def create_splits(
        self, datasets: list[str], duration: int = 2
    ) -> dict[str, dict[str, list[int]]]:
        """
        Create train/val/test splits for all datasets.

        Args:
            datasets: List of dataset names (e.g., ['AS_1', '23M74M'])
            duration: Chunk duration in seconds

        Returns:
            Dictionary mapping dataset names to split dictionaries
        """
        splits_file = self.splits_dir / f"splits_{duration}s_seed{self.seed}.json"

        # Load existing splits if available
        if splits_file.exists():
            self.logger.info(f"Loading existing splits from {splits_file}")
            with open(splits_file) as f:
                return json.load(f)

        self.logger.info(f"Creating new data splits with seed {self.seed}")
        all_splits = {}

        for dataset_name in datasets:
            # Load labels to get chunk indices
            labels_file = (
                Path(__file__).parent
                / "labels"
                / f"{dataset_name}_{duration}s_labels.pkl"
            )

            if not labels_file.exists():
                self.logger.warning(f"Labels file not found: {labels_file}")
                continue

            with open(labels_file, "rb") as f:
                labels = pickle.load(f)

            chunk_indices = list(labels.keys())
            chunk_indices.sort()  # Ensure consistent ordering

            # Shuffle indices
            random.shuffle(chunk_indices)

            # Calculate split sizes
            total_chunks = len(chunk_indices)
            train_size = int(total_chunks * self.train_ratio)
            val_size = int(total_chunks * self.val_ratio)
            # test_size is the remainder

            # Create splits
            train_indices = chunk_indices[:train_size]
            val_indices = chunk_indices[train_size : train_size + val_size]
            test_indices = chunk_indices[train_size + val_size :]

            splits = {"train": train_indices, "val": val_indices, "test": test_indices}

            all_splits[dataset_name] = splits

            self.logger.info(f"Dataset {dataset_name} splits:")
            self.logger.info(
                f"  Train: {len(train_indices)} chunks ({len(train_indices)/total_chunks*100:.1f}%)"
            )
            self.logger.info(
                f"  Val: {len(val_indices)} chunks ({len(val_indices)/total_chunks*100:.1f}%)"
            )
            self.logger.info(
                f"  Test: {len(test_indices)} chunks ({len(test_indices)/total_chunks*100:.1f}%)"
            )

        # Save splits to file
        with open(splits_file, "w") as f:
            json.dump(all_splits, f, indent=2)

        self.logger.info(f"Saved splits to {splits_file}")
        return all_splits

    def get_splits(
        self, datasets: list[str], duration: int = 2
    ) -> dict[str, dict[str, list[int]]]:
        """
        Get existing splits or create new ones.

        Args:
            datasets: List of dataset names
            duration: Chunk duration in seconds

        Returns:
            Dictionary mapping dataset names to split dictionaries
        """
        return self.create_splits(datasets, duration)

    def get_split_summary(
        self, datasets: list[str], duration: int = 2
    ) -> dict[str, dict[str, int]]:
        """
        Get summary statistics for splits.

        Args:
            datasets: List of dataset names
            duration: Chunk duration in seconds

        Returns:
            Dictionary with split summary statistics
        """
        splits = self.get_splits(datasets, duration)
        summary = {}

        for dataset_name, dataset_splits in splits.items():
            summary[dataset_name] = {
                "train_count": len(dataset_splits["train"]),
                "val_count": len(dataset_splits["val"]),
                "test_count": len(dataset_splits["test"]),
                "total_count": len(dataset_splits["train"])
                + len(dataset_splits["val"])
                + len(dataset_splits["test"]),
            }

        return summary
