"""Multi-class dataset for bowel sound classification."""

import pickle
from pathlib import Path

import librosa
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from .config import MultiClassifierConfig
    from .data_splits import DataSplitter
    from .logging_config import get_logger
except ImportError:
    from config import MultiClassifierConfig
    from data_splits import DataSplitter
    from logging_config import get_logger


class MultiClassDataset(Dataset):
    """Dataset for multi-class bowel sound frame classification."""

    def __init__(
        self,
        config: MultiClassifierConfig,
        split: str = "train",
        data_splitter: DataSplitter | None = None,
    ):
        """
        Initialize the multi-class dataset.

        Args:
            config: Configuration object
            split: 'train', 'val', or 'test'
            data_splitter: DataSplitter instance for consistent splits
        """
        self.config = config
        self.split = split
        self.logger = get_logger()

        # Initialize data splitter
        if data_splitter is None:
            data_splitter = DataSplitter(config.data_dir)
        self.data_splitter = data_splitter

        # Get splits
        self.splits = self.data_splitter.get_splits(config.datasets, config.duration)

        # Load label files - use absolute path or relative to current script
        self.labels_dir = Path(__file__).parent / "labels"
        self.audio_paths = {}
        self.labels = {}

        # Load data for each dataset
        self._load_datasets()

        # Create samples list
        self.samples = self._create_samples()

        self.logger.info(f"Loaded {len(self.samples)} samples for {split} split")

        # Log class distribution
        self._log_class_distribution()

    def _load_datasets(self):
        """Load label files and audio paths for all datasets."""
        for dataset_name in self.config.datasets:
            # Load labels
            label_file = (
                self.labels_dir / f"{dataset_name}_{self.config.duration}s_labels.pkl"
            )
            if label_file.exists():
                with open(label_file, "rb") as f:
                    dataset_labels = pickle.load(f)
                self.labels[dataset_name] = dataset_labels

                # Set up audio paths
                dataset_path = Path(self.config.data_dir) / f"{dataset_name}-dataset"
                chunk_dir = dataset_path / f"{self.config.duration}s-chunks"
                self.audio_paths[dataset_name] = chunk_dir

                self.logger.info(
                    f"Loaded {len(dataset_labels)} total chunks from {dataset_name}"
                )
            else:
                self.logger.warning(f"Label file not found: {label_file}")

    def _create_samples(self):
        """Create list of samples for the specified split."""
        samples = []

        for dataset_name, dataset_labels in self.labels.items():
            if dataset_name not in self.splits:
                self.logger.warning(f"No splits found for dataset {dataset_name}")
                continue

            # Get chunk indices for this split
            split_indices = self.splits[dataset_name][self.split]

            for chunk_idx in split_indices:
                if chunk_idx in dataset_labels:
                    samples.append(
                        {
                            "dataset": dataset_name,
                            "chunk_idx": chunk_idx,
                            "labels": dataset_labels[chunk_idx],
                        }
                    )
                else:
                    self.logger.warning(
                        f"Chunk {chunk_idx} not found in {dataset_name} labels"
                    )

        return samples

    def _log_class_distribution(self):
        """Log the distribution of classes in the dataset."""
        class_counts = {cls: 0 for cls in self.config.class_to_idx.keys()}

        for sample in self.samples:
            labels = sample["labels"]
            unique, counts = np.unique(labels, return_counts=True)

            for label, count in zip(unique, counts, strict=False):
                class_counts[label] += count

        total_frames = sum(class_counts.values())

        self.logger.info(f"Class distribution for {self.split} split:")
        for class_name, count in class_counts.items():
            percentage = count / total_frames * 100 if total_frames > 0 else 0
            self.logger.info(f"  {class_name}: {count} ({percentage:.1f}%)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        sample = self.samples[idx]

        # Construct audio filename
        dataset_name = sample["dataset"]
        chunk_idx = sample["chunk_idx"]

        # Try different filename patterns
        possible_filenames = [
            f"{dataset_name}_chunk_{chunk_idx:04d}.wav",
            f"{dataset_name}_chunk_{chunk_idx:04d}_padded.wav",
        ]

        audio_file = None
        for filename in possible_filenames:
            potential_path = self.audio_paths[dataset_name] / filename
            if potential_path.exists():
                audio_file = potential_path
                break

        if audio_file is None:
            self.logger.error(
                f"Audio file not found for {dataset_name} chunk {chunk_idx}"
            )
            # Return zeros as fallback
            audio = np.zeros(int(self.config.max_length * self.config.sample_rate))
        else:
            # Load audio with librosa
            try:
                audio, sr = librosa.load(
                    audio_file, sr=self.config.sample_rate, mono=True
                )
            except Exception as e:
                self.logger.error(f"Error loading audio {audio_file}: {e}")
                # Return zeros as fallback
                audio = np.zeros(int(self.config.max_length * self.config.sample_rate))

        # Ensure audio is the right length
        target_length = int(self.config.max_length * self.config.sample_rate)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), "constant")
        elif len(audio) > target_length:
            audio = audio[:target_length]

        # Convert string labels to indices
        labels = sample["labels"]
        label_indices = np.array([self.config.class_to_idx[label] for label in labels])

        return {
            "audio": torch.FloatTensor(audio),
            "labels": torch.LongTensor(label_indices),
            "dataset": dataset_name,
            "chunk_idx": chunk_idx,
        }


def create_dataloaders(
    config: MultiClassifierConfig, data_splitter: DataSplitter | None = None
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.

    Args:
        config: Configuration object
        data_splitter: DataSplitter instance for consistent splits

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    logger = get_logger()

    if data_splitter is None:
        data_splitter = DataSplitter(config.data_dir)

    # Create datasets
    train_dataset = MultiClassDataset(
        config, split="train", data_splitter=data_splitter
    )
    val_dataset = MultiClassDataset(config, split="val", data_splitter=data_splitter)
    test_dataset = MultiClassDataset(config, split="test", data_splitter=data_splitter)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # Log dataset sizes
    logger.info("Dataset sizes:")
    logger.info(f"  Train: {len(train_dataset)} samples")
    logger.info(f"  Val: {len(val_dataset)} samples")
    logger.info(f"  Test: {len(test_dataset)} samples")

    return train_loader, val_loader, test_loader


def get_dataset_statistics(config: MultiClassifierConfig) -> dict[str, dict[str, int]]:
    """
    Get dataset statistics for all splits.

    Args:
        config: Configuration object

    Returns:
        Dictionary with dataset statistics
    """
    data_splitter = DataSplitter(config.data_dir)
    return data_splitter.get_split_summary(config.datasets, config.duration)
