import argparse
import logging
import math
import os
import pickle
from datetime import datetime
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import Wav2Vec2ForAudioFrameClassification

import wandb
from src.train.hubert_for_audio_frame_classification import (
    HubertForAudioFrameClassification,
)


def setup_logging(log_level="INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


class BowelSoundDataset(Dataset):
    def __init__(
        self,
        logger: logging.Logger,
        data_dir,
        split="train",
        max_length=2.0,
        sample_rate=16000,
    ):
        """
        Dataset for bowel sound detection

        Args:
            data_dir: Directory containing wav files and csv annotations
            split: 'train' or 'test'
            max_length: Maximum audio length in seconds (default: 2.0 for 2s segments)
            sample_rate: Audio sample rate
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_length = max_length
        self.sample_rate = sample_rate

        # Determine frame rate based on model
        self.frame_rate = 49.5  # Hz (output frequency for all models)

        self.stride_ms = 20  # ms between samples
        self.receptive_field_ms = 25  # ms receptive field

        # Cache directory for processed data
        self.cache_dir = self.data_dir / "processed_cache"
        self.cache_dir.mkdir(exist_ok=True)

        # Cache filename based on parameters
        self.cache_filename = f"{split}_maxlen{max_length}_sr{sample_rate}.pkl"
        self.cache_path = self.cache_dir / self.cache_filename

        # Load file list
        self.files = self._load_file_list()

        # Load or preprocess data
        self.processed_data = self._load_or_preprocess_data(logger)

    def _load_file_list(self):
        """Load list of files for the specified split"""
        files_df = pd.read_csv(self.data_dir / "files.csv")
        split_files = files_df[files_df["train/test"] == self.split][
            "filename"
        ].tolist()
        return [f for f in split_files if f.endswith(".wav")]

    def _time_to_frame_index(self, time_seconds):
        """Convert time in seconds to frame index at 49Hz"""
        return math.ceil(time_seconds * self.frame_rate)

    def _load_or_preprocess_data(self, logger: logging.Logger):
        """Load cached data if available, otherwise preprocess and cache"""
        if self.cache_path.exists():
            logger.info(f"Loading cached data from {self.cache_path}")
            try:
                with open(self.cache_path, "rb") as f:
                    processed_data = pickle.load(f)
                logger.info(f"✓ Loaded {len(processed_data)} cached samples")
                return processed_data
            except Exception as e:
                logger.error(f"Error loading cache: {e}")
                logger.info("Will reprocess data...")

        logger.info(f"Processing data for {self.split} split...")
        processed_data = self._preprocess_data(logger)

        # Cache the processed data
        logger.info(f"Saving processed data to {self.cache_path}")
        try:
            with open(self.cache_path, "wb") as f:
                pickle.dump(processed_data, f)
            logger.info(f"✓ Cached {len(processed_data)} samples")
        except Exception as e:
            logger.warning(f"Warning: Could not cache data: {e}")

        return processed_data

    def _preprocess_data(self, logger: logging.Logger):
        """Preprocess all data and cache labels"""
        processed_data = []

        # Track some statistics for debugging
        total_annotations = 0
        total_bowel_sound_frames = 0
        sample_labels_shown = 0

        for wav_file in tqdm(self.files, desc=f"Preprocessing {self.split} data"):
            wav_path = self.data_dir / "bs-dataset" / wav_file
            csv_file = wav_file.replace(".wav", ".csv")
            csv_path = self.data_dir / "bs-dataset" / csv_file

            if not csv_path.exists():
                continue

            # Load audio
            try:
                audio, sr = librosa.load(wav_path, sr=self.sample_rate)
            except Exception as e:
                logger.error(f"Error loading {wav_file}: {e}")
                continue

            # Load annotations
            try:
                annotations = pd.read_csv(csv_path)
            except Exception as e:
                logger.error(f"Error loading {csv_file}: {e}")
                continue

            # Calculate number of frames for this audio
            audio_duration = len(audio) / self.sample_rate
            num_frames = int(audio_duration * self.frame_rate)

            # Initialize labels as zeros
            labels = np.zeros(num_frames, dtype=np.int64)

            # Mark bowel sound regions as 1
            file_annotations = 0
            for _, row in annotations.iterrows():
                if pd.isna(row["start"]) or pd.isna(row["end"]):  # type: ignore
                    continue

                start_frame = self._time_to_frame_index(row["start"])
                end_frame = self._time_to_frame_index(row["end"])

                # Ensure frame indices are within bounds
                start_frame = max(0, min(start_frame, num_frames - 1))
                end_frame = max(0, min(end_frame, num_frames - 1))

                if start_frame < end_frame:
                    labels[start_frame : end_frame + 1] = 1
                    file_annotations += 1
                    total_bowel_sound_frames += end_frame - start_frame + 1

            total_annotations += file_annotations

            # For 2-second segments, we don't need to chunk - each file is one segment
            chunk_duration_frames = int(
                self.max_length * self.frame_rate
            )  # 98 frames for 2s

            # Ensure we have enough frames, pad if necessary
            if num_frames < chunk_duration_frames:
                labels = np.pad(
                    labels, (0, chunk_duration_frames - num_frames), "constant"
                )
                num_frames = chunk_duration_frames

            # Take the first chunk_duration_frames (or all if shorter)
            chunk_labels = labels[:chunk_duration_frames]

            # Calculate corresponding audio chunk
            chunk_audio = audio[
                : int(self.max_length * self.sample_rate)
            ]  # First 2 seconds

            # Pad audio if necessary
            if len(chunk_audio) < int(self.max_length * self.sample_rate):
                chunk_audio = np.pad(
                    chunk_audio,
                    (0, int(self.max_length * self.sample_rate) - len(chunk_audio)),
                    "constant",
                )

            processed_data.append(
                {
                    "audio": chunk_audio,
                    "labels": chunk_labels,
                    "filename": wav_file,
                    "chunk_idx": 0,  # Only one chunk per file
                }
            )

            # Show sample labels for first few files
            if sample_labels_shown < 3:
                bowel_sound_frames = (chunk_labels == 1).sum()
                total_frames = len(chunk_labels)
                chunk_start_time = 0.0
                chunk_end_time = self.max_length

                logger.info(f"\n📊 Sample labels from {wav_file}:")
                logger.info(
                    f"   - Time range: {chunk_start_time:.2f}s - {chunk_end_time:.2f}s"
                )
                logger.info(f"   - Total frames: {total_frames}")
                logger.info(
                    f"   - Bowel sound frames: {bowel_sound_frames} ({100 * bowel_sound_frames / total_frames:.1f}%)"
                )
                logger.info(f"   - All labels: {chunk_labels.tolist()}")

                # Show original CSV annotations for this file
                logger.info(f"\n📋 Original CSV annotations for {csv_file}:")
                logger.info(annotations.to_string(index=False))

                # Show which annotations fall within this 2-second segment
                chunk_annotations = []
                for _, row in annotations.iterrows():
                    if pd.isna(row["start"]) or pd.isna(row["end"]):  # type: ignore
                        continue
                    # Check if annotation overlaps with this chunk
                    if row["start"] < chunk_end_time and row["end"] > chunk_start_time:
                        chunk_annotations.append(row)

                if chunk_annotations:
                    logger.info("\n🎯 Annotations that overlap with this 2s segment:")
                    for ann in chunk_annotations:
                        logger.info(f"   - {ann['start']:.3f}s to {ann['end']:.3f}s")
                        # Convert to frame indices
                        start_frame = self._time_to_frame_index(ann["start"])
                        end_frame = self._time_to_frame_index(ann["end"])
                        # Adjust to chunk-relative indices
                        chunk_start_frame = max(0, start_frame)
                        chunk_end_frame = min(chunk_duration_frames, end_frame + 1)
                        logger.info(
                            f"     -> frames {chunk_start_frame} to {chunk_end_frame} in chunk"
                        )
                else:
                    logger.info("\n🎯 No annotations overlap with this 2s segment")

                if bowel_sound_frames > 0:
                    # Find all bowel sound regions in this chunk
                    bowel_regions = []
                    in_bowel = False
                    start_frame = 0
                    for frame_idx, label in enumerate(chunk_labels):
                        if label == 1 and not in_bowel:
                            start_frame = frame_idx
                            in_bowel = True
                        elif label == 0 and in_bowel:
                            bowel_regions.append((start_frame, frame_idx - 1))
                            in_bowel = False
                    if in_bowel:
                        bowel_regions.append((start_frame, len(chunk_labels) - 1))

                    logger.info("\n🔍 Bowel sound regions in this chunk:")
                    for start, end in bowel_regions:
                        start_time = start / self.frame_rate
                        end_time = end / self.frame_rate
                        logger.info(
                            f"   - Frames {start}-{end}: {start_time:.3f}s - {end_time:.3f}s"
                        )

                sample_labels_shown += 1

        # Print summary statistics
        total_frames = sum(len(item["labels"]) for item in processed_data)
        total_bowel_frames = sum((item["labels"] == 1).sum() for item in processed_data)

        logger.info(f"\n📈 {self.split.capitalize()} Data Summary:")
        logger.info(f"   - Total files processed: {len(self.files)}")
        logger.info(f"   - Total chunks created: {len(processed_data)}")
        logger.info(f"   - Total annotations: {total_annotations}")
        logger.info(f"   - Total frames: {total_frames}")
        logger.info(f"   - Total bowel sound frames: {total_bowel_frames}")
        logger.info(
            f"   - Overall bowel sound ratio: {100 * total_bowel_frames / total_frames:.2f}%"
        )

        return processed_data

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        item = self.processed_data[idx]
        return {
            "audio": torch.FloatTensor(item["audio"]),
            "labels": torch.LongTensor(item["labels"]),
            "filename": item["filename"],
            "chunk_idx": item["chunk_idx"],
        }

    def clear_cache(self):
        """Clear the cached data for this dataset"""
        if self.cache_path.exists():
            self.cache_path.unlink()
            print(f"Cleared cache: {self.cache_path}")

    @classmethod
    def clear_all_cache(cls, data_dir):
        """Clear all cached data for a data directory"""
        cache_dir = Path(data_dir) / "processed_cache"
        if cache_dir.exists():
            for cache_file in cache_dir.glob("*.pkl"):
                cache_file.unlink()
            print(f"Cleared all cache files in {cache_dir}")


def create_dataloaders(
    logger: logging.Logger, data_dir, batch_size=8, num_workers=4
) -> tuple[DataLoader[BowelSoundDataset], DataLoader[BowelSoundDataset], dict, dict]:
    """Create train and evaluation dataloaders"""
    train_dataset: BowelSoundDataset = BowelSoundDataset(
        data_dir=data_dir, split="train", logger=logger
    )
    eval_dataset: BowelSoundDataset = BowelSoundDataset(
        data_dir=data_dir, split="test", logger=logger
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Calculate dataset statistics
    train_stats = _calculate_dataset_stats(train_dataset)
    eval_stats = _calculate_dataset_stats(eval_dataset)

    return train_loader, eval_loader, train_stats, eval_stats


def _calculate_dataset_stats(dataset: BowelSoundDataset) -> dict:
    """Calculate statistics for a dataset"""
    total_frames = sum(len(item["labels"]) for item in dataset.processed_data)
    total_bowel_frames = sum(
        (item["labels"] == 1).sum() for item in dataset.processed_data
    )

    return {
        "total_files": len(dataset.files),
        "total_chunks": len(dataset.processed_data),
        "total_frames": total_frames,
        "total_bowel_frames": total_bowel_frames,
        "bowel_sound_ratio": 100 * total_bowel_frames / total_frames
        if total_frames > 0
        else 0,
    }


def cleanup_checkpoints(checkpoint_dir: str):
    """Remove all checkpoint files in the directory"""
    checkpoint_path = Path(checkpoint_dir)
    for checkpoint_file in checkpoint_path.glob("checkpoint_*.pt"):
        try:
            checkpoint_file.unlink()
            print(f"Removed checkpoint: {checkpoint_file.name}")
        except Exception as e:
            print(f"Failed to remove checkpoint {checkpoint_file.name}: {e}")


def get_checkpoint_dir(output_dir: str) -> str:
    """Create a checkpoint directory with today's date, hour, minute, and second"""
    output_path = Path(output_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_dir = output_path / "checkpoints" / timestamp
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return str(checkpoint_dir)


def train_epoch(
    model,
    train_loader,
    optimizer,
    criterion,
    device,
    epoch=None,
    logger=None,
    global_step=0,
    wandb_available=False,
):
    """Train for one epoch"""
    if logger is None:
        logger = logging.getLogger(__name__)
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    step_losses = []  # Track loss for each step

    progress_bar = tqdm(train_loader, desc="Training")

    for batch in progress_bar:
        audio = batch["audio"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(audio)
        logits = outputs.logits  # Shape: (batch_size, sequence_length, num_labels)

        # Get the actual sequence length from model output
        batch_size, seq_len, num_labels = logits.shape

        # Ensure labels match the model's output sequence length
        if labels.size(1) != seq_len:
            # Pad or truncate labels to match model output
            if labels.size(1) > seq_len:
                # Truncate labels
                labels = labels[:, :seq_len]
            else:
                # Pad labels with zeros
                padding = torch.zeros(
                    batch_size,
                    seq_len - labels.size(1),
                    dtype=labels.dtype,
                    device=labels.device,
                )
                labels = torch.cat([labels, padding], dim=1)

        # Reshape for loss calculation
        logits_flat = logits.view(-1, num_labels)
        labels_flat = labels.view(-1)

        # Calculate loss
        loss = criterion(logits_flat, labels_flat)

        # Backward pass
        loss.backward()

        # Debug: Check if gradients are being computed (only on first batch of first epoch)
        if epoch == 0 and progress_bar.n == 0:
            logger.debug("\n🔍 Gradient debugging (first batch):")
            total_grad_norm = 0
            num_params_with_grad = 0
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    total_grad_norm += grad_norm
                    num_params_with_grad += 1
                    logger.debug(f"   ✓ {name}: grad_norm = {grad_norm:.6f}")
                elif param.requires_grad:
                    logger.debug(f"   ❌ {name}: No gradient computed!")

            if num_params_with_grad > 0:
                logger.debug(
                    f"   - Average gradient norm: {total_grad_norm / num_params_with_grad:.6f}"
                )
                logger.debug(f"   - Parameters with gradients: {num_params_with_grad}")
            else:
                logger.debug("   ⚠️  No gradients computed for any parameters!")

        optimizer.step()

        # Calculate accuracy
        predictions = torch.argmax(logits_flat, dim=1)
        correct_predictions += (predictions == labels_flat).sum().item()
        total_predictions += labels_flat.size(0)

        total_loss += loss.item()
        step_losses.append(loss.item())  # Store step loss

        # Log step metrics to wandb
        current_step = global_step + progress_bar.n
        if wandb_available:
            wandb.log(
                {
                    "train/step_loss": loss.item(),
                    "train/step_accuracy": correct_predictions / total_predictions,
                    "train/step": current_step,
                }
            )

        # Update progress bar
        progress_bar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "acc": f"{100 * correct_predictions / total_predictions:.2f}%",
            }
        )

    return (
        total_loss / len(train_loader),
        correct_predictions / total_predictions,
        step_losses,
    )


def evaluate_subset(model, eval_loader, criterion, device, max_batches=10):
    """Evaluate on a subset of the evaluation set for faster evaluation during training"""
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    num_batches = 0

    with torch.no_grad():
        for batch in eval_loader:
            if num_batches >= max_batches:
                break

            audio = batch["audio"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(audio)
            logits = outputs.logits

            # Get the actual sequence length from model output
            batch_size, seq_len, num_labels = logits.shape

            # Ensure labels match the model's output sequence length
            if labels.size(1) != seq_len:
                # Pad or truncate labels to match model output
                if labels.size(1) > seq_len:
                    # Truncate labels
                    labels = labels[:, :seq_len]
                else:
                    # Pad labels with zeros
                    padding = torch.zeros(
                        batch_size,
                        seq_len - labels.size(1),
                        dtype=labels.dtype,
                        device=labels.device,
                    )
                    labels = torch.cat([labels, padding], dim=1)

            # Reshape for loss calculation
            logits_flat = logits.view(-1, num_labels)
            labels_flat = labels.view(-1)

            # Calculate loss
            loss = criterion(logits_flat, labels_flat)

            # Calculate accuracy
            predictions = torch.argmax(logits_flat, dim=1)
            correct_predictions += (predictions == labels_flat).sum().item()
            total_predictions += labels_flat.size(0)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches, correct_predictions / total_predictions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train HuBERT/Wav2Vec2 for bowel sound detection"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing data (files should be in data/bs-dataset/)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs", help="Output directory for model"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=5e-4, help="Learning rate"
    )
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument(
        "--max_length",
        type=float,
        default=2.0,
        help="Maximum audio length in seconds (default: 2.0 for 2s segments)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for dataloader"
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=float("inf"),
        help="Additionally evaluate every N steps, by default disabled (evals after every epoch)",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="bowel-sound-monitor",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Weights & Biases run name (optional)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/hubert-xlarge-ll60k",
        choices=[
            "facebook/wav2vec2-large",
            "facebook/wav2vec2-base",
            "facebook/hubert-xlarge-ll60k",
        ],
        help="Audio model to use for training",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training from",
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.log_level)

    # Initialize wandb
    try:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "data_dir": args.data_dir,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "num_epochs": args.num_epochs,
                "max_length": args.max_length,
                "num_workers": args.num_workers,
                "eval_steps": args.eval_steps,
                "model": args.model,
                "num_labels": 2,
            },
        )
        logger.info("Weights & Biases logging initialized")
        wandb_available = True
    except Exception as e:
        logger.warning(f"Failed to initialize Weights & Biases: {e}")
        logger.warning("Continuing without wandb logging")
        wandb_available = False

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create checkpoint directory for today
    checkpoint_dir = get_checkpoint_dir(args.output_dir)
    logger.info(f"Checkpoint directory: {checkpoint_dir}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, eval_loader, train_stats, eval_stats = create_dataloaders(
        logger, args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers
    )

    logger.info(f"Train samples: {len(train_loader.dataset)}")  # type: ignore
    logger.info(f"Eval samples: {len(eval_loader.dataset)}")  # type: ignore
    logger.info(f"Using frame rate: {train_loader.dataset.frame_rate} Hz")  # type: ignore

    # Log dataset statistics to wandb
    if wandb_available:
        wandb.log(
            {
                "dataset/train_samples": len(train_loader.dataset),  # type: ignore
                "dataset/eval_samples": len(eval_loader.dataset),  # type: ignore
                "dataset/train_files": train_stats["total_files"],
                "dataset/train_chunks": train_stats["total_chunks"],
                "dataset/train_frames": train_stats["total_frames"],
                "dataset/train_bowel_frames": train_stats["total_bowel_frames"],
                "dataset/train_bowel_ratio": train_stats["bowel_sound_ratio"],
                "dataset/eval_files": eval_stats["total_files"],
                "dataset/eval_chunks": eval_stats["total_chunks"],
                "dataset/eval_frames": eval_stats["total_frames"],
                "dataset/eval_bowel_frames": eval_stats["total_bowel_frames"],
                "dataset/eval_bowel_ratio": eval_stats["bowel_sound_ratio"],
                "dataset/frame_rate": train_loader.dataset.frame_rate,  # type: ignore
            }
        )

    # Show sample data from first batch
    logger.debug("Sample data from first training batch:")
    sample_batch = next(iter(train_loader))
    logger.debug(f"  - Batch audio shape: {sample_batch['audio'].shape}")
    logger.debug(f"  - Batch labels shape: {sample_batch['labels'].shape}")

    # Show label distribution in this batch
    batch_labels = sample_batch["labels"].flatten()
    num_zeros = (batch_labels == 0).sum().item()
    num_ones = (batch_labels == 1).sum().item()
    total_labels = batch_labels.numel()
    logger.debug(f"  - Label distribution: {num_zeros} zeros, {num_ones} ones")
    logger.debug(f"  - Bowel sound ratio: {100 * num_ones / total_labels:.2f}%")

    # Show sample labels from first item
    first_item_labels = sample_batch["labels"][0]
    logger.debug(
        f"  - First item labels (first 30 frames): {first_item_labels[:30].tolist()}"
    )

    # Show filenames in this batch
    logger.debug(f"  - Files in batch: {sample_batch['filename']}")

    # Initialize model
    logger.info(f"Initializing model: {args.model}")

    # Determine the appropriate model class based on the model name
    if "hubert" in args.model.lower():
        model = HubertForAudioFrameClassification.from_pretrained(
            args.model,
            num_labels=2,  # 0 = no bowel sound, 1 = bowel sound
        )
    else:
        # For wav2vec2 models, use the frame classification head
        model = Wav2Vec2ForAudioFrameClassification.from_pretrained(
            args.model,
            num_labels=2,  # 0 = no bowel sound, 1 = bowel sound
        )

    # Freeze feature encoder (only for models that support it)
    if hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()
        logger.info("Feature encoder frozen")
    else:
        logger.info("Feature encoder freezing not supported for this model")

    # Debug: Check which parameters are trainable
    logger.debug("Parameter training status:")
    total_params = 0
    trainable_params = 0
    frozen_params = 0

    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            logger.debug(f"  ✓ Trainable: {name} ({param.numel():,} params)")
        else:
            frozen_params += param.numel()
            logger.debug(f"  ❌ Frozen: {name} ({param.numel():,} params)")

    logger.info("Parameter Summary:")
    logger.info(f"  - Total parameters: {total_params:,}")
    logger.info(
        f"  - Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.1f}%)"
    )
    logger.info(
        f"  - Frozen parameters: {frozen_params:,} ({100 * frozen_params / total_params:.1f}%)"
    )

    # Log model parameters to wandb
    if wandb_available:
        wandb.log(
            {
                "model/total_parameters": total_params,
                "model/trainable_parameters": trainable_params,
                "model/frozen_parameters": frozen_params,
                "model/trainable_percentage": 100 * trainable_params / total_params,
            }
        )

    # Move model to device
    model = model.to(device)  # type: ignore

    # Test model output shape with sample data
    logger.debug("Testing model output shape:")
    model.eval()
    with torch.no_grad():
        sample_audio = sample_batch["audio"].to(device)
        sample_outputs = model(sample_audio)
        logger.debug(f"  - Model output logits shape: {sample_outputs.logits.shape}")
        logger.debug(f"  - Expected labels shape: {sample_batch['labels'].shape}")
        logger.debug(
            f"  - Sequence length mismatch: {sample_outputs.logits.shape[1]} vs {sample_batch['labels'].shape[1]}"
        )
    model.train()

    # Setup training
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Initialize training state variables
    start_epoch = 0
    global_step = 0
    best_acc = 0.0
    all_step_losses = []

    # Resume from checkpoint if specified
    if args.resume_from:
        if os.path.exists(args.resume_from):
            logger.info(f"Resuming from checkpoint: {args.resume_from}")
            checkpoint = torch.load(args.resume_from, map_location=device)

            # Load model state
            model.load_state_dict(checkpoint["model_state_dict"])

            # Load optimizer state
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            # Load training state
            start_epoch = checkpoint["epoch"] + 1
            global_step = checkpoint["global_step"]
            # Reset best_acc to 0.0 for current run tracking (not from previous checkpoint)
            best_acc = 0.0
            all_step_losses = checkpoint.get("step_losses", [])

            logger.info(f"Resumed from epoch {checkpoint['epoch']}, step {global_step}")
            logger.info("Best accuracy tracking reset to 0.0 for current run")

            # Show checkpoint metadata if available
            if "timestamp" in checkpoint:
                logger.info(f"Checkpoint timestamp: {checkpoint['timestamp']}")
            if "model_name" in checkpoint:
                logger.info(f"Checkpoint model: {checkpoint['model_name']}")
            if "learning_rate" in checkpoint:
                logger.info(f"Checkpoint learning rate: {checkpoint['learning_rate']}")

            # Log resume info to wandb
            if wandb_available:
                wandb.log(
                    {
                        "training/resumed_from": args.resume_from,
                        "training/resume_epoch": checkpoint["epoch"],
                        "training/resume_step": global_step,
                        "training/resume_best_acc_reset": True,
                    }
                )
        else:
            logger.warning(f"Checkpoint file not found: {args.resume_from}")
            logger.warning("Starting training from scratch")

    # Log optimizer and criterion info to wandb
    if wandb_available:
        wandb.log(
            {
                "training/optimizer": "AdamW",
                "training/learning_rate": args.learning_rate,
                "training/criterion": "CrossEntropyLoss",
            }
        )

    # Training loop
    logger.info("Starting training...")

    for epoch in range(start_epoch, args.num_epochs):
        logger.info(f"Epoch {epoch + 1}/{args.num_epochs}")

        # Train
        train_loss, train_acc, step_losses = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch,
            logger,
            global_step,
            wandb_available,
        )

        # Log step losses for this epoch
        for step_loss in step_losses:
            global_step += 1
            all_step_losses.append(step_loss)
            logger.debug(f"Step {global_step}: loss = {step_loss:.4f}")

            # Evaluate periodically
            if global_step % args.eval_steps == 0:
                eval_loss, eval_acc = evaluate_subset(
                    model, eval_loader, criterion, device
                )
                logger.debug(
                    f"Step {global_step} - Eval Loss: {eval_loss:.4f}, Eval Acc: {eval_acc:.4f}"
                )
                # Log evaluation metrics to wandb
                if wandb_available:
                    wandb.log(
                        {
                            "eval/step_loss": eval_loss,
                            "eval/step_accuracy": eval_acc,
                            "eval/step": global_step,
                        }
                    )

        # Evaluation at end of epoch (using subset of test data for efficiency)
        eval_loss, eval_acc = evaluate_subset(model, eval_loader, criterion, device)

        logger.info(f"Epoch {epoch + 1} Summary:")
        logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logger.info(f"  Eval Loss: {eval_loss:.4f}, Eval Acc: {eval_acc:.4f}")

        # Log epoch metrics to wandb
        if wandb_available:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train/epoch_loss": train_loss,
                    "train/epoch_accuracy": train_acc,
                    "eval/epoch_loss": eval_loss,
                    "eval/epoch_accuracy": eval_acc,
                }
            )

        # Save checkpoint only if this is the best eval_acc so far in this run
        if eval_acc > best_acc:
            best_acc = eval_acc

            # Remove any existing checkpoints
            cleanup_checkpoints(checkpoint_dir)

            # Save new best checkpoint
            checkpoint = {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "train_acc": train_acc,
                "eval_loss": eval_loss,
                "eval_acc": eval_acc,
                "best_acc": best_acc,
                "step_losses": all_step_losses,
                "timestamp": datetime.now().isoformat(),
                "model_name": args.model,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
            }
            checkpoint_filename = f"checkpoint_epoch{epoch+1}_step{global_step}.pt"
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
            torch.save(checkpoint, checkpoint_path)
            logger.info(
                f"New best checkpoint saved: {checkpoint_filename} (eval_acc: {best_acc:.4f})"
            )

            # Save best model
            model.save_pretrained(os.path.join(checkpoint_dir, "best_model"))
            logger.info(f"New best model saved with accuracy: {best_acc:.4f}")

            # Log new best accuracy to wandb
            if wandb_available:
                wandb.log({"best_accuracy": best_acc})
        else:
            logger.info(
                f"No new best accuracy. Current best: {best_acc:.4f}, This epoch: {eval_acc:.4f}"
            )

    logger.info(f"Training completed! Best accuracy: {best_acc:.4f}")
    logger.info(f"Model saved to: {args.output_dir}")

    # Log final training summary to wandb
    if wandb_available:
        wandb.log(
            {
                "training/final_best_accuracy": best_acc,
                "training/total_steps": global_step,
                "training/total_epochs": args.num_epochs,
            }
        )

    # Save training history
    training_history = {
        "step_losses": all_step_losses,
        "global_step": global_step,
        "best_acc": best_acc,
    }
    torch.save(training_history, os.path.join(checkpoint_dir, "training_history.pt"))
    logger.info(f"Training history saved to: {checkpoint_dir}/training_history.pt")

    # Finish wandb run
    if wandb_available:
        wandb.finish()


if __name__ == "__main__":
    main()
