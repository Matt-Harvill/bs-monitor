import argparse
import logging
import math
import os
import pickle
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


def setup_logging(log_level="INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


class BowelSoundDataset(Dataset):
    def __init__(self, data_dir, split="train", max_length=2.0, sample_rate=16000):
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

        # Wav2Vec2 parameters
        self.frame_rate = 49.5  # Hz (Wav2Vec2 large output frequency)
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
        self.processed_data = self._load_or_preprocess_data()

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

    def _load_or_preprocess_data(self):
        """Load cached data if available, otherwise preprocess and cache"""
        if self.cache_path.exists():
            print(f"Loading cached data from {self.cache_path}")
            try:
                with open(self.cache_path, "rb") as f:
                    processed_data = pickle.load(f)
                print(f"✓ Loaded {len(processed_data)} cached samples")
                return processed_data
            except Exception as e:
                print(f"Error loading cache: {e}")
                print("Will reprocess data...")

        print(f"Processing data for {self.split} split...")
        processed_data = self._preprocess_data()

        # Cache the processed data
        print(f"Saving processed data to {self.cache_path}")
        try:
            with open(self.cache_path, "wb") as f:
                pickle.dump(processed_data, f)
            print(f"✓ Cached {len(processed_data)} samples")
        except Exception as e:
            print(f"Warning: Could not cache data: {e}")

        return processed_data

    def _preprocess_data(self):
        """Preprocess all data and cache labels"""
        processed_data = []

        # Track some statistics for debugging
        total_annotations = 0
        total_bowel_sound_frames = 0
        sample_labels_shown = 0

        for wav_file in tqdm(self.files, desc=f"Preprocessing {self.split} data"):
            wav_path = self.data_dir / wav_file
            csv_file = wav_file.replace(".wav", ".csv")
            csv_path = self.data_dir / csv_file

            if not csv_path.exists():
                continue

            # Load audio
            try:
                audio, sr = librosa.load(wav_path, sr=self.sample_rate)
            except Exception as e:
                print(f"Error loading {wav_file}: {e}")
                continue

            # Load annotations
            try:
                annotations = pd.read_csv(csv_path)
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
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

                print(f"\n📊 Sample labels from {wav_file}:")
                print(
                    f"   - Time range: {chunk_start_time:.2f}s - {chunk_end_time:.2f}s"
                )
                print(f"   - Total frames: {total_frames}")
                print(
                    f"   - Bowel sound frames: {bowel_sound_frames} ({100 * bowel_sound_frames / total_frames:.1f}%)"
                )
                print(f"   - All labels: {chunk_labels.tolist()}")

                # Show original CSV annotations for this file
                print(f"\n📋 Original CSV annotations for {csv_file}:")
                print(annotations.to_string(index=False))

                # Show which annotations fall within this 2-second segment
                chunk_annotations = []
                for _, row in annotations.iterrows():
                    if pd.isna(row["start"]) or pd.isna(row["end"]):  # type: ignore
                        continue
                    # Check if annotation overlaps with this chunk
                    if row["start"] < chunk_end_time and row["end"] > chunk_start_time:
                        chunk_annotations.append(row)

                if chunk_annotations:
                    print("\n🎯 Annotations that overlap with this 2s segment:")
                    for ann in chunk_annotations:
                        print(f"   - {ann['start']:.3f}s to {ann['end']:.3f}s")
                        # Convert to frame indices
                        start_frame = self._time_to_frame_index(ann["start"])
                        end_frame = self._time_to_frame_index(ann["end"])
                        # Adjust to chunk-relative indices
                        chunk_start_frame = max(0, start_frame)
                        chunk_end_frame = min(chunk_duration_frames, end_frame + 1)
                        print(
                            f"     -> frames {chunk_start_frame} to {chunk_end_frame} in chunk"
                        )
                else:
                    print("\n🎯 No annotations overlap with this 2s segment")

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

                    print("\n🔍 Bowel sound regions in this chunk:")
                    for start, end in bowel_regions:
                        start_time = start / self.frame_rate
                        end_time = end / self.frame_rate
                        print(
                            f"   - Frames {start}-{end}: {start_time:.3f}s - {end_time:.3f}s"
                        )

                sample_labels_shown += 1

        # Print summary statistics
        total_frames = sum(len(item["labels"]) for item in processed_data)
        total_bowel_frames = sum((item["labels"] == 1).sum() for item in processed_data)

        print(f"\n📈 {self.split.capitalize()} Data Summary:")
        print(f"   - Total files processed: {len(self.files)}")
        print(f"   - Total chunks created: {len(processed_data)}")
        print(f"   - Total annotations: {total_annotations}")
        print(f"   - Total frames: {total_frames}")
        print(f"   - Total bowel sound frames: {total_bowel_frames}")
        print(
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
    data_dir, batch_size=8, num_workers=4
) -> tuple[DataLoader[BowelSoundDataset], DataLoader[BowelSoundDataset]]:
    """Create train and validation dataloaders"""
    train_dataset: BowelSoundDataset = BowelSoundDataset(data_dir, split="train")
    test_dataset: BowelSoundDataset = BowelSoundDataset(data_dir, split="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


def train_epoch(
    model, train_loader, optimizer, criterion, device, epoch=None, logger=None
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


def evaluate(model, test_loader, criterion, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
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

    return total_loss / len(test_loader), correct_predictions / total_predictions


def evaluate_subset(model, test_loader, criterion, device, max_batches=10):
    """Evaluate on a subset of the test set for faster evaluation during training"""
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    num_batches = 0

    with torch.no_grad():
        for batch in test_loader:
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
        description="Train Wav2Vec2 for bowel sound detection"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data", help="Directory containing data"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs", help="Output directory for model"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
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
        "--eval_steps", type=int, default=50, help="Evaluate every N steps"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.log_level)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, test_loader = create_dataloaders(
        args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers
    )

    logger.info(f"Train samples: {len(train_loader.dataset)}")  # type: ignore
    logger.info(f"Test samples: {len(test_loader.dataset)}")  # type: ignore

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
    logger.info("Initializing model...")
    model = Wav2Vec2ForAudioFrameClassification.from_pretrained(
        "facebook/wav2vec2-large",
        num_labels=2,  # 0 = no bowel sound, 1 = bowel sound
    )

    # Freeze feature encoder
    model.freeze_feature_encoder()
    logger.info("Feature encoder frozen")

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

    # Training loop
    logger.info("Starting training...")
    best_acc = 0.0
    global_step = 0
    all_step_losses = []  # Track all step losses across epochs

    for epoch in range(args.num_epochs):
        logger.info(f"Epoch {epoch + 1}/{args.num_epochs}")

        # Train
        train_loss, train_acc, step_losses = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, logger
        )

        # Log step losses for this epoch
        for step_loss in step_losses:
            global_step += 1
            all_step_losses.append(step_loss)
            logger.info(f"Step {global_step}: loss = {step_loss:.4f}")

            # Evaluate periodically
            if global_step % args.eval_steps == 0:
                eval_loss, eval_acc = evaluate_subset(
                    model, test_loader, criterion, device
                )
                logger.info(
                    f"Step {global_step} - Eval Loss: {eval_loss:.4f}, Eval Acc: {eval_acc:.4f}"
                )

        # Full evaluation at end of epoch
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        logger.info(f"Epoch {epoch + 1} Summary:")
        logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logger.info(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            model.save_pretrained(os.path.join(args.output_dir, "best_model"))
            logger.info(f"New best model saved with accuracy: {best_acc:.4f}")

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "best_acc": best_acc,
            "step_losses": all_step_losses,
        }
        torch.save(checkpoint, os.path.join(args.output_dir, "checkpoint.pt"))

    logger.info(f"Training completed! Best accuracy: {best_acc:.4f}")
    logger.info(f"Model saved to: {args.output_dir}")

    # Save training history
    training_history = {
        "step_losses": all_step_losses,
        "global_step": global_step,
        "best_acc": best_acc,
    }
    torch.save(training_history, os.path.join(args.output_dir, "training_history.pt"))
    logger.info(f"Training history saved to: {args.output_dir}/training_history.pt")


if __name__ == "__main__":
    main()
