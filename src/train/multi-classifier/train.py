"""Train multi-class classifier for bowel sound classification."""

import argparse
import math
import os
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from transformers import Wav2Vec2ForAudioFrameClassification

import wandb

# Use absolute imports when running as module
try:
    from .config import MultiClassifierConfig
    from .data_splits import DataSplitter
    from .datasets import create_dataloaders
    from .hubert_for_audio_frame_classification import HubertForAudioFrameClassification
    from .logging_config import get_logger
except ImportError:
    # Use direct imports when running as script
    from config import MultiClassifierConfig
    from data_splits import DataSplitter
    from datasets import create_dataloaders
    from hubert_for_audio_frame_classification import HubertForAudioFrameClassification
    from logging_config import get_logger


def train_epoch(
    model,
    train_loader,
    optimizer,
    criterion,
    device,
    accumulation_steps=1,
    wandb_log=True,
    log_every_n_steps=10,
    scheduler=None,
):
    """Train for one epoch with gradient accumulation."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    accumulated_loss = 0
    optimizer_steps = 0

    # Calculate total optimizer steps for progress bar
    total_optimizer_steps = len(train_loader) // accumulation_steps
    if len(train_loader) % accumulation_steps != 0:
        total_optimizer_steps += 1

    progress_bar = tqdm(total=total_optimizer_steps, desc="Training (optimizer steps)")

    for idx, batch in enumerate(train_loader):
        audio = batch["audio"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        outputs = model(audio)
        logits = outputs.logits  # (batch_size, seq_len, num_classes)

        # Ensure labels match model output length
        seq_len = logits.size(1)
        if labels.size(1) != seq_len:
            if labels.size(1) > seq_len:
                labels = labels[:, :seq_len]
            else:
                labels = torch.nn.functional.pad(labels, (0, seq_len - labels.size(1)))

        # Calculate loss - NOT scaled by accumulation steps for tracking
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        # Scale loss by accumulation steps only for backward
        scaled_loss = loss / accumulation_steps
        scaled_loss.backward()

        # Track the accumulated loss (unscaled)
        accumulated_loss += loss.item()

        # Calculate accuracy
        predictions = torch.argmax(logits, dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.numel()

        # Update weights every accumulation_steps
        if (idx + 1) % accumulation_steps == 0 or (idx + 1) == len(train_loader):
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
            optimizer_steps += 1

            # Calculate average loss for this optimizer step
            step_loss = accumulated_loss / accumulation_steps
            total_loss += step_loss
            accumulated_loss = 0

            # Update progress bar
            progress_bar.update(1)
            effective_batch_size = train_loader.batch_size * accumulation_steps
            progress_bar.set_postfix(
                {
                    "loss": f"{step_loss:.4f}",
                    "acc": f"{100 * correct / total:.2f}%",
                    "eff_batch": effective_batch_size,
                }
            )

            # Log to wandb every N optimizer steps
            if wandb_log and optimizer_steps % log_every_n_steps == 0:
                import wandb

                log_dict = {
                    "train/step_loss": total_loss / optimizer_steps,
                    "train/step_accuracy": correct / total,
                    "train/optimizer_steps": optimizer_steps,
                }

                # Add current learning rate if scheduler is used
                if scheduler is not None:
                    log_dict["train/learning_rate"] = scheduler.get_last_lr()[0]

                wandb.log(log_dict)

    progress_bar.close()
    return total_loss / optimizer_steps, correct / total


def evaluate(model, loader, criterion, device):
    """Evaluate model on given loader."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            audio = batch["audio"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(audio)
            logits = outputs.logits

            # Ensure labels match model output length
            seq_len = logits.size(1)
            if labels.size(1) != seq_len:
                if labels.size(1) > seq_len:
                    labels = labels[:, :seq_len]
                else:
                    labels = torch.nn.functional.pad(
                        labels, (0, seq_len - labels.size(1))
                    )

            # Calculate loss
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

            # Calculate accuracy
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.numel()

            total_loss += loss.item()

    return total_loss / len(loader), correct / total


def get_lr_schedule_fn(warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.0):
    """Create learning rate schedule function with warmup + cosine decay."""

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            # Linear warmup from 1/warmup_steps to 1
            return (current_step + 1) / warmup_steps
        else:
            # Cosine decay
            progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
            progress = min(progress, 1.0)  # Clamp to [0, 1]
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay

    return lr_lambda


def get_checkpoint_dir(output_dir: str) -> str:
    """Create checkpoint directory with timestamp."""
    output_path = Path(output_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_dir = output_path / "checkpoints" / timestamp
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return str(checkpoint_dir)


def main():
    parser = argparse.ArgumentParser(description="Train multi-class classifier")
    parser.add_argument("--config", type=str, help="Path to config file (optional)")
    parser.add_argument("--learning_rate", type=float, help="Override learning rate")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument(
        "--accumulation_steps", type=int, help="Gradient accumulation steps"
    )
    parser.add_argument("--num_epochs", type=int, help="Override number of epochs")
    parser.add_argument(
        "--log_every_n_steps", type=int, help="Log to wandb every N optimizer steps"
    )
    parser.add_argument("--wandb_run_name", type=str, help="W&B run name")
    parser.add_argument("--resume_from", type=str, help="Resume from checkpoint")
    parser.add_argument(
        "--skip_optimizer_load",
        action="store_true",
        help="Skip loading optimizer state when resuming",
    )
    parser.add_argument(
        "--no_lr_schedule", action="store_true", help="Disable LR schedule"
    )
    parser.add_argument("--warmup_steps", type=int, help="Number of warmup steps")
    parser.add_argument(
        "--min_lr_ratio", type=float, help="Min LR as ratio of initial LR"
    )

    args = parser.parse_args()

    # Initialize config
    config = MultiClassifierConfig()

    # Override config with command line args
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.accumulation_steps:
        config.accumulation_steps = args.accumulation_steps
    if args.num_epochs:
        config.num_epochs = args.num_epochs
    if args.log_every_n_steps:
        config.log_every_n_steps = args.log_every_n_steps
    if args.wandb_run_name:
        config.wandb_run_name = args.wandb_run_name
    if args.resume_from:
        config.resume_from = args.resume_from
    if args.no_lr_schedule:
        config.use_lr_schedule = False
    if args.warmup_steps:
        config.warmup_steps = args.warmup_steps
    if args.min_lr_ratio is not None:
        config.min_lr_ratio = args.min_lr_ratio

    # Setup logging
    logger = get_logger()

    # Log effective batch size
    effective_batch_size = config.batch_size * config.accumulation_steps
    logger.info(
        f"Batch size: {config.batch_size}, Accumulation steps: {config.accumulation_steps}"
    )
    logger.info(f"Effective batch size: {effective_batch_size}")

    # Initialize wandb
    try:
        wandb_config = vars(config).copy()
        wandb_config["effective_batch_size"] = effective_batch_size
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=wandb_config,
        )
        logger.info("W&B initialized")
        wandb_available = True
    except Exception as e:
        logger.warning(f"Failed to initialize W&B: {e}")
        wandb_available = False

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create data splitter for consistent splits
    data_splitter = DataSplitter(config.data_dir)

    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(config, data_splitter)

    # Initialize model
    logger.info(f"Initializing model: {config.model_name}")
    if "hubert" in config.model_name.lower():
        model = HubertForAudioFrameClassification.from_pretrained(
            config.model_name, num_labels=config.num_classes
        )
    else:
        model = Wav2Vec2ForAudioFrameClassification.from_pretrained(
            config.model_name, num_labels=config.num_classes
        )

    # Freeze feature encoder
    if hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()
        logger.info("Feature encoder frozen")

    # Setup optimizer and criterion
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)

    # Setup learning rate scheduler if enabled
    scheduler = None
    if config.use_lr_schedule:
        # Calculate total training steps
        steps_per_epoch = len(train_loader) // config.accumulation_steps
        if len(train_loader) % config.accumulation_steps != 0:
            steps_per_epoch += 1
        total_steps = steps_per_epoch * config.num_epochs

        logger.info(
            f"Learning rate schedule: warmup_steps={config.warmup_steps}, total_steps={total_steps}"
        )
        logger.info(
            f"Initial LR: {config.learning_rate}, Min LR ratio: {config.min_lr_ratio}"
        )

        lr_lambda = get_lr_schedule_fn(
            config.warmup_steps, total_steps, config.min_lr_ratio
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Use weighted cross entropy if class weights provided
    if config.class_weights:
        weight = torch.tensor(config.class_weights, device=device)
        criterion = nn.CrossEntropyLoss(weight=weight)
        logger.info(
            f"Using weighted cross entropy with weights: {config.class_weights}"
        )
    else:
        criterion = nn.CrossEntropyLoss()

    # Initialize training state
    start_epoch = 0
    best_val_acc = 0.0
    checkpoint_dir = get_checkpoint_dir(config.output_dir)
    logger.info(f"Checkpoint directory: {checkpoint_dir}")

    # Move model to device before loading checkpoint
    model = model.to(device)

    # Resume from checkpoint if specified
    if config.resume_from and os.path.exists(config.resume_from):
        logger.info(f"Resuming from checkpoint: {config.resume_from}")
        # Load checkpoint more efficiently
        checkpoint = torch.load(config.resume_from, map_location="cpu")

        # Load model state
        model.load_state_dict(checkpoint["model_state_dict"])

        # Only load optimizer state if we're continuing training
        # This avoids memory issues when just evaluating
        if "optimizer_state_dict" in checkpoint and not args.skip_optimizer_load:
            # Create a new optimizer with current model parameters
            optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                # Move optimizer state to device
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)
            except RuntimeError as e:
                logger.warning(f"Failed to load optimizer state: {e}")
                logger.info("Creating fresh optimizer instead")
                optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        logger.info(f"Resumed from epoch {checkpoint['epoch']}")

        # Clean up checkpoint from memory
        del checkpoint
        torch.cuda.empty_cache()

    # Training loop
    logger.info("Starting training...")
    for epoch in range(start_epoch, config.num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{config.num_epochs}")

        # Train
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            config.accumulation_steps,
            wandb_available,
            config.log_every_n_steps,
            scheduler,
        )

        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Log to wandb
        if wandb_available:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "train/accuracy": train_acc,
                    "val/loss": val_loss,
                    "val/accuracy": val_acc,
                }
            )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc

            # Save checkpoint
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "best_val_acc": best_val_acc,
                "config": vars(config),
            }

            # Save scheduler state if available
            if scheduler is not None:
                checkpoint["scheduler_state_dict"] = scheduler.state_dict()

            checkpoint_path = os.path.join(checkpoint_dir, "best_checkpoint.pt")
            torch.save(checkpoint, checkpoint_path)

            # Save model in HuggingFace format
            model.save_pretrained(os.path.join(checkpoint_dir, "best_model"))

            logger.info(
                f"New best model saved with validation accuracy: {best_val_acc:.4f}"
            )

            if wandb_available:
                wandb.log({"best_val_accuracy": best_val_acc})

    # Final test evaluation
    logger.info("\nFinal evaluation on test set...")
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    logger.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    if wandb_available:
        wandb.log(
            {
                "test/loss": test_loss,
                "test/accuracy": test_acc,
            }
        )
        wandb.finish()

    logger.info(f"\nTraining completed! Best validation accuracy: {best_val_acc:.4f}")
    logger.info(f"Model saved to: {checkpoint_dir}")


if __name__ == "__main__":
    main()
