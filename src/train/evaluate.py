import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Wav2Vec2ForAudioFrameClassification

from src.train.hubert_for_audio_frame_classification import (
    HubertForAudioFrameClassification,
)
from src.train.train import BowelSoundDataset, setup_logging


def evaluate_full_dataset(
    model,
    eval_loader,
    criterion,
    device,
    logger,
):
    """Evaluate on the full evaluation dataset"""
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    # Track per-file metrics
    file_metrics = {}

    # Track confusion matrix
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    with torch.no_grad():
        progress_bar = tqdm(eval_loader, desc="Evaluating")

        for batch in progress_bar:
            audio = batch["audio"].to(device)
            labels = batch["labels"].to(device)
            filenames = batch["filename"]

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

            # Calculate accuracy and confusion matrix
            predictions = torch.argmax(logits_flat, dim=1)
            correct_predictions += (predictions == labels_flat).sum().item()
            total_predictions += labels_flat.size(0)

            # Update confusion matrix
            true_positives += ((predictions == 1) & (labels_flat == 1)).sum().item()
            false_positives += ((predictions == 1) & (labels_flat == 0)).sum().item()
            true_negatives += ((predictions == 0) & (labels_flat == 0)).sum().item()
            false_negatives += ((predictions == 0) & (labels_flat == 1)).sum().item()

            total_loss += loss.item()

            # Calculate per-file metrics
            for i, filename in enumerate(filenames):
                start_idx = i * seq_len
                end_idx = (i + 1) * seq_len

                file_pred = predictions[start_idx:end_idx]
                file_labels = labels_flat[start_idx:end_idx]

                file_correct = (file_pred == file_labels).sum().item()
                file_total = file_labels.size(0)

                if filename not in file_metrics:
                    file_metrics[filename] = {"correct": 0, "total": 0, "accuracy": 0.0}

                file_metrics[filename]["correct"] += file_correct
                file_metrics[filename]["total"] += file_total
                file_metrics[filename]["accuracy"] = (
                    file_metrics[filename]["correct"] / file_metrics[filename]["total"]
                )

            # Update progress bar
            current_acc = correct_predictions / total_predictions
            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{100 * current_acc:.2f}%",
                }
            )

    # Calculate final metrics
    final_loss = total_loss / len(eval_loader)
    final_accuracy = correct_predictions / total_predictions

    # Calculate precision, recall, F1-score
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0
    )
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    # Calculate specificity
    specificity = (
        true_negatives / (true_negatives + false_positives)
        if (true_negatives + false_positives) > 0
        else 0
    )

    return {
        "loss": final_loss,
        "accuracy": final_accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "specificity": specificity,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
        "total_predictions": total_predictions,
        "file_metrics": file_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate trained HuBERT/Wav2Vec2 model for bowel sound detection"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data", help="Directory containing data"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model (HuggingFace format with config.json and safetensors)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for evaluation"
    )
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
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save evaluation results (optional)",
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.log_level)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create test dataset and dataloader
    logger.info("Creating test dataset...")
    test_dataset = BowelSoundDataset(
        data_dir=args.data_dir, split="test", logger=logger, max_length=args.max_length
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    logger.info(f"Test samples: {len(test_dataset)}")
    logger.info(f"Using frame rate: {test_dataset.frame_rate} Hz")

    # Load model
    logger.info(f"Loading model from: {args.model_path}")

    # Check if model path exists
    if not os.path.exists(args.model_path):
        logger.error(f"Model path does not exist: {args.model_path}")
        return

    # Check if it's a HuggingFace format directory
    config_path = os.path.join(args.model_path, "config.json")
    if not os.path.exists(config_path):
        logger.error(f"Config file not found at: {config_path}")
        logger.error("Please provide a path to a HuggingFace format model directory")
        return

    try:
        # Load config to determine model type
        import json

        with open(config_path) as f:
            config = json.load(f)

        model_type = config.get("model_type", "").lower()

        if "hubert" in model_type:
            logger.info("Loading HuBERT model...")
            model = HubertForAudioFrameClassification.from_pretrained(args.model_path)
        elif "wav2vec2" in model_type:
            logger.info("Loading Wav2Vec2 model...")
            model = Wav2Vec2ForAudioFrameClassification.from_pretrained(args.model_path)
        else:
            logger.warning(f"Unknown model type: {model_type}")
            logger.info("Attempting to load as HuBERT model...")
            model = HubertForAudioFrameClassification.from_pretrained(args.model_path)

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return

    # Move model to device
    model = model.to(device)  # type: ignore
    model.eval()

    # Setup criterion
    criterion = nn.CrossEntropyLoss()

    # Test model with a sample batch
    logger.info("Testing model with sample data...")
    sample_batch = next(iter(test_loader))
    sample_audio = sample_batch["audio"].to(device)

    with torch.no_grad():
        sample_outputs = model(sample_audio)
        logger.info(f"Model output shape: {sample_outputs.logits.shape}")
        logger.info(f"Expected labels shape: {sample_batch['labels'].shape}")

    # Evaluate on full test set
    logger.info("Starting evaluation on full test set...")
    results = evaluate_full_dataset(model, test_loader, criterion, device, logger)

    # Print results
    logger.info("\n" + "=" * 50)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 50)
    logger.info(f"Test Loss: {results['loss']:.4f}")
    logger.info(
        f"Test Accuracy: {results['accuracy']:.4f} ({100 * results['accuracy']:.2f}%)"
    )
    logger.info(
        f"Precision: {results['precision']:.4f} ({100 * results['precision']:.2f}%)"
    )
    logger.info(f"Recall: {results['recall']:.4f} ({100 * results['recall']:.2f}%)")
    logger.info(
        f"F1-Score: {results['f1_score']:.4f} ({100 * results['f1_score']:.2f}%)"
    )
    logger.info(
        f"Specificity: {results['specificity']:.4f} ({100 * results['specificity']:.2f}%)"
    )
    logger.info(f"Total Predictions: {results['total_predictions']:,}")
    logger.info(f"True Positives: {results['true_positives']:,}")
    logger.info(f"False Positives: {results['false_positives']:,}")
    logger.info(f"True Negatives: {results['true_negatives']:,}")
    logger.info(f"False Negatives: {results['false_negatives']:,}")

    # Show some per-file metrics
    logger.info("\n" + "-" * 50)
    logger.info("PER-FILE METRICS (first 10 files)")
    logger.info("-" * 50)

    sorted_files = sorted(
        results["file_metrics"].items(), key=lambda x: x[1]["accuracy"], reverse=True
    )

    for i, (filename, metrics) in enumerate(sorted_files[:10]):
        logger.info(
            f"{i+1:2d}. {filename}: {100 * metrics['accuracy']:.2f}% "
            f"({metrics['correct']}/{metrics['total']})"
        )

    # Save results if output file specified
    if args.output_file:
        import json
        from datetime import datetime

        # Prepare results for saving
        save_results = {
            "timestamp": datetime.now().isoformat(),
            "model_path": args.model_path,
            "data_dir": args.data_dir,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "device": str(device),
            "metrics": {
                "loss": results["loss"],
                "accuracy": results["accuracy"],
                "precision": results["precision"],
                "recall": results["recall"],
                "f1_score": results["f1_score"],
                "specificity": results["specificity"],
                "true_positives": results["true_positives"],
                "false_positives": results["false_positives"],
                "true_negatives": results["true_negatives"],
                "false_negatives": results["false_negatives"],
                "total_predictions": results["total_predictions"],
            },
            "file_metrics": results["file_metrics"],
        }

        # Save to file
        with open(args.output_file, "w") as f:
            json.dump(save_results, f, indent=2)

        logger.info(f"\nResults saved to: {args.output_file}")

    logger.info("\nEvaluation completed!")


if __name__ == "__main__":
    main()
