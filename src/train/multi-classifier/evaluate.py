"""Evaluate multi-class classifier on test datasets."""

import argparse

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
from transformers import Wav2Vec2ForAudioFrameClassification

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


def evaluate_model(model, loader, device, config):
    """
    Evaluate model and collect predictions for detailed metrics.

    Returns:
        y_true: List of true labels
        y_pred: List of predicted labels
        dataset_info: List of (dataset_name, chunk_idx) tuples
    """
    model.eval()
    y_true = []
    y_pred = []
    dataset_info = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            audio = batch["audio"].to(device)
            labels = batch["labels"].to(device)
            dataset_names = batch["dataset"]
            chunk_idxs = batch["chunk_idx"]

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

            # Get predictions
            predictions = torch.argmax(logits, dim=-1)

            # Flatten and collect
            batch_size = labels.size(0)
            for i in range(batch_size):
                y_true.extend(labels[i].cpu().numpy())
                y_pred.extend(predictions[i].cpu().numpy())
                # Add dataset info for each frame
                frames = labels[i].size(0)
                dataset_info.extend([(dataset_names[i], chunk_idxs[i].item())] * frames)

    return y_true, y_pred, dataset_info


def compute_metrics_per_dataset(y_true, y_pred, dataset_info, config):
    """Compute metrics separately for each dataset."""
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Get unique datasets
    dataset_names = list({info[0] for info in dataset_info})

    results = {}

    for dataset_name in dataset_names:
        # Filter predictions for this dataset
        mask = np.array([info[0] == dataset_name for info in dataset_info])
        dataset_y_true = y_true[mask]
        dataset_y_pred = y_pred[mask]

        # Compute metrics
        accuracy = accuracy_score(dataset_y_true, dataset_y_pred)

        # Get classification report
        report = classification_report(
            dataset_y_true,
            dataset_y_pred,
            labels=list(config.idx_to_class.keys()),
            target_names=list(config.idx_to_class.values()),
            output_dict=True,
            zero_division=0,
        )

        # Get confusion matrix
        cm = confusion_matrix(
            dataset_y_true, dataset_y_pred, labels=list(config.idx_to_class.keys())
        )

        results[dataset_name] = {
            "accuracy": accuracy,
            "report": report,
            "confusion_matrix": cm,
            "n_samples": len(dataset_y_true),
        }

    return results


def print_results(results, config, split_name):
    """Print evaluation results in a formatted way."""
    print(f"\n{'='*80}")
    print(f"EVALUATION RESULTS - {split_name.upper()} SPLIT")
    print(f"{'='*80}\n")

    # Overall metrics across all datasets
    all_accuracies = []
    all_n_samples = []

    for _dataset_name, metrics in results.items():
        all_accuracies.append(metrics["accuracy"])
        all_n_samples.append(metrics["n_samples"])

    # Weighted average accuracy
    weighted_acc = np.average(all_accuracies, weights=all_n_samples)
    print(f"Overall Weighted Accuracy: {weighted_acc:.4f}")
    print(f"Total Samples: {sum(all_n_samples):,}")

    # Per-dataset results
    for dataset_name, metrics in sorted(results.items()):
        print(f"\n{'-'*60}")
        print(f"Dataset: {dataset_name}")
        print(f"Number of frames: {metrics['n_samples']:,}")
        print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
        print("\nPer-Class Metrics:")
        print(
            f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}"
        )
        print("-" * 60)

        report = metrics["report"]
        for class_name in config.idx_to_class.values():
            if class_name in report:
                class_metrics = report[class_name]
                print(
                    f"{class_name:<15} "
                    f"{class_metrics['precision']:<12.4f} "
                    f"{class_metrics['recall']:<12.4f} "
                    f"{class_metrics['f1-score']:<12.4f} "
                    f"{int(class_metrics['support']):<12,}"
                )

        # Print confusion matrix
        print("\nConfusion Matrix:")
        print(f"{'True\\Pred':<15}", end="")
        for class_name in config.idx_to_class.values():
            print(f"{class_name:<12}", end="")
        print()

        cm = metrics["confusion_matrix"]
        for i, true_class in enumerate(config.idx_to_class.values()):
            print(f"{true_class:<15}", end="")
            for j in range(len(config.idx_to_class)):
                print(f"{cm[i, j]:<12,}", end="")
            print()

    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate multi-class classifier")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file (optional, will use checkpoint config)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "val", "test", "all"],
        help="Which split(s) to evaluate on",
    )
    parser.add_argument(
        "--batch_size", type=int, help="Override batch size for evaluation"
    )

    args = parser.parse_args()

    # Setup logging
    logger = get_logger()

    # Load checkpoint
    logger.info(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    # Initialize config from checkpoint
    if "config" in checkpoint:
        checkpoint_config = checkpoint["config"]
        config = MultiClassifierConfig()
        # Update config with checkpoint values
        for key, value in checkpoint_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
    else:
        config = MultiClassifierConfig()

    # Override batch size if specified
    if args.batch_size:
        config.batch_size = args.batch_size

    logger.info(f"Using config: {config}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

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

    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Create data splitter
    data_splitter = DataSplitter(config.data_dir)

    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(config, data_splitter)

    # Determine which splits to evaluate
    splits_to_eval = []
    if args.split == "all":
        splits_to_eval = [
            ("train", train_loader),
            ("val", val_loader),
            ("test", test_loader),
        ]
    elif args.split == "train":
        splits_to_eval = [("train", train_loader)]
    elif args.split == "val":
        splits_to_eval = [("val", val_loader)]
    elif args.split == "test":
        splits_to_eval = [("test", test_loader)]

    # Evaluate on each split
    for split_name, loader in splits_to_eval:
        logger.info(f"\nEvaluating on {split_name} split...")

        # Get predictions
        y_true, y_pred, dataset_info = evaluate_model(model, loader, device, config)

        # Compute metrics per dataset
        results = compute_metrics_per_dataset(y_true, y_pred, dataset_info, config)

        # Print results
        print_results(results, config, split_name)

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
