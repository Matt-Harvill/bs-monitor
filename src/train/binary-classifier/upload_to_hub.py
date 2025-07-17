#!/usr/bin/env python3
"""
Script to upload a trained HuBERT-based bowel sound detection model to Hugging Face Hub.
"""

import argparse
import json
import logging
import os

from huggingface_hub import HfApi, create_repo, upload_folder

from src.train.train import setup_logging


def create_model_card(
    model_name: str, model_path: str, metrics: dict | None = None
) -> str:
    """Create a model card for the Hugging Face Hub"""

    # Format metrics if provided
    metrics_text = ""
    if metrics:
        metrics_text = f"""
## Model Performance

- **Test Accuracy**: {metrics.get('accuracy', 'N/A'):.4f} ({100 * metrics.get('accuracy', 0):.2f}%)
- **Precision**: {metrics.get('precision', 'N/A'):.4f} ({100 * metrics.get('precision', 0):.2f}%)
- **Recall**: {metrics.get('recall', 'N/A'):.4f} ({100 * metrics.get('recall', 0):.2f}%)
- **F1-Score**: {metrics.get('f1_score', 'N/A'):.4f} ({100 * metrics.get('f1_score', 0):.2f}%)
- **Specificity**: {metrics.get('specificity', 'N/A'):.4f} ({100 * metrics.get('specificity', 0):.2f}%)

### Confusion Matrix
- True Positives: {metrics.get('true_positives', 'N/A'):,}
- False Positives: {metrics.get('false_positives', 'N/A'):,}
- True Negatives: {metrics.get('true_negatives', 'N/A'):,}
- False Negatives: {metrics.get('false_negatives', 'N/A'):,}
"""

    model_card = f"""---
language:
- en
tags:
- audio
- audio-classification
- bowel-sounds
- medical
- healthcare
- hubert
license: mit
datasets:
- robertnowak/bowel-sounds
metrics:
- accuracy
- precision
- recall
- f1
---

# {model_name}

This model is trained for bowel sound detection in audio recordings using HuBERT (Hidden-Unit BERT). It can classify audio frames as either containing bowel sounds (class 1) or not containing bowel sounds (class 0).

## Model Details

- **Model Type**: HuBERT for Audio Frame Classification
- **Task**: Binary classification of audio frames for bowel sound detection
- **Input**: Audio waveforms (2-second segments at 16kHz)
- **Output**: Frame-level predictions (49.5 Hz frame rate)
- **Classes**: 0 (no bowel sound), 1 (bowel sound present)

## Training Details

- **Base Model**: Fine-tuned on pre-trained HuBERT
- **Dataset**: Bowel Sounds Dataset from Kaggle
- **Training Data**: 2-second audio segments with frame-level annotations
- **Evaluation**: Frame-level accuracy on test set

## Usage

```python
from src.train.hubert_for_audio_frame_classification import HubertForAudioFrameClassification
import torch
import librosa
import numpy as np

# Load model
model = HubertForAudioFrameClassification.from_pretrained("{model_name}")

# Load and preprocess audio
audio, sr = librosa.load("audio_file.wav", sr=16000)

# Ensure audio is 2 seconds (32000 samples at 16kHz)
if len(audio) < 32000:
    # Pad with zeros if shorter
    audio = np.pad(audio, (0, 32000 - len(audio)), 'constant')
else:
    # Truncate if longer
    audio = audio[:32000]

# Convert to tensor and add batch dimension
audio_tensor = torch.FloatTensor(audio).unsqueeze(0)  # Shape: (1, 32000)

# Get predictions
with torch.no_grad():
    outputs = model(audio_tensor)
    predictions = torch.argmax(outputs.logits, dim=-1)

# predictions contains frame-level classifications
# 0 = no bowel sound, 1 = bowel sound present
# Shape: (1, 99) - 99 frames for 2 seconds at 49.5 Hz
```

## Training and Evaluation

This repository includes training and evaluation scripts:

- **`src/train/train.py`**: Training script with the `BowelSoundDataset` class for loading and preprocessing bowel sound data
- **`src/train/evaluate.py`**: Evaluation script for testing model performance on the test set

To train your own model:

```bash
# Train a new model
uv run src/train/train.py --data_dir /path/to/data --output_dir ./my_model

# Evaluate a trained model
uv run src/train/evaluate.py --model_path ./my_model --data_dir /path/to/data
```

The `BowelSoundDataset` class handles:
- Loading audio files and CSV annotations
- Preprocessing 2-second audio segments
- Converting time annotations to frame-level labels
- Caching processed data for faster training

## Dataset

This model was trained on the [Bowel Sounds Dataset](https://www.kaggle.com/datasets/robertnowak/bowel-sounds) which contains audio recordings with manual annotations of bowel sound events.
I converted the dataset to a format that can be used with the `BowelSoundDataset` class (any type of bowel sound = class 1, no bowel sound = class 0).

## Limitations

- Trained on 2-second audio segments
- May not generalize to significantly different recording conditions
- Requires 16kHz audio input
- Frame-level predictions at 49.5 Hz rate

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{{{model_name.lower().replace('-', '_')},
  author = {{Matthew Harvill}},
  title = {{Bowel Sound Detection Model}},
  year = {{2025}},
  publisher = {{Hugging Face}},
  url = {{https://huggingface.co/{model_name}}}
}}
"""

    if metrics_text:
        model_card += metrics_text

    return model_card


def create_readme(model_name: str, model_path: str) -> str:
    """Create a README file for the repository"""

    return f"""# {model_name}

HuBERT-based bowel sound detection model trained on the Bowel Sounds Dataset.

## Quick Start

```python
from src.train.hubert_for_audio_frame_classification import HubertForAudioFrameClassification
import torch
import librosa
import numpy as np

# Load model
model = HubertForAudioFrameClassification.from_pretrained("{model_name}")

# Load and preprocess audio
audio, sr = librosa.load("audio_file.wav", sr=16000)

# Ensure audio is 2 seconds (32000 samples at 16kHz)
if len(audio) < 32000:
    # Pad with zeros if shorter
    audio = np.pad(audio, (0, 32000 - len(audio)), 'constant')
else:
    # Truncate if longer
    audio = audio[:32000]

# Convert to tensor and add batch dimension
audio_tensor = torch.FloatTensor(audio).unsqueeze(0)  # Shape: (1, 32000)

# Get predictions
with torch.no_grad():
    outputs = model(audio_tensor)
    predictions = torch.argmax(outputs.logits, dim=-1)
```

## Model Information

- **Architecture**: HuBERT (Hidden-Unit BERT) for Audio Frame Classification
- **Input**: 2-second audio segments at 16kHz
- **Output**: Frame-level predictions (49.5 Hz)
- **Classes**: 0 (no bowel sound), 1 (bowel sound)

For more details, see the model card.
"""


def prepare_model_for_upload(
    model_path: str, logger: logging.Logger | None = None
) -> str:
    """
    Prepare model directory for upload by copying necessary files

    Args:
        model_path: Path to the trained model directory
        logger: Logger instance

    Returns:
        Path to the prepared upload directory
    """

    if logger is None:
        logger = logging.getLogger(__name__)

    # Create a temporary upload directory
    import shutil
    import tempfile

    upload_dir = tempfile.mkdtemp(prefix="hf_upload_")
    logger.info(f"Created upload directory: {upload_dir}")

    # Copy all model files
    logger.info("Copying model files...")
    for item in os.listdir(model_path):
        src = os.path.join(model_path, item)
        dst = os.path.join(upload_dir, item)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
        elif os.path.isdir(src):
            shutil.copytree(src, dst)

    # Always copy the custom HuBERT model file
    logger.info("Copying custom HuBERT model file...")
    custom_model_file = "src/train/hubert_for_audio_frame_classification.py"
    if os.path.exists(custom_model_file):
        # Create src/train directory in upload dir
        os.makedirs(os.path.join(upload_dir, "src", "train"), exist_ok=True)
        shutil.copy2(custom_model_file, os.path.join(upload_dir, "src", "train"))
        logger.info("✓ Custom HuBERT model file copied")
    else:
        logger.warning(f"Custom model file not found: {custom_model_file}")

    # Copy training and evaluation scripts
    logger.info("Copying training and evaluation scripts...")
    training_files = ["src/train/train.py", "src/train/evaluate.py"]

    for file_path in training_files:
        if os.path.exists(file_path):
            # Create src/train directory in upload dir if not already created
            os.makedirs(os.path.join(upload_dir, "src", "train"), exist_ok=True)
            shutil.copy2(file_path, os.path.join(upload_dir, "src", "train"))
            logger.info(f"✓ {file_path} copied")
        else:
            logger.warning(f"Training file not found: {file_path}")

    return upload_dir


def upload_model_to_hub(
    model_path: str,
    repo_name: str,
    token: str,
    private: bool = False,
    metrics_file: str | None = None,
    logger: logging.Logger | None = None,
) -> str:
    """
    Upload a trained HuBERT model to Hugging Face Hub

    Args:
        model_path: Path to the trained model directory
        repo_name: Name for the repository on HF Hub (e.g., "username/model-name")
        token: Hugging Face API token
        private: Whether to make the repository private
        metrics_file: Optional path to evaluation metrics JSON file
        logger: Logger instance

    Returns:
        URL of the uploaded model
    """

    if logger is None:
        logger = logging.getLogger(__name__)

    # Validate model path
    if not os.path.exists(model_path):
        raise ValueError(f"Model path does not exist: {model_path}")

    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        raise ValueError(f"Config file not found at: {config_path}")

    # Validate that this is a HuBERT model
    with open(config_path) as f:
        config = json.load(f)
        model_type = config.get("model_type", "").lower()

    if "hubert" not in model_type:
        raise ValueError(
            f"This script is for HuBERT models only. Found model type: {model_type}"
        )

    logger.info(f"Validated HuBERT model: {model_type}")

    # Load metrics if provided
    metrics = None
    if metrics_file and os.path.exists(metrics_file):
        logger.info(f"Loading metrics from: {metrics_file}")
        with open(metrics_file) as f:
            metrics_data = json.load(f)
            metrics = metrics_data.get("metrics", {})

    # Initialize HF API
    api = HfApi(token=token)

    # Check if user is authenticated
    try:
        user = api.whoami()
        logger.info(f"Authenticated as: {user}")
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        raise

    # Create repository
    logger.info(f"Creating repository: {repo_name}")
    try:
        create_repo(repo_id=repo_name, token=token, private=private, exist_ok=True)
        logger.info(f"✓ Repository created/verified: {repo_name}")
    except Exception as e:
        logger.error(f"Failed to create repository: {e}")
        raise

    # Prepare model for upload
    logger.info("Preparing model for upload...")
    upload_dir = prepare_model_for_upload(model_path, logger)

    # Create model card
    logger.info("Creating model card...")
    model_card = create_model_card(repo_name, model_path, metrics)

    # Upload model files
    logger.info("Uploading model files...")
    try:
        # Upload the prepared model directory
        upload_folder(
            folder_path=upload_dir,
            repo_id=repo_name,
            token=token,
            commit_message="Upload trained HuBERT bowel sound detection model",
        )
        logger.info("✓ Model files uploaded")

        # Upload model card
        api.upload_file(
            path_or_fileobj=model_card.encode(),
            path_in_repo="README.md",
            repo_id=repo_name,
            token=token,
            commit_message="Add model card",
        )
        logger.info("✓ Model card uploaded")

        # Clean up temporary directory
        import shutil

        shutil.rmtree(upload_dir)
        logger.info("✓ Cleaned up temporary files")

    except Exception as e:
        logger.error(f"Failed to upload files: {e}")
        # Clean up on error
        try:
            import shutil

            shutil.rmtree(upload_dir)
        except Exception:
            pass
        raise

    # Get the model URL
    model_url = f"https://huggingface.co/{repo_name}"
    logger.info(f"✓ Model successfully uploaded to: {model_url}")

    return model_url


def main():
    parser = argparse.ArgumentParser(
        description="Upload trained HuBERT-based bowel sound detection model to Hugging Face Hub"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained HuBERT model directory (containing config.json, model.safetensors, etc.)",
    )
    parser.add_argument(
        "--repo_name",
        type=str,
        required=True,
        help="Repository name on HF Hub (e.g., 'username/bowel-sound-detector')",
    )
    parser.add_argument(
        "--token",
        type=str,
        required=True,
        help="Hugging Face API token (get from https://huggingface.co/settings/tokens)",
    )
    parser.add_argument(
        "--private", action="store_true", help="Make the repository private"
    )
    parser.add_argument(
        "--metrics_file",
        type=str,
        default=None,
        help="Path to evaluation metrics JSON file (optional)",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.log_level)

    logger.info("Starting HuBERT model upload to Hugging Face Hub...")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Repository: {args.repo_name}")
    logger.info(f"Private: {args.private}")

    try:
        # Upload model
        model_url = upload_model_to_hub(
            model_path=args.model_path,
            repo_name=args.repo_name,
            token=args.token,
            private=args.private,
            metrics_file=args.metrics_file,
            logger=logger,
        )

        logger.info("\n" + "=" * 60)
        logger.info("🎉 UPLOAD SUCCESSFUL!")
        logger.info("=" * 60)
        logger.info(f"Model URL: {model_url}")
        logger.info(f"Repository: {args.repo_name}")
        logger.info("\nYou can now use your model with:")
        logger.info(
            "from src.train.hubert_for_audio_frame_classification import HubertForAudioFrameClassification"
        )
        logger.info(
            f"model = HubertForAudioFrameClassification.from_pretrained('{args.repo_name}')"
        )

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
