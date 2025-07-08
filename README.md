# Bowel Sound Monitor

This repo is for exploring and training models on bowel sound data for better classification and representation learning.

## Supported Models

The training script supports multiple audio models:

- **Wav2Vec2 Large** (`facebook/wav2vec2-large`) - Default, best performance
- **Wav2Vec2 Base** (`facebook/wav2vec2-base`) - Faster training, smaller model
- **HuBERT XLarge** (`facebook/hubert-xlarge-ll60k`) - Alternative architecture with frame-level classification

## Quick Start

```bash
# Install dependencies
uv sync

# Train with default model (Wav2Vec2 Large)
uv run src/train/train.py

# Train with HuBERT XLarge
uv run src/train/train.py --model "facebook/hubert-xlarge-ll60k"

# Train with Wav2Vec2 Base
uv run src/train/train.py --model "facebook/wav2vec2-base"
```
