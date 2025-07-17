# Bowel Sound Monitor

This repo is for exploring and training models on bowel sound data for better classification and representation learning.

## Supported Models

The training script supports multiple audio models:

- **HuBERT XLarge** (`facebook/hubert-xlarge-ll60k`) - Default, Largest model
- **Wav2Vec2 Large** (`facebook/wav2vec2-large`) - Medium model
- **Wav2Vec2 Base** (`facebook/wav2vec2-base`) - Smallest model

## Quick Start

```bash
# Install dependencies
uv sync

# Train with default model (HuBERT XLarge)
uv run src/train/train.py

# Train with Wav2Vec2 Large
uv run src/train/train.py --model "facebook/wav2vec2-large"

# Train with Wav2Vec2 Base
uv run src/train/train.py --model "facebook/wav2vec2-base"
```
