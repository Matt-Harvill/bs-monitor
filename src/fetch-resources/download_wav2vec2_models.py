#!/usr/bin/env python3
"""Download wav2vec2 base and large models from Hugging Face."""

import os

from transformers import Wav2Vec2Model

models = ["facebook/wav2vec2-base", "facebook/wav2vec2-large"]
os.makedirs("models", exist_ok=True)

for model_name in models:
    print(f"Downloading {model_name}...")
    model = Wav2Vec2Model.from_pretrained(model_name)
    # model.save_pretrained(f"models/{model_name.replace('/', '_')}")
    print(f"✅ Downloaded {model_name}")
