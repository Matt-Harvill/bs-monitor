#!/usr/bin/env python3
"""Test script to determine HuBERT frame rates for different audio chunk sizes."""

from pathlib import Path

import librosa
import torch
from transformers import HubertModel


def test_hubert_frame_rates():
    """Test HuBERT output shapes for different audio durations."""

    model_name = "facebook/hubert-base-ls960"
    model = HubertModel.from_pretrained(model_name)
    model.eval()

    sample_rate = 16000

    durations = [1, 2, 4, 8]

    print("Testing HuBERT frame rates for different audio durations:")
    print("-" * 60)

    for duration in durations:
        num_samples = duration * sample_rate

        dummy_audio = torch.randn(1, num_samples)

        with torch.no_grad():
            outputs = model(dummy_audio)
            hidden_states = outputs.last_hidden_state

        num_frames = hidden_states.shape[1]
        frames_per_second = num_frames / duration

        print(f"Duration: {duration}s")
        print(f"  Input samples: {num_samples}")
        print(f"  Output frames: {num_frames}")
        print(f"  Frames per second: {frames_per_second:.2f}")
        print(f"  Frame duration: {1000/frames_per_second:.2f}ms")
        print()

    data_dir = Path("/home/matthew/Code/bs-monitor/data")

    print("\nTesting with actual audio files:")
    print("-" * 60)

    for dataset in ["AS_1-dataset", "23M74M-dataset"]:
        dataset_path = data_dir / dataset
        if not dataset_path.exists():
            continue

        print(f"\nDataset: {dataset}")

        for duration in durations:
            chunk_dir = dataset_path / f"{duration}s-chunks"
            if not chunk_dir.exists():
                continue

            wav_files = list(chunk_dir.glob("*.wav"))
            if not wav_files:
                continue

            sample_file = wav_files[0]
            waveform, sr = librosa.load(sample_file, sr=sample_rate, mono=True)

            waveform = torch.tensor(waveform).unsqueeze(0)

            with torch.no_grad():
                outputs = model(waveform)
                hidden_states = outputs.last_hidden_state

            num_frames = hidden_states.shape[1]
            actual_duration = waveform.shape[1] / sample_rate
            frames_per_second = num_frames / actual_duration

            print(f"  {duration}s chunks:")
            print(f"    Sample file: {sample_file.name}")
            print(f"    Actual duration: {actual_duration:.3f}s")
            print(f"    Output frames: {num_frames}")
            print(f"    Frames per second: {frames_per_second:.2f}")


if __name__ == "__main__":
    test_hubert_frame_rates()
