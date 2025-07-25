#!/usr/bin/env python3
"""Generate multi-class labels for HuBERT frame classification."""

import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

# HuBERT frame rates based on test results
FRAME_RATES = {1: 49, 2: 99, 4: 199, 8: 399}

# Label mapping
LABEL_MAPPING = {
    "sb": "single",
    "mb": "multiple",
    "h": "harmonic",
    "n": None,  # ignore noise labels
    "v": None,  # ignore voice labels
}

# Class priorities (higher number = higher priority)
CLASS_PRIORITIES = {"harmonic": 3, "multiple": 2, "single": 1, "none": 0}


def parse_clean_file(clean_file_path: Path) -> list[tuple[float, float, str]]:
    """Parse the clean annotation file to extract labels."""
    labels = []

    with open(clean_file_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Parse format: "0.000000	17.019255	n"
            parts = line.split("\t")
            if len(parts) != 3:
                continue

            # Extract start time, end time, and label
            try:
                start_time = float(parts[0])
                end_time = float(parts[1])
                label = parts[2]
            except ValueError:
                continue

            # Convert label using mapping
            mapped_label = LABEL_MAPPING.get(label)
            if mapped_label is not None:  # Skip noise labels and voice labels
                labels.append((start_time, end_time, mapped_label))

    return labels


def generate_frame_labels(
    labels: list[tuple[float, float, str]], duration: int, num_frames: int
) -> np.ndarray:
    """Generate frame-level labels for a given duration."""
    frame_labels = ["none"] * num_frames
    frame_duration = duration / num_frames

    for start_time, end_time, label in labels:
        # Calculate frame indices
        start_frame = int(start_time / frame_duration)
        end_frame = int(end_time / frame_duration)

        # Ensure frames are within bounds
        start_frame = max(0, min(start_frame, num_frames - 1))
        end_frame = max(0, min(end_frame, num_frames - 1))

        # Apply label with priority system
        for frame_idx in range(start_frame, end_frame + 1):
            current_label = frame_labels[frame_idx]
            current_priority = CLASS_PRIORITIES[current_label]
            new_priority = CLASS_PRIORITIES[label]

            if new_priority > current_priority:
                frame_labels[frame_idx] = label

    return np.array(frame_labels)


def generate_chunk_labels(
    dataset_name: str, dataset_path: Path, clean_file_path: Path, output_dir: Path
) -> None:
    """Generate labels for all chunks in a dataset."""
    print(f"Processing {dataset_name}...")

    # Parse clean file
    labels = parse_clean_file(clean_file_path)
    print(f"Found {len(labels)} label intervals")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each duration
    for duration in [1, 2, 4, 8]:
        chunk_dir = dataset_path / f"{duration}s-chunks"
        if not chunk_dir.exists():
            print(f"Skipping {duration}s chunks - directory not found")
            continue

        wav_files = sorted(chunk_dir.glob("*.wav"))
        if not wav_files:
            print(f"No wav files found in {chunk_dir}")
            continue

        num_frames = FRAME_RATES[duration]
        chunk_labels = {}

        print(f"Generating labels for {duration}s chunks ({num_frames} frames each)...")

        for wav_file in tqdm(wav_files, desc=f"{duration}s chunks"):
            # Extract chunk index from filename
            # Handle files like "AS_1_chunk_0001.wav" or "AS_1_chunk_0001_padded.wav"
            stem_parts = wav_file.stem.split("_")
            if stem_parts[-1] == "padded":
                chunk_idx = int(stem_parts[-2])
            else:
                chunk_idx = int(stem_parts[-1])

            # Calculate time offset for this chunk
            chunk_start_time = chunk_idx * duration
            chunk_end_time = chunk_start_time + duration

            # Filter labels that overlap with this chunk
            chunk_label_list = []
            for start_time, end_time, label in labels:
                # Check if label overlaps with chunk
                if start_time < chunk_end_time and end_time > chunk_start_time:
                    # Adjust times relative to chunk start
                    adjusted_start = max(0, start_time - chunk_start_time)
                    adjusted_end = min(duration, end_time - chunk_start_time)
                    chunk_label_list.append((adjusted_start, adjusted_end, label))

            # Generate frame labels for this chunk
            frame_labels = generate_frame_labels(chunk_label_list, duration, num_frames)
            chunk_labels[chunk_idx] = frame_labels

        # Save labels
        output_file = output_dir / f"{dataset_name}_{duration}s_labels.pkl"
        with open(output_file, "wb") as f:
            pickle.dump(chunk_labels, f)

        print(f"Saved {len(chunk_labels)} chunk labels to {output_file}")

        # Print label distribution
        all_labels = np.concatenate(list(chunk_labels.values()))
        unique, counts = np.unique(all_labels, return_counts=True)
        print(f"Label distribution for {duration}s chunks:")
        for label, count in zip(unique, counts, strict=False):
            percentage = count / len(all_labels) * 100
            print(f"  {label}: {count} ({percentage:.1f}%)")
        print()


def main():
    """Main function to generate labels for all datasets."""
    data_dir = Path("/home/matthew/Code/bs-monitor/data")
    output_dir = Path("/home/matthew/Code/bs-monitor/src/train/multi-classifier/labels")

    # Process AS_1 dataset
    as1_dataset_path = data_dir / "AS_1-dataset"
    as1_clean_file = as1_dataset_path / "AS_1_clean.txt"

    if as1_dataset_path.exists() and as1_clean_file.exists():
        generate_chunk_labels("AS_1", as1_dataset_path, as1_clean_file, output_dir)
    else:
        print("AS_1 dataset or clean file not found")

    # Process 23M74M dataset
    m74m_dataset_path = data_dir / "23M74M-dataset"
    m74m_clean_file = m74m_dataset_path / "23M74M_clean.txt"

    if m74m_dataset_path.exists() and m74m_clean_file.exists():
        generate_chunk_labels("23M74M", m74m_dataset_path, m74m_clean_file, output_dir)
    else:
        print("23M74M dataset or clean file not found")

    print("Label generation complete!")


if __name__ == "__main__":
    main()
