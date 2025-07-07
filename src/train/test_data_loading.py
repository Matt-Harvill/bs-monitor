#!/usr/bin/env python3
"""
Test script to verify data loading functionality
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from train import BowelSoundDataset, create_dataloaders


def test_data_loading():
    """Test the data loading functionality"""
    print("Testing data loading...")

    # Test dataset creation (first run - will process and cache)
    print("\n=== First run (processing and caching) ===")
    try:
        train_dataset = BowelSoundDataset("data", split="train", max_length=2.0)
        print(f"✓ Train dataset created successfully with {len(train_dataset)} samples")

        # Test getting a sample
        sample = train_dataset[0]
        print("✓ Sample loaded successfully")
        print(f"  - Audio shape: {sample['audio'].shape}")
        print(f"  - Labels shape: {sample['labels'].shape}")
        print(f"  - Filename: {sample['filename']}")
        print(f"  - Chunk index: {sample['chunk_idx']}")

        # Check label distribution
        labels = sample["labels"].numpy()
        num_zeros = (labels == 0).sum()
        num_ones = (labels == 1).sum()
        print(f"  - Label distribution: {num_zeros} zeros, {num_ones} ones")
        print(f"  - Bowel sound ratio: {100 * num_ones / len(labels):.2f}%")

        # Show all labels
        print(f"  - All labels: {labels.tolist()}")

        # Show original CSV data for this file
        filename = sample["filename"]
        csv_file = filename.replace(".wav", ".csv")
        csv_path = f"data/{csv_file}"

        try:
            import pandas as pd

            annotations = pd.read_csv(csv_path)
            print(f"\n📋 Original CSV annotations for {csv_file}:")
            print(annotations.to_string(index=False))

            # Show which annotations fall within this 2-second segment
            chunk_idx = sample["chunk_idx"]
            chunk_start_time = chunk_idx * 2.0  # 2s segments
            chunk_end_time = (chunk_idx + 1) * 2.0

            print(
                f"\n🎯 2-second segment {chunk_idx} covers time range: {chunk_start_time:.1f}s - {chunk_end_time:.1f}s"
            )

            chunk_annotations = []
            for _, row in annotations.iterrows():
                if pd.isna(row["start"]) or pd.isna(row["end"]):
                    continue
                # Check if annotation overlaps with this chunk
                if row["start"] < chunk_end_time and row["end"] > chunk_start_time:
                    chunk_annotations.append(row)

            if chunk_annotations:
                print("Annotations that overlap with this 2s segment:")
                for ann in chunk_annotations:
                    print(f"  - {ann['start']:.3f}s to {ann['end']:.3f}s")
            else:
                print("No annotations overlap with this 2s segment")

        except Exception as e:
            print(f"Could not load CSV file: {e}")

        # Find all bowel sound regions if any
        if num_ones > 0:
            bowel_regions = []
            in_bowel = False
            start_frame = 0
            for frame_idx, label in enumerate(labels):
                if label == 1 and not in_bowel:
                    start_frame = frame_idx
                    in_bowel = True
                elif label == 0 and in_bowel:
                    bowel_regions.append((start_frame, frame_idx - 1))
                    in_bowel = False
            if in_bowel:
                bowel_regions.append((start_frame, len(labels) - 1))

            print("\n🔍 Bowel sound regions in this chunk:")
            for start, end in bowel_regions:
                start_time = chunk_start_time + start / 49.0
                end_time = chunk_start_time + end / 49.0
                print(f"  - Frames {start}-{end}: {start_time:.3f}s - {end_time:.3f}s")

    except Exception as e:
        print(f"✗ Error creating train dataset: {e}")
        return False

    try:
        test_dataset = BowelSoundDataset("data", split="test", max_length=2.0)
        print(f"✓ Test dataset created successfully with {len(test_dataset)} samples")
    except Exception as e:
        print(f"✗ Error creating test dataset: {e}")
        return False

    # Test dataloader creation
    try:
        train_loader, test_loader = create_dataloaders(
            "data", batch_size=2, num_workers=0
        )
        print("✓ Dataloaders created successfully")
        print(f"  - Train batches: {len(train_loader)}")
        print(f"  - Test batches: {len(test_loader)}")

        # Test getting a batch
        batch = next(iter(train_loader))
        print("✓ Batch loaded successfully")
        print(f"  - Batch audio shape: {batch['audio'].shape}")
        print(f"  - Batch labels shape: {batch['labels'].shape}")

    except Exception as e:
        print(f"✗ Error creating dataloaders: {e}")
        return False

    # Test caching functionality (second run - should load from cache)
    print("\n=== Second run (loading from cache) ===")
    try:
        train_dataset_cached = BowelSoundDataset("data", split="train", max_length=2.0)
        print(
            f"✓ Train dataset loaded from cache with {len(train_dataset_cached)} samples"
        )

        # Verify data is the same
        sample_cached = train_dataset_cached[0]
        if torch.equal(sample["audio"], sample_cached["audio"]) and torch.equal(
            sample["labels"], sample_cached["labels"]
        ):
            print("✓ Cached data matches original data")
        else:
            print("✗ Cached data does not match original data")
            return False

    except Exception as e:
        print(f"✗ Error loading from cache: {e}")
        return False

    print("\n✓ All tests passed! Data loading and caching is working correctly.")
    return True


if __name__ == "__main__":
    test_data_loading()
