#!/usr/bin/env python3
"""
Script to check .wav file lengths for entries where end time > 2.0s.
This helps identify if the .wav files are actually longer than 2.0s or if there's a data inconsistency.
"""

from pathlib import Path

import librosa
import pandas as pd


def get_wav_duration(wav_file_path):
    """Get the duration of a .wav file in seconds using librosa."""
    try:
        y, sr = librosa.load(str(wav_file_path), sr=None)
        duration = len(y) / sr
        return duration
    except Exception as e:
        print(f"Error reading {wav_file_path}: {e}")
        return None


def main():
    # Get the data directory path
    data_dir = Path("data")

    # Find all CSV files in the data directory (excluding files.csv which seems to be metadata)
    csv_files = list(data_dir.glob("*.csv"))
    csv_files = [f for f in csv_files if f.name != "files.csv"]

    print(f"Found {len(csv_files)} CSV files to process")

    # List of dataframes to concatenate
    dfs = []

    # Read each CSV file
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            # Add a column to track the source file
            df["source_file"] = csv_file.name
            dfs.append(df)
            print(f"Processed: {csv_file.name} ({len(df)} rows)")
        except Exception as e:
            print(f"Error reading {csv_file.name}: {e}")

    if not dfs:
        print("No CSV files could be processed!")
        return

    # Concatenate all dataframes
    print("\nConcatenating all CSV files...")
    combined_df = pd.concat(dfs, ignore_index=True)

    # Find entries where end > 2.0
    end_gt_2_mask = combined_df["end"] > 2.0
    end_gt_2_entries = combined_df[end_gt_2_mask]
    assert isinstance(end_gt_2_entries, pd.DataFrame)

    print(f"\nFound {len(end_gt_2_entries)} entries with end > 2.0s")

    if len(end_gt_2_entries) == 0:
        print("No entries found with end > 2.0s")
        return

    print("\n" + "=" * 80)
    print("ANALYSIS OF ENTRIES WITH END > 2.0s")
    print("=" * 80)

    # Group by source file to check .wav lengths
    for source_file in end_gt_2_entries["source_file"].unique():
        print(f"\nSource file: {source_file}")

        # Get the corresponding .wav file
        wav_file = data_dir / source_file.replace(".csv", ".wav")

        if wav_file.exists():
            wav_duration = get_wav_duration(wav_file)
            print(f"  .wav file: {wav_file.name}")
            if wav_duration is not None:
                print(f"  .wav duration: {wav_duration:.3f}s")

                # Get entries from this file with end > 2.0
                file_entries = end_gt_2_entries[
                    end_gt_2_entries["source_file"] == source_file
                ]

                print("  Entries with end > 2.0s:")
                for _, row in file_entries.iterrows():
                    start_val = row["start"]
                    end_val = row["end"]
                    fmin_val = row["fmin"]
                    fmax_val = row["fmax"]
                    category_val = row.get("category", "N/A")

                    print(
                        f"    Start: {start_val:.3f}s, End: {end_val:.3f}s, Duration: {end_val - start_val:.3f}s"
                    )
                    print(
                        f"    Frequency range: {fmin_val:.1f}Hz - {fmax_val:.1f}Hz, Category: {category_val}"
                    )

                    # Check if this is concerning
                    if wav_duration <= 2.0:
                        print(
                            f"    ⚠️  CONCERNING: .wav file is only {wav_duration:.3f}s but end time is {end_val:.3f}s"
                        )
                    else:
                        print(
                            f"    ✅ OK: .wav file is {wav_duration:.3f}s, so end time of {end_val:.3f}s is valid"
                        )
                    print()
            else:
                print("  ❌ Could not read .wav file duration")
        else:
            print(f"  ❌ .wav file not found: {wav_file}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    concerning_files = []
    ok_files = []

    for source_file in end_gt_2_entries["source_file"].unique():
        wav_file = data_dir / source_file.replace(".csv", ".wav")
        if wav_file.exists():
            wav_duration = get_wav_duration(wav_file)
            if wav_duration is not None:
                if wav_duration <= 2.0:
                    concerning_files.append((source_file, wav_duration))
                else:
                    ok_files.append((source_file, wav_duration))

    print(
        f"Files with end > 2.0s and .wav duration <= 2.0s (CONCERNING): {len(concerning_files)}"
    )
    for file_name, duration in concerning_files:
        print(f"  - {file_name}: .wav duration = {duration:.3f}s")

    print(f"\nFiles with end > 2.0s and .wav duration > 2.0s (OK): {len(ok_files)}")
    for file_name, duration in ok_files:
        print(f"  - {file_name}: .wav duration = {duration:.3f}s")


if __name__ == "__main__":
    main()
