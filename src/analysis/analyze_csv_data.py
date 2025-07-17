#!/usr/bin/env python3
"""
Script to analyze CSV files in the data folder.
Concatenates all CSV files and reports statistics on start/end values > 2.0.
"""

from pathlib import Path

import pandas as pd


def main():
    # Get the data directory path
    data_dir = Path("data/bs-dataset")

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

    print(f"Total rows in combined dataset: {len(combined_df)}")
    print(f"Columns: {list(combined_df.columns)}")

    # Display basic info about the data
    print("\nData info:")
    print(combined_df.info())

    print("\nFirst few rows:")
    print(combined_df.head())

    # Analyze categories
    print("\n" + "=" * 50)
    print("CATEGORY ANALYSIS")
    print("=" * 50)

    if "category" in combined_df.columns:
        # Count NaN values in category column
        nan_count = combined_df["category"].isna().sum()
        total_categories = len(combined_df)
        print(f"Total entries: {total_categories}")
        print(
            f"Entries with NaN category: {nan_count} ({nan_count/total_categories*100:.2f}%)"
        )

        # Get unique categories (excluding NaN)
        unique_categories = combined_df["category"].dropna().unique()
        print(f"\nUnique categories found: {len(unique_categories)}")
        print("Categories:", sorted(unique_categories))

        # Count each category
        print("\nCategory counts:")
        category_counts = combined_df["category"].value_counts(dropna=False)
        for category, count in category_counts.items():
            percentage = count / total_categories * 100
            if pd.isna(category):
                print(f"  NaN: {count} ({percentage:.2f}%)")
            else:
                print(f"  '{category}': {count} ({percentage:.2f}%)")
    else:
        print("No 'category' column found in the data")

    # Analyze start and end values
    print("\n" + "=" * 50)
    print("ANALYSIS OF START AND END VALUES")
    print("=" * 50)

    # Count total start and end entries (excluding NaN values)
    total_start_entries = combined_df["start"].notna().sum()
    total_end_entries = combined_df["end"].notna().sum()

    print(f"Total start entries (non-NaN): {total_start_entries}")
    print(f"Total end entries (non-NaN): {total_end_entries}")

    # Count entries where start > 2.0
    start_gt_2 = (combined_df["start"] > 2.0).sum()
    start_gt_2_percentage = (
        (start_gt_2 / total_start_entries * 100) if total_start_entries > 0 else 0
    )

    # Count entries where end > 2.0
    end_gt_2 = (combined_df["end"] > 2.0).sum()
    end_gt_2_percentage = (
        (end_gt_2 / total_end_entries * 100) if total_end_entries > 0 else 0
    )

    print(
        f"\nStart values > 2.0: {start_gt_2} out of {total_start_entries} ({start_gt_2_percentage:.2f}%)"
    )
    print(
        f"End values > 2.0: {end_gt_2} out of {total_end_entries} ({end_gt_2_percentage:.2f}%)"
    )

    # Count entries where start < 0.01
    start_lt_01 = (combined_df["start"] < 0.01).sum()
    start_lt_01_percentage = (
        (start_lt_01 / total_start_entries * 100) if total_start_entries > 0 else 0
    )
    print(
        f"Start values < 0.01: {start_lt_01} out of {total_start_entries} ({start_lt_01_percentage:.2f}%)"
    )

    # Combined analysis (either start OR end > 2.0)
    either_gt_2 = ((combined_df["start"] > 2.0) | (combined_df["end"] > 2.0)).sum()
    total_entries = len(combined_df)
    either_gt_2_percentage = (
        (either_gt_2 / total_entries * 100) if total_entries > 0 else 0
    )

    print(
        f"\nRows with either start OR end > 2.0: {either_gt_2} out of {total_entries} ({either_gt_2_percentage:.2f}%)"
    )

    # Show some examples of entries with values > 2.0
    print("\nExamples of entries with start or end > 2.0:")
    gt_2_mask = (combined_df["start"] > 2.0) | (combined_df["end"] > 2.0)
    examples = combined_df[gt_2_mask].head(10)
    if not examples.empty:
        print(examples[["start", "end", "fmin", "fmax", "category", "source_file"]])
    else:
        print("No entries found with start or end > 2.0")

    # Summary statistics
    print("\n" + "=" * 50)
    print("SUMMARY STATISTICS")
    print("=" * 50)
    print(f"Total CSV files processed: {len(csv_files)}")
    print(f"Total rows in combined dataset: {len(combined_df)}")
    print(f"Percentage of entries with start > 2.0: {start_gt_2_percentage:.2f}%")
    print(f"Percentage of entries with end > 2.0: {end_gt_2_percentage:.2f}%")
    print(f"Percentage of entries with start < 0.01: {start_lt_01_percentage:.2f}%")
    print(
        f"Percentage of rows with either start OR end > 2.0: {either_gt_2_percentage:.2f}%"
    )


if __name__ == "__main__":
    main()
