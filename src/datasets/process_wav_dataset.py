import argparse
import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def create_chunked_dataset(wav_file_path, chunk_durations: list | None = None):
    """
    Process a WAV file and create datasets with different chunk sizes.

    Args:
        wav_file_path (str): Path to the input WAV file
        chunk_durations (list): List of chunk durations in seconds
    """
    # Get the base name without extension
    base_name = Path(wav_file_path).stem

    # Create main dataset directory
    dataset_dir = Path("data") / f"{base_name}-dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Load the audio file
    audio, sample_rate = librosa.load(wav_file_path, sr=None)

    print(f"Processing {wav_file_path}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Duration: {len(audio) / sample_rate:.2f} seconds")

    # Process each chunk duration
    for duration in chunk_durations:
        chunk_size = int(duration * sample_rate)

        # Create subdirectory for this chunk size
        chunk_dir = dataset_dir / f"{duration}s-chunks"
        chunk_dir.mkdir(exist_ok=True)

        # Calculate number of chunks
        num_chunks = len(audio) // chunk_size

        print(f"\nCreating {num_chunks} chunks of {duration}s each...")

        # Create chunks
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size

            chunk = audio[start_idx:end_idx]

            # Save chunk
            chunk_filename = f"{base_name}_chunk_{i:04d}.wav"
            chunk_path = chunk_dir / chunk_filename

            sf.write(chunk_path, chunk, sample_rate)

        print(f"Saved {num_chunks} chunks to {chunk_dir}")

        # Handle remaining audio if it's significant
        remaining_samples = len(audio) % chunk_size
        if remaining_samples > sample_rate * 0.5:  # If more than 0.5 seconds remaining
            remaining_chunk = audio[-remaining_samples:]

            # Pad with zeros to match chunk size
            padded_chunk = np.pad(
                remaining_chunk, (0, chunk_size - remaining_samples), mode="constant"
            )

            chunk_filename = f"{base_name}_chunk_{num_chunks:04d}_padded.wav"
            chunk_path = chunk_dir / chunk_filename

            sf.write(chunk_path, padded_chunk, sample_rate)
            print(f"Saved padded remainder chunk to {chunk_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Process WAV file into chunked datasets"
    )
    parser.add_argument("wav_file", help="Path to the input WAV file")
    parser.add_argument(
        "--chunk-durations",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8],
        help="Chunk durations in seconds (default: 1 2 4 8)",
    )

    args = parser.parse_args()

    # Check if file exists
    if not os.path.exists(args.wav_file):
        print(f"Error: File {args.wav_file} not found!")
        return

    # Check if it's a WAV file
    if not args.wav_file.lower().endswith(".wav"):
        print(f"Error: {args.wav_file} is not a WAV file!")
        return

    # Process the file
    create_chunked_dataset(args.wav_file, args.chunk_durations)

    print("\nDataset creation complete!")


if __name__ == "__main__":
    main()
