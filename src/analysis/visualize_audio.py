import glob
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

FMAX = 2000


def visualize_audio_file(file_path: str, figsize: tuple[int, int] = (15, 10)) -> Figure:
    """
    Visualize a single audio file with multiple plots
    """
    # Load audio file
    y, sr = librosa.load(file_path, sr=None)  # sr=None preserves original sample rate

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f"Audio Analysis: {Path(file_path).name}", fontsize=16)

    # 1. Waveform
    time = np.linspace(0, len(y) / sr, len(y))
    axes[0, 0].plot(time, y)
    axes[0, 0].set_title("Waveform")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Spectrogram (limited to 2000Hz)
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    img = librosa.display.specshow(
        S_db,
        sr=sr,
        x_axis="time",
        y_axis="hz",
        ax=axes[0, 1],
        cmap="viridis",
        fmax=FMAX,
    )
    axes[0, 1].set_ylim(0, FMAX)  # Explicitly set y-axis limit
    axes[0, 1].set_title("Spectrogram")
    axes[0, 1].set_ylabel("Frequency (Hz)")
    fig.colorbar(img, ax=axes[0, 1], format="%+2.0f dB")

    # 3. RMS Energy over time
    hop_length = 512
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    times_rms = librosa.times_like(rms, sr=sr, hop_length=hop_length)
    axes[1, 0].plot(times_rms, rms)
    axes[1, 0].set_title("RMS Energy Over Time")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("RMS Energy")
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Frequency spectrum (average over time)
    fft = np.fft.fft(y)
    freqs = np.fft.fftfreq(len(fft), 1 / sr)
    magnitude = np.abs(fft)

    # Only plot positive frequencies up to 2000Hz
    pos_mask = (freqs >= 0) & (freqs <= FMAX)
    axes[1, 1].plot(freqs[pos_mask], magnitude[pos_mask])
    axes[1, 1].set_title(f"Frequency Spectrum (0-{FMAX} Hz)")
    axes[1, 1].set_xlabel("Frequency (Hz)")
    axes[1, 1].set_ylabel("Magnitude")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure to figures directory
    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)

    # Create filename from original file path
    original_filename = Path(file_path).stem
    output_filename = figures_dir / f"{original_filename}_analysis.png"
    fig.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(f"Figure saved to: {output_filename}")

    return fig


def analyze_multiple_files(
    directory_path: str, file_pattern: str = "*.wav", max_files: int = 5
) -> None:
    """
    Analyze multiple audio files in a directory
    """
    files = glob.glob(str(Path(directory_path) / file_pattern))

    if not files:
        print(f"No files found matching pattern {file_pattern} in {directory_path}")
        return

    print(f"Found {len(files)} audio files")
    files_to_analyze = files[:max_files]  # Limit to avoid too many plots

    for i, file_path in enumerate(files_to_analyze):
        print(f"\nAnalyzing file {i+1}/{len(files_to_analyze)}: {Path(file_path).name}")
        try:
            fig = visualize_audio_file(file_path)
            plt.close(fig)  # Close the figure to free memory
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")


def quick_stats(file_path: str) -> None:
    """
    Print quick statistics about the audio file
    """
    y, sr = librosa.load(file_path, sr=None)

    print(f"\n--- Audio Statistics for {Path(file_path).name} ---")
    print(f"Sample Rate: {sr} Hz")
    print(f"Duration: {len(y)/sr:.2f} seconds")
    print(f"Channels: {'Mono' if y.ndim == 1 else 'Stereo'}")
    print(f"Max Amplitude: {np.max(np.abs(y)):.4f}")
    print(f"RMS Energy: {np.sqrt(np.mean(y**2)):.4f}")
    print(f"Zero Crossing Rate: {np.mean(librosa.feature.zero_crossing_rate(y)):.4f}")


# Example usage:
if __name__ == "__main__":
    # Replace with your directory path
    audio_directory = "data/bs-dataset"

    # Method 1: Analyze multiple files
    analyze_multiple_files(audio_directory, "*.wav", max_files=10)

    # # Method 2: Analyze single file
    # single_file = "data/bs-dataset/0_a.wav"
    # visualize_audio_file(single_file)
    # quick_stats(single_file)
