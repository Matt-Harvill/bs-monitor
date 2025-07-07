import torch
from transformers import Wav2Vec2ForAudioFrameClassification


def test_actual_lengths():
    model = Wav2Vec2ForAudioFrameClassification.from_pretrained(
        "facebook/wav2vec2-base", num_labels=2
    )

    test_lengths = [16000, 32000, 48000, 64000]  # 1s, 2s, 3s, 4s

    for length in test_lengths:
        dummy_input = torch.zeros(1, length)
        with torch.no_grad():
            output = model(dummy_input, labels=None)

        theoretical = int(length / 16000 * 49)  # Your calculation
        actual = output.logits.shape[1]

        print(f"Input: {length} samples ({length/16000:.1f}s)")
        print(f"  Theoretical (49Hz): {theoretical}")
        print(f"  Actual output: {actual}")
        print(f"  Difference: {actual - theoretical}")
        print()


test_actual_lengths()

# Calculate actual frame rate
actual_rates = []
for length, actual_frames in [(16000, 49), (32000, 99), (48000, 149), (64000, 199)]:
    duration = length / 16000
    actual_rate = actual_frames / duration
    actual_rates.append(actual_rate)
    print(f"{duration}s: {actual_frames} frames = {actual_rate:.2f} Hz")

print(f"Average actual rate: {sum(actual_rates)/len(actual_rates):.2f} Hz")
