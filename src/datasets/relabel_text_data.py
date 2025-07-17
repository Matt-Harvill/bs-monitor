import argparse
import os
from pathlib import Path


def relabel_text_file(input_file_path):
    """
    Relabel the third column in a text file by replacing 'b' and 'v' labels with 'sb'.

    Args:
        input_file_path (str): Path to the input text file
    """
    input_path = Path(input_file_path)

    # Create output filename with '_clean' suffix
    output_filename = input_path.stem + "_clean" + input_path.suffix
    output_path = input_path.parent / output_filename

    print(f"Processing {input_file_path}")
    print(f"Output will be saved to {output_path}")

    lines_processed = 0
    labels_changed = 0

    with (
        open(input_path, encoding="utf-8") as infile,
        open(output_path, "w", encoding="utf-8") as outfile,
    ):
        for line in infile:
            line = line.strip()
            if not line:
                outfile.write("\n")
                continue

            # Split the line into components
            parts = line.split("\t")

            if len(parts) >= 3:
                # Get the third column (index 2)
                original_label = parts[2]

                # Replace 'b' and 'v' with 'sb'
                if original_label == "b" or original_label == "v":
                    parts[2] = "sb"
                    labels_changed += 1
                    print(f"Line {lines_processed + 1}: '{original_label}' -> 'sb'")

                # Write the modified line
                outfile.write("\t".join(parts) + "\n")
            else:
                # If line doesn't have enough columns, write as-is
                outfile.write(line + "\n")

            lines_processed += 1

    print("\nProcessing complete!")
    print(f"Total lines processed: {lines_processed}")
    print(f"Labels changed: {labels_changed}")
    print(f"Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Relabel third column in text files by replacing b/v with sb"
    )
    parser.add_argument("text_file", help="Path to the input text file")

    args = parser.parse_args()

    # Check if file exists
    if not os.path.exists(args.text_file):
        print(f"Error: File {args.text_file} not found!")
        return

    # Check if it's a text file
    if not args.text_file.lower().endswith(".txt"):
        print(f"Error: {args.text_file} is not a text file!")
        return

    # Process the file
    relabel_text_file(args.text_file)


if __name__ == "__main__":
    main()
