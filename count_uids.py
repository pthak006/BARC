import json
import os
import argparse
from collections import Counter # Can also use a set

def count_unique_uids(jsonl_file_path):
    """
    Reads a JSONL file, extracts the 'uid' from each line,
    and counts the number of unique UIDs.

    Args:
        jsonl_file_path (str): The path to the JSONL file.

    Returns:
        int: The number of unique UIDs found, or None if an error occurs.
    """
    if not os.path.exists(jsonl_file_path):
        print(f"ERROR: File not found at '{jsonl_file_path}'")
        return None

    unique_uids = set()
    line_count = 0
    error_count = 0

    print(f"Processing file: {jsonl_file_path}")
    try:
        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line_count += 1
                try:
                    # Load the JSON object from the line
                    data = json.loads(line)

                    # Extract the UID, handle missing key gracefully
                    uid = data.get("uid")

                    if uid is not None:
                        # Add the UID to the set (sets automatically handle uniqueness)
                        # Convert to string for consistent comparison if UIDs might be numeric
                        unique_uids.add(str(uid))
                    else:
                        print(f"Warning: 'uid' key missing or null in line {line_count}. Skipping line.")
                        error_count += 1

                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON in line {line_count}. Skipping line.")
                    error_count += 1
                except Exception as e:
                    print(f"Warning: Unexpected error processing line {line_count}: {e}. Skipping line.")
                    error_count += 1

    except Exception as e:
        print(f"ERROR: Failed to read file '{jsonl_file_path}': {e}")
        return None

    print(f"\nFinished processing.")
    print(f"Total lines read: {line_count}")
    if error_count > 0:
        print(f"Lines skipped due to errors or missing UID: {error_count}")

    num_unique = len(unique_uids)
    print(f"Number of unique UIDs found: {num_unique}")
    return num_unique

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count unique UIDs in a JSONL file.")
    parser.add_argument(
        "jsonl_file",
        help="Path to the input JSONL file (e.g., output from eval_code_samples.py)."
    )

    args = parser.parse_args()
    count_unique_uids(args.jsonl_file)
