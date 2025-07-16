import os
import pickle
import numpy as np
import pandas as pd
import argparse

# --- Import the original functions from your project ---
from timebase.data.reader import split_into_sets
from timebase.data.static import DICT_STATE 

# ==============================================================================
# --- CORE LOGIC ---
# ==============================================================================

def parse_args():
    """
    Parses all necessary command-line arguments for the script.
    """
    parser = argparse.ArgumentParser(description="Generate train/val/test splits from segmented metadata.")
    
    parser.add_argument(
        "--metadata_path",
        type=str,
        default="data/preprocessed/sl512_ss128/metadata.pkl",
        help="Path to your metadata.pkl file in the segmented data folder."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="inference/data_splits",
        help="Directory where the output split files will be saved."
    )
    parser.add_argument(
        "--split_mode",
        type=int,
        default=0,
        choices=[0, 1],
        help="The split mode used for training (0 = time-split, 1 = subject-split)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="The random seed used for training."
    )
    
    return parser.parse_args()

def main(args):
    """
    Main function to load metadata, perform splitting, and save the results.
    Accepts an 'args' object containing all configurations.
    """
    print(f"Loading metadata from: {args.metadata_path}")
    if not os.path.exists(args.metadata_path):
        print(f"ERROR: Metadata file not found at '{args.metadata_path}'")
        return
        
    with open(args.metadata_path, 'rb') as f:
        metadata = pickle.load(f)

    # Add other necessary attributes to the args object for the split_into_sets function
    args.ds_info = metadata.get('ds_info', {})
    args.e4selflearning = False
    args.task_mode = 1  # 1 for fine-tuning/classification
    args.pretext_task = None
    args.exclude_anomalies = False

    print("Extracting necessary data for splitting...")
    y = metadata['sessions_labels']
    sleep_status = metadata['sessions_sleep_status']
    recording_id = metadata['recording_id']
    all_paths = metadata['sessions_paths']

    print(f"Performing data split (mode={args.split_mode}, seed={args.seed})...")
    try:
        # Use the imported original function and the args object for splitting
        split_indices = split_into_sets(args, y=y, sleep_status=sleep_status, recording_id=recording_id)
    except Exception as e:
        print(f"An error occurred during splitting: {e}")
        print("Please ensure your PYTHONPATH is set correctly and the `timebase` module is accessible.")
        return

    # Create the output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving split file lists to '{args.output_dir}/' directory...")

    for set_name, indices in split_indices.items():
        # Get the full file paths for the current set
        set_paths = all_paths[indices]
        
        # Print summary
        print(f"\n--- {set_name.upper()} Set ---")
        print(f"Total segments: {len(set_paths)}")
        if len(set_paths) > 0:
            print(f"Example segment path: {set_paths[0]}")

        # Save the full list of paths to a text file
        output_filename = os.path.join(args.output_dir, f"{set_name}_set_paths.txt")
        with open(output_filename, 'w') as f:
            for path in set_paths:
                f.write(f"{path}\n")
        print(f"Full list saved to: {output_filename}")

    print("\nScript finished successfully.")

if __name__ == '__main__':
    # Parse all arguments from the command line
    args = parse_args()
    # Pass the populated args object to the main function
    main(args)
