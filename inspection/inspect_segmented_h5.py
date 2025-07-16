import h5py
import numpy as np
import os

class Colors:
    """A simple class to add color to terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def inspect_segmented_h5(filepath: str):
    """
    Loads and inspects a segmented H5 file (a single segment), 
    printing a summary of its contents.
    """
    print(f"{Colors.BOLD}Inspecting segmented H5 file: {filepath}{Colors.ENDC}")
    if not os.path.exists(filepath):
        print(f"--- ERROR: File not found. ---")
        return

    try:
        with h5py.File(filepath, 'r') as h5f:
            print(f"{Colors.HEADER}{'='*15} File Content Summary {'='*15}{Colors.ENDC}")
            
            keys = list(h5f.keys())
            print(f"File contains {len(keys)} datasets (channels/features): {keys}\n")

            # Iterate through each dataset in the file
            for key in keys:
                dataset = h5f[key]
                
                # For segment files, the data volume is small, so we can load it all
                data_content = dataset[:]
                
                print(f"   {Colors.BLUE}--- Dataset: {key} ---{Colors.ENDC}")
                print(f"   {Colors.CYAN}{'Shape':<25}{Colors.ENDC}{Colors.GREEN}{str(dataset.shape)}{Colors.ENDC}")
                print(f"   {Colors.CYAN}{'Data Type (Dtype)':<25}{Colors.ENDC}{Colors.GREEN}{dataset.dtype}{Colors.ENDC}")
                # Only print the first 5 elements as a preview to avoid long output
                print(f"   {Colors.CYAN}{'Data Preview (first 5 items)':<25}{Colors.ENDC}{Colors.GREEN}\n{data_content.flatten()[:5]}{Colors.ENDC}\n")

    except Exception as e:
        print(f"--- ERROR: Could not read H5 file: {e} ---")

if __name__ == '__main__':
    path_to_segmented_file = "data/preprocessed/sl512_ss128/wesad/S5/S5_E4_Data/0.h5"
    inspect_segmented_h5(path_to_segmented_file)

    # Example for an EmotiBit file
    path_to_emotibit_file = "data/preprocessed/sl512_ss128/emotibit/2025-06-11_10-53-14-135414/0.h5"
    inspect_segmented_h5(path_to_emotibit_file)