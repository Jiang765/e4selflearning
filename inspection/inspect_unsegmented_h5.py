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

def inspect_preprocessed_h5(filepath: str):
    """
    Loads and inspects a preprocessed (unsegmented) H5 file,
    printing a summary of its contents.
    """
    print(f"{Colors.BOLD}Inspecting preprocessed H5 file: {filepath}{Colors.ENDC}")
    if not os.path.exists(filepath):
        print(f"--- ERROR: File not found. ---")
        return

    try:
        with h5py.File(filepath, 'r') as h5f:
            print(f"{Colors.HEADER}{'='*15} File Content Summary {'='*15}{Colors.ENDC}")
            
            keys = list(h5f.keys())
            print(f"File contains {len(keys)} datasets (channels): {keys}\n")

            # Iterate through each dataset (i.e., each channel) in the file
            for key in keys:
                dataset = h5f[key]
                
                # To avoid loading all data, we only read a small slice for preview
                preview_slice = dataset[:5] if len(dataset.shape) == 1 else dataset[:5, :]
                
                print(f"   {Colors.BLUE}--- Channel: {key} ---{Colors.ENDC}")
                print(f"   {Colors.CYAN}{'Shape':<25}{Colors.ENDC}{Colors.GREEN}{str(dataset.shape)}{Colors.ENDC}")
                print(f"   {Colors.CYAN}{'Data Type (Dtype)':<25}{Colors.ENDC}{Colors.GREEN}{dataset.dtype}{Colors.ENDC}")
                print(f"   {Colors.CYAN}{'Data Preview (first 5 items)':<25}{Colors.ENDC}{Colors.GREEN}\n{preview_slice}{Colors.ENDC}\n")

    except Exception as e:
        print(f"--- ERROR: Could not read H5 file: {e} ---")

if __name__ == '__main__':
    path_to_preprocessed_file = "data/preprocessed/unsegmented/wesad/S2/S2_E4_Data/channels.h5"
    inspect_preprocessed_h5(path_to_preprocessed_file)

    # Example for an EmotiBit file
    path_to_emotibit_file = "data/preprocessed/unsegmented/emotibit/2025-06-11_10-53-14-135414/channels.h5"
    inspect_preprocessed_h5(path_to_emotibit_file)
