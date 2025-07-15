import os
import pickle
import pandas as pd
import numpy as np

class Colors:
    """A simple class to add color to terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def _print_section_header(title):
    """Prints a styled header for a section."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 20} {title.upper()} {'=' * 20}{Colors.ENDC}")

def _print_key_value(key, value, indent=1):
    """Prints a styled key-value pair."""
    indent_str = "   " * indent
    print(f"{indent_str}{Colors.CYAN}{key:<35}{Colors.ENDC}{Colors.GREEN}{value}{Colors.ENDC}")

def _handle_array(key, data_array, description):
    """
    (Improved)
    Generic handler for printing details and useful examples of a numpy array.
    """
    _print_key_value(f"Key: '{key}'", "")
    _print_key_value("  - Description", description)
    _print_key_value("  - Type", type(data_array))
    _print_key_value("  - Shape", data_array.shape)
    _print_key_value("  - Example (first 5 elements)", data_array[:5])
    
    # --- MODIFIED: Show ALL unique values ---
    unique_values = np.unique(data_array)
    _print_key_value("  - Unique Values Count", len(unique_values))
    # .tolist() ensures the full array is printed nicely
    _print_key_value("  - Unique Values (first 5 elements)", unique_values.tolist()[:5])
    
def print_segmented_metadata_info(metadata_path: str):
    """
    Loads and prints a detailed, readable summary of the segmented metadata.pkl file.
    """
    print(f"{Colors.BOLD}Inspecting segmented metadata file: {metadata_path}{Colors.ENDC}")
    if not os.path.exists(metadata_path):
        print(f"{Colors.FAIL}--- ERROR: File not found. ---{Colors.ENDC}")
        return

    try:
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
    except Exception as e:
        print(f"{Colors.FAIL}--- ERROR: Could not read file: {e} ---{Colors.ENDC}")
        return

    # --- TOP-LEVEL SUMMARY ---
    _print_section_header("DATASET OVERALL SUMMARY")
    total_segments = len(metadata.get('sessions_paths', []))
    _print_key_value("Top-Level Keys", sorted(list(metadata.keys())))
    _print_key_value("Total Segments Found", total_segments)
    if 'ds_info' in metadata:
        _print_key_value("Segment Length", f"{metadata['ds_info'].get('segment_length', 'N/A')} seconds")
        if 'step_size' in metadata['ds_info']:
            _print_key_value("Step Size", f"{metadata['ds_info'].get('step_size')} seconds")

    # --- DETAILED KEY-VALUE INFO ---
    _print_section_header("METADATA KEY DETAILS")

    key_handlers = {
        "sessions_paths": lambda v: _handle_array("sessions_paths", v, "Array of absolute paths to each segment's .h5 file."),
        "recording_id": lambda v: _handle_array("recording_id", v, "Array assigning a unique recording ID to each segment for cross-validation."),
        "sessions_sleep_status": lambda v: _handle_array("sessions_sleep_status", v, "Array marking the sleep state for each segment (0=wake)."),
        "sessions_segments_unix_t0": lambda v: _handle_array("sessions_segments_unix_t0", v, "Array of start timestamps for each segment."),
    }

    for key in sorted(metadata.keys()):
        value = metadata[key]
        if key in key_handlers:
            key_handlers[key](value)
        elif key == "sessions_labels":
            _print_key_value("Key: 'sessions_labels'", "")
            _print_key_value("  - Description", "Dictionary of labels, where each key is a label name.")
            _print_key_value("  - Type", type(value))
            _print_key_value("  - Available Labels", list(value.keys()))
            # Show an example for one of the labels
            if "status" in value:
                example_key = "status"
                _print_key_value(f"  - Example from '{example_key}' shape", value[example_key].shape)
                _print_key_value(f"  - Example from '{example_key}' values", value[example_key][:5])
        elif key == "ds_info":
            _print_key_value("Key: 'ds_info'", "")
            _print_key_value("  - Description", "Configuration information about the dataset creation process.")
            _print_key_value("  - Type", type(value))
            for sub_key, sub_val in value.items():
                if sub_key == "wake_sleep_off":
                     _print_key_value(f"    - {sub_key}", f"Dictionary with {len(sub_val)} entries.")
                     # Print first 2 example entries from the wake_sleep_off dict
                     for i, (session_id, times) in enumerate(sub_val.items()):
                         if i >= 2: break
                         _print_key_value(f"      - Example Session", f"'{session_id}': {times}")
                else:
                    _print_key_value(f"    - {sub_key}", sub_val)
        # Skip printing full session/clinical info unless needed, as they are large
        elif key in ["sessions_info", "clinical_info"]:
             _print_key_value(f"Key: '{key}'", f"Contains pre-segmentation metadata. (Details omitted for brevity).")


    print(f"\n{Colors.BOLD}--- End of Report ---{Colors.ENDC}")

if __name__ == '__main__':
    path_to_metadata = "data/preprocessed/sl512_ss128/metadata.pkl"
    
    print_segmented_metadata_info(path_to_metadata)
