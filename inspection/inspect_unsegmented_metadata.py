import os
import pickle
import pandas as pd
import json

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
    print(f"{indent_str}{Colors.CYAN}{key:<20}{Colors.ENDC}{Colors.GREEN}{value}{Colors.ENDC}")

def _print_session_details(session_id, session_info, device_type):
    """
    (Improved)
    A helper function to clearly print all details of a single session's metadata.
    """
    color = Colors.BLUE
    if "Unlabelled" in device_type:
        color = Colors.WARNING
    
    print(f"\n   {color}{'─' * 10} Example: {device_type} {'─' * 10}{Colors.ENDC}")
    _print_key_value("Session ID", f"'{session_id}'", indent=2)

    # Loop through all key-value pairs in the session_info dictionary
    for key, value in sorted(session_info.items()):
        # Keep the original key name, do not convert case
        key_name = key

        # Format the output for different types of values
        if isinstance(value, dict):
            # Use json.dumps to pretty-print dictionaries
            formatted_value = json.dumps(value, indent=4)
            print(f"   {Colors.CYAN}{key_name:<20}{Colors.ENDC}\n{Colors.GREEN}{formatted_value}{Colors.ENDC}")
        elif isinstance(value, list):
            _print_key_value(key_name, f"List (total {len(value)} items): {value}", indent=2)
        elif isinstance(value, bool):
            status_color = Colors.GREEN if value else Colors.WARNING
            _print_key_value(key_name, f"{status_color}{value}{Colors.ENDC}", indent=2)
        else:
            # Handle other types like strings, numbers, etc.
            _print_key_value(key_name, value, indent=2)


def print_metadata_info(metadata_path: str):
    """
    Loads and prints a clear, readable summary of the metadata.pkl file.
    """
    print(f"{Colors.BOLD}Inspecting metadata file: {metadata_path}{Colors.ENDC}")
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
    _print_section_header("OVERALL SUMMARY")
    toplevel_keys = list(metadata.keys())
    sessions_info = metadata.get('sessions_info', {})
    invalid_sessions = metadata.get('invalid_sessions', [])
    
    _print_key_value("Top-Level Keys", f"{len(toplevel_keys)} found: {toplevel_keys}")
    _print_key_value("Processed Sessions", len(sessions_info))
    _print_key_value("Invalid/Skipped", len(invalid_sessions))


    # --- SESSION INFO ---
    if 'sessions_info' in metadata and sessions_info:
        _print_section_header("SESSION DETAILS")
        
        first_session_info = list(sessions_info.values())[0]
        session_keys = list(first_session_info.keys())
        _print_key_value("Keys per session", sorted(session_keys))

        # --- NEW LOGIC FOR FINDING EXAMPLES ---
        labelled_example = None
        unlabelled_emotibit_example = None
        unlabelled_e4_example = None

        for session_id, s_info in sessions_info.items():
            is_labelled = s_info.get('labelled', False)
            # A simple check for the device type based on the path
            is_emotibit = "emotibit" in session_id

            if is_labelled and not labelled_example:
                labelled_example = (session_id, s_info)
            
            elif not is_labelled and is_emotibit and not unlabelled_emotibit_example:
                unlabelled_emotibit_example = (session_id, s_info)

            elif not is_labelled and not is_emotibit and not unlabelled_e4_example:
                unlabelled_e4_example = (session_id, s_info)

            # Stop when we have found all three types
            if labelled_example and unlabelled_emotibit_example and unlabelled_e4_example:
                break
        
        # --- NEW PRINTING LOGIC ---
        if labelled_example:
            # Generic name for the labelled session as requested
            _print_session_details(labelled_example[0], labelled_example[1], "Labelled Session")
        
        if unlabelled_emotibit_example:
            _print_session_details(unlabelled_emotibit_example[0], unlabelled_emotibit_example[1], "Unlabelled EmotiBit Session")
        
        if unlabelled_e4_example:
            _print_session_details(unlabelled_e4_example[0], unlabelled_e4_example[1], "Unlabelled Empatica E4 Session")

    # --- CLINICAL INFO ---
    if 'clinical_info' in metadata:
        _print_section_header("CLINICAL INFO")
        clinical_df = metadata['clinical_info']
        _print_key_value("Data Type", "pandas.DataFrame")
        _print_key_value("Shape", f"{clinical_df.shape[0]} rows, {clinical_df.shape[1]} columns")
        _print_key_value("Columns", clinical_df.columns.tolist())
        print(f"\n   {Colors.CYAN}Example (first 3 rows):{Colors.ENDC}")
        with pd.option_context('display.max_rows', 3, 'display.max_columns', 10, 'display.width', 120):
             print(clinical_df.head(3).to_string())
    
    # --- INVALID SESSIONS ---
    if invalid_sessions:
        _print_section_header("INVALID SESSIONS")
        _print_key_value("Count", len(invalid_sessions))
        _print_key_value("Examples", invalid_sessions[:3])

    print(f"\n{Colors.BOLD}--- End of Report ---{Colors.ENDC}")

if __name__ == '__main__':
    # Make sure this path points to your actual metadata file
    path_to_metadata = "data/preprocessed/unsegmented/metadata.pkl"
    print_metadata_info(path_to_metadata)
