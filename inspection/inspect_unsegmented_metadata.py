import os
import pickle
import pandas as pd

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
    """Helper function to neatly print the details of a single session."""
    color = Colors.BLUE if "Empatica" in device_type else Colors.WARNING
    print(f"\n   {color}{'─' * 10} Example: {device_type} Session {'─' * 10}{Colors.ENDC}")
    _print_key_value("Session ID", f"'{session_id}'", indent=2)

    for key, value in session_info.items():
        if key == "channel_names":
            _print_key_value("Channels", f"({len(value)} found) {value}", indent=2)
        elif key == "sampling_rates":
            _print_key_value("Sampling Rates (Hz)", "", indent=2)
            for ch, rate in value.items():
                _print_key_value(f"-> {ch}", rate, indent=3)
        elif key == "labelled":
             status_color = Colors.GREEN if value else Colors.WARNING
             _print_key_value(key.capitalize(), f"{status_color}{value}{Colors.ENDC}", indent=2)
        else:
            if isinstance(value, (str, bool, int, float)):
                 _print_key_value(key.capitalize(), value, indent=2)

def print_metadata_info(metadata_path: str):
    """
    Loads and prints a clear, readable summary of the metadata.pkl file.
    """
    print(f"{Colors.BOLD}Inspecting metadata file at: {metadata_path}{Colors.ENDC}")
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
    _print_section_header("Overall Summary")
    toplevel_keys = list(metadata.keys())
    sessions_info = metadata.get('sessions_info', {})
    invalid_sessions = metadata.get('invalid_sessions', [])
    
    # This line was added to show the key count and names
    _print_key_value("Top-Level Keys", f"{len(toplevel_keys)} found: {toplevel_keys}")
    _print_key_value("Processed Sessions", len(sessions_info))
    _print_key_value("Invalid/Skipped", len(invalid_sessions))


    # --- SESSIONS INFO ---
    if 'sessions_info' in metadata and sessions_info:
        _print_section_header("Session Details")
        
        emotibit_example = None
        empatica_unlabelled_example = None
        empatica_labelled_example = None

        for session_id, s_info in sessions_info.items():
            if "emotibit" in session_id:
                if not emotibit_example:
                    emotibit_example = (session_id, s_info)
            else:
                if s_info.get('labelled') is True and not empatica_labelled_example:
                    empatica_labelled_example = (session_id, s_info)
                elif s_info.get('labelled') is False and not empatica_unlabelled_example:
                    empatica_unlabelled_example = (session_id, s_info)
            if emotibit_example and empatica_unlabelled_example and empatica_labelled_example:
                break
        
        if empatica_unlabelled_example:
            _print_session_details(empatica_unlabelled_example[0], empatica_unlabelled_example[1], "Empatica (E4) - Unlabelled")
        if empatica_labelled_example:
            _print_session_details(empatica_labelled_example[0], empatica_labelled_example[1], "Empatica (E4) - Labelled")
        if emotibit_example:
            _print_session_details(emotibit_example[0], emotibit_example[1], "EmotiBit")

    # --- CLINICAL INFO ---
    if 'clinical_info' in metadata:
        _print_section_header("Clinical Info")
        clinical_df = metadata['clinical_info']
        _print_key_value("Data Type", "pandas.DataFrame")
        _print_key_value("Shape", f"{clinical_df.shape[0]} rows, {clinical_df.shape[1]} columns")
        _print_key_value("Columns", clinical_df.columns.tolist())
        print(f"\n   {Colors.CYAN}Example (first 3 rows):{Colors.ENDC}")
        with pd.option_context('display.max_rows', 3, 'display.max_columns', 10, 'display.width', 120):
             print(clinical_df.head(3).to_string())
    
    # --- INVALID SESSIONS ---
    if invalid_sessions:
        _print_section_header("Invalid Sessions")
        _print_key_value("Count", len(invalid_sessions))
        _print_key_value("Examples", invalid_sessions[:3])

    print(f"\n{Colors.BOLD}--- End of Report ---{Colors.ENDC}")

if __name__ == '__main__':
    path_to_metadata = "data/preprocessed/unsegmented/metadata.pkl"
    print_metadata_info(path_to_metadata)
