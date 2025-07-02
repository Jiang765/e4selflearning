import pandas as pd
import numpy as np
from timebase.data.static import LABEL_COLS

def get_dummy_clinical_info():
    """
    Generates a dummy clinical dataset using ONLY the 15 available WESAD subjects
    (S2-S17, excluding S12) to ensure the script's filtering logic works.

    This function assigns roles to the 15 subjects:
    - 8 subjects are designated as "cases".
    - 7 subjects are designated as "controls".
    """
    
    # Define the exact list of 15 valid WESAD subject IDs
    valid_wesad_ids = [i for i in range(2, 18) if i != 12]
    
    # Split these 15 IDs into two groups for our dummy data
    case_subject_ids = valid_wesad_ids[:8]      # First 8 subjects (S2-S9) will be cases
    control_subject_ids = valid_wesad_ids[8:]   # Remaining 7 subjects will be controls

    # --- Create data for the 8 "Case" subjects ---
    cases_data = {
        'Sub_ID': [f'{i:02d}' for i in range(len(case_subject_ids))],
        'age': np.random.randint(25, 55, len(case_subject_ids)),
        'sex': np.random.randint(0, 2, len(case_subject_ids)),
        'status': 'MDE_BD',
        'time': 'T0',
        'YMRS_SUM': np.random.randint(8, 25, len(case_subject_ids)).astype(float),
        'HDRS_SUM': np.random.randint(8, 25, len(case_subject_ids)).astype(float),
        'Session_Code': [f'data/raw_data/barcelona/S{sid}/S{sid}_E4_Data' for sid in case_subject_ids],
    }

    # --- Create data for the 7 "Control" subjects ---
    controls_data = {
        'Sub_ID': [f'{i:02d}' for i in range(len(control_subject_ids))],
        'age': np.random.randint(25, 55, len(control_subject_ids)),
        'sex': np.random.randint(0, 2, len(control_subject_ids)),
        'status': 'Eu_BD',
        'time': 'T0',
        'YMRS_SUM': np.random.randint(0, 7, len(control_subject_ids)).astype(float),
        'HDRS_SUM': np.random.randint(0, 7, len(control_subject_ids)).astype(float),
        'Session_Code': [f'data/raw_data/barcelona/S{sid}/S{sid}_E4_Data' for sid in control_subject_ids],
    }

    # Combine into DataFrames
    df_cases = pd.DataFrame(cases_data)
    df_controls = pd.DataFrame(controls_data)

    # Create the final 15-row DataFrame
    df = pd.concat([df_cases, df_controls], ignore_index=True)

    # --- Fill in the remaining required columns with dummy values ---
    total_subjects = len(df)
    for col in LABEL_COLS:
        if col not in df.columns:
            if 'YMRS' in col or 'HDRS' in col:
                df[col] = np.random.randint(0, 4, total_subjects).astype(float)
            elif col == 'IPAQ_total':
                df[col] = 1500.0
            else:
                 df[col] = 0.0

    # Ensure the final DataFrame has the exact column order
    df = df[LABEL_COLS]

    return df