import pandas as pd
import numpy as np
import os
from timebase.data.static import LABEL_COLS # Assuming LABEL_COLS is accessible

def get_dummy_clinical_info_emotibit():
    """
    Generates a dummy clinical dataset specifically for the prepared EmotiBit data.

    This function creates a 14-subject dataset based on the following folder structure:
    - Folders '1-1' through '1-7' are treated as 7 unique "case" subjects.
    - Folders '2-8' through '2-14' are treated as 7 unique "control" subjects.
    """
    
    # --- Define the paths and subject IDs for Cases and Controls ---
    base_path = "data/raw_data/barcelona"
    
    # 7 Case Subjects
    case_subject_ids = [f"E{i:02d}" for i in range(1, 8)] # E01 to E07
    case_session_folders = [f"1-{i}" for i in range(1, 8)] # 1-1 to 1-7

    # 7 Control Subjects
    control_subject_ids = [f"E{i:02d}" for i in range(8, 15)] # E08 to E14
    control_session_folders = [f"2-{i}" for i in range(8, 15)] # 2-8 to 2-14

    # --- Create data for the 7 "Case" subjects ---
    cases_data = {
        'Sub_ID': case_subject_ids,
        'age': np.random.randint(25, 55, len(case_subject_ids)),
        'sex': np.random.randint(0, 2, len(case_subject_ids)),
        'status': 'MDE_BD', # Assigning a "case" status
        'time': 'T0',
        'YMRS_SUM': np.random.randint(8, 25, len(case_subject_ids)).astype(float),
        'HDRS_SUM': np.random.randint(8, 25, len(case_subject_ids)).astype(float),
        'Session_Code': [os.path.join(base_path, folder) for folder in case_session_folders],
    }

    # --- Create data for the 7 "Control" subjects ---
    controls_data = {
        'Sub_ID': control_subject_ids,
        'age': np.random.randint(25, 55, len(control_subject_ids)),
        'sex': np.random.randint(0, 2, len(control_subject_ids)),
        'status': 'Eu_BD', # Assigning a "control" status
        'time': 'T0',
        'YMRS_SUM': np.random.randint(0, 7, len(control_subject_ids)).astype(float),
        'HDRS_SUM': np.random.randint(0, 7, len(control_subject_ids)).astype(float),
        'Session_Code': [os.path.join(base_path, folder) for folder in control_session_folders],
    }

    # Combine into DataFrames
    df_cases = pd.DataFrame(cases_data)
    df_controls = pd.DataFrame(controls_data)

    # Create the final 14-row DataFrame
    df = pd.concat([df_cases, df_controls], ignore_index=True)

    # --- Fill in any other required columns from LABEL_COLS with dummy values ---
    total_subjects = len(df)
    for col in LABEL_COLS:
        if col not in df.columns:
            if 'YMRS' in col or 'HDRS' in col:
                # Fill other scale items with low random values
                df[col] = np.random.randint(0, 4, total_subjects).astype(float)
            elif col == 'IPAQ_total':
                df[col] = 1500.0
            else:
                 # Use a default that won't cause issues
                 df[col] = 0.0

    # Ensure the final DataFrame has the exact column order as defined in LABEL_COLS
    df = df[LABEL_COLS]

    return df

if __name__ == '__main__':
    # You can run this script directly to test it and see the output
    dummy_emotibit_df = get_dummy_clinical_info_emotibit()
    print("Generated Dummy EmotiBit Clinical Info:")
    print(dummy_emotibit_df.head())
    print("\nDataFrame Info:")
    dummy_emotibit_df.info()
    
    # Optional: Save to an Excel file for inspection
    # dummy_emotibit_df.to_excel("dummy_emotibit_clinical_info.xlsx", index=False)
    # print("\nSaved to dummy_emotibit_clinical_info.xlsx")
