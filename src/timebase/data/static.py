# All the static variables related to the dataset

import os

import numpy as np

FILE_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

HR_OFFSET = 10  # HR data record 10s after the t0
# channel names of the csv recorded data 
CSV_CHANNELS = ["ACC", "BVP", "EDA", "HR", "TEMP", "IBI"]
# dictionary of channel:sampling frequency -> key: value pairs
CHANNELS_FREQ = {
    "BVP": 64,
    "EDA": 4,
    "HR": 1,
    "TEMP": 4,
    "ACC_x": 32,
    "ACC_y": 32,
    "ACC_z": 32,
}

# Sample rate dictionary
EMOTIBIT_NOMINAL_SAMPLE_RATES = {
    # Accelerometer
    'AX': 25, 'AY': 25, 'AZ': 25,
    # Gyroscope
    'GX': 25, 'GY': 25, 'GZ': 25,
    # Magnetometer
    'MX': 25, 'MY': 25, 'MZ': 25,
    # Electrodermal Activity
    'EA': 15, 'EL': 15,
    # Skin Conductance Response Frequency
    'SF': 3,
    # Temperature Sensors (7.5Hz is rounded to 8)
    'T1': 8, 'TH': 8,
    # PPG (Photoplethysmogram)
    'PI': 25, 'PR': 25, 'PG': 25,
    # For variable-rate or event-driven signals, set to -1 to trigger calculation.
    'BI': -1, 'HR': -1, 'SA': -1, 'SR': -1, 
}

EMOTIBIT_CHANNELS = ["AX", "AY", "AZ", "GX", "GY", "GZ", "EA", "PI", "PR", "PG", "T1"]

EMOTIBIT_CHANNELS_FREQ = {
    channel: EMOTIBIT_NOMINAL_SAMPLE_RATES[channel] 
    for channel in EMOTIBIT_CHANNELS
}

# maximum values of each individual symptom in YMRS and HDRS
ITEM_MAX = {
    "YMRS1": 4,
    "YMRS2": 4,
    "YMRS3": 4,
    "YMRS4": 4,
    "YMRS5": 8,
    "YMRS6": 8,
    "YMRS7": 4,
    "YMRS8": 8,
    "YMRS9": 8,
    "YMRS10": 4,
    "YMRS11": 4,
    "HDRS1": 4,
    "HDRS2": 4,
    "HDRS3": 4,
    "HDRS4": 2,
    "HDRS5": 2,
    "HDRS6": 2,
    "HDRS7": 4,
    "HDRS8": 4,
    "HDRS9": 4,
    "HDRS10": 4,
    "HDRS11": 4,
    "HDRS12": 2,
    "HDRS13": 2,
    "HDRS14": 2,
    "HDRS15": 4,
    "HDRS16": 3,
    "HDRS17": 2,
}

YMRS_item_ranks = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
HDRS_item_ranks = [5, 5, 5, 3, 3, 3, 5, 5, 5, 5, 5, 3, 3, 3, 5, 4, 3]
# combined item ranks
ITEM_RANKS = {
    "YMRS1": 5,
    "YMRS2": 5,
    "YMRS3": 5,
    "YMRS4": 5,
    "YMRS5": 5,
    "YMRS6": 5,
    "YMRS7": 5,
    "YMRS8": 5,
    "YMRS9": 5,
    "YMRS10": 5,
    "YMRS11": 5,
    "HDRS1": 5,
    "HDRS2": 5,
    "HDRS3": 5,
    "HDRS4": 3,
    "HDRS5": 3,
    "HDRS6": 3,
    "HDRS7": 5,
    "HDRS8": 5,
    "HDRS9": 5,
    "HDRS10": 5,
    "HDRS11": 5,
    "HDRS12": 3,
    "HDRS13": 3,
    "HDRS14": 3,
    "HDRS15": 5,
    "HDRS16": 3,
    "HDRS17": 3,
}

SLEEP_DICT = {0.0: "wake", 1.0: "sleep", 2.0: "off-body"}

SLEEP_COLOR_DICT = {
    "wake": "cyan",
    "sleep": "magenta",
    "off-body": "black",
}

# some items of the YMRS are scored with 2 unit interval spaces
RANK_NORMALIZER = {
    "YMRS1": 1,
    "YMRS2": 1,
    "YMRS3": 1,
    "YMRS4": 1,
    "YMRS5": 2,
    "YMRS6": 2,
    "YMRS7": 1,
    "YMRS8": 2,
    "YMRS9": 2,
    "YMRS10": 1,
    "YMRS11": 1,
    "HDRS1": 1,
    "HDRS2": 1,
    "HDRS3": 1,
    "HDRS4": 1,
    "HDRS5": 1,
    "HDRS6": 1,
    "HDRS7": 1,
    "HDRS8": 1,
    "HDRS9": 1,
    "HDRS10": 1,
    "HDRS11": 1,
    "HDRS12": 1,
    "HDRS13": 1,
    "HDRS14": 1,
    "HDRS15": 1,
    "HDRS16": 1,
    "HDRS17": 1,
}

STATES = {
    "MDE_BD": "Bipolar Depression",
    "MDE_MDD": "Unipolar Depression",
    "ME": "Manic Episode",
    "MX": "Mixed Episode",
    "PSY_EP": "Psychotic Episode",
    "Eu_BD": "Euthymia BD",
    "Eu_MDD": "Euthymia MDD",
    "SCZ_REM": "Schizophrenia Remission",
    "HC": "Controls",
    "INC": "Incomplete",
    "SUD": "SUD",
}

DICT_STATE = {
    "MDE_BD": 0.0,
    "MDE_MDD": 1.0,
    "ME": 2.0,
    "MX": 3.0,
    "PSY_EP": 4.0,
    "Eu_BD": 5.0,
    "Eu_MDD": 6.0,
    "SCZ_REM": 7.0,
    "HC": 8.0,
    "INC": 9.0,
    "SUD": 10.0,
}

DICT_STATE_COLOR = {
    "MDE_BD": "forestgreen",
    "MDE_MDD": "royalblue",
    "ME": "red",
    "MX": "gold",
    "PSY_EP": "grey",
    "Eu_BD": "lightsalmon",
    "Eu_MDD": "chocolate",
    "SCZ_REM": "fuchsia",
    "HC": "lawngreen",
    "INC": "pink",
    "SUD": "olive",
}

DICT_TIME = {"T0": 0.0, "T1": 1.0, "T2": 2.0, "T3": 3.0, "T4": 4.0}

LABEL_COLS = [
    "Sub_ID",
    "age",
    "sex",
    "status",
    "time",
    "Session_Code",
    "YMRS1",
    "YMRS2",
    "YMRS3",
    "YMRS4",
    "YMRS5",
    "YMRS6",
    "YMRS7",
    "YMRS8",
    "YMRS9",
    "YMRS10",
    "YMRS11",
    "YMRS_SUM",
    "HDRS1",
    "HDRS2",
    "HDRS3",
    "HDRS4",
    "HDRS5",
    "HDRS6",
    "HDRS7",
    "HDRS8",
    "HDRS9",
    "HDRS10",
    "HDRS11",
    "HDRS12",
    "HDRS13",
    "HDRS14",
    "HDRS15",
    "HDRS16",
    "HDRS17",
    "HDRS_SUM",
    "IPAQ_total",
    "YMRS_discretized",
    "HDRS_discretized",
]

YOUNG_HAMILTON_DICT = {
    key: idx
    for idx, key in enumerate(
        [f"young{y}_hamilton{h}" for y in range(5) for h in range(5)]
    )
}

UNLABELLED_DATA_PATHS = {
    "adarp": "ADARP/Data",
    "big-ideas": "big-ideas/big-ideas-lab-glycemic-variability-and-wearable"
    "-device-data-1.1.1",
    "in-gauge_en-gage": "in-gauge_en-gage/in-gauge-and-en-gage-understanding-occupants-behaviour-engagement-emotion-and-comfort-indoors-with-heterogeneous-sensors-and-wearables-1.0.0/raw_wearable_data",
    "k_emocon": "K_EmoCon/e4_data",
    "pgg_dalia": "PPG_DaLiA/PPG_FieldStudy",
    "stress_detection_nurses_hospital": "stress_detection_nurses_hospital/Stress_dataset",
    "spd": "SPD/Raw_data",
    "toadstool": "Toadstool/toadstool/participants",
    "ue4w": "ue4w",
    "weee": "WEEE/dataset",
    "wesad": "WESAD",
    "wesd": "WESD/Data",
    "emotibit": "EmotiBit",
}

# Second column of IBI.csv file has two IBI entries
CORRUPTED_FILES = [
    "data/raw_data/unlabelled_data/in-gauge_en-gage/in-gauge-and-en-gage-understanding-occupants-behaviour-engagement-emotion-and-comfort-indoors-with-heterogeneous-sensors-and-wearables-1.0.0/raw_wearable_data/Week4_4/qlwtxd_19",
    "data/raw_data/unlabelled_data/stress_detection_nurses_hospital/Stress_dataset/F5/F5_1594920460",
    "data/raw_data/unlabelled_data/stress_detection_nurses_hospital/Stress_dataset/BG/BG_1607218219",
]

COLLECTIONS_DICT = {
    "barcelona": 0,
    "adarp": 1,
    "big-ideas": 2,
    "in-gauge_en-gage": 3,
    "k_emocon": 4,
    "pgg_dalia": 5,
    "spd": 6,
    "stress_detection_nurses_hospital": 7,
    "toadstool": 8,
    "ue4w": 9,
    "weee": 10,
    "wesad": 11,
    "wesd": 12,
    "emotibit": 13,
}

COLLECTIONS_COLOR_DICT = {
    "barcelona": "forestgreen",
    "adarp": "red",
    "big-ideas": "thistle",
    "in-gauge_en-gage": "skyblue",
    "k_emocon": "gold",
    "pgg_dalia": "grey",
    "spd": "royalblue",
    "stress_detection_nurses_hospital": "rosybrown",
    "toadstool": "lightsalmon",
    "ue4w": "olivedrab",
    "weee": "chocolate",
    "wesad": "fuchsia",
    "wesd": "lawngreen",
}

FLIRT_EDA = [
    "tonic_mean",
    "tonic_std",
    "tonic_min",
    "tonic_max",
    "tonic_ptp",
    "tonic_sum",
    "tonic_energy",
    "tonic_skewness",
    "tonic_kurtosis",
    "tonic_peaks",
    "tonic_rms",
    "tonic_lineintegral",
    "tonic_n_above_mean",
    "tonic_n_below_mean",
    "tonic_n_sign_changes",
    "tonic_iqr",
    "tonic_iqr_5_95",
    "tonic_pct_5",
    "tonic_pct_95",
    "tonic_entropy",
    "tonic_perm_entropy",
    "tonic_svd_entropy",
    "phasic_mean",
    "phasic_std",
    "phasic_min",
    "phasic_max",
    "phasic_ptp",
    "phasic_sum",
    "phasic_energy",
    "phasic_skewness",
    "phasic_kurtosis",
    "phasic_peaks",
    "phasic_rms",
    "phasic_lineintegral",
    "phasic_n_above_mean",
    "phasic_n_below_mean",
    "phasic_n_sign_changes",
    "phasic_iqr",
    "phasic_iqr_5_95",
    "phasic_pct_5",
    "phasic_pct_95",
    "phasic_entropy",
    "phasic_perm_entropy",
    "phasic_svd_entropy",
]

FLIRT_ACC = [
    "acc_x_mean",
    "acc_x_std",
    "acc_x_min",
    "acc_x_max",
    "acc_x_ptp",
    "acc_x_sum",
    "acc_x_energy",
    "acc_x_skewness",
    "acc_x_kurtosis",
    "acc_x_peaks",
    "acc_x_rms",
    "acc_x_lineintegral",
    "acc_x_n_above_mean",
    "acc_x_n_below_mean",
    "acc_x_n_sign_changes",
    "acc_x_iqr",
    "acc_x_iqr_5_95",
    "acc_x_pct_5",
    "acc_x_pct_95",
    "acc_x_entropy",
    "acc_x_perm_entropy",
    "acc_x_svd_entropy",
    "acc_y_mean",
    "acc_y_std",
    "acc_y_min",
    "acc_y_max",
    "acc_y_ptp",
    "acc_y_sum",
    "acc_y_energy",
    "acc_y_skewness",
    "acc_y_kurtosis",
    "acc_y_peaks",
    "acc_y_rms",
    "acc_y_lineintegral",
    "acc_y_n_above_mean",
    "acc_y_n_below_mean",
    "acc_y_n_sign_changes",
    "acc_y_iqr",
    "acc_y_iqr_5_95",
    "acc_y_pct_5",
    "acc_y_pct_95",
    "acc_y_entropy",
    "acc_y_perm_entropy",
    "acc_y_svd_entropy",
    "acc_z_mean",
    "acc_z_std",
    "acc_z_min",
    "acc_z_max",
    "acc_z_ptp",
    "acc_z_sum",
    "acc_z_energy",
    "acc_z_skewness",
    "acc_z_kurtosis",
    "acc_z_peaks",
    "acc_z_rms",
    "acc_z_lineintegral",
    "acc_z_n_above_mean",
    "acc_z_n_below_mean",
    "acc_z_n_sign_changes",
    "acc_z_iqr",
    "acc_z_iqr_5_95",
    "acc_z_pct_5",
    "acc_z_pct_95",
    "acc_z_entropy",
    "acc_z_perm_entropy",
    "acc_z_svd_entropy",
    "l2_mean",
    "l2_std",
    "l2_min",
    "l2_max",
    "l2_ptp",
    "l2_sum",
    "l2_energy",
    "l2_skewness",
    "l2_kurtosis",
    "l2_peaks",
    "l2_rms",
    "l2_lineintegral",
    "l2_n_above_mean",
    "l2_n_below_mean",
    "l2_n_sign_changes",
    "l2_iqr",
    "l2_iqr_5_95",
    "l2_pct_5",
    "l2_pct_95",
    "l2_entropy",
    "l2_perm_entropy",
    "l2_svd_entropy",
]

FLIRT_HRV = [
    "num_ibis",
    "hrv_mean_nni",
    "hrv_median_nni",
    "hrv_range_nni",
    "hrv_sdsd",
    "hrv_rmssd",
    "hrv_nni_50",
    "hrv_pnni_50",
    "hrv_nni_20",
    "hrv_pnni_20",
    "hrv_cvsd",
    "hrv_sdnn",
    "hrv_cvnni",
    "hrv_mean_hr",
    "hrv_min_hr",
    "hrv_max_hr",
    "hrv_std_hr",
    "hrv_total_power",
    "hrv_vlf",
    "hrv_lf",
    "hrv_hf",
    "hrv_lf_hf_ratio",
    "hrv_lfnu",
    "hrv_hfnu",
    "hrv_SD1",
    "hrv_SD2",
    "hrv_SD2SD1",
    "hrv_CSI",
    "hrv_CVI",
    "hrv_CSI_Modified",
    "hrv_mean",
    "hrv_std",
    "hrv_min",
    "hrv_max",
    "hrv_ptp",
    "hrv_sum",
    "hrv_energy",
    "hrv_skewness",
    "hrv_kurtosis",
    "hrv_peaks",
    "hrv_rms",
    "hrv_lineintegral",
    "hrv_n_above_mean",
    "hrv_n_below_mean",
    "hrv_n_sign_changes",
    "hrv_iqr",
    "hrv_iqr_5_95",
    "hrv_pct_5",
    "hrv_pct_95",
    "hrv_entropy",
    "hrv_perm_entropy",
    "hrv_svd_entropy",
]

FLIRT_TEMP = ["temp_avg", "temp_std"]

TRANSFORMATION_NAME_DICT = {
    0: "identity",
    1: "magnitude_warp",
    2: "time_warp",
    3: "permutation",
    4: "cropping",
    5: "gaussian noise",
}

NUM_MASKS = int((10**6) * 1)  # NUM_MASKS = 100

MEDS = [
    "Lithium",
    "SSRI",
    "SNRI",
    "Tryciclics",
    "MAOI",
    "Other_AD",
    "AP_1st",
    "AP_2nd",
    "Anticonvulsants",
    "Beta-blockers",
    "Opioids",
    "Amphetamines",
    "Antihistamines",
    "Antiarrhythmic",
    "Other_medication_with_anticholinergic_effects",
    "BZD",
]
