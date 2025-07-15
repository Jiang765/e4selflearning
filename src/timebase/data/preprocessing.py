"""
Helper functions to preprocess CSV files
Reference on data export and formatting of Empatica E4 wristband
https://support.empatica.com/hc/en-us/articles/201608896-Data-export-and-formatting-from-E4-connect-
"""


import re
import shutil
import typing as t
import warnings
from datetime import datetime
from datetime import timedelta
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib import concurrent

from timebase.data import utils
from timebase.data.filter_data import scripps_clinic_algorithm
from timebase.data.filter_data import van_hees_algorithm
from timebase.data.static import *
from timebase.utils.utils import get_sequences_boundaries_index

warnings.simplefilter("error", RuntimeWarning)


def read_clinical_info(filename: str):
    """Read clinical EXCEL file"""
    assert os.path.isfile(filename), f"clinical file {filename} does not exists."
    xls = pd.ExcelFile(filename)
    info = pd.read_excel(xls, sheet_name=None)  # read all sheets
    return pd.concat(info)


def split_acceleration(
    channel_data: t.Dict[str, np.ndarray],
    sampling_rates: t.Dict[str, int],
):
    """Split 3D ACC into ACC_x, ACC_y and ACC_z"""
    channel_data["ACC_x"] = channel_data["ACC"][:, 0]
    channel_data["ACC_y"] = channel_data["ACC"][:, 1]
    channel_data["ACC_z"] = channel_data["ACC"][:, 2]
    del channel_data["ACC"]
    sampling_rates["ACC_x"] = sampling_rates["ACC"]
    sampling_rates["ACC_y"] = sampling_rates["ACC"]
    sampling_rates["ACC_z"] = sampling_rates["ACC"]
    del sampling_rates["ACC"]


def load_channel(recording_dir: str, channel: str):
    """Load channel CSV data from file
    Returns
      unix_t0: int, the start time of the recording in UNIX time
      sampling_rate: int, sampling rate of the recording (if exists)
      data: np.ndarray, the raw recording data
    """
    assert channel in CSV_CHANNELS, f"unknown channel {channel}"
    filepath = os.path.join(recording_dir, f"{channel}.csv")
    try:
        if channel == "IBI":
            raw_data = pd.read_csv(filepath, delimiter=",")
        else:
            raw_data = pd.read_csv(filepath, delimiter=",", header=None).values

        unix_t0, sampling_rate, data = None, -1.0, None
        if channel == "IBI":
            unix_t0 = np.float64(raw_data.columns[0])
            data = raw_data.values
        else:
            unix_t0 = raw_data[0] if raw_data.ndim == 1 else raw_data[0, 0]
            sampling_rate = raw_data[1] if raw_data.ndim == 1 else raw_data[1, 0]
            data = raw_data[2:]
        assert sampling_rate.is_integer(), "sampling rate must be an integer"
        data = np.squeeze(data)
        return int(unix_t0), int(sampling_rate), data.astype(np.float32)

    except pd.errors.EmptyDataError:
        # If the file is empty, print a warning and skip it
        print(f"--- [WARNING] File is empty. Skipping. ---\n    File: {filepath}\n-------------------------------------------")
        return np.nan, np.nan, np.nan
        
    except (ValueError, AttributeError, IndexError) as e:
        # Catch the ValueError we encountered before, plus other possible format errors
        # like an IndexError from a file with too few lines, or an AttributeError from wrong data types.
        print(f"--- [ERROR] A format or data conversion error occurred while processing the file. ---")
        print(f"    File: {filepath}")
        print(f"    Error Message: {e}")
        print(f"    This file's content is likely corrupted or not in the standard format. Skipping file.")
        print(f"--------------------------------------------------------------------------------------")
        # Return NaN values to let the main program know this file failed, but without crashing.
        return np.nan, np.nan, np.nan

def preprocess_channel(recording_dir: str, channel: str):
    """
    Load and downsample channel using args.downsampling s.t. each time-step
    corresponds to one second in wall-time
    """
    assert channel in CSV_CHANNELS
    unix_t0, sampling_rate, data = load_channel(
        recording_dir=recording_dir, channel=channel
    )
    # transform to g for acceleration
    if channel == "ACC":
        data = data * 2 / 128
    # despike, apply filter on EDA and TEMP data
    # note: kleckner2018 uses a length of 2057 for a signal sampled at 32Hz,
    # EDA from Empatica E4 is sampled at 4Hz (1/8)
    # if channel == "EDA" or channel == "TEMP":
    #     data = low_pass_filter(recording=data, sampling_rate=sampling_rate)
    if channel not in ("HR", "IBI"):
        # HR begins at t0 + 10s, remove first 10s from channels other than HR
        data = data[sampling_rate * HR_OFFSET :]
    return data, sampling_rate, unix_t0


def sleep_wake_detection(args, t0: int, session_info: t.Dict, channel_data: t.Dict):
    if (args.sleep_algorithm == "scripps_clinic") and (args.wear_minimum_minutes < 30):
        raise TypeError(
            "Scripps Clinic algorithm requires a minimum of 30 " "minutes observed"
        )
    session_info["sampling_rates"]["SLEEP"] = session_info["sampling_rates"]["ACC"]
    session_info["mask_names"].append("SLEEP")

    acc = channel_data["ACC"].copy()
    timestamps = pd.to_datetime(t0, unit="s", origin="unix") + np.arange(
        len(acc)
    ) * timedelta(seconds=session_info["sampling_rates"]["ACC"] ** -1)
    df_acc = pd.DataFrame(data=acc, columns=["acc_x", "acc_y", "acc_z"])
    df_acc = df_acc.set_index(pd.DatetimeIndex(data=timestamps, tz="UTC", name="time"))

    # Empatica E4 samples EDA at 4 Hz, ACC at 32 Hz, thus up-sample no_wear_mask
    # derived from EDA in order to align it to ACC
    upsampled_no_wear_mask = np.reshape(
        channel_data["WEAR"],
        newshape=(-1, session_info["sampling_rates"]["WEAR"]),
        order="C",
    )
    upsampled_no_wear_mask = np.repeat(
        upsampled_no_wear_mask,
        repeats=session_info["sampling_rates"]["ACC"]
        // session_info["sampling_rates"]["WEAR"],
        axis=1,
    )
    upsampled_no_wear_mask = np.reshape(upsampled_no_wear_mask, newshape=-1, order="C")
    indexes = get_sequences_boundaries_index(arr=upsampled_no_wear_mask, value=1)

    match args.sleep_algorithm:
        case "van_hees":
            sleep_wake_extractor = van_hees_algorithm
        case "scripps_clinic":
            sleep_wake_extractor = scripps_clinic_algorithm

    sleep_wake_mask = sleep_wake_extractor(
        indexes=indexes,
        df_acc=df_acc,
        timestamps=timestamps,
        acc_freq=session_info["sampling_rates"]["ACC"],
    )
    channel_data["SLEEP"] = sleep_wake_mask
    mask_labels, values = np.unique(sleep_wake_mask, return_counts=True)
    session_info["seconds_per_status"] = {
        k: (v / session_info["sampling_rates"]["SLEEP"])
        for k, v in zip(mask_labels, values)
    }


def no_wear_detection(args, t0: int, session_info: t.Dict, channel_data: t.Dict):
    # 1) Values of EDA outside the range 0.05 and 100 are considered as
    # invalid. If a sampling cycle, contains any such value the whole
    # sampling cycle is discarded
    # https://archive.arch.ethz.ch/esum/downloads/manuals/emaptics.pdf
    # https://box.empatica.com/documentation/20141119_E4_TechSpecs.pdf
    session_info["sampling_rates"]["WEAR"] = session_info["sampling_rates"]["EDA"]
    session_info["mask_names"].append("WEAR")
    eda = channel_data["EDA"].copy()
    temp = channel_data["TEMP"].copy()

    # wear = 1, no-wear = 0

    mask = np.where(
        (np.logical_or(eda < 0.05, eda > 100) | np.logical_or(temp < 30, temp > 40)),
        True,
        False,
    )
    mask = np.reshape(mask, (-1, session_info["sampling_rates"]["EDA"]))
    mask = np.sum(mask, axis=1)
    mask = np.where(mask > 0, 0, 1)
    mask = np.repeat(mask, session_info["sampling_rates"]["EDA"])
    session_info["no_wear_percentage"] = np.sum(mask == 0) / len(mask)

    # 2) Valid sampling cycles are kept only if they occur as a sequence whose
    # length (in wall-time) is over a given time threshold

    timestamps = pd.to_datetime(t0, unit="s", origin="unix") + np.arange(
        len(eda)
    ) * timedelta(seconds=session_info["sampling_rates"]["EDA"] ** -1)
    indexes = get_sequences_boundaries_index(arr=mask, value=1)
    for i in indexes:
        start, stop = i[0], i[1]
        if (timestamps[stop] - timestamps[start]) < pd.Timedelta(
            minutes=args.wear_minimum_minutes
        ):
            mask[start : stop + 1] = 0

    channel_data["WEAR"] = mask


def preprocess_dir(args, recording_dir: str, labelled: bool):
    """
    Preprocess channels in recording_dir and return the preprocessed features
    and corresponding label obtained from spreadsheet.
    Returns:
      features: np.ndarray, preprocessed channels in SAVE_CHANNELS format
    """
    is_emotibit = "emotibit" in recording_dir

    if is_emotibit:
        # New logic for EmotiBit
        durations, channel_data, sampling_rates, unix_t0s = [], {}, {}, {}

        # Load all specified EmotiBit channels
        for channel in EMOTIBIT_CHANNELS:
            data, sampling_rate, unix_t0 = preprocess_channel_emotibit(
                recording_dir=recording_dir, channel=channel
            )

            # Skip if the file was empty or corrupted
            if isinstance(data, float) and np.isnan(data):
                continue

            channel_data[channel] = data
            sampling_rates[channel] = sampling_rate
            unix_t0s[channel] = unix_t0
            durations.append(len(data) // sampling_rate)

        # Exit if no valid channels were loaded
        if not durations:
            print(f"--- [WARNING] No valid channels found in the EmotiBit directory: {recording_dir}. Skipping directory. ---")
            return {}, {}, True  # Return empty dicts and short_section=True

        session_info = {
            "channel_names": utils.get_channel_names(channel_data),
            "sampling_rates": sampling_rates,
            "unix_t0": unix_t0s,
            "mask_names": [],
            "labelled": labelled,
        }

        # Crop all channels to the shortest duration
        min_duration = min(durations)
        short_section = min_duration < (args.minimum_recorded_time * 60)

        if short_section:
            # Discard the session if it's shorter than the specified minimum
            return channel_data, session_info, short_section
        else:
            for channel, recording in channel_data.items():
                # Crop the data array to the minimum duration
                channel_data[channel] = recording[: min_duration * sampling_rates[channel]]

            # We hardcode it to report that 100% of the valid time (`min_duration`)
            # was spent in the "wake" state (status 0.0).
            session_info["seconds_per_status"] = {0.0: float(min_duration)}
        
            return channel_data, session_info, short_section

    # print(f"processing {session_id} ")
    durations, channel_data, sampling_rates, unix_t0s = [], {}, {}, {}
    # load and preprocess all channels except IBI
    for channel in CSV_CHANNELS:
        if channel != "IBI":
            channel_data[channel], sampling_rate, unix_t0 = preprocess_channel(
                recording_dir=recording_dir, channel=channel
            )
            durations.append(len(channel_data[channel]) // sampling_rate)
            sampling_rates[channel] = sampling_rate
            unix_t0s[channel] = unix_t0
        else:
            channel_data[channel], _, unix_t0 = preprocess_channel(
                recording_dir=recording_dir, channel=channel
            )
            if isinstance(channel_data[channel], float):
                channel_data[channel] = np.array([channel_data[channel]])
            unix_t0s[channel] = unix_t0

    session_info = {
        "channel_names": utils.get_channel_names(channel_data),
        "sampling_rates": sampling_rates,
        "unix_t0": unix_t0s,
        "mask_names": [],
        "labelled": labelled,
    }

    # all channels should have the same durations, but as a failsafe, crop
    # each channel to the shortest duration
    min_duration = min(durations)
    short_section = min_duration < (args.minimum_recorded_time * 60)

    if short_section:
        # drop session is shorter than args.shortest_acceptable_duration minutes
        return channel_data, session_info, short_section
    else:
        for channel, recording in channel_data.items():
            if channel != "IBI":
                channel_data[channel] = recording[
                    : min_duration * sampling_rates[channel]
                ]

        # artifact = 0, valid = 1
        # artifact_mask is sampled at EDA frequency (4Hz)
        # artifact_mask = artifactual_data_removal(
        #     eda=channel_data["EDA"], session_info=session_info
        # )

        # # no-wear = 0, wear = 1
        # # no_wear_mask is sampled at EDA frequency (4Hz)
        # no_wear_detection(
        #     args,
        #     channel_data=channel_data,
        #     t0=unix_t0s["HR"],
        #     session_info=session_info,
        # )
        # # wake = 0, sleep = 1, can't tell = 2
        # # sleep_wake_mask is sampled at ACC frequency (32Hz)
        # sleep_wake_detection(
        #     args,
        #     channel_data=channel_data,
        #     t0=unix_t0s["HR"],
        #     session_info=session_info,
        # )

        # We hardcode it to report that 100% of the valid time (`min_duration`)
        # was spent in the "wake" state (status 0.0).
        session_info["seconds_per_status"] = {0.0: float(min_duration)}

        split_acceleration(channel_data=channel_data, sampling_rates=sampling_rates)

        return channel_data, session_info, short_section


def get_channel_from_filename(filename: str, channels: t.List):
    # Create a regular expression that matches any of the strings in `strings`.
    regex = re.compile("|".join(channels))

    # Check whether the regular expression matches the string.
    match = regex.search(filename)

    # If the regular expression matches the string, return the matched string.
    if match:
        return match.group()
    else:
        return None


def reformat_kemocon_csv(filename: str, output_file: str):
    channel = get_channel_from_filename(filename=filename, channels=CSV_CHANNELS)
    df = pd.read_csv(filename)
    if channel in ("BVP", "EDA", "HR", "TEMP"):
        data = np.insert(
            np.float32(df["value"].values),
            0,
            np.float32(CHANNELS_FREQ[channel]),
        )
        data = pd.DataFrame(
            data=data, columns=[str(np.float64(df["timestamp"][0]) / 1000)]
        )
    elif channel == "ACC":
        data = np.concatenate(
            [
                np.insert(
                    np.float32(df["x"].values),
                    0,
                    np.float32(CHANNELS_FREQ[f"{channel}_{axis}"]),
                )[..., np.newaxis]
                for axis in ["x", "y", "z"]
            ],
            axis=1,
        )
        data = pd.DataFrame(
            data=data,
            columns=[str(np.float64(df["timestamp"][0]) / 1000)] * data.shape[1],
        )

    else:
        t0 = pd.read_csv(filename.replace("IBI", "EDA"))["timestamp"][0]
        col1 = np.array((df["timestamp"] - t0) / 1000).astype(np.float64)[
            ..., np.newaxis
        ]
        col2 = (df["value"].astype(np.float64).values / 1000)[..., np.newaxis]
        data = np.concatenate([col1, col2], axis=1)
        data = pd.DataFrame(
            data=data,
            columns=[str(np.float64(t0) / 1000), "IBI"],
        )
    data.to_csv(output_file, index=False)


def reformat_in_gauge_en_gage(filename: str, output_file: str):
    channel = get_channel_from_filename(filename=filename, channels=CSV_CHANNELS)
    df = pd.read_csv(filename)
    # Recording ID (specifically "%Y-%m-%d) t0 may be incorrect but this is not
    # relevant for the current analysis
    match = re.search(r"Week(\d_\d)", filename)
    week_number, day_of_week = int(match.group()[-3]), int(match.group()[-1])
    month_num = datetime.strptime("September", "%B").month
    first_day_of_month = datetime(2019, month_num, 1)
    first_weekday = first_day_of_month.weekday()
    days_to_add = (week_number - 1) * 7 + (day_of_week - first_weekday) % 7
    target_date = (first_day_of_month + timedelta(days=days_to_add)).strftime(
        "%Y-%m-%d"
    )
    t0 = str(
        datetime.strptime(
            f"{target_date} " + df.columns[0], "%Y-%m-%d %H:%M:%S"
        ).timestamp()
    )

    if channel in ("BVP", "EDA", "HR", "TEMP"):
        cols = [t0]
    elif channel == "ACC":
        cols = [t0] * 3
    else:
        cols = [t0, "IBI"]
    data = pd.DataFrame(data=df.values, columns=cols)
    data.to_csv(output_file, index=False)


def reformat_big_ideas(filename: str, output_file: str):
    channel = get_channel_from_filename(filename=filename, channels=CSV_CHANNELS)
    df = pd.read_csv(filename)
    date_format = "%Y-%m-%d %H:%M:%S.%f" if channel != "HR" else "%Y-%m-%d %H:%M:%S"
    date_obj = datetime.strptime(df["datetime"][0], date_format)
    t0 = int(date_obj.timestamp())

    if channel in ("BVP", "EDA", "HR", "TEMP"):
        data = np.insert(
            np.float32(df.iloc[:, 1:].values),
            0,
            np.float32(CHANNELS_FREQ[channel]),
        )
        data = pd.DataFrame(data=data, columns=[t0])
    elif channel == "ACC":
        data = np.concatenate(
            (np.float32(np.array([[32, 32, 32]])), np.float32(df.iloc[:, 1:].values)),
            axis=0,
        )
        data = pd.DataFrame(
            data=data,
            columns=[t0] * 3,
        )

    else:
        timestamps = np.array(
            [
                datetime.strptime(timestamp, "%Y-%m-%d " "%H:%M:%S.%f").timestamp()
                for timestamp in df["datetime"]
            ]
        )
        col1 = np.array(timestamps[1:] - t0).astype(np.float64)[..., np.newaxis]
        col2 = (df.iloc[1:, 1].astype(np.float64).values)[..., np.newaxis]

        data = np.concatenate([col1, col2], axis=1)
        data = pd.DataFrame(
            data=data,
            columns=[t0, "IBI"],
        )
    data.to_csv(output_file, index=False)


REFORMAT_COLLECTION_DICT = {
    "k_emocon": reformat_kemocon_csv,
    "in-gauge_en-gage": reformat_in_gauge_en_gage,
    "big-ideas": reformat_big_ideas,
}


def check_faulty_folder(dirpath: str, files2dismiss: t.List):
    files = []
    num_empty = 0
    unique_channels = []
    for f in os.listdir(dirpath):
        # TODO avoid for loop
        for c in CSV_CHANNELS:
            if c in f:
                files.append(os.path.join(dirpath, f))
                unique_channels.append(c)
                if not os.path.getsize(os.path.join(dirpath, f)):
                    num_empty += 1
    # files within a folder are removed from analyses if 1) there are more than
    # len(CSV_CHANNELS) channel files or 2) at least one channel file is empty
    if (len(unique_channels) > len(CSV_CHANNELS)) or (num_empty > 0):
        files2dismiss.extend(files)


def recast_collection(args, collection: str, path: str):
    dirs = []
    output_dir_collection = os.path.join(
        "data/raw_data/unlabelled_data/recast/", collection
    )
    root_dir = os.path.join(args.path2unlabelled_data, path)

    if collection == "emotibit":
        # tqdm.write(f">>> Running recast for EmotiBit collection...")

        # Automatically create a source-to-target file mapping based on the sample rate dictionary.
        # Ignore signals with a sample rate of -1 (e.g., BI, HR, SA, SR).
        source_to_target_map = {
            tag: f"{tag}.csv"
            for tag, rate in EMOTIBIT_NOMINAL_SAMPLE_RATES.items()
            if rate != -1
        }

        for dirpath, dirnames, _ in os.walk(root_dir):
            # EmotiBit data is typically in a specific subfolder.
            subfolder_name = "separated by signal with EmotiBitDataParserApp"
            if subfolder_name in dirnames:
                
                session_dir = dirpath
                source_data_dir = os.path.join(session_dir, subfolder_name)
                
                relative_path = os.path.relpath(session_dir, root_dir)
                output_dir_session = os.path.join(output_dir_collection, relative_path)
                os.makedirs(output_dir_session, exist_ok=True)
                dirs.append(output_dir_session)
                
                # tqdm.write(f"Processing session: {os.path.basename(session_dir)}")

                # The file prefix is based on the session directory name.
                prefix = os.path.basename(session_dir)

                # Iterate through the dynamically generated map to process all matching files.
                for source_tag, target_filename in source_to_target_map.items():
                    source_filename = f"{prefix}_{source_tag}.csv"
                    input_path = os.path.join(source_data_dir, source_filename)

                    if os.path.exists(input_path):
                        output_path = os.path.join(output_dir_session, target_filename)
                        # Call the reformatting function.
                        reformat_csv_emotibit(input_path, output_path)

        # tqdm.write(f">>> Complete recast for EmotiBit collection")
        return list(set(dirs))

    files2dismiss = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".csv") and get_channel_from_filename(
                filename, CSV_CHANNELS
            ):
                if dirpath in CORRUPTED_FILES:
                    continue
                check_faulty_folder(dirpath=dirpath, files2dismiss=files2dismiss)
                if os.path.join(dirpath, filename) in files2dismiss:
                    continue
                output_file = os.path.join(
                    output_dir_collection,
                    dirpath.rsplit(f"{path}/", 1)[-1],
                    f"{get_channel_from_filename(filename, CSV_CHANNELS)}.csv",
                ).replace(" ", "_")
                if not os.path.exists(output_file.rsplit("/", 1)[0]):
                    os.makedirs(output_file.rsplit("/", 1)[0])
                    dirs.append(output_file.rsplit("/", 1)[0])
                if collection in REFORMAT_COLLECTION_DICT.keys():
                    REFORMAT_COLLECTION_DICT[collection](
                        filename=os.path.join(dirpath, filename),
                        output_file=output_file,
                    )
                else:
                    shutil.copyfile(os.path.join(dirpath, filename), output_file)
    return dirs


def recast_wrapper(args, output_dir, collection, path):
    output_dir_collection = os.path.join(output_dir, collection)
    if os.path.isdir(output_dir_collection):
        if args.overwrite:
            shutil.rmtree(output_dir_collection)
        else:
            raise FileExistsError(
                f"output_dir {output_dir_collection} already exists. Add --overwrite "
                f" flag to overwrite the existing preprocessed data."
            )
    os.makedirs(output_dir_collection)
    dirs = recast_collection(args, collection=collection, path=path)
    return dirs


def recast_unlabelled_data(args):
    output_dir = os.path.join("data/raw_data/unlabelled_data/recast/")
    if os.path.isdir(output_dir):
        if args.overwrite:
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(
                f"output_dir {output_dir} already exists. Add --overwrite "
                f" flag to overwrite the existing preprocessed data."
            )
    os.makedirs(output_dir)
    dirs_collector = []
    res = concurrent.process_map(
        partial(recast_wrapper, args, output_dir),
        UNLABELLED_DATA_PATHS.keys(),
        UNLABELLED_DATA_PATHS.values(),
        max_workers=args.num_workers,
        chunksize=1,
        desc="Recasting unlabelled data",
    )
    for i in range(len(UNLABELLED_DATA_PATHS)):
        # for i, (collection, path) in tqdm(enumerate(UNLABELLED_DATA_PATHS.items())):
        #     output_dir_collection = os.path.join(output_dir, collection)
        #     if os.path.isdir(output_dir_collection):
        #         if args.overwrite:
        #             shutil.rmtree(output_dir_collection)
        #         else:
        #             raise FileExistsError(
        #                 f"output_dir {output_dir_collection} already exists. Add --overwrite "
        #                 f" flag to overwrite the existing preprocessed data."
        #             )
        #     os.makedirs(output_dir_collection)
        #     dirs = recast_collection(args, collection=collection, path=path)
        dirs = res[i]
        if len(dirs):
            dirs_collector.extend(dirs)

    return dirs_collector


def reformat_csv_emotibit(filename: str, output_file: str):
    """
    Converts a single EmotiBit CSV file to the project's required standard format.

    The standard format is a single-column CSV file:
    - Row 1: The initial LocalTimestamp (t0).
    - Row 2: The signal's sampling rate in Hz.
    - Row 3 and onwards: The continuous signal data points.

    Args:
        filename (str): The path to the input raw EmotiBit CSV file.
        output_file (str): The path for the output standardized CSV file.
    """
    try:
        # 1. Read the raw CSV file.
        df_raw = pd.read_csv(filename)

        # If the file is empty or has too few rows, skip it.
        if len(df_raw) < 2:
            print(f"Warning: File '{os.path.basename(filename)}' is too short or empty. Skipping.")
            return

        # 2. Extract key information.
        # Get the initial timestamp from the first row.
        t0 = df_raw['LocalTimestamp'].iloc[0]
        # Get the signal type from the 'TypeTag' column of the first row;
        # this is also used as the data column's name.
        data_col_name = df_raw['TypeTag'].iloc[0]

        # Confirm that the data column actually exists.
        if data_col_name not in df_raw.columns:
            print(f"Error: Data column '{data_col_name}' not found in file '{filename}'.")
            return

        # 3. Determine the sampling rate.
        # First, try to get the nominal sample rate from the static dictionary.
        sampling_rate = EMOTIBIT_NOMINAL_SAMPLE_RATES.get(data_col_name)

        # If it's not in the dictionary or is marked as -1, calculate it dynamically.
        if sampling_rate is None or sampling_rate == -1:
            num_samples = len(df_raw)
            # Calculate the total duration.
            time_delta = df_raw['LocalTimestamp'].iloc[-1] - df_raw['LocalTimestamp'].iloc[0]
            
            if time_delta > 0:
                # Sample Rate = (Number of Samples - 1) / Total Duration
                calculated_rate = (num_samples - 1) / time_delta
                sampling_rate = int(round(calculated_rate))
                print(f"Info: Calculated sampling rate for {data_col_name}: {sampling_rate} Hz")
            else:
                # If the time delta is zero, we can't calculate a rate, so default to 1.
                sampling_rate = 1

        # 4. Extract the signal data.
        data_values = df_raw[data_col_name]

        # 5. Assemble the final single-column list.
        final_values = [t0, sampling_rate]
        final_values.extend(data_values.tolist())

        # 6. Create the final DataFrame and save it.
        df_standard = pd.DataFrame(final_values)

        # Save to CSV without an index or header.
        df_standard.to_csv(output_file, index=False, header=False)

    except (pd.errors.EmptyDataError, IndexError):
        print(f"Warning: File '{filename}' is empty or has incorrect format. Skipping.")
    except FileNotFoundError:
        print(f"Error: File not found at '{filename}'. Skipping.")
    except Exception as e:
        print(f"An unexpected error occurred while processing '{filename}': {e}")


def load_channel_emotibit(recording_dir: str, channel: str):
    """
    Load pre-reformatted EmotiBit channel CSV data from a file.
    The format is a single-column CSV:
    - Row 1: Initial Unix timestamp (t0)
    - Row 2: Signal's sampling rate (Hz)
    - Row 3 and onwards: Continuous signal data points
    """
    filepath = os.path.join(recording_dir, f"{channel}.csv")
    try:
        # Data is a single column: t0, sampling_rate, data...
        raw_data = pd.read_csv(filepath, header=None).values

        if raw_data.shape[0] < 3:  # At least t0, rate, and one data point are needed
            raise ValueError("File has insufficient data rows.")

        unix_t0 = raw_data[0, 0]
        sampling_rate = raw_data[1, 0]
        data = raw_data[2:]

        # Basic validation
        assert np.issubdtype(type(unix_t0), np.number), "t0 must be a number."
        assert np.issubdtype(type(sampling_rate), np.number) and sampling_rate > 0, "Sampling rate must be a positive number."

        data = np.squeeze(data)
        return int(unix_t0), int(sampling_rate), data.astype(np.float32)

    except (pd.errors.EmptyDataError, FileNotFoundError):
        print(f"--- [WARNING] File not found or is empty. Skipping. ---\n    File: {filepath}\n-------------------------------------------")
        return np.nan, np.nan, np.nan

    except (ValueError, AttributeError, IndexError, AssertionError) as e:
        print(f"--- [ERROR] A format or data conversion error occurred while processing the EmotiBit file. ---")
        print(f"    File: {filepath}")
        print(f"    Error Message: {e}")
        print(f"    Skipping this file.")
        print(f"--------------------------------------------------------------------------------------")
        return np.nan, np.nan, np.nan


def preprocess_channel_emotibit(recording_dir: str, channel: str):
    """Load an EmotiBit channel and perform basic preprocessing."""
    assert channel in EMOTIBIT_CHANNELS, f"Unknown EmotiBit channel: {channel}"

    unix_t0, sampling_rate, data = load_channel_emotibit(
        recording_dir=recording_dir, channel=channel
    )

    # No specific preprocessing like scaling or filtering is needed for now.
    # The HR_OFFSET logic from the original function is specific to Empatica E4 and is not applicable here.

    return data, sampling_rate, unix_t0
