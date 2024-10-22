import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from tdt import read_block


def load_block_data(block_path):
    return read_block(block_path)


def process_fluorescence_data(block, serotonin_channel="_465A", isos_channel="_405A"):
    """
    Processes fluorescence data from a given block and computes the delta F/F.
    Args:
        block (dict): A dictionary containing the data streams.
        serotonin_channel (str, optional): The key for the serotonin channel data in the block. Defaults to '_465A'.
        isos_channel (str, optional): The key for the isosbestic channel data in the block. Defaults to '_405A'.
    Returns:
        tuple: A tuple containing:
            - delF (numpy.ndarray): The delta F/F values.
            - block (dict): The original block dictionary.
    """
    x1 = block["streams"][serotonin_channel].data
    x2 = block["streams"][isos_channel].data

    reg = np.polyfit(x2, x1, 1)
    f0 = reg[0] * x2 + reg[1] if reg[0] >= 0 else np.mean(x1)
    delF = 100 * (x1 - f0) / f0

    return delF, block


def calculate_zscore(corrected_signal, baseline_end_sec, t, fs):
    """
    Calculate the z-score of a corrected signal based on a baseline period.
    Parameters:
    corrected_signal (numpy.ndarray): The corrected signal data.
    baseline_end_sec (float): The end time of the baseline period in seconds.
    t (float): The start time for the analysis in seconds.
    fs (float): The sampling frequency of the signal.
    Returns:
    tuple: A tuple containing:
        - time_x (numpy.ndarray): The time vector starting from time t.
        - zscore_signal (numpy.ndarray): The z-scored signal.
        - baseline_end_idx (int): The index in the time vector where the baseline period ends.
    """
    time_x = np.linspace(1, len(corrected_signal), len(corrected_signal)) / fs
    inds = np.where(time_x > t)
    ind = inds[0][0]
    time_x = time_x[ind:]
    corrected_signal = corrected_signal[ind:]

    baseline_end_idx = np.where(time_x > (baseline_end_sec - t))[0][0]
    baseline_mean = np.mean(corrected_signal[:baseline_end_idx])
    baseline_sd = np.std(corrected_signal[:baseline_end_idx])
    zscore_signal = (corrected_signal - baseline_mean) / baseline_sd

    return time_x, zscore_signal, baseline_end_idx


def save_to_csv(data, filename):
    pd.DataFrame(data, columns=["delF"]).to_csv(filename, index=False)


def load_and_process_behavior_data(h5_path):
    # TODO: add an argument called 'scorer' which takes out the scorer value from the
    # dlc .h5 file. This is because the scorer value is different for each experiment.
    """
    Load and process behavior data from an HDF5 file.
    This function reads behavior data from the specified HDF5 file, filters the data based on
    valid coordinates and likelihood, and calculates the mean velocity and total distance moved.
    Parameters:
    h5_path (str): The file path to the HDF5 file containing the behavior data.
    Returns:
    tuple: A tuple containing:
        - mean_velocity (float): The mean velocity of the subject.
        - total_distance (float): The total distance moved by the subject.
    """
    beh_df = pd.read_hdf(h5_path)

    x_coords = beh_df[(beh_df.columns.values[0][0], "back1", "x")]
    y_coords = beh_df[(beh_df.columns.values[0][0], "back1", "y")]

    valid_points = (x_coords > 0) & (y_coords > 0)
    likelihood_filter = (
        beh_df[("DLC_resnet50_pcb_testJul2shuffle1_1030000", "back1", "likelihood")]
        > 0.8
    )
    x_coords = x_coords[valid_points & likelihood_filter]
    y_coords = y_coords[valid_points & likelihood_filter]

    dist_moved = np.sqrt(np.diff(x_coords) ** 2 + np.diff(y_coords) ** 2)
    velocity = dist_moved / (1 / 30)

    mean_velocity = velocity.mean()
    total_distance = dist_moved.sum()

    return mean_velocity, total_distance


def preprocess_velocity_data(velocity_path, serotonin_path):
    """
    Preprocesses velocity and serotonin data by reading CSV files, renaming columns,
    filtering, and merging the data.
    Args:
        velocity_path (str): The file path to the velocity data CSV.
        serotonin_path (str): The file path to the serotonin data CSV.
    Returns:
        pd.DataFrame: A merged DataFrame containing the preprocessed velocity and serotonin data.
    """
    velocity_data = pd.read_csv(velocity_path)
    serotonin_data = pd.read_csv(serotonin_path)
    serotonin_data = serotonin_data.iloc[::100, :]

    velocity_data.rename(
        columns={"Adjusted Time (seconds)": "Time (seconds)"}, inplace=True
    )
    serotonin_data.rename(columns={"Time (s)": "Time (seconds)"}, inplace=True)

    velocity_data = velocity_data[velocity_data["Smoothed Velocity (cm/s)"] <= 450]
    merged_data = pd.merge_asof(
        velocity_data, serotonin_data, on="Time (seconds)", direction="nearest"
    )
    merged_data.dropna(inplace=True)

    return merged_data


def gaussian_smooth(data, sigma=2):
    return gaussian_filter1d(data, sigma)


def plot_velocity(velocity_data, sigma=2):
    smoothed_velocity = gaussian_smooth(velocity_data["Velocity (cm/s)"].values, sigma)
    plt.figure(figsize=(12, 6))
    plt.plot(
        velocity_data["Time (seconds)"],
        velocity_data["Velocity (cm/s)"],
        "k",
        label="Original Data",
    )
    plt.plot(
        velocity_data["Time (seconds)"],
        smoothed_velocity,
        "--",
        label=f"Filtered, sigma={sigma}",
    )
    plt.xlabel("Time (seconds)")
    plt.ylabel("Velocity (cm/s)")
    plt.title("Original and Smoothed Velocity Data")
    plt.legend()
    plt.grid()
    plt.show()
    return smoothed_velocity
