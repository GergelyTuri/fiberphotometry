from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def time_to_seconds(time_str: str) -> float:
    """
    Convert time strings to seconds.

    Parameters:
    ----------
    time_str : str
        Time string in the format '%I:%M:%S%p'

    Returns:
    -------
    float
        Time in seconds.
    """
    t = datetime.strptime(time_str, "%I:%M:%S%p")
    return timedelta(hours=t.hour, minutes=t.minute, seconds=t.second).total_seconds()


def process_behavior_data(
    file_path: str, start_time_str: str, injection_time_str: str
) -> tuple:
    """
    Load and process behavior data from a CSV file.

    Parameters:
    ----------
    file_path : str
        Path to the CSV file.
    start_time_str : str
        Start time as a string in the format '%I:%M:%S%p'.
    injection_time_str : str
        Injection time as a string in the format '%I:%M:%S%p'.

    Returns:
    -------
    tuple
        baseline_behavior_df: pd.DataFrame
            DataFrame containing the baseline behavior data.
        post_injection_behavior_df: pd.DataFrame
            DataFrame containing the post-injection behavior data.
        baseline_duration: float
            Duration of the baseline period in seconds.
        post_injection_duration: float
            Duration of the post-injection period in seconds.
    """
    behavior_df = pd.read_csv(file_path)
    start_time = time_to_seconds(start_time_str)
    injection_time = time_to_seconds(injection_time_str)

    behavior_df["Time (s)"] = behavior_df["Time (s)"] + start_time
    baseline_behavior_df = behavior_df[behavior_df["Time (s)"] < injection_time]
    post_injection_behavior_df = behavior_df[
        behavior_df["Time (s)"] >= injection_time
    ].copy()

    post_injection_behavior_df.loc[:, "Time (s)"] = (
        post_injection_behavior_df["Time (s)"] - injection_time
    )
    baseline_duration = injection_time - start_time
    post_injection_duration = behavior_df["Time (s)"].max() - injection_time

    return (
        baseline_behavior_df,
        post_injection_behavior_df,
        baseline_duration,
        post_injection_duration,
    )


def calculate_behavior_frequencies(
    behavior_df: pd.DataFrame, total_duration: float
) -> dict:
    """
    Calculate behavior frequencies per second.

    Parameters:
    ----------
    behavior_df : pd.DataFrame
        DataFrame containing the behavior data.
    total_duration : float
        Total duration of the period in seconds.

    Returns:
    -------
    dict
        Dictionary containing the frequencies of each behavior.
    """
    behavior_counts = behavior_df["Behavior"].value_counts()
    behavior_frequencies = {
        behavior: count / total_duration for behavior, count in behavior_counts.items()
    }
    return behavior_frequencies


def ensure_all_behaviors(frequencies: dict, behavior_labels: dict) -> None:
    """
    Ensure all behaviors are in the dictionaries, even if their frequency is zero.

    Parameters:
    ----------
    frequencies : dict
        Dictionary containing the frequencies of each behavior.
    behavior_labels : dict
        Dictionary containing the behavior labels.
    """
    for key in behavior_labels.keys():
        if key not in frequencies:
            frequencies[key] = 0


def calculate_z_score_differences(
    baseline_frequencies: dict, post_injection_frequencies: dict
) -> dict:
    """
    Calculate the z-score differences between baseline and post-injection frequencies for given behaviors.
    This function computes the z-scores for both baseline and post-injection frequencies, and then calculates
    the differences in z-scores for each behavior.
    Args:
        baseline_frequencies (dict): A dictionary where keys are behavior names and values are their corresponding
                                     frequencies during the baseline period.
        post_injection_frequencies (dict): A dictionary where keys are behavior names and values are their corresponding
                                           frequencies after injection.
    Returns:
        dict: A dictionary where keys are behavior names and values are the differences in z-scores between
              post-injection and baseline frequencies.
    """
    all_frequencies = list(baseline_frequencies.values()) + list(
        post_injection_frequencies.values()
    )
    mean_freq = np.mean(all_frequencies)
    std_freq = np.std(all_frequencies)

    z_baseline_frequencies = {
        behavior: (freq - mean_freq) / std_freq
        for behavior, freq in baseline_frequencies.items()
    }
    z_post_injection_frequencies = {
        behavior: (freq - mean_freq) / std_freq
        for behavior, freq in post_injection_frequencies.items()
    }

    z_score_differences = {
        behavior: z_post_injection_frequencies[behavior]
        - z_baseline_frequencies[behavior]
        for behavior in baseline_frequencies.keys()
    }
    return z_score_differences

    return z_score_differences
