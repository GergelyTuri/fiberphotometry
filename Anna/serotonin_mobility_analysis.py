import numpy as np
import pandas as pd
from scipy.stats import ttest_ind


def load_and_merge_data(
    control_mobility_paths,
    control_serotonin_paths,
    pcb_mobility_paths,
    pcb_serotonin_paths,
):
    """
    Loads and merges mobility and serotonin data from given file paths.
    Parameters:
    control_mobility_paths (list of str): List of file paths for control mobility data CSV files.
    control_serotonin_paths (list of str): List of file paths for control serotonin data CSV files.
    pcb_mobility_paths (list of str): List of file paths for PCB mobility data CSV files.
    pcb_serotonin_paths (list of str): List of file paths for PCB serotonin data CSV files.
    Returns:
    tuple: A tuple containing:
        - merged_control_df (pd.DataFrame): Merged DataFrame of control mobility and serotonin data.
        - merged_pcb_df (pd.DataFrame): Merged DataFrame of PCB mobility and serotonin data.
        - merged_control_dfs (list of pd.DataFrame): List of individual merged DataFrames for control data.
        - merged_pcb_dfs (list of pd.DataFrame): List of individual merged DataFrames for PCB data.
    """

    def load_data(paths):
        return [pd.read_csv(path) for path in paths]

    control_mobility_dfs = load_data(control_mobility_paths)
    control_serotonin_dfs = load_data(control_serotonin_paths)
    pcb_mobility_dfs = load_data(pcb_mobility_paths)
    pcb_serotonin_dfs = load_data(pcb_serotonin_paths)

    for df in control_serotonin_dfs + pcb_serotonin_dfs:
        df["Time (s)"] = df["Time (s)"].round(2)

    merged_control_dfs = [
        pd.merge(mob_df, ser_df, on="Time (s)", how="inner")
        for mob_df, ser_df in zip(control_mobility_dfs, control_serotonin_dfs)
    ]
    merged_pcb_dfs = [
        pd.merge(mob_df, ser_df, on="Time (s)", how="inner")
        for mob_df, ser_df in zip(pcb_mobility_dfs, pcb_serotonin_dfs)
    ]

    merged_control_df = pd.concat(merged_control_dfs)
    merged_pcb_df = pd.concat(merged_pcb_dfs)

    return merged_control_df, merged_pcb_df, merged_control_dfs, merged_pcb_dfs


def extract_and_compare_serotonin(merged_control_df, merged_pcb_df):
    """
    Extracts serotonin Z-scores for mobile and immobile states from control and PCB dataframes,
    performs independent t-tests between control and PCB groups for both states, and prints the results.
    Parameters:
    merged_control_df (pd.DataFrame): DataFrame containing control group data with 'mob' and 'Z-score' columns.
    merged_pcb_df (pd.DataFrame): DataFrame containing PCB group data with 'mob' and 'Z-score' columns.
    Returns:
    tuple: A tuple containing four pandas Series:
        - control_mobile_serotonin (pd.Series): Z-scores for mobile state in control group.
        - control_immobile_serotonin (pd.Series): Z-scores for immobile state in control group.
        - pcb_mobile_serotonin (pd.Series): Z-scores for mobile state in PCB group.
        - pcb_immobile_serotonin (pd.Series): Z-scores for immobile state in PCB group.
    """
    control_mobile_serotonin = merged_control_df[merged_control_df["mob"] == 0][
        "Z-score"
    ]
    control_immobile_serotonin = merged_control_df[merged_control_df["mob"] == 1][
        "Z-score"
    ]
    pcb_mobile_serotonin = merged_pcb_df[merged_pcb_df["mob"] == 0]["Z-score"]
    pcb_immobile_serotonin = merged_pcb_df[merged_pcb_df["mob"] == 1]["Z-score"]

    t_stat_mobile, p_value_mobile = ttest_ind(
        control_mobile_serotonin, pcb_mobile_serotonin
    )
    t_stat_immobile, p_value_immobile = ttest_ind(
        control_immobile_serotonin, pcb_immobile_serotonin
    )

    print(
        f"T-test for mobile state: t-statistic = {t_stat_mobile}, p-value = {p_value_mobile}"
    )
    print(
        f"T-test for immobile state: t-statistic = {t_stat_immobile}, p-value = {p_value_immobile}"
    )

    return (
        control_mobile_serotonin,
        control_immobile_serotonin,
        pcb_mobile_serotonin,
        pcb_immobile_serotonin,
    )


def calculate_average_serotonin(merged_control_dfs, merged_pcb_dfs):
    """
    Calculate the average serotonin Z-scores for mobile and immobile states in control and PCB groups.
    Args:
        merged_control_dfs (list of pandas.DataFrame): List of DataFrames containing control group data.
        merged_pcb_dfs (list of pandas.DataFrame): List of DataFrames containing PCB group data.
    Returns:
        tuple: A tuple containing:
            - averages (list of dict): List of dictionaries with average Z-scores for each DataFrame pair.
            - overall_averages (dict): Dictionary with overall average Z-scores across all DataFrame pairs.
    """

    def calculate_mean(df, state):
        return df[df["mob"] == state]["Z-score"].mean()

    averages = []
    for control_df, pcb_df in zip(merged_control_dfs, merged_pcb_dfs):
        averages.append(
            {
                "control_mobile": calculate_mean(control_df, 0),
                "control_immobile": calculate_mean(control_df, 1),
                "pcb_mobile": calculate_mean(pcb_df, 0),
                "pcb_immobile": calculate_mean(pcb_df, 1),
            }
        )

    overall_averages = {
        "control_mobile": sum(avg["control_mobile"] for avg in averages)
        / len(averages),
        "control_immobile": sum(avg["control_immobile"] for avg in averages)
        / len(averages),
        "pcb_mobile": sum(avg["pcb_mobile"] for avg in averages) / len(averages),
        "pcb_immobile": sum(avg["pcb_immobile"] for avg in averages) / len(averages),
    }

    return averages, overall_averages


def load_and_process_file(file_path, frame_rate=30, likelihood_threshold=0.8):
    """
    Load and process HDF file to calculate mean velocity.

    Parameters:
    ----------
    file_path : str
        Path to the HDF file.
    frame_rate : int, optional
        Frame rate of the video (default is 30 fps).
    likelihood_threshold : float, optional
        Threshold for likelihood to filter out low-confidence points (default is 0.8).

    Returns:
    -------
    float
        Mean velocity calculated from the file.
    """
    beh_df = pd.read_hdf(file_path)

    # Extract coordinates
    x_coords = beh_df[(beh_df.columns.values[0][0], "back1", "x")]
    y_coords = beh_df[(beh_df.columns.values[0][0], "back1", "y")]

    # Remove noisy points where both x and y are zero
    valid_points = (x_coords > 0) & (y_coords > 0)
    likelihood_filter = (
        beh_df[("DLC_resnet50_pcb_testJul2shuffle1_1030000", "back1", "likelihood")]
        > likelihood_threshold
    )
    x_coords = x_coords[valid_points & likelihood_filter]
    y_coords = y_coords[valid_points & likelihood_filter]

    # Compute velocity
    dist_moved = np.sqrt(np.diff(x_coords) ** 2 + np.diff(y_coords) ** 2)
    velocity = dist_moved / (1 / frame_rate)

    # Calculate mean velocity
    mean_velocity = velocity.mean()

    return mean_velocity


def calculate_total_distance(file_path, likelihood_threshold=0.8):
    """
    Load and process HDF file to calculate total distance traveled.

    Parameters:
    ----------
    file_path : str
        Path to the HDF file.
    likelihood_threshold : float, optional
        Threshold for likelihood to filter out low-confidence points (default is 0.8).

    Returns:
    -------
    float
        Total distance traveled calculated from the file.
    """
    beh_df = pd.read_hdf(file_path)

    # Extract coordinates
    x_coords = beh_df[(beh_df.columns.values[0][0], "back1", "x")]
    y_coords = beh_df[(beh_df.columns.values[0][0], "back1", "y")]

    # Remove noisy points where both x and y are zero
    valid_points = (x_coords > 0) & (y_coords > 0)
    likelihood_filter = (
        beh_df[("DLC_resnet50_pcb_testJul2shuffle1_1030000", "back1", "likelihood")]
        > likelihood_threshold
    )
    x_coords = x_coords[valid_points & likelihood_filter]
    y_coords = y_coords[valid_points & likelihood_filter]

    # Compute distance moved
    dist_moved = np.sqrt(np.diff(x_coords) ** 2 + np.diff(y_coords) ** 2)
    total_distance = dist_moved.sum()

    return total_distance

    return total_distance
