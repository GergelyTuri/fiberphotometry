import pandas as pd
from scipy.stats import ttest_ind
import numpy as np
from datetime import datetime

def load_and_merge_data(control_mobility_paths, control_serotonin_paths, pcb_mobility_paths, pcb_serotonin_paths):
    def load_data(paths):
        return [pd.read_csv(path) for path in paths]

    control_mobility_dfs = load_data(control_mobility_paths)
    control_serotonin_dfs = load_data(control_serotonin_paths)
    pcb_mobility_dfs = load_data(pcb_mobility_paths)
    pcb_serotonin_dfs = load_data(pcb_serotonin_paths)

    for df in control_serotonin_dfs + pcb_serotonin_dfs:
        df['Time (s)'] = df['Time (s)'].round(2)

    merged_control_dfs = [pd.merge(mob_df, ser_df, on='Time (s)', how='inner')
                          for mob_df, ser_df in zip(control_mobility_dfs, control_serotonin_dfs)]
    merged_pcb_dfs = [pd.merge(mob_df, ser_df, on='Time (s)', how='inner')
                      for mob_df, ser_df in zip(pcb_mobility_dfs, pcb_serotonin_dfs)]

    merged_control_df = pd.concat(merged_control_dfs)
    merged_pcb_df = pd.concat(merged_pcb_dfs)

    return merged_control_df, merged_pcb_df, merged_control_dfs, merged_pcb_dfs


def extract_and_compare_serotonin(merged_control_df, merged_pcb_df):
    control_mobile_serotonin = merged_control_df[merged_control_df['mob'] == 0]['Z-score']
    control_immobile_serotonin = merged_control_df[merged_control_df['mob'] == 1]['Z-score']
    pcb_mobile_serotonin = merged_pcb_df[merged_pcb_df['mob'] == 0]['Z-score']
    pcb_immobile_serotonin = merged_pcb_df[merged_pcb_df['mob'] == 1]['Z-score']

    t_stat_mobile, p_value_mobile = ttest_ind(control_mobile_serotonin, pcb_mobile_serotonin)
    t_stat_immobile, p_value_immobile = ttest_ind(control_immobile_serotonin, pcb_immobile_serotonin)

    print(f"T-test for mobile state: t-statistic = {t_stat_mobile}, p-value = {p_value_mobile}")
    print(f"T-test for immobile state: t-statistic = {t_stat_immobile}, p-value = {p_value_immobile}")

    return (control_mobile_serotonin, control_immobile_serotonin, pcb_mobile_serotonin, pcb_immobile_serotonin)

def calculate_average_serotonin(merged_control_dfs, merged_pcb_dfs):
    def calculate_mean(df, state):
        return df[df['mob'] == state]['Z-score'].mean()

    averages = []
    for control_df, pcb_df in zip(merged_control_dfs, merged_pcb_dfs):
        averages.append({
            'control_mobile': calculate_mean(control_df, 0),
            'control_immobile': calculate_mean(control_df, 1),
            'pcb_mobile': calculate_mean(pcb_df, 0),
            'pcb_immobile': calculate_mean(pcb_df, 1)
        })

    overall_averages = {
        'control_mobile': sum(avg['control_mobile'] for avg in averages) / len(averages),
        'control_immobile': sum(avg['control_immobile'] for avg in averages) / len(averages),
        'pcb_mobile': sum(avg['pcb_mobile'] for avg in averages) / len(averages),
        'pcb_immobile': sum(avg['pcb_immobile'] for avg in averages) / len(averages)
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
    x_coords = beh_df[(beh_df.columns.values[0][0], 'back1', 'x')]
    y_coords = beh_df[(beh_df.columns.values[0][0], 'back1', 'y')]

    # Remove noisy points where both x and y are zero
    valid_points = (x_coords > 0) & (y_coords > 0)
    likelihood_filter = beh_df[('DLC_resnet50_pcb_testJul2shuffle1_1030000', 'back1', 'likelihood')] > likelihood_threshold
    x_coords = x_coords[valid_points & likelihood_filter]
    y_coords = y_coords[valid_points & likelihood_filter]

    # Compute velocity
    dist_moved = np.sqrt(np.diff(x_coords)**2 + np.diff(y_coords)**2)
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
    x_coords = beh_df[(beh_df.columns.values[0][0], 'back1', 'x')]
    y_coords = beh_df[(beh_df.columns.values[0][0], 'back1', 'y')]

    # Remove noisy points where both x and y are zero
    valid_points = (x_coords > 0) & (y_coords > 0)
    likelihood_filter = beh_df[('DLC_resnet50_pcb_testJul2shuffle1_1030000', 'back1', 'likelihood')] > likelihood_threshold
    x_coords = x_coords[valid_points & likelihood_filter]
    y_coords = y_coords[valid_points & likelihood_filter]

    # Compute distance moved
    dist_moved = np.sqrt(np.diff(x_coords)**2 + np.diff(y_coords)**2)
    total_distance = dist_moved.sum()

    return total_distance


def read_block(block_path):
    # Assuming that this function reads the data block from the specified path.
    # Replace this with the actual implementation.
    class Stream:
        def __init__(self, data, fs):
            self.data = data
            self.fs = fs

    block = {
        'streams': {
            '_465A': Stream(np.random.rand(1000), fs=30),  # Example data with sampling frequency
            '_405A': Stream(np.random.rand(1000), fs=30)   # Example data with sampling frequency
        }
    }
    return block

def read_and_process_signal(block_path, serotonin_channel='_465A', isos_channel='_405A'):
    block = read_block(block_path)
    x1 = block['streams'][serotonin_channel].data
    x2 = block['streams'][isos_channel].data

    reg = np.polyfit(x2, x1, 1)

    if reg[0] < 0:
        f0 = np.mean(x1)
    else:
        f0 = reg[0] * x2 + reg[1]

    delF = 100 * (x1 - f0) / f0
    return delF, block

def calculate_zscore_signal(corrected_signal, baseline_end_idx):
    baseline_mean = np.mean(corrected_signal[:baseline_end_idx])
    baseline_sd = np.std(corrected_signal[:baseline_end_idx])
    zscore_signal = (corrected_signal - baseline_mean) / baseline_sd
    return zscore_signal

def process_behavior_data(h5_path):
    beh_df = pd.read_hdf(h5_path)

    x_coords = beh_df[(beh_df.columns.values[0][0], 'back1', 'x')]
    y_coords = beh_df[(beh_df.columns.values[0][0], 'back1', 'y')]

    valid_points = (x_coords > 0) & (y_coords > 0)
    likelihood_filter = beh_df[('DLC_resnet50_pcb_testJul2shuffle1_1030000', 'back1', 'likelihood')] > 0.8
    x_coords = x_coords[valid_points & likelihood_filter]
    y_coords = y_coords[valid_points & likelihood_filter]

    dist_moved = np.sqrt(np.diff(x_coords)**2 + np.diff(y_coords)**2)
    velocity = dist_moved / (1 / 30)

    mean_velocity = velocity.mean()
    total_distance = dist_moved.sum()

    return mean_velocity, total_distance

def load_and_merge_data(velocity_path, serotonin_path):
    velocity_data = pd.read_csv(velocity_path)
    serotonin_data = pd.read_csv(serotonin_path)
    serotonin_data = serotonin_data.iloc[::100, :]
    
    velocity_data.rename(columns={'Adjusted Time (seconds)': 'Time (seconds)'}, inplace=True)
    serotonin_data.rename(columns={'Time (s)': 'Time (seconds)'}, inplace=True)
    
    velocity_data = velocity_data[velocity_data['Smoothed Velocity (cm/s)'] <= 450]
    merged_data = pd.merge_asof(velocity_data, serotonin_data, on='Time (seconds)', direction='nearest')
    merged_data.dropna(inplace=True)

    return merged_data

def preprocess_signal_data(saline_path, pcb_path, serotonin_channel='_465A', isos_channel='_405A'):
    delF_saline, saline_block = read_and_process_signal(saline_path, serotonin_channel, isos_channel)
    delF_pcb, pcb_block = read_and_process_signal(pcb_path, serotonin_channel, isos_channel)

    return delF_saline, saline_block, delF_pcb, pcb_block

def calculate_corrected_signal(delF, block, isos_channel='_405A'):
    corrected_signal = delF - block['streams'][isos_channel].data
    return corrected_signal