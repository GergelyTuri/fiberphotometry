"""
dlc_alignment.py
────────────────
Functions for aligning DLC tracking / hand-scored behavior CSVs
to true TDT timestamps using Cam1 epoc onsets.

TWO USE CASES:

  1. align_dlc_velocity()
     You have a DLC-derived velocity CSV (frame-by-frame).
     Assigns true TDT timestamps to each frame and saves output.

  2. align_behavior_csv()
     You have a hand-scored behavior CSV with a 'Time' column in seconds
     (based on video FPS). Converts those times to true TDT timestamps.

Both functions need:
  - A TDT block path (to read Cam1 epoc onsets)
  - An input CSV path
  - An output CSV path
"""

import numpy as np
import pandas as pd
from tdt import read_block


# ─────────────────────────────────────────────────────────────────────────────
# SHARED HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _load_tdt_frame_times(block_path: str) -> np.ndarray:
    """
    Read a TDT block and return the Cam1 epoc onset timestamps (in seconds).
    These are the true timestamps for each camera frame recorded by TDT.
    """
    data = read_block(block_path, store=['Cam1'])
    frame_times = data.epocs['Cam1']['onset']
    print(f"TDT Cam1: {len(frame_times)} frames, "
          f"duration = {frame_times[-1]:.2f} s")
    return frame_times


# ─────────────────────────────────────────────────────────────────────────────
# USE CASE 1 — Align a DLC velocity CSV to TDT frame times
# ─────────────────────────────────────────────────────────────────────────────

def align_dlc_velocity(
    block_path:  str,
    dlc_path:    str,
    output_path: str,
) -> pd.DataFrame:
    """
    Assign true TDT timestamps to each row of a DLC velocity CSV.

    The DLC CSV rows are assumed to correspond 1-to-1 with camera frames.
    If the number of DLC rows and TDT frames differ, the shorter one is used.

    Parameters
    ----------
    block_path  : path to the TDT block folder
    dlc_path    : path to the DLC velocity CSV (one row per frame)
    output_path : where to save the aligned CSV

    Returns
    -------
    DataFrame with a new 'true_time' column added
    """
    frame_times = _load_tdt_frame_times(block_path)

    dlc_data = pd.read_csv(dlc_path)
    print(f"DLC CSV rows: {len(dlc_data)}")

    # Truncate to whichever is shorter
    n_frames = min(len(dlc_data), len(frame_times))
    dlc_data = dlc_data.iloc[:n_frames].copy()
    dlc_data['true_time'] = frame_times[:n_frames]

    dlc_data.to_csv(output_path, index=False)
    print(f"Saved aligned DLC CSV → {output_path}")

    return dlc_data


# ─────────────────────────────────────────────────────────────────────────────
# USE CASE 2 — Align a hand-scored behavior CSV to TDT frame times
# ─────────────────────────────────────────────────────────────────────────────

def align_behavior_csv(
    block_path:    str,
    behavior_path: str,
    output_path:   str,
    behavior_fps:  float = 20.0,
    time_col:      str   = 'Time',
) -> pd.DataFrame:
    """
    Convert hand-scored behavior timestamps to true TDT timestamps.

    Your behavior CSV has a 'Time' column in seconds (based on video FPS).
    This function converts those times → frame indices → true TDT times.

    Parameters
    ----------
    block_path    : path to the TDT block folder
    behavior_path : path to the hand-scored behavior CSV
    output_path   : where to save the aligned CSV
    behavior_fps  : FPS used when scoring behavior (default 20)
    time_col      : name of the time column in your behavior CSV

    Returns
    -------
    DataFrame with 'Time' column replaced by true TDT timestamps
    """
    frame_times = _load_tdt_frame_times(block_path)
    max_idx     = len(frame_times) - 1

    behav_df = pd.read_csv(behavior_path)
    print(f"Behavior CSV rows: {len(behav_df)}")

    # Convert time (s) → frame index → clip to valid range → map to TDT time
    frame_indices = (behav_df[time_col] * behavior_fps).round().astype(int)
    frame_indices = frame_indices.clip(0, max_idx)
    behav_df[time_col] = frame_indices.apply(lambda i: frame_times[i])

    behav_df.to_csv(output_path, index=False)
    print(f"Saved aligned behavior CSV → {output_path}")
    print(behav_df[[time_col, 'Behavior']].head())

    return behav_df


