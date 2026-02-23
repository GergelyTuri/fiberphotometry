"""
split_at_injection.py
─────────────────────
For animals where baseline and post-injection are in ONE recording,
split the z-score or velocity CSV into two separate files at the injection point.

After running this, every animal will have:
    - a baseline CSV
    - a recording (post-injection) CSV

...regardless of whether they originally came from one file or two.
This means all downstream analysis functions can treat every animal the same way.
"""

import pandas as pd
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — convert clock times to injection time in seconds
# ─────────────────────────────────────────────────────────────────────────────

def clock_to_injection_seconds(
    start_time_str:     str,
    injection_time_str: str,
) -> float:
    """
    Convert a start time and injection clock time into injection time in seconds
    relative to the start of the recording.

    Parameters
    ----------
    start_time_str     : recording start time, e.g. '10:52:44am'
    injection_time_str : injection clock time, e.g. '11:21:31am'

    Returns
    -------
    float — injection time in seconds from recording start

    Example
    -------
    >>> clock_to_injection_seconds('10:52:44am', '11:21:31am')
    1727.0
    """
    fmt        = '%I:%M:%S%p'
    t_start    = datetime.strptime(start_time_str,     fmt)
    t_inject   = datetime.strptime(injection_time_str, fmt)
    seconds    = (t_inject - t_start).total_seconds()
    print(f"Injection at {seconds:.1f} s after recording start")
    return seconds


# ─────────────────────────────────────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def split_at_injection(
    input_path:       str,
    baseline_out:     str,
    recording_out:    str,
    injection_time:   float,
    time_col:         str  = 'Time (s)',
    reset_time:       bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a single CSV into a baseline and post-injection CSV at a given time.

    Parameters
    ----------
    input_path      : path to the combined CSV (z-score or velocity)
    baseline_out    : where to save the baseline portion
    recording_out   : where to save the post-injection portion
    injection_time  : time in seconds where injection happened
                      (use clock_to_injection_seconds() to get this)
    time_col        : name of the time column (default 'Time (s)')
    reset_time      : if True, both output CSVs start at t=0 (default True)

    Returns
    -------
    (baseline_df, recording_df)
    """
    df = pd.read_csv(input_path)

    baseline  = df[df[time_col] <  injection_time].copy()
    recording = df[df[time_col] >= injection_time].copy()

    if reset_time:
        baseline[time_col]  = baseline[time_col]  - baseline[time_col].iloc[0]
        recording[time_col] = recording[time_col] - recording[time_col].iloc[0]

    baseline.to_csv(baseline_out,  index=False)
    recording.to_csv(recording_out, index=False)

    print(f"Baseline:  {len(baseline)} rows  → {baseline_out}")
    print(f"Recording: {len(recording)} rows → {recording_out}")

    return baseline, recording


