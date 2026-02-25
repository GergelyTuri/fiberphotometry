"""
loader.py
─────────────────────────────────────────────────────────────────────────────
Functions for loading and preprocessing fiber photometry and BORIS behavior data.

Usage:
    from loader import load_behavior, load_serotonin

BORIS export notes:
    - Export as "Observations list" CSV with all fields
    - Behaviors coded as consecutive POINT pairs (onset, offset)
    - Time column = absolute video timestamp; script auto-corrects to recording start

Serotonin CSV notes:
    - Expected columns: 'time' (seconds), 'dff' (ΔF/F or z-score from your preprocessing)
    - ~100 Hz sampling rate assumed; any rate works automatically
"""

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

# ── Behavior label map ────────────────────────────────────────────────────────
# Edit here to match your BORIS behavior codes
BEHAVIOR_LABELS = {
    'e': 'Exploring',
    'g': 'Grooming',
    'd': 'Digging',
    'r': 'Rearing',
}

# ── Default parameters ────────────────────────────────────────────────────────
DEFAULT_MIN_BOUT_S  = 3.0   # minimum bout duration to keep (seconds)
DEFAULT_SMOOTH_SIGMA = 2    # gaussian smoothing kernel width (samples)


# ─────────────────────────────────────────────────────────────────────────────
# BEHAVIOR LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_behavior(path, min_bout_s=DEFAULT_MIN_BOUT_S, verbose=True):
    """
    Load a BORIS-exported CSV and return behavior bout epochs.

    Parameters
    ----------
    path : str
        Path to BORIS CSV file.
    min_bout_s : float
        Minimum bout duration in seconds. Bouts shorter than this are dropped.
    verbose : bool
        Print summary of bouts per behavior.

    Returns
    -------
    bouts : dict
        Keys = behavior code (str), values = np.ndarray of shape (n_bouts, 2)
        where column 0 = onset (s), column 1 = offset (s), relative to
        recording start (t=0).

    Notes
    -----
    Time alignment: BORIS stores absolute video timestamps. This function
    subtracts the recording start time so t=0 = start of photometry recording.
    If your BORIS file has a non-zero 'Time offset (s)' field, that is also
    applied automatically.
    """
    df = pd.read_csv(path)

    # ── Time alignment ────────────────────────────────────────────────────
    time_offset = float(df['Time offset (s)'].iloc[0]) if 'Time offset (s)' in df.columns else 0.0
    rec_start   = df['Time'].min() - time_offset
    df['time_adj'] = df['Time'] - rec_start

    bouts = {}
    for beh_code in sorted(df['Behavior'].unique()):
        events = df[df['Behavior'] == beh_code]['time_adj'].sort_values().values

        # Warn and trim if odd number of events
        if len(events) % 2 != 0:
            print(f"  WARNING [{beh_code}]: odd number of events ({len(events)}), "
                  f"dropping last unpaired event.")
            events = events[:-1]

        if len(events) == 0:
            continue

        pairs     = events.reshape(-1, 2)
        durations = pairs[:, 1] - pairs[:, 0]
        valid     = pairs[durations >= min_bout_s]

        if verbose:
            label = BEHAVIOR_LABELS.get(beh_code, beh_code)
            n_dropped = len(pairs) - len(valid)
            print(f"  [{beh_code}] {label}: "
                  f"{len(pairs)} bouts total, "
                  f"{n_dropped} dropped (<{min_bout_s}s), "
                  f"{len(valid)} kept | "
                  f"dur {durations.min():.1f}–{durations.max():.1f}s "
                  f"(median {np.median(durations):.1f}s)")

        bouts[beh_code] = valid

    return bouts


# ─────────────────────────────────────────────────────────────────────────────
# SEROTONIN LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_serotonin(path, smooth=True, smooth_sigma=DEFAULT_SMOOTH_SIGMA, verbose=True):
    """
    Load a serotonin fiber photometry CSV.

    Parameters
    ----------
    path : str
        Path to CSV with columns 'time' and 'dff'.
    smooth : bool
        Apply light gaussian smoothing for visualization (does not affect
        per-trial z-score normalization in the extractor).
    smooth_sigma : float
        Sigma for gaussian kernel in samples.
    verbose : bool
        Print signal summary.

    Returns
    -------
    t : np.ndarray
        Time vector (seconds).
    dff : np.ndarray
        ΔF/F signal (smoothed if smooth=True).
    sr : float
        Estimated sampling rate (Hz).
    """
    df  = pd.read_csv(path)
    t   = df['time'].values.astype(float)
    dff = df['dff'].values.astype(float)

    # Estimate sampling rate from median interval
    sr = 1.0 / np.median(np.diff(t))

    if smooth:
        dff = gaussian_filter1d(dff, sigma=smooth_sigma)

    if verbose:
        print(f"  Signal: {len(t)} samples | "
              f"{t[-1]:.1f}s duration | "
              f"~{sr:.1f} Hz | "
              f"dF/F range [{dff.min():.2f}, {dff.max():.2f}]")

    return t, dff, sr


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_behavior_label(code):
    """Return human-readable label for a behavior code."""
    return BEHAVIOR_LABELS.get(code, code)


def get_all_behavior_codes():
    """Return list of all defined behavior codes."""
    return list(BEHAVIOR_LABELS.keys())