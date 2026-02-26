"""
periplot_loader.py
─────────────────────────────────────────────────────────────────────────────
Functions for loading and preprocessing fiber photometry and BORIS behavior data.

Usage:
    from periplot_loader import load_behavior, load_serotonin

BORIS export notes:
    - Export as "Observations list" CSV with all fields
    - Behaviors coded as consecutive POINT pairs (onset, offset)
    - Time column = absolute video timestamp; script auto-corrects to recording start

Serotonin CSV notes:
    - Expected columns: 'Time (s)', 'Z-score' (or similar variations)
    - Any sampling rate works — detected automatically
"""

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

# ── Per-animal behavior code remapping ───────────────────────────────────────
# Standardizes inconsistent BORIS coding across animals
# Final standard: e=Exploring, g=Grooming, r=Rearing, eat=Eating, s=Nose out of box
BEHAVIOR_REMAP = {
    'nia11': {'e': 'eat', 'p': 'e', 'y': 'eat'},
    'nia2':  {'e': 'eat', 'p': 'e', 'y': 'eat'},
    'nia4':  {'e': 'eat', 'p': 'e', 'y': 'eat'},
    'nia44': {'y': 'eat'},
    'nia41': {'y': 'eat'},
    'nia35': {'y': 'eat'},
}

# ── Behavior label map ────────────────────────────────────────────────────────
BEHAVIOR_LABELS = {
    'e':   'Exploring',
    'g':   'Grooming',
    'r':   'Rearing',
    'd':   'Digging',
    'eat': 'Eating',
    's':   'Nose out of box',
}

# ── Default parameters ────────────────────────────────────────────────────────
DEFAULT_MIN_BOUT_S   = 3.0  # minimum bout duration to keep (seconds)
DEFAULT_SMOOTH_SIGMA = 2    # gaussian smoothing kernel width (samples)


# ─────────────────────────────────────────────────────────────────────────────
# BEHAVIOR LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_behavior(path, animal_id='', min_bout_s=DEFAULT_MIN_BOUT_S, verbose=True):
    """
    Load a BORIS-exported CSV and return behavior bout epochs.

    Parameters
    ----------
    path : str
        Path to BORIS CSV file.
    animal_id : str
        Animal identifier — used to apply per-animal behavior code remapping.
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
    """
    df = pd.read_csv(path)

    # ── Per-animal behavior code remapping ────────────────────────────────
    remap = BEHAVIOR_REMAP.get(animal_id, {})
    if remap:
        df['Behavior'] = df['Behavior'].replace(remap)
        if verbose:
            print(f"  Applied behavior remap for {animal_id}: {remap}")

    # ── Time alignment ────────────────────────────────────────────────────
    time_offset = float(df['Time offset (s)'].iloc[0]) if 'Time offset (s)' in df.columns else 0.0
    time_col = next((c for c in df.columns if 'time' in c.lower()), None)
    if time_col is None:
        raise KeyError(f"No time column found. Columns: {list(df.columns)}")
    rec_start      = df[time_col].min() - time_offset
    df['time_adj'] = df[time_col] - rec_start

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
            label     = BEHAVIOR_LABELS.get(beh_code, beh_code)
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

    Returns
    -------
    t : np.ndarray       — time vector (seconds)
    dff : np.ndarray     — z-score signal (smoothed if smooth=True)
    sr : float           — sampling rate (Hz)
    """
    df = pd.read_csv(path)

    # Flexible time column detection
    time_col = None
    for candidate in ['time', 'Time', 'Time (s)', 'time (s)', 'timestamp', 'Timestamp']:
        if candidate in df.columns:
            time_col = candidate
            break
    if time_col is None:
        raise KeyError(f"No time column found. Columns present: {list(df.columns)}")

    # Flexible signal column detection
    dff_col = None
    for candidate in ['dff', 'DFF', 'dF/F', 'df/f', 'signal', 'Signal',
                       'zscore', 'z_score', 'Z-score', 'Z_score']:
        if candidate in df.columns:
            dff_col = candidate
            break
    if dff_col is None:
        raise KeyError(f"No signal column found. Columns present: {list(df.columns)}")

    if verbose:
        print(f"  Using columns: time='{time_col}', signal='{dff_col}'")

    t   = df[time_col].values.astype(float)
    dff = df[dff_col].values.astype(float)

    # Estimate sampling rate from median interval
    sr = 1.0 / np.median(np.diff(t))

    if smooth:
        dff = gaussian_filter1d(dff, sigma=smooth_sigma)

    if verbose:
        print(f"  Signal: {len(t)} samples | "
              f"{t[-1]:.1f}s duration | "
              f"~{sr:.1f} Hz | "
              f"z-score range [{dff.min():.2f}, {dff.max():.2f}]")

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