"""
dlc_to_velocity.py
──────────────────
Convert a DLC .h5 tracking file into a smoothed velocity CSV.

One function to call per file:
    dlc_to_smoothed_velocity()

Output CSV columns:
    'Time (s)'               — true time in seconds (from TDT frame times if provided,
                               otherwise estimated from frame index / fps)
    'Smoothed Velocity (cm/s)' — Gaussian-smoothed velocity with artifact frames removed
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


# ─────────────────────────────────────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def dlc_to_smoothed_velocity(
    h5_path:          str,
    output_path:      str,
    fps:              float,
    bodypart:         str   = 'back1',
    likelihood_thresh: float = 0.8,
    velocity_cutoff:  float = 200.0,
    sigma:            float = 3.0,
    frame_times:      np.ndarray | None = None,
    plot:             bool  = True,
) -> pd.DataFrame:
    """
    Load a DLC .h5 file, compute smoothed velocity, and save to CSV.

    Parameters
    ----------
    h5_path           : path to the DLC .h5 tracking file
    output_path       : where to save the output CSV
    fps               : camera frame rate (used if frame_times not provided)
    bodypart          : which body part to track (default 'back1')
    likelihood_thresh : frames below this likelihood are treated as NaN (default 0.8)
    velocity_cutoff   : velocity values above this are removed as artifacts (default 200 cm/s)
    sigma             : Gaussian smoothing sigma (default 3)
    frame_times       : optional array of true TDT timestamps (one per frame).
                        If provided, these are used as the time axis instead of fps.
    plot              : show a QC plot of raw vs smoothed velocity

    Returns
    -------
    DataFrame with columns 'Time (s)' and 'Smoothed Velocity (cm/s)'
    """
    # Load DLC data
    beh_df     = pd.read_hdf(h5_path)
    model_name = beh_df.columns.values[0][0]
    print(f"Model: {model_name}")

    # Extract x/y coords and likelihood for chosen bodypart
    x          = beh_df[(model_name, bodypart, 'x')].copy()
    y          = beh_df[(model_name, bodypart, 'y')].copy()
    likelihood = beh_df[(model_name, bodypart, 'likelihood')]

    # Mask out low-likelihood and zero-coordinate frames
    bad = (likelihood <= likelihood_thresh) | ((x == 0) & (y == 0))
    x[bad] = np.nan
    y[bad] = np.nan

    # Interpolate gaps
    x = x.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
    y = y.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')

    # Compute frame-by-frame velocity (pixels/frame → cm/s needs px_per_cm if you have it)
    dist      = np.sqrt(np.diff(x.to_numpy())**2 + np.diff(y.to_numpy())**2)
    velocity  = dist * fps   # pixels per second — scale by px_per_cm if known

    # Pad to match original frame count
    velocity  = np.concatenate([[np.nan], velocity])

    # Build time axis
    n_frames = len(velocity)
    if frame_times is not None:
        n_use  = min(n_frames, len(frame_times))
        time_x = frame_times[:n_use]
        velocity = velocity[:n_use]
    else:
        time_x = np.arange(n_frames) / fps

    # Remove extreme velocity artifacts
    velocity_cleaned = np.where(velocity > velocity_cutoff, np.nan, velocity)

    # Interpolate over removed artifacts before smoothing
    filled = pd.Series(velocity_cleaned).interpolate(limit_direction='both').to_numpy()

    # Gaussian smoothing
    smoothed = gaussian_filter1d(filled, sigma=sigma)

    # Restore NaNs at artifact positions
    smoothed[np.isnan(velocity_cleaned)] = np.nan

    # QC plot
    if plot:
        plt.figure(figsize=(12, 5))
        plt.plot(time_x, velocity_cleaned, color='gray', alpha=0.4, label='Raw')
        plt.plot(time_x, smoothed, color='blue', linewidth=1.2, label=f'Smoothed (σ={sigma})')
        plt.axhline(velocity_cutoff, color='red', linestyle='--', label=f'{velocity_cutoff} cutoff')
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (px/s)')
        plt.title(f'Smoothed Velocity — {h5_path.split("/")[-1]}')
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Save
    df_out = pd.DataFrame({'Time (s)': time_x, 'Smoothed Velocity (cm/s)': smoothed})
    df_out.to_csv(output_path, index=False)
    print(f"Saved {len(df_out)} rows → {output_path}")

    return df_out


