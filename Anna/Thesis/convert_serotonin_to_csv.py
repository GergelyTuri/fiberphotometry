# CASE 1 — Single TDT file with an injection point
#     Use: zscore_single_file()
#     Baseline = pre-injection period inside the same recording.
#     Output = z-scored signal from post-injection onward (or full recording).

# CASE 2 — Two separate TDT files (baseline block + experiment block)
#     Use: zscore_two_files()
#     Baseline stats computed from the baseline block.
#     Output = z-scored experiment signal.

# Shared pipeline for both:
#     1. Read 465A (signal) and 405A (isos) streams
#     2. Fit isos → signal via linear regression (dF/F style)
#     3. Compute ΔF/F = 100 * (signal - fitted_isos) / fitted_isos
#     4. Trim first `artifact_sec` seconds (default 8 s single-file, 30 s two-file)
#     5. Z-score using baseline mean and std
#     6. Save to CSV with columns: 'Time (s)', 'Z-score'


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from tdt import read_block


SEROTONIN_CH = '_465A'
ISOS_CH      = '_405A'



def _compute_dff(signal: np.ndarray, isos: np.ndarray) -> np.ndarray:
 
    # Fit isosbestic to signal with a linear regression, then compute ΔF/F.
    # Falls back to mean(signal) as f0 if the fit slope is negative.

    reg = np.polyfit(isos, signal, 1)
    if reg[0] < 0:
        f0 = np.mean(signal)          # rare edge case
    else:
        f0 = reg[0] * isos + reg[1]   # normal: fitted isos
    return 100 * (signal - f0) / f0


def _trim_artifact(time_x: np.ndarray, signal: np.ndarray,
                   artifact_sec: float) -> tuple[np.ndarray, np.ndarray]:
    # Remove the first `artifact_sec` seconds from time and signal arrays
    ind = np.where(time_x > artifact_sec)[0][0]
    return time_x[ind:], signal[ind:]


def _zscore(signal: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (signal - mean) / std


def _save_and_plot(time_x: np.ndarray, zscore_signal: np.ndarray,
                   output_path: str, title: str, plot: bool) -> pd.DataFrame:
    df = pd.DataFrame({'Time (s)': time_x, 'Z-score': zscore_signal})
    df.to_csv(output_path, index=False)
    print(f'✅ Saved z-score CSV → {output_path}')

    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(time_x, zscore_signal, color='blue', linewidth=1.2)
        plt.axhline(0, color='red', linestyle='--', linewidth=1, label='Baseline (z=0)')
        plt.xlabel('Time (s)')
        plt.ylabel('Z-score')
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return df


# ─────────────────────────────────────────────────────────────────────────────
# CASE 1 — Single TDT file with injection point
# ─────────────────────────────────────────────────────────────────────────────

def zscore_single_file(
    tdt_path:       str,
    start_time_str: str,
    bline_end_str:  str,
    output_path:    str,
    artifact_sec:   float = 8.0,
    post_injection_only: bool = True,
    plot:           bool  = True,
    serotonin_ch:   str   = SEROTONIN_CH,
    isos_ch:        str   = ISOS_CH,
) -> pd.DataFrame:
    """
    Convert a single TDT recording (with an injection point) to a z-score CSV.

    The baseline period is everything from `start_time_str` to `bline_end_str`.
    Z-scoring uses the mean and std of that pre-injection window.

    Parameters
    ----------
    tdt_path          : path to the TDT block folder
    start_time_str    : recording start time, e.g. '10:52:44am'
    bline_end_str     : end of baseline / injection time, e.g. '11:21:31am'
    output_path       : where to save the CSV
    artifact_sec      : seconds to trim from the start (default 8)
    post_injection_only: if True, output only contains post-injection signal
    plot              : whether to show a quick QC plot
    serotonin_ch      : TDT stream name for 465 channel
    isos_ch           : TDT stream name for 405 channel

    Returns
    -------
    DataFrame with columns 'Time (s)' and 'Z-score'
    """
    # Load block
    block = read_block(tdt_path)

    # Compute ΔF/F
    x1  = block['streams'][serotonin_ch].data
    x2  = block['streams'][isos_ch].data
    dff = _compute_dff(x1, x2)

    # Build time axis
    fs     = block['streams'][serotonin_ch].fs
    npts   = len(x1)
    time_x = np.linspace(1, npts, npts) / fs

    # Trim artifact
    time_x, dff = _trim_artifact(time_x, dff, artifact_sec)

    # Convert baseline end time → seconds relative to recording start
    fmt        = '%I:%M:%S%p'
    t_start    = datetime.strptime(start_time_str, fmt)
    t_bline    = datetime.strptime(bline_end_str,  fmt)
    bline_sec  = (t_bline - t_start).total_seconds()

    # Adjust for artifact trim
    bline_adj  = bline_sec - artifact_sec
    bline_idx  = np.where(time_x > bline_adj)[0][0]

    # Z-score using pre-injection baseline stats
    bline_mean = np.mean(dff[:bline_idx])
    bline_std  = np.std(dff[:bline_idx])
    z          = _zscore(dff, bline_mean, bline_std)

    # Optionally keep only post-injection signal
    if post_injection_only:
        time_x = time_x[bline_idx:]
        z      = z[bline_idx:]
        # Reset time so post-injection starts at t=0
        time_x = time_x - time_x[0]

    return _save_and_plot(time_x, z, output_path,
                          title=f'Z-score — {tdt_path.split("/")[-1]}', plot=plot)


# ─────────────────────────────────────────────────────────────────────────────
# CASE 2 — Two separate TDT files (baseline block + experiment block)
# ─────────────────────────────────────────────────────────────────────────────

def zscore_two_files(
        
    baseline_path:   str,
    experiment_path: str,
    output_path:     str,
    artifact_sec:    float = 30.0,
    plot:            bool  = True,
    serotonin_ch:    str   = SEROTONIN_CH,
    isos_ch:         str   = ISOS_CH,
) -> pd.DataFrame:
    """
    Convert a TDT experiment recording to a z-score CSV using a separate
    baseline block for normalization.

    Pipeline:
      - Compute ΔF/F for the baseline block → extract mean & std
      - Compute ΔF/F for the experiment block → z-score with baseline stats
      - Trim first `artifact_sec` seconds from the experiment signal

    Parameters
    ----------
    baseline_path   : path to the TDT baseline block folder
    experiment_path : path to the TDT experiment block folder
    output_path     : where to save the CSV
    artifact_sec    : seconds to trim from start of experiment (default 30)
    plot            : whether to show a quick QC plot
    serotonin_ch    : TDT stream name for 465 channel
    isos_ch         : TDT stream name for 405 channel

    Returns
    -------
    DataFrame with columns 'Time (s)' and 'Z-score'
    """
    # ── Baseline block ────────────────────────────────────────────────────────
    baseline   = read_block(baseline_path)
    x1_base    = baseline['streams'][serotonin_ch].data
    x2_base    = baseline['streams'][isos_ch].data
    dff_base   = _compute_dff(x1_base, x2_base)
    bline_mean = np.mean(dff_base)
    bline_std  = np.std(dff_base)
    print(f'Baseline → mean: {bline_mean:.4f}, std: {bline_std:.4f}')

    # ── Experiment block ──────────────────────────────────────────────────────
    experiment = read_block(experiment_path)
    x1_exp     = experiment['streams'][serotonin_ch].data
    x2_exp     = experiment['streams'][isos_ch].data
    dff_exp    = _compute_dff(x1_exp, x2_exp)

    # Build time axis
    fs     = experiment['streams'][serotonin_ch].fs
    npts   = len(x1_exp)
    time_x = np.linspace(0, npts / fs, npts)

    # Trim artifact
    time_x, dff_exp = _trim_artifact(time_x, dff_exp, artifact_sec)

    # Z-score using baseline stats
    z = _zscore(dff_exp, bline_mean, bline_std)

    label = experiment_path.rstrip('/').split('/')[-1]
    return _save_and_plot(time_x, z, output_path,
                          title=f'Z-score — {label}', plot=plot)


def zscore_from_dff_csvs(
    baseline_path:  str,
    recording_path: str,
    output_path:    str,
    time_col:       str  = 'time',
    dff_col:        str  = 'dff',
    plot:           bool = True,
) -> pd.DataFrame:
    """
    Z-score a recording dff CSV using the mean and std of a baseline dff CSV.

    Parameters
    ----------
    baseline_path  : path to baseline dff CSV
    recording_path : path to recording dff CSV
    output_path    : where to save the z-scored CSV
    time_col       : name of time column (default 'time')
    dff_col        : name of dff column (default 'dff')
    plot           : show a quick QC plot

    Returns
    -------
    DataFrame with columns 'Time (s)' and 'Z-score'
    """
    baseline  = pd.read_csv(baseline_path)
    recording = pd.read_csv(recording_path)

    # Compute baseline stats
    bline_mean = baseline[dff_col].mean()
    bline_std  = baseline[dff_col].std()
    print(f'Baseline → mean: {bline_mean:.4f}, std: {bline_std:.4f}')

    # Z-score the recording using baseline stats
    z = (recording[dff_col] - bline_mean) / bline_std

    df_out = pd.DataFrame({
        'Time (s)': recording[time_col].values,
        'Z-score':  z.values,
    })

    df_out.to_csv(output_path, index=False)
    print(f'Saved → {output_path}')

    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(df_out['Time (s)'], df_out['Z-score'],
                 color='blue', linewidth=1.2)
        plt.axhline(0, color='red', linestyle='--', linewidth=1,
                    label='Baseline (z=0)')
        plt.xlabel('Time (s)')
        plt.ylabel('Z-score')
        plt.title(f'Z-score — {recording_path.split("/")[-1]}')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return df_out