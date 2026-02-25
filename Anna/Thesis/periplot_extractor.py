"""
extractor.py
─────────────────────────────────────────────────────────────────────────────
Functions for extracting peri-event serotonin traces aligned to behavior
onset or offset, with per-trial baseline normalization and quantification.

Usage:
    from extractor import extract_peri_event, quantify_trials, build_animal_dataset
"""

import numpy as np
from scipy.stats import sem as scipy_sem

# ── Default window parameters (seconds) ──────────────────────────────────────
DEFAULT_PRE_S         = 5.0   # window before event
DEFAULT_POST_S        = 10.0  # window after event
DEFAULT_BASELINE_PRE  = 3.0   # baseline window start (seconds before event)
DEFAULT_BASELINE_END  = 0.5   # baseline window end (seconds before event)
DEFAULT_QUANT_WINDOW  = (0.0, 5.0)  # post-event window for AUC / peak quantification


# ─────────────────────────────────────────────────────────────────────────────
# CORE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_peri_event(
    t_sero,
    dff_sero,
    event_times,
    pre_s          = DEFAULT_PRE_S,
    post_s         = DEFAULT_POST_S,
    baseline_pre_s = DEFAULT_BASELINE_PRE,
    baseline_end_s = DEFAULT_BASELINE_END,
    align          = 'onset',
    verbose        = True,
):
    """
    Extract peri-event serotonin traces aligned to a list of event times.

    Each trial is independently z-score normalized using a pre-event baseline
    window, so drift and between-trial differences in baseline do not contaminate
    the signal.

    Parameters
    ----------
    t_sero : np.ndarray
        Photometry time vector (seconds).
    dff_sero : np.ndarray
        Photometry signal (dF/F).
    event_times : np.ndarray
        1D array of event timestamps (seconds, same reference as t_sero).
    pre_s : float
        Seconds before event to include in window.
    post_s : float
        Seconds after event to include in window.
    baseline_pre_s : float
        Baseline window starts this many seconds before event.
    baseline_end_s : float
        Baseline window ends this many seconds before event.
        Window = [event - baseline_pre_s : event - baseline_end_s]
    align : str
        Label for the alignment type ('onset' or 'offset'), used for printing only.
    verbose : bool
        Print extraction summary.

    Returns
    -------
    time_axis : np.ndarray
        Time relative to event (seconds), shape (n_samples,).
    traces_z : np.ndarray
        Per-trial z-scored traces, shape (n_valid_trials, n_samples).
    traces_dff : np.ndarray
        Per-trial baseline-subtracted dF/F traces, shape (n_valid_trials, n_samples).
    trial_meta : list of dict
        Per-trial metadata: {'event_time', 'baseline_mean', 'baseline_std', 'idx'}.
    """
    sr      = 1.0 / np.median(np.diff(t_sero))
    n_pre   = int(pre_s  * sr)
    n_post  = int(post_s * sr)
    n_total = n_pre + n_post

    # Build common time axis
    time_axis = np.linspace(-pre_s, post_s, n_total)

    traces_z   = []
    traces_dff = []
    trial_meta = []
    n_skipped  = 0

    for ev_t in event_times:
        # Find nearest sample index
        idx = np.searchsorted(t_sero, ev_t)
        i_start = idx - n_pre
        i_end   = idx + n_post

        # Skip if window extends outside recording
        if i_start < 0 or i_end > len(dff_sero):
            n_skipped += 1
            continue

        trial = dff_sero[i_start:i_end]
        if len(trial) != n_total:
            n_skipped += 1
            continue

        # Per-trial baseline
        bl_start_idx = int((pre_s - baseline_pre_s) * sr)
        bl_end_idx   = int((pre_s - baseline_end_s)  * sr)
        baseline_samples = trial[bl_start_idx:bl_end_idx]

        if len(baseline_samples) < 5:
            n_skipped += 1
            continue

        bl_mean = np.mean(baseline_samples)
        bl_std  = np.std(baseline_samples)

        dff_trial = trial - bl_mean  # baseline-subtracted dF/F

        if bl_std > 1e-6:
            z_trial = dff_trial / bl_std
        else:
            z_trial = dff_trial  # fallback if std is ~0

        traces_z.append(z_trial)
        traces_dff.append(dff_trial)
        trial_meta.append({
            'event_time':    ev_t,
            'baseline_mean': bl_mean,
            'baseline_std':  bl_std,
            'sample_idx':    idx,
        })

    if verbose:
        n_valid = len(traces_z)
        print(f"  {align} alignment: {n_valid}/{len(event_times)} trials extracted "
              f"({n_skipped} skipped — out of recording bounds)")

    if not traces_z:
        empty = np.empty((0, n_total))
        return time_axis, empty, empty, trial_meta

    return time_axis, np.array(traces_z), np.array(traces_dff), trial_meta


def extract_onset_and_offset(
    t_sero,
    dff_sero,
    bouts,
    behavior_code,
    **kwargs,
):
    """
    Convenience wrapper: extract both onset- and offset-aligned traces for one behavior.

    Parameters
    ----------
    t_sero, dff_sero : as in extract_peri_event
    bouts : dict
        Output of loader.load_behavior()
    behavior_code : str
        Which behavior to extract (e.g. 'e').
    **kwargs :
        Passed through to extract_peri_event (pre_s, post_s, etc.)

    Returns
    -------
    dict with keys 'time_axis', 'onset_z', 'onset_dff', 'offset_z', 'offset_dff',
    'onset_meta', 'offset_meta', 'durations'
    """
    bout_array = bouts.get(behavior_code, np.array([]))
    if len(bout_array) == 0:
        print(f"  No valid bouts found for behavior '{behavior_code}'")
        return None

    onsets    = bout_array[:, 0]
    offsets   = bout_array[:, 1]
    durations = offsets - onsets

    t_ax, on_z, on_dff, on_meta = extract_peri_event(
        t_sero, dff_sero, onsets, align='onset', **kwargs)
    _,    off_z, off_dff, off_meta = extract_peri_event(
        t_sero, dff_sero, offsets, align='offset', **kwargs)

    return {
        'time_axis':   t_ax,
        'onset_z':     on_z,
        'onset_dff':   on_dff,
        'onset_meta':  on_meta,
        'offset_z':    off_z,
        'offset_dff':  off_dff,
        'offset_meta': off_meta,
        'durations':   durations,
        'n_bouts':     len(bout_array),
    }


# ─────────────────────────────────────────────────────────────────────────────
# QUANTIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def quantify_trials(
    time_axis,
    traces_z,
    quant_window = DEFAULT_QUANT_WINDOW,
):
    """
    Quantify signal in a post-event window per trial.

    Parameters
    ----------
    time_axis : np.ndarray
        Time vector relative to event.
    traces_z : np.ndarray
        z-scored traces, shape (n_trials, n_samples).
    quant_window : tuple (float, float)
        (start, end) in seconds relative to event for quantification.

    Returns
    -------
    dict with per-trial and summary statistics:
        'peak'     : peak z-score per trial
        'auc'      : area under curve per trial (trapezoidal)
        'mean_peak': grand mean of peak values
        'sem_peak' : SEM of peak values
        'mean_auc' : grand mean of AUC
        'sem_auc'  : SEM of AUC
        'n'        : number of trials
    """
    if len(traces_z) == 0:
        return {k: np.nan for k in
                ['peak', 'auc', 'mean_peak', 'sem_peak', 'mean_auc', 'sem_auc', 'n']}

    q_mask = (time_axis >= quant_window[0]) & (time_axis <= quant_window[1])
    t_q    = time_axis[q_mask]
    tr_q   = traces_z[:, q_mask]

    peak_vals = np.max(tr_q, axis=1)
    auc_vals  = np.trapezoid(tr_q, t_q, axis=1) if hasattr(np, 'trapezoid') else np.trapz(tr_q, t_q, axis=1)

    return {
        'peak':      peak_vals,
        'auc':       auc_vals,
        'mean_peak': np.mean(peak_vals),
        'sem_peak':  scipy_sem(peak_vals),
        'mean_auc':  np.mean(auc_vals),
        'sem_auc':   scipy_sem(auc_vals),
        'n':         len(peak_vals),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-ANIMAL DATASET BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_animal_dataset(animal_id, condition, bouts, t_sero, dff_sero, **kwargs):
    """
    Package all extracted traces for one animal into a single dataset dict.
    This is the object you save per animal and then pass to the plotter for
    group-level figures.

    Parameters
    ----------
    animal_id : str
        Unique animal identifier (e.g. 'nia4').
    condition : str
        Condition label (e.g. 'saline', 'psi').
    bouts : dict
        Output of loader.load_behavior().
    t_sero, dff_sero : arrays
        Output of loader.load_serotonin().
    **kwargs :
        Passed to extract_onset_and_offset (pre_s, post_s, etc.)

    Returns
    -------
    dataset : dict
        {
          'animal_id'  : str,
          'condition'  : str,
          'behaviors'  : {
              beh_code : {
                  'time_axis', 'onset_z', 'onset_dff',
                  'offset_z', 'offset_dff', 'durations', 'n_bouts',
                  'quant_onset', 'quant_offset'   ← from quantify_trials
              }, ...
          }
        }
    """
    print(f"\n── Building dataset: {animal_id} | {condition} ──")
    dataset = {
        'animal_id': animal_id,
        'condition': condition,
        'behaviors': {},
    }

    for beh_code in bouts:
        print(f"\n  Behavior: {beh_code}")
        result = extract_onset_and_offset(
            t_sero, dff_sero, bouts, beh_code, **kwargs)

        if result is None:
            continue

        result['quant_onset']  = quantify_trials(result['time_axis'], result['onset_z'])
        result['quant_offset'] = quantify_trials(result['time_axis'], result['offset_z'])

        dataset['behaviors'][beh_code] = result

    return dataset


def save_dataset(dataset, path):
    """Save a dataset dict to a .npz file for later group-level analysis."""
    np.save(path, dataset, allow_pickle=True)
    print(f"  Saved dataset → {path}")


def load_dataset(path):
    """Load a previously saved dataset .npy file."""
    return np.load(path, allow_pickle=True).item()