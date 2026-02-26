"""
periplot_extractor.py
─────────────────────────────────────────────────────────────────────────────
Extracts peri-event serotonin traces aligned to post-immobility behavior onset.

Core logic:
  1. Find gaps ≥ MIN_IMMOBILITY_S between any coded behaviors → immobility epochs
  2. Align to end of each gap = onset of the next behavior
  3. Split trials by which behavior follows (e, g, d, r)
  4. Window: pre_s before → post_s after (default −10s to +50s)
  5. NO per-trial normalization — signal is already globally z-scored

Usage:
    from periplot_extractor import (find_immobility_gaps,
                                    extract_post_immobility,
                                    build_animal_dataset,
                                    save_dataset, load_dataset)
"""

import numpy as np
from scipy.stats import sem as scipy_sem

# ── Default parameters ────────────────────────────────────────────────────────
DEFAULT_MIN_IMMOBILITY_S = 10.0   # minimum gap to count as immobility
DEFAULT_PRE_S            = 10.0   # window before behavior onset (into immobility)
DEFAULT_POST_S           = 50.0   # window after behavior onset (full recovery)
DEFAULT_QUANT_WINDOW     = (0.0, 15.0)  # post-onset window for peak/AUC stats


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — FIND IMMOBILITY GAPS
# ─────────────────────────────────────────────────────────────────────────────

def find_immobility_gaps(bouts, min_immobility_s=DEFAULT_MIN_IMMOBILITY_S, verbose=True):
    """
    Find gaps between ANY coded behavior bouts that are ≥ min_immobility_s.
    These gaps = immobility epochs.

    Parameters
    ----------
    bouts : dict
        Output of periplot_loader.load_behavior().
        Keys = behavior code, values = np.ndarray (n_bouts, 2) of [onset, offset].
    min_immobility_s : float
        Minimum gap duration to count as immobility.
    verbose : bool

    Returns
    -------
    gaps : np.ndarray, shape (n_gaps, 2)
        Each row = [gap_start, gap_end] in seconds.
        gap_end = onset of the behavior that follows.
    next_behaviors : list of str
        Behavior code of the behavior that starts at each gap_end.
    """
    # Flatten all bouts from all behaviors into one sorted list of (onset, offset, code)
    all_events = []
    for code, bout_array in bouts.items():
        for onset, offset in bout_array:
            all_events.append((onset, offset, code))

    if not all_events:
        return np.empty((0, 2)), []

    all_events.sort(key=lambda x: x[0])  # sort by onset time

    gaps          = []
    next_behaviors = []

    for i in range(1, len(all_events)):
        prev_offset = all_events[i - 1][1]   # end of previous behavior
        curr_onset  = all_events[i][0]        # start of current behavior
        curr_code   = all_events[i][2]

        gap_duration = curr_onset - prev_offset

        if gap_duration >= min_immobility_s:
            gaps.append([prev_offset, curr_onset])
            next_behaviors.append(curr_code)

    gaps = np.array(gaps) if gaps else np.empty((0, 2))

    if verbose:
        print(f"  Found {len(gaps)} immobility gaps ≥ {min_immobility_s}s")
        if len(gaps) > 0:
            durations = gaps[:, 1] - gaps[:, 0]
            print(f"  Gap durations: {durations.min():.1f}–{durations.max():.1f}s "
                  f"(median {np.median(durations):.1f}s)")
            from collections import Counter
            counts = Counter(next_behaviors)
            for code, n in sorted(counts.items()):
                print(f"    → {code}: {n} trials")

    return gaps, next_behaviors


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — EXTRACT TRACES (no per-trial normalization)
# ─────────────────────────────────────────────────────────────────────────────

def extract_post_immobility(
    t_sero,
    dff_sero,
    gaps,
    next_behaviors,
    behavior_code,
    pre_s  = DEFAULT_PRE_S,
    post_s = DEFAULT_POST_S,
    verbose = True,
):
    """
    Extract serotonin traces aligned to the END of each immobility gap
    (= onset of the next behavior), filtered to one behavior type.

    NO per-trial normalization is applied — signal is used as-is.

    Parameters
    ----------
    t_sero : np.ndarray
        Photometry time vector (seconds).
    dff_sero : np.ndarray
        Already-normalized z-score signal.
    gaps : np.ndarray (n_gaps, 2)
        Output of find_immobility_gaps().
    next_behaviors : list of str
        Output of find_immobility_gaps().
    behavior_code : str
        Which behavior to extract trials for (e.g. 'e').
    pre_s : float
        Seconds before behavior onset to include (looking back into immobility).
    post_s : float
        Seconds after behavior onset to include.
    verbose : bool

    Returns
    -------
    time_axis : np.ndarray
        Time relative to behavior onset (s). Negative = during immobility.
    traces : np.ndarray
        Shape (n_trials, n_samples). Raw z-score, no normalization.
    gap_durations : np.ndarray
        Duration of immobility gap for each kept trial.
    trial_meta : list of dict
        Per-trial metadata.
    """
    sr      = 1.0 / np.median(np.diff(t_sero))
    n_pre   = int(pre_s  * sr)
    n_post  = int(post_s * sr)
    n_total = n_pre + n_post

    time_axis = np.linspace(-pre_s, post_s, n_total)

    traces        = []
    gap_durations = []
    trial_meta    = []
    n_skipped     = 0

    for i, (gap, next_beh) in enumerate(zip(gaps, next_behaviors)):
        if next_beh != behavior_code:
            continue

        event_t      = gap[1]   # end of gap = onset of next behavior
        gap_duration = gap[1] - gap[0]

        idx     = np.searchsorted(t_sero, event_t)
        i_start = idx - n_pre
        i_end   = idx + n_post

        if i_start < 0 or i_end > len(dff_sero):
            n_skipped += 1
            continue

        trial = dff_sero[i_start:i_end]
        if len(trial) != n_total:
            n_skipped += 1
            continue

        traces.append(trial)
        gap_durations.append(gap_duration)
        trial_meta.append({
            'event_time':    event_t,
            'gap_start':     gap[0],
            'gap_duration':  gap_duration,
            'behavior_code': next_beh,
        })

    if verbose:
        n_valid = len(traces)
        print(f"  [{behavior_code}] {n_valid} trials extracted "
              f"({n_skipped} skipped — out of recording bounds)")

    if not traces:
        return time_axis, np.empty((0, n_total)), np.array([]), []

    return time_axis, np.array(traces), np.array(gap_durations), trial_meta


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — QUANTIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def quantify_trials(time_axis, traces, quant_window=DEFAULT_QUANT_WINDOW):
    """
    Quantify peak z-score and AUC in a post-onset window.

    Parameters
    ----------
    time_axis : np.ndarray
    traces : np.ndarray (n_trials, n_samples)
    quant_window : tuple (start_s, end_s) relative to behavior onset

    Returns
    -------
    dict with 'peak', 'auc', 'mean_peak', 'sem_peak', 'mean_auc', 'sem_auc', 'n'
    """
    if len(traces) == 0:
        return {k: np.nan for k in
                ['peak', 'auc', 'mean_peak', 'sem_peak', 'mean_auc', 'sem_auc', 'n']}

    q_mask    = (time_axis >= quant_window[0]) & (time_axis <= quant_window[1])
    t_q       = time_axis[q_mask]
    tr_q      = traces[:, q_mask]

    peak_vals = np.max(tr_q, axis=1)
    auc_vals  = (np.trapezoid(tr_q, t_q, axis=1)
                 if hasattr(np, 'trapezoid')
                 else np.trapz(tr_q, t_q, axis=1))

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
# STEP 4 — BUILD FULL ANIMAL DATASET
# ─────────────────────────────────────────────────────────────────────────────

def build_animal_dataset(
    animal_id,
    condition,
    bouts,
    t_sero,
    dff_sero,
    min_immobility_s = DEFAULT_MIN_IMMOBILITY_S,
    pre_s            = DEFAULT_PRE_S,
    post_s           = DEFAULT_POST_S,
    quant_window     = DEFAULT_QUANT_WINDOW,
    verbose          = True,
):
    """
    Full pipeline for one animal:
      1. Find immobility gaps
      2. Extract traces per post-immobility behavior type
      3. Quantify

    Returns
    -------
    dataset : dict
        {
          'animal_id'  : str,
          'condition'  : str,
          'gaps'       : np.ndarray (n_gaps, 2),
          'next_behaviors' : list of str,
          'behaviors'  : {
              beh_code : {
                  'time_axis', 'traces', 'gap_durations',
                  'trial_meta', 'quant'
              }
          }
        }
    """
    print(f"\n── Building dataset: {animal_id} | {condition} ──")

    # 1. Find gaps
    print("\n  Finding immobility gaps...")
    gaps, next_behaviors = find_immobility_gaps(
        bouts, min_immobility_s=min_immobility_s, verbose=verbose)

    dataset = {
        'animal_id':      animal_id,
        'condition':      condition,
        'gaps':           gaps,
        'next_behaviors': next_behaviors,
        'behaviors':      {},
    }

    if len(gaps) == 0:
        print("  No immobility gaps found — check your data or lower min_immobility_s")
        return dataset

    # 2. Extract per behavior type
    behavior_codes = sorted(set(next_behaviors))
    print(f"\n  Extracting traces (pre={pre_s}s, post={post_s}s, no normalization)...")

    for beh_code in behavior_codes:
        time_axis, traces, gap_durs, meta = extract_post_immobility(
            t_sero, dff_sero, gaps, next_behaviors,
            behavior_code = beh_code,
            pre_s         = pre_s,
            post_s        = post_s,
            verbose       = verbose,
        )

        quant = quantify_trials(time_axis, traces, quant_window=quant_window)

        dataset['behaviors'][beh_code] = {
            'time_axis':     time_axis,
            'traces':        traces,
            'gap_durations': gap_durs,
            'trial_meta':    meta,
            'quant':         quant,
        }

    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# SAVE / LOAD
# ─────────────────────────────────────────────────────────────────────────────

def save_dataset(dataset, path):
    """Save dataset dict to .npy file."""
    np.save(path, dataset, allow_pickle=True)
    print(f"  Saved → {path}")


def load_dataset(path):
    """Load a saved dataset .npy file."""
    return np.load(path, allow_pickle=True).item()