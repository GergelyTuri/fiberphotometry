
"""
mobility_serotonin.py
─────────────────────
Compute and plot average serotonin Z-score during mobile vs immobile
periods for psi vs ctrl conditions, averaged across animals.

PIPELINE PER ANIMAL:
    1. Load velocity CSV and serotonin CSV
    2. Label each serotonin timepoint as mobile or immobile using
       nearest velocity value and a threshold
    3. Compute mean serotonin for mobile and immobile periods
    4. Collect one mean value per animal per category

THEN ACROSS ANIMALS:
    5. Average across animals per condition (psi / ctrl)
    6. Plot bar graph with individual animal dots connected by lines
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — label serotonin timepoints as mobile or immobile
# ─────────────────────────────────────────────────────────────────────────────

def get_mobile_immobile_serotonin(
    serotonin_path:   str,
    velocity_path:    str,
    mobility_threshold: float = 5.0,
    serotonin_time_col: str   = 'Time (s)',
    serotonin_val_col:  str   = 'Z-score',
    velocity_time_col:  str   = 'Time (s)',
    velocity_val_col:   str   = 'Smoothed Velocity (cm/s)',
) -> dict[str, float]:
    """
    For one animal, compute mean serotonin Z-score during mobile
    and immobile periods.

    Each serotonin timepoint is matched to the nearest velocity
    timepoint. If velocity >= threshold → mobile, else → immobile.

    Parameters
    ----------
    serotonin_path      : path to serotonin z-score CSV
    velocity_path       : path to smoothed velocity CSV
    mobility_threshold  : velocity threshold in cm/s (default 5.0)
    serotonin_time_col  : time column name in serotonin CSV
    serotonin_val_col   : z-score column name in serotonin CSV
    velocity_time_col   : time column name in velocity CSV
    velocity_val_col    : velocity column name in velocity CSV

    Returns
    -------
    {'mobile': float, 'immobile': float}  — mean z-score per state
    """
    sero_df = pd.read_csv(serotonin_path)
    velo_df = pd.read_csv(velocity_path).dropna(subset=[velocity_val_col])

    sero_df[serotonin_time_col] = pd.to_numeric(sero_df[serotonin_time_col], errors='coerce')
    velo_df[velocity_time_col]  = pd.to_numeric(velo_df[velocity_time_col],  errors='coerce')
    sero_df = sero_df.dropna(subset=[serotonin_time_col, serotonin_val_col])

    # For each serotonin timepoint, find the nearest velocity value
    sero_times = sero_df[serotonin_time_col].to_numpy()
    velo_times = velo_df[velocity_time_col].to_numpy()
    velo_vals  = velo_df[velocity_val_col].to_numpy()

    # Match each serotonin time to nearest velocity time
    indices  = np.searchsorted(velo_times, sero_times, side='left')
    indices  = np.clip(indices, 0, len(velo_times) - 1)
    matched_velocity = velo_vals[indices]

    # Label mobile vs immobile
    is_mobile   = matched_velocity >= mobility_threshold
    sero_vals   = sero_df[serotonin_val_col].to_numpy()

    mobile_mean   = np.nanmean(sero_vals[is_mobile])   if is_mobile.any()  else np.nan
    immobile_mean = np.nanmean(sero_vals[~is_mobile])  if (~is_mobile).any() else np.nan

    print(f"  mobile pts: {is_mobile.sum()}, immobile pts: {(~is_mobile).sum()}")
    print(f"  mobile mean: {mobile_mean:.3f}, immobile mean: {immobile_mean:.3f}")

    return {'mobile': mobile_mean, 'immobile': immobile_mean}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — run across all animals and collect per-animal means
# ─────────────────────────────────────────────────────────────────────────────

def collect_condition_data(
    animal_list:        list[dict],
    mobility_threshold: float = 5.0,
) -> dict[str, list[float]]:
    """
    Run get_mobile_immobile_serotonin() for each animal in a condition
    and collect the per-animal means.

    Parameters
    ----------
    animal_list : list of dicts, each with keys:
                    'serotonin_path' and 'velocity_path'
    mobility_threshold : velocity threshold in cm/s

    Returns
    -------
    {'mobile': [mean_a1, mean_a2, ...], 'immobile': [...]}

    Example
    -------
    psi_animals = [
        {'serotonin_path': '.../nia35_pcb_zscore.csv',
         'velocity_path':  '.../nia35_pcb_velocity.csv'},
        {'serotonin_path': '.../nia41_pcb_zscore.csv',
         'velocity_path':  '.../nia41_pcb_velocity.csv'},
    ]
    psi_data = collect_condition_data(psi_animals)
    """
    collected = {'mobile': [], 'immobile': []}

    for i, animal in enumerate(animal_list):
        print(f"Animal {i+1}: {animal['serotonin_path'].split('/')[-1]}")
        result = get_mobile_immobile_serotonin(
            serotonin_path     = animal['serotonin_path'],
            velocity_path      = animal['velocity_path'],
            mobility_threshold = mobility_threshold,
        )
        collected['mobile'].append(result['mobile'])
        collected['immobile'].append(result['immobile'])

    return collected


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_mobile_immobile(
    psi_data:  dict[str, list[float]],
    ctrl_data: dict[str, list[float]],
    mobility_threshold: float = 5.0,
    title:     str  = 'Serotonin: Mobile vs Immobile',
    ylabel:    str  = 'Mean Z-score',
) -> plt.Figure:
    """
    Bar graph with individual animal dots and paired lines,
    comparing psi vs ctrl for mobile and immobile states.

    Parameters
    ----------
    psi_data  : output of collect_condition_data() for psilocybin animals
    ctrl_data : output of collect_condition_data() for control animals
    mobility_threshold : shown in the title for reference
    title     : plot title
    ylabel    : y-axis label
    """
    categories = ['mobile', 'immobile']
    x          = np.arange(len(categories))
    width      = 0.35

    psi_color  = '#E87722'
    ctrl_color = '#4878CF'
    psi_dot    = '#8b3d00'
    ctrl_dot   = '#1a3d6e'

    psi_means  = [np.nanmean(psi_data[c])  for c in categories]
    ctrl_means = [np.nanmean(ctrl_data[c]) for c in categories]
    psi_sems   = [sem(psi_data[c],  nan_policy='omit') for c in categories]
    ctrl_sems  = [sem(ctrl_data[c], nan_policy='omit') for c in categories]

    fig, ax = plt.subplots(figsize=(7, 6))

    # Bars
    ax.bar(x - width / 2, psi_means,  width, yerr=psi_sems,
           color=psi_color,  label='Psilocybin', capsize=5,
           error_kw=dict(elinewidth=1.5, ecolor='black'), zorder=2)
    ax.bar(x + width / 2, ctrl_means, width, yerr=ctrl_sems,
           color=ctrl_color, label='Control',    capsize=5,
           error_kw=dict(elinewidth=1.5, ecolor='black'), zorder=2)

    # Individual animal dots + paired lines
    rng = np.random.default_rng(42)
    for i, cat in enumerate(categories):
        psi_vals  = psi_data[cat]
        ctrl_vals = ctrl_data[cat]

        jp = rng.uniform(-0.06, 0.06, size=len(psi_vals))
        jc = rng.uniform(-0.06, 0.06, size=len(ctrl_vals))

        ax.scatter(x[i] - width / 2 + jp, psi_vals,
                   color=psi_dot,  s=60, zorder=5, alpha=0.9)
        ax.scatter(x[i] + width / 2 + jc, ctrl_vals,
                   color=ctrl_dot, s=60, zorder=5, alpha=0.9)

        # Connect paired animals (assumes same order in both lists)
        n_pairs = min(len(psi_vals), len(ctrl_vals))
        for j in range(n_pairs):
            ax.plot(
                [x[i] - width / 2 + jp[j], x[i] + width / 2 + jc[j]],
                [psi_vals[j], ctrl_vals[j]],
                color='gray', linewidth=0.9, alpha=0.6, zorder=3
            )

    ax.set_xticks(x)
    ax.set_xticklabels(['Mobile', 'Immobile'], fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(f'{title}\n(threshold = {mobility_threshold} cm/s)', fontsize=13)
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)
    ax.legend(fontsize=11)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()

    return fig