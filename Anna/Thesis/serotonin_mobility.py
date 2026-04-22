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

def _pval_to_stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'NS'


def _add_significance_bar(ax, x1, x2, y, p_val, fontsize=11):
    h = abs(y) * 0.05 + 0.05
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y],
            color='black', linewidth=1.2)
    ax.text((x1 + x2) / 2, y + h * 1.3, _pval_to_stars(p_val),
            ha='center', va='bottom', fontsize=fontsize)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — plot
# ─────────────────────────────────────────────────────────────────────────────
def plot_mobile_immobile(
    psi_data,
    ctrl_data,
    mobility_threshold=5.0,
    title='Avg Serotonin: Mobile vs Immobile',
    ylabel='Mean Z-score',
    output_path=None,
    ctrl_color='#4A4A4A',  # Dark grey for saline/control
    psi_color='#1F77B4',   # Professional blue for psilocybin
):
    """
    Publication-grade bar graph with grey/blue styling.
    Saline     = dark grey (#4A4A4A)
    Psilocybin = professional blue (#1F77B4)
    Dots       = white with black outline, paired lines in grey.
    Significance bar per category (paired t-test).

    Parameters
    ----------
    psi_data    : output of collect_condition_data() for psilocybin animals
    ctrl_data   : output of collect_condition_data() for control animals
    output_path : if provided, saves figure here at 300 dpi
    ctrl_color  : hex color for saline condition (default: #4A4A4A - dark grey)
    psi_color   : hex color for psilocybin condition (default: #1F77B4 - blue)
    """
    from matplotlib.patches import Patch
    from scipy.stats import ttest_rel

    categories = ['mobile', 'immobile']
    xlabels    = ['Mobile', 'Immobile']
    x          = np.arange(len(categories))
    width      = 0.35

    psi_means  = [np.nanmean(psi_data[c])  for c in categories]
    ctrl_means = [np.nanmean(ctrl_data[c]) for c in categories]
    psi_sems   = [sem(psi_data[c],  nan_policy='omit') for c in categories]
    ctrl_sems  = [sem(ctrl_data[c], nan_policy='omit') for c in categories]

    fig, ax = plt.subplots(figsize=(7, 6))

    # Bars — control (ctrl) left, psi right (with black borders)
    ax.bar(x - width / 2, ctrl_means, width, yerr=ctrl_sems,
           color=ctrl_color, edgecolor='black', linewidth=2.5,
           label='Saline', capsize=5,
           error_kw=dict(elinewidth=2.0, ecolor='black'), zorder=2)
    ax.bar(x + width / 2, psi_means, width, yerr=psi_sems,
           color=psi_color, edgecolor='black', linewidth=2.5,
           label='Psilocybin', capsize=5,
           error_kw=dict(elinewidth=2.0, ecolor='black'), zorder=2)

    # Individual dots + paired connecting lines (white with black outline)
    rng = np.random.default_rng(42)
    for i, cat in enumerate(categories):
        cv = ctrl_data[cat]
        pv = psi_data[cat]

        jc = rng.uniform(-0.04, 0.04, size=len(cv))
        jp = rng.uniform(-0.04, 0.04, size=len(pv))

        ax.scatter(x[i] - width / 2 + jc, cv,
                   color='white', edgecolors='black', s=75,
                   linewidths=1.2, zorder=5)
        ax.scatter(x[i] + width / 2 + jp, pv,
                   color='white', edgecolors='black', s=75,
                   linewidths=1.2, zorder=5)

        n_pairs = min(len(cv), len(pv))
        for j in range(n_pairs):
            ax.plot(
                [x[i] - width / 2 + jc[j], x[i] + width / 2 + jp[j]],
                [cv[j], pv[j]],
                color='gray', linewidth=0.8, alpha=0.4, zorder=3
            )

        # Significance bar
        if n_pairs >= 3:
            _, p = ttest_rel(cv[:n_pairs], pv[:n_pairs])
        else:
            p = 1.0

        all_vals = [v for v in list(cv) + list(pv) if not np.isnan(v)]
        y_top = max(all_vals)
        _add_significance_bar(ax, x[i] - width / 2, x[i] + width / 2, y_top, p)

    # Legend
    legend_elements = [
        Patch(facecolor=ctrl_color, edgecolor='black', linewidth=2.0, label='Saline'),
        Patch(facecolor=psi_color, edgecolor='black', linewidth=2.0, label='Psilocybin'),
    ]
    ax.legend(handles=legend_elements, fontsize=12, frameon=True, framealpha=0.95, edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=13, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
    ax.set_title(f'{title}\n(threshold = {mobility_threshold} cm/s)', fontsize=13, fontweight='bold')
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.4)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_linewidth(1.5)
    ax.tick_params(axis='both', labelsize=11)
    ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] * 1.2)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f'Saved -> {output_path}')

    return fig

def plot_serotonin_velocity_overlay(
    serotonin_path:     str,
    velocity_path:      str,
    animal_name:        str   = '',
    mobility_threshold: float = 7.0,
    serotonin_time_col: str   = 'Time (s)',
    serotonin_val_col:  str   = 'Z-score',
    velocity_time_col:  str   = 'Time (s)',
    velocity_val_col:   str   = 'Smoothed Velocity (cm/s)',
    output_path:        str   | None = None,
) -> plt.Figure:
    """
    Plot serotonin z-score (top) and velocity (bottom) on a shared time axis
    for one animal. Shades mobile periods in both panels so you can see
    exactly when the animal was moving and what the serotonin was doing.

    Parameters
    ----------
    serotonin_path      : path to serotonin z-score CSV
    velocity_path       : path to smoothed velocity CSV
    animal_name         : label for the plot title
    mobility_threshold  : velocity threshold used to define mobile (cm/s)
    output_path         : if provided, saves the figure here
    """
    sero_df = pd.read_csv(serotonin_path)
    velo_df = pd.read_csv(velocity_path).dropna(subset=[velocity_val_col])

    sero_df[serotonin_time_col] = pd.to_numeric(sero_df[serotonin_time_col], errors='coerce')
    velo_df[velocity_time_col]  = pd.to_numeric(velo_df[velocity_time_col],  errors='coerce')
    sero_df = sero_df.dropna(subset=[serotonin_time_col, serotonin_val_col])

    sero_time = sero_df[serotonin_time_col].to_numpy()
    sero_vals = sero_df[serotonin_val_col].to_numpy()
    velo_time = velo_df[velocity_time_col].to_numpy()
    velo_vals = velo_df[velocity_val_col].to_numpy()

    # Find mobile periods in velocity
    is_mobile = velo_vals >= mobility_threshold

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 7))
    from matplotlib import gridspec
    gs  = gridspec.GridSpec(2, 1, hspace=0.08)

    ax1 = fig.add_subplot(gs[0])  # serotonin
    ax2 = fig.add_subplot(gs[1], sharex=ax1)  # velocity

    # Shade mobile periods on both panels
    in_mobile = False
    mobile_start = None
    for i, mobile in enumerate(is_mobile):
        t = velo_time[i]
        if mobile and not in_mobile:
            mobile_start = t
            in_mobile = True
        elif not mobile and in_mobile:
            ax1.axvspan(mobile_start, t, alpha=0.12, color='green', zorder=0)
            ax2.axvspan(mobile_start, t, alpha=0.12, color='green', zorder=0)
            in_mobile = False
    # close last span if still open
    if in_mobile:
        ax1.axvspan(mobile_start, velo_time[-1], alpha=0.12, color='green', zorder=0)
        ax2.axvspan(mobile_start, velo_time[-1], alpha=0.12, color='green', zorder=0)

    # Serotonin panel
    ax1.plot(sero_time, sero_vals, color='#C1440E', linewidth=1.0, label='Serotonin Z-score')
    ax1.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax1.set_ylabel('Serotonin Z-score', fontsize=12)
    ax1.spines[['top', 'right']].set_visible(False)
    ax1.tick_params(labelbottom=False)

    # Mark mean serotonin during mobile vs immobile
    # Match each serotonin time to nearest velocity
    indices          = np.searchsorted(velo_time, sero_time, side='left')
    indices          = np.clip(indices, 0, len(velo_time) - 1)
    matched_mobile   = velo_vals[indices] >= mobility_threshold
    mobile_mean      = np.nanmean(sero_vals[matched_mobile])
    immobile_mean    = np.nanmean(sero_vals[~matched_mobile])
    ax1.axhline(mobile_mean,   color='green',  linestyle=':', linewidth=1.2,
                label=f'Mobile mean: {mobile_mean:.2f}')
    ax1.axhline(immobile_mean, color='purple', linestyle=':', linewidth=1.2,
                label=f'Immobile mean: {immobile_mean:.2f}')
    ax1.legend(fontsize=9, frameon=False, loc='upper right')

    # Velocity panel
    ax2.plot(velo_time, velo_vals, color='#4878CF', linewidth=1.0, label='Velocity')
    ax2.axhline(mobility_threshold, color='green', linestyle='--',
                linewidth=1.0, label=f'Threshold ({mobility_threshold} cm/s)')
    ax2.set_ylabel('Velocity (cm/s)', fontsize=12)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.spines[['top', 'right']].set_visible(False)
    ax2.legend(fontsize=9, frameon=False, loc='upper right')

    fig.suptitle(f'{animal_name}  |  green shading = mobile periods',
                 fontsize=13, y=1.01)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'Saved → {output_path}')

    return fig

def plot_all_animals(
    animal_list: list[dict],
    mobility_threshold: float = 7.0,
    output_folder: str | None = None,
) -> None:
    """
    Run plot_serotonin_velocity_overlay() for every animal in a list.

    Parameters
    ----------
    animal_list : list of dicts with keys:
                    'serotonin_path'
                    'velocity_path'
                    'name'           — label for the plot title
    mobility_threshold : velocity threshold in cm/s
    output_folder : if provided, saves each figure as {name}_overlay.png
    """
    for animal in animal_list:
        out = None
        if output_folder:
            out = f"{output_folder}/{animal['name']}_overlay.png"

        plot_serotonin_velocity_overlay(
            serotonin_path     = animal['serotonin_path'],
            velocity_path      = animal['velocity_path'],
            animal_name        = animal['name'],
            mobility_threshold = mobility_threshold,
            output_path        = out,
        )
        plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE USAGE
# ─────────────────────────────────────────────────────────────────────────────

