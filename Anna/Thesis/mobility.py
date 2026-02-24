"""
mobility_plot.py
────────────────
Plot the fraction of time spent mobile vs immobile
for psi vs ctrl conditions, averaged across animals.

No serotonin needed — just velocity CSVs.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem


def compute_mobility_fractions(
    velocity_path:      str,
    mobility_threshold: float = 5.0,
    velocity_time_col:  str   = 'Time (s)',
    velocity_val_col:   str   = 'Smoothed Velocity (cm/s)',
) -> dict[str, float]:
    """
    Compute the fraction of time spent mobile and immobile for one animal.

    Parameters
    ----------
    velocity_path       : path to smoothed velocity CSV
    mobility_threshold  : velocity >= this = mobile (default 5.0 cm/s)
    velocity_time_col   : name of time column
    velocity_val_col    : name of velocity column

    Returns
    -------
    {'mobile': float, 'immobile': float}  — fractions (sum to 1.0)
    """
    df = pd.read_csv(velocity_path).dropna(subset=[velocity_val_col])

    is_mobile   = df[velocity_val_col] >= mobility_threshold
    mobile_frac   = is_mobile.sum()   / len(is_mobile)
    immobile_frac = (~is_mobile).sum() / len(is_mobile)

    print(f"  mobile: {mobile_frac:.2%}, immobile: {immobile_frac:.2%}")
    return {'mobile': mobile_frac, 'immobile': immobile_frac}


def collect_mobility_data(
    animal_paths:       list[str],
    mobility_threshold: float = 5.0,
) -> dict[str, list[float]]:
    """
    Run compute_mobility_fractions() for each animal in a condition.

    Parameters
    ----------
    animal_paths        : list of velocity CSV paths, one per animal
    mobility_threshold  : velocity threshold in cm/s

    Returns
    -------
    {'mobile': [frac_a1, frac_a2, ...], 'immobile': [...]}
    """
    collected = {'mobile': [], 'immobile': []}

    for path in animal_paths:
        print(path.split('/')[-1])
        result = compute_mobility_fractions(path, mobility_threshold)
        collected['mobile'].append(result['mobile'])
        collected['immobile'].append(result['immobile'])

    return collected


def plot_mobility_fractions(
    psi_data:           dict[str, list[float]],
    ctrl_data:          dict[str, list[float]],
    mobility_threshold: float = 5.0,
    title:              str   = 'Mobility vs Immobility',
    ylabel:             str   = 'Fraction of Time',
) -> plt.Figure:
    """
    Bar graph with individual animal dots and paired lines,
    comparing psi vs ctrl for mobile and immobile fractions.

    Parameters
    ----------
    psi_data   : output of collect_mobility_data() for psi animals
    ctrl_data  : output of collect_mobility_data() for ctrl animals
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

    ax.bar(x - width / 2, psi_means,  width, yerr=psi_sems,
           color=psi_color,  label='Psilocybin', capsize=5,
           error_kw=dict(elinewidth=1.5, ecolor='black'), zorder=2)
    ax.bar(x + width / 2, ctrl_means, width, yerr=ctrl_sems,
           color=ctrl_color, label='Saline', capsize=5,
           error_kw=dict(elinewidth=1.5, ecolor='black'), zorder=2)

    # Individual dots + paired lines
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
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=11)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE USAGE
