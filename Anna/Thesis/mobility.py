"""
mobility_plot.py
────────────────
Publication-grade bar plot of mobile vs immobile fraction
for psi vs ctrl conditions, averaged across animals.

TWO PLOTS:
  1. Absolute fraction of time mobile/immobile during recording
  2. Change in mobility fraction from baseline to recording

Styling: clean, minimal, publication-ready with significance bar.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem, wilcoxon, ttest_rel


# ─────────────────────────────────────────────────────────────────────────────
# CORE — compute mobile/immobile fraction for one CSV
# ─────────────────────────────────────────────────────────────────────────────

def compute_mobility_fractions(
    velocity_path:      str,
    mobility_threshold: float = 5.0,
    velocity_val_col:   str   = 'Smoothed Velocity (cm/s)',
) -> dict[str, float]:
    """Compute fraction of time mobile and immobile for one CSV."""
    df        = pd.read_csv(velocity_path).dropna(subset=[velocity_val_col])
    is_mobile = df[velocity_val_col] >= mobility_threshold
    return {
        'mobile':   is_mobile.sum()    / len(is_mobile),
        'immobile': (~is_mobile).sum() / len(is_mobile),
    }


# ─────────────────────────────────────────────────────────────────────────────
# COLLECT — run across all animals in a condition
# ─────────────────────────────────────────────────────────────────────────────

def collect_mobility_data(
    animal_list:        list[dict],
    mobility_threshold: float = 5.0,
) -> dict[str, list[float]]:
    """
    Collect mobility fractions for all animals in one condition.

    Parameters
    ----------
    animal_list : list of dicts with keys:
                    'recording_path' — velocity CSV for the recording
                    'baseline_path'  — velocity CSV for the baseline
    mobility_threshold : velocity threshold in cm/s

    Returns
    -------
    {
      'mobile':          [frac_a1, ...],
      'immobile':        [frac_a1, ...],
      'mobile_change':   [delta_a1, ...],  # recording - baseline
      'immobile_change': [delta_a1, ...],
    }
    """
    collected = {'mobile': [], 'immobile': [],
                 'mobile_change': [], 'immobile_change': []}

    for animal in animal_list:
        print(animal['recording_path'].split('/')[-1])
        rec  = compute_mobility_fractions(animal['recording_path'], mobility_threshold)
        base = compute_mobility_fractions(animal['baseline_path'],  mobility_threshold)
        collected['mobile'].append(rec['mobile'])
        collected['immobile'].append(rec['immobile'])
        collected['mobile_change'].append(rec['mobile']    - base['mobile'])
        collected['immobile_change'].append(rec['immobile'] - base['immobile'])

    return collected


# ─────────────────────────────────────────────────────────────────────────────
# STATS HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _pval_to_stars(p: float) -> str:
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'


def _add_significance_bar(ax, x1, x2, y, p_val, fontsize=11):
    """Draw a significance bracket between two x positions."""
    h = y * 0.03
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color='black', linewidth=1.2)
    ax.text((x1 + x2) / 2, y + h * 1.2, _pval_to_stars(p_val),
            ha='center', va='bottom', fontsize=fontsize)


# ─────────────────────────────────────────────────────────────────────────────
# PLOT — absolute mobility only (publication grade)
# ─────────────────────────────────────────────────────────────────────────────

def plot_mobility(
    psi_data:           dict[str, list[float]],
    ctrl_data:          dict[str, list[float]],
    mobility_threshold: float = 5.0,
    title:              str   = 'Mobility During Recording',
    ylabel:             str   = 'Fraction of Time',
    output_path:        str   | None = None,
) -> plt.Figure:
    """
    Clean publication-grade bar plot: mobile only, psi vs ctrl.
    Paired dots connected by lines. Significance bar.

    Parameters
    ----------
    psi_data    : output of collect_mobility_data() for psi animals
    ctrl_data   : output of collect_mobility_data() for ctrl animals
    output_path : if provided, saves the figure here
    """
    ctrl_color  = '#BEBEBE'   # light grey (saline)
    psi_color   = '#606060'   # dark grey (psilocybin)

    fig, ax = plt.subplots(figsize=(6, 6))

    # Get mobile data only
    cv = ctrl_data['mobile']
    pv = psi_data['mobile']
    
    ctrl_mean = np.nanmean(cv)
    psi_mean = np.nanmean(pv)
    ctrl_sem = sem(cv, nan_policy='omit')
    psi_sem = sem(pv, nan_policy='omit')

    # Bars — ctrl on left, psi on right
    x = np.array([0, 1])
    width = 0.15
    
    ax.bar(x[0] - width / 2, ctrl_mean, width, yerr=ctrl_sem,
           color=ctrl_color, capsize=5, zorder=2,
           error_kw=dict(elinewidth=1.5, ecolor='black'))
    ax.bar(x[1] + width / 2, psi_mean, width, yerr=psi_sem,
           color=psi_color, capsize=5, zorder=2,
           error_kw=dict(elinewidth=1.5, ecolor='black'))

    # Paired dots + connecting lines
    rng = np.random.default_rng(42)
    jc = rng.uniform(-0.04, 0.04, size=len(cv))
    jp = rng.uniform(-0.04, 0.04, size=len(pv))

    ax.scatter(x[0] - width / 2 + jc, cv,
               color='white', edgecolors='black', s=55, zorder=5,
               linewidths=1.2)
    ax.scatter(x[1] + width / 2 + jp, pv,
               color='white', edgecolors='black', s=55, zorder=5,
               linewidths=1.2)

    # Connect paired animals
    for j in range(min(len(cv), len(pv))):
        ax.plot(
            [x[0] - width / 2 + jc[j], x[1] + width / 2 + jp[j]],
            [cv[j], pv[j]],
            color='gray', linewidth=0.8, alpha=0.5, zorder=3
        )

    # Significance bar (paired t-test)
    n = min(len(cv), len(pv))
    if n >= 3:
        _, p = ttest_rel(cv[:n], pv[:n])
    else:
        p = 1.0

    y_top = max(ctrl_mean, psi_mean) * 1.08
    _add_significance_bar(ax,
                          x[0] - width / 2,
                          x[1] + width / 2,
                          y_top, p)

    # Legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#BEBEBE', edgecolor='black', label='Saline'),
        Patch(facecolor='#606060', edgecolor='black', label='Psilocybin'),
    ]
    ax.legend(handles=legend_elements, fontsize=11, frameon=False)

    # Clean axes
    ax.set_xticks(x)
    ax.set_xticklabels(['Saline', 'Psilocybin'], fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(f'{title}\n(threshold = {mobility_threshold} cm/s)', fontsize=13)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='both', labelsize=11)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f'Saved → {output_path}')

    return fig
# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE USAGE
# ─────────────────────────────────────────────────────────────────────────────

