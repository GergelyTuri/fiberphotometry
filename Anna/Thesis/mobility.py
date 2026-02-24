"""
mobility_plot.py
────────────────
Plot mobility vs immobility for psi vs ctrl conditions, averaged across animals.

TWO PLOTS:
  1. Absolute fraction of time mobile/immobile during recording
  2. Change in mobility fraction from baseline to recording
     (recording fraction - baseline fraction)

Only needs velocity CSVs — no serotonin required.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem


# ─────────────────────────────────────────────────────────────────────────────
# CORE — compute mobile/immobile fraction for one CSV
# ─────────────────────────────────────────────────────────────────────────────

def compute_mobility_fractions(
    velocity_path:      str,
    mobility_threshold: float = 5.0,
    velocity_val_col:   str   = 'Smoothed Velocity (cm/s)',
) -> dict[str, float]:
    """
    Compute fraction of time spent mobile and immobile for one CSV.

    Returns
    -------
    {'mobile': float, 'immobile': float}  — fractions that sum to 1.0
    """
    df        = pd.read_csv(velocity_path).dropna(subset=[velocity_val_col])
    is_mobile = df[velocity_val_col] >= mobility_threshold

    mobile_frac   = is_mobile.sum()    / len(is_mobile)
    immobile_frac = (~is_mobile).sum() / len(is_mobile)

    print(f"  mobile: {mobile_frac:.2%}, immobile: {immobile_frac:.2%}")
    return {'mobile': mobile_frac, 'immobile': immobile_frac}


# ─────────────────────────────────────────────────────────────────────────────
# COLLECT — run across all animals in a condition
# ─────────────────────────────────────────────────────────────────────────────

def collect_mobility_data(
    animal_list:        list[dict],
    mobility_threshold: float = 5.0,
) -> dict[str, list[float]]:
    """
    Collect mobility fractions and baseline-normalized changes for all animals
    in one condition.

    Parameters
    ----------
    animal_list : list of dicts, each with keys:
                    'recording_path' — velocity CSV for the recording
                    'baseline_path'  — velocity CSV for the baseline
    mobility_threshold : velocity threshold in cm/s

    Returns
    -------
    {
      'mobile':          [frac_a1, ...],   # absolute mobile fraction
      'immobile':        [frac_a1, ...],   # absolute immobile fraction
      'mobile_change':   [delta_a1, ...],  # recording - baseline mobile
      'immobile_change': [delta_a1, ...],  # recording - baseline immobile
    }
    """
    collected = {
        'mobile':          [],
        'immobile':        [],
        'mobile_change':   [],
        'immobile_change': [],
    }

    for animal in animal_list:
        name = animal['recording_path'].split('/')[-1]
        print(f"\n{name}")

        rec  = compute_mobility_fractions(animal['recording_path'], mobility_threshold)
        base = compute_mobility_fractions(animal['baseline_path'],  mobility_threshold)

        collected['mobile'].append(rec['mobile'])
        collected['immobile'].append(rec['immobile'])
        collected['mobile_change'].append(rec['mobile']   - base['mobile'])
        collected['immobile_change'].append(rec['immobile'] - base['immobile'])

    return collected


# ─────────────────────────────────────────────────────────────────────────────
# PLOT HELPER — shared bar + dots + lines logic
# ─────────────────────────────────────────────────────────────────────────────

def _bar_plot(ax, psi_vals, ctrl_vals, labels, ylabel, title,
              psi_color, ctrl_color, psi_dot, ctrl_dot):
    x     = np.arange(len(labels))
    width = 0.35
    rng   = np.random.default_rng(42)

    psi_means  = [np.nanmean(psi_vals[c])  for c in labels]
    ctrl_means = [np.nanmean(ctrl_vals[c]) for c in labels]
    psi_sems   = [sem(psi_vals[c],  nan_policy='omit') for c in labels]
    ctrl_sems  = [sem(ctrl_vals[c], nan_policy='omit') for c in labels]

    ax.bar(x - width / 2, psi_means,  width, yerr=psi_sems,
           color=psi_color,  label='Psilocybin', capsize=5,
           error_kw=dict(elinewidth=1.5, ecolor='black'), zorder=2)
    ax.bar(x + width / 2, ctrl_means, width, yerr=ctrl_sems,
           color=ctrl_color, label='Saline', capsize=5,
           error_kw=dict(elinewidth=1.5, ecolor='black'), zorder=2)

    for i, cat in enumerate(labels):
        pv = psi_vals[cat]
        cv = ctrl_vals[cat]
        jp = rng.uniform(-0.06, 0.06, size=len(pv))
        jc = rng.uniform(-0.06, 0.06, size=len(cv))

        ax.scatter(x[i] - width / 2 + jp, pv, color=psi_dot,  s=60, zorder=5, alpha=0.9)
        ax.scatter(x[i] + width / 2 + jc, cv, color=ctrl_dot, s=60, zorder=5, alpha=0.9)

        for j in range(min(len(pv), len(cv))):
            ax.plot(
                [x[i] - width / 2 + jp[j], x[i] + width / 2 + jc[j]],
                [pv[j], cv[j]],
                color='gray', linewidth=0.9, alpha=0.6, zorder=3
            )

    ax.set_xticks(x)
    ax.set_xticklabels(['Mobile', 'Immobile'], fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=13)
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)
    ax.legend(fontsize=11)
    ax.spines[['top', 'right']].set_visible(False)


# ─────────────────────────────────────────────────────────────────────────────
# PLOT — both absolute and change-from-baseline side by side
# ─────────────────────────────────────────────────────────────────────────────

def plot_mobility(
    psi_data:           dict[str, list[float]],
    ctrl_data:          dict[str, list[float]],
    mobility_threshold: float = 5.0,
) -> plt.Figure:
    """
    Two side-by-side plots:
      Left  — absolute fraction of time mobile/immobile during recording
      Right — change from baseline (recording - baseline)

    Parameters
    ----------
    psi_data  : output of collect_mobility_data() for psi animals
    ctrl_data : output of collect_mobility_data() for ctrl animals
    """
    psi_color  = '#E87722'
    ctrl_color = '#4878CF'
    psi_dot    = '#8b3d00'
    ctrl_dot   = '#1a3d6e'

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: absolute fractions
    psi_abs  = {'mobile': psi_data['mobile'],   'immobile': psi_data['immobile']}
    ctrl_abs = {'mobile': ctrl_data['mobile'],  'immobile': ctrl_data['immobile']}
    _bar_plot(axes[0], psi_abs, ctrl_abs,
              labels=['mobile', 'immobile'],
              ylabel='Fraction of Time',
              title=f'Mobility During Recording\n(threshold = {mobility_threshold} cm/s)',
              psi_color=psi_color, ctrl_color=ctrl_color,
              psi_dot=psi_dot, ctrl_dot=ctrl_dot)
    axes[0].set_ylim(0, 1.0)

    # Right: change from baseline
    psi_chg  = {'mobile': psi_data['mobile_change'],   'immobile': psi_data['immobile_change']}
    ctrl_chg = {'mobile': ctrl_data['mobile_change'],  'immobile': ctrl_data['immobile_change']}
    _bar_plot(axes[1], psi_chg, ctrl_chg,
              labels=['mobile', 'immobile'],
              ylabel='Change in Fraction (Recording − Baseline)',
              title=f'Change in Mobility from Baseline\n(threshold = {mobility_threshold} cm/s)',
              psi_color=psi_color, ctrl_color=ctrl_color,
              psi_dot=psi_dot, ctrl_dot=ctrl_dot)

    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE USAGE
# ─────────────────────────────────────────────────────────────────────────────

