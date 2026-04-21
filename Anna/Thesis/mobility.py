"""
mobility_plot_poster.py
───────────────────────
Publication-grade bar plot of mobile fraction for psi vs ctrl conditions.

Styling: FILLED bars (dark grey + professional blue), bars much closer together, matching grooming behavior poster plots.
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
# STATS HELPERS
# ─────────────────────────────────────────────────────────────────────────────
 
def _pval_to_stars(p: float) -> str:
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'
 
 
def _add_significance_bar(ax, x1, x2, y, p_val, fontsize=11):
    """Draw a significance bracket between two x positions."""
    h = y * 0.03
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color='black', linewidth=1.5)
    ax.text((x1 + x2) / 2, y + h * 1.2, _pval_to_stars(p_val),
            ha='center', va='bottom', fontsize=fontsize, fontweight='bold')
 
 
# ─────────────────────────────────────────────────────────────────────────────
# PLOT — absolute mobility (publication grade, poster style)
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
    Clean publication-grade bar plot: mobile fraction, psi vs ctrl.
    FILLED bars (dark grey + professional blue) matching grooming behavior poster style.
    Paired dots connected by lines. Significance bar. Bars much closer together.
 
    Parameters
    ----------
    psi_data    : output of collect_mobility_data() for psi animals
    ctrl_data   : output of collect_mobility_data() for ctrl animals
    output_path : if provided, saves the figure here
    """
 
    fig, ax = plt.subplots(figsize=(7, 7))

    # Professional poster colors - DARK GREY + PROFESSIONAL BLUE
    COLOR_CTRL = '#4A4A4A'      # Dark grey (saline/control)
    COLOR_PSI = '#1F77B4'       # Professional blue (psi/treatment)
    OUTLINE_COLOR = '#000000'   # Black outline for both bars

    # Get mobile data only
    cv = ctrl_data['mobile']
    pv = psi_data['mobile']
    
    ctrl_mean = np.nanmean(cv)
    psi_mean = np.nanmean(pv)
    ctrl_sem = sem(cv, nan_policy='omit')
    psi_sem = sem(pv, nan_policy='omit')
    
    # ═════════════════════════════════════════════════════════════════════════
    # STATISTICAL TESTS & PRINT P-VALUES
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("STATISTICAL ANALYSIS: MOBILITY FRACTION (Mobile)")
    print("="*70)
    
    n = min(len(cv), len(pv))
    print(f"\nSample size: {n} paired animals")
    print(f"Ctrl (Saline) Mean ± SEM: {ctrl_mean:.4f} ± {ctrl_sem:.4f}")
    print(f"PSI (Psilocybin) Mean ± SEM: {psi_mean:.4f} ± {psi_sem:.4f}")
    print(f"Mean difference: {psi_mean - ctrl_mean:.4f}")
    
    # Paired t-test
    if n >= 3:
        t_stat, p_ttest = ttest_rel(cv[:n], pv[:n])
        print(f"\nPaired t-test:")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_ttest:.6f} {_pval_to_stars(p_ttest)}")
    else:
        p_ttest = 1.0
        print(f"\nPaired t-test: N/A (n < 3)")
    
    # Wilcoxon signed-rank test (non-parametric alternative)
    if n >= 3:
        w_stat, p_wilcox = wilcoxon(cv[:n], pv[:n])
        print(f"\nWilcoxon signed-rank test (non-parametric):")
        print(f"  W-statistic: {w_stat:.4f}")
        print(f"  p-value: {p_wilcox:.6f} {_pval_to_stars(p_wilcox)}")
    else:
        p_wilcox = 1.0
        print(f"\nWilcoxon test: N/A (n < 3)")
    
    print("="*70 + "\n")
 
    # Bars — FILLED with black outlines, bars MUCH CLOSER together
    x = np.array([0, 0.35])  # Reduced spacing for bars to be much closer
    bar_width = 0.15
    
    ax.bar(x[0], ctrl_mean, bar_width, yerr=ctrl_sem,
           color=COLOR_CTRL, edgecolor=OUTLINE_COLOR, linewidth=2.5, capsize=5, zorder=2,
           error_kw=dict(elinewidth=1.5, ecolor='black'))
    ax.bar(x[1], psi_mean, bar_width, yerr=psi_sem,
           color=COLOR_PSI, edgecolor=OUTLINE_COLOR, linewidth=2.5, capsize=5, zorder=2,
           error_kw=dict(elinewidth=1.5, ecolor='black'))
 
    # Paired dots + connecting lines - white dots with black edges
    rng = np.random.default_rng(42)
    jc = rng.uniform(-0.03, 0.03, size=len(cv))
    jp = rng.uniform(-0.03, 0.03, size=len(pv))
 
    ax.scatter(x[0] + jc, cv,
               color='white', edgecolors='black', linewidths=1.2, s=70, alpha=0.9, zorder=5)
    ax.scatter(x[1] + jp, pv,
               color='white', edgecolors='black', linewidths=1.2, s=70, alpha=0.9, zorder=5)
 
    # Connect paired animals - darker gray
    for j in range(min(len(cv), len(pv))):
        ax.plot(
            [x[0] + jc[j], x[1] + jp[j]],
            [cv[j], pv[j]],
            color='#666666', linewidth=0.8, alpha=0.7, zorder=3
        )
 
    # Significance bar (paired t-test)
    n = min(len(cv), len(pv))
    if n >= 3:
        _, p = ttest_rel(cv[:n], pv[:n])
    else:
        p = 1.0
 
    y_top = max(ctrl_mean, psi_mean) * 1.08
    _add_significance_bar(ax, x[0], x[1], y_top, p)
 
    # Legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOR_CTRL, edgecolor=OUTLINE_COLOR, linewidth=2.5, label='Saline'),
        Patch(facecolor=COLOR_PSI, edgecolor=OUTLINE_COLOR, linewidth=2.5, label='Psilocybin'),
    ]
    ax.legend(handles=legend_elements, fontsize=11, frameon=True, 
              fancybox=False, edgecolor='black', framealpha=0.95)
 
    # Clean axes - match grooming behavior plot styling
    ax.set_xticks(x)
    ax.set_xticklabels(['Saline', 'Psilocybin'], fontsize=12, fontweight='600')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='600')
    ax.set_title(f'{title}\n(threshold = {mobility_threshold} cm/s)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_linewidth(0.8)
    ax.tick_params(axis='y', labelsize=11, length=3, width=0.8)
    ax.tick_params(axis='x', labelsize=11, length=0)
    ax.grid(False)
    fig.tight_layout()
 
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f'Saved → {output_path}')
 
    return fig