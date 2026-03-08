"""
mobility_plot_thesis.py
────────────────────────
Thesis-ready figure: Fraction of Time Mobile
Clean, bold labels and titles. Outliers removed.

Single publication-grade bar plot with paired individual data points.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem, ttest_rel


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


def collect_mobility_data(
    animal_list:        list[dict],
    mobility_threshold: float = 5.0,
) -> dict[str, list[float]]:
    """Collect mobility fractions for all animals in one condition."""
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


def remove_outliers_iqr(data, k=1.5):
    """
    Remove outliers using IQR method.
    
    Parameters
    ----------
    data : list or array of values
    k    : IQR multiplier (default 1.5 is standard; use k=3 for more aggressive)
    
    Returns
    -------
    Cleaned array without outliers
    """
    data = np.array(data)
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - k * IQR
    upper_bound = Q3 + k * IQR
    return data[(data >= lower_bound) & (data <= upper_bound)]


def _pval_to_stars(p: float) -> str:
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'


def _add_significance_bar(ax, x1, x2, y, p_val, fontsize=12):
    """Draw a significance bracket between two x positions."""
    h = y * 0.03
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], 
            color='black', linewidth=1.5)
    ax.text((x1 + x2) / 2, y + h * 1.3, _pval_to_stars(p_val),
            ha='center', va='bottom', fontsize=fontsize, fontweight='bold')


def plot_mobility_thesis(
    psi_data:           dict[str, list[float]],
    ctrl_data:          dict[str, list[float]],
    mobility_threshold: float = 5.0,
    remove_outliers:    bool = True,
    outlier_k:          float = 1.5,
    output_path:        str | None = None,
) -> plt.Figure:
    """
    Thesis-ready publication figure: Fraction of Time Mobile ONLY.
    
    Parameters
    ----------
    psi_data, ctrl_data : output of collect_mobility_data()
    mobility_threshold  : velocity threshold (cm/s)
    remove_outliers     : if True, remove outliers using IQR method
    outlier_k           : IQR multiplier for outlier detection
    output_path         : if provided, saves figure here
    
    Returns
    -------
    fig : matplotlib Figure
    """
    
    # Extract mobile fraction data
    ctrl_mobile = np.array(ctrl_data['mobile'])
    psi_mobile  = np.array(psi_data['mobile'])
    
    # Remove outliers if requested
    if remove_outliers:
        ctrl_mobile_orig_len = len(ctrl_mobile)
        psi_mobile_orig_len = len(psi_mobile)
        
        ctrl_mobile = remove_outliers_iqr(ctrl_mobile, k=outlier_k)
        psi_mobile  = remove_outliers_iqr(psi_mobile, k=outlier_k)
        
        print(f'Control: {len(ctrl_mobile)}/{ctrl_mobile_orig_len} animals retained (removed {ctrl_mobile_orig_len - len(ctrl_mobile)})')
        print(f'Psilocybin: {len(psi_mobile)}/{psi_mobile_orig_len} animals retained (removed {psi_mobile_orig_len - len(psi_mobile)})')
    
    # Compute statistics
    ctrl_mean = np.nanmean(ctrl_mobile)
    psi_mean  = np.nanmean(psi_mobile)
    ctrl_sem  = sem(ctrl_mobile, nan_policy='omit')
    psi_sem   = sem(psi_mobile, nan_policy='omit')
    
    # Statistical test (paired t-test on matched animals)
    n_paired = min(len(ctrl_mobile), len(psi_mobile))
    _, p_val = ttest_rel(ctrl_mobile[:n_paired], psi_mobile[:n_paired])
    
    print(f'\nFraction of Time Mobile:')
    print(f'  Saline:      {ctrl_mean:.3f} ± {ctrl_sem:.3f}')
    print(f'  Psilocybin:  {psi_mean:.3f} ± {psi_sem:.3f}')
    print(f'  p-value: {p_val:.4f}')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(7, 6))
    
    x = np.array([0, 1])
    width = 0.35
    
    ctrl_color = '#BEBEBE'   # light grey
    psi_color  = '#606060'   # dark grey
    
    # Bars with error bars
    ax.bar(x[0] - width/2, ctrl_mean, width, yerr=ctrl_sem,
           color=ctrl_color, edgecolor='black', linewidth=1.5, capsize=6,
           error_kw=dict(elinewidth=1.5, ecolor='black'), zorder=2)
    ax.bar(x[1] + width/2, psi_mean, width, yerr=psi_sem,
           color=psi_color, edgecolor='black', linewidth=1.5, capsize=6,
           error_kw=dict(elinewidth=1.5, ecolor='black'), zorder=2)
    
    # Overlay individual data points (jittered)
    rng = np.random.default_rng(42)
    jitter_strength = 0.04
    
    ctrl_jitter = rng.uniform(-jitter_strength, jitter_strength, size=len(ctrl_mobile))
    psi_jitter  = rng.uniform(-jitter_strength, jitter_strength, size=len(psi_mobile))
    
    ax.scatter(x[0] - width/2 + ctrl_jitter, ctrl_mobile,
               color='white', edgecolors='black', s=70, zorder=5,
               linewidths=1.2, alpha=0.9)
    ax.scatter(x[1] + width/2 + psi_jitter, psi_mobile,
               color='white', edgecolors='black', s=70, zorder=5,
               linewidths=1.2, alpha=0.9)
    
    # Connect paired animals with light lines
    for i in range(n_paired):
        ax.plot([x[0] - width/2 + ctrl_jitter[i], x[1] + width/2 + psi_jitter[i]],
                [ctrl_mobile[i], psi_mobile[i]],
                color='gray', linewidth=0.7, alpha=0.4, zorder=1)
    
    # Significance bracket
    y_sig = max(ctrl_mean, psi_mean) * 1.12
    _add_significance_bar(ax, x[0] - width/2, x[1] + width/2, y_sig, p_val, fontsize=13)
    
    # Formatting
    ax.set_xticks([x[0], x[1]])
    ax.set_xticklabels(['Saline', 'Psilocybin'], fontsize=13, fontweight='bold')
    ax.set_ylabel('Fraction of Time Mobile', fontsize=14, fontweight='bold')
    ax.set_title('Mobile Behavior During Recording', fontsize=15, fontweight='bold', pad=15)
    
    ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.tick_params(axis='both', labelsize=12, length=6, width=1.2)
    
    fig.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f'\n✓ Saved → {output_path}')
    
    return fig
# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE USAGE
# ─────────────────────────────────────────────────────────────────────────────

