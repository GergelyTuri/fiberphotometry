"""
periplot_behavior_structure.py
─────────────────────────────────────────────────────────────────────────────
Behavioral structure metrics from BORIS bout data:

  1. bout_duration_figure()      — mean bout duration per behavior per condition
  2. fragmentation_figure()      — bout count, mean duration, inter-bout interval
  3. sequence_predictability()   — transition matrix + entropy score

Updated: Fragmentation figure now uses thesis-ready layout with three
independent metrics (Count, Duration, IBI) shown in separate figures.

Usage:
    from periplot_behavior_structure import (
        bout_duration_figure,
        fragmentation_figure,
        sequence_predictability,
        behavior_summary_table,
    )
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import sem as scipy_sem, mannwhitneyu, ttest_ind
from collections import Counter
import itertools

CONDITION_COLORS = {
    'saline':  '#4A4A4A',     # Dark grey
    'vehicle': '#4A4A4A',     # Dark grey
    'psi':     '#1F77B4',     # Professional blue
    'drug':    '#1F77B4',     # Professional blue
}
BEHAVIOR_LABELS = {
    'e': 'Exploring',
    'g': 'Grooming',
    'd': 'Digging',
    'r': 'Rearing',
    'eat': 'Eating',
}
BEHAVIOR_COLORS = {
    'e': '#4CAF93',
    'g': '#9B59B6',
    'd': '#E67E22',
    'r': '#E74C3C',
    'eat': '#F39C12',
}

def _cond_color(c):
    return CONDITION_COLORS.get(c.lower(), '#555')

def _p_stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'


# ─────────────────────────────────────────────────────────────────────────────
# 1. BOUT DURATION
# ─────────────────────────────────────────────────────────────────────────────

def bout_duration_figure(animal_datasets, outdir='.', save=True, test='mwu'):
    """
    Bar + strip plot of mean bout duration per behavior, split by condition.

    Parameters
    ----------
    animal_datasets : list of dict
        Each dict: {'animal_id', 'condition', 'bouts'}
        where bouts = output of load_behavior()
    """
    os.makedirs(outdir, exist_ok=True)

    behaviors  = sorted({b for d in animal_datasets for b in d['bouts']})
    conditions = list(dict.fromkeys(d['condition'] for d in animal_datasets))
    n_behs     = len(behaviors)

    fig, axes = plt.subplots(1, n_behs, figsize=(4.5 * n_behs, 5),
                              gridspec_kw={'wspace': 0.45})
    if n_behs == 1:
        axes = [axes]

    fig.suptitle('Bout Duration by Behavior and Condition',
                 fontsize=13, fontweight='bold')

    for ax, beh in zip(axes, behaviors):
        ax.spines[['top', 'right']].set_visible(False)
        beh_name = BEHAVIOR_LABELS.get(beh, beh)
        beh_col  = BEHAVIOR_COLORS.get(beh, '#555')

        cond_vals = {}
        for xi, cond in enumerate(conditions):
            cond_animals = [d for d in animal_datasets if d['condition'] == cond]
            # Mean bout duration per animal
            per_animal_means = []
            for d in cond_animals:
                bouts = d['bouts'].get(beh, np.empty((0, 2)))
                if len(bouts) == 0:
                    continue
                durs = bouts[:, 1] - bouts[:, 0]
                per_animal_means.append(np.mean(durs))

            vals = np.array(per_animal_means)
            cond_vals[cond] = vals
            color = _cond_color(cond)

            if len(vals) == 0:
                continue

            ax.bar(xi, np.mean(vals),
                   yerr=scipy_sem(vals) if len(vals) > 1 else 0,
                   color=color, alpha=0.7, width=0.55, capsize=5,
                   error_kw={'lw': 2}, zorder=2, edgecolor='black', linewidth=2.0)
            jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(vals))
            ax.scatter(xi + jitter, vals, color='white', s=50, alpha=0.85,
                       edgecolors='black', linewidths=1.2, zorder=3)

        # Stats
        if test and len(conditions) >= 2:
            g1 = cond_vals.get(conditions[0], np.array([]))
            g2 = cond_vals.get(conditions[1], np.array([]))
            if len(g1) > 1 and len(g2) > 1:
                _, pval = mannwhitneyu(g1, g2, alternative='two-sided')
                y_br = ax.get_ylim()[1] * 1.08
                ax.plot([0, 1], [y_br, y_br], color='black', lw=1.0)
                ax.text(0.5, y_br * 1.01, _p_stars(pval) + f'\np={pval:.3f}',
                        ha='center', va='bottom', fontsize=7)

        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels(conditions, fontsize=9)
        ax.set_ylabel('Mean bout duration (s)', fontsize=9)
        ax.set_title(beh_name, fontsize=11, fontweight='bold', color=beh_col)

    if save:
        fname = os.path.join(outdir, 'bout_duration.png')
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved → {fname}")
        return fname
    plt.show()
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 2. FRAGMENTATION FIGURE — THESIS LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

def fragmentation_figure(animal_datasets, session_duration_s=1800.0,
                         outdir='.', save=True, test='mwu'):
    """
    Three fragmentation metrics shown in thesis-ready layout:
      - Bout count: number of bouts per behavior
      - Mean duration: average duration per bout
      - Mean IBI: inter-bout interval (time between consecutive onsets)

    Figure layout:
      Three separate figures, one per metric.
      Each figure shows all behaviors side-by-side with paired bars (condition 1 vs condition 2).
      Styled with grey for saline and blue for PSI.
      
    This layout is publication-ready and independent of behavior count.
    
    Parameters
    ----------
    animal_datasets : list of dict
        Each dict: {'animal_id', 'condition', 'bouts'}
    session_duration_s : float
        Length of recording session (default 1800s = 30 min)
    outdir : str
        Output directory for saving figures
    save : bool
        Whether to save figures
    test : str
        Statistical test: 'mwu' (Mann-Whitney U) or 't' (t-test)
    """
    os.makedirs(outdir, exist_ok=True)

    behaviors  = sorted({b for d in animal_datasets for b in d['bouts']})
    conditions = list(dict.fromkeys(d['condition'] for d in animal_datasets))
    
    if len(conditions) != 2:
        print(f"⚠ Warning: Expected 2 conditions, found {len(conditions)}")

    # ── DEFINE METRICS ───────────────────────────────────────────────────────
    def metric_count(durs, onsets, offsets):
        """Total number of bouts"""
        return len(durs)

    def metric_duration(durs, onsets, offsets):
        """Mean duration per bout"""
        return np.mean(durs) if len(durs) > 0 else 0

    def metric_ibi(durs, onsets, offsets):
        """Mean inter-bout interval (time between consecutive onsets)"""
        if len(onsets) > 1:
            return np.mean(np.diff(onsets))
        return np.nan

    metrics = [
        ('count',    'Bout Count',                metric_count),
        ('duration', 'Mean Bout Duration (s)',    metric_duration),
        ('ibi',      'Mean Inter-Bout Interval (s)', metric_ibi),
    ]

    # ── COMPUTE PER-ANIMAL VALUES FOR EACH METRIC/BEHAVIOR ────────────────────
    metric_data = {}
    
    for metric_key, metric_label, metric_fn in metrics:
        metric_data[metric_key] = {}
        
        for beh in behaviors:
            metric_data[metric_key][beh] = {}
            
            for cond in conditions:
                cond_animals = [d for d in animal_datasets
                                if d['condition'] == cond]
                
                per_animal = []
                for d in cond_animals:
                    bouts = d['bouts'].get(beh, np.empty((0, 2)))
                    
                    if len(bouts) == 0:
                        val = 0 if metric_key != 'ibi' else np.nan
                        per_animal.append(val)
                        continue

                    durs    = bouts[:, 1] - bouts[:, 0]
                    onsets  = bouts[:, 0]
                    offsets = bouts[:, 1]

                    val = metric_fn(durs, onsets, offsets)
                    per_animal.append(val)

                # Store values (filter NaN for IBI)
                vals = np.array(per_animal, dtype=float)
                vals = vals[~np.isnan(vals)]
                metric_data[metric_key][beh][cond] = vals

    # ── CREATE FIGURES (ONE PER METRIC) ──────────────────────────────────────
    for metric_key, metric_label, metric_fn in metrics:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)

        # X-axis positioning: behaviors with paired bars
        n_behs = len(behaviors)
        x_positions = []
        x_labels = []
        bar_width = 0.35
        group_spacing = 1.2  # Space between behavior groups

        x_offset = 0
        for beh_idx, beh in enumerate(behaviors):
            beh_name = BEHAVIOR_LABELS.get(beh, beh)
            
            # Position for first condition
            x_cond1 = x_offset
            # Position for second condition
            x_cond2 = x_offset + bar_width + 0.05
            
            x_positions.append((beh, conditions[0], x_cond1))
            x_positions.append((beh, conditions[1], x_cond2))
            
            # Label at center of pair
            x_center = (x_cond1 + x_cond2) / 2
            x_labels.append((x_center, beh_name))
            
            x_offset += group_spacing

        # ── Plot bars for each behavior × condition ──────────────────────────
        cond_vals_all = {cond: {} for cond in conditions}
        
        for beh, cond, x_pos in x_positions:
            vals = metric_data[metric_key][beh].get(cond, np.array([]))
            cond_vals_all[cond][beh] = vals
            
            if len(vals) == 0:
                continue

            color = _cond_color(cond)
            bar_mean = np.mean(vals)
            bar_err = scipy_sem(vals) if len(vals) > 1 else 0

            # Bar with grey/blue fill and black border
            ax.bar(x_pos, bar_mean, yerr=bar_err, color=color, alpha=0.9,
                   width=bar_width, capsize=5, error_kw={'lw': 2, 'capthick': 2},
                   zorder=2, edgecolor='black', linewidth=2.5)

            # Individual points (white with black edges)
            jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(vals))
            ax.scatter(x_pos + jitter, vals, color='white', s=50, alpha=0.8,
                       edgecolors='black', linewidths=1.2, zorder=3)

        # ── Add statistics (comparisons within each behavior) ────────────────
        if test and len(conditions) == 2:
            for beh_idx, beh in enumerate(behaviors):
                beh_name = BEHAVIOR_LABELS.get(beh, beh)
                
                g1 = cond_vals_all[conditions[0]].get(beh, np.array([]))
                g2 = cond_vals_all[conditions[1]].get(beh, np.array([]))
                
                if len(g1) > 1 and len(g2) > 1:
                    if test == 'mwu':
                        _, pval = mannwhitneyu(g1, g2, alternative='two-sided')
                    else:
                        _, pval = ttest_ind(g1, g2)
                    
                    # Draw line above the two bars
                    x_cond1 = beh_idx * group_spacing
                    x_cond2 = x_cond1 + bar_width + 0.05
                    x_line_center = (x_cond1 + x_cond2) / 2
                    
                    y_max = max(np.max(g1), np.max(g2))
                    y_line = y_max * 1.10
                    
                    ax.plot([x_cond1, x_cond2], [y_line, y_line], color='black', lw=1.0)
                    ax.text(x_line_center, y_line * 1.02, _p_stars(pval),
                            ha='center', va='bottom', fontsize=9, fontweight='bold')

        # ── Format axes ──────────────────────────────────────────────────────
        x_tick_positions = [label[0] for label in x_labels]
        x_tick_labels = [label[1] for label in x_labels]
        
        ax.set_xticks(x_tick_positions)
        ax.set_xticklabels(x_tick_labels, fontsize=11, fontweight='bold')
        ax.set_ylabel(metric_label, fontsize=12, fontweight='bold')
        ax.set_title(f'Behavioral Fragmentation: {metric_label}',
                     fontsize=13, fontweight='bold', pad=15)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=_cond_color(conditions[0]), alpha=0.9, 
                  edgecolor='black', linewidth=2.0, label=conditions[0].capitalize()),
            Patch(facecolor=_cond_color(conditions[1]), alpha=0.9,
                  edgecolor='black', linewidth=2.0, label=conditions[1].capitalize()),
        ]
        ax.legend(handles=legend_elements, fontsize=11, frameon=True, 
                  edgecolor='black', framealpha=0.95, loc='upper left')

        # Grid for readability
        ax.yaxis.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis='both', labelsize=11)

        # Save
        if save:
            fname = os.path.join(outdir, f'fragmentation_{metric_key}.png')
            fig.savefig(fname, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved → {fname}")
        else:
            plt.show()

    print(f"✓ Created 3 figures: count, duration, IBI (thesis-ready layout)")


# ─────────────────────────────────────────────────────────────────────────────
# 3. SEQUENCE PREDICTABILITY (TRANSITION MATRIX + ENTROPY)
# ─────────────────────────────────────────────────────────────────────────────

def sequence_predictability(animal_datasets, outdir='.', save=True, test='mwu'):
    """
    For each animal:
      1. Build behavior sequence (ordered list of behaviors across session)
      2. Compute transition matrix (probability of A→B)
      3. Compute Shannon entropy of transitions (high = unpredictable/diverse,
         low = stereotyped/repetitive)

    Plots:
      - Mean transition matrix per condition (heatmap)
      - Entropy score per animal per condition (bar + strip)
    """
    os.makedirs(outdir, exist_ok=True)

    behaviors  = sorted({b for d in animal_datasets for b in d['bouts']})
    conditions = list(dict.fromkeys(d['condition'] for d in animal_datasets))
    n_behs     = len(behaviors)

    def get_sequence(bouts_dict):
        """Return time-ordered sequence of behavior codes."""
        all_events = []
        for code, bout_array in bouts_dict.items():
            for onset, _ in bout_array:
                all_events.append((onset, code))
        all_events.sort()
        return [code for _, code in all_events]

    def transition_matrix(sequence, behaviors):
        """Row = from, col = to. Rows sum to 1."""
        n    = len(behaviors)
        beh_idx = {b: i for i, b in enumerate(behaviors)}
        mat  = np.zeros((n, n))
        for a, b in zip(sequence[:-1], sequence[1:]):
            if a in beh_idx and b in beh_idx:
                mat[beh_idx[a], beh_idx[b]] += 1
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        return mat / row_sums

    def sequence_entropy(trans_mat):
        """Mean row entropy of transition matrix (bits)."""
        entropies = []
        for row in trans_mat:
            row = row[row > 0]
            if len(row) > 0:
                entropies.append(-np.sum(row * np.log2(row)))
        return np.mean(entropies) if entropies else 0.0

    # ── Compute per animal ───────────────────────────────────────────────────
    animal_results = []
    for d in animal_datasets:
        seq  = get_sequence(d['bouts'])
        tmat = transition_matrix(seq, behaviors)
        ent  = sequence_entropy(tmat)
        animal_results.append({
            'animal_id': d['animal_id'],
            'condition': d['condition'],
            'sequence':  seq,
            'trans_mat': tmat,
            'entropy':   ent,
        })

    # ── Figure ────────────────────────────────────────────────────────────────
    n_conds = len(conditions)
    fig = plt.figure(figsize=(6 * n_conds + 5, 6))
    gs  = gridspec.GridSpec(1, n_conds + 1, figure=fig,
                             wspace=0.45, width_ratios=[3]*n_conds + [2])
    fig.suptitle('Behavior Sequence Predictability',
                 fontsize=13, fontweight='bold')

    # Mean transition matrix per condition
    for ci, cond in enumerate(conditions):
        cond_results = [r for r in animal_results if r['condition'] == cond]
        if not cond_results:
            continue
        mean_tmat = np.mean([r['trans_mat'] for r in cond_results], axis=0)

        ax = fig.add_subplot(gs[ci])
        im = ax.imshow(mean_tmat, cmap='Blues', vmin=0, vmax=1,
                       aspect='auto')
        plt.colorbar(im, ax=ax, label='Transition probability',
                     fraction=0.046, pad=0.04)
        ax.set_xticks(range(n_behs))
        ax.set_yticks(range(n_behs))
        labels = [BEHAVIOR_LABELS.get(b, b) for b in behaviors]
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel('To', fontsize=9)
        ax.set_ylabel('From', fontsize=9)
        ax.set_title(f'{cond}\nMean transition matrix', fontsize=10,
                     fontweight='bold', color=_cond_color(cond))

        # Annotate cells
        for i in range(n_behs):
            for j in range(n_behs):
                ax.text(j, i, f'{mean_tmat[i,j]:.2f}',
                        ha='center', va='center', fontsize=7,
                        color='white' if mean_tmat[i,j] > 0.5 else 'black')

    # Entropy bar + strip
    ax_e = fig.add_subplot(gs[-1])
    ax_e.spines[['top', 'right']].set_visible(False)
    cond_vals = {}
    for xi, cond in enumerate(conditions):
        entropies = [r['entropy'] for r in animal_results
                     if r['condition'] == cond]
        vals  = np.array(entropies)
        color = _cond_color(cond)
        cond_vals[cond] = vals

        ax_e.bar(xi, np.mean(vals),
                 yerr=scipy_sem(vals) if len(vals) > 1 else 0,
                 color=color, alpha=0.7, width=0.55, capsize=5,
                 error_kw={'lw': 2}, zorder=2, edgecolor='black', linewidth=2.0)
        jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(vals))
        ax_e.scatter(xi + jitter, vals, color='white', s=50, alpha=0.85,
                     edgecolors='black', linewidths=1.2, zorder=3)

    if test and len(conditions) >= 2:
        g1 = cond_vals.get(conditions[0], np.array([]))
        g2 = cond_vals.get(conditions[1], np.array([]))
        if len(g1) > 1 and len(g2) > 1:
            _, pval = mannwhitneyu(g1, g2, alternative='two-sided')
            y_br = ax_e.get_ylim()[1] * 1.08
            ax_e.plot([0, 1], [y_br, y_br], color='black', lw=1.0)
            ax_e.text(0.5, y_br * 1.01,
                      _p_stars(pval) + f'\np={pval:.3f}',
                      ha='center', va='bottom', fontsize=8)

    ax_e.set_xticks(range(len(conditions)))
    ax_e.set_xticklabels(conditions, fontsize=10)
    ax_e.set_ylabel('Transition entropy (bits)', fontsize=9)
    ax_e.set_title('Sequence\npredictability', fontsize=10, fontweight='bold')

    # Add reference lines for entropy
    max_entropy = np.log2(n_behs)
    ax_e.axhline(max_entropy, color='gray', lw=1.0, ls='--', alpha=0.6,
                 label=f'Max ({max_entropy:.1f} bits)')
    ax_e.legend(fontsize=7, frameon=False)

    if save:
        fname = os.path.join(outdir, 'sequence_predictability.png')
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved → {fname}")
        return fname
    plt.show()
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 4. SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

def behavior_summary_table(animal_datasets, session_duration_s=1800.0,
                            outdir='.', save=True):
    """
    Export a CSV summary table: one row per animal × behavior with
    bout count, mean duration, total time, % session, IBI.
    """
    os.makedirs(outdir, exist_ok=True)
    rows = []

    for d in animal_datasets:
        for beh, bouts in d['bouts'].items():
            if len(bouts) == 0:
                continue
            durs    = bouts[:, 1] - bouts[:, 0]
            onsets  = bouts[:, 0]
            ibi     = np.mean(np.diff(onsets)) if len(onsets) > 1 else np.nan
            rows.append({
                'animal_id':       d['animal_id'],
                'condition':       d['condition'],
                'behavior':        beh,
                'behavior_label':  BEHAVIOR_LABELS.get(beh, beh),
                'n_bouts':         len(bouts),
                'mean_duration_s': np.mean(durs),
                'median_duration_s': np.median(durs),
                'total_time_s':    durs.sum(),
                'pct_session':     durs.sum() / session_duration_s * 100,
                'mean_ibi_s':      ibi,
                'bout_rate_per_min': len(bouts) / (session_duration_s / 60),
            })

    df = pd.DataFrame(rows)
    if save:
        fname = os.path.join(outdir, 'behavior_summary.csv')
        df.to_csv(fname, index=False)
        print(f"  Saved → {fname}")

    return df