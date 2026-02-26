"""
periplot_behavior_structure.py
─────────────────────────────────────────────────────────────────────────────
Behavioral structure metrics from BORIS bout data:

  1. bout_duration_figure()      — mean bout duration per behavior per condition
  2. fragmentation_figure()      — number of bouts, bout rate, inter-bout interval
  3. sequence_predictability()   — transition matrix + entropy score

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
from scipy.stats import sem as scipy_sem, mannwhitneyu
from collections import Counter
import itertools

CONDITION_COLORS = {
    'saline':  '#5B8DB8',
    'vehicle': '#5B8DB8',
    'psi':     '#C0664A',
    'drug':    '#C0664A',
}
BEHAVIOR_LABELS = {
    'e': 'Exploring',
    'g': 'Grooming',
    'd': 'Digging',
    'r': 'Rearing',
}
BEHAVIOR_COLORS = {
    'e': '#4CAF93',
    'g': '#9B59B6',
    'd': '#E67E22',
    'r': '#E74C3C',
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
                   error_kw={'lw': 2}, zorder=2)
            jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(vals))
            ax.scatter(xi + jitter, vals, color=color, s=50, alpha=0.85,
                       edgecolors='white', linewidths=0.5, zorder=3)

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
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved → {fname}")
        return fname
    plt.show()
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 2. FRAGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

def fragmentation_figure(animal_datasets, session_duration_s=1800.0,
                          outdir='.', save=True, test='mwu'):
    """
    Three fragmentation metrics per behavior per condition:
      - Bout count
      - Bout rate (bouts/min)
      - Mean inter-bout interval (IBI, s)

    Parameters
    ----------
    animal_datasets : list of dict
        {'animal_id', 'condition', 'bouts'}
    session_duration_s : float
        Length of recording session for rate calculation.
    """
    os.makedirs(outdir, exist_ok=True)

    behaviors  = sorted({b for d in animal_datasets for b in d['bouts']})
    conditions = list(dict.fromkeys(d['condition'] for d in animal_datasets))

    metrics = [
        ('count',    'Bout count',          lambda durs, onsets, offsets: len(durs)),
        ('rate',     'Bout rate (per min)', lambda durs, onsets, offsets:
             len(durs) / (session_duration_s / 60)),
        ('ibi',      'Mean IBI (s)',        lambda durs, onsets, offsets:
             np.mean(np.diff(onsets)) if len(onsets) > 1 else np.nan),
    ]

    fig, axes = plt.subplots(len(behaviors), 3,
                              figsize=(13, 4 * len(behaviors)),
                              gridspec_kw={'hspace': 0.55, 'wspace': 0.40})
    if len(behaviors) == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle('Behavioral Fragmentation by Condition',
                 fontsize=13, fontweight='bold')

    for row, beh in enumerate(behaviors):
        beh_name = BEHAVIOR_LABELS.get(beh, beh)
        beh_col  = BEHAVIOR_COLORS.get(beh, '#555')

        for col, (metric_key, metric_label, metric_fn) in enumerate(metrics):
            ax = axes[row, col]
            ax.spines[['top', 'right']].set_visible(False)

            cond_vals = {}
            for xi, cond in enumerate(conditions):
                cond_animals = [d for d in animal_datasets
                                if d['condition'] == cond]
                per_animal = []
                for d in cond_animals:
                    bouts = d['bouts'].get(beh, np.empty((0, 2)))
                    if len(bouts) == 0:
                        per_animal.append(0 if metric_key != 'ibi' else np.nan)
                        continue
                    durs    = bouts[:, 1] - bouts[:, 0]
                    onsets  = bouts[:, 0]
                    offsets = bouts[:, 1]
                    per_animal.append(metric_fn(durs, onsets, offsets))

                vals = np.array(per_animal, dtype=float)
                vals = vals[~np.isnan(vals)]
                cond_vals[cond] = vals

                color = _cond_color(cond)
                if len(vals) == 0:
                    continue
                ax.bar(xi, np.mean(vals),
                       yerr=scipy_sem(vals) if len(vals) > 1 else 0,
                       color=color, alpha=0.7, width=0.55, capsize=5,
                       error_kw={'lw': 2}, zorder=2)
                jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(vals))
                ax.scatter(xi + jitter, vals, color=color, s=45, alpha=0.85,
                           edgecolors='white', linewidths=0.5, zorder=3)

            if test and len(conditions) >= 2:
                g1 = cond_vals.get(conditions[0], np.array([]))
                g2 = cond_vals.get(conditions[1], np.array([]))
                if len(g1) > 1 and len(g2) > 1:
                    _, pval = mannwhitneyu(g1, g2, alternative='two-sided')
                    y_br = ax.get_ylim()[1] * 1.08
                    ax.plot([0, 1], [y_br, y_br], color='black', lw=1.0)
                    ax.text(0.5, y_br * 1.01,
                            _p_stars(pval) + f'\np={pval:.3f}',
                            ha='center', va='bottom', fontsize=7)

            ax.set_xticks(range(len(conditions)))
            ax.set_xticklabels(conditions, fontsize=9)
            ax.set_ylabel(metric_label, fontsize=9)
            if col == 0:
                ax.set_title(f'{beh_name}', fontsize=11,
                             fontweight='bold', color=beh_col)

    if save:
        fname = os.path.join(outdir, 'fragmentation.png')
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved → {fname}")
        return fname
    plt.show()
    return None


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
                 error_kw={'lw': 2}, zorder=2)
        jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(vals))
        ax_e.scatter(xi + jitter, vals, color=color, s=50, alpha=0.85,
                     edgecolors='white', linewidths=0.5, zorder=3)

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
        fig.savefig(fname, dpi=150, bbox_inches='tight')
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