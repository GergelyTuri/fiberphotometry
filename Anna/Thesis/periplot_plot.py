"""
periplot_plot.py
─────────────────────────────────────────────────────────────────────────────
Publication-quality peri-event serotonin figures.

Updated for immobility-gap aligned analysis:
  - Traces keyed as 'traces' (not onset_z/offset_z)
  - No per-trial normalization — signal is globally z-scored
  - Window is −pre_s to +post_s around behavior onset following immobility
  - Negative time = during immobility, positive = behavioral response

Figure types:
  1. per_animal_figure()       — heatmap + mean trace per behavior
  2. all_behaviors_summary()   — all behaviors on one page
  3. comparison_figure()       — side-by-side two conditions
  4. comparison_overlay_figure() — overlaid mean traces
  5. quantification_figure()   — bar + strip + stats
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import sem as scipy_sem, ttest_ind, mannwhitneyu

from periplot_loader import get_behavior_label

# ── Style ─────────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    'font.family':     'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size':        9,
    'axes.labelsize':  10,
    'axes.titlesize':  11,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'figure.dpi':      150,
    'savefig.dpi':     300,
    'savefig.bbox':    'tight',
})

CONDITION_COLORS = {
    'saline':  '#5B8DB8',
    'vehicle': '#5B8DB8',
    'psi':     '#C0664A',
    'drug':    '#C0664A',
}

HEATMAP_CMAP  = 'RdBu_r'
QUANT_WINDOW  = (0.0, 15.0)  # keep in sync with extractor


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _condition_color(condition):
    return CONDITION_COLORS.get(condition.lower(), '#555555')


def _draw_heatmap(ax, time_axis, traces, title=''):
    """Heatmap of trials × time, sorted by peak latency post-onset."""
    if len(traces) == 0:
        ax.set_visible(False)
        return None

    # Sort by peak latency in post-onset window only
    post_mask  = time_axis >= 0
    peak_idx   = np.argmax(traces[:, post_mask], axis=1)
    sort_order = np.argsort(peak_idx)
    sorted_tr  = traces[sort_order]

    vmax = np.nanpercentile(np.abs(traces), 95)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax.imshow(
        sorted_tr,
        aspect='auto',
        origin='lower',
        extent=[time_axis[0], time_axis[-1], 0.5, len(traces) + 0.5],
        cmap=HEATMAP_CMAP,
        norm=norm,
        interpolation='nearest',
    )

    # Immobility region shading
    ax.axvspan(time_axis[0], -0.001, alpha=0.08, color='gray', zorder=0,
           label='Immobility')
    ax.axvline(0, color='white', lw=1.8, ls='--', alpha=0.95)

    ax.set_ylabel('Trial (sorted by peak)', fontsize=8)
    ax.set_xlabel('Time from behavior onset (s)', fontsize=8)
    if title:
        ax.set_title(title, fontsize=10, fontweight='bold', pad=4)
    return im


def _draw_mean_trace(ax, time_axis, traces, color, median_gap_s=None):
    """Mean ± SEM trace. Shades immobility period and quantification window."""
    if len(traces) == 0:
        ax.set_visible(False)
        return

    mean_tr = np.mean(traces, axis=0)
    err_tr  = scipy_sem(traces, axis=0)

    # Immobility shading
    ax.axvspan(time_axis[0], 0, alpha=0.08, color='gray', zorder=0,
               label='Immobility')

    # Quantification window shading
    ax.axvspan(QUANT_WINDOW[0], QUANT_WINDOW[1], alpha=0.08, color=color, zorder=0)

    ax.fill_between(time_axis, mean_tr - err_tr, mean_tr + err_tr,
                    alpha=0.25, color=color, linewidth=0)
    ax.plot(time_axis, mean_tr, color=color, lw=2.0, zorder=3)

    ax.axvline(0, color='black', lw=1.2, ls='--', alpha=0.7, label='Behavior onset')
    ax.axhline(0, color='gray',  lw=0.7, alpha=0.4, zorder=1)

    # Median gap duration marker (shows how deep into immobility we're seeing)
    if median_gap_s is not None:
        marker_t = -median_gap_s
        if marker_t >= time_axis[0]:
            ax.axvline(marker_t, color='gray', lw=1.0, ls=':',
                       label=f'Median immobility start (−{median_gap_s:.1f}s)')

    ax.legend(fontsize=7, frameon=False, loc='upper left')
    ax.set_xlabel('Time from behavior onset (s)', fontsize=9)
    ax.set_ylabel('z-score', fontsize=9)
    ax.set_xlim(time_axis[0], time_axis[-1])
    ax.tick_params(labelsize=8)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — PER ANIMAL
# ─────────────────────────────────────────────────────────────────────────────

def per_animal_figure(dataset, outdir='.', behaviors=None, save=True):
    """
    One figure per behavior: heatmap (top) + mean trace (bottom).
    Negative time = immobility, t=0 = behavior onset, positive = response.
    """
    os.makedirs(outdir, exist_ok=True)
    animal_id = dataset['animal_id']
    condition = dataset['condition']
    color     = _condition_color(condition)
    saved     = []

    beh_list = behaviors or list(dataset['behaviors'].keys())

    for beh_code in beh_list:
        beh_data = dataset['behaviors'].get(beh_code)
        if beh_data is None or len(beh_data['traces']) == 0:
            continue

        beh_name      = get_behavior_label(beh_code)
        time_axis     = beh_data['time_axis']
        traces        = beh_data['traces']
        gap_durations = beh_data['gap_durations']
        n_trials      = len(traces)
        median_gap    = np.median(gap_durations) if len(gap_durations) else None

        fig = plt.figure(figsize=(10, 7))
        fig.suptitle(
            f'{beh_name} following immobility  |  {animal_id} – {condition}  |  '
            f'n={n_trials} trials',
            fontsize=12, fontweight='bold', y=0.99
        )

        gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.45,
                               top=0.93, bottom=0.09, left=0.10, right=0.93)

        ax_h = fig.add_subplot(gs[0])
        ax_m = fig.add_subplot(gs[1])

        im = _draw_heatmap(ax_h, time_axis, traces,
                           title=f'n={n_trials} trials  |  '
                                 f'median immobility {median_gap:.1f}s')
        if im:
            plt.colorbar(im, ax=ax_h, label='z-score',
                         fraction=0.03, pad=0.02, shrink=0.9)

        _draw_mean_trace(ax_m, time_axis, traces, color,
                         median_gap_s=median_gap)

        if save:
            fname = os.path.join(outdir,
                                 f'{animal_id}_{condition}_{beh_code}_postimmobility.png')
            fig.savefig(fname)
            plt.close(fig)
            print(f"  Saved → {fname}")
            saved.append(fname)
        else:
            plt.show()

    return saved


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — ALL BEHAVIORS SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def all_behaviors_summary(dataset, align='onset', outdir='.', save=True):
    """
    One-page summary: each row = one behavior, columns = heatmap | mean trace.
    (align parameter kept for API compatibility but not used — always onset-aligned)
    """
    os.makedirs(outdir, exist_ok=True)
    animal_id = dataset['animal_id']
    condition = dataset['condition']
    color     = _condition_color(condition)

    behs = [b for b in dataset['behaviors']
            if len(dataset['behaviors'][b]['traces']) > 0]

    if not behs:
        print("  No valid behaviors to plot.")
        return None

    n_rows = len(behs)
    fig, axes = plt.subplots(n_rows, 2, figsize=(13, 4.5 * n_rows),
                             gridspec_kw={'wspace': 0.38, 'hspace': 0.55})
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        f'{animal_id} – {condition}  |  Post-immobility behavior summary\n'
        f'Gray = immobility  |  Dashed = behavior onset  |  '
        f'Blue shade = quantification window',
        fontsize=12, fontweight='bold', y=1.01
    )

    for row, beh_code in enumerate(behs):
        beh_data      = dataset['behaviors'][beh_code]
        time_axis     = beh_data['time_axis']
        traces        = beh_data['traces']
        gap_durations = beh_data['gap_durations']
        beh_name      = get_behavior_label(beh_code)
        median_gap    = np.median(gap_durations) if len(gap_durations) else None

        ax_h = axes[row, 0]
        ax_m = axes[row, 1]

        im = _draw_heatmap(ax_h, time_axis, traces,
                           title=f'{beh_name}  (n={len(traces)})')
        if im:
            plt.colorbar(im, ax=ax_h, label='z-score',
                         fraction=0.03, pad=0.02, shrink=0.9)

        _draw_mean_trace(ax_m, time_axis, traces, color,
                         median_gap_s=median_gap)
        ax_m.set_title(beh_name, fontsize=10, fontweight='bold')

    if save:
        fname = os.path.join(outdir,
                             f'{animal_id}_{condition}_onset_all_behaviors.png')
        fig.savefig(fname)
        plt.close(fig)
        print(f"  Saved → {fname}")
        return fname
    else:
        plt.show()
        return None


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — CONDITION COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def comparison_figure(datasets, behavior_code, align='onset', outdir='.', save=True):
    """Side-by-side heatmap + mean trace for two conditions, one behavior."""
    os.makedirs(outdir, exist_ok=True)
    beh_name = get_behavior_label(behavior_code)

    valid_ds = [d for d in datasets
                if behavior_code in d['behaviors']
                and len(d['behaviors'][behavior_code]['traces']) > 0]

    if not valid_ds:
        print(f"  No valid data for '{behavior_code}'")
        return None

    n_cols = len(valid_ds)
    fig, axes = plt.subplots(2, n_cols, figsize=(6.5 * n_cols, 8),
                             gridspec_kw={'hspace': 0.48, 'wspace': 0.40})
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle(f'{beh_name} — Post-immobility Comparison',
                 fontsize=13, fontweight='bold', y=0.99)

    for col, d in enumerate(valid_ds):
        beh_data      = d['behaviors'][behavior_code]
        time_axis     = beh_data['time_axis']
        traces        = beh_data['traces']
        gap_durations = beh_data['gap_durations']
        condition     = d['condition']
        color         = _condition_color(condition)
        median_gap    = np.median(gap_durations) if len(gap_durations) else None

        ax_h = axes[0, col]
        ax_m = axes[1, col]

        im = _draw_heatmap(ax_h, time_axis, traces,
                           title=f'{condition}  ({d["animal_id"]})\n'
                                 f'n={len(traces)} trials')
        if im:
            plt.colorbar(im, ax=ax_h, label='z-score',
                         fraction=0.03, pad=0.02, shrink=0.9)

        _draw_mean_trace(ax_m, time_axis, traces, color,
                         median_gap_s=median_gap)
        ax_m.text(0.02, 0.95, condition, transform=ax_m.transAxes,
                  fontsize=9, fontweight='bold', color=color, va='top')

    if save:
        fname = os.path.join(outdir, f'comparison_{behavior_code}_postimmobility.png')
        fig.savefig(fname)
        plt.close(fig)
        print(f"  Saved → {fname}")
        return fname
    else:
        plt.show()
        return None


def comparison_overlay_figure(datasets, behavior_code, align='onset',
                               outdir='.', save=True):
    """Overlaid mean traces for all conditions on one axes."""
    os.makedirs(outdir, exist_ok=True)
    beh_name = get_behavior_label(behavior_code)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.suptitle(f'{beh_name} — Post-immobility (overlay)',
                 fontsize=12, fontweight='bold')

    time_axis = None
    for d in datasets:
        beh_data = d['behaviors'].get(behavior_code)
        if beh_data is None or len(beh_data['traces']) == 0:
            continue

        traces    = beh_data['traces']
        time_axis = beh_data['time_axis']
        condition = d['condition']
        color     = _condition_color(condition)

        mean_tr = np.mean(traces, axis=0)
        err_tr  = scipy_sem(traces, axis=0)

        ax.fill_between(time_axis, mean_tr - err_tr, mean_tr + err_tr,
                        alpha=0.20, color=color)
        ax.plot(time_axis, mean_tr, color=color, lw=2.0,
                label=f'{condition} (n={len(traces)})')

    if time_axis is not None:
        ax.axvspan(time_axis[0], 0, alpha=0.07, color='gray', label='Immobility')
        ax.axvspan(QUANT_WINDOW[0], QUANT_WINDOW[1], alpha=0.07, color='steelblue')
        ax.axvline(0, color='black', lw=1.2, ls='--', alpha=0.6)
        ax.axhline(0, color='gray',  lw=0.7, alpha=0.4)
        ax.set_xlim(time_axis[0], time_axis[-1])

    ax.set_xlabel('Time from behavior onset (s)', fontsize=10)
    ax.set_ylabel('z-score', fontsize=10)
    ax.legend(fontsize=9, frameon=False)

    if save:
        fname = os.path.join(outdir, f'overlay_{behavior_code}_postimmobility.png')
        fig.savefig(fname)
        plt.close(fig)
        print(f"  Saved → {fname}")
        return fname
    else:
        plt.show()
        return None


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — QUANTIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def quantification_figure(datasets, behavior_code, metric='peak', align='onset',
                           outdir='.', save=True, test='mwu'):
    """Bar + strip plot of peak z-score or AUC, with optional stats."""
    os.makedirs(outdir, exist_ok=True)
    beh_name = get_behavior_label(behavior_code)

    fig, ax = plt.subplots(figsize=(3.5 * len(datasets), 5))
    ax.spines[['top', 'right']].set_visible(False)

    condition_vals  = {}
    xtick_positions = []
    xtick_labels    = []

    for xi, d in enumerate(datasets):
        beh_data = d['behaviors'].get(behavior_code)
        if beh_data is None:
            continue

        quant     = beh_data.get('quant', {})
        vals      = quant.get(metric, np.array([]))
        if isinstance(vals, float) or len(vals) == 0:
            continue

        condition = d['condition']
        color     = _condition_color(condition)
        condition_vals.setdefault(condition, []).extend(vals.tolist())

        bar_mean = np.mean(vals)
        bar_err  = scipy_sem(vals)
        ax.bar(xi, bar_mean, yerr=bar_err, color=color, alpha=0.70, width=0.55,
               capsize=5, error_kw={'lw': 2, 'capthick': 2}, zorder=2)

        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(vals))
        ax.scatter(xi + jitter, vals, color=color, s=35, alpha=0.75,
                   edgecolors='white', linewidths=0.4, zorder=3)

        xtick_positions.append(xi)
        xtick_labels.append(f'{condition}\n(n={len(vals)})')

    # Stats between first two conditions
    cond_list = list(condition_vals.keys())
    if test and len(cond_list) >= 2:
        g1 = np.array(condition_vals[cond_list[0]])
        g2 = np.array(condition_vals[cond_list[1]])
        if test == 'mwu':
            stat, pval = mannwhitneyu(g1, g2, alternative='two-sided')
            test_label = 'MWU'
        else:
            stat, pval = ttest_ind(g1, g2)
            test_label = 't-test'

        y_max = ax.get_ylim()[1]
        y_br  = y_max * 1.05
        ax.plot([0, 1], [y_br, y_br], color='black', lw=1.2)
        ax.text(0.5, y_br * 1.02, f'{_p_to_stars(pval)}\n{test_label}: p={pval:.3f}',
                ha='center', va='bottom', fontsize=8)

    metric_label = 'Peak z-score' if metric == 'peak' else 'AUC (z·s)'
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels, fontsize=10)
    ax.set_ylabel(f'{metric_label}\n({QUANT_WINDOW[0]}–{QUANT_WINDOW[1]}s post-onset)',
                  fontsize=10)
    ax.set_title(f'{beh_name} — {metric.capitalize()} post-immobility',
                 fontsize=11, fontweight='bold')
    ax.axhline(0, color='gray', lw=0.8, alpha=0.5)

    if save:
        fname = os.path.join(outdir, f'quant_{behavior_code}_{metric}.png')
        fig.savefig(fname)
        plt.close(fig)
        print(f"  Saved → {fname}")
        return fname
    else:
        plt.show()
        return None


def _p_to_stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'