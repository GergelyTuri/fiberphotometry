"""
plotter.py
─────────────────────────────────────────────────────────────────────────────
Publication-quality peri-event serotonin figures.

Three main figure types:
  1. per_animal_figure()     — heatmap + mean trace, onset & offset, per behavior
  2. comparison_figure()     — side-by-side two conditions, one behavior
  3. quantification_figure() — bar + strip plot, peak z-score by condition

Usage:
    from plotter import per_animal_figure, comparison_figure, quantification_figure
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import sem as scipy_sem, ttest_ind, mannwhitneyu

from loader import get_behavior_label

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

# ── Condition color palette ───────────────────────────────────────────────────
CONDITION_COLORS = {
    'saline':  '#5B8DB8',
    'vehicle': '#5B8DB8',
    'psi':     '#C0664A',
    'drug':    '#C0664A',
}

BEHAVIOR_COLORS = {
    'e': '#4CAF93',
    'g': '#9B59B6',
    'd': '#E67E22',
    'r': '#E74C3C',
}

HEATMAP_CMAP = 'RdBu_r'

# ── Quantification window (keep in sync with extractor.py) ───────────────────
QUANT_WINDOW = (0.0, 5.0)


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _condition_color(condition):
    return CONDITION_COLORS.get(condition.lower(), '#555555')


def _draw_heatmap(ax, time_axis, traces_z, title='', align_label='Event'):
    """
    Draw a single heatmap panel (trials × time), sorted by peak latency.
    Returns the image object for colorbar attachment.
    """
    if len(traces_z) == 0:
        ax.set_visible(False)
        return None

    # Sort trials by peak latency so structure is visible
    peak_idx   = np.argmax(traces_z, axis=1)
    sort_order = np.argsort(peak_idx)
    sorted_tr  = traces_z[sort_order]

    vmax = np.nanpercentile(np.abs(traces_z), 95)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax.imshow(
        sorted_tr,
        aspect='auto',
        origin='lower',
        extent=[time_axis[0], time_axis[-1], 0.5, len(traces_z) + 0.5],
        cmap=HEATMAP_CMAP,
        norm=norm,
        interpolation='nearest',
    )
    ax.axvline(0, color='white', lw=1.5, ls='--', alpha=0.9, label=align_label)
    ax.set_ylabel('Trial (sorted by peak)', fontsize=8)
    ax.set_xlabel(f'Time from {align_label} (s)', fontsize=8)
    if title:
        ax.set_title(title, fontsize=10, fontweight='bold', pad=4)
    return im


def _draw_mean_trace(ax, time_axis, traces_z, color, align_label='Event',
                     median_offset_s=None, shade_quant=True):
    """
    Draw mean ± SEM trace. Optionally marks median bout offset (for onset plots).
    """
    if len(traces_z) == 0:
        ax.set_visible(False)
        return

    mean_tr = np.mean(traces_z, axis=0)
    err_tr  = scipy_sem(traces_z, axis=0)

    ax.fill_between(time_axis, mean_tr - err_tr, mean_tr + err_tr,
                    alpha=0.25, color=color, linewidth=0)
    ax.plot(time_axis, mean_tr, color=color, lw=2.0, zorder=3)

    ax.axvline(0, color='black', lw=1.2, ls='--', alpha=0.6)
    ax.axhline(0, color='gray',  lw=0.7, alpha=0.4, zorder=1)

    # Shade quantification window
    if shade_quant:
        ax.axvspan(QUANT_WINDOW[0], QUANT_WINDOW[1], alpha=0.07, color=color, zorder=0)

    # Median bout duration marker (onset-aligned panels only)
    if median_offset_s is not None:
        ax.axvline(median_offset_s, color='gray', lw=1.0, ls=':', alpha=0.8,
                   label=f'Median offset ({median_offset_s:.1f}s)')
        ax.legend(fontsize=7, frameon=False, loc='upper right')

    ax.set_xlabel(f'Time from {align_label} (s)', fontsize=9)
    ax.set_ylabel('z-score ΔF/F', fontsize=9)
    ax.set_xlim(time_axis[0], time_axis[-1])
    ax.tick_params(labelsize=8)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — PER ANIMAL
# ─────────────────────────────────────────────────────────────────────────────

def per_animal_figure(dataset, outdir='.', behaviors=None, save=True):
    """
    One figure per behavior: 2×2 layout.
      Top row:    onset heatmap  | offset heatmap
      Bottom row: onset mean     | offset mean

    Parameters
    ----------
    dataset : dict
        Output of extractor.build_animal_dataset()
    outdir : str
        Directory to save figures.
    behaviors : list of str or None
        Which behaviors to plot. None = all in dataset.
    save : bool
        Save to disk if True.

    Returns
    -------
    saved_paths : list of str
    """
    os.makedirs(outdir, exist_ok=True)
    animal_id = dataset['animal_id']
    condition = dataset['condition']
    color     = _condition_color(condition)
    saved     = []

    beh_list = behaviors or list(dataset['behaviors'].keys())

    for beh_code in beh_list:
        beh_data = dataset['behaviors'].get(beh_code)
        if beh_data is None:
            continue

        beh_name   = get_behavior_label(beh_code)
        time_axis  = beh_data['time_axis']
        on_z       = beh_data['onset_z']
        off_z      = beh_data['offset_z']
        durations  = beh_data['durations']
        n_bouts    = beh_data['n_bouts']
        n_on       = len(on_z)
        n_off      = len(off_z)
        med_dur    = np.median(durations) if len(durations) else None

        fig = plt.figure(figsize=(13, 8))
        fig.suptitle(
            f'{beh_name}  |  {animal_id} – {condition}  |  '
            f'n={n_bouts} bouts  ({n_on} onset / {n_off} offset trials)',
            fontsize=12, fontweight='bold', y=0.99
        )

        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.50, wspace=0.40,
                               top=0.93, bottom=0.09, left=0.09, right=0.96)

        # ── Onset column ──────────────────────────────────────────────────
        ax_h_on = fig.add_subplot(gs[0, 0])
        ax_m_on = fig.add_subplot(gs[1, 0])

        im_on = _draw_heatmap(ax_h_on, time_axis, on_z,
                              title=f'Onset-aligned  (n={n_on})',
                              align_label='Onset')
        if im_on is not None:
            plt.colorbar(im_on, ax=ax_h_on, label='z-score',
                         fraction=0.046, pad=0.04, shrink=0.85)

        _draw_mean_trace(ax_m_on, time_axis, on_z, color,
                         align_label='Onset', median_offset_s=med_dur)

        # ── Offset column ─────────────────────────────────────────────────
        ax_h_off = fig.add_subplot(gs[0, 1])
        ax_m_off = fig.add_subplot(gs[1, 1])

        im_off = _draw_heatmap(ax_h_off, time_axis, off_z,
                               title=f'Offset-aligned  (n={n_off})',
                               align_label='Offset')
        if im_off is not None:
            plt.colorbar(im_off, ax=ax_h_off, label='z-score',
                         fraction=0.046, pad=0.04, shrink=0.85)

        _draw_mean_trace(ax_m_off, time_axis, off_z, color,
                         align_label='Offset')

        if save:
            fname = os.path.join(outdir, f'{animal_id}_{condition}_{beh_code}_perianimal.png')
            fig.savefig(fname)
            plt.close(fig)
            print(f"  Saved → {fname}")
            saved.append(fname)
        else:
            plt.show()

    return saved


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — CONDITION COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def comparison_figure(datasets, behavior_code, align='onset', outdir='.', save=True):
    """
    Side-by-side heatmap + mean trace for two (or more) conditions.

    Parameters
    ----------
    datasets : list of dict
        List of dataset dicts (one per animal OR per condition group).
        Each must have keys: 'animal_id', 'condition', 'behaviors'.
    behavior_code : str
        Which behavior to compare.
    align : str
        'onset' or 'offset'
    outdir : str
    save : bool

    Returns
    -------
    path : str or None
    """
    os.makedirs(outdir, exist_ok=True)
    beh_name  = get_behavior_label(behavior_code)
    trace_key = f'{align}_z'

    # Filter datasets that have this behavior
    valid_ds = [d for d in datasets if behavior_code in d['behaviors']
                and len(d['behaviors'][behavior_code][trace_key]) > 0]

    if not valid_ds:
        print(f"  No valid data for behavior '{behavior_code}' ({align})")
        return None

    n_cols = len(valid_ds)
    fig, axes = plt.subplots(2, n_cols, figsize=(6.5 * n_cols, 8),
                             gridspec_kw={'hspace': 0.48, 'wspace': 0.40})
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle(f'{beh_name} — {align.capitalize()}-aligned Comparison',
                 fontsize=13, fontweight='bold', y=0.99)

    for col, d in enumerate(valid_ds):
        beh_data   = d['behaviors'][behavior_code]
        time_axis  = beh_data['time_axis']
        traces_z   = beh_data[trace_key]
        condition  = d['condition']
        animal_id  = d['animal_id']
        color      = _condition_color(condition)
        durations  = beh_data['durations']
        med_dur    = np.median(durations) if align == 'onset' and len(durations) else None

        ax_h = axes[0, col]
        ax_m = axes[1, col]

        im = _draw_heatmap(ax_h, time_axis, traces_z,
                           title=f'{condition}  ({animal_id})\nn={len(traces_z)} trials',
                           align_label=align.capitalize())
        if im is not None:
            plt.colorbar(im, ax=ax_h, label='z-score',
                         fraction=0.046, pad=0.04, shrink=0.85)

        _draw_mean_trace(ax_m, time_axis, traces_z, color,
                         align_label=align.capitalize(), median_offset_s=med_dur)

        # Label condition on mean trace
        ax_m.text(0.02, 0.95, f'{condition}', transform=ax_m.transAxes,
                  fontsize=9, fontweight='bold', color=color, va='top')

    if save:
        fname = os.path.join(outdir, f'comparison_{behavior_code}_{align}.png')
        fig.savefig(fname)
        plt.close(fig)
        print(f"  Saved → {fname}")
        return fname
    else:
        plt.show()
        return None


def comparison_overlay_figure(datasets, behavior_code, align='onset', outdir='.', save=True):
    """
    Overlay mean traces from all conditions on a single axes (no heatmap).
    Best for group-level data where each dataset = one condition's grand mean.

    Parameters
    ----------
    datasets : list of dict
        Each dict should have 'condition', 'behaviors', 'animal_id'.
    behavior_code : str
    align : str
        'onset' or 'offset'
    outdir, save : as above
    """
    os.makedirs(outdir, exist_ok=True)
    beh_name  = get_behavior_label(behavior_code)
    trace_key = f'{align}_z'

    fig, ax = plt.subplots(figsize=(6, 4.5))
    fig.suptitle(f'{beh_name} — {align.capitalize()}-aligned (overlay)',
                 fontsize=12, fontweight='bold')

    for d in datasets:
        beh_data = d['behaviors'].get(behavior_code)
        if beh_data is None or len(beh_data[trace_key]) == 0:
            continue

        traces_z  = beh_data[trace_key]
        time_axis = beh_data['time_axis']
        condition = d['condition']
        color     = _condition_color(condition)
        n         = len(traces_z)

        mean_tr = np.mean(traces_z, axis=0)
        err_tr  = scipy_sem(traces_z, axis=0)

        ax.fill_between(time_axis, mean_tr - err_tr, mean_tr + err_tr,
                        alpha=0.20, color=color)
        ax.plot(time_axis, mean_tr, color=color, lw=2.0,
                label=f'{condition} (n={n})')

    ax.axvline(0, color='black', lw=1.2, ls='--', alpha=0.6)
    ax.axhline(0, color='gray',  lw=0.7, alpha=0.4)
    ax.axvspan(QUANT_WINDOW[0], QUANT_WINDOW[1], alpha=0.06, color='gray')
    ax.set_xlabel(f'Time from {align.capitalize()} (s)', fontsize=10)
    ax.set_ylabel('z-score ΔF/F', fontsize=10)
    ax.legend(fontsize=9, frameon=False)
    ax.set_xlim(beh_data['time_axis'][0], beh_data['time_axis'][-1])

    if save:
        fname = os.path.join(outdir, f'overlay_{behavior_code}_{align}.png')
        fig.savefig(fname)
        plt.close(fig)
        print(f"  Saved → {fname}")
        return fname
    else:
        plt.show()
        return None


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — QUANTIFICATION (bar + strip)
# ─────────────────────────────────────────────────────────────────────────────

def quantification_figure(
    datasets,
    behavior_code,
    metric    = 'peak',
    align     = 'onset',
    outdir    = '.',
    save      = True,
    test      = 'mwu',
):
    """
    Bar + individual-trial strip plot of a quantification metric.
    Optionally runs a statistical test between the first two conditions.

    Parameters
    ----------
    datasets : list of dict
        Dataset dicts — one per animal or averaged condition.
    behavior_code : str
    metric : str
        'peak' (peak z-score in quant window) or 'auc'.
    align : str
        'onset' or 'offset'
    outdir, save : as above
    test : str or None
        'mwu' (Mann-Whitney U), 'ttest' (unpaired t-test), or None.

    Returns
    -------
    path : str or None
    """
    os.makedirs(outdir, exist_ok=True)
    beh_name  = get_behavior_label(behavior_code)
    quant_key = f'quant_{align}'

    fig, ax = plt.subplots(figsize=(3.5 * len(datasets), 5))
    ax.spines[['top', 'right']].set_visible(False)

    condition_vals = {}   # condition → all trial values (for stats)
    xtick_positions = []
    xtick_labels    = []

    for xi, d in enumerate(datasets):
        beh_data = d['behaviors'].get(behavior_code)
        if beh_data is None:
            continue

        quant    = beh_data.get(quant_key, {})
        vals     = quant.get(metric, np.array([]))
        if isinstance(vals, float) or len(vals) == 0:
            continue

        condition = d['condition']
        color     = _condition_color(condition)

        # Store for stats
        condition_vals.setdefault(condition, []).extend(vals.tolist())

        # Bar with SEM
        bar_mean = np.mean(vals)
        bar_err  = scipy_sem(vals)
        ax.bar(xi, bar_mean, yerr=bar_err, color=color, alpha=0.70, width=0.55,
               capsize=5, error_kw={'lw': 2, 'capthick': 2}, zorder=2)

        # Individual trial points with jitter
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(vals))
        ax.scatter(xi + jitter, vals, color=color, s=35, alpha=0.75,
                   edgecolors='white', linewidths=0.4, zorder=3)

        xtick_positions.append(xi)
        xtick_labels.append(f'{condition}\n(n={len(vals)})')

    # ── Statistics ────────────────────────────────────────────────────────
    cond_list = list(condition_vals.keys())
    if test and len(cond_list) >= 2:
        g1 = np.array(condition_vals[cond_list[0]])
        g2 = np.array(condition_vals[cond_list[1]])
        if test == 'mwu':
            stat, pval = mannwhitneyu(g1, g2, alternative='two-sided')
            test_label = 'MWU'
        else:
            stat, pval = ttest_ind(g1, g2)
            test_label = "t-test"

        # Significance bracket
        y_max = ax.get_ylim()[1]
        y_br  = y_max * 1.05
        ax.plot([0, 1], [y_br, y_br], color='black', lw=1.2)
        sig_str = _p_to_stars(pval)
        ax.text(0.5, y_br * 1.02, f'{sig_str}\n{test_label}: p={pval:.3f}',
                ha='center', va='bottom', fontsize=8)

    metric_label = ('Peak z-score' if metric == 'peak' else 'AUC (z-score·s)')
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels, fontsize=10)
    ax.set_ylabel(f'{metric_label}\n({QUANT_WINDOW[0]}–{QUANT_WINDOW[1]}s post-{align})',
                  fontsize=10)
    ax.set_title(f'{beh_name} — {metric.capitalize()} ({align.capitalize()})',
                 fontsize=11, fontweight='bold')
    ax.axhline(0, color='gray', lw=0.8, alpha=0.5)

    if save:
        fname = os.path.join(outdir, f'quant_{behavior_code}_{align}_{metric}.png')
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


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — ALL BEHAVIORS SUMMARY (single figure, one row per behavior)
# ─────────────────────────────────────────────────────────────────────────────

def all_behaviors_summary(dataset, align='onset', outdir='.', save=True):
    """
    One-page summary: each row = one behavior, columns = heatmap | mean trace.
    Good for a quick overview of all behaviors for one animal.

    Parameters
    ----------
    dataset : dict
    align : str  ('onset' or 'offset')
    outdir, save : as above
    """
    os.makedirs(outdir, exist_ok=True)
    trace_key = f'{align}_z'
    behs      = [b for b in dataset['behaviors']
                 if len(dataset['behaviors'][b][trace_key]) > 0]

    if not behs:
        print("  No valid behaviors to plot.")
        return None

    animal_id = dataset['animal_id']
    condition = dataset['condition']
    color     = _condition_color(condition)
    n_rows    = len(behs)

    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4.5 * n_rows),
                             gridspec_kw={'wspace': 0.40, 'hspace': 0.55})
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(f'{animal_id} – {condition} | {align.capitalize()}-aligned Summary',
                 fontsize=13, fontweight='bold', y=1.01)

    for row, beh_code in enumerate(behs):
        beh_data   = dataset['behaviors'][beh_code]
        time_axis  = beh_data['time_axis']
        traces_z   = beh_data[trace_key]
        durations  = beh_data['durations']
        beh_name   = get_behavior_label(beh_code)
        med_dur    = np.median(durations) if align == 'onset' and len(durations) else None

        ax_h = axes[row, 0]
        ax_m = axes[row, 1]

        im = _draw_heatmap(ax_h, time_axis, traces_z,
                           title=f'{beh_name}  (n={len(traces_z)})',
                           align_label=align.capitalize())
        if im:
            plt.colorbar(im, ax=ax_h, label='z-score',
                         fraction=0.046, pad=0.04, shrink=0.85)

        _draw_mean_trace(ax_m, time_axis, traces_z, color,
                         align_label=align.capitalize(), median_offset_s=med_dur)
        ax_m.set_title(beh_name, fontsize=10, fontweight='bold')

    if save:
        fname = os.path.join(outdir,
                             f'{animal_id}_{condition}_{align}_all_behaviors.png')
        fig.savefig(fname)
        plt.close(fig)
        print(f"  Saved → {fname}")
        return fname
    else:
        plt.show()
        return None