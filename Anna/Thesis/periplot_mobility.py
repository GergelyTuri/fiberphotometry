"""
periplot_mobility_analysis.py
─────────────────────────────────────────────────────────────────────────────
Four analyses linking serotonin signal to locomotor state:

  1. mobile_vs_immobile_serotonin()  — bar + strip plot, mean z-score by state
  2. velocity_serotonin_correlation() — continuous cross-correlation with lags
  3. peri_crossing_figure()           — peri-event around velocity threshold crossings
  4. detrend_signal()                 — remove slow drift before BORIS peri-event

Usage:
    from periplot_mobility_analysis import (
        mobile_vs_immobile_serotonin,
        velocity_serotonin_correlation,
        peri_crossing_figure,
        detrend_signal,
    )
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import sem as scipy_sem, mannwhitneyu
from scipy.signal import correlate, detrend
from scipy.interpolate import interp1d

# ── Colors ────────────────────────────────────────────────────────────────────
CONDITION_COLORS = {
    'saline':  '#5B8DB8',
    'vehicle': '#5B8DB8',
    'psi':     '#C0664A',
    'drug':    '#C0664A',
}
MOBILE_COLOR   = '#4CAF93'
IMMOBILE_COLOR = '#9B59B6'


def _cond_color(condition):
    return CONDITION_COLORS.get(condition.lower(), '#555')


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — interpolate serotonin onto velocity timebase (or vice versa)
# ─────────────────────────────────────────────────────────────────────────────

def _align_signals(t_sero, dff_sero, t_vel, vel):
    """
    Interpolate both signals onto a common time grid (serotonin timebase,
    since it is typically higher resolution).
    Returns t, dff_aligned, vel_aligned on the overlapping time range.
    """
    t_min = max(t_sero[0],  t_vel[0])
    t_max = min(t_sero[-1], t_vel[-1])

    mask_s = (t_sero >= t_min) & (t_sero <= t_max)
    t_out  = t_sero[mask_s]
    dff_out = dff_sero[mask_s]

    vel_interp = interp1d(t_vel, vel, bounds_error=False,
                          fill_value='extrapolate')(t_out)

    return t_out, dff_out, vel_interp


# ─────────────────────────────────────────────────────────────────────────────
# 1. DETRENDING
# ─────────────────────────────────────────────────────────────────────────────

def detrend_signal(t_sero, dff_sero, method='linear', window_s=None, verbose=True):
    """
    Remove slow drift from serotonin signal before peri-event extraction.

    Parameters
    ----------
    method : str
        'linear'   — subtract best-fit line across session
        'poly2'    — subtract quadratic fit (better for psilocybin S-curve)
        'rolling'  — subtract rolling mean with window_s window
    window_s : float or None
        Required for 'rolling'. Typical: 120–300s.

    Returns
    -------
    dff_detrended : np.ndarray  — detrended signal
    trend         : np.ndarray  — the trend that was removed
    """
    if method == 'linear':
        coeffs = np.polyfit(t_sero, dff_sero, 1)
        trend  = np.polyval(coeffs, t_sero)

    elif method == 'poly2':
        coeffs = np.polyfit(t_sero, dff_sero, 2)
        trend  = np.polyval(coeffs, t_sero)

    elif method == 'rolling':
        if window_s is None:
            raise ValueError("window_s required for rolling detrend")
        sr       = 1.0 / np.median(np.diff(t_sero))
        win_samp = int(window_s * sr)
        # Pandas rolling mean
        import pandas as pd
        trend = (pd.Series(dff_sero)
                 .rolling(win_samp, center=True, min_periods=1)
                 .mean().values)
    else:
        raise ValueError(f"Unknown method: {method}")

    dff_detrended = dff_sero - trend

    if verbose:
        print(f"  Detrending ({method}): "
              f"original std={dff_sero.std():.3f}, "
              f"detrended std={dff_detrended.std():.3f}, "
              f"trend range={trend.min():.2f}–{trend.max():.2f}")

    return dff_detrended, trend


def plot_detrending(t_sero, dff_sero, dff_detrended, trend,
                    animal_id='', condition='', outdir='.', save=True):
    """Show original, trend, and detrended signal for QC."""
    fig, axes = plt.subplots(2, 1, figsize=(15, 6), sharex=True,
                              gridspec_kw={'hspace': 0.35})
    color = _cond_color(condition)

    axes[0].plot(t_sero, dff_sero, color=color, lw=0.6, alpha=0.8, label='Original')
    axes[0].plot(t_sero, trend, color='black', lw=2.0, ls='--', label='Trend')
    axes[0].set_ylabel('z-score', fontsize=9)
    axes[0].set_title(f'{animal_id} | {condition} — Original + trend', fontsize=10)
    axes[0].legend(fontsize=8, frameon=False)
    axes[0].spines[['top', 'right']].set_visible(False)

    axes[1].plot(t_sero, dff_detrended, color=color, lw=0.6, alpha=0.8)
    axes[1].axhline(0, color='gray', lw=0.8, alpha=0.5)
    axes[1].set_ylabel('z-score (detrended)', fontsize=9)
    axes[1].set_xlabel('Time (s)', fontsize=9)
    axes[1].set_title('Detrended signal', fontsize=10)
    axes[1].spines[['top', 'right']].set_visible(False)

    if save:
        os.makedirs(outdir, exist_ok=True)
        fname = os.path.join(outdir, f'{animal_id}_{condition}_detrending_qc.png')
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved → {fname}")
        return fname
    plt.show()
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 2. MOBILE VS IMMOBILE SEROTONIN
# ─────────────────────────────────────────────────────────────────────────────

def mobile_vs_immobile_serotonin(
    datasets_with_velocity,
    outdir='.',
    save=True,
    test='mwu',
):
    """
    For each animal: extract mean serotonin during mobile vs immobile epochs.
    Plot as bar + individual-animal strip, compare conditions.

    Parameters
    ----------
    datasets_with_velocity : list of dict, each containing:
        {
          'animal_id'  : str,
          'condition'  : str,
          't_sero'     : np.ndarray,
          'dff_sero'   : np.ndarray,
          't_vel'      : np.ndarray,
          'vel'        : np.ndarray,
          'mobile_mask': np.ndarray bool  (from classify_mobility),
        }
    outdir, save, test : as usual

    Returns
    -------
    results : list of dict — per-animal mobile/immobile means
    """
    os.makedirs(outdir, exist_ok=True)
    results = []

    for d in datasets_with_velocity:
        t_sero, dff_sero = d['t_sero'], d['dff_sero']
        t_vel,  vel      = d['t_vel'],  d['vel']

        # Interpolate mobile mask onto serotonin timebase
        mobile_interp = interp1d(t_vel, d['mobile_mask'].astype(float),
                                  kind='nearest', bounds_error=False,
                                  fill_value=0)(t_sero)
        is_mobile = mobile_interp > 0.5

        mobile_mean   = np.mean(dff_sero[is_mobile])   if is_mobile.any()  else np.nan
        immobile_mean = np.mean(dff_sero[~is_mobile])  if (~is_mobile).any() else np.nan

        results.append({
            'animal_id':    d['animal_id'],
            'condition':    d['condition'],
            'mobile_mean':  mobile_mean,
            'immobile_mean':immobile_mean,
            'diff':         mobile_mean - immobile_mean,
        })

    # ── Plot ─────────────────────────────────────────────────────────────────
    # Group by condition
    conditions = list(dict.fromkeys(d['condition'] for d in datasets_with_velocity))
    n_conds    = len(conditions)

    fig, axes = plt.subplots(1, 2, figsize=(5 * n_conds, 5),
                              gridspec_kw={'wspace': 0.45})
    if n_conds == 1:
        axes = [axes[0], axes[1]]

    fig.suptitle('Mean serotonin: Mobile vs Immobile', fontsize=13,
                 fontweight='bold')

    # Panel 1: Mobile vs Immobile per condition
    ax = axes[0]
    ax.spines[['top', 'right']].set_visible(False)

    for xi, cond in enumerate(conditions):
        cond_results = [r for r in results if r['condition'] == cond]
        mob_vals  = [r['mobile_mean']   for r in cond_results]
        imm_vals  = [r['immobile_mean'] for r in cond_results]
        color     = _cond_color(cond)

        x_mob = xi * 2.5
        x_imm = xi * 2.5 + 1.0

        for vals, x, col, label in [
            (mob_vals,  x_mob, MOBILE_COLOR,   'Mobile'),
            (imm_vals,  x_imm, IMMOBILE_COLOR, 'Immobile'),
        ]:
            vals = np.array(vals)
            ax.bar(x, np.nanmean(vals), yerr=scipy_sem(vals[~np.isnan(vals)]),
                   color=col, alpha=0.7, width=0.7, capsize=5,
                   error_kw={'lw': 2}, zorder=2,
                   label=label if xi == 0 else '')
            jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(vals))
            ax.scatter(x + jitter, vals, color=col, s=45, alpha=0.8,
                       edgecolors='white', linewidths=0.5, zorder=3)

        # Connect paired points (same animal)
        for mob, imm in zip(mob_vals, imm_vals):
            ax.plot([x_mob, x_imm], [mob, imm], color='gray',
                    lw=0.8, alpha=0.4, zorder=1)

        # Condition label
        ax.text(xi * 2.5 + 0.5, ax.get_ylim()[0] - 0.05,
                cond, ha='center', fontsize=9, color=color, fontweight='bold',
                transform=ax.transData)

    ax.axhline(0, color='gray', lw=0.8, alpha=0.5)
    ax.set_ylabel('Mean z-score', fontsize=10)
    ax.set_title('By locomotor state', fontsize=10)
    ax.legend(fontsize=9, frameon=False)
    ax.set_xticks([])

    # Panel 2: Mobile − Immobile difference per condition
    ax2 = axes[1]
    ax2.spines[['top', 'right']].set_visible(False)

    all_by_cond = {}
    for xi, cond in enumerate(conditions):
        diffs = [r['diff'] for r in results if r['condition'] == cond]
        diffs = np.array(diffs)
        all_by_cond[cond] = diffs
        color = _cond_color(cond)

        ax2.bar(xi, np.nanmean(diffs), yerr=scipy_sem(diffs[~np.isnan(diffs)]),
                color=color, alpha=0.7, width=0.55, capsize=5,
                error_kw={'lw': 2}, zorder=2)
        jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(diffs))
        ax2.scatter(xi + jitter, diffs, color=color, s=45, alpha=0.8,
                    edgecolors='white', linewidths=0.5, zorder=3)

    # Stats between conditions
    if test and len(conditions) >= 2:
        g1 = all_by_cond[conditions[0]]
        g2 = all_by_cond[conditions[1]]
        g1 = g1[~np.isnan(g1)]
        g2 = g2[~np.isnan(g2)]
        if len(g1) > 1 and len(g2) > 1:
            _, pval = mannwhitneyu(g1, g2, alternative='two-sided')
            y_br = ax2.get_ylim()[1] * 1.05
            ax2.plot([0, 1], [y_br, y_br], color='black', lw=1.2)
            ax2.text(0.5, y_br * 1.02, _p_stars(pval) + f'\np={pval:.3f}',
                     ha='center', va='bottom', fontsize=8)

    ax2.axhline(0, color='gray', lw=0.8, alpha=0.5)
    ax2.set_xticks(range(len(conditions)))
    ax2.set_xticklabels(conditions, fontsize=10)
    ax2.set_ylabel('Mobile − Immobile z-score', fontsize=10)
    ax2.set_title('Difference score by condition', fontsize=10)

    if save:
        fname = os.path.join(outdir, 'mobile_vs_immobile_serotonin.png')
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved → {fname}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3. VELOCITY × SEROTONIN CROSS-CORRELATION
# ─────────────────────────────────────────────────────────────────────────────

def velocity_serotonin_correlation(
    datasets_with_velocity,
    max_lag_s   = 30.0,
    outdir      = '.',
    save        = True,
):
    """
    Cross-correlate serotonin z-score with velocity for each animal.
    Positive lag = serotonin follows velocity (velocity leads).
    Negative lag = serotonin precedes velocity (serotonin leads).

    Plots mean ± SEM cross-correlation per condition.
    """
    os.makedirs(outdir, exist_ok=True)

    conditions  = list(dict.fromkeys(d['condition'] for d in datasets_with_velocity))
    fig, ax     = plt.subplots(figsize=(9, 5))
    ax.spines[['top', 'right']].set_visible(False)

    all_xcorrs = {c: [] for c in conditions}
    lag_axis   = None

    for d in datasets_with_velocity:
        t_out, dff_out, vel_out = _align_signals(
            d['t_sero'], d['dff_sero'], d['t_vel'], d['vel'])

        sr       = 1.0 / np.median(np.diff(t_out))
        max_samp = int(max_lag_s * sr)

        # Z-score velocity for fair comparison
        vel_z = (vel_out - vel_out.mean()) / vel_out.std()
        dff_z = (dff_out - dff_out.mean()) / dff_out.std()

        # Full cross-correlation, normalized
        xcorr = correlate(dff_z, vel_z, mode='full') / len(dff_z)
        lags  = np.arange(-len(dff_z) + 1, len(dff_z)) / sr

        # Trim to ±max_lag_s
        center  = len(dff_z) - 1
        i_start = center - max_samp
        i_end   = center + max_samp + 1
        xcorr_trim = xcorr[i_start:i_end]
        lags_trim  = lags[i_start:i_end]

        if lag_axis is None:
            lag_axis = lags_trim

        all_xcorrs[d['condition']].append(xcorr_trim)

    # Plot per condition
    for cond in conditions:
        xcorrs = np.array(all_xcorrs[cond])
        if len(xcorrs) == 0:
            continue
        color   = _cond_color(cond)
        mean_xc = np.mean(xcorrs, axis=0)
        err_xc  = scipy_sem(xcorrs, axis=0)

        ax.fill_between(lag_axis, mean_xc - err_xc, mean_xc + err_xc,
                        alpha=0.2, color=color)
        ax.plot(lag_axis, mean_xc, color=color, lw=2.0,
                label=f'{cond} (n={len(xcorrs)})')

        # Mark peak lag
        peak_idx = np.argmax(np.abs(mean_xc))
        ax.axvline(lag_axis[peak_idx], color=color, lw=1.0, ls='--', alpha=0.6)
        ax.text(lag_axis[peak_idx], mean_xc[peak_idx],
                f' peak {lag_axis[peak_idx]:.1f}s',
                color=color, fontsize=7, va='bottom')

    ax.axvline(0, color='black', lw=1.2, ls='-', alpha=0.5)
    ax.axhline(0, color='gray',  lw=0.8, alpha=0.4)
    ax.set_xlabel('Lag (s)  [positive = serotonin follows velocity]', fontsize=10)
    ax.set_ylabel('Cross-correlation', fontsize=10)
    ax.set_title('Velocity × Serotonin Cross-Correlation', fontsize=12,
                 fontweight='bold')
    ax.legend(fontsize=9, frameon=False)

    # Annotate directionality
    ax.text(0.02, 0.97, '← Serotonin leads', transform=ax.transAxes,
            fontsize=8, color='gray', va='top')
    ax.text(0.98, 0.97, 'Velocity leads →', transform=ax.transAxes,
            fontsize=8, color='gray', va='top', ha='right')

    if save:
        fname = os.path.join(outdir, 'velocity_serotonin_xcorr.png')
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved → {fname}")
        return fname
    plt.show()
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 4. PERI-EVENT AROUND VELOCITY THRESHOLD CROSSINGS
# ─────────────────────────────────────────────────────────────────────────────

def peri_crossing_figure(
    datasets_with_velocity,
    crossing_times_list,    # list of np.ndarray, one per dataset
    pre_s   = 10.0,
    post_s  = 50.0,
    outdir  = '.',
    save    = True,
):
    """
    Peri-event serotonin traces aligned to immobile→mobile velocity crossings.
    No normalization applied — signal already globally z-scored.

    Parameters
    ----------
    datasets_with_velocity : list of dicts with 't_sero', 'dff_sero', etc.
    crossing_times_list    : list of np.ndarray — one array of crossing times per dataset
    """
    os.makedirs(outdir, exist_ok=True)
    conditions = list(dict.fromkeys(d['condition'] for d in datasets_with_velocity))

    fig, axes = plt.subplots(2, len(conditions),
                              figsize=(7 * len(conditions), 8),
                              gridspec_kw={'hspace': 0.45, 'wspace': 0.40})
    if len(conditions) == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle('Serotonin around Immobile→Mobile transitions',
                 fontsize=13, fontweight='bold')

    for col_idx, cond in enumerate(conditions):
        cond_datasets  = [(d, ct) for d, ct in
                          zip(datasets_with_velocity, crossing_times_list)
                          if d['condition'] == cond]

        all_traces = []
        time_axis  = None

        for d, crossing_times in cond_datasets:
            t_sero   = d['t_sero']
            dff_sero = d['dff_sero']
            sr       = 1.0 / np.median(np.diff(t_sero))
            n_pre    = int(pre_s  * sr)
            n_post   = int(post_s * sr)
            n_total  = n_pre + n_post
            time_axis = np.linspace(-pre_s, post_s, n_total)

            for ev_t in crossing_times:
                idx     = np.searchsorted(t_sero, ev_t)
                i_start = idx - n_pre
                i_end   = idx + n_post
                if i_start < 0 or i_end > len(dff_sero):
                    continue
                trial = dff_sero[i_start:i_end]
                if len(trial) == n_total:
                    all_traces.append(trial)

        if not all_traces or time_axis is None:
            continue

        traces = np.array(all_traces)
        color  = _cond_color(cond)

        # Heatmap
        ax_h = axes[0, col_idx]
        peak_idx   = np.argmax(traces[:, time_axis >= 0], axis=1)
        sort_order = np.argsort(peak_idx)
        sorted_tr  = traces[sort_order]

        vmax = np.nanpercentile(np.abs(traces), 95)
        from matplotlib.colors import TwoSlopeNorm
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im = ax_h.imshow(sorted_tr, aspect='auto', origin='lower',
                         extent=[time_axis[0], time_axis[-1],
                                 0.5, len(traces) + 0.5],
                         cmap='RdBu_r', norm=norm, interpolation='nearest')
        ax_h.axvspan(time_axis[0], 0, alpha=0.08, color='gray')
        ax_h.axvline(0, color='white', lw=1.8, ls='--')
        plt.colorbar(im, ax=ax_h, label='z-score', fraction=0.03,
                     pad=0.02, shrink=0.85)
        ax_h.set_title(f'{cond}\nn={len(traces)} crossings', fontsize=11,
                       fontweight='bold')
        ax_h.set_ylabel('Trial (sorted by peak)', fontsize=8)
        ax_h.set_xlabel('Time from movement onset (s)', fontsize=8)

        # Mean trace
        ax_m = axes[1, col_idx]
        mean_tr = np.mean(traces, axis=0)
        err_tr  = scipy_sem(traces, axis=0)

        ax_m.axvspan(time_axis[0], 0, alpha=0.08, color='gray', label='Immobility')
        ax_m.fill_between(time_axis, mean_tr - err_tr, mean_tr + err_tr,
                          alpha=0.25, color=color)
        ax_m.plot(time_axis, mean_tr, color=color, lw=2.0)
        ax_m.axvline(0, color='black', lw=1.2, ls='--', alpha=0.7,
                     label='Movement onset')
        ax_m.axhline(0, color='gray', lw=0.7, alpha=0.4)
        ax_m.legend(fontsize=8, frameon=False)
        ax_m.set_xlabel('Time from movement onset (s)', fontsize=9)
        ax_m.set_ylabel('z-score', fontsize=9)
        ax_m.set_xlim(time_axis[0], time_axis[-1])
        ax_m.spines[['top', 'right']].set_visible(False)

    if save:
        fname = os.path.join(outdir, 'peri_velocity_crossing.png')
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved → {fname}")
        return fname
    plt.show()
    return None


# ─────────────────────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _p_stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'