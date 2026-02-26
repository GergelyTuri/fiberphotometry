"""
periplot_velocity.py
─────────────────────────────────────────────────────────────────────────────
Load DLC velocity data, classify mobile/immobile epochs, extract threshold
crossing events for peri-event alignment.

Expected CSV format:
    Time (s), Smoothed Velocity (cm/s)

Usage:
    from periplot_velocity import load_velocity, classify_mobility, get_threshold_crossings
"""

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.signal import argrelmin
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

DEFAULT_IMMOBILE_THRESH_CMS = 5.0    # cm/s below = immobile
DEFAULT_MIN_BOUT_S          = 3.0    # minimum bout duration to keep
DEFAULT_MIN_IMMOBILE_S      = 10.0   # minimum immobile bout for peri-event anchor


# ─────────────────────────────────────────────────────────────────────────────
# LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_velocity(path, verbose=True):
    """
    Load DLC velocity CSV.

    Returns
    -------
    t_vel : np.ndarray   — time vector (s)
    vel   : np.ndarray   — smoothed velocity (cm/s), NaNs interpolated
    sr    : float        — sampling rate (Hz)
    """
    df  = pd.read_csv(path)

    # Flexible column detection
    time_col = next((c for c in df.columns
                     if 'time' in c.lower()), None)
    vel_col  = next((c for c in df.columns
                     if 'veloc' in c.lower() or 'speed' in c.lower()), None)

    if time_col is None or vel_col is None:
        raise KeyError(f"Could not find time/velocity columns. Got: {list(df.columns)}")

    t_vel = df[time_col].values.astype(float)
    vel   = df[vel_col].values.astype(float)

    # Interpolate NaNs
    nan_mask = np.isnan(vel)
    if nan_mask.any():
        vel[nan_mask] = np.interp(
            np.flatnonzero(nan_mask),
            np.flatnonzero(~nan_mask),
            vel[~nan_mask]
        )

    sr = 1.0 / np.median(np.diff(t_vel))

    if verbose:
        print(f"  Velocity: {len(t_vel)} samples | {t_vel[-1]:.1f}s | "
              f"~{sr:.1f} Hz | {nan_mask.sum()} NaNs interpolated")
        print(f"  Range: {vel.min():.1f}–{vel.max():.1f} cm/s | "
              f"median {np.median(vel):.1f} | mean {vel.mean():.1f}")

    return t_vel, vel, sr


# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLD SELECTION
# ─────────────────────────────────────────────────────────────────────────────

def suggest_threshold(vel, outdir=None, animal_id='', condition='', save=True):
    """
    Plot velocity histogram with KDE and candidate thresholds so you can
    pick an informed cutoff. Saves figure to outdir if provided.

    Returns
    -------
    candidates : list of float — local minima in KDE (natural valleys)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4),
                              gridspec_kw={'wspace': 0.35})

    # Full distribution
    ax = axes[0]
    ax.hist(vel, bins=100, color='#5B8DB8', alpha=0.6, density=True)
    x = np.linspace(0, np.percentile(vel, 99), 1000)
    kde = gaussian_kde(vel, bw_method=0.15)
    density = kde(x)
    ax.plot(x, density, color='navy', lw=2)

    # Mark candidate thresholds
    minima_idx = argrelmin(density, order=30)[0]
    candidates = [x[m] for m in minima_idx if x[m] < 30]
    for c in candidates:
        ax.axvline(c, color='red', lw=1.5, ls='--', label=f'KDE min: {c:.1f} cm/s')

    # Mark common field thresholds
    for t, col in [(5, '#E67E22'), (8, '#E74C3C')]:
        ax.axvline(t, color=col, lw=1.2, ls=':', alpha=0.8,
                   label=f'{t} cm/s ({np.mean(vel<t)*100:.0f}% immobile)')

    ax.set_xlabel('Velocity (cm/s)', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(f'{animal_id} {condition} — Full velocity distribution', fontsize=10)
    ax.set_xlim(0, np.percentile(vel, 99))
    ax.legend(fontsize=8, frameon=False)
    ax.spines[['top', 'right']].set_visible(False)

    # Zoomed low-velocity region
    ax2 = axes[1]
    low_vel = vel[vel < 25]
    ax2.hist(low_vel, bins=60, color='#5B8DB8', alpha=0.6, density=True)
    x2 = np.linspace(0, 25, 500)
    kde2 = gaussian_kde(low_vel, bw_method=0.2)
    ax2.plot(x2, kde2(x2), color='navy', lw=2)
    for t, col in [(5, '#E67E22'), (8, '#E74C3C'), (10, '#9B59B6')]:
        pct = np.mean(vel < t) * 100
        ax2.axvline(t, color=col, lw=1.5, ls='--',
                    label=f'{t} cm/s → {pct:.0f}% immobile')
    ax2.set_xlabel('Velocity (cm/s)', fontsize=10)
    ax2.set_ylabel('Density', fontsize=10)
    ax2.set_title('Zoomed 0–25 cm/s', fontsize=10)
    ax2.set_xlim(0, 25)
    ax2.legend(fontsize=8, frameon=False)
    ax2.spines[['top', 'right']].set_visible(False)

    fig.suptitle('Velocity distribution — pick your immobility threshold',
                 fontsize=12, fontweight='bold')

    if save and outdir:
        os.makedirs(outdir, exist_ok=True)
        fname = os.path.join(outdir, f'{animal_id}_{condition}_velocity_histogram.png')
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved → {fname}")
        return candidates, fname
    else:
        plt.show()
        return candidates, None


# ─────────────────────────────────────────────────────────────────────────────
# MOBILITY CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def classify_mobility(t_vel, vel, thresh_cms=DEFAULT_IMMOBILE_THRESH_CMS,
                      min_bout_s=DEFAULT_MIN_BOUT_S, verbose=True):
    """
    Classify each sample as mobile (1) or immobile (0).
    Short bouts below min_bout_s are merged into surrounding state.

    Returns
    -------
    mobile_mask  : np.ndarray bool — True = mobile
    mobile_bouts : np.ndarray (n, 2) — [onset, offset] of mobile epochs
    immobile_bouts : np.ndarray (n, 2) — [onset, offset] of immobile epochs
    """
    sr   = 1.0 / np.median(np.diff(t_vel))
    mask = (vel >= thresh_cms).astype(int)

    # Remove short bouts by minimum duration filtering
    min_samples = int(min_bout_s * sr)

    def extract_bouts(binary_mask, state=1):
        bouts  = []
        in_bout = False
        start   = 0
        for i, v in enumerate(binary_mask):
            if v == state and not in_bout:
                start   = i
                in_bout = True
            elif v != state and in_bout:
                if (i - start) >= min_samples:
                    bouts.append([t_vel[start], t_vel[i - 1]])
                in_bout = False
        if in_bout and (len(binary_mask) - start) >= min_samples:
            bouts.append([t_vel[start], t_vel[-1]])
        return np.array(bouts) if bouts else np.empty((0, 2))

    mobile_bouts   = extract_bouts(mask, state=1)
    immobile_bouts = extract_bouts(mask, state=0)

    if verbose:
        pct_mobile = np.mean(mask) * 100
        print(f"  Threshold: {thresh_cms} cm/s → "
              f"{pct_mobile:.1f}% mobile | {100-pct_mobile:.1f}% immobile")
        print(f"  Mobile bouts (≥{min_bout_s}s): {len(mobile_bouts)}")
        print(f"  Immobile bouts (≥{min_bout_s}s): {len(immobile_bouts)}")
        if len(mobile_bouts):
            durs = mobile_bouts[:,1] - mobile_bouts[:,0]
            print(f"    Mobile dur: {durs.min():.1f}–{durs.max():.1f}s "
                  f"(median {np.median(durs):.1f}s)")
        if len(immobile_bouts):
            durs = immobile_bouts[:,1] - immobile_bouts[:,0]
            print(f"    Immobile dur: {durs.min():.1f}–{durs.max():.1f}s "
                  f"(median {np.median(durs):.1f}s)")

    mobile_mask = mask.astype(bool)
    return mobile_mask, mobile_bouts, immobile_bouts


# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLD CROSSING EVENTS
# ─────────────────────────────────────────────────────────────────────────────

def get_threshold_crossings(t_vel, vel, thresh_cms=DEFAULT_IMMOBILE_THRESH_CMS,
                             min_immobile_s=DEFAULT_MIN_IMMOBILE_S,
                             min_bout_s=DEFAULT_MIN_BOUT_S, verbose=True):
    """
    Find immobile→mobile transitions where immobility lasted ≥ min_immobile_s.
    These are your peri-event anchors for velocity-based alignment.

    Returns
    -------
    crossing_times : np.ndarray — timestamps of immobile→mobile transitions (s)
    pre_immobile_durs : np.ndarray — duration of preceding immobility (s)
    """
    _, mobile_bouts, immobile_bouts = classify_mobility(
        t_vel, vel, thresh_cms=thresh_cms,
        min_bout_s=min_bout_s, verbose=False)

    if len(immobile_bouts) == 0 or len(mobile_bouts) == 0:
        return np.array([]), np.array([])

    crossing_times    = []
    pre_immobile_durs = []

    for imm_onset, imm_offset in immobile_bouts:
        dur = imm_offset - imm_onset
        if dur < min_immobile_s:
            continue
        # The crossing = end of this immobility bout = start of next movement
        crossing_times.append(imm_offset)
        pre_immobile_durs.append(dur)

    crossing_times    = np.array(crossing_times)
    pre_immobile_durs = np.array(pre_immobile_durs)

    if verbose:
        print(f"  Immobile→Mobile crossings (immobility ≥{min_immobile_s}s): "
              f"{len(crossing_times)}")
        if len(pre_immobile_durs):
            print(f"  Preceding immobility: "
                  f"{pre_immobile_durs.min():.1f}–{pre_immobile_durs.max():.1f}s "
                  f"(median {np.median(pre_immobile_durs):.1f}s)")

    return crossing_times, pre_immobile_durs