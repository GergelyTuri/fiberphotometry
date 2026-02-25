"""
run_analysis.py
─────────────────────────────────────────────────────────────────────────────
Main entry point. Shows how to use loader → extractor → plotter for:
  - Single animal
  - Multi-animal group comparison

Edit the ANIMALS list at the top, then run:
    python run_analysis.py

Outputs go to ./figures/ by default.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import numpy as np

from loader    import load_behavior, load_serotonin
from extractor import build_animal_dataset, save_dataset, load_dataset
from plotter   import (per_animal_figure, comparison_figure,
                       comparison_overlay_figure, quantification_figure,
                       all_behaviors_summary)

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURE YOUR ANIMALS HERE
# Each entry: (animal_id, condition, behavior_csv_path, serotonin_csv_path)
# ═══════════════════════════════════════════════════════════════════════════

ANIMALS = [
    # Vehicle group
    ('nia4',  'saline',  'data/nia4_saline_behaviors_boris.csv',  'data/nia4_saline_serotonin.csv'),
    # ('nia5',  'saline',  'data/nia5_saline_behaviors_boris.csv',  'data/nia5_saline_serotonin.csv'),

    # Drug group
    # ('nia2',  'psi',     'data/nia2_psi_behaviors_boris.csv',     'data/nia2_psi_serotonin.csv'),
    # ('nia3',  'psi',     'data/nia3_psi_behaviors_boris.csv',     'data/nia3_psi_serotonin.csv'),
]

# ── Extraction parameters ────────────────────────────────────────────────────
EXTRACT_PARAMS = dict(
    pre_s          = 5.0,
    post_s         = 10.0,
    baseline_pre_s = 3.0,
    baseline_end_s = 0.5,
)

MIN_BOUT_S = 3.0  # drop bouts shorter than this

# ── Output ───────────────────────────────────────────────────────────────────
OUTDIR     = './figures'
CACHE_DIR  = './datasets'   # saved .npy files per animal so you don't re-extract
BEHAVIORS  = ['e', 'g', 'd', 'r']

os.makedirs(OUTDIR,    exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD + EXTRACT (per animal)
# ═══════════════════════════════════════════════════════════════════════════

def run_extraction(use_cache=True):
    """Load data for each animal, extract peri-event traces, save to cache."""
    all_datasets = []

    for animal_id, condition, beh_path, sero_path in ANIMALS:
        cache_path = os.path.join(CACHE_DIR, f'{animal_id}_{condition}.npy')

        if use_cache and os.path.exists(cache_path):
            print(f"\nLoading cached dataset: {animal_id} | {condition}")
            ds = load_dataset(cache_path)
        else:
            print(f"\n{'═'*60}")
            print(f"Processing: {animal_id} | {condition}")
            print(f"{'═'*60}")

            print("Loading behavior...")
            bouts = load_behavior(beh_path, min_bout_s=MIN_BOUT_S)

            print("\nLoading serotonin...")
            t_sero, dff_sero, sr = load_serotonin(sero_path)

            ds = build_animal_dataset(
                animal_id  = animal_id,
                condition  = condition,
                bouts      = bouts,
                t_sero     = t_sero,
                dff_sero   = dff_sero,
                **EXTRACT_PARAMS,
            )
            save_dataset(ds, cache_path)

        all_datasets.append(ds)

    return all_datasets


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 — FIGURES (per animal)
# ═══════════════════════════════════════════════════════════════════════════

def run_per_animal_figures(all_datasets):
    """Per-animal: heatmap + mean trace for each behavior × alignment."""
    for ds in all_datasets:
        animal_id = ds['animal_id']
        condition = ds['condition']
        subdir    = os.path.join(OUTDIR, 'per_animal', f'{animal_id}_{condition}')

        print(f"\nGenerating per-animal figures: {animal_id} | {condition}")

        # Detailed per-behavior figure (onset + offset side by side)
        per_animal_figure(ds, outdir=subdir)

        # Quick one-page summary of all behaviors
        for align in ['onset', 'offset']:
            all_behaviors_summary(ds, align=align, outdir=subdir)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 — FIGURES (condition comparison)
# ═══════════════════════════════════════════════════════════════════════════

def run_comparison_figures(all_datasets):
    """Side-by-side and overlay comparison figures across conditions."""
    subdir = os.path.join(OUTDIR, 'comparison')

    for beh_code in BEHAVIORS:
        for align in ['onset', 'offset']:
            print(f"\nComparison figure: {beh_code} | {align}")

            # Heatmap side-by-side
            comparison_figure(
                all_datasets,
                behavior_code = beh_code,
                align         = align,
                outdir        = subdir,
            )

            # Overlay mean traces (good for group avg)
            comparison_overlay_figure(
                all_datasets,
                behavior_code = beh_code,
                align         = align,
                outdir        = subdir,
            )


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 — QUANTIFICATION
# ═══════════════════════════════════════════════════════════════════════════

def run_quantification_figures(all_datasets):
    """Bar + strip quantification plots with stats."""
    subdir = os.path.join(OUTDIR, 'quantification')

    for beh_code in BEHAVIORS:
        for align in ['onset', 'offset']:
            for metric in ['peak', 'auc']:
                print(f"\nQuantification: {beh_code} | {align} | {metric}")

                quantification_figure(
                    all_datasets,
                    behavior_code = beh_code,
                    metric        = metric,
                    align         = align,
                    outdir        = subdir,
                    test          = 'mwu',   # or 'ttest' or None
                )


# ═══════════════════════════════════════════════════════════════════════════
# DEMO MODE — runs on the two example files shipped with this repo
# ═══════════════════════════════════════════════════════════════════════════

def run_demo():
    """
    Demo with mismatched example files (different animals).
    Replace with your real matched pairs in ANIMALS above.
    """
    from loader    import load_behavior, load_serotonin
    from extractor import build_animal_dataset

    print("=" * 60)
    print("DEMO MODE — using example files")
    print("=" * 60)

    demo_animals = [
        ('nia4', 'saline',
         'data/nai4_saline_behaviors_boris.csv',
         'data/nia2_psi_serotonin_baseline.csv'),   # mismatched on purpose for demo
    ]

    datasets = []
    for animal_id, condition, beh_path, sero_path in demo_animals:
        if not os.path.exists(beh_path) or not os.path.exists(sero_path):
            print(f"  Demo files not found — place CSVs in ./data/")
            return

        print(f"\nLoading {animal_id} | {condition}...")
        bouts              = load_behavior(beh_path, min_bout_s=MIN_BOUT_S)
        t_sero, dff, _     = load_serotonin(sero_path)
        ds                 = build_animal_dataset(
            animal_id, condition, bouts, t_sero, dff, **EXTRACT_PARAMS)
        datasets.append(ds)

    subdir = os.path.join(OUTDIR, 'demo')
    for ds in datasets:
        per_animal_figure(ds, outdir=subdir)
        all_behaviors_summary(ds, align='onset',  outdir=subdir)
        all_behaviors_summary(ds, align='offset', outdir=subdir)

    print(f"\nDemo figures saved to {subdir}/")


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Peri-event serotonin analysis')
    parser.add_argument('--demo',       action='store_true', help='Run demo mode')
    parser.add_argument('--no-cache',   action='store_true', help='Ignore cached datasets')
    parser.add_argument('--only-step',  type=int, default=0,
                        help='Run only step 1/2/3/4 (0 = all)')
    args = parser.parse_args()

    if args.demo:
        run_demo()
    else:
        use_cache   = not args.no_cache
        all_datasets = run_extraction(use_cache=use_cache)

        step = args.only_step
        if step in (0, 2): run_per_animal_figures(all_datasets)
        if step in (0, 3): run_comparison_figures(all_datasets)
        if step in (0, 4): run_quantification_figures(all_datasets)

        print(f"\nDone. All figures saved to {OUTDIR}/")