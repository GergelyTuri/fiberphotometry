"""
bdnf_protein_poster.py
──────────────────────
Publication-grade BDNF protein figures for DG and CA3.
One figure per region showing Vehicle vs Psi bars for Day 1 and Day 7 side-by-side.
Styling: black + professional blue filled bars matching grooming behavior poster plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import ttest_ind


# Professional poster colors - BLACK + PROFESSIONAL BLUE
COLOR_VEHICLE = '#000000'      # Black (vehicle/control)
COLOR_PSI = '#1F77B4'          # Professional blue (psi/treatment)
OUTLINE_COLOR = '#000000'      # Black outline for both bars


def plot_bdnf_by_region(merged_df, region, region_label, save_dir=None):
    """
    Create publication-grade BDNF protein plot for one brain region.
    
    Shows Day 1 and Day 7 on same axis with paired Vehicle vs Psi bars.
    Matches grooming behavior poster styling: black + blue filled bars, 
    bars close together, white dots with black edges, significance brackets.
    
    Parameters
    ----------
    merged_df : pd.DataFrame
        Merged data with columns: Region, Condition, Sex, Intensity, Timepoint
    region : str
        'DG' or 'CA3' - the brain region to plot
    region_label : str
        Display label (e.g., 'Dentate Gyrus' or 'CA3')
    save_dir : str, optional
        Directory to save figure
    """
    
    # Filter to this region
    region_data = merged_df[merged_df['Region'] == region].copy()
    
    if region_data.empty:
        print(f"No data for region: {region}")
        return
    
    region_data = region_data.dropna(subset=['Intensity', 'Condition', 'Timepoint'])
    
    print(f"\n{'='*70}")
    print(f"Plotting {region_label} ({region}): {len(region_data)} data points")
    print(f"{'='*70}")
    
    conditions = ['Veh', '1 mg/kg PSI']
    timepoints = ['Day 1', 'Day 7']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # X-axis positions: [Day 1 Veh, Day 1 Psi, Day 7 Veh, Day 7 Psi]
    # Group timepoints with bars very close together
    x_day1 = np.array([0, 0.15])      # Day 1: Vehicle at 0, Psi at 0.15
    x_day7 = np.array([0.5, 0.65])    # Day 7: Vehicle at 0.5, Psi at 0.65
    
    bar_width = 0.12
    
    # Find global y_max for consistent scaling
    y_max = region_data['Intensity'].max() * 1.35
    
    # ─────────────────────────────────────────────────────────────────────────
    # Plot Day 1 and Day 7
    # ─────────────────────────────────────────────────────────────────────────
    
    for timepoint_idx, (timepoint, x_positions) in enumerate([(tp, [x_day1, x_day7][i]) 
                                                                for i, tp in enumerate(timepoints)]):
        timepoint_data = region_data[region_data['Timepoint'] == timepoint]
        
        for condition_idx, condition in enumerate(conditions):
            cond_data = timepoint_data[timepoint_data['Condition'] == condition]
            
            if cond_data.empty:
                continue
            
            # Calculate stats
            intensity_vals = cond_data['Intensity'].values
            mean_val = np.mean(intensity_vals)
            sem_val = np.std(intensity_vals, ddof=1) / np.sqrt(len(intensity_vals))
            
            # Determine color
            bar_color = COLOR_VEHICLE if condition == 'Veh' else COLOR_PSI
            
            # Plot bar
            x_pos = x_positions[condition_idx]
            ax.bar(
                x_pos,
                mean_val,
                bar_width,
                yerr=sem_val,
                color=bar_color,
                edgecolor=OUTLINE_COLOR,
                linewidth=2.5,
                capsize=5,
                error_kw=dict(elinewidth=1.5, ecolor='black'),
                zorder=2
            )
            
            # Plot individual points - white dots with black edges
            np.random.seed(42)
            jitters = np.random.normal(0, 0.02, size=len(intensity_vals))
            
            ax.scatter(
                x_pos + jitters,
                intensity_vals,
                color='white',
                edgecolors='black',
                linewidths=1.2,
                s=70,
                alpha=0.9,
                zorder=5
            )
        
        # Significance test (Vehicle vs Psi for this timepoint)
        veh_data = timepoint_data[timepoint_data['Condition'] == 'Veh']['Intensity']
        psi_data = timepoint_data[timepoint_data['Condition'] == '1 mg/kg PSI']['Intensity']
        
        if len(veh_data) >= 2 and len(psi_data) >= 2:
            t_stat, p_val = ttest_ind(veh_data, psi_data, equal_var=False)
            
            # Significance label
            if p_val < 0.001:
                sig_label = '***'
            elif p_val < 0.01:
                sig_label = '**'
            elif p_val < 0.05:
                sig_label = '*'
            else:
                sig_label = 'ns'
            
            # Draw significance bracket
            bracket_y = y_max * 0.05 + max(
                (veh_data.max() if len(veh_data) > 0 else 0),
                (psi_data.max() if len(psi_data) > 0 else 0)
            )
            tick_h = y_max * 0.02
            
            ax.plot([x_positions[0], x_positions[0], x_positions[1], x_positions[1]],
                   [bracket_y, bracket_y + tick_h, bracket_y + tick_h, bracket_y],
                   lw=1.2, c='black', zorder=6)
            ax.text((x_positions[0] + x_positions[1]) / 2,
                   bracket_y + tick_h + y_max * 0.01,
                   sig_label,
                   ha='center', va='bottom',
                   fontsize=11, fontweight='bold',
                   color='black', zorder=7)
            
            print(f"\n{timepoint}:")
            print(f"  Vehicle: {np.mean(veh_data):.4f} ± {np.std(veh_data, ddof=1)/np.sqrt(len(veh_data)):.4f}")
            print(f"  Psi:     {np.mean(psi_data):.4f} ± {np.std(psi_data, ddof=1)/np.sqrt(len(psi_data)):.4f}")
            print(f"  p-value: {p_val:.4f} ({sig_label})")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Axes styling - match grooming behavior plots
    # ─────────────────────────────────────────────────────────────────────────
    
    ax.set_title(f'{region_label} — BDNF Protein Expression', 
                fontsize=14, fontweight='bold', pad=12)
    ax.set_ylabel('Mean BDNF Protein Intensity', fontsize=13, fontweight='600')
    
    # X-axis: Day 1 and Day 7 labels
    ax.set_xticks([0.075, 0.575])  # Midpoint between Veh and Psi for each timepoint
    ax.set_xticklabels(['Day 1', 'Day 7'], fontsize=12, fontweight='600')
    
    ax.set_ylim(0, y_max)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.tick_params(axis='y', labelsize=11, length=3, width=0.8)
    ax.tick_params(axis='x', labelsize=11, length=0)
    ax.grid(False)
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COLOR_VEHICLE, edgecolor=OUTLINE_COLOR, 
                      linewidth=2.5, label='Vehicle'),
        mpatches.Patch(facecolor=COLOR_PSI, edgecolor=OUTLINE_COLOR, 
                      linewidth=2.5, label='1 mg/kg Psi'),
    ]
    ax.legend(handles=legend_elements, fontsize=11, loc='upper right',
             frameon=True, framealpha=0.95, edgecolor='black')
    
    plt.tight_layout()
    
    # Save if requested
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        fpath = os.path.join(save_dir, f'BDNF_Protein_{region}.png')
        fig.savefig(fpath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f'\nSaved → {fpath}')
    
    plt.show()
    return fig


def main():
    """
    Example usage - load your merged BDNF data and plot by region.
    """
    
    # This assumes you have a merged_df from load_and_merge_data()
    # with columns: Region, Condition, Sex, Intensity, Timepoint
    
    # Example: if using the BDNF protein code from earlier
    # merged_df = load_and_merge_data(df1_path, df2_path)
    
    # Plot each region
    regions = [
        ('DG', 'Dentate Gyrus'),
        ('CA3', 'CA3'),
    ]
    
    for region, label in regions:
        # Uncomment and update path when ready
        # plot_bdnf_by_region(merged_df, region, label, save_dir='/content/drive/MyDrive/figures/')
        pass


if __name__ == "__main__":
    main()
