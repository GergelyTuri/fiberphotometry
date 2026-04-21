import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats


def plot_area(merged_df, area_name):
    """
    Plot data for a specific brain area with group bars and sex-colored scatter points.

    Parameters:
    ----------
    merged_df : pd.DataFrame
        The merged DataFrame containing the data.
    area_name : str
        The area name to filter the data by and plot.
    """

    # Filter for the target area
    area_df = merged_df[merged_df['area'] == area_name].copy()

    # Standardize sex column to uppercase
    area_df['sex'] = area_df['sex'].str.upper()

    # Custom group label mapping
    label_map = {
        'ctrl': 'Control',
        'pcb1': '1 Day',
        'pcb7': '7 Day'
    }
    area_df['label'] = area_df['group'].map(label_map)

    # Group stats: mean and standard error
    grouped = area_df.groupby('label')['mean/volume'].agg(['mean', 'sem']).reset_index()

    # Bar colors for groups
    bar_colors = {
        'Control': '#A0C4FF',
        '1 Day': '#7B9FAB',
        '7 Day': '#B7C3D0'
    }

    # Point colors for sex
    sex_colors = {'M': '#4A90E2', 'F': '#FF69B4'}

    # Initialize plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Bar plots with error bars
    for label in grouped['label']:
        group_data = grouped[grouped['label'] == label]
        ax.bar(label, group_data['mean'].values[0],
               yerr=group_data['sem'].values[0],
               color=bar_colors.get(label, 'gray'),
               capsize=5)

    # Overlay individual data points colored by sex
    for label in grouped['label']:
        scatter_data = area_df[area_df['label'] == label]
        for _, row in scatter_data.iterrows():
            ax.scatter(
                label,
                row['mean/volume'],
                color=sex_colors.get(row['sex'], 'gray'),
                edgecolor='black',
                alpha=0.8,
                s=80
            )

    # Axes labels and title
    ax.set_ylabel('Mean Volume', fontsize=16)
    ax.set_xlabel('Group', fontsize=16)
    ax.set_title(f'Mean BDNF in {area_name}', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Sex legend
    for sex, color in sex_colors.items():
        ax.scatter([], [], color=color, edgecolor='black', label=sex)
    ax.legend(title='Sex', fontsize=12, title_fontsize=12)

    plt.tight_layout()
    plt.show()


def plot_bdnf_figure(df):
    """
    Expects aggregated dataframe from load_and_merge_data()
    """

    # Remove missing sex rows
    df = df.dropna(subset=["sex"])

    # Order areas
    area_order = ["ca1", "ca3", "hilus"]
    df["area"] = pd.Categorical(df["area"], categories=area_order, ordered=True)

    # Compute mean + SEM per area per sex
    summary = (
        df
        .groupby(["area", "sex"])
        ["mean/volume"]
        .agg(["mean", "sem"])
        .reset_index()
    )

    areas = area_order
    sexes = ["M", "F"]

    x = np.arange(len(areas))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 5))

    for i, sex in enumerate(sexes):
        data = summary[summary["sex"] == sex]
        means = []
        sems = []

        for area in areas:
            row = data[data["area"] == area]
            if len(row) > 0:
                means.append(row["mean"].values[0])
                sems.append(row["sem"].values[0])
            else:
                means.append(0)
                sems.append(0)

        ax.bar(
            x + (i - 0.5) * width,
            means,
            width,
            yerr=sems,
            capsize=5,
            label=sex
        )

    ax.set_xticks(x)
    ax.set_xticklabels(["CA1", "CA3", "Hilus"])
    ax.set_ylabel("Mean / Volume")
    ax.set_xlabel("Region")
    ax.legend(title="Sex")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_bdnf_by_sex(df):
    """
    Produces two separate 2x2 publication-quality figures of BDNF,
    one for Male and one for Female. Each shows Vehicle vs PSI
    across areas (DG, CA3) and timepoints (Day 1, Day 7).
    """

    df = df.dropna(subset=["area", "sex", "group", "mean/volume"]).copy()
    df["sex"] = df["sex"].str.upper()
    df["group"] = df["group"].str.lower()

    areas = ["hilus", "ca3"]
    timepoints = ["pcb1", "pcb7"]
    col_titles = ["Post-Drug Day 1", "Post-Drug Day 7"]
    row_labels = ["DG", "CA3"]

    bar_colors = {"Vehicle": "#D4B896", "PSI": "#CC5533"}

    sex_configs = {
        "M": {"label": "Male",   "dot_color": "black",   "filename": "bdnf_male.png"},
        "F": {"label": "Female", "dot_color": "#E0409A", "filename": "bdnf_female.png"},
    }

    for sex_key, sex_cfg in sex_configs.items():

        sex_df = df[df["sex"] == sex_key].copy()

        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        plt.subplots_adjust(hspace=0.4, wspace=0.25, right=0.82, top=0.88)

        fig.suptitle(
            f"Psilocybin Does Not Alter BDNF Protein Expression — {sex_cfg['label']}",
            fontsize=13, fontweight="bold", y=0.97
        )

        for i, area in enumerate(areas):
            for j, tp in enumerate(timepoints):
                ax = axes[i, j]

                plot_df = sex_df[
                    (sex_df["area"] == area) & (sex_df["group"].isin([tp, "ctrl"]))
                ].copy()
                plot_df["plot_group"] = plot_df["group"].map(
                    lambda x: "Vehicle" if x == "ctrl" else "PSI"
                )

                group_order = ["Vehicle", "PSI"]
                x_positions = {g: k for k, g in enumerate(group_order)}

                summary = (
                    plot_df.groupby("plot_group")["mean/volume"]
                    .agg(["mean", "sem"])
                    .reindex(group_order)
                    .reset_index()
                )

                # Bars
                for _, row in summary.iterrows():
                    ax.bar(
                        x_positions[row["plot_group"]],
                        row["mean"],
                        yerr=row["sem"],
                        color=bar_colors[row["plot_group"]],
                        edgecolor="black",
                        linewidth=0.8,
                        capsize=5,
                        width=0.55,
                        error_kw={"elinewidth": 1.2, "capthick": 1.2}
                    )

                # Individual points — open circles, sex color edge
                jitter_strength = 0.07
                for _, row in plot_df.iterrows():
                    xpos = x_positions[row["plot_group"]] + np.random.uniform(
                        -jitter_strength, jitter_strength
                    )
                    ax.scatter(
                        xpos,
                        row["mean/volume"],
                        facecolors="white",
                        edgecolors=sex_cfg["dot_color"],
                        linewidths=1.2,
                        s=30,
                        zorder=5
                    )

                # NS bracket
                bar_top = (summary["mean"] + summary["sem"]).max()
                tick_height = bar_top * 0.04
                bracket_y = bar_top * 1.12

                ax.plot([0, 1], [bracket_y, bracket_y], color="black", lw=1.2)
                ax.plot([0, 0], [bracket_y - tick_height, bracket_y], color="black", lw=1.2)
                ax.plot([1, 1], [bracket_y - tick_height, bracket_y], color="black", lw=1.2)
                ax.text(0.5, bracket_y + bar_top * 0.02, "ns", ha="center", va="bottom", fontsize=9)

                # Formatting
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.set_xticks([0, 1])
                ax.set_xticklabels(["Veh", "1 mg/kg\nPSI"], fontsize=10, fontweight="bold")
                ax.tick_params(axis="y", labelsize=10)
                for label in ax.get_yticklabels():
                    label.set_fontweight("bold")
                ax.set_xlim(-0.5, 1.5)

                if i == 0:
                    ax.set_title(col_titles[j], fontsize=11, fontweight="bold", pad=10)

                if j == 0:
                    ax.set_ylabel("Mean BDNF Protein Intensity", fontsize=10, fontweight="bold")
                else:
                    ax.set_yticklabels([])

        # Row labels
        for i, label in enumerate(row_labels):
            fig.text(
                0.01,
                axes[i, 0].get_position().y0 + axes[i, 0].get_position().height / 2,
                label,
                va="center", ha="left",
                fontsize=11, fontweight="bold",
                rotation=90
            )

        # Legend
        legend_elements = [
            mpatches.Patch(facecolor="#D4B896", edgecolor="black", label="Veh"),
            mpatches.Patch(facecolor="#CC5533", edgecolor="black", label="1 mg/kg PSI"),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
                       markeredgecolor=sex_cfg["dot_color"], markeredgewidth=1.5,
                       markersize=7, label=sex_cfg["label"]),
        ]
        fig.legend(
            handles=legend_elements, loc="upper right",
            bbox_to_anchor=(0.99, 0.88), frameon=True, fontsize=9
        )

        plt.savefig(sex_cfg["filename"], dpi=150, bbox_inches="tight")
        plt.show()
        print(f"Saved: {sex_cfg['filename']}")


def plot_bdnf_2x2_pub(df):
    """
    Two separate figures (one per region: DG and CA3).
    Each figure shows Day 1 and Day 7 on SAME AXIS with paired Vehicle vs Psi bars.
    Styling: black + professional blue filled bars, white dots with black edges.
    """

    # Clean data
    df = df.dropna(subset=["area", "sex", "group", "mean/volume"]).copy()
    df["sex"] = df["sex"].str.upper()
    df["group"] = df["group"].str.lower()

    areas = ["hilus", "ca3"]
    timepoints = ["pcb1", "pcb7"]
    area_labels = {"hilus": "DG", "ca3": "CA3"}

    # Professional poster colors - BLACK + PROFESSIONAL BLUE
    bar_colors = {"Vehicle": "#000000", "PSI": "#1F77B4"}
    sex_facecolors = {"M": "white", "F": "white"}
    sex_edgecolors = {"M": "black", "F": "#E0409A"}

    # Create one figure per region
    for area in areas:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        area_label = area_labels[area]
        fig.suptitle(f"{area_label} — BDNF Protein Expression", fontsize=14, fontweight="bold", y=0.98)
        
        # X-axis positions: Day 1 and Day 7, with Veh and PSI bars close together
        x_day1 = np.array([0, 0.15])         # Day 1: Veh at 0, PSI at 0.15
        x_day7 = np.array([0.5, 0.65])       # Day 7: Veh at 0.5, PSI at 0.65
        bar_width = 0.12
        
        # Find global y_max for consistent scaling
        area_data = df[df["area"] == area]
        y_max = area_data["mean/volume"].max() * 1.35
        
        # Plot each timepoint
        for tp_idx, (tp, x_pos) in enumerate([(tp, [x_day1, x_day7][i]) for i, tp in enumerate(timepoints)]):
            plot_df = df[(df["area"] == area) & (df["group"].isin([tp, "ctrl"]))].copy()
            plot_df["plot_group"] = plot_df["group"].map(lambda x: "Vehicle" if x == "ctrl" else "PSI")
            
            group_order = ["Vehicle", "PSI"]
            summary = plot_df.groupby("plot_group")["mean/volume"].agg(["mean", "sem"]).reindex(group_order).reset_index()
            
            # Bars - FILLED with black outlines
            for cond_idx, cond in enumerate(group_order):
                cond_summary = summary[summary["plot_group"] == cond]
                if len(cond_summary) > 0:
                    ax.bar(
                        x_pos[cond_idx],
                        cond_summary["mean"].values[0],
                        bar_width,
                        yerr=cond_summary["sem"].values[0],
                        color=bar_colors[cond],
                        edgecolor="black",
                        linewidth=2.5,
                        capsize=5,
                        error_kw={"elinewidth": 1.5, "capthick": 2},
                        zorder=2
                    )
            
            # Individual data points - white with black/pink edges
            jitter_strength = 0.02
            for _, row in plot_df.iterrows():
                cond_idx = 0 if row["plot_group"] == "Vehicle" else 1
                xpos = x_pos[cond_idx] + np.random.uniform(-jitter_strength, jitter_strength)
                ax.scatter(
                    xpos,
                    row["mean/volume"],
                    facecolors=sex_facecolors[row["sex"]],
                    edgecolors=sex_edgecolors[row["sex"]],
                    linewidths=1.2,
                    s=70,
                    zorder=5
                )
            
            # Significance bracket for this timepoint
            bar_top = summary["mean"].max() + summary["sem"].max()
            tick_height = y_max * 0.04
            bracket_y = bar_top + y_max * 0.05
            
            ax.plot([x_pos[0], x_pos[0], x_pos[1], x_pos[1]], 
                   [bracket_y, bracket_y + tick_height, bracket_y + tick_height, bracket_y], 
                   color="black", lw=1.2, zorder=6)
            ax.text((x_pos[0] + x_pos[1]) / 2, bracket_y + tick_height + y_max * 0.02, 
                   "ns", ha="center", va="bottom", fontsize=10, fontweight="bold", zorder=7)
        
        # Axes formatting
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.set_xticks([0.075, 0.575])  # Midpoint of each timepoint pair
        ax.set_xticklabels(["Day 1", "Day 7"], fontsize=12, fontweight="600")
        ax.set_ylabel("Mean BDNF Protein Intensity", fontsize=13, fontweight="600")
        ax.tick_params(axis="y", labelsize=11, length=3, width=0.8)
        ax.tick_params(axis="x", labelsize=11, length=0)
        ax.set_ylim(0, y_max)
        ax.grid(False)
        
        # Legend
        legend_elements = [
            mpatches.Patch(facecolor=bar_colors["Vehicle"], edgecolor="black", linewidth=2.5, label="Veh"),
            mpatches.Patch(facecolor=bar_colors["PSI"], edgecolor="black", linewidth=2.5, label="1 mg/kg PSI"),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
                       markeredgecolor='black', markeredgewidth=1.5, markersize=8, label='Male'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
                       markeredgecolor='#E0409A', markeredgewidth=1.5, markersize=8, label='Female'),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=11, frameon=True, framealpha=0.95, edgecolor="black")
        
        plt.tight_layout()
        
        # Save
        filename = f"bdnf_protein_{area}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {filename}")
        plt.show()
    
    # ─── PRINT P-VALUES ───────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("BDNF STATISTICAL RESULTS")
    print("="*70)
    
    for area in areas:
        area_label = area_labels[area]
        print(f"\n{area_label}:")
        for tp in timepoints:
            plot_df = df[(df["area"] == area) & (df["group"].isin([tp, "ctrl"]))].copy()
            plot_df["plot_group"] = plot_df["group"].map(lambda x: "Vehicle" if x == "ctrl" else "PSI")
            
            veh_data = plot_df[plot_df["plot_group"] == "Vehicle"]["mean/volume"].values
            psi_data = plot_df[plot_df["plot_group"] == "PSI"]["mean/volume"].values
            
            if len(veh_data) > 0 and len(psi_data) > 0:
                t_stat, p_val = stats.ttest_ind(veh_data, psi_data)
                tp_label = "Day 1" if tp == "pcb1" else "Day 7"
                
                print(f"  {tp_label}:")
                print(f"    Vehicle:  {np.mean(veh_data):.4f} ± {np.std(veh_data)/np.sqrt(len(veh_data)):.4f}  (n={len(veh_data)})")
                print(f"    PSI:      {np.mean(psi_data):.4f} ± {np.std(psi_data)/np.sqrt(len(psi_data)):.4f}  (n={len(psi_data)})")
                print(f"    p-value:  {p_val:.4f}")
    
    print("\n" + "="*70)