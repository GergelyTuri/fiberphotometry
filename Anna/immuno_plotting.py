import matplotlib.pyplot as plt

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

import matplotlib.pyplot as plt
import numpy as np

def plot_bdnf_figure(merged_df):

    # Ensure standardized columns
    df = merged_df.copy()

    regions = ["dg", "ca3"]
    timepoints = ["pcb1", "pcb7"]  # 1 day, 7 day

    region_labels = {"dg": "DG", "ca3": "CA3"}
    time_labels = {"pcb1": "Post-Drug Day 1",
                   "pcb7": "Post-Drug Day 7"}

    group_labels = {"ctrl": "Veh",
                    "pcb1": "1 mg/kg PSI",
                    "pcb7": "1 mg/kg PSI"}

    bar_colors = {
        "ctrl": "#d8c4ad",
        "pcb1": "#d97759",
        "pcb7": "#d97759"
    }

    sex_colors = {"M": "black", "F": "#d81b60"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=True)

    for i, region in enumerate(regions):
        for j, time in enumerate(timepoints):

            ax = axes[i, j]

            # Filter data
            sub = df[
                (df["area"] == region) &
                ((df["group"] == "ctrl") | (df["group"] == time))
            ]

            # Group stats
            grouped = sub.groupby("group")["mean/volume"].agg(["mean", "sem"]).reset_index()

            x_positions = np.arange(len(grouped))

            for k, row in grouped.iterrows():
                ax.bar(
                    x_positions[k],
                    row["mean"],
                    yerr=row["sem"],
                    color=bar_colors[row["group"]],
                    edgecolor="black",
                    capsize=5,
                    width=0.6
                )

            # Scatter points
            for k, group in enumerate(grouped["group"]):
                scatter_data = sub[sub["group"] == group]
                for _, r in scatter_data.iterrows():
                    ax.scatter(
                        x_positions[k],
                        r["mean/volume"],
                        color=sex_colors[r["sex"]],
                        edgecolor="black",
                        s=70,
                        zorder=3
                    )

            ax.set_xticks(x_positions)
            ax.set_xticklabels([group_labels[g] for g in grouped["group"]], fontsize=11)

            if i == 0:
                ax.set_title(time_labels[time], fontsize=14)

            if j == 0:
                ax.set_ylabel("Mean BDNF Protein Intensity", fontsize=12)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Add simple "ns" bracket
            if len(grouped) == 2:
                y_max = grouped["mean"].max() + grouped["sem"].max() + 20
                ax.plot([0, 0, 1, 1], [y_max, y_max+5, y_max+5, y_max], color="black")
                ax.text(0.5, y_max+7, "ns", ha="center")

            ax.set_ylim(0, df["mean/volume"].max() * 1.3)

            if j == 0:
                ax.annotate(region_labels[region],
                            xy=(-0.6, 0.5),
                            xycoords="axes fraction",
                            rotation=90,
                            va="center",
                            fontsize=13)

    # Legend
    handles = []
    for sex, color in sex_colors.items():
        handles.append(
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=color, markeredgecolor="black",
                       markersize=8, label=sex)
        )

    fig.legend(handles=handles, title="Sex",
               loc="upper right", bbox_to_anchor=(1.05, 0.95))

    fig.suptitle("Psilocybin Does Not Alter BDNF Protein Expression",
                 fontsize=16)

    plt.tight_layout()
    plt.show()
