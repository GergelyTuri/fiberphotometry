import matplotlib.pyplot as plt

def plot_area(merged_df, area_name):
        # Remove missing sex or area rows
    df = df.dropna(subset=["sex", "area"])

    # Keep only known areas
    area_order = ["ca1", "ca3", "hilus"]
    df = df[df["area"].isin(area_order)].copy()

    # Convert to categorical
    df["area"] = pd.Categorical(df["area"], categories=area_order, ordered=True)

    ...
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
