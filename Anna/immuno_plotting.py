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

