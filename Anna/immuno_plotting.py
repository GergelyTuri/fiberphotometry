import matplotlib.pyplot as plt

def plot_area(merged_df, area_name):
    """
    Plot data for a specific area with muted colors.

    Parameters:
    ----------
    merged_df : pd.DataFrame
        The merged DataFrame containing the data.
    area_name : str
        The area name to filter the data by and plot.
    """
    area_df = merged_df[merged_df['area'] == area_name]

    # Calculate the mean and standard error for each group
    grouped = area_df.groupby('group')['mean/volume'].agg(['mean', 'sem']).reset_index()

    # Define muted colors
    colors = {'ctrl': '#A0C4FF', 'pcb1': '#7B9FAB', 'pcb7': '#B7C3D0'}

    # Plot the data
    fig, ax = plt.subplots(figsize=(10, 6))

    # Bar plots with error bars
    for group, color in colors.items():
        group_data = grouped[grouped['group'] == group]
        ax.bar(group, group_data['mean'], yerr=group_data['sem'], color=color, capsize=5, label=group)

    # Overlay data points
    for group in colors.keys():
        group_data = area_df[area_df['group'] == group]['mean/volume']
        ax.scatter([group] * len(group_data), group_data, color='black', alpha=0.6)

    # Labels and title with increased font size
    ax.set_ylabel('Mean Volume', fontsize=16)
    ax.set_xlabel('Group', fontsize=16)
    ax.set_title(f'Mean BDNF in {area_name}', fontsize=16)

    # Set tick parameters
    ax.tick_params(axis='both', which='major', labelsize=12)

    ax.legend(fontsize=12)

    # Show the plot
    plt.show()
