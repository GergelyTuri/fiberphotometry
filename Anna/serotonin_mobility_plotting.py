import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines
import matplotlib.patches as mpatches


# ─────────────────────────────────────────────────────────────────────────────
# COLOR CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    'Control': '#4A4A4A',    # Dark grey
    'PSI':     '#1F77B4',    # Professional blue
}

EDGE_COLORS = {
    'Control': '#000000',    # Black outline
    'PSI':     '#000000',    # Black outline
}


def _bar_handle(condition):
    """Filled-patch legend handle for a condition bar with colored edge."""
    return mpatches.Patch(
        facecolor=COLORS[condition],
        edgecolor=EDGE_COLORS[condition],
        linewidth=2.0,
        label=condition,
    )


def plot_serotonin_levels(averages, overall_averages):
    """
    Plots the average serotonin levels during mobile and immobile states for control and PSI groups.
    Styled with grey for control and blue for PSI, matching BDNF figure style.
    
    Parameters:
    averages (list of dict): A list of dictionaries containing individual serotonin level averages for each condition.
                             Each dictionary should have the keys 'control_mobile', 'control_immobile', 'psi_mobile', and 'psi_immobile'.
    overall_averages (dict): A dictionary containing the overall average serotonin levels for each condition.
                             Should have the keys 'control_mobile', 'control_immobile', 'psi_mobile', and 'psi_immobile'.
    Returns:
    fig: The matplotlib figure object.
    """
    categories = ["Mobile", "Immobile"]
    control_means = [
        overall_averages["control_mobile"],
        overall_averages["control_immobile"],
    ]
    psi_means = [overall_averages["psi_mobile"], overall_averages["psi_immobile"]]

    control_mobile_points = [avg["control_mobile"] for avg in averages]
    psi_mobile_points = [avg["psi_mobile"] for avg in averages]
    control_immobile_points = [avg["control_immobile"] for avg in averages]
    psi_immobile_points = [avg["psi_immobile"] for avg in averages]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    # Bars with grey for control and blue for PSI
    bars1 = ax.bar(
        x - width / 2, 
        control_means, 
        width, 
        label="Control",
        color=COLORS['Control'],
        edgecolor=EDGE_COLORS['Control'],
        linewidth=2.5,
    )
    bars2 = ax.bar(
        x + width / 2, 
        psi_means, 
        width, 
        label="PSI",
        color=COLORS['PSI'],
        edgecolor=EDGE_COLORS['PSI'],
        linewidth=2.5,
    )

    # Scatter points (individual data)
    ax.scatter(
        [x[0] - width / 2] * len(control_mobile_points),
        control_mobile_points,
        color="#FFFFFF",
        edgecolors="#000000",
        linewidths=1.2,
        s=60,
        zorder=10,
    )
    ax.scatter(
        [x[0] + width / 2] * len(psi_mobile_points),
        psi_mobile_points,
        color="#FFFFFF",
        edgecolors="#000000",
        linewidths=1.2,
        s=60,
        zorder=10,
    )
    ax.scatter(
        [x[1] - width / 2] * len(control_immobile_points),
        control_immobile_points,
        color="#FFFFFF",
        edgecolors="#000000",
        linewidths=1.2,
        s=60,
        zorder=10,
    )
    ax.scatter(
        [x[1] + width / 2] * len(psi_immobile_points),
        psi_immobile_points,
        color="#FFFFFF",
        edgecolors="#000000",
        linewidths=1.2,
        s=60,
        zorder=10,
    )

    # Connecting lines between control and PSI for each animal
    for i in range(len(averages)):
        ax.plot(
            [x[0] - width / 2, x[0] + width / 2],
            [control_mobile_points[i], psi_mobile_points[i]],
            color="#4F4F4F",
            linestyle="--",
            linewidth=1.0,
            zorder=1,
        )
        ax.plot(
            [x[1] - width / 2, x[1] + width / 2],
            [control_immobile_points[i], psi_immobile_points[i]],
            color="#4F4F4F",
            linestyle="--",
            linewidth=1.0,
            zorder=1,
        )

    # Axes styling
    ax.set_xlabel("Condition", fontsize=14, fontweight='bold')
    ax.set_ylabel("Average Serotonin Activity (Z-Score)", fontsize=14, fontweight='bold')
    ax.set_title("Average Serotonin Level During Mobile and Immobile States", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12, fontweight='600')
    
    # Legend with bar colors
    handles = [_bar_handle('Control'), _bar_handle('PSI')]
    ax.legend(
        handles=handles,
        fontsize=11,
        loc='upper left',
        frameon=True,
        framealpha=0.95,
        edgecolor='black',
    )
    
    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.tick_params(axis='y', labelsize=11)
    ax.tick_params(axis='x', labelsize=11)

    plt.tight_layout()
    return fig


def plot_velocities(
    velocities,
    labels,
    title="Average Velocity Comparison",
    xlabel="Group",
    ylabel="Average Velocity",
    colors=None,
):
    """
    Plot a bar graph of velocities in the grey/blue style.

    Parameters:
    ----------
    velocities : list of float
        List of mean velocities to plot.
    labels : list of str
        Corresponding labels for the velocities (e.g., ['Control', 'PSI']).
    title : str, optional
        Title of the plot.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    colors : list of str, optional
        List of colors for the bars. If None, uses Control/PSI default colors.
    """
    if colors is None:
        colors = [COLORS.get(label, '#888888') for label in labels]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bars = ax.bar(
        labels, 
        velocities, 
        color=colors,
        edgecolor='#000000',
        linewidth=2.5,
    )
    
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')
    
    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.tick_params(axis='y', labelsize=11)
    ax.tick_params(axis='x', labelsize=11)
    
    plt.tight_layout()
    return fig


def plot_distances(
    distances,
    labels,
    title="Total Distance Traveled Comparison",
    xlabel="Group",
    ylabel="Total Distance Traveled",
    colors=None,
):
    """
    Plot a bar graph of distances traveled in the grey/blue style.

    Parameters:
    ----------
    distances : list of float
        List of total distances to plot.
    labels : list of str
        Corresponding labels for the distances (e.g., ['Control', 'PSI']).
    title : str, optional
        Title of the plot.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    colors : list of str, optional
        List of colors for the bars. If None, uses Control/PSI default colors.
    """
    if colors is None:
        colors = [COLORS.get(label, '#888888') for label in labels]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bars = ax.bar(
        labels, 
        distances, 
        color=colors,
        edgecolor='#000000',
        linewidth=2.5,
    )
    
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')
    
    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.tick_params(axis='y', labelsize=11)
    ax.tick_params(axis='x', labelsize=11)
    
    plt.tight_layout()
    return fig