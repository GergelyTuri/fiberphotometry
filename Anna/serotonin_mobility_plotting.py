import matplotlib.pyplot as plt
import numpy as np


def plot_serotonin_levels(averages, overall_averages):
    """
    Plots the average serotonin levels during mobile and immobile states for control and PCB groups.
    Parameters:
    averages (list of dict): A list of dictionaries containing individual serotonin level averages for each condition.
                             Each dictionary should have the keys 'control_mobile', 'control_immobile', 'pcb_mobile', and 'pcb_immobile'.
    overall_averages (dict): A dictionary containing the overall average serotonin levels for each condition.
                             Should have the keys 'control_mobile', 'control_immobile', 'pcb_mobile', and 'pcb_immobile'.
    Returns:
    None: This function displays a bar plot with scatter points and connecting lines.
    """
    categories = ["Mobile", "Immobile"]
    control_means = [
        overall_averages["control_mobile"],
        overall_averages["control_immobile"],
    ]
    pcb_means = [overall_averages["pcb_mobile"], overall_averages["pcb_immobile"]]

    control_mobile_points = [avg["control_mobile"] for avg in averages]
    pcb_mobile_points = [avg["pcb_mobile"] for avg in averages]
    control_immobile_points = [avg["control_immobile"] for avg in averages]
    pcb_immobile_points = [avg["pcb_immobile"] for avg in averages]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(
        x - width / 2, control_means, width, label="Control", color="#7B9FAB"
    )
    bars2 = ax.bar(x + width / 2, pcb_means, width, label="PCB", color="#B7C3D0")

    ax.scatter(
        [x[0] - width / 2] * len(control_mobile_points),
        control_mobile_points,
        color="#4F4F4F",
        zorder=10,
    )
    ax.scatter(
        [x[0] + width / 2] * len(pcb_mobile_points),
        pcb_mobile_points,
        color="#4F4F4F",
        zorder=10,
    )
    ax.scatter(
        [x[1] - width / 2] * len(control_immobile_points),
        control_immobile_points,
        color="#4F4F4F",
        zorder=10,
    )
    ax.scatter(
        [x[1] + width / 2] * len(pcb_immobile_points),
        pcb_immobile_points,
        color="#4F4F4F",
        zorder=10,
    )

    for i in range(len(averages)):
        ax.plot(
            [x[0] - width / 2, x[0] + width / 2],
            [control_mobile_points[i], pcb_mobile_points[i]],
            color="#4F4F4F",
            linestyle="--",
        )
        ax.plot(
            [x[1] - width / 2, x[1] + width / 2],
            [control_immobile_points[i], pcb_immobile_points[i]],
            color="#4F4F4F",
            linestyle="--",
        )

    ax.set_xlabel("Condition", fontsize=16)
    ax.set_ylabel("Average Serotonin Activity (Z-Score)", fontsize=16)
    ax.set_title(
        "Average Serotonin Level During Mobile and Immobile States", fontsize=17
    )
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=13)
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_velocities(
    velocities,
    labels,
    title="Average Velocity Comparison",
    xlabel="Files",
    ylabel="Average Velocity",
    colors=None,
):
    """
    Plot a bar graph of velocities.

    Parameters:
    ----------
    velocities : list of float
        List of mean velocities to plot.
    labels : list of str
        Corresponding labels for the velocities.
    title : str, optional
        Title of the plot (default is "Average Velocity Comparison").
    xlabel : str, optional
        Label for the x-axis (default is "Files").
    ylabel : str, optional
        Label for the y-axis (default is "Average Velocity").
    colors : list of str, optional
        List of colors for the bars (default is None, which uses default colors).
    """
    plt.figure(figsize=(8, 6))
    plt.bar(labels, velocities, color=colors if colors else ["blue", "orange"])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def plot_distances(
    distances,
    labels,
    title="Total Distance Traveled Comparison",
    xlabel="Files",
    ylabel="Total Distance Traveled",
    colors=None,
):
    """
    Plot a bar graph of distances traveled.

    Parameters:
    ----------
    distances : list of float
        List of total distances to plot.
    labels : list of str
        Corresponding labels for the distances.
    title : str, optional
        Title of the plot (default is "Total Distance Traveled Comparison").
    xlabel : str, optional
        Label for the x-axis (default is "Files").
    ylabel : str, optional
        Label for the y-axis (default is "Total Distance Traveled").
    colors : list of str, optional
        List of colors for the bars (default is None, which uses default colors).
    """
    plt.figure(figsize=(8, 6))
    plt.bar(labels, distances, color=colors if colors else ["blue", "orange"])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.title(title)
