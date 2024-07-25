import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go

def plot_serotonin_levels(averages, overall_averages):
    categories = ['Mobile', 'Immobile']
    control_means = [overall_averages['control_mobile'], overall_averages['control_immobile']]
    pcb_means = [overall_averages['pcb_mobile'], overall_averages['pcb_immobile']]

    control_mobile_points = [avg['control_mobile'] for avg in averages]
    pcb_mobile_points = [avg['pcb_mobile'] for avg in averages]
    control_immobile_points = [avg['control_immobile'] for avg in averages]
    pcb_immobile_points = [avg['pcb_immobile'] for avg in averages]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width / 2, control_means, width, label='Control', color='#7B9FAB')
    bars2 = ax.bar(x + width / 2, pcb_means, width, label='PCB', color='#B7C3D0')

    ax.scatter([x[0] - width / 2] * len(control_mobile_points), control_mobile_points, color='#4F4F4F', zorder=10)
    ax.scatter([x[0] + width / 2] * len(pcb_mobile_points), pcb_mobile_points, color='#4F4F4F', zorder=10)
    ax.scatter([x[1] - width / 2] * len(control_immobile_points), control_immobile_points, color='#4F4F4F', zorder=10)
    ax.scatter([x[1] + width / 2] * len(pcb_immobile_points), pcb_immobile_points, color='#4F4F4F', zorder=10)

    for i in range(len(averages)):
        ax.plot([x[0] - width / 2, x[0] + width / 2], [control_mobile_points[i], pcb_mobile_points[i]], color='#4F4F4F', linestyle='--')
        ax.plot([x[1] - width / 2, x[1] + width / 2], [control_immobile_points[i], pcb_immobile_points[i]], color='#4F4F4F', linestyle='--')

    ax.set_xlabel('Condition', fontsize=16)
    ax.set_ylabel('Average Serotonin Activity (Z-Score)', fontsize=16)
    ax.set_title('Average Serotonin Level During Mobile and Immobile States', fontsize=17)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=13)
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_velocities(velocities, labels, title="Average Velocity Comparison", xlabel="Files", ylabel="Average Velocity", colors=None):
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
    plt.bar(labels, velocities, color=colors if colors else ['blue', 'orange'])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
def plot_distances(distances, labels, title="Total Distance Traveled Comparison", xlabel="Files", ylabel="Total Distance Traveled", colors=None):
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
    plt.bar(labels, distances, color=colors if colors else ['blue', 'orange'])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

def plot_zscore_signal(time_x, zscore_signal, baseline_end_idx, title="Z-score of 5HT2A Signal"):
    baseline_zscore = np.mean(zscore_signal[:baseline_end_idx])

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(time_x, zscore_signal, linewidth=2, color='blue', label='Z-score (serotonin - ISOS)')
    ax1.axhline(y=baseline_zscore, color='red', linestyle='--', label='Baseline')
    
    ax1.set_ylabel('Z-score')
    ax1.set_xlabel('Seconds')
    ax1.set_title(title)
    ax1.legend()
    
    fig.tight_layout()
    plt.show()

def plot_overlay(velocity_data, serotonin_data, injection_relative_time, stop_time_seconds, start_time_seconds):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=velocity_data['Time (seconds)'], y=velocity_data['Smoothed Velocity (cm/s)'], mode='lines', name='Mobility'))
    fig.add_trace(go.Scatter(x=serotonin_data['Time (seconds)'], y=serotonin_data['Z-score'], mode='lines', name='Z-score (serotonin - ISOS)', yaxis='y2'))
    fig.add_vline(x=injection_relative_time, line_width=2, line_dash="dash", line_color="red", annotation_text="Injection", annotation_position="top")
    
    fig.update_layout(
        title='Overlay of Mobility and Serotonin Z-score Over Time',
        xaxis_title='Time (s)',
        yaxis_title='Mobility',
        xaxis=dict(range=[0, stop_time_seconds - start_time_seconds]),
        yaxis=dict(title='Velocity (cm/s)', tickmode='array'),
        yaxis2=dict(title='Z-score', overlaying='y', side='right'),
        legend=dict(x=0, y=1.1, orientation='h')
    )
    
    fig.show()

def plot_velocities(velocities, labels, title="Average Velocity Comparison", xlabel="Files", ylabel="Average Velocity", colors=None):
    plt.figure(figsize=(8, 6))
    plt.bar(labels, velocities, color=colors if colors else ['blue', 'orange'])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def plot_distances(distances, labels, title="Total Distance Traveled Comparison", xlabel="Files", ylabel="Total Distance Traveled", colors=None):
    plt.figure(figsize=(8, 6))
    plt.bar(labels, distances, color=colors if colors else ['blue', 'orange'])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
