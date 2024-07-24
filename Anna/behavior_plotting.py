import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_behavior_frequencies(pcb_baseline_frequencies: dict, pcb_post_injection_frequencies: dict, saline_baseline_frequencies: dict, saline_post_injection_frequencies: dict, behavior_labels: dict, colors: list) -> None:
    """
    Plot the behavior frequencies for PCB and saline conditions.

    Parameters:
    ----------
    pcb_baseline_frequencies : dict
        Dictionary containing the PCB baseline behavior frequencies.
    pcb_post_injection_frequencies : dict
        Dictionary containing the PCB post-injection behavior frequencies.
    saline_baseline_frequencies : dict
        Dictionary containing the saline baseline behavior frequencies.
    saline_post_injection_frequencies : dict
        Dictionary containing the saline post-injection behavior frequencies.
    behavior_labels : dict
        Dictionary containing the behavior labels.
    colors : list
        List of colors for the bars in the plot.
    """
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 12))

    behaviors = list(behavior_labels.keys())

    # PCB baseline
    axes[0, 0].bar([behavior_labels[key] for key in behaviors], [pcb_baseline_frequencies[key] for key in behaviors], color=colors)
    axes[0, 0].set_xlabel('Behavior')
    axes[0, 0].set_ylabel('Frequency (occurrences per second)')
    axes[0, 0].set_title('PCB: Behavior Frequency During Baseline')

    # PCB post-injection
    axes[0, 1].bar([behavior_labels[key] for key in behaviors], [pcb_post_injection_frequencies[key] for key in behaviors], color=colors)
    axes[0, 1].set_xlabel('Behavior')
    axes[0, 1].set_ylabel('Frequency (occurrences per second)')
    axes[0, 1].set_title('PCB: Behavior Frequency After Injection')

    # Saline baseline
    axes[1, 0].bar([behavior_labels[key] for key in behaviors], [saline_baseline_frequencies[key] for key in behaviors], color=colors)
    axes[1, 0].set_xlabel('Behavior')
    axes[1, 0].set_ylabel('Frequency (occurrences per second)')
    axes[1, 0].set_title('Saline: Behavior Frequency During Baseline')

    # Saline post-injection
    axes[1, 1].bar([behavior_labels[key] for key in behaviors], [saline_post_injection_frequencies[key] for key in behaviors], color=colors)
    axes[1, 1].set_xlabel('Behavior')
    axes[1, 1].set_ylabel('Frequency (occurrences per second)')
    axes[1, 1].set_title('Saline: Behavior Frequency After Injection')

    plt.tight_layout()
    plt.show()

    # Save the figure as a PNG file with 300 DPI resolution
    fig.savefig('/gdrive/Shareddrives/Turi_lab/Data/psilocybin_project/PCB_Serotonin/behavior_frequencies.png', format='png', dpi=300)

    # Save the figure as an SVG file
    fig.savefig('/gdrive/Shareddrives/Turi_lab/Data/psilocybin_project/PCB_Serotonin/behavior_frequencies.svg', format='svg')


def plot_z_score_differences_v2(z_score_differences_pcb_mean: dict, z_score_differences_saline_mean: dict, z_score_differences_pcb: dict, z_score_differences_saline: dict, behavior_labels: dict) -> None:
    """
    Plot the averaged Z-score differences for PCB and saline conditions.

    Parameters:
    ----------
    z_score_differences_pcb_mean : dict
        Dictionary containing the mean Z-score differences for PCB.
    z_score_differences_saline_mean : dict
        Dictionary containing the mean Z-score differences for saline.
    z_score_differences_pcb : dict
        Dictionary containing the Z-score differences for individual PCB animals.
    z_score_differences_saline : dict
        Dictionary containing the Z-score differences for individual saline animals.
    behavior_labels : dict
        Dictionary containing the behavior labels.
    """
    x = np.arange(len(behavior_labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 8))
    bars_pcb = ax.bar(x - width/2, z_score_differences_pcb_mean.values(), width, label='PCB', color='#377eb8')
    bars_saline = ax.bar(x + width/2, z_score_differences_saline_mean.values(), width, label='Saline', color='#808080')

    # Overlay individual data points and connect the dots with lines
    for i, key in enumerate(behavior_labels.keys()):
        pcb_y_values = z_score_differences_pcb[key]
        saline_y_values = z_score_differences_saline[key]

        pcb_scatter = ax.scatter([x[i] - width/2] * len(pcb_y_values), pcb_y_values, color='#377eb8', edgecolor='black', zorder=5)
        saline_scatter = ax.scatter([x[i] + width/2] * len(saline_y_values), saline_y_values, color='#808080', edgecolor='black', zorder=5)

        # Connect the points between the animals
        for j in range(len(pcb_y_values)):
            ax.plot([x[i] - width/2, x[i] + width/2], [pcb_y_values[j], saline_y_values[j]], color='black', linestyle='-', zorder=4)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Behavior', fontsize=16)
    ax.set_ylabel('Z-score Difference (Post-Injection - Baseline)', fontsize=17)
    ax.set_title('Behavior Frequency After Injection (Normalized with Baseline)', fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels([behavior_labels[key] for key in z_score_differences_pcb.keys()], fontsize=14)
    ax.legend()

    ax.axhline(0, color='grey', linestyle='--')
    fig.tight_layout()

    plt.show()
