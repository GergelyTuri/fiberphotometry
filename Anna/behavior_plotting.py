import matplotlib.pyplot as plt
import pandas as pd

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
