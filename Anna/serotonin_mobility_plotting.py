import matplotlib.pyplot as plt
import numpy as np

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
