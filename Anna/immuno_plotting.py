import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import ttest_ind

COLORS = {
    'Control': '#E8D4C0',
    '1 Day':   '#E07A5F',
    '7 Day':   '#B5451B',
}

SEX_COLORS = {
    'M': '#1A1A1A',
    'F': '#C2185B',
}

def get_star(p):
    if p < 0.001:    return '***'
    elif p < 0.01:   return '**'
    elif p <= 0.055: return '*'
    else:            return 'ns'

def plot_area(merged_df, area_name):
    area_df = merged_df[merged_df['area'] == area_name].copy()
    area_df['sex'] = area_df['sex'].str.upper()

    label_map = {'ctrl': 'Control', 'pcb1': '1 Day', 'pcb7': '7 Day'}
    area_df['label'] = area_df['group'].map(label_map)
    area_df = area_df.dropna(subset=['label', 'mean/volume', 'sex'])

    group_order = ['Control', '1 Day', '7 Day']
    grouped = (
        area_df.groupby('label')['mean/volume']
        .agg(['mean', 'sem'])
        .reindex(group_order)
        .reset_index()
    )
    x_pos = np.arange(len(group_order)) * 0.6

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.bar(x_pos, grouped['mean'].fillna(0),
           color=[COLORS.get(g, '#888888') for g in group_order],
           edgecolor='black', linewidth=1.0, width=0.35, zorder=2)

    for i, row in grouped.iterrows():
        if not np.isnan(row['mean']):
            ax.errorbar(x_pos[i], row['mean'], yerr=row['sem'],
                        fmt='none', ecolor='black',
                        elinewidth=1.2, capsize=5, capthick=1.2, zorder=4)

    dot_max = 0.0
    for i, group_label in enumerate(group_order):
        group_rows = area_df[area_df['label'] == group_label]
        n = len(group_rows)
        jitters = np.linspace(-0.12, 0.12, n) if n > 1 else [0.0]
        np.random.shuffle(jitters)
        for (_, row), jit in zip(group_rows.iterrows(), jitters):
            dot_color = SEX_COLORS.get(row['sex'], '#333333')
            ax.scatter(x_pos[i] + jit, row['mean/volume'],
                       facecolors='white', edgecolors=dot_color,
                       linewidths=1.5, s=50, zorder=5, marker='o')
            dot_max = max(dot_max, row['mean/volume'])

    y_max = dot_max * 1.35
    ctrl_vals = area_df[area_df['label'] == 'Control']['mean/volume']
    for j, psi_label in enumerate(['1 Day', '7 Day']):
        psi_vals = area_df[area_df['label'] == psi_label]['mean/volume']
        if len(ctrl_vals) >= 2 and len(psi_vals) >= 2:
            _, pval = ttest_ind(ctrl_vals, psi_vals, equal_var=False)
            label = get_star(pval)
            x0, x1 = x_pos[0], x_pos[group_order.index(psi_label)]
            bracket_y = dot_max + y_max * 0.08 * (j + 1)
            tick_h = y_max * 0.02
            ax.plot([x0, x0, x1, x1],
                    [bracket_y, bracket_y + tick_h, bracket_y + tick_h, bracket_y],
                    lw=1.0, c='black', zorder=6)
            ax.text((x0 + x1) / 2, bracket_y + tick_h + y_max * 0.005, label,
                    ha='center', va='bottom',
                    fontsize=11 if label != 'ns' else 8,
                    color='black' if label != 'ns' else '#666666',
                    fontweight='bold' if label != 'ns' else 'normal', zorder=7)

    ax.set_title(f'Mean BDNF in {area_name}', fontsize=13, fontweight='bold', pad=10)
    ax.set_xlim(x_pos[0] - 0.45, x_pos[-1] + 0.45)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(group_order, fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean/Volume', fontsize=13, fontweight='bold')
    ax.set_ylim(0, y_max)
    ax.tick_params(axis='y', labelsize=11, length=3)
    ax.tick_params(axis='x', labelsize=12, length=3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    handles = [mpatches.Patch(facecolor=COLORS[g], edgecolor='black',
                               linewidth=0.8, label=g) for g in group_order]
    handles += [
        mlines.Line2D([], [], marker='o', linestyle='None', markersize=9,
                      markerfacecolor='white', markeredgecolor=SEX_COLORS['M'],
                      markeredgewidth=1.8, label='Male'),
        mlines.Line2D([], [], marker='o', linestyle='None', markersize=9,
                      markerfacecolor='white', markeredgecolor=SEX_COLORS['F'],
                      markeredgewidth=1.8, label='Female'),
    ]
    ax.legend(handles=handles, fontsize=10, loc='upper right',
              frameon=True, framealpha=0.92, edgecolor='#cccccc')

    plt.tight_layout()
    save_path = f'/gdrive/MyDrive/csvs/Mean_BDNF_{area_name}.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'  Saved -> {save_path}')
    plt.show()
