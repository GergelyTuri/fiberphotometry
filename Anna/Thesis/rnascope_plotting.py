import matplotlib.pyplot as plt
import pandas as pd
def plot_data(stats: pd.DataFrame, df: pd.DataFrame, area_name: str, colors: list, value_col: str = 'back_minus_value', group_col: str = 'drug') -> None:
    """
    Plot the data with bar plots and scatter points.

    Parameters:
    ----------
    stats : pd.DataFrame
        Aggregated statistics (mean and std).
    df : pd.DataFrame
        Original data.
    area_name : str
        Name of the area (CA3 or DG).
    colors : list
        List of colors for plotting.
    value_col : str, optional
        Column name for values, by default 'back_minus_value'.
    group_col : str, optional
        Column name for groups, by default 'drug'.
    """
    plt.figure(figsize=(10, 6))
    plt.bar(stats.index, stats['mean'], yerr=stats['std'], color=colors, capsize=5)
    plt.scatter(df[group_col], df[value_col], color='black', alpha=0.6)
    plt.title(f'{area_name} Mean BDNF', fontsize=17)
    plt.xlabel('Drug Group', fontsize=16)
    plt.ylabel('Mean BDNF', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()
