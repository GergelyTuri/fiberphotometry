import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import ttest_ind

def time_to_seconds(time_str: str) -> float:
    """
    Convert time strings to seconds.

    Parameters:
    ----------
    time_str : str
        Time string in the format '%I:%M:%S%p'

    Returns:
    -------
    float
        Time in seconds.
    """
    t = datetime.strptime(time_str, "%I:%M:%S%p")
    return timedelta(hours=t.hour, minutes=t.minute, seconds=t.second).total_seconds()

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the CSV data from the specified file path.

    Parameters:
    ----------
    file_path : str
        Path to the CSV file.

    Returns:
    -------
    pd.DataFrame
        Loaded data.
    """
    df = pd.read_csv(file_path)
    print("Loaded data:")
    print(df.head())
    return df

def filter_data(df: pd.DataFrame, ca3_keyword: str = 'ca3', dg_keyword: str = 'dg', value_col: str = 'back_minus_value') -> tuple:
    """
    Remove negative values and filter data for CA3 and DG.

    Parameters:
    ----------
    df : pd.DataFrame
        The dataframe to filter.
    ca3_keyword : str, optional
        Keyword to filter CA3 data, by default 'ca3'.
    dg_keyword : str, optional
        Keyword to filter DG data, by default 'dg'.
    value_col : str, optional
        Column name for values, by default 'back_minus_value'.

    Returns:
    -------
    tuple
        Filtered dataframes for CA3 and DG.
    """
    df = df[df[value_col] >= 0]
    df_ca3 = df[df['image_id'].str.contains(ca3_keyword)]
    df_dg = df[df['image_id'].str.contains(dg_keyword)]
    print("\nFiltered CA3 data:")
    print(df_ca3.head())
    print("\nFiltered DG data:")
    print(df_dg.head())
    return df_ca3, df_dg

def aggregate_data(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    """
    Aggregate data by drug group and calculate mean and std.

    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame to aggregate.
    group_col : str
        Column name to group by.
    value_col : str
        Column name for values.

    Returns:
    -------
    pd.DataFrame
        Aggregated data.
    """
    stats = df.groupby(group_col)[value_col].agg(['mean', 'std']).reindex(['c', 'p1', 'p7'])
    return stats

def perform_ttest(group1: pd.Series, group2: pd.Series, group_name: str) -> tuple:
    """
    Perform a t-test between two groups and print the result.

    Parameters:
    ----------
    group1 : pd.Series
        First group of data.
    group2 : pd.Series
        Second group of data.
    group_name : str
        Name of the group for printing.

    Returns:
    -------
    tuple
        t-test result (t-statistic and p-value).
    """
    ttest_result = ttest_ind(group1, group2)
    print(f"\n{group_name} Control vs PCB T-test:")
    print(ttest_result)
    return ttest_result
