import pandas as pd

def load_and_merge_data(df1_path, df2_path):
    """
    Load two CSV files and merge them on 'animal_ID'.

    Parameters:
    ----------
    df1_path : str
        Path to the first CSV file.
    df2_path : str
        Path to the second CSV file.

    Returns:
    -------
    pd.DataFrame
        Merged DataFrame.
    """
    # Load the two CSV files into DataFrames
    df1 = pd.read_csv(df1_path)
    df2 = pd.read_csv(df2_path)

    # Merge the DataFrames on 'animal_ID'
    merged_df = pd.merge(df1, df2, on='animal_ID')

    # Update the 'group' column based on 'drug_condition' from df1
    merged_df['group'] = merged_df['drug_condition']

    return merged_df
