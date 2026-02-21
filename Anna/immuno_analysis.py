import pandas as pd

import pandas as pd

def load_and_merge_data(df1_path, df2_path):
    """
    Load two CSV files, merge on 'animal_ID', and standardize key columns.
    """

    # Load CSVs
    df1 = pd.read_csv(df1_path)
    df2 = pd.read_csv(df2_path)

    # Merge
    merged_df = pd.merge(df1, df2, on='animal_ID')

    # Standardize GROUP (drug condition)
    merged_df['group'] = (
        merged_df['drug_condition']
        .str.strip()
        .str.lower()
    )

    # Standardize AREA
    merged_df['area'] = (
        merged_df['area']
        .str.strip()
        .str.lower()
    )

    # Standardize SEX
    merged_df['sex'] = (
        merged_df['sex']
        .str.strip()
        .str.upper()
    )

    return merged_df
