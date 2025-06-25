import pandas as pd

def load_and_merge_data(df1_path, df2_path):
    df1 = pd.read_csv(df1_path)
    df2 = pd.read_csv(df2_path)
    merged_df = pd.merge(df1, df2, on='animal_ID')

    # Standardize group label to match plotting keys
    merged_df['group'] = merged_df['drug_condition'].str.extract(r'(pcb|c|ctrl)', expand=False) + \
                         merged_df['sac-injection'].astype(str)

    # Special case: if 'ctrl' already means control across timepoints, keep it as 'ctrl'
    merged_df.loc[merged_df['drug_condition'].str.contains('ctrl', case=False, na=False), 'group'] = 'ctrl'

    return merged_df
