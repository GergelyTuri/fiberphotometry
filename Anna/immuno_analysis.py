import pandas as pd

import pandas as pd

def load_and_merge_data(df1_path, df2_path):
    df1 = pd.read_csv(df1_path)
    df1.columns = df1.columns.str.strip()
    df2 = pd.read_csv(df2_path)
    df2.columns = df2.columns.str.strip()

    # drop junk rows that have no real animal data
    df1 = df1.dropna(subset=['sex', 'weight'])

    merged_df = pd.merge(df1, df2, on='animal_ID')

    merged_df['group'] = merged_df['drug_condition'].str.extract(r'(pcb|c|ctrl)', expand=False) + \
                         merged_df['sac-injection'].astype(str)
    merged_df.loc[merged_df['drug_condition'].str.contains('ctrl', case=False, na=False), 'group'] = 'ctrl'

    return merged_df
