import pandas as pd

import pandas as pd

def load_and_merge_data(df1_path, df2_path):

    df1 = pd.read_csv(df1_path)
    df2 = pd.read_csv(df2_path)

    # 🔥 CLEAN animal_ID BEFORE merging
    df1["animal_ID"] = (
        df1["animal_ID"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df2["animal_ID"] = (
        df2["animal_ID"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # Merge AFTER cleaning
    merged_df = pd.merge(df1, df2, on="animal_ID", how="left")

    # Standardize other columns
    merged_df["group"] = (
        merged_df["drug_condition"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    merged_df["area"] = (
        merged_df["area"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    merged_df["sex"] = (
        merged_df["sex"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    merged_df.loc[~merged_df["sex"].isin(["M", "F"]), "sex"] = None

    return merged_df
    return merged_df
