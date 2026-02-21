import pandas as pd


def load_and_merge_data(bdnf_path, metadata_path):
    """
    Loads BDNF image-level data and animal metadata.
    Cleans IDs.
    Merges safely.
    Aggregates to ONE value per animal per area.
    """

    # Load
    bdnf = pd.read_csv(bdnf_path)
    meta = pd.read_csv(metadata_path)

    # Clean animal IDs
    bdnf["animal_ID"] = (
        bdnf["animal_ID"].astype(str).str.strip().str.upper()
    )

    meta["animal_ID"] = (
        meta["animal_ID"].astype(str).str.strip().str.upper()
    )

    # Keep only needed metadata columns
    meta = meta[["animal_ID", "sex", "drug_condition"]]

    # Ensure one row per animal in metadata
    meta = meta.drop_duplicates(subset="animal_ID")

    # Merge
    merged = pd.merge(
        bdnf,
        meta,
        on="animal_ID",
        how="left"
    )

    # Clean categorical fields
    merged["area"] = (
        merged["area"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    merged["group"] = (
        merged["drug_condition"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    merged["sex"] = (
        merged["sex"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    merged.loc[~merged["sex"].isin(["M", "F"]), "sex"] = None

    # Ensure numeric
    merged["mean/volume"] = pd.to_numeric(
        merged["mean/volume"],
        errors="coerce"
    )

    # 🔬 CRITICAL STEP:
    # Aggregate to ONE value per animal per area
    aggregated = (
        merged
        .groupby(["animal_ID", "area", "sex", "group"], as_index=False)
        ["mean/volume"]
        .mean()
    )

    return aggregated
