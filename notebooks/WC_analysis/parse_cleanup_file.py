# script for parsing the cleanup file and creating a new file with the relevant information

from os.path import join

import metadata
import pandas as pd

from src import TDT_experiment as tdt


def main():
    # Read the cleanup metadata
    cleanup_metadata = pd.read_excel(metadata.CLEANUP_METADATA)

    df = cleanup_metadata.copy()
    # Convert to lowercase and split by '_'
    split_col = df["Folder Name"].str.lower().str.split("_")

    # Validate that all rows split into exactly 6 parts
    if not (split_col.apply(len) == 6).all():
        raise ValueError("Some rows do not split into exactly 6 parts.")

    expanded = split_col.apply(pd.Series)

    df["mouse_id"] = expanded[0] + "_" + expanded[1]
    df["sensor_type"] = expanded[2]
    df["context"] = expanded[3]
    df["day"] = expanded[4] + "_" + expanded[5]

    df["Age (months)"] = df["Age (months)"].astype(int)

    def age_mapper(x: int) -> str:
        return "Young" if x <= 2 else "Old"

    df["Age (binary)"] = df["Age (months)"].map(age_mapper)

    # adding data based on the TDT notes.txt file
    records = []
    for index, row in df.iterrows():
        exp = tdt.TDTExperiment(join(metadata.ROOT_DATA_DIR, row["Folder Name"]))
        records.append(
            {
                "Folder Name": row["Folder Name"],
                "tdt_mouse_id": exp.subject_name,
                "exp_start": exp.experiment_start_stop()[0],
                "exp_stop": exp.experiment_start_stop()[1],
            }
        )

    tdt_df = pd.DataFrame(records)

    # merging the two dataframes
    merged_df = df.merge(tdt_df, on="Folder Name", how="left")

    merged_df.to_excel(
        metadata.CLEANUP_METADATA.replace(".xlsx", "_parsed.xlsx"), index=False
    )

    # filtering out good quality data
    good_df = merged_df[
        ((merged_df["Fiber Signal"] == "good") | (merged_df["Fiber Signal"] == "great"))
        & (merged_df["Motion Data"] != "discard")
    ]

    good_df.to_excel(
        metadata.CLEANUP_METADATA.replace(".xlsx", "_parsed_good.xlsx"), index=False
    )


if __name__ == "__main__":
    main()
