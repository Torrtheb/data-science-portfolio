from pathlib import Path
from typing import Union
import pandas as pd


def convert_csv_to_parquet(
    csv_path: Union[str, Path],
    parquet_path: Union[str, Path],
) -> None:
    """
    Convert a CSV file to Parquet format.

    Parameters
    ----------
    csv_path : str | pathlib.Path
        Path to the source CSV file.
    parquet_path : str | pathlib.Path
        Destination path for the Parquet file
        (the “.parquet” extension is recommended but not required).
    """
    csv_path = Path(csv_path)
    parquet_path = Path(parquet_path)

    print(f"Reading {csv_path} ...")
    df = pd.read_csv(csv_path)

    print(f"Writing {parquet_path} ...")
    df.to_parquet(parquet_path, index=False)

    print("Conversion complete.")
