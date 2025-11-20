import logging
import re
import time
from pathlib import Path
from typing import Any

import dask.dataframe as dd
import joblib
import numpy as np
import pandas as pd
from dask.dataframe import DataFrame

from feature_config import (
    HOUSING_TYPE_GROUP_MAP,
    INCOME_TYPE_GROUP_MAP,
    OCCUPATION_TYPE_GROUP_MAP,
    ORGANIZATION_TYPE_GROUP_MAP,
    SELECTED_FEATURES,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bureau_final = None
p_final_merged = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_csvs() -> None:
    """
    Lazily load two pre-computed Parquet datasets into global variables.

    The function populates the module-level variables
    bureau_final and p_final_merged the first time it is called.
    Subsequent calls are no-ops, allowing the DataFrames to act as an
    in-memory cache across the application.

    Raises
    ------
    FileNotFoundError
        If either Parquet file cannot be located.
    OSError
        For low-level I/O errors during reading.
    Any other exception emitted by :pyfunc:dask.dataframe.read_parquet
        is propagated upward after being logged.
    """
    global bureau_final, p_final_merged
    if bureau_final is None or p_final_merged is None:
        try:
            logger.info("📦 Loading bureau_final.parquet with Dask...")
            bureau_final = dd.read_parquet("bureau_final.parquet")
            logger.info("✅ bureau_final.parquet loaded lazily.")
        except Exception as e:
            logger.error(f"Failed to load bureau_final.parquet: {e}")
            raise

        try:
            logger.info("📦 Loading p_final_merged.parquet with Dask...")
            p_final_merged = dd.read_parquet("p_final_merged.parquet")
            logger.info("✅ p_final_merged.parquet loaded lazily.")
        except Exception as e:
            logger.error(f"Failed to load p_final_merged.parquet: {e}")
            raise


def load_model() -> Any:
    """
    Lazily load the trained LightGBM model (best_lgbm_model.pkl)
    and cache it on the function object itself.

    Looks for the model in several common locations so that it works
    both inside the Docker image (/app) and when run locally.
    """
    if not hasattr(load_model, "model"):
        candidates = [
            PROJECT_ROOT / "best_lgbm_model.pkl",
            PROJECT_ROOT / "notebooks_and_initial_tables/notebooks/best_lgbm_model.pkl",
            Path("best_lgbm_model.pkl"),
        ]
        model_path = next((p for p in candidates if p.exists()), None)
        if model_path is None:
            raise FileNotFoundError(
                "best_lgbm_model.pkl not found in expected locations"
            )
        load_model.model = joblib.load(model_path)
    return load_model.model


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean DataFrame column names by replacing any character
    that is not a letter, number, or underscore with an underscore.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with original column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with cleaned column names.
    """
    df = df.copy()
    df.columns = [re.sub(r"[^\w_]", "_", col) for col in df.columns]
    return df


def basic_cleaning(df: DataFrame) -> DataFrame:
    """
    Perform basic preprocessing and cleaning on a Home Credit-style DataFrame.

    Operations performed:
    ---------------------
    1. Replace all occurrences of 'XNA' (string) with NaN.
    2. Enable pandas' future behavior for silent downcasting (opt-in).
    3. Use infer_objects(copy=False) to convert columns to inferred types.
    4. Replace placeholder 365243 in DAYS_EMPLOYED with NaN (likely sentinel).
    5. Set OWN_CAR_AGE to 0 if missing and applicant does not own a car.
    6. Clip AMT_INCOME_TOTAL to its 99th percentile to reduce outliers.

    Parameters
    ----------
    df : pd.DataFrame
        The raw input data containing application and demographic features.

    Returns
    -------
    pd.DataFrame
        A cleaned version of the input DataFrame, ready for further processing.
    """
    df = df.replace("XNA", np.nan)
    pd.set_option("future.no_silent_downcasting", True)
    df = df.infer_objects(copy=False)
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
    df.loc[
        (df["OWN_CAR_AGE"].isnull()) & (df["FLAG_OWN_CAR"] == "N"), "OWN_CAR_AGE"
    ] = 0
    upper_limit = df["AMT_INCOME_TOTAL"].quantile(0.99)
    df["AMT_INCOME_TOTAL"] = df["AMT_INCOME_TOTAL"].clip(upper=upper_limit)
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering on the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame to perform feature engineering on.

    Returns:
        pd.DataFrame: DataFrame with engineered features.
    """
    df = df.copy()
    df["OCCUPATION_TYPE_GROUPED"] = df["OCCUPATION_TYPE"].map(OCCUPATION_TYPE_GROUP_MAP)
    df["NAME_INCOME_TYPE_GROUPED"] = df["NAME_INCOME_TYPE"].map(INCOME_TYPE_GROUP_MAP)
    df["ORGANIZATION_TYPE_GROUPED"] = df["ORGANIZATION_TYPE"].map(
        ORGANIZATION_TYPE_GROUP_MAP
    )
    df["NAME_HOUSING_TYPE_GROUPED"] = df["NAME_HOUSING_TYPE"].map(
        HOUSING_TYPE_GROUP_MAP
    )
    df = df.drop(
        columns=[
            "OCCUPATION_TYPE",
            "NAME_INCOME_TYPE",
            "ORGANIZATION_TYPE",
            "NAME_HOUSING_TYPE",
        ]
    )

    df["credit_annuity_ratio"] = (
        (df["AMT_CREDIT"] / df["AMT_ANNUITY"].replace(0, np.nan))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )
    df["age_score_ratio"] = (
        (df["DAYS_BIRTH"] / df["EXT_SOURCE_1"].replace(0, np.nan))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )
    df["score_credit_ratio"] = (
        (df["EXT_SOURCE_2"] / df["AMT_CREDIT"].replace(0, np.nan))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )
    df["income_goods_ratio"] = (
        (df["AMT_INCOME_TOTAL"] / df["AMT_GOODS_PRICE"].replace(0, np.nan))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    document_flags = [col for col in df.columns if col.startswith("FLAG_DOCUMENT_")]
    df["SUM_FLAG_DOCUMENT"] = df[document_flags].sum(axis=1)

    return df


def encode_and_clean(df: DataFrame) -> DataFrame:
    """
    Encode categorical variables, clean column names, and prepare a DataFrame
    for modeling.

    Steps performed:
    ----------------
    1. Fill missing values in object-type (string) columns with "missing".
    2. Convert object columns to pandas category dtype.
    3. Apply one-hot encoding via pd.get_dummies, keeping all categories.
    4. Sanitize column names via clean_column_names() (assumed to exist).
    5. Convert boolean columns to integers (0/1).
    6. Remove duplicated columns, if any.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame with raw features including categorical variables.

    Returns
    -------
    pd.DataFrame
        A fully numeric and cleaned DataFrame ready for use in ML models.
    """
    cat_features = df.select_dtypes(include="object").columns.tolist()
    for col in cat_features:
        df[col] = df[col].fillna("missing")
        df[col] = df[col].astype("category")

    df = pd.get_dummies(df, drop_first=False)
    df = clean_column_names(df)

    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    df = df.loc[:, ~df.columns.duplicated()]

    return df


def preprocess_raw_input(input_data: dict) -> pd.DataFrame:
    """
    Preprocess a single applicant dictionary for model prediction.

    This function:
    - Validates input and checks for SK_ID_CURR
    - Applies basic and engineered feature transformations
    - Joins external aggregated features from bureau_final and p_final_merged
    - Applies encoding and cleaning to ensure consistency with training
    - Ensures all selected_features are present in final DataFrame

    Parameters
    ----------
    input_data : dict
        A dictionary representing a single applicant's input data.
        Must include 'SK_ID_CURR' to allow joining with external data.

    Returns
    -------
    pd.DataFrame
        A single-row DataFrame containing all preprocessed and encoded features,
        ready for input into a predictive model.

    Raises
    ------
    ValueError
        If 'SK_ID_CURR' is missing or not found in external datasets.
    Exception
        If an error occurs during external feature merging or Dask computation.
    """
    sk_id_curr = input_data.get("SK_ID_CURR")
    if sk_id_curr is None:
        raise ValueError("Missing SK_ID_CURR in input data.")

    df = pd.DataFrame([input_data])
    df = basic_cleaning(df)
    df = feature_engineering(df)

    if bureau_final is None or p_final_merged is None:
        raise ValueError(
            "External features not loaded. Call load_csvs() before prediction."
        )

    try:
        t0 = time.time()
        bureau_cols = [
            col
            for col in bureau_final.columns
            if col in SELECTED_FEATURES or col == "SK_ID_CURR"
        ]
        prev_cols = [
            col
            for col in p_final_merged.columns
            if col in SELECTED_FEATURES or col == "SK_ID_CURR"
        ]

        bureau_row_dd = bureau_final[bureau_final["SK_ID_CURR"] == sk_id_curr][
            bureau_cols
        ]
        prev_row_dd = p_final_merged[p_final_merged["SK_ID_CURR"] == sk_id_curr][
            prev_cols
        ]
        logger.info(
            f"bureau_row_dd columns: {bureau_row_dd.columns}, npartitions: {bureau_row_dd.npartitions}"
        )
        logger.info(
            f"prev_row_dd columns: {prev_row_dd.columns}, npartitions: {prev_row_dd.npartitions}"
        )

        bureau_row = bureau_row_dd.head(1, compute=True)
        prev_row = prev_row_dd.head(1, compute=True)

        logger.info(
            f"Filtering and loading external features took {time.time() - t0:.2f} seconds."
        )
        logger.info(
            f"bureau_row shape: {bureau_row.shape}, prev_row shape: {prev_row.shape}"
        )
    except Exception as e:
        logger.error(f"Exception during Dask filtering or compute: {e}")
        raise

    if len(bureau_row.index) == 0:
        logger.error(f"SK_ID_CURR {sk_id_curr} not found in bureau_final.")
        raise ValueError(f"SK_ID_CURR {sk_id_curr} not found in bureau_final.")
    if len(prev_row.index) == 0:
        logger.error(f"SK_ID_CURR {sk_id_curr} not found in p_final_merged.")
        raise ValueError(f"SK_ID_CURR {sk_id_curr} not found in p_final_merged.")

    df = df.merge(
        bureau_row.drop(columns=["SK_ID_CURR"]),
        left_index=True,
        right_index=True,
        how="left",
    )
    df = df.merge(
        prev_row.drop(columns=["SK_ID_CURR"]),
        left_index=True,
        right_index=True,
        how="left",
    )

    df = encode_and_clean(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    for col in SELECTED_FEATURES:
        if col not in df.columns:
            df[col] = 0

    df = df[SELECTED_FEATURES]

    return df
