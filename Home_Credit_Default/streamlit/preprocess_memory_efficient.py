from __future__ import annotations
import json
import logging
import os
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import dask.dataframe as dd
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from google.cloud import storage
from google.oauth2 import service_account

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deploy.feature_config import (
    OCCUPATION_TYPE_GROUP_MAP,
    INCOME_TYPE_GROUP_MAP,
    ORGANIZATION_TYPE_GROUP_MAP,
    HOUSING_TYPE_GROUP_MAP,
    SELECTED_FEATURES,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@st.cache_resource(show_spinner=False)
def get_storage_client() -> storage.Client:
    """Create a GCS client using the service account info in Streamlit secrets."""
    # Expect `GCP_SERVICE_ACCOUNT_JSON` to be a secrets *table*, not a raw JSON string.
    key_info = dict(st.secrets["GCP_SERVICE_ACCOUNT_JSON"])
    creds = service_account.Credentials.from_service_account_info(key_info)
    project_id = key_info.get("project_id")
    return storage.Client(project=project_id, credentials=creds)

# ────────────────────────────── Lazy global datasets ─────────────────────────

bureau_final: dd.DataFrame | None = None
p_final_merged: dd.DataFrame | None = None
selected_features = SELECTED_FEATURES

# ────────────────────────────── Parquet loaders ──────────────────────────────


def load_csvs_memory_efficient() -> None:
    """Populate *bureau_final* and *p_final_merged* global variables lazily."""
    global bureau_final, p_final_merged

    if bureau_final is None:
        logger.info("📦 Loading bureau_final.parquet (lazy)…")
        bucket = os.getenv("GCS_BUCKET") or st.secrets.get("GCS_BUCKET", "")
        bureau_blob = os.getenv("GCS_BUREAU_PARQUET") or st.secrets.get(
            "GCS_BUREAU_PARQUET", ""
        )

        if bucket and bureau_blob:
            tmp_path = Path("/tmp/bureau_final.parquet")
            client = get_storage_client()
            client.bucket(bucket).blob(bureau_blob).download_to_filename(tmp_path)
            bureau_df = dd.read_parquet(tmp_path, engine="pyarrow")
            logger.info("✅ bureau_final loaded from GCS: %s", bureau_blob)
        else:
            candidates = [
                PROJECT_ROOT / "deploy" / "bureau_final.parquet",
                Path(__file__).resolve().parent / "bureau_final.parquet",
            ]
            bureau_path = next((p for p in candidates if p.exists()), None)
            if bureau_path is None:
                raise FileNotFoundError(
                    "bureau_final.parquet not found in expected locations"
                )
            bureau_df = dd.read_parquet(bureau_path, engine="pyarrow")
            logger.info("✅ bureau_final loaded from local path: %s", bureau_path)

        bureau_final = bureau_df
        logger.info("✅ bureau_final: %d columns", len(bureau_final.columns))

    if p_final_merged is None:
        logger.info("📦 Loading p_final_merged.parquet (lazy)…")
        bucket = os.getenv("GCS_BUCKET") or st.secrets.get("GCS_BUCKET", "")
        prev_blob = os.getenv("GCS_PREV_PARQUET") or st.secrets.get(
            "GCS_PREV_PARQUET", ""
        )

        if bucket and prev_blob:
            tmp_path = Path("/tmp/p_final_merged.parquet")
            client = get_storage_client()
            client.bucket(bucket).blob(prev_blob).download_to_filename(tmp_path)
            prev_df = dd.read_parquet(tmp_path, engine="pyarrow")
            logger.info("✅ p_final_merged loaded from GCS: %s", prev_blob)
        else:
            candidates = [
                PROJECT_ROOT / "deploy" / "p_final_merged.parquet",
                Path(__file__).resolve().parent / "p_final_merged.parquet",
            ]
            prev_path = next((p for p in candidates if p.exists()), None)
            if prev_path is None:
                raise FileNotFoundError(
                    "p_final_merged.parquet not found in expected locations"
                )
            prev_df = dd.read_parquet(prev_path, engine="pyarrow")
            logger.info("✅ p_final_merged loaded from local path: %s", prev_path)

        p_final_merged = prev_df
        logger.info("✅ p_final_merged: %d columns", len(p_final_merged.columns))


# ────────────────────────────── External feature join ────────────────────────


def _filter_and_fetch(ddf: dd.DataFrame, sk_id: int) -> pd.DataFrame:
    """Return a *single‑row* pandas DataFrame for one borrower."""
    cols = [c for c in ddf.columns if c in selected_features or c == "SK_ID_CURR"]
    row = ddf[ddf["SK_ID_CURR"] == sk_id][cols].head(1, compute=True)
    return row


def get_features_for_client(sk_id_curr: int) -> Dict[str, Any]:
    if bureau_final is None or p_final_merged is None:
        raise RuntimeError(
            "Parquet sources not initialised – call load_csvs_memory_efficient() first"
        )

    bureau_row = _filter_and_fetch(bureau_final, sk_id_curr)
    prev_row = _filter_and_fetch(p_final_merged, sk_id_curr)

    if bureau_row.empty or prev_row.empty:
        raise ValueError(f"Client {sk_id_curr} not found in external datasets")

    features: Dict[str, Any] = (
        bureau_row.drop(columns=["SK_ID_CURR"]).iloc[0].to_dict()
        | prev_row.drop(columns=["SK_ID_CURR"]).iloc[0].to_dict()
    )
    return features


# ────────────────────────────── Pre‑processing steps ─────────────────────────


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"[^\w_]", "_", col) for col in df.columns]
    return df


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace("XNA", np.nan)
    pd.set_option("future.no_silent_downcasting", True)
    df = df.infer_objects(copy=False)

    if "DAYS_EMPLOYED" in df.columns:
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)

    if {"OWN_CAR_AGE", "FLAG_OWN_CAR"}.issubset(df.columns):
        df.loc[df["FLAG_OWN_CAR"].eq("N") & df["OWN_CAR_AGE"].isna(), "OWN_CAR_AGE"] = 0

    if "AMT_INCOME_TOTAL" in df.columns:
        upper = df["AMT_INCOME_TOTAL"].quantile(0.99)
        df["AMT_INCOME_TOTAL"] = df["AMT_INCOME_TOTAL"].clip(upper=upper)

    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # -------- grouping maps (identical to reference pipeline) --------
    group_map_1 = OCCUPATION_TYPE_GROUP_MAP
    if "OCCUPATION_TYPE" in df.columns:
        df["OCCUPATION_TYPE_GROUPED"] = df["OCCUPATION_TYPE"].map(group_map_1)

    group_map_2 = INCOME_TYPE_GROUP_MAP
    if "NAME_INCOME_TYPE" in df.columns:
        df["NAME_INCOME_TYPE_GROUPED"] = df["NAME_INCOME_TYPE"].map(group_map_2)

    group_map_3 = ORGANIZATION_TYPE_GROUP_MAP
    if "ORGANIZATION_TYPE" in df.columns:
        df["ORGANIZATION_TYPE_GROUPED"] = df["ORGANIZATION_TYPE"].map(group_map_3)

    group_map_4 = HOUSING_TYPE_GROUP_MAP
    if "NAME_HOUSING_TYPE" in df.columns:
        df["NAME_HOUSING_TYPE_GROUPED"] = df["NAME_HOUSING_TYPE"].map(group_map_4)

    for col in [
        "OCCUPATION_TYPE",
        "NAME_INCOME_TYPE",
        "ORGANIZATION_TYPE",
        "NAME_HOUSING_TYPE",
    ]:
        if col in df.columns:
            df = df.drop(columns=col)

    if {"AMT_CREDIT", "AMT_ANNUITY"}.issubset(df.columns):
        df["credit_annuity_ratio"] = (
            (df["AMT_CREDIT"] / df["AMT_ANNUITY"].replace(0, np.nan))
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )

    if {"DAYS_BIRTH", "EXT_SOURCE_1"}.issubset(df.columns):
        df["age_score_ratio"] = (
            (df["DAYS_BIRTH"] / df["EXT_SOURCE_1"].replace(0, np.nan))
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )

    if {"EXT_SOURCE_2", "AMT_CREDIT"}.issubset(df.columns):
        df["score_credit_ratio"] = (
            (df["EXT_SOURCE_2"] / df["AMT_CREDIT"].replace(0, np.nan))
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )

    if {"AMT_INCOME_TOTAL", "AMT_GOODS_PRICE"}.issubset(df.columns):
        df["income_goods_ratio"] = (
            (df["AMT_INCOME_TOTAL"] / df["AMT_GOODS_PRICE"].replace(0, np.nan))
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )

    doc_flags = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]
    if doc_flags:
        df["SUM_FLAG_DOCUMENT"] = df[doc_flags].sum(axis=1)

    return df


def encode_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        df[col] = df[col].fillna("missing").astype("category")

    df = pd.get_dummies(df, drop_first=False)
    df = clean_column_names(df)

    bool_cols = df.select_dtypes(include="bool").columns
    if not bool_cols.empty:
        df[bool_cols] = df[bool_cols].astype(int)

    df = df.loc[:, ~df.columns.duplicated()]
    return df


# ────────────────────────────── Master preprocess ────────────────────────────


def preprocess_raw_input_memory_efficient(input_data: Dict[str, Any]) -> pd.DataFrame:
    """Full pipeline for a **single borrower** – returns 375‑col DataFrame."""
    sk_id = input_data.get("SK_ID_CURR")
    if sk_id is None:
        raise ValueError("Missing SK_ID_CURR in input data")

    df = pd.DataFrame([input_data])
    df = basic_cleaning(df)
    df = feature_engineering(df)
    ext_features = get_features_for_client(sk_id)
    ext_df = pd.DataFrame(ext_features, index=df.index)

    df = df.merge(ext_df, left_index=True, right_index=True, how="left")

    df = encode_and_clean(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    sel = selected_features
    missing = [c for c in sel if c not in df.columns]
    if missing:
        add_df = pd.DataFrame(0, index=df.index, columns=missing)
        df = pd.concat([df, add_df], axis=1)
    df = df[sel]
    logger.info("Client %s – returning %d features", sk_id, df.shape[1])
    return df


def preprocess_prospective_input(input_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Preprocess a *prospective applicant* who is not yet in the bureau tables.

    This version:
    - Uses only self-reported application features (no SK_ID_CURR required).
    - Applies the same basic cleaning, feature engineering, encoding and
      column alignment as the main pipeline.
    - Sets all missing model features (including bureau/previous aggregates)
      to 0, which effectively represents "no external credit history".
    """
    df = pd.DataFrame([input_data])
    df = basic_cleaning(df)
    df = feature_engineering(df)
    df = encode_and_clean(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    sel = selected_features
    missing = [c for c in sel if c not in df.columns]
    if missing:
        add_df = pd.DataFrame(0, index=df.index, columns=missing)
        df = pd.concat([df, add_df], axis=1)
    df = df[sel]
    logger.info("Prospective applicant – returning %d features", df.shape[1])
    return df
