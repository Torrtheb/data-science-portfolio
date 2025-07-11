from __future__ import annotations
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict
import dask.dataframe as dd
import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ────────────────────────────── Lazy global datasets ─────────────────────────

bureau_final: dd.DataFrame | None = None  
p_final_merged: dd.DataFrame | None = None

# ────────────────────────────── Helper: model columns ────────────────────────

@lru_cache(maxsize=1)
def get_selected_features() -> list[str]:
    """Read `feature_name_` from *best_lgbm_model.pkl* once and cache it."""
    model_path = Path("best_lgbm_model.pkl")
    if not model_path.exists():
        raise FileNotFoundError(
            "best_lgbm_model.pkl not found – cannot derive selected_features")
    model = joblib.load(model_path)
    names = getattr(model, "feature_name_", None) or getattr(model, "feature_names", None)
    if not names:
        raise AttributeError("Model file has no feature name attribute")
    return list(names)

selected_features = get_selected_features()

# ────────────────────────────── Parquet loaders ──────────────────────────────

def load_csvs_memory_efficient() -> None:
    """Populate *bureau_final* and *p_final_merged* global variables lazily."""
    global bureau_final, p_final_merged

    if bureau_final is None:
        logger.info("📦 Loading bureau_final.parquet (lazy)…")
        bureau_final = dd.read_parquet("bureau_final.parquet", engine="pyarrow")
        logger.info("✅ bureau_final: %d columns", len(bureau_final.columns))

    if p_final_merged is None:
        logger.info("📦 Loading p_final_merged.parquet (lazy)…")
        p_final_merged = dd.read_parquet("p_final_merged.parquet", engine="pyarrow")
        logger.info("✅ p_final_merged: %d columns", len(p_final_merged.columns))

# ────────────────────────────── External feature join ────────────────────────

def _filter_and_fetch(ddf: dd.DataFrame, sk_id: int) -> pd.DataFrame:
    """Return a *single‑row* pandas DataFrame for one borrower."""
    cols = [c for c in ddf.columns if c in selected_features or c == "SK_ID_CURR"]
    row = (
        ddf[ddf["SK_ID_CURR"] == sk_id][cols]
        .head(1, compute=True)  
    )
    return row


def get_features_for_client(sk_id_curr: int) -> Dict[str, Any]:
    if bureau_final is None or p_final_merged is None:
        raise RuntimeError("Parquet sources not initialised – call load_csvs_memory_efficient() first")

    bureau_row = _filter_and_fetch(bureau_final, sk_id_curr)
    prev_row = _filter_and_fetch(p_final_merged, sk_id_curr)

    if bureau_row.empty or prev_row.empty:
        raise ValueError(f"Client {sk_id_curr} not found in external datasets")

    features: Dict[str, Any] = (
        bureau_row.drop(columns=["SK_ID_CURR"]).iloc[0].to_dict() |
        prev_row.drop(columns=["SK_ID_CURR"]).iloc[0].to_dict()
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
    group_map_1 = {
        "Accountants": "Accountants",
        "Cleaning staff": "Service",
        "Cooking staff": "Service",
        "Core staff": "Core staff",
        "Drivers": "Drivers",
        "HR staff": "Other",
        "High skill tech staff": "Tech",
        "IT staff": "Tech",
        "Laborers": "Laborers",
        "Low-skil Laborers": "Laborers",
        "Managers": "Managers",
        "Medicine staff": "Medicine",
        "Private service staff": "Other",
        "Realty agents": "Other",
        "Sales staff": "Sales",
        "Secretaries": "Other",
        "Security staff": "Other",
        "Waiters/barmen staff": "Service",
    }
    if "OCCUPATION_TYPE" in df.columns:
        df["OCCUPATION_TYPE_GROUPED"] = df["OCCUPATION_TYPE"].map(group_map_1)

    group_map_2 = {
        "Businessman": "Other",
        "Commercial associate": "Commercial associate",
        "Pensioner": "Pensioner",
        "State servant": "State servant",
        "Student": "Other",
        "Unemployed": "Other",
        "Working": "Working",
    }
    if "NAME_INCOME_TYPE" in df.columns:
        df["NAME_INCOME_TYPE_GROUPED"] = df["NAME_INCOME_TYPE"].map(group_map_2)

    group_map_3 = {
        "Business Entity Type 3": "Business",
        "Business Entity Type 2": "Business",
        "Business Entity Type 1 ": "Business",
        "XNA": "Unknown", 
        "Self-employed": "Self-employed",         
        "Other": "Other",                  
        "Medicine": "Public Sector",                
        "Government": "Public Sector",                
        "School": "Public Sector",  
        "Kindergarten": "Public Sector",
        "Security Ministries": "Public Sector",                  
        "Housing": "Public Sector",
        "Military": "Public Sector",
        "Police": "Public Sector",
        "Postal": "Public Sector",
        "University": "Public Sector",
        "Emergency": "Public Sector",
        "Trade: type 7": "Trade",
        "Trade: type 6": "Trade",
        "Trade: type 5": "Trade",
        "Trade: type 4": "Trade",
        "Trade: type 3": "Trade",
        "Trade: type 2": "Trade", 
        "Trade: type 1": "Trade", 
        "Electricity ": "Trade",        
        "Construction": "Trade",              
        "Transport: type 4": "Transport",
        "Transport: type 3": "Transport",
        "Transport: type 2": "Transport",
        "Transport: type 1": "Transport",
        "Industry: type 13": "Industry",
        "Industry: type 12 ": "Industry",
        "Industry: type 10": "Industry",
        "Industry: type 9": "Industry",
        "Industry: type 8": "Industry",
        "Industry: type 7": "Industry", 
        "Industry: type 6": "Industry",
        "Industry: type 5": "Industry",
        "Industry: type 4 ": "Industry",       
        "Industry: type 3": "Industry", 
        "Industry: type 2 ": "Industry",
        "Industry: type 1 ": "Industry",      
        "Industry: type 11": "Industry", 
        "Agriculture": "Industry",
        "Services ": "Service",  
        "Hotel": "Service",
        "Restaurant": "Service",
        "Cleaning": "Service",
        "Realtor": "Service",
        "Legal Services": "Service",
        "Advertising": "Other",   
        "Religion": "Other",
        "Culture": "Other",
        "Bank": "Finance",
        "Insurance": "Finance",
        "Telecom": "Other",
        "Mobile": "Other"
    }
    if "ORGANIZATION_TYPE" in df.columns:
        df["ORGANIZATION_TYPE_GROUPED"] = df["ORGANIZATION_TYPE"].map(group_map_3)

    group_map_4 = {
        "House / apartment": "House / apartment",
        "Municipal apartment": "Other apartment",
        "Office apartment": "Other apartment",
        "Rented apartment": "Other apartment",
        "With parents": "With parents",
        "Co-op apartment": "Other apartment",
    }
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
            df["AMT_CREDIT"] / df["AMT_ANNUITY"].replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).fillna(0)

    if {"DAYS_BIRTH", "EXT_SOURCE_1"}.issubset(df.columns):
        df["age_score_ratio"] = (
            df["DAYS_BIRTH"] / df["EXT_SOURCE_1"].replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).fillna(0)

    if {"EXT_SOURCE_2", "AMT_CREDIT"}.issubset(df.columns):
        df["score_credit_ratio"] = (
            df["EXT_SOURCE_2"] / df["AMT_CREDIT"].replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).fillna(0)

    if {"AMT_INCOME_TOTAL", "AMT_GOODS_PRICE"}.issubset(df.columns):
        df["income_goods_ratio"] = (
            df["AMT_INCOME_TOTAL"] / df["AMT_GOODS_PRICE"].replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).fillna(0)

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
        df[missing] = 0
    df = df[sel]
    logger.info("Client %s – returning %d features", sk_id, df.shape[1])
    return df
