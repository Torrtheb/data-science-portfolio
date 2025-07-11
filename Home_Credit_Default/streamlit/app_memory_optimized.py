from __future__ import annotations
import gc
import logging
import os
from pathlib import Path
from typing import Optional
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from google.cloud import storage
from preprocess_memory_efficient import (
    preprocess_raw_input_memory_efficient,
    load_csvs_memory_efficient,
)

# ─────────────────────────────── Logging ─────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────── Helpers ─────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_valid_ids() -> set[int]:
    """Read SK_ID_CURR universe once from *valid_data.csv* (local or GCS)."""
    bucket = os.getenv("GCS_BUCKET")
    blob = os.getenv("GCS_VALID_DATA")

    if bucket and blob:
        tmp_path = Path("/tmp/valid_data.csv")
        storage.Client().bucket(bucket).blob(blob).download_to_filename(tmp_path)
        ids = pd.read_csv(tmp_path, usecols=["SK_ID_CURR"])
        tmp_path.unlink(missing_ok=True)
    else:
        ids = pd.read_csv("valid_data.csv", usecols=["SK_ID_CURR"])

    logger.info("Loaded %d valid IDs", len(ids))
    return set(ids["SK_ID_CURR"].tolist())


def _download_blob(bucket: str, blob: str, dest: str) -> None:
    """Download *blob* from *bucket* to *dest* with basic retries."""
    try:
        storage.Client().bucket(bucket).blob(blob).download_to_filename(dest)
    except Exception as exc:
        raise RuntimeError(f"Failed to download {blob} from {bucket}: {exc}") from exc


# ────────────────────────── Model & data loaders ─────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    """Load LightGBM model from GCS or local file."""
    bucket = os.getenv("GCS_BUCKET")
    blob = os.getenv("GCS_MODEL")

    if bucket and blob:
        tmp_path = "/tmp/best_lgbm_model.pkl"
        _download_blob(bucket, blob, tmp_path)
        model_ = joblib.load(tmp_path)
        Path(tmp_path).unlink(missing_ok=True)
        logger.info("Loaded model from GCS")
    else:
        model_ = joblib.load("best_lgbm_model.pkl")
        logger.info("Loaded model from local disk")

    return model_


@st.cache_resource(show_spinner=False)
def initialize_data_sources() -> bool:
    """Fire the memory-efficient parquet readers once."""
    try:
        load_csvs_memory_efficient()
        return True
    except Exception as exc:
        logger.error("Failed to initialise external data: %s", exc)
        return False


@st.cache_data(show_spinner=False, ttl=3600)
def _sample_client_ids(n_samples: int = 100) -> pd.DataFrame:
    """Return *n_samples* random IDs for demo / testing UI."""
    bucket = os.getenv("GCS_BUCKET")
    blob = os.getenv("GCS_VALID_DATA")

    if bucket and blob:
        tmp_path = "/tmp/valid_data_sample.csv"
        _download_blob(bucket, blob, tmp_path)
        df = pd.read_csv(tmp_path, usecols=["SK_ID_CURR"])
        Path(tmp_path).unlink(missing_ok=True)
    else:
        df = pd.read_csv("valid_data.csv", usecols=["SK_ID_CURR"])

    return df.sample(min(n_samples, len(df))).reset_index(drop=True)


# ────────────────────────────── Core logic ───────────────────────────────────
def predict_default_risk(
    client_id: int,
    manual_data: Optional[dict] | None = None,
) -> float:
    """Return probability of default for *client_id* using minimal memory."""

    valid_ids = load_valid_ids()
    if client_id not in valid_ids:
        raise ValueError(
            f"Client {client_id} is not present in validation data – probability not calculated."
        )

    bucket = os.getenv("GCS_BUCKET")
    blob = os.getenv("GCS_VALID_DATA")

    if bucket and blob:
        tmp_path = Path("/tmp/valid_data_full.csv")
        _download_blob(bucket, blob, str(tmp_path))
        valid_df = pd.read_csv(tmp_path)
        tmp_path.unlink(missing_ok=True)
    else:
        valid_df = pd.read_csv("valid_data.csv")

    client_data = valid_df[valid_df["SK_ID_CURR"] == client_id]
    if client_data.empty:
        raise ValueError(f"Client {client_id} data not found in valid_data.csv")

    raw = client_data.iloc[0].to_dict()

    if manual_data:
        raw.update(manual_data)

    X = preprocess_raw_input_memory_efficient(raw)

    model = load_model()
    X = X.reindex(columns=model.feature_name_, fill_value=np.nan).astype(np.float32)

    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(X)[0, 1])
    else:
        prob = float(model.predict(X)[0])

    del X
    gc.collect()
    return prob


# ────────────────────────────── Streamlit UI ─────────────────────────────────
st.set_page_config(
    page_title="Home Credit Default-Risk Predictor",
    layout="wide",
    initial_sidebar_state="auto",
)

st.title("🎯 Home Credit Default-Risk Prediction")

with st.sidebar.expander("⚙️ Settings – Prevent cold starts", expanded=False):
    keep_warm = st.toggle(
        "Auto-refresh every 14 minutes (keeps server warm)", value=False
    )
    if keep_warm:
        REFRESH_MS = 14 * 60 * 1000
        try:
            from streamlit_autorefresh import st_autorefresh

            st_autorefresh(interval=REFRESH_MS, key="keepalive_refresh")
            st.caption(
                "⏳ Auto-refresh enabled – this tab will ping the app every 14 minutes."
            )
        except ModuleNotFoundError:
            st.warning("Install *streamlit-autorefresh* to enable background pings.")

with st.spinner("Initialising resources …"):
    if not initialize_data_sources():
        st.error("Failed to initialise external parquet sources – see logs.")
        st.stop()

st.subheader("Client Selection")

mode = st.radio(
    "Choose mode:", ["Specific Client ID", "Random Sample"], horizontal=True
)

if mode == "Specific Client ID":
    with st.form("client_form"):
        client_id_inp = st.number_input(
            "Enter Client ID:", min_value=1, value=100001, step=1
        )
        predict_clicked = st.form_submit_button("🔍 Predict Risk")

    if predict_clicked:
        with st.spinner("Scoring borrower …"):
            try:
                risk = predict_default_risk(int(client_id_inp))
            except ValueError as err:
                st.info(str(err))
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
            else:
                st.success(f"Default-Risk Probability: **{risk:.2%}**")
                if risk < 0.30:
                    st.info("🟢 Low Risk")
                elif risk < 0.70:
                    st.warning("🟡 Medium Risk")
                else:
                    st.error("🔴 High Risk")
else:
    if st.button("🎲 Predict Random Client"):
        with st.spinner("Sampling & scoring …"):
            try:
                random_id = int(_sample_client_ids(1).iloc[0]["SK_ID_CURR"])
                risk = predict_default_risk(random_id)
            except Exception as exc:
                st.error(f"Random prediction failed: {exc}")
            else:
                st.success(f"Client ID: **{random_id}** – Probability: **{risk:.2%}**")
                if risk < 0.30:
                    st.info("🟢 Low Risk")
                elif risk < 0.70:
                    st.warning("🟡 Medium Risk")
                else:
                    st.error("🔴 High Risk")
