from __future__ import annotations
import gc
import json
import logging
import os
from pathlib import Path
from typing import Optional
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from google.cloud import storage
from google.oauth2 import service_account
from preprocess_memory_efficient import (
    preprocess_raw_input_memory_efficient,
    load_csvs_memory_efficient,
    preprocess_prospective_input,
)

# ─────────────────────────────── Logging ─────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────── Helpers ─────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_storage_client() -> storage.Client:
    key_info = dict(st.secrets["GCP_SERVICE_ACCOUNT_JSON"])
    creds = service_account.Credentials.from_service_account_info(key_info)
    project_id = key_info.get("project_id")
    return storage.Client(project=project_id, credentials=creds)


def _secret(name: str, default: str = "") -> str:
    """Safe helper to read a string secret, checking both top-level and nested tables."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
        svc = st.secrets.get("GCP_SERVICE_ACCOUNT_JSON", {})
        try:
            svc_dict = dict(svc) if svc is not None else {}
        except TypeError:
            svc_dict = {}
        if name in svc_dict:
            return str(svc_dict[name])
    except Exception:
        pass
    return default


@st.cache_data(show_spinner=False)
def load_valid_ids() -> set[int]:
    """Read SK_ID_CURR universe once from *valid_data.csv* (local or GCS)."""
    bucket = os.getenv("GCS_BUCKET") or _secret("GCS_BUCKET", "")
    blob = os.getenv("GCS_VALID_DATA") or _secret("GCS_VALID_DATA", "")

    if bucket and blob:
        tmp_path = Path("/tmp/valid_data.csv")
        client = get_storage_client()
        client.bucket(bucket).blob(blob).download_to_filename(tmp_path)
        ids = pd.read_csv(tmp_path, usecols=["SK_ID_CURR"])
        tmp_path.unlink(missing_ok=True)
    else:
        base = Path(__file__).resolve().parents[1]
        candidates = [
            base / "valid_data.csv",
            base / "notebooks_and_initial_tables/notebooks/valid_data.csv",
        ]
        csv_path = next((p for p in candidates if p.exists()), None)
        if csv_path is None:
            raise FileNotFoundError("valid_data.csv not found in expected locations")
        ids = pd.read_csv(csv_path, usecols=["SK_ID_CURR"])

    logger.info("Loaded %d valid IDs", len(ids))
    return set(ids["SK_ID_CURR"].tolist())


@st.cache_data(show_spinner=False)
def load_valid_df() -> pd.DataFrame:
    """
    Load the full valid_data.csv into a DataFrame (local or GCS).

    Used for random sampling of client IDs; cached so we only hit disk once.
    """
    bucket = os.getenv("GCS_BUCKET") or _secret("GCS_BUCKET", "")
    blob = os.getenv("GCS_VALID_DATA") or _secret("GCS_VALID_DATA", "")

    if bucket and blob:
        tmp_path = Path("/tmp/valid_data_full.csv")
        client = get_storage_client()
        client.bucket(bucket).blob(blob).download_to_filename(tmp_path)
        df = pd.read_csv(tmp_path)
        tmp_path.unlink(missing_ok=True)
    else:
        base = Path(__file__).resolve().parents[1]
        candidates = [
            base / "valid_data.csv",
            base / "notebooks_and_initial_tables/notebooks/valid_data.csv",
        ]
        csv_path = next((p for p in candidates if p.exists()), None)
        if csv_path is None:
            raise FileNotFoundError("valid_data.csv not found in expected locations")
        df = pd.read_csv(csv_path)

    return df


def _download_blob(bucket: str, blob: str, dest: str) -> None:
    """Download *blob* from *bucket* to *dest* with basic retries."""
    try:
        client = get_storage_client()
        client.bucket(bucket).blob(blob).download_to_filename(dest)
    except Exception as exc:
        raise RuntimeError(f"Failed to download {blob} from {bucket}: {exc}") from exc


# ────────────────────────── Model & data loaders ─────────────────────────────
@st.cache_data(show_spinner=False)
def get_template_client_id() -> int:
    """
    Return a stable reference SK_ID_CURR from the validation universe.

    This ID is used as a template for prospective applicants: we keep
    its external bureau / previous-loan aggregates, but overwrite
    application-level fields with user inputs.
    """
    ids = sorted(load_valid_ids())
    return ids[0]


@st.cache_resource(show_spinner=False)
def load_model():
    """Load LightGBM model from GCS or local file."""
    bucket = os.getenv("GCS_BUCKET") or _secret("GCS_BUCKET", "")
    blob = os.getenv("GCS_MODEL") or _secret("GCS_MODEL", "")

    if bucket and blob:
        tmp_path = "/tmp/best_lgbm_model.pkl"
        _download_blob(bucket, blob, tmp_path)
        model_ = joblib.load(tmp_path)
        Path(tmp_path).unlink(missing_ok=True)
        logger.info("Loaded model from GCS")
    else:
        base = Path(__file__).resolve().parents[1]
        candidates = [
            base / "best_lgbm_model.pkl",
            base / "notebooks_and_initial_tables/notebooks/best_lgbm_model.pkl",
            Path("best_lgbm_model.pkl"),
        ]
        model_path = next((p for p in candidates if p.exists()), None)
        if model_path is None:
            raise FileNotFoundError(
                "best_lgbm_model.pkl not found in expected locations"
            )
        model_ = joblib.load(model_path)
        logger.info("Loaded model from local disk: %s", model_path)

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


def _sample_client_ids(n_samples: int = 100) -> pd.DataFrame:
    """Return *n_samples* random IDs for demo / testing UI."""
    df = load_valid_df()[["SK_ID_CURR"]]
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
        base = Path(__file__).resolve().parents[1]
        candidates = [
            base / "valid_data.csv",
            base / "notebooks_and_initial_tables/notebooks/valid_data.csv",
        ]
        csv_path = next((p for p in candidates if p.exists()), None)
        if csv_path is None:
            raise FileNotFoundError("valid_data.csv not found in expected locations")
        valid_df = pd.read_csv(csv_path)

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
    "Choose mode:",
    ["Specific Client ID", "Random Sample", "Prospective Applicant"],
    horizontal=True,
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
elif mode == "Random Sample":
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
else:
    st.markdown(
        "Use this mode as an **approximate pre-screening tool**. "
        "The estimate is based on your self-reported information and "
        "typical external data; the actual production model also "
        "uses full credit-bureau and historical repayment information."
    )

    with st.form("prospective_form"):
        col1, col2 = st.columns(2)

        with col1:
            age_years = st.number_input(
                "Age (years)", min_value=18, max_value=80, value=35
            )
            years_employed = st.number_input(
                "Years in current employment", min_value=0.0, max_value=60.0, value=5.0
            )
            ext_score = st.slider(
                "External credit score (0 = poor, 1 = excellent)",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.01,
            )
            annual_income = st.number_input(
                "Annual income (dataset currency units)",
                min_value=0.0,
                value=30000.0,
                step=1000.0,
            )
            loan_amount = st.number_input(
                "Requested loan amount (dataset currency units)",
                min_value=0.0,
                value=200000.0,
                step=10000.0,
            )
            annuity = st.number_input(
                "Expected monthly instalment (annuity, dataset currency units)",
                min_value=0.0,
                value=8000.0,
                step=500.0,
            )

        with col2:
            gender = st.selectbox("Gender", ["F", "M"])
            family_status = st.selectbox(
                "Family status",
                [
                    "Single / not married",
                    "Married",
                    "Separated",
                    "Widow",
                    "Civil marriage",
                ],
            )
            education = st.selectbox(
                "Highest education",
                [
                    "Secondary / secondary special",
                    "Higher education",
                    "Incomplete higher",
                    "Lower secondary",
                    "Academic degree",
                ],
            )
            housing = st.selectbox(
                "Housing type",
                [
                    "House / apartment",
                    "Rented apartment",
                    "With parents",
                    "Municipal apartment",
                    "Office apartment",
                    "Co-op apartment",
                ],
            )
            owns_car = st.checkbox("Owns a car", value=False)
            owns_realty = st.checkbox("Owns real estate", value=False)

        submit_prospective = st.form_submit_button("💡 Estimate Default Risk")

    if submit_prospective:
        with st.spinner("Estimating risk …"):
            try:
                manual_data = {
                    "DAYS_BIRTH": int(-365.25 * age_years),
                    "DAYS_EMPLOYED": int(-365.25 * years_employed),
                    "EXT_SOURCE_1": ext_score,
                    "EXT_SOURCE_2": ext_score,
                    "EXT_SOURCE_3": ext_score,
                    "AMT_INCOME_TOTAL": annual_income,
                    "AMT_CREDIT": loan_amount,
                    "AMT_ANNUITY": annuity,
                    "CODE_GENDER": gender,
                    "NAME_FAMILY_STATUS": family_status,
                    "NAME_EDUCATION_TYPE": education,
                    "NAME_HOUSING_TYPE": housing,
                    "FLAG_OWN_CAR": "Y" if owns_car else "N",
                    "FLAG_OWN_REALTY": "Y" if owns_realty else "N",
                }

                X = preprocess_prospective_input(manual_data)
                model = load_model()
                X = X.reindex(columns=model.feature_name_, fill_value=np.nan).astype(
                    np.float32
                )
                if hasattr(model, "predict_proba"):
                    risk = float(model.predict_proba(X)[0, 1])
                else:
                    risk = float(model.predict(X)[0])
            except Exception as exc:
                st.error(f"Estimation failed: {exc}")
            else:
                st.success(f"Estimated default-risk probability: **{risk:.2%}**")
                if risk < 0.30:
                    st.info(
                        "🟢 Low estimated risk (based on limited self-reported data)."
                    )
                elif risk < 0.70:
                    st.warning(
                        "🟡 Medium estimated risk – additional information could change this."
                    )
                else:
                    st.error(
                        "🔴 High estimated risk – improving credit history or reducing debt may help."
                    )
