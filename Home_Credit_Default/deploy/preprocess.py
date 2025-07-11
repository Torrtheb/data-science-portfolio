import pandas as pd
import numpy as np
import logging
import re
import pyarrow.parquet as pq
import joblib
import dask.dataframe as dd
import time
from typing import Any, Dict
from dask.dataframe import DataFrame

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bureau_final = None
p_final_merged = None

selected_features = [
    "EXT_SOURCE_3",
    "EXT_SOURCE_2",
    "EXT_SOURCE_1",
    "DAYS_EMPLOYED",
    "credit_annuity_ratio",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "age_score_ratio",
    "NAME_EDUCATION_TYPE_Higher_education",
    "p_NUM_DAYS_LAST_DUE_1ST_VERSION_max",
    "score_credit_ratio",
    "NAME_FAMILY_STATUS_Married",
    "CODE_GENDER_F",
    "BUREAU_bureau_NUM_AMT_CREDIT_SUM_DEBT_mean_mean",
    "DAYS_BIRTH",
    "p_NUM_ip_NUM_AMT_PAYMENT_min_mean",
    "p_NUM_ip_NUM_late_payment_lag_max_mean",
    "FLAG_OWN_CAR_N",
    "prev_cat_CNT_NAME_CONTRACT_STATUS_Refused",
    "CODE_GENDER_M",
    "BUREAU_bureau_NUM_AMT_CREDIT_MAX_OVERDUE_mean_mean",
    "BUREAU_bureau_CNT_CREDIT_ACTIVE_Closed_mean",
    "prev_cat_CNT_NAME_YIELD_GROUP_low_action",
    "LAST3_ip_NUM_late_payment_lag_max_mean",
    "ORGANIZATION_TYPE_GROUPED_Public_Sector",
    "BUREAU_bureau_NUM_credit_usage_mean_mean",
    "SUM_FLAG_DOCUMENT",
    "AMT_CREDIT",
    "REGION_RATING_CLIENT_W_CITY",
    "p_NUM_AMT_DOWN_PAYMENT_max",
    "prev_cat_CNT_NAME_YIELD_GROUP_high",
    "DAYS_ID_PUBLISH",
    "p_NUM_DAYS_LAST_DUE_1ST_VERSION_mean",
    "p_NUM_ccb_NUM_CNT_DRAWINGS_ATM_CURRENT_mean_max",
    "p_NUM_DAYS_LAST_DUE_max",
    "p_NUM_ip_NUM_AMT_PAYMENT_min_max",
    "p_NUM_ip_NUM_late_payment_lag_max_max",
    "BUREAU_bureau_NUM_DAYS_CREDIT_mean_mean",
    "BUREAU_bureau_NUM_bureau_debt_over_limit_mean_mean",
    "p_NUM_POS_NUM_CNT_INSTALMENT_FUTURE_min_mean",
    "DEF_60_CNT_SOCIAL_CIRCLE",
    "FLAG_DOCUMENT_3",
    "p_NUM_POS_NUM_MONTHS_BALANCE_max_max",
    "p_NUM_POS_NUM_CNT_INSTALMENT_FUTURE_mean_mean",
    "BUREAU_bureau_NUM_AMT_CREDIT_SUM_mean_mean",
    "OWN_CAR_AGE",
    "prev_cat_CNT_PRODUCT_COMBINATION_Cash_X_Sell__low",
    "LAST3_ip_NUM_late_payment_lag_max_max",
    "prev_cat_CNT_CODE_REJECT_REASON_XAP",
    "p_NUM_ip_NUM_NUM_INSTALMENT_VERSION_mean_max",
    "NAME_EDUCATION_TYPE_Secondary_secondary_special",
    "BUREAU_bureau_CNT_CREDIT_TYPE_Mortgage_mean",
    "NAME_CONTRACT_TYPE_Cash_loans",
    "LAST3_ip_NUM_AMT_PAYMENT_min_max",
    "REG_CITY_NOT_LIVE_CITY",
    "p_NUM_POS_NUM_CNT_INSTALMENT_std_mean",
    "LAST3_ip_NUM_late_payment_lag_mean_max",
    "DAYS_LAST_PHONE_CHANGE",
    "p_NUM_AMT_ANNUITY_mean",
    "LAST3_ip_NUM_DAYS_INSTALMENT_std_max",
    "prev_cat_CNT_NAME_PRODUCT_TYPE_walk_in",
    "LAST3_POS_NUM_CNT_INSTALMENT_FUTURE_mean_std",
    "BUREAU_bureau_NUM_DAYS_CREDIT_ENDDATE_mean_mean",
    "AMT_REQ_CREDIT_BUREAU_QRT",
    "p_NUM_POS_NUM_CNT_INSTALMENT_std_max",
    "p_NUM_ip_NUM_NUM_INSTALMENT_VERSION_std_max",
    "LAST3_ip_NUM_AMT_PAYMENT_min_mean",
    "p_NUM_AMT_DOWN_PAYMENT_mean",
    "LAST1_ip_NUM_late_payment_lag_max_mean",
    "NAME_INCOME_TYPE_GROUPED_Working",
    "DAYS_REGISTRATION",
    "p_NUM_POS_NUM_MONTHS_BALANCE_mean_max",
    "p_NUM_ip_NUM_NUM_INSTALMENT_VERSION_std_mean",
    "BUREAU_bureau_CNT_CREDIT_TYPE_Microloan_mean",
    "p_NUM_CNT_PAYMENT_mean",
    "p_NUM_ip_NUM_AMT_INSTALMENT_min_mean",
    "p_NUM_ip_NUM_AMT_PAYMENT_mean_max",
    "income_goods_ratio",
    "prev_cat_CNT_PRODUCT_COMBINATION_POS_industry_with_interest",
    "p_NUM_ip_NUM_DAYS_ENTRY_PAYMENT_max_max",
    "APARTMENTS_MEDI",
    "LAST3_ip_NUM_late_payment_lag_std_mean",
    "p_NUM_ip_NUM_AMT_INSTALMENT_min_max",
    "AMT_INCOME_TOTAL",
    "LAST3_ccb_NUM_CNT_DRAWINGS_CURRENT_std_mean",
    "p_NUM_ccb_NUM_CNT_DRAWINGS_CURRENT_std_max",
    "BUREAU_bureau_NUM_AMT_CREDIT_SUM_LIMIT_mean_mean",
    "BUREAU_bureau_CNT_CREDIT_ACTIVE_Active_mean",
    "p_NUM_POS_NUM_credit_term_ratio_mean_mean",
    "p_NUM_POS_NUM_CNT_INSTALMENT_FUTURE_min_max",
    "p_NUM_POS_NUM_CNT_INSTALMENT_FUTURE_mean_max",
    "LAST3_ip_NUM_late_payment_lag_std_max",
    "p_NUM_ip_NUM_DAYS_INSTALMENT_std_max",
    "DEF_30_CNT_SOCIAL_CIRCLE",
    "prev_cat_CNT_NAME_GOODS_CATEGORY_Furniture",
    "p_NUM_ccb_NUM_CNT_DRAWINGS_ATM_CURRENT_mean_mean",
    "LAST3_DAYS_LAST_DUE_1ST_VERSION_max",
    "p_NUM_RATE_DOWN_PAYMENT_max",
    "LAST3_POS_NUM_SK_DPD_DEF_std_std",
    "p_NUM_ip_NUM_DAYS_INSTALMENT_min_mean",
    "p_NUM_HOUR_APPR_PROCESS_START_max",
    "p_NUM_ccb_NUM_AMT_BALANCE_max_max",
    "p_NUM_POS_CNT_NAME_CONTRACT_STATUS_Active_max",
    "p_NUM_ip_NUM_DAYS_ENTRY_PAYMENT_std_max",
    "LAST3_HOUR_APPR_PROCESS_START_max",
    "LAST3_ip_NUM_DAYS_ENTRY_PAYMENT_std_max",
    "APARTMENTS_AVG",
    "p_NUM_DAYS_FIRST_DRAWING_max",
    "BUREAU_bureau_CNT_CREDIT_TYPE_Car_loan_mean",
    "OCCUPATION_TYPE_GROUPED_Core_staff",
    "p_NUM_RATE_DOWN_PAYMENT_mean",
    "LAST3_RATE_DOWN_PAYMENT_max",
    "prev_cat_CNT_NAME_PRODUCT_TYPE_XNA",
    "p_NUM_AMT_GOODS_PRICE_mean",
    "LAST3_RATE_DOWN_PAYMENT_std",
    "LAST1_ip_NUM_AMT_PAYMENT_min_mean",
    "p_NUM_HOUR_APPR_PROCESS_START_mean",
    "LAST1_AMT_GOODS_PRICE_mean",
    "LAST1_HOUR_APPR_PROCESS_START_mean",
    "p_NUM_CNT_PAYMENT_max",
    "p_NUM_POS_CNT_NAME_CONTRACT_STATUS_Active_mean",
    "LAST3_ip_NUM_DAYS_ENTRY_PAYMENT_max_max",
    "LAST3_POS_NUM_CNT_INSTALMENT_mean_std",
    "p_NUM_DAYS_DECISION_mean",
    "OCCUPATION_TYPE_GROUPED_Drivers",
    "REGION_POPULATION_RELATIVE",
    "BUREAU_bureau_NUM_bureau_CNT_STATUS_1_mean_mean",
    "LAST3_POS_NUM_credit_term_ratio_mean_mean",
    "p_NUM_ip_NUM_DAYS_ENTRY_PAYMENT_std_mean",
    "LAST3_CNT_PAYMENT_mean",
    "LAST1_ip_NUM_late_payment_lag_std_mean",
    "p_NUM_ccb_NUM_CNT_DRAWINGS_ATM_CURRENT_std_mean",
    "prev_cat_CNT_CODE_REJECT_REASON_LIMIT",
    "ORGANIZATION_TYPE_GROUPED_Self_employed",
    "BUREAU_bureau_NUM_bureau_CNT_STATUS_0_mean_mean",
    "LAST3_AMT_ANNUITY_std",
    "p_NUM_ccb_NUM_AMT_CREDIT_LIMIT_ACTUAL_mean_mean",
    "p_NUM_DAYS_TERMINATION_max",
    "LAST3_ip_NUM_DAYS_ENTRY_PAYMENT_std_mean",
    "LAST3_AMT_CREDIT_mean",
    "prev_cat_CNT_NAME_CONTRACT_STATUS_Approved",
    "TOTALAREA_MODE",
    "YEARS_BEGINEXPLUATATION_MEDI",
    "LAST3_DAYS_LAST_DUE_1ST_VERSION_std",
    "prev_cat_CNT_NAME_CLIENT_TYPE_New",
    "p_NUM_POS_NUM_SK_DPD_DEF_std_max",
    "LAST3_HOUR_APPR_PROCESS_START_std",
    "p_NUM_ip_NUM_AMT_PAYMENT_max_max",
    "p_NUM_POS_NUM_credit_term_ratio_min_mean",
    "prev_cat_CNT_CHANNEL_TYPE_AP___Cash_loan_",
    "LAST3_CNT_PAYMENT_max",
    "p_NUM_ccb_NUM_AMT_RECEIVABLE_PRINCIPAL_mean_max",
    "BUREAU_bureau_NUM_DAYS_CREDIT_UPDATE_mean_mean",
    "BUREAU_bureau_NUM_bureau_NUM_MONTHS_BALANCE_std_mean_mean",
    "p_NUM_AMT_APPLICATION_max",
    "YEARS_BEGINEXPLUATATION_MODE",
    "LAST3_ip_NUM_late_payment_lag_max_std",
    "p_NUM_DAYS_LAST_DUE_mean",
    "LAST3_AMT_ANNUITY_max",
    "BUREAU_bureau_NUM_DAYS_ENDDATE_FACT_mean_mean",
    "p_NUM_ip_NUM_DAYS_INSTALMENT_std_mean",
    "prev_cat_CNT_CODE_REJECT_REASON_HC",
    "LAST3_POS_NUM_credit_term_ratio_std_mean",
    "prev_cat_CNT_PRODUCT_COMBINATION_Cash_X_Sell__high",
    "LAST3_POS_NUM_SK_DPD_DEF_std_mean",
    "LAST3_RATE_DOWN_PAYMENT_mean",
    "p_NUM_ccb_NUM_CNT_DRAWINGS_CURRENT_mean_mean",
    "p_NUM_ip_NUM_AMT_PAYMENT_max_mean",
    "p_NUM_ccb_NUM_max_drawings_receivable_ratio_std_max",
    "p_NUM_ip_NUM_late_payment_lag_std_mean",
    "p_NUM_ccb_NUM_CNT_DRAWINGS_CURRENT_mean_max",
    "LAST3_POS_NUM_credit_term_ratio_min_std",
    "OCCUPATION_TYPE_GROUPED_Accountants",
    "p_NUM_ccb_NUM_max_drawings_receivable_ratio_mean_mean",
    "p_NUM_DAYS_FIRST_DRAWING_mean",
    "OCCUPATION_TYPE_GROUPED_Laborers",
    "LAST3_AMT_DOWN_PAYMENT_max",
    "LAST1_ip_NUM_late_payment_lag_mean_mean",
    "YEARS_BUILD_MODE",
    "p_NUM_ip_NUM_NUM_INSTALMENT_VERSION_mean_mean",
    "LAST3_ip_NUM_NUM_INSTALMENT_VERSION_std_mean",
    "LAST3_ip_NUM_DAYS_INSTALMENT_min_mean",
    "p_NUM_DAYS_DECISION_max",
    "p_NUM_POS_NUM_CNT_INSTALMENT_FUTURE_std_mean",
    "p_NUM_POS_NUM_credit_term_ratio_std_max",
    "p_NUM_ccb_NUM_max_drawings_receivable_ratio_max_max",
    "prev_cat_CNT_CHANNEL_TYPE_Channel_of_corporate_sales",
    "LAST3_POS_NUM_SK_DPD_DEF_mean_mean",
    "LAST3_ip_NUM_NUM_INSTALMENT_VERSION_mean_mean",
    "p_NUM_ccb_NUM_AMT_BALANCE_mean_mean",
    "p_NUM_ip_NUM_AMT_PAYMENT_mean_mean",
    "LAST3_ip_NUM_AMT_INSTALMENT_min_max",
    "p_NUM_ip_NUM_NUM_INSTALMENT_VERSION_max_max",
    "LAST1_SELLERPLACE_AREA_mean",
    "LAST1_ip_NUM_DAYS_INSTALMENT_std_mean",
    "p_NUM_ip_NUM_late_payment_lag_mean_mean",
    "LAST1_DAYS_TERMINATION_mean",
    "YEARS_BEGINEXPLUATATION_AVG",
    "LAST3_ip_NUM_late_payment_lag_min_std",
    "LAST3_SELLERPLACE_AREA_mean",
    "p_NUM_POS_NUM_SK_DPD_DEF_max_mean",
    "LAST3_ip_NUM_NUM_INSTALMENT_NUMBER_max_max",
    "p_NUM_SELLERPLACE_AREA_max",
    "LIVINGAPARTMENTS_MODE",
    "ORGANIZATION_TYPE_GROUPED_Finance",
    "LAST3_ip_NUM_NUM_INSTALMENT_VERSION_mean_max",
    "BUREAU_bureau_NUM_bureau_CNT_STATUS_C_mean_mean",
    "p_NUM_ip_NUM_AMT_PAYMENT_std_mean",
    "LAST3_DAYS_DECISION_std",
    "LAST3_ip_NUM_AMT_INSTALMENT_mean_max",
    "LAST3_POS_NUM_SK_DPD_mean_std",
    "LAST3_SELLERPLACE_AREA_std",
    "prev_cat_CNT_PRODUCT_COMBINATION_POS_industry_without_interest",
    "p_NUM_ip_NUM_AMT_INSTALMENT_std_mean",
    "p_NUM_ccb_NUM_AMT_CREDIT_LIMIT_ACTUAL_mean_max",
    "p_NUM_ccb_NUM_AMT_DRAWINGS_CURRENT_std_max",
    "p_NUM_ip_NUM_late_payment_lag_mean_max",
    "LAST3_HOUR_APPR_PROCESS_START_mean",
    "p_NUM_POS_NUM_MONTHS_BALANCE_std_mean",
    "LAST3_DAYS_TERMINATION_std",
    "LAST3_AMT_GOODS_PRICE_mean",
    "LAST3_POS_NUM_credit_term_ratio_min_max",
    "LAST3_ip_NUM_AMT_INSTALMENT_max_mean",
    "LAST3_AMT_ANNUITY_mean",
    "LAST3_POS_NUM_SK_DPD_DEF_std_max",
    "p_NUM_POS_NUM_CNT_INSTALMENT_max_mean",
    "p_NUM_ip_NUM_late_payment_lag_std_max",
    "LAST3_ip_NUM_DAYS_INSTALMENT_std_mean",
    "LAST3_POS_NUM_credit_term_ratio_mean_max",
    "LAST3_DAYS_LAST_DUE_max",
    "LAST3_DAYS_TERMINATION_max",
    "LAST3_POS_NUM_CNT_INSTALMENT_FUTURE_mean_mean",
    "LAST3_DAYS_DECISION_mean",
    "p_NUM_ccb_NUM_AMT_PAYMENT_TOTAL_CURRENT_mean_mean",
    "LIVINGAREA_MODE",
    "p_NUM_POS_NUM_MONTHS_BALANCE_std_max",
    "p_NUM_ip_NUM_DAYS_INSTALMENT_mean_max",
    "prev_cat_CNT_NAME_CONTRACT_TYPE_Consumer_loans",
    "LAST1_ip_NUM_DAYS_ENTRY_PAYMENT_max_mean",
    "p_NUM_ip_NUM_DAYS_INSTALMENT_mean_mean",
    "YEARS_BUILD_AVG",
    "p_NUM_POS_NUM_MONTHS_BALANCE_min_mean",
    "LAST3_ccb_NUM_CNT_DRAWINGS_CURRENT_mean_mean",
    "LAST3_POS_NUM_CNT_INSTALMENT_std_mean",
    "FLOORSMAX_AVG",
    "LAST3_ip_NUM_late_payment_lag_mean_mean",
    "LAST3_ip_NUM_AMT_INSTALMENT_min_mean",
    "NAME_INCOME_TYPE_GROUPED_State_servant",
    "p_NUM_POS_NUM_MONTHS_BALANCE_min_max",
    "prev_cat_CNT_NAME_PORTFOLIO_POS",
    "p_NUM_ccb_NUM_CNT_DRAWINGS_ATM_CURRENT_std_max",
    "p_NUM_ip_NUM_AMT_INSTALMENT_max_mean",
    "LAST1_ip_NUM_AMT_INSTALMENT_min_mean",
    "prev_cat_CNT_NAME_YIELD_GROUP_low_normal",
    "LAST3_POS_NUM_CNT_INSTALMENT_std_std",
    "LAST3_ip_NUM_late_payment_lag_std_std",
    "p_NUM_ip_NUM_DAYS_INSTALMENT_max_max",
    "LAST3_POS_NUM_SK_DPD_max_std",
    "LAST3_ip_NUM_late_payment_lag_min_max",
    "LAST1_ip_NUM_NUM_INSTALMENT_VERSION_mean_mean",
    "LAST3_ip_NUM_late_payment_lag_mean_std",
    "LAST1_ip_NUM_AMT_INSTALMENT_std_mean",
    "FLAG_WORK_PHONE",
    "BUREAU_bureau_NUM_bureau_NUM_MONTHS_BALANCE_mean_mean_mean",
    "APARTMENTS_MODE",
    "p_NUM_POS_NUM_CNT_INSTALMENT_FUTURE_std_max",
    "LAST3_ip_NUM_NUM_INSTALMENT_VERSION_std_max",
    "LAST3_POS_NUM_credit_term_ratio_mean_std",
    "LAST3_POS_NUM_MONTHS_BALANCE_std_std",
    "prev_cat_CNT_PRODUCT_COMBINATION_POS_household_without_interest",
    "LAST3_CNT_PAYMENT_std",
    "LANDAREA_AVG",
    "p_NUM_AMT_CREDIT_mean",
    "p_NUM_ccb_NUM_max_drawings_receivable_ratio_std_mean",
    "BUREAU_bureau_NUM_bureau_CNT_STATUS_X_mean_mean",
    "BUREAU_bureau_NUM_AMT_ANNUITY_mean_mean",
    "BUREAU_bureau_CNT_CREDIT_TYPE_Consumer_credit_mean",
    "LAST3_ip_NUM_DAYS_INSTALMENT_min_std",
    "LAST3_AMT_DOWN_PAYMENT_mean",
    "p_NUM_ip_NUM_AMT_INSTALMENT_max_max",
    "BUREAU_bureau_NUM_bureau_NUM_MONTHS_BALANCE_min_mean_mean",
    "LAST1_RATE_DOWN_PAYMENT_mean",
    "p_NUM_POS_NUM_MONTHS_BALANCE_mean_mean",
    "LAST3_ip_NUM_DAYS_INSTALMENT_mean_std",
    "LIVINGAREA_AVG",
    "LAST3_DAYS_FIRST_DUE_mean",
    "LAST3_POS_NUM_MONTHS_BALANCE_min_mean",
    "LAST3_ip_NUM_late_payment_lag_min_mean",
    "p_NUM_POS_NUM_credit_term_ratio_min_max",
    "LAST3_ccb_NUM_AMT_INST_MIN_REGULARITY_std_mean",
    "NONLIVINGAREA_AVG",
    "LAST3_ip_NUM_DAYS_ENTRY_PAYMENT_mean_mean",
    "p_NUM_ccb_NUM_AMT_PAYMENT_CURRENT_max_mean",
    "LAST3_POS_CNT_NAME_CONTRACT_STATUS_Active_max",
    "p_NUM_POS_NUM_SK_DPD_DEF_max_max",
    "LAST1_POS_NUM_credit_term_ratio_mean_mean",
    "BUREAU_bureau_NUM_AMT_CREDIT_SUM_OVERDUE_mean_mean",
    "FLOORSMAX_MEDI",
    "LAST1_POS_NUM_credit_term_ratio_std_mean",
    "LAST3_ip_NUM_NUM_INSTALMENT_NUMBER_std_max",
    "p_NUM_ip_NUM_late_payment_lag_min_max",
    "LAST1_ip_NUM_DAYS_ENTRY_PAYMENT_min_mean",
    "prev_cat_CNT_NAME_SELLER_INDUSTRY_Connectivity",
    "LAST1_POS_NUM_SK_DPD_std_mean",
    "LAST1_ccb_NUM_CNT_DRAWINGS_CURRENT_std_mean",
    "LAST3_POS_CNT_NAME_CONTRACT_STATUS_Active_std",
    "p_NUM_ccb_NUM_AMT_PAYMENT_CURRENT_mean_mean",
    "p_NUM_ccb_NUM_AMT_RECEIVABLE_PRINCIPAL_mean_mean",
    "LAST1_DAYS_LAST_DUE_mean",
    "LAST3_POS_NUM_CNT_INSTALMENT_FUTURE_min_max",
    "p_NUM_SELLERPLACE_AREA_mean",
    "p_NUM_ip_NUM_DAYS_ENTRY_PAYMENT_min_mean",
    "LAST1_POS_NUM_MONTHS_BALANCE_mean_mean",
    "LAST1_ip_NUM_AMT_PAYMENT_max_mean",
    "LAST3_ip_NUM_AMT_PAYMENT_std_mean",
    "p_NUM_POS_NUM_SK_DPD_DEF_std_mean",
    "p_NUM_POS_NUM_CNT_INSTALMENT_mean_max",
    "LAST1_DAYS_LAST_DUE_1ST_VERSION_mean",
    "LAST3_ccb_NUM_max_drawings_receivable_ratio_mean_mean",
    "LAST3_AMT_APPLICATION_std",
    "p_NUM_ip_NUM_late_payment_lag_min_mean",
    "LIVINGAPARTMENTS_AVG",
    "LAST3_POS_NUM_credit_term_ratio_std_std",
    "p_NUM_ccb_NUM_AMT_CREDIT_LIMIT_ACTUAL_min_mean",
    "LAST3_POS_NUM_CNT_INSTALMENT_FUTURE_min_mean",
    "p_NUM_ccb_NUM_CNT_DRAWINGS_CURRENT_max_max",
    "p_NUM_ccb_NUM_AMT_BALANCE_min_max",
    "LAST1_POS_CNT_NAME_CONTRACT_STATUS_Active_mean",
    "LAST1_ip_NUM_AMT_INSTALMENT_mean_mean",
    "LAST3_ccb_NUM_AMT_INST_MIN_REGULARITY_max_mean",
    "p_NUM_ip_NUM_NUM_INSTALMENT_NUMBER_mean_mean",
    "p_NUM_ccb_NUM_CNT_INSTALMENT_MATURE_CUM_std_mean",
    "HOUR_APPR_PROCESS_START",
    "LAST3_ccb_NUM_AMT_BALANCE_mean_max",
    "LAST3_ip_NUM_AMT_INSTALMENT_max_max",
    "LAST3_POS_CNT_NAME_CONTRACT_STATUS_Active_mean",
    "LIVINGAPARTMENTS_MEDI",
    "OBS_60_CNT_SOCIAL_CIRCLE",
    "prev_cat_CNT_NAME_GOODS_CATEGORY_Photo___Cinema_Equipment",
    "BASEMENTAREA_AVG",
    "LAST3_DAYS_FIRST_DRAWING_mean",
    "LAST1_ip_NUM_AMT_PAYMENT_std_mean",
    "FLAG_DOCUMENT_11",
    "BASEMENTAREA_MEDI",
    "p_NUM_AMT_CREDIT_max",
    "p_NUM_AMT_APPLICATION_mean",
    "LAST3_ip_NUM_AMT_PAYMENT_mean_max",
    "LAST3_ccb_NUM_CNT_DRAWINGS_ATM_CURRENT_std_mean",
    "BUREAU_bureau_NUM_bureau_NUM_MONTHS_BALANCE_max_mean_mean",
    "p_NUM_ccb_NUM_AMT_DRAWINGS_CURRENT_mean_mean",
    "LAST3_POS_NUM_MONTHS_BALANCE_std_mean",
    "LAST3_POS_NUM_CNT_INSTALMENT_min_mean",
    "LAST1_POS_NUM_CNT_INSTALMENT_FUTURE_std_mean",
    "LAST3_ip_NUM_AMT_PAYMENT_max_std",
    "LAST3_ip_NUM_DAYS_ENTRY_PAYMENT_mean_std",
    "LAST3_POS_CNT_NAME_CONTRACT_STATUS_Signed_mean",
    "p_NUM_ccb_NUM_AMT_RECIVABLE_mean_max",
    "BUREAU_bureau_CNT_CREDIT_TYPE_Credit_card_mean",
    "LAST1_POS_NUM_CNT_INSTALMENT_FUTURE_mean_mean",
    "p_NUM_ip_NUM_NUM_INSTALMENT_NUMBER_mean_max",
    "LAST3_DAYS_LAST_DUE_std",
    "LAST3_ip_NUM_DAYS_INSTALMENT_max_std",
    "LAST3_ip_NUM_AMT_PAYMENT_std_std",
    "p_NUM_POS_NUM_CNT_INSTALMENT_mean_mean",
    "p_NUM_ccb_NUM_AMT_BALANCE_mean_max",
    "LAST1_ip_NUM_DAYS_INSTALMENT_mean_mean",
    "LAST3_POS_NUM_CNT_INSTALMENT_mean_mean",
    "prev_cat_CNT_NAME_CASH_LOAN_PURPOSE_XAP",
    "LAST3_POS_NUM_CNT_INSTALMENT_std_max",
    "LAST1_ip_NUM_NUM_INSTALMENT_VERSION_std_mean",
    "LAST3_ip_NUM_AMT_INSTALMENT_max_std",
    "LAST3_DAYS_LAST_DUE_mean",
    "LAST3_ip_NUM_DAYS_ENTRY_PAYMENT_max_std",
    "LAST1_AMT_ANNUITY_mean",
    "p_NUM_ccb_NUM_AMT_BALANCE_std_mean",
]


def load_csvs() -> None:
    """
    Lazily load two pre-computed Parquet datasets into global variables.

    The function populates the module-level variables
    ``bureau_final`` and ``p_final_merged`` the first time it is called.
    Subsequent calls are no-ops, allowing the DataFrames to act as an
    in-memory cache across the application.

    Raises
    ------
    FileNotFoundError
        If either Parquet file cannot be located.
    OSError
        For low-level I/O errors during reading.
    Any other exception emitted by :pyfunc:`dask.dataframe.read_parquet`
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
    Lazily load the trained LightGBM model (`best_lgbm_model.pkl`)
    and cache it on the function object itself.

    Returns
    -------
    Any
        The un-pickled model object (typically a ``lightgbm.Booster`` or
        scikit-learn–compatible estimator), loaded once and reused on
        subsequent calls.
    """
    if not hasattr(load_model, "model"):
        load_model.model = joblib.load("best_lgbm_model.pkl")
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
    3. Use `infer_objects(copy=False)` to convert columns to inferred types.
    4. Replace placeholder 365243 in `DAYS_EMPLOYED` with NaN (likely sentinel).
    5. Set `OWN_CAR_AGE` to 0 if missing and applicant does not own a car.
    6. Clip `AMT_INCOME_TOTAL` to its 99th percentile to reduce outliers.

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
        "Security Ministries": "Public Sector",
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
        "Mobile": "Other",
    }

    df["ORGANIZATION_TYPE_GROUPED"] = df["ORGANIZATION_TYPE"].map(group_map_3)

    group_map_4 = {
        "House / apartment": "House / apartment",
        "Municipal apartment": "Other apartment",
        "Office apartment": "Other apartment",
        "Rented apartment": "Other apartment",
        "With parents": "With parents",
        "Co-op apartment": "Other apartment",
    }

    df["NAME_HOUSING_TYPE_GROUPED"] = df["NAME_HOUSING_TYPE"].map(group_map_4)
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
    2. Convert object columns to pandas `category` dtype.
    3. Apply one-hot encoding via `pd.get_dummies`, keeping all categories.
    4. Sanitize column names via `clean_column_names()` (assumed to exist).
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
    - Joins external aggregated features from `bureau_final` and `p_final_merged`
    - Applies encoding and cleaning to ensure consistency with training
    - Ensures all `selected_features` are present in final DataFrame

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
            if col in selected_features or col == "SK_ID_CURR"
        ]
        prev_cols = [
            col
            for col in p_final_merged.columns
            if col in selected_features or col == "SK_ID_CURR"
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

    for col in selected_features:
        if col not in df.columns:
            df[col] = 0

    df = df[selected_features]

    return df
