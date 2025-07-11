from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Optional
import logging
import asyncio
from contextlib import asynccontextmanager
import time

from preprocess import (
    preprocess_raw_input,
    load_csvs,
    load_model,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Set up fastapi lifespan to load model and CSVs lazily
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("Loading model lazily...")
        model = load_model()
        logger.info("Model loaded.")

        logger.info("Loading external CSV features...")
        await asyncio.to_thread(load_csvs)
        logger.info("External data loaded.")

        app.state.model = model
        yield
    except Exception as e:
        logger.exception("Error during startup: %s", e)
        raise HTTPException(status_code=500, detail="Error during startup")
    finally:
        logger.info("Shutting down...")


app = FastAPI(title="Home Credit Default API", lifespan=lifespan)


class ApplicationInput(BaseModel):
    SK_ID_CURR: int
    NAME_CONTRACT_TYPE: Optional[str] = None
    CODE_GENDER: Optional[str] = None
    FLAG_OWN_CAR: Optional[str] = None
    FLAG_OWN_REALTY: Optional[str] = None
    CNT_CHILDREN: Optional[int] = None
    AMT_INCOME_TOTAL: Optional[float] = None
    AMT_CREDIT: Optional[float] = None
    AMT_ANNUITY: Optional[float] = None
    AMT_GOODS_PRICE: Optional[float] = None
    NAME_TYPE_SUITE: Optional[str] = None
    NAME_INCOME_TYPE: Optional[str] = None
    NAME_EDUCATION_TYPE: Optional[str] = None
    NAME_FAMILY_STATUS: Optional[str] = None
    NAME_HOUSING_TYPE: Optional[str] = None
    REGION_POPULATION_RELATIVE: Optional[float] = None
    DAYS_BIRTH: Optional[float] = None
    DAYS_EMPLOYED: Optional[float] = None
    DAYS_REGISTRATION: Optional[float] = None
    DAYS_ID_PUBLISH: Optional[float] = None
    OWN_CAR_AGE: Optional[float] = None
    FLAG_MOBIL: Optional[int] = None
    FLAG_EMP_PHONE: Optional[int] = None
    FLAG_WORK_PHONE: Optional[int] = None
    FLAG_CONT_MOBILE: Optional[int] = None
    FLAG_PHONE: Optional[int] = None
    FLAG_EMAIL: Optional[int] = None
    OCCUPATION_TYPE: Optional[str] = None
    CNT_FAM_MEMBERS: Optional[float] = None
    REGION_RATING_CLIENT: Optional[int] = None
    REGION_RATING_CLIENT_W_CITY: Optional[int] = None
    WEEKDAY_APPR_PROCESS_START: Optional[str] = None
    HOUR_APPR_PROCESS_START: Optional[int] = None
    REG_REGION_NOT_LIVE_REGION: Optional[int] = None
    REG_REGION_NOT_WORK_REGION: Optional[int] = None
    LIVE_REGION_NOT_WORK_REGION: Optional[int] = None
    REG_CITY_NOT_LIVE_CITY: Optional[int] = None
    REG_CITY_NOT_WORK_CITY: Optional[int] = None
    LIVE_CITY_NOT_WORK_CITY: Optional[int] = None
    ORGANIZATION_TYPE: Optional[str] = None
    EXT_SOURCE_1: Optional[float] = None
    EXT_SOURCE_2: Optional[float] = None
    EXT_SOURCE_3: Optional[float] = None
    FLAG_DOCUMENT_2: Optional[int] = 0
    FLAG_DOCUMENT_3: Optional[int] = 0
    FLAG_DOCUMENT_4: Optional[int] = 0
    FLAG_DOCUMENT_5: Optional[int] = 0
    FLAG_DOCUMENT_6: Optional[int] = 0
    FLAG_DOCUMENT_7: Optional[int] = 0
    FLAG_DOCUMENT_8: Optional[int] = 0
    FLAG_DOCUMENT_9: Optional[int] = 0
    FLAG_DOCUMENT_10: Optional[int] = 0
    FLAG_DOCUMENT_11: Optional[int] = 0
    FLAG_DOCUMENT_12: Optional[int] = 0
    FLAG_DOCUMENT_13: Optional[int] = 0
    FLAG_DOCUMENT_14: Optional[int] = 0
    FLAG_DOCUMENT_15: Optional[int] = 0
    FLAG_DOCUMENT_16: Optional[int] = 0
    FLAG_DOCUMENT_17: Optional[int] = 0
    FLAG_DOCUMENT_18: Optional[int] = 0
    FLAG_DOCUMENT_19: Optional[int] = 0
    FLAG_DOCUMENT_20: Optional[int] = 0
    FLAG_DOCUMENT_21: Optional[int] = 0


# Define the response model for predictions
class PredictionResponse(BaseModel):
    SK_ID_CURR: int
    default_probability: float


@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: ApplicationInput, request: Request):
    start_time = time.time()
    try:
        input_dict = input_data.model_dump()
        features_df = preprocess_raw_input(input_dict)
        import pandas as pd

        if not isinstance(features_df, pd.DataFrame):
            raise HTTPException(
                status_code=500,
                detail="Preprocessing did not return a pandas DataFrame.",
            )

        logger.info(f"Feature shape: {features_df.shape}")
        logger.info(f"Feature columns: {features_df.columns.tolist()}")

        model = request.app.state.model
        prediction = model.predict_proba(features_df)[:, 1][0]

        duration = time.time() - start_time
        logger.info(
            f"Prediction for SK_ID_CURR={input_data.SK_ID_CURR} took {duration:.3f} seconds. Default probability: {prediction:.6f}"
        )

        return {
            "SK_ID_CURR": input_data.SK_ID_CURR,
            "default_probability": float(prediction),
        }

    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            f"Prediction error for SK_ID_CURR={getattr(input_data, 'SK_ID_CURR', None)} after {duration:.3f} seconds: {e}"
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def read_root():
    logger.info("🔍 Health check accessed")
    return {"message": "Home Credit API is running!"}


@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring.
    """
    return {"status": "ok"}
