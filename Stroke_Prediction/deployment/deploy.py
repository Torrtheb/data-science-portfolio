from fastapi import FastAPI
import pandas as pd
import joblib
from pydantic import BaseModel
import uvicorn
import os
from typing import Optional
import numpy as np

app = FastAPI()

MODEL_PATH = "/app/best_brf.joblib" 
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

BMI_PIPELINE_PATH = "/app/bmi_imputation_pipeline.joblib"
PREPROCESSING_PIPELINE_PATH = "/app/preprocessing_pipeline.joblib"

if not os.path.exists(BMI_PIPELINE_PATH):
    raise FileNotFoundError(f"Pipeline file not found: {BMI_PIPELINE_PATH}")
if not os.path.exists(PREPROCESSING_PIPELINE_PATH):
    raise FileNotFoundError(f"Pipeline file not found: {PREPROCESSING_PIPELINE_PATH}")

pipeline_1 = joblib.load(BMI_PIPELINE_PATH)
pipeline_2 = joblib.load(PREPROCESSING_PIPELINE_PATH)

cols_to_drop = [
    'smoking_status_formerly smoked', 
    'Residence_type_Urban', 'work_type_Never_worked', 
    'gender_Female', 'ever_married_Yes', 'work_type_Private', 'stroke', 'id'
]
num_features = ['age', 'bmi', 'avg_glucose_level']

class ModelInput(BaseModel): 
    id: int
    gender: str
    age: float
    hypertension: int
    heart_disease: int
    ever_married: str 
    work_type: str
    Residence_type: str
    avg_glucose_level: float
    bmi: Optional[float]
    smoking_status: str
    stroke: int

@app.post("/predict/")

def predict(data: ModelInput) -> dict:
    """
    Preprocess input data, handle missing values, apply transformations, 
    and make a prediction using the loaded model.

    Args:
        data (ModelInput): Input data in the form of a Pydantic model.

    Returns:
        dict: A dictionary containing the prediction probability or an error message.
    """
    input_df = pd.DataFrame([data.model_dump()])
    
    if input_df['bmi'].isnull().any():
        input_df['bmi'] = np.nan
    
    imp_data = pipeline_1.transform(input_df)
    imp_data = pd.DataFrame(
        imp_data, 
        columns=['age', 'bmi', 'bmi_imputed'] + [col for col in input_df.columns if col not in ['age', 'bmi']]
    )

    imp_data['log_bmi'] = np.log1p(imp_data['bmi'].astype(float))  
    imp_data['log_glucose'] = np.log1p(imp_data['avg_glucose_level'].astype(float))
    imp_data = imp_data.drop(['avg_glucose_level', 'bmi'], axis=1)

    processed = pipeline_2.transform(imp_data)
    processed = pd.DataFrame(
        processed, 
        columns=pipeline_2.named_steps['column_transform'].get_feature_names_out()
    )
    processed.columns = processed.columns.str.replace('onehot__', '', regex=True)
    processed.columns = processed.columns.str.replace('scaler__', '', regex=True)
    processed.columns = processed.columns.str.replace('remainder__', '', regex=True)
    
    for col in processed.columns:
        if col in num_features:  
            processed[col] = processed[col].astype(float)
        else: 
            processed[col] = processed[col].astype(int)
    
    processed = processed.drop(columns=[col for col in cols_to_drop if col in processed.columns], errors="ignore")

    final_input = processed.to_numpy()
    
    prediction = model.predict_proba(final_input)[0, 1]
    
    return {"prediction_probability": prediction, "status": "success"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
