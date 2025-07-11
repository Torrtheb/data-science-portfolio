FROM python:3.11

WORKDIR /app

COPY . /app
COPY best_brf.joblib bmi_imputation_pipeline.joblib preprocessing_pipeline.joblib /app/

RUN pip install --no-cache-dir --upgrade uv && \
    uv venv .venv && \
    uv pip install --system -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "deploy:app", "--host", "0.0.0.0", "--port", "8000"]
