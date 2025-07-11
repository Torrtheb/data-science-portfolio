FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential  \
        libgomp1         \
        curl             \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /install
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt

FROM python:3.11-slim
COPY --from=builder /install /usr/local

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        curl && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -m appuser
USER appuser

WORKDIR /app
COPY . .

ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_MAX_UPLOAD_SIZE=50 \
    STREAMLIT_SERVER_MAX_MESSAGE_SIZE=50 \
    STREAMLIT_GLOBAL_DEVELOPMENT_MODE=false \
    STREAMLIT_GLOBAL_SUPPRESS_WARNING=true

ENV GCS_BUCKET="" \
    GCS_MODEL="" \
    GCS_VALID_DATA=""

ENV PYTHONHASHSEED=0 \
    MALLOC_ARENA_MAX=2

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app_memory_optimized.py", "--server.maxUploadSize=50", "--server.maxMessageSize=50"]
