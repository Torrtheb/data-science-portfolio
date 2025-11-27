# Backend Dockerfile (FastAPI on Uvicorn)
# Build with project root (aieng_3) as context:
#   docker build -f backend/Dockerfile -t backend:local .
# Cloud Build uses aieng_3 as context and backend/Dockerfile

FROM python:3.13-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Minimal system deps; psycopg[binary] avoids needing libpq
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install locked dependencies via uv (use uv.lock + pyproject.toml)
# IMPORTANT: Build with aieng_3 as context so these COPY paths resolve.
COPY pyproject.toml ./pyproject.toml
COPY uv.lock ./uv.lock
# Create and use an in-project virtualenv; add to PATH for runtime
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN python -m pip install --upgrade pip && pip install uv \
 && uv sync --frozen --no-dev

# Copy backend source
COPY backend /app/backend

# Default port for Cloud Run
ENV PORT=8080
EXPOSE 8080

WORKDIR /app/backend

# Start FastAPI with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
