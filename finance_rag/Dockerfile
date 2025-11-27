# ---------- Build stage ----------
FROM python:3.13-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip wheel --no-cache-dir -r requirements.txt -w /build/wheels


# ---------- Runtime stage ----------
FROM python:3.13-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin"

# non-root user
RUN useradd -m -u 10001 appuser

# system deps + WeasyPrint runtime + tini for proper signal handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git bash \
    libglib2.0-0 libcairo2 libpango-1.0-0 libpangocairo-1.0-0 \
    libgdk-pixbuf-2.0-0 shared-mime-info fonts-dejavu tzdata \
    tini \
 && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /app

# python deps from wheels
COPY --from=builder /build/wheels /wheels
COPY requirements.txt /app/requirements.txt
RUN pip install --no-index --find-links=/wheels -r /app/requirements.txt && \
    rm -rf /wheels

# app code
COPY back_app/ /app/back_app/
RUN [ -f /app/back_app/__init__.py ] || touch /app/back_app/__init__.py

# startup script
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# writable dir for DBs, etc.
RUN mkdir -p /data && chown -R appuser:appuser /app /data

# -------- Defaults (override in Cloud Run) --------
ENV PORT=8080
ENV PYTHONPATH=/app

# Run the proxy INSIDE the container and point the backend at it (same as your local flow)
ENV MCP_ENABLE=true
ENV MCP_PROXY_ENABLE=true
ENV MCP_PROXY_ENDPOINT=http://127.0.0.1:8077/mcp
# Make sure we do NOT try to spawn stdio directly in prod
ENV MCP_SERVER_WORLD_BANK=
ENV MCP_TRANSPORT=
ENV MCP_WORLD_BANK_CMD=
ENV MCP_WORLD_BANK_ARGS=

USER appuser
EXPOSE 8080

# JSON-form CMD (no shell), tini as PID 1 for clean shutdown
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/app/start.sh"]
