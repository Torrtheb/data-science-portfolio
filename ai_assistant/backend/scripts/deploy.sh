#!/usr/bin/env bash
set -euo pipefail

# deploy.sh — Build + Deploy FastAPI backend to Cloud Run
#
# Usage:
#   export PROJECT_ID=your-gcp-project
#   export REGION=us-east1
#   export SERVICE=aieng3-backend
#   export REPO=containers
#   export BACKEND_DATABASE_URL='postgresql+psycopg://neondb_owner:PW@ep-…-pooler…/neondb?sslmode=require'
#   export CORS_ALLOWED_ORIGINS='http://localhost:3000,http://127.0.0.1:3000'
#   # Optional (if you want this script to create/update secrets):
#   export OPENAI_API_KEY_VALUE='sk-…'
#   export LANGSMITH_API_KEY_VALUE='lsv2_…'
#   export NEXTAUTH_SECRET_VALUE='your-32+random-string'
#   
#   bash scripts/deploy.sh

# Defaults
PROJECT_ID=${PROJECT_ID:-}
REGION=${REGION:-us-east1}
SERVICE=${SERVICE:-aieng3-backend}
REPO=${REPO:-containers}
BACKEND_DATABASE_URL=${BACKEND_DATABASE_URL:-}
CORS_ALLOWED_ORIGINS=${CORS_ALLOWED_ORIGINS:-"http://localhost:3000,http://127.0.0.1:3000"}
SERVICE_ACCOUNT=${SERVICE_ACCOUNT:-}
# Control migration behavior and timeouts
RUN_DB_MIGRATIONS=${RUN_DB_MIGRATIONS:-1}
RUN_DB_MIGRATE_WAIT_SECS=${RUN_DB_MIGRATE_WAIT_SECS:-120}

# Secret names in Secret Manager
OPENAI_API_KEY_SM=${OPENAI_API_KEY_SM:-OPENAI_API_KEY}
LANGSMITH_API_KEY_SM=${LANGSMITH_API_KEY_SM:-LANGSMITH_API_KEY}
NEXTAUTH_SECRET_SM=${NEXTAUTH_SECRET_SM:-NEXTAUTH_SECRET}
# SMTP secret (password) and optional value for seeding/rotating
SMTP_PASS_SM=${SMTP_PASS_SM:-SMTP_PASS}

# Optional convenience: if provided, used to create/update the SMTP_PASS secret
SMTP_PASS_VALUE=${SMTP_PASS_VALUE:-}

if [[ -z "${PROJECT_ID}" ]]; then
  echo "ERROR: PROJECT_ID not set" >&2
  exit 1
fi
if [[ -z "${BACKEND_DATABASE_URL}" ]]; then
  echo "ERROR: BACKEND_DATABASE_URL not set (use Neon pooler DSN with sslmode=require)" >&2
  exit 1
fi

GCLOUD=$(command -v gcloud || true)
if [[ -z "$GCLOUD" ]]; then
  echo "ERROR: gcloud CLI is required" >&2
  exit 1
fi

echo "[deploy] Project: $PROJECT_ID  Region: $REGION  Service: $SERVICE"
gcloud config set project "$PROJECT_ID" >/dev/null
gcloud config set run/region "$REGION" >/dev/null

echo "[deploy] Enabling required services (idempotent)…"
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com secretmanager.googleapis.com >/dev/null

echo "[deploy] Ensuring Artifact Registry repo: $REPO"
gcloud artifacts repositories create "$REPO" \
  --repository-format=docker \
  --location="$REGION" >/dev/null 2>&1 || true
gcloud auth configure-docker "$REGION-docker.pkg.dev" -q >/dev/null

IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/$SERVICE:prod"
echo "[deploy] Building & pushing image via Cloud Build → $IMAGE"
# Resolve repository root regardless of current working directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Project root = two levels up from backend/scripts → aieng_3
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CB_CONFIG="$REPO_ROOT/backend/cloudbuild.docker.yaml"
if [[ ! -f "$CB_CONFIG" ]]; then
  echo "ERROR: Cloud Build config not found: $CB_CONFIG" >&2
  exit 1
fi
# Submit with repo root as build context so Dockerfile COPY paths work
gcloud builds submit \
  --config "$CB_CONFIG" \
  --substitutions _IMAGE="$IMAGE" \
  "$REPO_ROOT"

# Optionally create/update Secret Manager secrets from *_VALUE envs
create_or_update_secret() {
  local name="$1"; shift
  local value_var="$1"; shift
  local value="${!value_var:-}"
  if [[ -z "$value" ]]; then
    echo "[deploy] Secret $name: value not provided; expecting it already exists"
    return 0
  fi
  if gcloud secrets describe "$name" >/dev/null 2>&1; then
    echo "[deploy] Updating secret version: $name"
    printf '%s' "$value" | gcloud secrets versions add "$name" --data-file=- >/dev/null
  else
    echo "[deploy] Creating secret: $name"
    printf '%s' "$value" | gcloud secrets create "$name" --data-file=- >/dev/null
  fi
}

create_or_update_secret "$OPENAI_API_KEY_SM" OPENAI_API_KEY_VALUE
create_or_update_secret "$LANGSMITH_API_KEY_SM" LANGSMITH_API_KEY_VALUE
create_or_update_secret "$NEXTAUTH_SECRET_SM" NEXTAUTH_SECRET_VALUE
create_or_update_secret "$SMTP_PASS_SM" SMTP_PASS_VALUE

# Ensure service account can access secrets
SA_EMAIL=$(gcloud run services describe "$SERVICE" --format='value(spec.template.spec.serviceAccountName)' --region "$REGION" 2>/dev/null || true)
# Determine a valid service account to use if none configured yet
if [[ -z "$SA_EMAIL" ]]; then
  if [[ -n "$SERVICE_ACCOUNT" ]]; then
    SA_EMAIL="$SERVICE_ACCOUNT"
  else
    PN=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')
    SA_EMAIL="${PN}-compute@developer.gserviceaccount.com"
  fi
fi
echo "[deploy] Using service account: $SA_EMAIL"
echo "[deploy] Ensuring Secret Manager accessor role for $SA_EMAIL"
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/secretmanager.secretAccessor" >/dev/null || true

echo "[deploy] Deploying Cloud Run service: $SERVICE"

# If ENV_VARS_FILE is provided, use it and avoid any --set-env-vars (they can't be mixed)
DEPLOY_ARGS=(
  --image "$IMAGE"
  --allow-unauthenticated
  --region "$REGION"
  --port 8080
  --cpu 1 --memory 1Gi
  --max-instances 10 --min-instances 0
  --concurrency 40
  --timeout 300
  --service-account "$SA_EMAIL"
)

if [[ -n "${ENV_VARS_FILE:-}" ]]; then
  # Resolve ENV_VARS_FILE to an absolute path. Try as-is, then relative to repo root.
  if [[ ! -f "$ENV_VARS_FILE" ]]; then
    CANDIDATE="$REPO_ROOT/${ENV_VARS_FILE#./}"
    if [[ -f "$CANDIDATE" ]]; then
      ENV_VARS_FILE="$CANDIDATE"
    fi
  fi
  if [[ ! -f "$ENV_VARS_FILE" ]]; then
    echo "ERROR: ENV_VARS_FILE not found: $ENV_VARS_FILE" >&2
    exit 1
  fi
  echo "[deploy] Using env file: $ENV_VARS_FILE (replaces all env vars)"
  DEPLOY_ARGS+=( --env-vars-file "$ENV_VARS_FILE" )
else
  # Escape commas in CORS value for --set-env-vars (gcloud splits on commas)
  CORS_ESCAPED="${CORS_ALLOWED_ORIGINS//,/\,}"
  DEPLOY_ARGS+=(
    --set-env-vars "BACKEND_DATABASE_URL=$BACKEND_DATABASE_URL"
    --set-env-vars "RUN_DB_MIGRATIONS=$RUN_DB_MIGRATIONS"
    --set-env-vars "RUN_DB_MIGRATE_WAIT_SECS=$RUN_DB_MIGRATE_WAIT_SECS"
    --set-env-vars "CORS_ALLOWED_ORIGINS=$CORS_ESCAPED"
    --set-env-vars "LANGCHAIN_TRACING_V2=${LANGCHAIN_TRACING_V2:-true}"
    --set-env-vars "LANGCHAIN_PROJECT=${LANGCHAIN_PROJECT:-aieng-3-prod}"
  )
  # Optionally pass SMTP config (non-secret values) if present in shell env
  if [[ -n "${SMTP_HOST:-}" ]]; then DEPLOY_ARGS+=( --set-env-vars "SMTP_HOST=$SMTP_HOST" ); fi
  if [[ -n "${SMTP_PORT:-}" ]]; then DEPLOY_ARGS+=( --set-env-vars "SMTP_PORT=$SMTP_PORT" ); fi
  if [[ -n "${SMTP_USER:-}" ]]; then DEPLOY_ARGS+=( --set-env-vars "SMTP_USER=$SMTP_USER" ); fi
  if [[ -n "${MAIL_FROM:-}" ]]; then DEPLOY_ARGS+=( --set-env-vars "MAIL_FROM=$MAIL_FROM" ); fi
  if [[ -n "${SMTP_FROM_EMAIL:-}" ]]; then DEPLOY_ARGS+=( --set-env-vars "SMTP_FROM_EMAIL=$SMTP_FROM_EMAIL" ); fi
  if [[ -n "${SMTP_FROM_NAME:-}" ]]; then DEPLOY_ARGS+=( --set-env-vars "SMTP_FROM_NAME=$SMTP_FROM_NAME" ); fi
  if [[ -n "${SMTP_STARTTLS:-}" ]]; then DEPLOY_ARGS+=( --set-env-vars "SMTP_STARTTLS=$SMTP_STARTTLS" ); fi
fi

# Always attach secrets via Secret Manager
DEPLOY_ARGS+=(
  --update-secrets "OPENAI_API_KEY=$OPENAI_API_KEY_SM:latest"
  --update-secrets "LANGSMITH_API_KEY=$LANGSMITH_API_KEY_SM:latest"
  --update-secrets "NEXTAUTH_SECRET=$NEXTAUTH_SECRET_SM:latest"
  --update-secrets "SMTP_PASS=$SMTP_PASS_SM:latest"
)

# shellcheck disable=SC2068
gcloud run deploy "$SERVICE" ${DEPLOY_ARGS[@]}

URL=$(gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)')
echo "[deploy] Service URL: $URL"
echo "[deploy] Health: curl $URL/healthz"
