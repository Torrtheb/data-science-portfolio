# AI Scheduling Assistant

This project is a full-stack AI scheduling assistant for a small music studio. It combines a FastAPI + Postgres/pgvector backend with a Next.js frontend and a LangGraph agent that can propose, book, and manage appointments, handle credit wallets, and draft emails, while keeping a human owner in control.


## Overview
- Full‑stack AI scheduling assistant for a small music studio business.
- Backend: FastAPI (appointments, availability, wallets, admin fees, email drafts, AI agent).
- Frontend: Next.js App Router (owner dashboard, client portal, agent chat, email approval).
- Live frontend: https://music-studio-rho.vercel.app (Vercel).
- Demo login (owner): email `owner@example.com`, password `demo-pass-123`.


## Highlights
- Implements availability rules, special openings, time-off blocks, and conflict-aware booking/rescheduling/cancelling flows.
- Manages client credit bundles, including auto-applying credits to appointments/admin fees and handling refunds.
- Uses a LangGraph-based agent to coordinate tools for slot finding, pricing, wallet operations, analytics, and email drafting.
- Provides guarded SSE chat with retries, JWT authentication, and per-owner rate limiting to keep conversations safe and reliable.


## Repo Layout
- /backend — FastAPI service, Alembic migrations, agent toolchain.
- /my-app — Next.js (NextAuth, Prisma for auth DB, proxy to backend).
- /docker-compose.yml — local Postgres (pgvector) + Adminer.
- See architecture.md for system overview. 

Prerequisites
- Python 3.11+
- Node 18+
- Postgres 14+ with pgvector (local via Docker recommended).


## Quickstart (Local Dev)
1) Database (from repo root): 'docker compose up -d db'
2) Backend setup:
   - Copy /backend/.env.example → /backend/.env and edit.
   - In dev you may set AUTH_DISABLED=1 and DEV_FAKE_OWNER_ID.
   - From /backend: alembic upgrade head
   - Start API: uvicorn main:app --reload
3) Frontend setup:
   - Edit /my-app/.env.local:
     - NEXTAUTH_SECRET (must match backend)
     - NEXTAUTH_URL (e.g., http://localhost:3000)
     - BACKEND_URL (e.g., http://localhost:8000)
     - FRONTEND_DATABASE_URL (Postgres DSN for NextAuth)
   - From /my-app: npm install && npm run dev
4) Open http://localhost:3000 and sign in. Owner dashboard exposes agent chat and admin panels.

Key Concepts
- Auth: NextAuth (credentials or Google). Frontend signs a short‑lived HS256 JWT with 'NEXTAUTH_SECRET' and proxies API calls to the backend.
- CORS: Backend must allow the frontend origin. In production, restrict to Vercel domain.
- SSE: Agent chat uses stream responses; ensure proxies/load balancers support streaming.


### Demo data seeding 
- Seed script: `cd backend && python scripts/seed_demo.py`
  - Creates an owner (`owner-demo-1`), a client (`client-demo-1`), one client account/person/email, a special opening (next Friday 9–12 local), and a time-off block (next Wednesday 13–15 local).
  - Idempotent; safe to rerun.
- Customizable IDs/emails inside `backend/scripts/seed_demo.py`.

### Demo scripts
- “Book client tomorrow at 10am for 60 minutes” → find slots → book_appointment → confirmation.
- “Block off next Friday from 9am to 5pm” → add_time_off → calendar updates.
- “What does client owe me?” → financial dashboard / payments summary.
- “Draft an email to client saying I’ll be 10 minutes late” → UI:EMAIL_DRAFT card, then approve/send from Outbox.


## Testing
- Backend tests live in `backend/tests`. Example: `cd backend && pytest tests/test_agent_chat.py tests/test_services/test_wallets.py`.
- Chat tests cover UUID validation and rate limiting; wallet tests cover auto-apply flows.
- Health (local): `curl -s http://localhost:8000/healthz` should return `{"status":"ok"}`


## Future work
- Global rate limiting with Redis/Upstash so limits hold across scaled Cloud Run instances.
- Sandbox SMTP (e.g., Mailtrap) for safe outbound email demos.
- Richer seeded data: past/future appointments, payments, wallet movements to showcase dashboards.
- Observability: structured logs + basic traces/metrics for easier debugging.
- Optional OAuth (Google) to show a smoother demo login alongside the seeded owner.
- Enhancing usability, booking logic, llm bugs, and user experience for specific use cases. 
