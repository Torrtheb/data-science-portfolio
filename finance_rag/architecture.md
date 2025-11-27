# Architecture Overview

Concise view of how the frontend, backend, vector store, and tools fit together.

## System diagram

```mermaid
flowchart LR
  subgraph Client
    UI[Next.js UI<br/>Chat + sources drawer]
  end

  subgraph Frontend
    FE[Next.js / React<br/>App Router]
  end

  subgraph Backend
    API[FastAPI<br/>/api/chat, /api/chat/stream]
    Guard[Guards<br/>scope, size, rate limit]
    Agent[LangChain Agent<br/>prompts + tools]
  end

  subgraph Data
    Qdrant[(Qdrant<br/>vectors + metadata)]
    DB[(SQLite)]
    Logs[(Log files)]
  end

  subgraph External
    OpenAI[(OpenAI API)]
    Finnhub[(Finnhub API)]
    Twelve[(TwelveData API)]
    WorldBank[(World Bank MCP)]
  end

  UI -->|HTTP / SSE| FE --> API
  API --> Guard --> Agent
  Agent --> Qdrant
  Agent --> OpenAI
  Agent --> Finnhub
  Agent --> Twelve
  Agent --> WorldBank
  Agent --> DB
  API --> Logs
```

## Request flow (chat)
1) Frontend calls `/api/chat` or `/api/chat/stream` with user text and session.
2) Backend guards the request (finance domain, length limits, rate limits).
3) Agent builds a hybrid retriever (Qdrant) + tool list (market data, calculators, World Bank).
4) OpenAI chat model generates with strict citation rails; tools run when invoked.
5) Response streams back with answer, citations, and usage.

## Ingestion flow (docs → Qdrant)
```mermaid
flowchart TD
  Raw[PDFs in data/raw] --> Ingest[ingest_pdfs_qdrant.py]
  Ingest --> Split[Chunk + embed]
  Split --> Qdrant[(Qdrant collection)]
```

## Environments & secrets
- `.env` holds OpenAI, Finnhub, TwelveData, Qdrant, and MCP settings.
- `.env.local` for frontend base URL.


## Deployment notes
- Frontend: Vercel (Next.js).
- Backend: Cloud Run (FastAPI). Ensure CORS matches frontend domain; keep MCP disabled if not configured.
- Persist Qdrant externally or snapshot/import for consistent vectors across deploys.
