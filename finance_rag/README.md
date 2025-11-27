# Finance Assistant RAG Chatbot (FinAssist)

FinAssist is a financial assistant chatbot that combines RAG over curated investing books, real‑time market data, and calculator tools.  
The backend is FastAPI + LangChain, the frontend is Next.js/React, and Qdrant is used as the vector store (local and prod).

## Live Demo

- Deployed app: https://front-app-nine.vercel.app/

The demo supports both standard HTTP responses and streaming answers with live token updates.


## Features 
This app: 
- Uses Retrieval-Augmented Generation (RAG) to correctly answer user's financial questions using an investment-focused database.
- Has a sources drawer which aggregates citations across the entire session.
- Uses a Qdrant vector store with OpenAI embeddings for all document chunks. 
- Provides real-time US market insights such as company news, profile, stock price, or analyst trends via Finnhub API, and graphs from TwelveData API. 
- Uses financial calculation tools: simple or compound interest, compound annual growth rate, loan amortization, net present value, or investment return. 
- Uses a world bank MCP tool to obtain inflation, unemployment rate, or Gross Domestic Product per person, adjusted for Purchasing Power Parity (GDP, PPP constant). 
- Exports chat history as JSON, PDF (WeasyPrint), or CSV from the UI. 

- **Safety and observability**
  - Finance‑only domain guardrail: non‑finance questions are refused with a clear, friendly message.
  - Additional guard when there are no useful docs/tools, to avoid hallucinated answers. If a query has no confident RAG matches and no applicable tools, the backend refuses with a clear message. 
  - Token accounting and cost estimation for each turn.


## Architecture

```text
         Frontend (Next.js)
                  │  HTTP (JSON / SSE)
                  ▼
            FastAPI (main.py)
        ┌──────────┼───────────┐
        │          │           │
  Pydantic models  │     Security / CORS / Rate‑limit
     (models.py)   │          (middlewares)
                   ▼
             Chat + RAG API
                   │
                   ▼
         LangChain + Agent (rag.py)
        ┌──────────┴──────────┐
        │                     │
   Qdrant vector store         Tooling & Calculators
         (rag.py)              (tools.py, agent_tools.py)
                   ▲
          Prompts, policies (prompts.py)
                   ▲
          Settings & secrets (settings.py, .env)
```

**Flow**  
- Frontend calls `/api/chat` (JSON) or `/api/chat/stream` (SSE) in `back_app/main.py`.
- Backend:
  - Validates and guards the query (finance scope, size limits, rate limits).
  - Uses the hybrid retriever to fetch relevant documents and tools for live data.
  - Calls the LangChain agent to generate an answer, with strict citation rails.
  - Logs usage and guardrails, and returns answer + sources + usage to the UI.

## Tech stack

- Python 3.13
- FastAPI for the HTTP API
- LangChain for RAG and tool orchestration
- Qdrant as the vector store
- OpenAI for chat (`gpt-4o-mini`) and embeddings (`text-embedding-3-small`)
- Finnhub and TwelveData for market data/charts
- World Bank MCP server for macro indicators
- Next.js/React + Tailwind for the frontend
- Pydantic + pydantic‑settings for configuration
- loguru for logging, tiktoken for token estimation
- WeasyPrint for PDF export
- Deployment: Vercel (frontend) + Google Cloud Run (backend)
- Testing with pytest
- [Architecture](./architecture.md)

## Running locally

1. **Clone the repository**
   - `git clone <repo>`  
   - `cd data-science-portfolio/finance_rag`

2. **Backend setup**
   - Create and activate a virtualenv.
   - Install deps: `pip install -r requirements.txt`
   - Create `.env` in the repo root with (at minimum):
     - `OPENAI_API_KEY`
     - `FINNHUB_API_KEY`
     - `TWELVEDATA_API_KEY`
     - `QDRANT_URL`
     - `QDRANT_API_KEY`
   - Optional RAG ingestion:
     - Place PDFs under `data/raw` or set `DOCS_DIR`.
     - Run: `python ingest_pdfs_qdrant.py` to populate Qdrant with chunks + keywords.
   - Start the API:
     - `cd back_app && uvicorn main:app --reload`

3. **Frontend setup**
   - `cd front_app && npm install`
   - Create `.env.local` with:
     - `NEXT_PUBLIC_BACKEND_URL=http://localhost:8000`
   - Run: `npm run dev` and open `http://localhost:3000`

4. **(Optional) MCP / World Bank tool**
   - From `back_app`, run:  
     `mcp-streamablehttp-proxy --stdio "world-bank-mcp-server" --port 8077`

## Testing

- Backend: `cd back_app && pytest`
- Frontend: `cd front_app && npm test` (add/adjust tests as needed)
- Notes: backend tests cover RAG helpers, citation guards, and tool/canonicalization logic; use `pytest -k <pattern>` to focus. Add CI badge here if/when Actions are enabled.

## Key improvements: 

Project improvements include: 
- Improving answer latency (current answers take too long to be returned to the user). 
- Integration of tool usage and llm answers, for a user to ask a question and use a tool simultaneously.
- Dedicated analytics page.
- Larger and more diverse knowledge base.
- Source tightening to include page numbers with clickable links to source in llm answer.
- Multi model support. 


## Acknowledgements

- OpenAI: https://platform.openai.com/docs/overview
- MCP World Bank server: https://github.com/anshumax/world_bank_mcp_server, https://pypi.org/project/world-bank-mcp-server/
- Finnhub api: https://finnhub.io
- TwelveData api: https://twelvedata.com

### Knowledge base:
  - Aliche, Tiffany. Get Good with Money : Ten Simple Steps to Becoming Financially Whole. New York, Rodale, 2021.
  - Arnold, G., and S. Kyle. Intermediate Financial Accounting. Calgary, Alberta, Canada, Lyres Learning Inc., 15 Dec. 2020.
  - Bogle, John C. Common Sense on Mutual Funds. Hoboken, N.J., Wiley, 2010
  - Fisher, Philip A. Common Stocks and Uncommon Profits and Other Writings. New York ; Chichester, Wiley, 2003.
  - Graham, Benjamin. The Intelligent Investor. New York, Harper, 1973
  - Graham, Benjamin, and David L Dodd. Security Analysis : Principles and Technique. 1934. New York, Mcgraw-Hill, 2009.
  - Hill, Napoleon. Think and Grow Rich. S.D. Classic Good Books, 2007
  - Lancaster, Marcus P. The Psychology of Money. eBookIt.com, 22 Jan. 2025, dokumen.pub. Accessed 18 Aug. 2025.
  - Larimore, Taylor, et al. The Bogleheads’ Guide to Investing. Hoboken, N.J, Wiley, 2014.
    ---. The Bogleheads’ Guide to Retirement Planning. John Wiley & Sons, 24 Sept. 2009.
  - Lowry, Erin. Broke Millennial. Penguin, 2 May 2017.
    ---. Broke Millennial Takes on Investing : A Beginner’s Guide to Leveling up Your Money. New York, A Tarcherperigee Book, 2019.
  - Lynch, Peter, and John Rothchild. One up on Wall Street : How to Use What You Already Know to Make Money in the Market. Norwalk, Conn., Easton Press, 2000.
  - Malkiel, Burton G. A Random Walk down Wall Street. New York, W.W. Norton & Company, 1973.
  - Mandelbrot, Benoit B., and Richard L. Hudson. The (Mis)Behavior of Markets : A Fractal View of Risk, Ruin, and Reward. London, Profile, 2008.
  - Pabrai, Mohnish. The Dhandho Investor : The Low-Risk Value Method to High Returns. Hoboken, N.J., John Wiley, 2007.
  - Munger, Charles T. Poor Charlie’s Almanack. The Donning Company Publishers, 2005.
  - Nassim, Nicholas Taleb. Fooled by Randomness : The Hidden Role of Chance in Life and in the Markets. New York, Random House, 2016.
  - Ramsey, Dave. The Total Money Makeover. Nashville, Tn, Nelson Current, 2013.
  - Watson, Richard Thomas. Electronic Commerce : The Strategic Perspective. Fla., Orange Grove Texts plus, 2008.
  - Sethi, Ramit . I Will Teach You to Be Rich. Workman Pub., 2009, pdfcoffee.com . Accessed 18 Aug. 2025.
  - Shiller, Robert J. Irrational Exuberance. Princeton, Princeton University Press, 2015.

