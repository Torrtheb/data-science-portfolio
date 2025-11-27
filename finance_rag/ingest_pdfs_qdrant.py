import os, glob, hashlib, sys, re
from pathlib import Path
from typing import List, Dict

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import uuid
import hashlib

# Load .env if present so required keys are available when running locally
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass

# ----------- CONFIG (env-driven) ----------------
DOCS_DIR = os.getenv(
    "DOCS_DIR", str(Path(__file__).resolve().parents[1] / "data" / "raw")
)
QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "finance_docs")
OVERWRITE_COLLECTION = os.getenv("OVERWRITE_COLLECTION", "0") not in (
    "0",
    "false",
    "False",
    "no",
    "No",
)

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
# Lightweight keyword extraction model (can be same as chat model)
OPENAI_KEYWORD_MODEL = os.getenv(
    "OPENAI_KEYWORD_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini")
)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
# Smaller batches → lower risk of HTTP timeouts to Qdrant; override via EMBED_BATCH
BATCH = int(os.getenv("EMBED_BATCH", "16"))
KEYWORDS_PER_CHUNK = int(os.getenv("RAG_KEYWORDS_PER_CHUNK", "8"))
# Turn off keyword extraction to speed up ingestion for large corpora (default: off)
ENABLE_KEYWORDS = os.getenv("ENABLE_KEYWORDS", "0") not in (
    "0",
    "false",
    "False",
    "no",
    "No",
)
# HTTP timeout (seconds) for Qdrant requests
QDRANT_TIMEOUT = float(os.getenv("QDRANT_TIMEOUT", "30"))
# Once a moderation/permission issue is hit, disable further keyword calls to avoid spamming 403s.
KEYWORDS_AVAILABLE = True


def resolve_dims(model: str) -> int:
    m = model.strip().lower()
    if "text-embedding-3-large" in m:
        return 3072
    if "text-embedding-3-small" in m:
        return 1536
    if "text-embedding-ada-002" in m:
        return 1536
    raise ValueError(f"Unknown embedding dim for model={model}")


def prettify_title(path: str) -> str:
    """
    Generate a human-friendly title from a file path.
    Falls back to the basename if we cannot prettify.
    """
    stem = Path(path).stem
    stem = stem.replace("_", " ").replace("-", " ")
    stem = re.sub(r"\s+", " ", stem).strip()
    stem = stem.replace("’", "'")
    return stem or os.path.basename(path)


def chunk_pdf(path: str) -> List[Dict]:
    reader = PdfReader(path)
    pages = [p.extract_text() or "" for p in reader.pages]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )
    chunks: List[Dict] = []
    abspath = os.path.abspath(path)
    title = prettify_title(path)
    filename = os.path.basename(path)
    url = f"file://{abspath}"
    for page_idx, txt in enumerate(pages, start=1):
        parts = splitter.split_text(txt)
        for chunk_idx, c in enumerate(parts):
            chunks.append(
                {
                    "page_content": c,
                    "metadata": {
                        "source": abspath,
                        "title": title,
                        "url": url,
                        "page": page_idx,
                        "chunk": chunk_idx,
                        "source_file": filename,
                    },
                }
            )
    return chunks


def embed_texts(oai: OpenAI, texts: List[str]) -> List[List[float]]:
    res = oai.embeddings.create(model=OPENAI_EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in res.data]


def extract_keywords(
    oai: OpenAI, text: str, *, max_keywords: int = KEYWORDS_PER_CHUNK
) -> List[str]:
    """
    Use an LLM to extract a small set of finance-relevant keywords for a chunk.

    Returns a list of short phrases (strings). On error, returns [].
    """
    global KEYWORDS_AVAILABLE
    if (not ENABLE_KEYWORDS) or (not KEYWORDS_AVAILABLE):
        return []
    txt = (text or "").strip()
    if not txt:
        return []

    prompt = (
        "Extract at most {k} concise finance-related keywords or short phrases "
        "from the following text. Focus on investing, markets, instruments, or "
        "personal finance concepts. Respond with a comma-separated list only, "
        "no explanations.\n\n"
        'Text:\n"""\n{body}\n"""'
    ).format(k=max_keywords, body=txt[:1200])

    try:
        resp = oai.chat.completions.create(
            model=OPENAI_KEYWORD_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a financial domain keyword extractor.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=64,
        )
        content = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        # If the provider rejects (e.g., 403 policy), turn off keywords for the rest of the run.
        KEYWORDS_AVAILABLE = False
        print(f"⚠️ keyword extraction disabled after error: {e}", file=sys.stderr)
        return []

    raw = content.replace("\n", ",")
    parts = [p.strip(" ,;-") for p in raw.split(",") if p.strip(" ,;-")]
    # De-duplicate while preserving order
    seen = set()
    out: List[str] = []
    for p in parts:
        pl = p.lower()
        if pl in seen:
            continue
        seen.add(pl)
        out.append(p)
        if len(out) >= max_keywords:
            break
    return out


def stable_uuid(md: dict, text: str) -> str:
    src = str((md or {}).get("source", ""))
    page = str((md or {}).get("page", ""))
    chk = str((md or {}).get("chunk", ""))
    head = (text or "")[:80]
    key = f"{src}|{page}|{chk}|{head}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def main():
    dims = resolve_dims(OPENAI_EMBEDDING_MODEL)
    print(f"🔧 model={OPENAI_EMBEDDING_MODEL} (dims={dims})")
    print(f"📂 DOCS_DIR={DOCS_DIR}")

    files = sorted(
        f
        for f in glob.glob(os.path.join(DOCS_DIR, "**/*.pdf"), recursive=True)
        if os.path.isfile(f)
    )
    if not files:
        print("⚠️ No PDF files found. Set DOCS_DIR or add documents.")
        return

    qd = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=QDRANT_TIMEOUT)
    if qd.collection_exists(QDRANT_COLLECTION):
        print(f"✅ Collection exists: {QDRANT_COLLECTION}")
        try:
            info = qd.get_collection(QDRANT_COLLECTION)
            vecs = (
                getattr(info.config.params, "vectors", None)
                if hasattr(info, "config")
                else None
            )
            size = None
            if vecs is None:
                vecs = (
                    info.get("config", {}).get("params", {}).get("vectors", {})
                    if isinstance(info, dict)
                    else None
                )
            if isinstance(vecs, dict):
                size = vecs.get("size")
            elif vecs is not None:
                size = getattr(vecs, "size", None)
            if size and size != dims:
                msg = (
                    f"Collection vector size={size} but embedding model dims={dims}. "
                    "Either set OPENAI_EMBEDDING_MODEL back to the original size, or drop/recreate the collection."
                )
                if OVERWRITE_COLLECTION:
                    print(
                        f"⚠️ {msg} Recreating collection due to OVERWRITE_COLLECTION=1 ..."
                    )
                    qd.delete_collection(QDRANT_COLLECTION)
                    qd.create_collection(
                        collection_name=QDRANT_COLLECTION,
                        vectors_config=VectorParams(
                            size=dims, distance=Distance.COSINE
                        ),
                    )
                else:
                    raise SystemExit(
                        f"ERROR: {msg} Set OVERWRITE_COLLECTION=1 to recreate."
                    )
        except SystemExit:
            raise
        except Exception as e:
            print(f"⚠️ Could not verify collection vector size: {e}", file=sys.stderr)
    else:
        print(f"🆕 Creating collection: {QDRANT_COLLECTION}")
        qd.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=dims, distance=Distance.COSINE),
        )

    oai = OpenAI(api_key=OPENAI_API_KEY)

    total = 0
    for path in files:
        chunks = chunk_pdf(path)
        print(f"• {os.path.basename(path)} → {len(chunks)} chunks")
        total += len(chunks)
        for i in range(0, len(chunks), BATCH):
            batch = chunks[i : i + BATCH]
            texts = [b["page_content"] for b in batch]
            vecs = embed_texts(oai, texts)
            points = []
            for j in range(len(batch)):
                pc = (
                    batch[j]["page_content"]
                    if "page_content" in batch[j]
                    else batch[j]["text"]
                )
                md = (
                    batch[j]["metadata"]
                    if "metadata" in batch[j]
                    else {
                        "source": batch[j].get("source"),
                        "title": batch[j].get("title"),
                        "url": batch[j].get("url"),
                        "page": batch[j].get("page"),
                        "chunk": j,
                    }
                )
                # Enrich metadata with LLM-extracted keywords for better retrieval filtering.
                kws = []
                if ENABLE_KEYWORDS:
                    try:
                        kws = extract_keywords(oai, pc)
                    except Exception:
                        kws = []
                if kws:
                    md = {**md, "keywords": kws}
                points.append(
                    PointStruct(
                        id=stable_uuid(md, pc),
                        vector=vecs[j],
                        payload={
                            "page_content": pc,
                            "metadata": md,
                        },
                    )
                )

            qd.upsert(collection_name=QDRANT_COLLECTION, points=points)

    try:
        count = qd.count(QDRANT_COLLECTION, exact=True).count
    except Exception:
        count = None
    print(f"✅ Done. Upserted {total} chunks into '{QDRANT_COLLECTION}'. Count≈{count}")


if __name__ == "__main__":
    for k in ("OPENAI_API_KEY", "QDRANT_URL", "QDRANT_API_KEY"):
        if not os.getenv(k):
            print(f"Missing env: {k}", file=sys.stderr)
            sys.exit(2)
    main()
