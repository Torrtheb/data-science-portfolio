from __future__ import annotations
import os, glob, time, re, asyncio
from functools import lru_cache
from typing import List, Dict, Any, Optional, Callable, TypeVar, Set, Tuple
from loguru import logger
from rank_bm25 import BM25Okapi
from pydantic import PrivateAttr
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

from langchain.chains import RetrievalQA
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.runnables import RunnableConfig
from qdrant_client import QdrantClient
try:
    from langchain_qdrant import Qdrant as LCQdrant
except Exception:
    from langchain_community.vectorstores import Qdrant as LCQdrant  # fallback
from qdrant_client import QdrantClient
from ..core.settings import settings
from .prompts import (
    QA_SYSTEM,
    QA_USER_TEMPLATE,
    ANSWER_WITH_CITATIONS,
    INJECTION_GUARD,
    AGENT_SYSTEM,
    STRICT_QA_USER_TEMPLATE,
)

from ..utils.citations import dedupe_sources, SourceEntry
from ..analytics.tokenlog import with_token_log


# ---- Config ----------------------------------------------------------
FINNHUB_RPS = getattr(settings, "finnhub_rps", int(os.getenv("FINNHUB_RPS", "20")))
FINNHUB_PERIOD = getattr(
    settings, "finnhub_period", float(os.getenv("FINNHUB_PERIOD", "1"))
)
EXTERNAL_TIMEOUT = getattr(
    settings, "external_timeout", float(os.getenv("EXTERNAL_TIMEOUT", "15"))
)
REPO_ROOT = Path(__file__).resolve().parents[1]
JUNK: Set[str] = {".DS_Store", ".gitkeep", ".keep"}
DOCS_DIR = os.getenv("DOCS_DIR", str(REPO_ROOT / "data" / "raw"))
CHUNK_SIZE = int(getattr(settings, "rag_chunk_size", 800))
CHUNK_OVERLAP = int(getattr(settings, "rag_chunk_overlap", 120))
TOP_K = int(getattr(settings, "rag_top_k", 4))
T = TypeVar("T")
ALLOWED_EXTS_DEFAULT: Set[str] = {
    ".pdf",
    ".txt",
    ".md",
    ".html",
    ".htm",
    ".docx",
    ".pptx",
    ".xlsx",
    ".csv",
    ".json",
    ".yaml",
    ".yml",
}
# --- vector backend toggle ---
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "finance_docs")
# Default to Qdrant; VECTORSTORE env is kept only for backward compatibility.
VECTORSTORE = os.getenv("VECTORSTORE", "qdrant")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# ---- LLMs / Embeddings ----------------------------------------------
_embeddings_singleton: Optional[OpenAIEmbeddings] = None
_llm_singleton: Optional[BaseChatModel] = None


def _resolve_openai_key() -> Optional[str]:
    """
    Resolve an OpenAI API key without side effects.

    Order of precedence:
        1) settings.openai_api_key (if present, pulled via get_secret_value)
        2) Environment variable OPENAI_API_KEY
        3) None (caller must handle)

    Returns:
        The API key string or None if not available.
    """
    try:
        if getattr(settings, "openai_api_key", None):
            return settings.openai_api_key.get_secret_value()
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY")


def get_embeddings() -> OpenAIEmbeddings:
    """
    Return a process-wide singleton for OpenAIEmbeddings (lazy-initialized).

    Model resolution:
        - settings.embedding_model if present
        - env OPENAI_EMBEDDING_MODEL
        - default "text-embedding-3-small"

    Returns:
        OpenAIEmbeddings: A shared embeddings instance.
    """
    global _embeddings_singleton
    if _embeddings_singleton is None:
        model_name = None
        try:
            model_name = getattr(settings, "embedding_model", None)
        except Exception:
            pass
        model_name = model_name or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        _embeddings_singleton = OpenAIEmbeddings(
            model=model_name,
            api_key=_resolve_openai_key(),
        )
        logger.info("Embeddings ready | model=%s", model_name)
    return _embeddings_singleton



def get_llm() -> BaseChatModel:
    """
    Return a process-wide ChatOpenAI (LangChain) instance (lazy-initialized).

    Uses:
        - settings.openai_model for the model name.
        - temperature=0.1, max_retries=3, request_timeout=60
        - API key from _resolve_openai_key()

    Returns:
        BaseChatModel: A LangChain-compatible chat model.
    """
    global _llm_singleton
    if _llm_singleton is None:
        _llm_singleton = ChatOpenAI(
            model=settings.openai_model,
            temperature=0.1,
            max_retries=3,
            request_timeout=60,
            api_key=_resolve_openai_key(),
        )
    return _llm_singleton


def _model_name_for_logging(llm: BaseChatModel) -> str:
    """
    Extract a friendly model name for analytics logs.

    Args:
        llm: Any LangChain chat model.

    Returns:
        str: Best-effort model name (falls back to "openai").
    """
    return getattr(llm, "model", None) or getattr(llm, "model_name", "openai")


# ---- Token estimation for RAG embedding & prompts ---------------------
try:
    import tiktoken
except Exception:
    tiktoken = None

_enc_cache: Dict[str, Any] = {}


def _count_tokens_text(text: str, model: str) -> int:
    """
    Estimate tokens for observability (embeddings/prompt previews).

    Strategy:
        - Try tiktoken.encoding_for_model(model)
        - Fallback to "cl100k_base"
        - Final fallback: len(text) / 4 heuristic

    Args:
        text: Input string.
        model: Model name used to pick encoding.

    Returns:
        Estimated token count (int, >= 0).
    """
    if not text:
        return 0
    if tiktoken:
        try:
            enc = _enc_cache.get(model) or tiktoken.encoding_for_model(model)
            _enc_cache[model] = enc
        except Exception:
            enc = _enc_cache.get("cl100k_base") or tiktoken.get_encoding("cl100k_base")
            _enc_cache["cl100k_base"] = enc
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    return max(1, int(len(text) / 4))


# ---- Utilities -------------------------------------------------------


def _tag_retrieval_docs(
    docs: List[Document],
) -> Tuple[List[Document], List[SourceEntry]]:
    """
    Add stable citation tags to retrieved documents and build source metadata.

    Each input Document receives:
        - A synthetic header line: "[S{i}] {title}" prepended to page_content.
        - metadata["sid"] = "S{i}"

    SourceEntry list is built to power UI citations.

    Args:
        docs: Raw retrieved documents.

    Returns:
        (tagged_docs, rag_sources):
            - tagged_docs: Documents with header + "sid" in metadata.
            - rag_sources: List[SourceEntry] for UI/source panels.
    """
    tagged: List[Document] = []
    sources: List[SourceEntry] = []
    for i, d in enumerate(docs, 1):
        sid = f"S{i}"
        title = (
            (d.metadata.get("title") if isinstance(d.metadata, dict) else None)
            or (d.metadata.get("source") if isinstance(d.metadata, dict) else None)
            or "Source"
        )
        href = None
        if isinstance(d.metadata, dict):
            href = (
                d.metadata.get("url")
                or d.metadata.get("href")
                or d.metadata.get("source")
            )
        head = f"[{sid}] {title}\n"
        tagged.append(
            Document(
                page_content=f"{head}{d.page_content}",
                metadata={**(d.metadata or {}), "sid": sid},
            )
        )
        sources.append(
            SourceEntry(
                id=sid,
                display=str(title),
                href=href,
                meta={
                    "kind": "retrieval",
                    **{
                        k: v
                        for k, v in (d.metadata or {}).items()
                        if k != "page_content"
                    },
                },
                snippet=(d.page_content[:300] + "…") if d.page_content else None,
            )
        )
    return tagged, sources


def docs_to_sources(docs) -> List[Dict]:
    """
    Convert LangChain Document objects to normalized source dicts for the UI.

    Preference order:
        1) File-like paths (so citation de-dup can map to local PDFs/books)
        2) {url, title}
        3) Entire metadata as fallback

    Env:
        RAG_LOG_SOURCES=1 → emit source composition details to logs.

    Args:
        docs: Iterable of Document-like objects.

    Returns:
        List[dict]: Normalized, deduplicated sources ready for rendering.
    """
    raw: List[Dict | str] = []
    for d in docs or []:
        meta = getattr(d, "metadata", {}) or {}
        f = meta.get("source") or meta.get("file") or meta.get("path")
        if isinstance(f, str) and f.strip():
            raw.append(f.strip())
            continue

        url = meta.get("url")
        title = meta.get("title")
        if url or title:
            raw.append({"url": url, "title": title})
            continue

        if meta:
            raw.append(meta)

    sources = dedupe_sources(raw)
    if os.getenv("RAG_LOG_SOURCES", "") == "1":
        books = [s for s in sources if (s.get("type") == "book")]
        webs = [s for s in sources if (s.get("type") == "web")]
        logger.info(
            "Source types → books={} web={} other={}",
            len(books),
            len(webs),
            len(sources) - len(books) - len(webs),
        )
        unknown_books = [
            s for s in books if not s.get("display") or s["display"].endswith(".pdf")
        ]
        for ub in unknown_books:
            logger.warning(
                "Book source did not map to citation (showing filename): {}",
                ub.get("display"),
            )

    return sources


def retry_with_backoff(
    func: Callable[[], T], max_retries: int = 3, backoff_factor: int = 1
) -> T:
    """
    Execute a callable with exponential backoff on failure.

    Wait schedule:
        wait = backoff_factor * (2 ** attempt)
        where attempt starts at 0.

    Args:
        func: No-arg callable to invoke.
        max_retries: Max attempts before re-raising the last exception.
        backoff_factor: Base wait in seconds.

    Returns:
        The callable's return value if a retry succeeds.

    Raises:
        Exception: The final exception after exhausting retries.
    """
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = backoff_factor * (2**attempt)
            print(f"⚠️ Attempt {attempt+1} failed, retrying in {wait}s: {e}")
            time.sleep(wait)


def _is_hidden(filename: str) -> bool:
    """
    Check whether a filename is "hidden" (starts with '.').

    Args:
        filename: Basename or path component.

    Returns:
        True if hidden; False otherwise.
    """
    return os.path.basename(filename).startswith(".")


def _load_docs_from_path(
    path: str,
    *,
    allowed_exts: Optional[Set[str]] = None,
    max_bytes: int = 50 * 1024 * 1024,
    skip_hidden: bool = True,
    junk: Optional[Set[str]] = None,
    follow_symlinks: bool = False,
    use_text_fallback: bool = True,
) -> List[Document]:
    """
    Recursively load files under a directory into LangChain Documents with guards.

    Features:
        - Validates path existence/type
        - Skips junk/hidden files
        - Extension allowlist (avoid unwanted binaries)
        - Filesize guard
        - Robust loader selection with text fallback
        - Adds metadata: source, title, url (file://), chunk index

    Args:
        path: Root folder to scan (recursive).
        allowed_exts: Lowercased file extensions to include. None → default allowlist;
            empty set → allow all.
        max_bytes: Skip files larger than this many bytes (default 50MB).
        skip_hidden: Skip dotfiles/directories if True.
        junk: Extra explicit filenames to skip (e.g., ".DS_Store"); defaults to JUNK.
        follow_symlinks: Follow symbolic links during traversal.
        use_text_fallback: Use TextLoader when UnstructuredFileLoader is unavailable.

    Returns:
        List[Document]: Flattened list of loaded chunk documents.
    """
    docs: List[Document] = []

    junk = JUNK if junk is None else junk
    if allowed_exts is None:
        allowed_exts = ALLOWED_EXTS_DEFAULT
    allow_all = isinstance(allowed_exts, set) and len(allowed_exts) == 0

    if not os.path.exists(path):
        logger.warning("Documents path does not exist: {}", path)
        return []
    if not os.path.isdir(path):
        logger.warning("Documents path is not a directory: {}", path)
        return []

    logger.info("Scanning documents under: {}", os.path.abspath(path))

    for root, dirnames, filenames in os.walk(path, followlinks=follow_symlinks):
        if skip_hidden:
            dirnames[:] = [d for d in dirnames if not _is_hidden(d)]

        for name in filenames:
            if name in junk:
                continue
            if skip_hidden and _is_hidden(name):
                continue

            f = os.path.join(root, name)

            ext = os.path.splitext(name)[1].lower()
            if not (allow_all or ext in allowed_exts or ext == ".pdf"):
                logger.debug("Skipping disallowed extension: {}", f)
                continue
            try:
                size = os.path.getsize(f)
                if size > max_bytes:
                    logger.warning(
                        "Skipping large file ({} bytes > {}): {}", size, max_bytes, f
                    )
                    continue
            except OSError as e:
                logger.error("Could not get size for {}: {}", f, e)
                continue
            abs_path = os.path.abspath(f)
            try:
                if ext == ".pdf":
                    from langchain_community.document_loaders import PyPDFLoader

                    loader = PyPDFLoader(abs_path)
                else:
                    try:
                        from langchain_community.document_loaders import (
                            UnstructuredFileLoader,
                        )

                        loader = UnstructuredFileLoader(abs_path)
                    except Exception as UE:
                        if not use_text_fallback:
                            logger.error(
                                "Unstructured loader unavailable and text fallback disabled for {}: {}",
                                abs_path,
                                UE,
                            )
                            continue
                        from langchain_community.document_loaders import TextLoader

                        loader = TextLoader(abs_path, encoding="utf-8")
                file_docs = loader.load()
                for i, d in enumerate(file_docs):
                    d.metadata.update(
                        {
                            "source": abs_path,
                            "title": os.path.basename(abs_path),
                            "url": f"file://{abs_path}",
                            "chunk": i,
                        }
                    )
                docs.extend(file_docs)
                logger.info("Loaded {} chunk(s) from {}", len(file_docs), name)

            except Exception as e:
                logger.error("Failed to load {}: {}", name, e)

    logger.info("Finished scanning. Total chunks loaded: {}", len(docs))
    return docs


def _chunk(docs: List[Document]) -> List[Document]:
    """
    Split documents into overlapping chunks tuned for RAG.

    Uses RecursiveCharacterTextSplitter with:
        - chunk_size = CHUNK_SIZE
        - chunk_overlap = CHUNK_OVERLAP
        - separators = ["\\n\\n", "\\n", " ", ""]

    Args:
        docs: Input documents.

    Returns:
        List[Document]: Chunked documents (may be empty).
    """
    if not docs:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )
    return splitter.split_documents(docs)


def _guess_qdrant_payload_keys(qd: "QdrantClient", collection: str) -> tuple[str, str]:
    """
    Guess payload keys for text and metadata in a Qdrant collection.

    Precedence:
        1) Env overrides: QDRANT_CONTENT_KEY / QDRANT_METADATA_KEY
        2) Inspect one point from the collection for common conventions:
           content ∈ {"page_content","text","content","document"}
           metadata ∈ {"metadata","meta"}
        3) Defaults to ("page_content", "metadata")

    Args:
        qd: QdrantClient bound to the target cluster.
        collection: Name of the Qdrant collection.

    Returns:
        (content_key, metadata_key) strings.
    """
    ck_env = os.getenv("QDRANT_CONTENT_KEY")
    mk_env = os.getenv("QDRANT_METADATA_KEY")
    if ck_env and mk_env:
        return ck_env, mk_env

    try:
        pts, _ = qd.scroll(collection_name=collection, limit=1)
        if pts:
            keys = set(pts[0].payload.keys())
            for cand in ("page_content", "text", "content", "document"):
                if cand in keys:
                    content_key = cand
                    break
            else:
                content_key = "page_content"

            for cand in ("metadata", "meta"):
                if cand in keys:
                    meta_key = cand
                    break
            else:
                meta_key = "metadata"
            return ck_env or content_key, mk_env or meta_key
    except Exception:
        pass

    return ck_env or "page_content", mk_env or "metadata"


def build_vectorstore_qdrant(
    collection_name: str | None = None,
    docs_dir: str | None = None,
) -> "LCQdrant":
    """
    Create a LangChain Qdrant vector store wrapper, using existing points.

    Reads QDRANT_URL/QDRANT_API_KEY from env and inspects the collection
    to auto-detect content/metadata payload keys (with env overrides).

    Args:
        collection_name: Qdrant collection to use (default: env QDRANT_COLLECTION).
        docs_dir: Unused here (kept for symmetry with build_vectorstore).

    Returns:
        LCQdrant: A LangChain vector store bound to the named collection.

    Raises:
        RuntimeError: If QDRANT_URL/API_KEY are not set.
    """
    collection_name = collection_name or QDRANT_COLLECTION
    rag_log = logger.bind(component="RAG", collection=collection_name)

    if not (QDRANT_URL and QDRANT_API_KEY):
        raise RuntimeError("QDRANT_URL/QDRANT_API_KEY not set")

    qd = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    content_key, meta_key = _guess_qdrant_payload_keys(qd, collection_name)
    rag_log.info("Qdrant payload keys | content='%s' metadata='%s'", content_key, meta_key)

    vs = LCQdrant(
        client=qd,
        collection_name=collection_name,
        embeddings=get_embeddings(),
        content_payload_key=content_key,
        metadata_payload_key=meta_key,
    )
    rag_log.info("Qdrant vector store ready | collection='%s'", collection_name)
    return vs



def build_functions_agent(
    llm: BaseChatModel,
    tools: List,
    debug: Optional[bool] = None,
) -> AgentExecutor:
    """
    Build an OpenAI Tools agent with system rails and scratchpad support.

    Use this for tasks where tool usage is preferable (pricing, calculators, etc.),
    as the agent will call tools rather than guess answers.

    Args:
        llm: LangChain chat model to drive the agent.
        tools: List of tool definitions (LangChain tools).
        debug: If None, read AGENT_DEBUG env; otherwise force True/False.

    Returns:
        AgentExecutor: Configured agent executor.
    """
    if debug is None:
        resolved_debug = bool(int(os.getenv("AGENT_DEBUG", "1")))
    else:
        resolved_debug = bool(debug)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", INJECTION_GUARD),
            ("system", AGENT_SYSTEM),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_openai_tools_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=resolved_debug,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        max_iterations=8,
        early_stopping_method="generate",
    )


# ---- Hybrid retriever (BaseRetriever) --------------------------------
def _tok(s: str) -> list[str]:
    """
    Lightweight tokenizer for BM25.

    Splits on alphanumeric word chunks in lowercase to be language-agnostic.

    Args:
        s: Input string.

    Returns:
        List[str]: Token list.
    """
    return re.findall(r"\w+", (s or "").lower())


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever: fuse dense (vector store) and sparse (BM25) results using RRF.

    Safe on empty vector stores: if no documents are present, BM25 is disabled
    and only dense leg (if any) contributes.
    """
    vs: Any
    k: int = TOP_K

    _docs: list[str] = PrivateAttr(default_factory=list)
    _metas: list[dict] = PrivateAttr(default_factory=list)
    _bm25: Any = PrivateAttr(default=None)

    def model_post_init(self, __ctx):
        """
        Build a BM25 index from the underlying store if document text is available.

        Attempts to pull raw documents+metadata from Chroma's private handle
        (vs._collection) and constructs BM25Okapi(corpus). If unavailable, BM25 remains None.
        """

        raw = {}
        try:
            raw = self.vs._collection.get(include=["documents", "metadatas"]) or {}
        except Exception:
            raw = {}
        try:
            self._docs = raw.get("documents") or []
            self._metas = raw.get("metadatas") or []
        except Exception:
            self._docs, self._metas = [], []

        try:
            if self._docs:
                self._bm25 = BM25Okapi([_tok(doc) for doc in self._docs])
            else:
                self._bm25 = None
        except Exception:
            self._bm25 = None

    def _bm25_topk(self, q: str, k: int) -> List[Document]:
        """
        Retrieve top-k documents via BM25.

        Args:
            q: Query string.
            k: Number of results to return.

        Returns:
            List[Document]: Documents annotated with {"bm25_score": float}.
        """

        try:
            bm = getattr(self, "_bm25", None)
        except Exception:
            bm = None
        if not bm:
            return []
        import numpy as np

        try:
            scores = bm.get_scores(_tok(q))
        except Exception:
            return []
        if len(scores) == 0:
            return []
        idx = np.argsort(scores)[-k:][::-1]
        out: List[Document] = []
        for i in idx:
            meta = {**(self._metas[i] or {}), "bm25_score": float(scores[i])}
            out.append(Document(page_content=self._docs[i], metadata=meta))
        return out

    def _dense_topk(self, q: str, k: int) -> List[Document]:
        """
        Retrieve top-k documents via the dense vector store interface.

        Uses:
            - similarity_search_with_relevance_scores → adds "dense_relevance"
            - fallback similarity_search_with_score (distance) → adds "dense_similarity"

        Args:
            q: Query string.
            k: Number of results.

        Returns:
            List[Document]: Documents with merged metadata including dense score.
        """
        try:
            pairs = self.vs.similarity_search_with_relevance_scores(q, k=k)
            to_meta = lambda rel: {"dense_relevance": float(rel)}
        except Exception:
            pairs = self.vs.similarity_search_with_score(q, k=k)
            to_meta = lambda dist: {"dense_similarity": 1.0 / (1.0 + float(dist))}
        out: List[Document] = []
        for doc, val in pairs:
            meta = {**(doc.metadata or {}), **to_meta(val)}
            out.append(Document(page_content=doc.page_content, metadata=meta))
        return out

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> List[Document]:
        """
        Fuse dense and sparse results with Reciprocal Rank Fusion (RRF).

        Steps:
            1) Dense leg from vector store; annotate with dense scores.
            2) Sparse leg from BM25 (if available); annotate with bm25_score.
            3) Deduplicate candidates by (source/id/hash(page), chunk).
            4) RRF score: score += 1 / (C + rank), C=50, rank starts at 1.
            5) Return top self.k by fused score.

        Args:
            query: Natural language query.
            run_manager: Optional LangChain callback manager (not used directly).

        Returns:
            List[Document]: Top-k fused documents, optionally filtered by keyword metadata.
        """
        C = 50
        dense = self._dense_topk(query, self.k)
        sparse = self._bm25_topk(query, self.k)
        if not dense and not sparse:
            return []

        pool: Dict[Any, Dict[str, Any]] = {}

        def key_for(d: Document):
            return (
                d.metadata.get("source")
                or d.metadata.get("id")
                or hash(d.page_content),
                d.metadata.get("chunk"),
            )

        def add(doc: Document, rank: int):
            key = key_for(doc)
            pool.setdefault(key, {"doc": doc, "score": 0.0})
            pool[key]["score"] += 1.0 / (C + rank)

        for i, d in enumerate(dense, start=1):
            add(d, i)
        for i, d in enumerate(sparse, start=1):
            add(d, i)

        fused = sorted(pool.values(), key=lambda x: x["score"], reverse=True)
        docs = [x["doc"] for x in fused]

        # --- Keyword-aware filtering (post re-ranking) --------------------
        # If chunks were ingested with metadata["keywords"] (list[str]), prefer
        # documents whose keywords overlap with the user's query. Fall back to
        # the original ranking if no keyword-filtered docs remain.
        q_lower = (query or "").lower()
        filtered: List[Document] = []
        for d in docs:
            md = d.metadata or {}
            kws = md.get("keywords") or []
            if not isinstance(kws, (list, tuple)):
                continue
            for kw in kws:
                try:
                    k = str(kw).strip().lower()
                except Exception:
                    continue
                if not k:
                    continue
                if k in q_lower:
                    filtered.append(d)
                    break

        if filtered:
            return filtered[: self.k]
        return docs[: self.k]

def _estimate_collection_size(vs) -> int | None:
    """
    Best-effort count of vectors stored in a vector store.

    Supports:
        - Chroma: vs._collection.count() or len(ids)
        - Qdrant: client.count(collection, exact=True)

    Args:
        vs: LangChain vector store wrapper.

    Returns:
        int | None: Number of vectors, or None if unavailable.
    """
    if hasattr(vs, "_collection"):
        try:
            return int(vs._collection.count())
        except Exception:
            try:
                got = vs._collection.get(include=["ids"])
                return int(len(got.get("ids", []) or []))
            except Exception:
                return None

    if hasattr(vs, "client") and hasattr(vs, "collection_name"):
        try:
            resp = vs.client.count(vs.collection_name, exact=True)
            return int(getattr(resp, "count", 0))
        except Exception:
            return None

    return None


# ---- Build a merged retriever that is also a BaseRetriever -----------
def build_retriever(
    vs: Any, mode: str = "hybrid", *, k: int = TOP_K, llm=None
) -> BaseRetriever:
    """
    Build a retriever over a vector store: "mmr" or "hybrid" pipeline.

    Modes:
        - "mmr": returns vs.as_retriever(search_type="mmr", ...)
        - "hybrid" (default): multi-stage with query expansion, compression,
          and HybridRetriever (dense + BM25) with final merge/dedupe.

    Empty store behavior:
        If the underlying collection is empty, BM25 is skipped and the function
        returns only the compressed retriever.

    Args:
        vs: Vector store instance (e.g., Qdrant LangChain wrapper).
        mode: "mmr" or "hybrid".
        k: Per-leg target top-k to retrieve.
        llm: LLM for query expansion/compression. If None, uses get_llm().

    Returns:
        BaseRetriever: A ready-to-use retriever instance.
    """
    if mode == "mmr":
        return vs.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "fetch_k": min(20, k * 3),
                "lambda_mult": 0.7,
                "score_threshold": 0.3,
            },
        )

    llm = llm or get_llm()
    base = vs.as_retriever(search_kwargs={"k": k})
    mqr = MultiQueryRetriever.from_llm(retriever=base, llm=llm)
    compressor = LLMChainExtractor.from_llm(llm)
    ccr = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=mqr)

    col_size = _estimate_collection_size(vs)
    if not col_size:
        return ccr

    hybrid = HybridRetriever(vs=vs, k=k)

    class CombinedRetriever(BaseRetriever):
        """
        Combine compressed multi-query retrieval with hybrid (dense+BM25) results.

        Deduplicates by (source/id/hash(page), chunk) and caps output to out_k.
        """

        ccr: BaseRetriever
        hybrid: BaseRetriever
        out_k: int = 8

        def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: CallbackManagerForRetrieverRun | None = None,
        ) -> List[Document]:
            """
            Retrieve from both legs (compressed + hybrid), then deduplicate.

                Deduplication key:
                    (metadata['source'] or metadata['id'] or hash(page_content),
                     metadata.get('chunk'))

                Returns up to 'self.out_k' documents.
            """
            config = RunnableConfig(
                callbacks=run_manager.get_child() if run_manager else None
            )

            a = self.ccr.invoke(query, config=config)
            b = self.hybrid.invoke(query, config=config)
            seen: set[tuple] = set()
            out: List[Document] = []
            for d in a + b:
                key = (
                    d.metadata.get("source")
                    or d.metadata.get("id")
                    or hash(d.page_content),
                    d.metadata.get("chunk"),
                )
                if key in seen:
                    continue
                seen.add(key)
                out.append(d)
            return out[: self.out_k]

        async def _aget_relevant_documents(
            self,
            query: str,
            *,
            run_manager: CallbackManagerForRetrieverRun | None = None,
        ) -> List[Document]:
            """
            Async variant to keep event loop responsive (preferred in streaming path).
            """
            config = RunnableConfig(
                callbacks=run_manager.get_child() if run_manager else None
            )
            # Prefer async calls when retrievers expose them; otherwise fall back.
            async def _call(r, q):
                if hasattr(r, "ainvoke"):
                    return await r.ainvoke(q, config=config)
                if hasattr(r, "aget_relevant_documents"):
                    return await r.aget_relevant_documents(q, callbacks=config.get("callbacks"))
                return r.invoke(q, config=config) if hasattr(r, "invoke") else []

            a, b = await asyncio.gather(_call(self.ccr, query), _call(self.hybrid, query))

            seen: set[tuple] = set()
            out: List[Document] = []
            for d in list(a or []) + list(b or []):
                key = (
                    d.metadata.get("source")
                    or d.metadata.get("id")
                    or hash(d.page_content),
                    d.metadata.get("chunk"),
                )
                if key in seen:
                    continue
                seen.add(key)
                out.append(d)
            return out[: self.out_k]

    return CombinedRetriever(ccr=ccr, hybrid=hybrid, out_k=min(8, k))


# ---- Prompt builder ---------------------------------------------------
def make_answer_prompt(
    query: str, docs: List[Document] | None = None
) -> ChatPromptTemplate:
    """
    Build a one-shot prompt (system+human) for strict citation answers.

    The provided documents are truncated and prefixed with [S{i}] tags to guide
    citation formatting via ANSWER_WITH_CITATIONS.

    Args:
        query: User question.
        docs: Optional list of context documents to inline in the prompt.

    Returns:
        ChatPromptTemplate: Ready for llm.invoke(prompt.format_messages()).
    """
    docs = docs or []
    ctx = "\n\n".join(
        f"[S{i+1}] {(d.page_content or '').strip()[:1200]}" for i, d in enumerate(docs)
    )
    human_msg = f"""Question: {query}

Context:
{ctx}

Instructions:
{ANSWER_WITH_CITATIONS}
"""
    return ChatPromptTemplate.from_messages(
        [
            ("system", INJECTION_GUARD),
            ("system", AGENT_SYSTEM),
            ("human", human_msg),
        ]
    )


def build_qa_chain(retriever: BaseRetriever) -> RetrievalQA:
    """
    Build a RetrievalQA chain configured with finance rails and citation policy.

    Prompt:
        - system: INJECTION_GUARD
        - system: QA_SYSTEM
        - human:  QA_USER_TEMPLATE (expects {{context}}, {{question}})

    Args:
        retriever: Any LangChain retriever.

    Returns:
        RetrievalQA: A chain returning "result" and "source_documents".
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", INJECTION_GUARD.strip()),
            ("system", QA_SYSTEM.strip()),
            ("human", QA_USER_TEMPLATE),
        ]
    )

    return RetrievalQA.from_chain_type(
        llm=get_llm(),
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
    )


# ---- Relevance guard --------------------------------------------------------
def _doc_max_score(docs: list[Document]) -> float:
    """
    Compute a conservative "max relevance" from doc metadata (dense/BM25).

    Considers:
        - dense_relevance
        - dense_similarity
        - bm25_score / 10

    Args:
        docs: Candidate documents.

    Returns:
        float: Max score seen (0.0 if unavailable).
    """
    best = 0.0
    for d in docs or []:
        m = d.metadata or {}
        dense_rel = float(m.get("dense_relevance") or 0.0)
        dense_sim = float(m.get("dense_similarity") or 0.0)
        bm25 = float(m.get("bm25_score") or 0.0)
        score = max(dense_rel, dense_sim, bm25 / 10.0)
        if score > best:
            best = score
    return best


def _strict_answer_with_tagged_context(
    query: str,
    retriever: BaseRetriever,
    max_docs: int = 6,
) -> tuple[str, List[Document]]:
    """
    Retrieve documents, tag them [S#], and ask the LLM with strict citation rails.

    Args:
        query: User question.
        retriever: BaseRetriever to fetch documents.
        max_docs: Limit of documents to pass to the LLM.

    Returns:
        (answer_text, used_docs):
            answer_text: Model output (may contain [S#] citations).
            used_docs: Subset of tagged docs filtered by citations or fallback heuristic.
    """
    raw_docs = retriever.invoke(query or "", config=None)[:max_docs]
    if not raw_docs:
        return ("", [])

    tagged_docs, _rag_sources = _tag_retrieval_docs(raw_docs)

    ctx = "\n\n".join(d.page_content for d in tagged_docs)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", INJECTION_GUARD.strip()),
            ("system", QA_SYSTEM.strip()),
            ("human", STRICT_QA_USER_TEMPLATE.format(context=ctx, question=query)),
        ]
    )
    msgs = prompt.format_messages()
    llm = get_llm()
    model_name = _model_name_for_logging(llm)
    with with_token_log(model_name, tag="rag_strict") as usage:
        resp = llm.invoke(msgs, config={"callbacks": usage["callbacks"]})
    text = (getattr(resp, "content", None) or "").strip()

    cited_ids = set(_extract_cited_ids(text))
    if cited_ids:
        used_docs = [
            d for d in tagged_docs if (d.metadata or {}).get("sid") in cited_ids
        ]
    else:
        used_docs = _filter_docs_by_answer(text, tagged_docs, top_n=5)

    return (text, used_docs)


def _filter_docs_by_answer(
    answer: str, docs: List[Document], top_n: int = 5
) -> List[Document]:
    """
    Filter retrieved docs to those most similar to the final answer via BM25.

    Used as a fallback when the model doesn't emit explicit [S#] citations.

    Args:
        answer: Final answer text.
        docs: Candidate documents (tagged).
        top_n: Maximum number of documents to keep.

    Returns:
        List[Document]: Subset of docs most relevant to the answer.
    """
    toks = [_tok(answer or "")]
    if not docs:
        return []
    corpus = [_tok(d.page_content or "") for d in docs]
    bm = BM25Okapi(corpus)
    scores = bm.get_scores(toks[0] if toks else [])
    pairs = list(enumerate(scores))
    if not pairs:
        return []
    max_score = max(s for _, s in pairs) or 0.0
    keep = [i for i, s in pairs if s >= (0.25 * max_score)]
    keep_sorted = sorted(keep, key=lambda i: scores[i], reverse=True)[:top_n]
    return [docs[i] for i in keep_sorted]


# --- Strict citation helpers -------------------------------------------------
_SID_RE = re.compile(r"\[(S\d+)\]")


def _extract_cited_ids(text: str) -> list[str]:
    """
    Extract unique cited IDs like "[S1]" in order of first appearance.

    Args:
        text: Model output possibly containing citations.

    Returns:
        List[str]: e.g., ["S1", "S3"].
    """
    seen, out = set(), []
    for m in _SID_RE.finditer(text or ""):
        sid = m.group(1)
        if sid not in seen:
            seen.add(sid)
            out.append(sid)
    return out


def _sources_from_tagged_docs(tagged_docs: List[Document]) -> list[dict]:
    """
    Convert tagged docs (with metadata['sid']) into source objects for the UI.

    Args:
        tagged_docs: Documents produced by _tag_retrieval_docs.

    Returns:
        List[dict]: {id, display, title, href, type} suitable for rendering.
    """
    out = []
    for d in tagged_docs or []:
        md = d.metadata or {}
        sid = md.get("sid") or ""
        title = (md.get("title") or md.get("source") or "Source").strip()
        href = md.get("url") or md.get("href") or None
        page = md.get("page")

        display = title
        if page not in (None, "", "?"):
            try:
                display = f"{title} | Page {int(page)}"
            except Exception:
                display = f"{title} | Page {page}"

        out.append(
            {
                "id": sid,
                "display": display,
                "title": title,
                "href": href,
                "page": page,
                "type": (
                    "web"
                    if (href and str(href).startswith(("http://", "https://")))
                    else "doc"
                ),
            }
        )
    return out


def _filter_sources_by_ids(cited: list[str], pool: list[dict]) -> list[dict]:
    """
    Filter a source pool by a list of cited IDs, preserving order.

    Args:
        cited: List of "S#" ids.
        pool: List of source dicts with "id" keys.

    Returns:
        List[dict]: Ordered subset matching cited ids.
    """
    if not pool or not cited:
        return []
    index = {s.get("id"): s for s in pool if s.get("id")}
    return [index[c] for c in cited if c in index]


@lru_cache(maxsize=100)
def cached_doc_search_with_sources(query: str) -> Tuple[str, List[Dict]]:
    """
    Strict-citation RAG: answer + only the sources actually cited as [S#].

    Steps:
        1) build_vectorstore_qdrant() → build_retriever(mode="hybrid")
        2) quick pre-check on preliminary relevance (via _doc_max_score)
        3) _strict_answer_with_tagged_context() to enforce [S#] usage
        4) Convert only cited [S#] docs into wire-format sources for UI
        5) Fallback to heuristic filtering if no explicit citations

    Args:
        query: User query string.

    Returns:
        (markdown, sources):
            markdown: Answer text (prefixed with "## 📚 Financial Literature Search").
            sources: Only the cited (or heuristically filtered) sources.
    """
    vs = get_vectorstore()
    r = build_retriever(vs, mode="hybrid")
    prelim = r.invoke(query or "", config=None)
    max_score = _doc_max_score(prelim)
    if not prelim or max_score < 0.20:
        msg = (
            "❌ Sorry — I couldn’t find anything about that in my knowledge base. "
            "I’m focused on investing and finance topics. Try asking about portfolios, fees, "
            "stock prices, or paste a document/link to index."
        )
        return (msg, [])

    answer_text, tagged_docs = _strict_answer_with_tagged_context(query, r)
    answer_text = (answer_text or "").strip()

    if not answer_text:
        return (
            "❌ Sorry — I couldn’t find context to answer that from the knowledge base.",
            [],
        )

    pool = _sources_from_tagged_docs(tagged_docs)
    cited_ids = _extract_cited_ids(answer_text)
    used_sources = _filter_sources_by_ids(cited_ids, pool)

    if not used_sources:
        filtered = _filter_docs_by_answer(answer_text, tagged_docs)
        if filtered:
            pool2 = _sources_from_tagged_docs(filtered)
            used_sources = pool2

    if not used_sources:
        out_md = "## 📚 Financial Literature Search\n" + answer_text
        return (out_md.strip(), [])

    out_md = "## 📚 Financial Literature Search\n" + answer_text
    return (out_md.strip(), used_sources)


@lru_cache(maxsize=100)
def _cached_doc_search(query: str) -> str:
    """
    Simpler cached RAG QA: answer with a deduplicated bullet list of sources.

    Pipeline:
        1) get_vectorstore() → build_retriever("hybrid")
        2) build_qa_chain(retriever)
        3) Prefix query with "financial advice investment " for domain bias
        4) Invoke and format Markdown with answer + "- {title} | Page {page}"

    Caching:
        Memoized per exact query string (lru_cache). Clear via _cached_doc_search.cache_clear().

    Args:
        query: Raw user query.

    Returns:
        Markdown string with answer and (if available) source bullets.
    """
    vs = get_vectorstore()
    r = build_retriever(vs, mode="hybrid")
    chain = build_qa_chain(r)
    enhanced = f"financial advice investment {query}".strip()
    llm = get_llm()
    model_name = _model_name_for_logging(llm)
    with with_token_log(model_name, tag="rag_qa") as usage:
        result = chain.invoke(
            {"query": enhanced}, config={"callbacks": usage["callbacks"]}
        )
    answer = result.get("result", "No answer found.")
    srcs = result.get("source_documents", [])
    out = ["## Financial Literature Search", answer, ""]
    if srcs:
        seen = set()
        for d in srcs:
            source = d.metadata.get("title") or os.path.basename(
                d.metadata.get("source", "Unknown")
            )
            page = d.metadata.get("page", "?")
            key = f"{source}|{page}"
            if key in seen:
                continue
            seen.add(key)
            out.append(f"- {source} | Page {page}")
    else:
        out.append("(No specific sources in the KB.)")
    return "\n".join(out)


# =========================
# Public, cached interfaces
# =========================

# Cached vector store handle (Qdrant).
_vectorstore_singleton: Any | None = None
_retriever_singleton: BaseRetriever | None = None


def get_vectorstore(*, force_rebuild: bool = False):
    """
    Return a cached Qdrant-based vector store handle.

    Env:
        VECTORSTORE=qdrant (default/only supported) → build_vectorstore_qdrant(...)

    Args:
        force_rebuild: Ignored for Qdrant (kept for signature compatibility).

    Returns:
        A LangChain Qdrant vector store instance.
    """
    global _vectorstore_singleton
    if _vectorstore_singleton is not None and not force_rebuild:
        return _vectorstore_singleton

    backend = (os.getenv("VECTORSTORE", "qdrant") or "qdrant").lower()
    logger.info("RAG vector backend = %s", backend)
    if backend != "qdrant":
        raise RuntimeError(
            f"Only Qdrant VECTORSTORE is supported; got VECTORSTORE={backend!r}"
        )
    vs = build_vectorstore_qdrant(collection_name=QDRANT_COLLECTION, docs_dir=DOCS_DIR)

    _vectorstore_singleton = vs
    return vs

def get_retriever(
    *, mode: str = "hybrid", k: int = TOP_K, force_rebuild: bool = False
) -> BaseRetriever:
    """
    Return a cached BaseRetriever constructed over the vector store.

    Args:
        mode: "hybrid" (default) or "mmr".
        k: Target top-k for each retrieval leg.
        force_rebuild: If True, rebuild the underlying vector store first.

    Returns:
        BaseRetriever: Ready to use with build_qa_chain or direct .get_relevant_documents().
    """
    global _retriever_singleton
    if _retriever_singleton is not None and not force_rebuild:
        return _retriever_singleton

    vs = get_vectorstore(force_rebuild=force_rebuild)
    r = build_retriever(vs, mode=mode, k=k, llm=get_llm())
    _retriever_singleton = r
    return r


def clear_caches() -> None:
    """
    Clear in-memory singletons and memoized RAG results.

    Use after reindexing or when refreshing the corpus.
    """

    global _vectorstore_singleton, _retriever_singleton
    _vectorstore_singleton = None
    _retriever_singleton = None
    try:
        _cached_doc_search.cache_clear()
    except Exception:
        pass
    try:
        cached_doc_search_with_sources.cache_clear()
    except Exception:
        pass


def reindex(*, persist_dir: str | None = None, docs_dir: str | None = None) -> None:
    """
    Refresh Qdrant-based RAG caches.

    Note:
        - Qdrant ingestion (creating/updating points) is handled externally,
          e.g. via 'ingest_pdfs_qdrant.py'.
        - This function clears in-memory caches and reconnects to the current
          collection so new data becomes visible without restarting the app.

    Args:
        persist_dir: Ignored (kept for backward compatibility).
        docs_dir: Ignored (kept for backward compatibility).

    Returns:
        None. After completion, get_vectorstore()/get_retriever() will use fresh state.
    """
    clear_caches()
    _ = get_vectorstore()
    _ = get_retriever()


def search_many(
    queries: List[str], *, k_per_query: int | None = None, out_k: int | None = None
) -> List[Document]:
    """
    Run retrieval across multiple sub-queries, deduplicate, and cap results.

    Args:
        queries: Reformulated query variants.
        k_per_query: Optional per-query cap (defaults to retriever's k).
        out_k: Final global cap (defaults to TOP_K).

    Returns:
        List[Document]: Deduped merged results up to out_k.
    """
    r = get_retriever()
    seen: set[tuple] = set()
    merged: List[Document] = []
    for q in queries or []:
        try:
            docs = r.invoke(q, config=None)
        except Exception:
            docs = []
        if k_per_query:
            docs = docs[:k_per_query]
        for d in docs:
            key = (
                d.metadata.get("source")
                or d.metadata.get("id")
                or hash(d.page_content),
                d.metadata.get("chunk"),
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(d)
    return merged[: (out_k or TOP_K)]


def get_functions_agent(debug: bool | None = None) -> AgentExecutor:
    """
    Convenience wrapper to construct a tools-first agent using the default LLM and rails.

    Args:
        debug: If provided, overrides AGENT_DEBUG for verbosity.

    Returns:
        AgentExecutor: Ready-to-run agent compatible with your registered tools.
    """
    return build_functions_agent(llm=get_llm(), debug=debug)
