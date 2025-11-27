from __future__ import annotations
from typing import Literal
from langchain_core.tools import tool, ToolException
from langchain_core.runnables import RunnableConfig

try:
    from routers.fun import welcome as fun_welcome
except Exception:
    fun_welcome = None

from agent.schemas import ToolFunImageIn, ToolFunImageOut


@tool("fun_cute_image", args_schema=ToolFunImageIn, return_direct=False)
def fun_cute_image_tool(
    source: Literal["cat", "dog", "fox", "random"] = "random",
    fresh: bool = False,
    config: RunnableConfig = {},
) -> ToolFunImageOut:
    """Return a cute, kid‑friendly animal image.

    Uses the same sources as the dashboard widget (Cat API, Dog API, RandomFox)
    via the server, with jpg/png and small sizes.

    Args:
        source: Prefer a specific source ("cat", "dog", "fox", or "random").
        fresh: If True, bypass server cache for a new image.
        config: Unused; present for consistency with other tools.

    Returns:
        'ToolFunImageOut' mapping with keys 'kind='image'', 'url', 'alt',
        and 'source'.

    Raises:
        ToolException: If the backend fun router is unavailable, or the
        response contains an invalid URL, or on unexpected failures.
    """
    if fun_welcome is None:
        raise ToolException("FUN_DISABLED: backend fun router not available")
    try:
        out = fun_welcome(source=source, fresh=bool(fresh))
        url = out.get("url")
        alt = out.get("alt") or f"Cute {out.get('source', 'animal')}"
        src = out.get("source")
        if not isinstance(url, str) or not url.startswith("http"):
            raise ToolException("FUN_BAD_URL")
        if src not in ("cat", "dog", "fox"):
            src = "fox"
        payload = ToolFunImageOut(
            kind="image", url=url, alt=alt, source=src
        ).model_dump()
        return payload
    except ToolException:
        raise
    except Exception as e:
        raise ToolException(f"FUN_FAILED: {e}")
