from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Dict, Any

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SETTINGS_FILE = DATA_DIR / "owner_settings.json"


def _ensure_dir() -> None:
    """Ensure the data directory exists.

    Creates 'backend/data' if missing. Swallows filesystem errors to avoid
    impacting callers that can tolerate missing persistence.
    """
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def _load_all() -> Dict[str, Dict[str, Any]]:
    """Load all owner flags from the JSON settings file.

    Returns an empty mapping on any read/parse error or if the file does not
    exist. The structure is '{owner_id: {flag_key: value}}'.
    """
    try:
        if SETTINGS_FILE.exists():
            with SETTINGS_FILE.open("r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
    except Exception:
        return {}
    return {}


def _save_all(obj: Dict[str, Dict[str, Any]]) -> None:
    """Persist all owner flags to the JSON settings file.

    Silently ignores write errors, as feature flags are best-effort.
    """
    _ensure_dir()
    try:
        with SETTINGS_FILE.open("w", encoding="utf-8") as f:
            json.dump(obj, f)
    except Exception:
        pass


def _env_bool(name: str, default: bool = False) -> bool:
    """Read a boolean from the environment with common truthy strings.

    Recognized truthy values: 1, true, yes, on (case-insensitive). If the
    variable is missing, returns the provided default.
    """
    v = os.getenv(name)
    if v is None:
        return bool(default)
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def get_owner_flags(owner_id: str) -> Dict[str, Any]:
    """Return all stored flags for an owner.

    Args:
        owner_id: Owner identifier as string (will be coerced to str).

    Returns:
        Mapping of '{flag_key: value}'. Empty dict when none are set.
    """
    all_ = _load_all()
    v = all_.get(str(owner_id))
    return v if isinstance(v, dict) else {}


def set_owner_flags(owner_id: str, **flags: Any) -> Dict[str, Any]:
    """Set one or more flags for an owner and persist them.

    Values are coerced to 'bool' for consistency.

    Args:
        owner_id: Owner identifier.
        **flags: Arbitrary key=value pairs representing feature flags.

    Returns:
        The updated mapping for the owner after persistence.
    """
    all_ = _load_all()
    cur = all_.get(str(owner_id)) or {}
    cur.update({k: bool(v) for k, v in flags.items()})
    all_[str(owner_id)] = cur
    _save_all(all_)
    return cur


def get_owner_flag(
    owner_id: str, key: str, env_name: str, *, default: bool = False
) -> bool:
    """Resolve a feature flag for an owner with environment fallback.

    Order of precedence:
      1) Per-owner value in 'owner_settings.json' if present.
      2) Environment variable 'env_name' (parsed as boolean string).
      3) Provided 'default' when neither is set.
    """
    flags = get_owner_flags(owner_id)
    if key in flags:
        return bool(flags[key])
    return _env_bool(env_name, default)
