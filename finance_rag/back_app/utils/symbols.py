from __future__ import annotations
import re
from typing import Set, Pattern, Tuple, List

# ---------------------------------
# Exchange / region suffix handling
# ---------------------------------

NON_US_SUFFIXES: Set[str] = {
    "TO",
    "V",
    "L",
    "AX",
    "HK",
    "SZ",
    "SS",
    "KS",
    "TW",
    "PA",
    "MI",
    "F",
    "SW",
    "BMV",
    "T",
    "NZ",
}

ALLOW_SUFFIX: Set[str] = set()
_CLASS_SHARE_ROOTS: Set[str] = {"BRK", "BF", "HEI"}
_CLASS_ALIASES: List[Tuple[Pattern[str], str]] = [
    (re.compile(r"\bBRK(?:[ \-/])?B\b", re.I), "BRK.B"),
    (re.compile(r"\bBRK(?:[ \-/])?A\b", re.I), "BRK.A"),
    (re.compile(r"\bBF(?:[ \-/])?B\b", re.I), "BF.B"),
    (re.compile(r"\bBF(?:[ \-/])?A\b", re.I), "BF.A"),
    (re.compile(r"\bHEI(?:[ \-/])?A\b", re.I), "HEI.A"),
]


def canonize_symbol(x: str) -> str:
    """
    Canonicalize an equity symbol into a stable, Finnhub-friendly form.

    Normalizations performed (in order):
      1) Class-share aliases → dot form for known roots:
         - 'BRK B', 'BRK/B', 'BRK-B'  → 'BRK.B'
         - 'BF A',  'BF/A',  'BF-A'   → 'BF.A'
         - 'HEI A', 'HEI/A', 'HEI-A'  → 'HEI.A'
      2) Strip a leading '$' (e.g., '$AAPL' → 'AAPL').
      3) Region/exchange hints → standard suffixes:
         - '.TSX', ':CA', '-T', '.TOR' → '.TO'   (Toronto)
         - '.TSXV'                     → '.V'    (TSX Venture)
         - ':GB'                       → '.L'    (London)
         - ':AU'                       → '.AX'   (ASX)
         - ':HK'                       → '.HK'   (Hong Kong)
      4) Remove explicit US qualifiers:
         - '.US' or ':US' → '' (bare US ticker)
      5) Collapse repeated dots and strip trailing dot:
         - '..' → '.',  'ABC.' → 'ABC'
      6) Fallback class-share pattern for known roots:
         - 'ROOT-CLASS', 'ROOT CLASS' → 'ROOT.CLASS' (if ROOT ∈ {'BRK','BF','HEI'})

    Leaves strictly US tickers bare (e.g., 'AAPL', 'MSFT').

    Args:
        x: Raw user-entered symbol text (may include spaces, punctuation, or hints).

    Returns:
        Canonicalized uppercase symbol string. Returns an empty string when 'x' is falsy.
    """
    if not x:
        return ""

    u = (x or "").strip().upper()
    for rx, rep in _CLASS_ALIASES:
        u = rx.sub(rep, u)
    u = re.sub(r"^\$", "", u)
    u = re.sub(r"\.TSX$", ".TO", u)
    u = re.sub(r":CA$", ".TO", u)
    u = re.sub(r"-T$", ".TO", u)
    u = re.sub(r"\.TSXV$", ".V", u)
    u = re.sub(r"\.TOR$", ".TO", u)
    u = re.sub(r":GB$", ".L", u)
    u = re.sub(r":AU$", ".AX", u)
    u = re.sub(r":HK$", ".HK", u)
    #u = re.sub(r"\.US\b", "", u)
    #u = re.sub(r":US\b", "", u)
    #u = re.sub(r"\.+", ".", u)
    u = re.sub(r"\.+", ".", u)
    u = re.sub(r"\.US\b", "", u)
    u = re.sub(r":US\b", "", u)
    u = re.sub(r"\.$", "", u)
    m = re.fullmatch(r"([A-Z]{1,5})[-/ ]([A-Z])", u)
    if m and m.group(1) in _CLASS_SHARE_ROOTS:
        u = f"{m.group(1)}.{m.group(2)}"

    return u
