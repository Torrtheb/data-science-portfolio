from __future__ import annotations
from functools import lru_cache
import re
import unicodedata
from typing import Iterable, Tuple
import pycountry
from rapidfuzz import process, fuzz

# ------------------------------- Normalization -------------------------------


def _strip_accents(s: str) -> str:
    """Remove diacritics: 'Côte d’Ivoire' -> 'Cote d’Ivoire'."""
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")


_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


def _norm(s: str) -> str:
    """
    Lowercase, strip accents, drop punctuation, squeeze spaces.
    Used for alias keys and fuzzy matching.
    """
    s = _strip_accents(s)
    s = s.lower()
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s


# ---------------------------- Pragmatic aliases ------------------------------

_ALIAS_TO_ISO3 = {
    _norm("USA"): "USA",
    _norm("U.S.A."): "USA",
    _norm("United States"): "USA",
    _norm("America"): "USA",
    _norm("UK"): "GBR",
    _norm("U.K."): "GBR",
    _norm("Great Britain"): "GBR",
    _norm("Britain"): "GBR",
    _norm("South Korea"): "KOR",
    _norm("North Korea"): "PRK",
    _norm("Russia"): "RUS",
    _norm("Vietnam"): "VNM",
    _norm("Laos"): "LAO",
    _norm("Ivory Coast"): "CIV",
    _norm("Cote d'Ivoire"): "CIV",
    _norm("Côte d’Ivoire"): "CIV",
    _norm("Cote d Ivoire"): "CIV",
    _norm("DRC"): "COD",
    _norm("Democratic Republic of the Congo"): "COD",
    _norm("Congo-Kinshasa"): "COD",
    _norm("Republic of the Congo"): "COG",
    _norm("Congo-Brazzaville"): "COG",
    _norm("Bolivia"): "BOL",
    _norm("Venezuela"): "VEN",
    _norm("Iran"): "IRN",
    _norm("Syria"): "SYR",
    _norm("Tanzania"): "TZA",
    _norm("Moldova"): "MDA",
    _norm("Macedonia"): "MKD",
    _norm("North Macedonia"): "MKD",
    _norm("Turkey"): "TUR",
    _norm("Türkiye"): "TUR",
    _norm("Czechia"): "CZE",
    _norm("Czech Republic"): "CZE",
    _norm("Eswatini"): "SWZ",
    _norm("Swaziland"): "SWZ",
    _norm("Burma"): "MMR",
    _norm("Myanmar"): "MMR",
    _norm("Cape Verde"): "CPV",
    _norm("Cabo Verde"): "CPV",
    _norm("South Sudan"): "SSD",
    _norm("Taiwan"): "TWN",
    _norm("UAE"): "ARE",
    _norm("United Arab Emirates"): "ARE",
    _norm("Brasil"): "BRA",
    _norm("Korea"): "KOR",
    _norm("Congo"): "COG",
    _norm("Kosovo"): "XKX",
}

_KOSOVO = ("XKX", "Kosovo")

# ------------------------------ Core resolver --------------------------------


def _pycountry_by_any_name(target: str):
    """
    Return a pycountry country object whose 'name', 'common_name', or 'official_name'
    exactly matches 'target'. Case sensitive.
    """
    for c in pycountry.countries:
        if c.name == target:
            return c
        if getattr(c, "common_name", None) == target:
            return c
        if getattr(c, "official_name", None) == target:
            return c
    return None


def _choices_for_fuzzy() -> Iterable[str]:
    """All country names we allow fuzzy matching against."""
    for c in pycountry.countries:
        yield c.name
        cn = getattr(c, "common_name", None)
        if cn:
            yield cn
        on = getattr(c, "official_name", None)
        if on:
            yield on
    yield "Kosovo"


@lru_cache(maxsize=2048)
def resolve_country(user_text: str) -> Tuple[str, str]:
    """
    Resolve a user-supplied country string to ISO-3166 alpha-3 and a canonical name.

    Accepts:
        Codes (alpha-3/alpha-2), official/common names, and many nicknames/misspellings.
        Examples: 'CAN', 'CA', 'canada', 'United States of America', 'Brasil', 'Cote d Ivoire'.

    Returns:
        (iso3, canonical_name)

    Raises:
        ValueError: if the input cannot be resolved.
    """
    if not user_text or not user_text.strip():
        raise ValueError("Empty country input.")

    raw = user_text.strip()
    alias_hit = _ALIAS_TO_ISO3.get(_norm(raw))
    if alias_hit:
        if alias_hit == "XKX":
            return _KOSOVO
        c = pycountry.countries.get(alpha_3=alias_hit)
        if c:
            return (c.alpha_3, c.name)

    c = pycountry.countries.get(alpha_3=raw.upper())
    if c:
        return (c.alpha_3, c.name)

    c = pycountry.countries.get(alpha_2=raw.upper())
    if c:
        return (c.alpha_3, c.name)

    try:
        c = pycountry.countries.lookup(raw)
        if c:
            return (c.alpha_3, c.name)
    except Exception:
        pass

    choices = list(_choices_for_fuzzy())
    match, score, _ = process.extractOne(
        raw, choices, scorer=fuzz.WRatio, score_cutoff=85
    )
    if match:
        if match == "Kosovo":
            return _KOSOVO
        c = _pycountry_by_any_name(match)
        if c:
            return (c.alpha_3, c.name)

    raise ValueError(f"Could not resolve country from '{user_text}'.")
