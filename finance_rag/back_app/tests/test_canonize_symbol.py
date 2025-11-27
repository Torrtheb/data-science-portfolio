# tests/test_canonize_symbol.py
from __future__ import annotations
import pytest

from back_app.utils.symbols import canonize_symbol


@pytest.mark.parametrize(
    "raw, expected",
    [
        # 1) Known class-share aliases → dot form
        ("BRK B", "BRK.B"),
        ("BRK/B", "BRK.B"),
        ("BRK-B", "BRK.B"),
        ("brk b", "BRK.B"),
        ("BF A", "BF.A"),
        ("BF/A", "BF.A"),
        ("BF-A", "BF.A"),
        ("HEI A", "HEI.A"),
        ("HEI/A", "HEI.A"),
        ("HEI-A", "HEI.A"),

        # 2) Strip leading '$'
        ("$AAPL", "AAPL"),
        ("$msft", "MSFT"),

        # 3) Region/exchange hints → standard suffixes
        ("SHOP.TSX", "SHOP.TO"),
        ("SHOP:CA", "SHOP.TO"),
        ("BNS-T", "BNS.TO"),
        ("CVE.TSXV", "CVE.V"),
        ("RY.TOR", "RY.TO"),
        ("BP:GB", "BP.L"),
        ("BHP:AU", "BHP.AX"),
        ("0700:HK", "0700.HK"),

        # 4) Remove explicit 'US' qualifiers
        ("AAPL.US", "AAPL"),
        ("MSFT:US", "MSFT"),
        ("GOOG..US", "GOOG"),  # also tests dot collapsing below

        # 5) Collapse multiple dots
        ("AAPL..TO", "AAPL.TO"),
        ("X...TO", "X.TO"),

        # 6) Class-shares of known roots written with dash/space (fallback regex)
        ("BRK C", "BRK.C"),     # not a real class, but tests the logic path
        ("BF-C", "BF.C"),
        ("HEI/A", "HEI.A"),     # already covered by alias but fine

        # 7) Idempotence for already canonical forms
        ("AAPL", "AAPL"),
        ("MSFT", "MSFT"),
        ("SHOP.TO", "SHOP.TO"),
        ("CVE.V", "CVE.V"),
        ("BP.L", "BP.L"),
        ("BHP.AX", "BHP.AX"),
        ("0700.HK", "0700.HK"),

        # 8) Trimming / case-normalization
        ("  msft  ", "MSFT"),
        ("\tBrK b\n", "BRK.B"),

        # 9) Falsy / empty
        ("", ""),
    ],
)
def test_canonize_symbol_parametrized(raw: str, expected: str):
    assert canonize_symbol(raw) == expected


def test_canonize_symbol_does_not_strip_unknown_suffixes():
    """
    The function only maps certain known hints and '.US'/':US'.
    Unknown suffixes should remain untouched (aside from uppercasing).
    """
    assert canonize_symbol("ACME.XY") == "ACME.XY"
    assert canonize_symbol("acme.zz") == "ACME.ZZ"


def test_canonize_symbol_mixed_transformations():
    """
    Multiple transforms in one go:
      - leading '$'
      - class alias
      - region hint
      - dot collapsing
    """
    raw = "$brk b..tsx"
    # Steps:
    #  - '$' removed → 'brk b..tsx'
    #  - alias → 'BRK.B..TSX'
    #  - collapse dots → 'BRK.B.TSX'
    #  - '.TSX' → '.TO'
    assert canonize_symbol(raw) == "BRK.B.TO"
