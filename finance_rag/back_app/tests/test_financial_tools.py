# tests/test_financial_tools.py
# Basic unit tests for pure financial helpers (no network, no DB).

import math
from decimal import Decimal
import pytest

# Adjust the import path if your module layout differs.
try:
    from back_app.llm.tools import (
        calculate_simple_interest,
        calculate_compound_interest,
        calculate_investment_return,
        investment_return_from_strings,
        calculate_loan_amortization,
        npv,
        cagr,
        _normalize_freq_key,          # internal but useful to sanity-check
        _validate_and_get_frequency,  # internal but useful to sanity-check
        FREQUENCY_MAP,
    )
except ModuleNotFoundError:
    # Fallback import if your file lives elsewhere (e.g., back_app/tools.py)
    from back_app.llm.tools import (
        calculate_simple_interest,
        calculate_compound_interest,
        calculate_investment_return,
        investment_return_from_strings,
        calculate_loan_amortization,
        npv,
        cagr,
        _normalize_freq_key,
        _validate_and_get_frequency,
        FREQUENCY_MAP,
    )


def test_calculate_simple_interest_basic():
    out = calculate_simple_interest(principal=1000, rate_percent=5, years=2)
    assert out["interest"] == 1000 * 0.05 * 2
    assert out["total_amount"] == 1100
    # no inflation -> real_value equals total_amount
    assert out["real_value"] == out["total_amount"]
    assert out["purchasing_power_loss"] == 0.0


def test_calculate_simple_interest_with_inflation():
    out = calculate_simple_interest(principal=1000, rate_percent=5, years=2, inflation_rate_percent=2)
    assert out["total_amount"] == 1100
    # Real value discounted by inflation
    assert out["real_value"] == pytest.approx(1100 / (1.02 ** 2), rel=1e-9)
    assert out["purchasing_power_loss"] == pytest.approx(1100 - out["real_value"], rel=1e-9)


def test_calculate_compound_interest_monthly():
    out = calculate_compound_interest(principal=2000, rate_percent=6, years=3, compounding_per_year=12)
    # Expected total using monthly compounding: P * (1 + r/m)^(m*t)
    expected_total = 2000 * ((1 + 0.06 / 12) ** (12 * 3))
    assert out["total_amount"] == pytest.approx(expected_total, rel=1e-12)
    assert out["interest"] == pytest.approx(expected_total - 2000, rel=1e-12)


def test_investment_return_plain_compound_vs_wrapper():
    base = calculate_investment_return(
        principal=10000,
        rate_percent=7,
        years=5,
        compounds_per_year=12,
        contribution_per_period=0,
        contribution_frequency_per_year=12,
    )
    wrapped = investment_return_from_strings(
        principal=10000,
        rate_percent=7,
        years=5,
        compound="monthly",
        regular_addition=0,
        regular_addition_every="monthly",
    )
    # core numerics should match (no contributions)
    assert wrapped["future_value"] == pytest.approx(base["future_value"], rel=1e-12)
    assert wrapped["total_return"] == pytest.approx(base["total_return"], rel=1e-12)


def test_investment_return_with_contributions_begin_vs_end():
    # same inputs except timing; "begin" should yield a strictly larger FV than "end"
    end = investment_return_from_strings(
        principal=1,
        rate_percent=6,
        years=2,
        compound="monthly",
        regular_addition=500,
        regular_addition_every="monthly",
        addition_timing="end",
    )
    begin = investment_return_from_strings(
        principal=1,
        rate_percent=6,
        years=2,
        compound="monthly",
        regular_addition=500,
        regular_addition_every="monthly",
        addition_timing="begin",
    )
    assert begin["future_value"] > end["future_value"]
    assert begin["number_of_contributions"] == end["number_of_contributions"] == 24
    assert begin["total_contributions"] == end["total_contributions"] == 500 * 24


def test_normalize_and_validate_frequency():
    # spot-check synonyms & normalization
    assert _normalize_freq_key("semi-annual") == "semiannually"
    assert _normalize_freq_key("bi-weekly") == "biweekly"
    assert _normalize_freq_key("Monthly") == "monthly"
    # validate returns numbers
    assert _validate_and_get_frequency("monthly", "field") == FREQUENCY_MAP["monthly"]
    assert _validate_and_get_frequency("semi-annual", "field") == FREQUENCY_MAP["semiannually"]
    with pytest.raises(ValueError):
        _validate_and_get_frequency("every-so-often", "field")


def test_loan_amortization_zero_rate():
    out = calculate_loan_amortization(
        principal=Decimal("12000"),
        annual_rate_percent=0.0,
        years=1,
        payments_per_year=12,
    )
    # Exactly 12 equal principal-only payments of 1000, no interest
    assert out["payment"] == Decimal("1000.00")
    assert out["total_interest"] == Decimal("0.00")


def test_loan_amortization_positive_rate():
    out = calculate_loan_amortization(
        principal=Decimal("250000"),
        annual_rate_percent=5.0,
        years=30,
        payments_per_year=12,
    )
    # Known around ~ $1342.05 for standard 30-year @5% (allow a small tolerance for quantization)
    assert out["payment"] == Decimal("1342.05")
    # total interest > 0 and plausible magnitude
    assert out["total_interest"] > Decimal("0.00")


def test_npv_basic():
    # Simple cashflows: at t=0: -1000, t=1: +600, t=2: +600; discount 10% per period
    value = npv(10.0, [Decimal("-1000"), Decimal("600"), Decimal("600")])
    # Manual: -1000 + 600/1.1 + 600/(1.1^2)
    expected = (-1000) + (600 / 1.1) + (600 / (1.1 ** 2))
    # Rounded to cents inside npv
    assert float(value) == pytest.approx(round(expected, 2), abs=1e-2)


def test_cagr_basic():
    out = cagr(1000, 1771.561, 5)
    # CAGR ≈ 12% (since 1000*(1.12)^5 ≈ 1762); allow small tolerance
    assert out["type"] == "cagr"
    assert out["cagr"] == pytest.approx(0.12, abs=0.005)


@pytest.mark.parametrize(
    "bad",
    [
        dict(principal=0, rate_percent=5, years=1),
        dict(principal=1000, rate_percent=-1, years=1),
        dict(principal=1000, rate_percent=5, years=0),
        dict(principal=1000, rate_percent=5, years=1, inflation_rate_percent=-0.1),
    ],
)
def test_simple_interest_validation(bad):
    with pytest.raises(ValueError):
        calculate_simple_interest(**bad)
