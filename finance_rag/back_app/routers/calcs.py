from __future__ import annotations
from typing import Any, Dict, TypedDict, TYPE_CHECKING
from fastapi import APIRouter, HTTPException

calc_api_router = APIRouter(prefix="/api/calc", tags=["Calculators"])

if TYPE_CHECKING:
    from ..llm.agent_tools import (
        SimpleInterestInput,
        CompoundInterestInput,
        InvestmentReturnInput,
        InvestmentReturnStringsInput,
        LoanAmortizationInput,
        NPVInput,
        CAGRInput,
    )


class ToolSource(TypedDict):
    """
    TypedDict describing metadata for a calculator tool result.

    Keys:
        type (str): Always "tool".
        name (str): Tool identifier (e.g. "simple_interest").
        title (str): Human-readable display title.
        meta (dict): Extra metadata (currently empty).
    """
    type: str
    name: str
    title: str
    meta: Dict[str, Any]


def _tool_source(name: str, title: str) -> ToolSource:
    """
    Construct a consistent source descriptor for UI display.

    Args:
        name: Tool identifier string.
        title: Human-readable tool title.

    Returns:
        ToolSource dict with type="tool", name, title, and empty meta.
    """
    return {"type": "tool", "name": name, "title": title, "meta": {}}


def _bad_request(msg: str) -> None:
    """
    Raise a standardized 400 Bad Request error.

    Args:
        msg: Error message to include in the HTTPException.

    Raises:
        HTTPException: Always, with status_code=400 and given detail.
    """
    raise HTTPException(status_code=400, detail=msg)


@calc_api_router.post("/simple-interest")
def http_simple_interest(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute simple interest over a given period.

    - Validates request body with 'SimpleInterestInput'.
    - Returns both markdown explanation and numeric results.
    - Supports inflation adjustment.

    Args:
        payload: JSON body matching SimpleInterestInput schema.

    Returns:
        Dict with:
          - tool (str)
          - ok (bool)
          - data (markdown + numeric results)
          - source (ToolSource)

    Raises:
        HTTPException 400: Invalid input.
    """
    try:
        from ..llm.agent_tools import (
            SimpleInterestInput,
            simple_interest_tool,
            calculate_simple_interest,
            _parse_percent_like,
        )

        p = SimpleInterestInput(**payload)
        md = simple_interest_tool(**p.model_dump())
        r = calculate_simple_interest(
            p.principal,
            float(_parse_percent_like(p.rate_percent)),
            p.years,
            float(_parse_percent_like(p.inflation_rate_percent or 0.0)),
        )
        return {
            "tool": "simple_interest",
            "ok": True,
            "data": {"markdown": md, **r},
            "source": _tool_source("simple_interest", "Simple Interest"),
        }
    except HTTPException:
        raise
    except Exception as e:
        _bad_request(f"Invalid simple-interest input: {e!s}")


@calc_api_router.post("/compound-interest")
def http_compound_interest(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute compound interest with optional inflation adjustment.

    - Validates input with 'CompoundInterestInput'.
    - Supports custom compounding frequency.
    - Returns markdown explanation and numeric results.

    Args:
        payload: JSON body with principal, rate, years, etc.

    Returns:
        Dict with tool metadata and results.

    Raises:
        HTTPException 400: Invalid input.
    """
    try:
        from ..llm.agent_tools import (
            CompoundInterestInput,
            compound_interest_tool,
            calculate_compound_interest,
            _parse_percent_like,
        )

        p = CompoundInterestInput(**payload)
        md = compound_interest_tool(**p.model_dump())
        r = calculate_compound_interest(
            p.principal,
            float(_parse_percent_like(p.rate_percent)),
            p.years,
            p.compounding_per_year,
            float(_parse_percent_like(p.inflation_rate_percent or 0.0)),
        )
        return {
            "tool": "compound_interest",
            "ok": True,
            "data": {"markdown": md, **r},
            "source": _tool_source("compound_interest", "Compound Interest"),
        }
    except HTTPException:
        raise
    except Exception as e:
        _bad_request(f"Invalid compound-interest input: {e!s}")


@calc_api_router.post("/investment-return")
def http_investment_return(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute investment return including contributions.

    - Handles contribution frequency and timing.
    - Validates with 'InvestmentReturnInput'.
    - Returns markdown and numeric results for visualization.

    Args:
        payload: JSON body describing investment parameters.

    Returns:
        Dict with tool metadata and results.

    Raises:
        HTTPException 400: Invalid input.
    """
    try:
        from ..llm.agent_tools import (
            InvestmentReturnInput,
            investment_return_tool,
            calculate_investment_return,
            _parse_percent_like,
        )

        p = InvestmentReturnInput(**payload)
        md = investment_return_tool(**p.model_dump())
        r = calculate_investment_return(
            principal=p.principal,
            rate_percent=float(_parse_percent_like(p.rate_percent)),
            years=p.years,
            compounds_per_year=p.compounds_per_year,
            contribution_per_period=p.contribution_per_period,
            contribution_frequency_per_year=p.contribution_frequency_per_year,
            contribution_timing=p.contribution_timing,
            inflation_rate_percent=float(
                _parse_percent_like(p.inflation_rate_percent or 0.0)
            ),
        )
        return {
            "tool": "investment_return",
            "ok": True,
            "data": {"markdown": md, **r},
            "source": _tool_source("investment_return", "Investment Return"),
        }
    except HTTPException:
        raise
    except Exception as e:
        _bad_request(f"Invalid investment-return input: {e!s}")


@calc_api_router.post("/investment-return-strings")
def http_investment_return_strings(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute investment return with stringly-typed parameters.

    - Accepts human-friendly values like "monthly", "end", "semiannual".
    - Validates with 'InvestmentReturnStringsInput'.
    - Returns markdown and numeric results.

    Args:
        payload: JSON body with textual investment parameters.

    Returns:
        Dict with tool metadata and results.

    Raises:
        HTTPException 400: Invalid input.
    """
    try:
        from ..llm.agent_tools import (
            InvestmentReturnStringsInput,
            investment_return_strings_tool,
            investment_return_from_strings,
            _parse_percent_like,
        )

        p = InvestmentReturnStringsInput(**payload)
        md = investment_return_strings_tool(**p.model_dump())
        r = investment_return_from_strings(
            principal=p.principal,
            rate_percent=float(_parse_percent_like(p.rate_percent)),
            years=p.years,
            compound=p.compound,
            regular_addition=p.regular_addition,
            regular_addition_every=p.regular_addition_every,
            addition_timing=p.addition_timing,
            inflation_rate_percent=float(
                _parse_percent_like(p.inflation_rate_percent or 0.0)
            ),
        )
        return {
            "tool": "investment_return_strings",
            "ok": True,
            "data": {"markdown": md, **r},
            "source": _tool_source(
                "investment_return_strings", "Investment Return (strings)"
            ),
        }
    except HTTPException:
        raise
    except Exception as e:
        _bad_request(f"Invalid investment-return-strings input: {e!s}")


@calc_api_router.post("/loan-amortization")
def http_loan_amortization(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a loan amortization schedule.

    - Uses 'Decimal' for principal to minimize rounding errors.
    - Validates with 'LoanAmortizationInput'.
    - Returns markdown plus per-period schedule data.

    Args:
        payload: JSON body with loan parameters.

    Returns:
        Dict with tool metadata and amortization results.

    Raises:
        HTTPException 400: Invalid input.
    """
    try:
        from decimal import Decimal
        from ..llm.agent_tools import (
            LoanAmortizationInput,
            loan_amortization_tool,
            calculate_loan_amortization,
            _parse_percent_like,
        )

        p = LoanAmortizationInput(**payload)
        md = loan_amortization_tool(**p.model_dump())
        r = calculate_loan_amortization(
            Decimal(str(p.principal)),
            float(_parse_percent_like(p.annual_rate_percent)),
            p.years,
            p.payments_per_year,
        )
        return {
            "tool": "loan_amortization",
            "ok": True,
            "data": {"markdown": md, **r},
            "source": _tool_source("loan_amortization", "Loan Amortization"),
        }
    except HTTPException:
        raise
    except Exception as e:
        _bad_request(f"Invalid loan-amortization input: {e!s}")


@calc_api_router.post("/npv")
def http_npv(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute Net Present Value (NPV) of a series of cashflows.

    - Expects per-period discount rate.
    - Cashflows are given as an ordered list.

    Args:
        payload: JSON body validated by 'NPVInput'.

    Returns:
        Dict with:
          - npv (float)
          - rate_percent_per_period (float)
          - markdown explanation

    Raises:
        HTTPException 400: Invalid input.
    """
    try:
        from decimal import Decimal
        from ..llm.agent_tools import (
            NPVInput,
            npv_tool,
            npv as npv_core,
            _parse_percent_like,
        )

        p = NPVInput(**payload)
        md = npv_tool(p.rate_percent_per_period, p.cashflows)
        rate = float(_parse_percent_like(p.rate_percent_per_period))
        value = float(npv_core(rate, [Decimal(str(x)) for x in p.cashflows]))
        return {
            "tool": "npv",
            "ok": True,
            "data": {
                "markdown": md,
                "npv": value,
                "rate_percent_per_period": rate,
            },
            "source": _tool_source("npv", "NPV Calculator"),
        }
    except HTTPException:
        raise
    except Exception as e:
        _bad_request(f"Invalid NPV input: {e!s}")


@calc_api_router.post("/cagr")
def http_cagr(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute Compound Annual Growth Rate (CAGR).

    - Validates with 'CAGRInput'.
    - Normalizes legacy shapes from core calculation.
    - Returns percent CAGR in data["CAGR"].

    Args:
        payload: JSON body with start_value, end_value, and years.

    Returns:
        Dict with tool metadata and CAGR result.

    Raises:
        HTTPException 400: Invalid input.
    """
    try:
        from ..llm.agent_tools import CAGRInput, cagr_tool, cagr as cagr_core

        p = CAGRInput(**payload)

        md = cagr_tool(p.start_value, p.end_value, p.years)
        res = cagr_core(p.start_value, p.end_value, p.years)

        if isinstance(res, dict):
            if "cagr_decimal" in res:
                rate_dec = float(res["cagr_decimal"])
                rate_pct = float(res.get("cagr_percent", rate_dec * 100.0))
            else:
                val = res.get("rate") or res.get("value") or res.get("cagr")
                if val is None:
                    _bad_request("cagr(...) returned dict without numeric rate")
                rate_dec = float(val)
                rate_pct = rate_dec * 100.0
        else:
            rate_dec = float(res)
            rate_pct = rate_dec * 100.0

        return {
            "tool": "cagr",
            "ok": True,
            "data": {
                "markdown": md,
                "CAGR": rate_pct,
            },
            "source": _tool_source("cagr", "CAGR Calculator"),
        }
    except HTTPException:
        raise
    except Exception as e:
        _bad_request(f"Invalid CAGR input: {e!s}")
