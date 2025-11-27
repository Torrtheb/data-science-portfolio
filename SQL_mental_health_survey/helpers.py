from __future__ import annotations

import pandas as pd
import sqlite3 as sql
import math as m
import textwrap
from typing import Final
from matplotlib.axes import Axes


database = "mental_health.sqlite"
connection = sql.connect(database)


def question_missing_stats(
    question_id: int,
    total: int,
    country_filter: bool = False,
) -> pd.DataFrame:
    """
    Compute missingness for a single survey question.

    Missing is defined as AnswerText in {"-1", "", " "} (after stripping)
    or a null value. The function can optionally restrict to respondents
    in the US/UK/Canada.

    Parameters
    ----------
    question_id : int
        QuestionID of the survey item to inspect.
    total : int
        Denominator for pct_missing_of_total, e.g. total_participants,
        total_top3 (US/UK/CA only), or total_usa.
    country_filter : bool, default False
        If True, restrict to respondents whose work country is in
        {"United States of America", "United States", "United Kingdom", "Canada"}.

    Returns
    -------
    pandas.DataFrame
        One-row DataFrame with columns:
        ["question_id", "n_answers", "n_missing",
         "pct_missing_of_answers", "pct_missing_of_total"].
    """
    if country_filter:
        q = f"""
            SELECT a.UserID, a.AnswerText
            FROM Answer a
            JOIN Answer c USING (UserID)
            WHERE a.QuestionID = {question_id}
              AND c.QuestionID = 50
              AND c.AnswerText IN (
                  'United States of America', 'United States', 'United Kingdom', 'Canada'
              )
        """
    else:
        q = f"""
            SELECT UserID, AnswerText
            FROM Answer
            WHERE QuestionID = {question_id}
        """

    df = pd.read_sql(q, connection)

    s = df["AnswerText"].astype(str).str.strip()
    missing = s.isin({"-1", "", " "}) | s.isna()

    n_answers: Final[int] = len(df)
    n_missing: Final[int] = int(missing.sum())

    return pd.DataFrame(
        {
            "question_id": [question_id],
            "n_answers": [n_answers],
            "n_missing": [n_missing],
            "pct_missing_of_answers": [
                (n_missing / n_answers) * 100 if n_answers else float("nan")
            ],
            "pct_missing_of_total": [
                (n_missing / total) * 100 if total else float("nan")
            ],
        }
    )


total_participants_USA_UK_Canada = 2294


def wrap_labels(
    ax: Axes,
    width: int,
    break_long_words: bool = False,
) -> None:
    """
    Wrap x-axis tick labels to a fixed character width.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes whose x-axis tick labels should be wrapped.
    width : int
        Maximum line width (characters) before wrapping.
    break_long_words : bool, default False
        Whether to break long words that exceed `width`.
    """
    tick_locs = ax.get_xticks()
    raw_labels = [lbl.get_text() for lbl in ax.get_xticklabels()]

    wrapped_labels = [
        textwrap.fill(text, width=width, break_long_words=break_long_words)
        for text in raw_labels
    ]
    ax.set_xticks(tick_locs)
    ax.set_xticklabels(wrapped_labels, rotation=0)


def ME(
    x: float,
    n: int = total_participants_USA_UK_Canada,
    z: float = 1.96,
) -> float | str:
    """
    Return the 95% margin of error for a proportion estimated from a count.

    Parameters
    ----------
    x : SupportsFloat
        Count in the category of interest (e.g. number of 'Yes' responses).
    n : int
        Total number of observations in the population or subgroup.
    z : float, default 1.96
        Z-score for the desired confidence level (1.96 for 95% CI).

    Returns
    -------
    float or str
        Margin of error for the estimated proportion, or a string
        message if the normal approximation is not appropriate
        (i.e. n * p_hat <= 5 or n * (1 - p_hat) <= 5).
    """
    n = int(n)
    if n <= 0:
        return "Not large enough sample size"

    phat = float(x) / float(n)
    if n * phat <= 5 or n * (1 - phat) <= 5:
        return "Not large enough sample size"

    se = m.sqrt((phat * (1 - phat)) / n)
    return z * se
