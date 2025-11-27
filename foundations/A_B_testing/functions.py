import numpy as np
from typing import List, Tuple
from typing import Any
import numpy as np
import pandas as pd
from matplotlib.axes import Axes


def plot_cdf_by_group(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    group_a: Any,
    group_b: Any,
    ax: Axes,
) -> None:
    """
    Plot CDFs of a numeric feature for two groups and show the KS statistic.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing the feature and group column.
    value_col : str
        Name of the numeric column to plot (e.g. 'SalesInThousands').
    group_col : str
        Column defining the groups (e.g. 'Promotion', 'version').
    group_a, group_b : Any
        The two group values to compare.
    ax : matplotlib.axes.Axes
        Axes to plot on.
    """

    data_a = df[df[group_col] == group_a][value_col].dropna()
    data_b = df[df[group_col] == group_b][value_col].dropna()

    x_a = np.sort(data_a)
    y_a = np.arange(1, len(x_a) + 1) / len(x_a)

    x_b = np.sort(data_b)
    y_b = np.arange(1, len(x_b) + 1) / len(x_b)

    all_x = np.sort(np.concatenate([x_a, x_b]))
    y_a_interp = np.searchsorted(x_a, all_x, side="right") / len(x_a)
    y_b_interp = np.searchsorted(x_b, all_x, side="right") / len(x_b)

    ks_stat = np.max(np.abs(y_a_interp - y_b_interp))
    ks_idx = np.argmax(np.abs(y_a_interp - y_b_interp))
    ks_x = all_x[ks_idx]
    ax.vlines(
        ks_x,
        y_a_interp[ks_idx],
        y_b_interp[ks_idx],
        color="black",
        linestyle="-",
        lw=2,
    )
    ax.text(
        0.05,
        0.05,
        f"KS = {ks_stat:.3f}",
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="black"),
    )

    ax.plot(x_a, y_a, label=f"{group_col} = {group_a}", color="blue")
    ax.plot(x_b, y_b, label=f"{group_col} = {group_b}", color="orange")

    ax.set_title(f"Cumulative Distribution: {value_col} by {group_col}", fontsize=12)
    ax.set_xlabel(value_col)
    ax.set_ylabel("CDF")
    ax.legend()


def bootstrap_means(
    group_a: List[int], group_b: List[int], n_iterations=10000
) -> Tuple[List[float]]:
    """
    Resamples randomly with replacement a given number of times from each of two lists of integers to compute the means of each group.

    Parameters:
    group_a (list of integers): the first group of numbers.
    group_b (list of integers): the second group of numbers.
    n_iterations=10000: the number of times to resample from a group, where the default is set to 10000.

    Returns:
    Tuple[np.array[float]]: A tuple composed of two arrays with floating point numbers which are the mean of each resample for each group.

    """
    means_a = []
    means_b = []

    for _ in range(n_iterations):
        resample_a = np.random.choice(group_a, size=len(group_a), replace=True)
        resample_b = np.random.choice(group_b, size=len(group_b), replace=True)

        means_a.append(np.mean(resample_a))
        means_b.append(np.mean(resample_b))

    return np.array(means_a), np.array(means_b)


def bootstrap_3means(
    group_a: List[float], group_b: List[float], group_c: List[float], n_iterations=10000
) -> Tuple[List[float]]:
    """
    Resamples randomly with replacement a given number of times from each of two lists of integers to compute the means of each group (added to lists: means_ab, means_ac, means_bc).

    Parameters:
    group_a (list of floats): the first group of numbers.
    group_b (list of floats): the second group of numbers.
    group_c (list of floats): the third group of numbers.
    n_iterations=10000: the number of times to resample from a group, where the default is set to 10000.

    Returns:
    Tuple[List[float]]: A tuple composed of two arrays with floating point numbers which are the mean of each resample for each group.

    """

    means_ab = []
    means_ac = []
    means_bc = []

    for _ in range(n_iterations):
        resample_a = np.random.choice(group_a, size=len(group_a), replace=True)
        resample_b = np.random.choice(group_b, size=len(group_b), replace=True)
        resample_c = np.random.choice(group_c, size=len(group_c), replace=True)

        means_ab.append(np.mean(resample_a) - np.mean(resample_b))
        means_ac.append(np.mean(resample_a) - np.mean(resample_c))
        means_bc.append(np.mean(resample_b) - np.mean(resample_c))

    return np.array(means_ab), np.array(means_ac), np.array(means_bc)


def bootstrap_medians(
    group_a: List[int], group_b: List[int], n_iterations=10000
) -> Tuple[List[float]]:
    """
    Resamples randomly with replacement a given number of times from each of two lists of integers to compute the medians of each group.

    Parameters:
    group_a (list of integers): the first group of numbers.
    group_b (list of integers): the second group of numbers.
    n_iterations=10000: the number of times to resample from a group, where the default is set to 10000.

    Returns:
    Tuple[np.array[float]]: A tuple composed of two arrays with floating point numbers which are the median of each resampled group for each input group.

    """
    medians_a = []
    medians_b = []

    for _ in range(n_iterations):
        resample_a = np.random.choice(group_a, size=len(group_a), replace=True)
        resample_b = np.random.choice(group_b, size=len(group_b), replace=True)

        medians_a.append(np.median(resample_a))
        medians_b.append(np.median(resample_b))

    return np.array(medians_a), np.array(medians_b)


def bootstrap_3medians(
    group_a: List[float], group_b: List[float], group_c: List[float], n_iterations=10000
) -> Tuple[List[float]]:
    """
    Resamples randomly with replacement a given number of times from each of two lists of integers to compute the medians of each group (kept in lists: medians_ab, medians_ac, medians_bc).

    Parameters:
    group_a (list of floats): the first group of numbers.
    group_b (list of floats): the second group of numbers.
    group_c (list of floats): the third group of numbers.
    n_iterations=10000: the number of times to resample from a group, where the default is set to 10000.

    Returns:
    Tuple[np.array[float]]: A tuple composed of two arrays with floating point numbers which are the median of each resampled group for each input group.

    """
    medians_ab = []
    medians_ac = []
    medians_bc = []

    for _ in range(n_iterations):
        resample_a = np.random.choice(group_a, size=len(group_a), replace=True)
        resample_b = np.random.choice(group_b, size=len(group_b), replace=True)
        resample_c = np.random.choice(group_c, size=len(group_c), replace=True)

        medians_ab.append(np.median(resample_a) - np.median(resample_b))
        medians_ac.append(np.median(resample_a) - np.median(resample_c))
        medians_bc.append(np.median(resample_b) - np.median(resample_c))

    return np.array(medians_ab), np.array(medians_ac), np.array(medians_bc)
