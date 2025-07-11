import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns
from scipy.stats import chi2_contingency, mannwhitneyu, skew
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from skopt import BayesSearchCV
from lightgbm import LGBMClassifier
from sklearn.utils import resample
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_validate, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    recall_score,
    average_precision_score,
    matthews_corrcoef,
    precision_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    make_scorer,
)
from typing import List, Tuple, Dict, Any, Union, Hashable
from pandas import DataFrame
import re

# Utility functions for DataFrame inspection and analysis

def print_df_info(df: pd.DataFrame) -> None:
    """
    Print summary information about a pandas DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to inspect.

    Returns:
    --------
    None
        This function prints information to the console and returns nothing.
    """
    print("Dataframe info:")
    print(df.info())
    print("\nSize of dataframe (rows, columns):", df.shape)


def print_dup_info(df: pd.DataFrame, id_1: str, id_2: str) -> None:
    """
    Print duplicate information for a pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to analyze for duplicates.
    id_1 : str
        The name of the first column to check for duplicated values.
    id_2 : str
        The name of the second column to check for duplicated values.

    Returns
    -------
    None
        This function prints duplicate statistics to the console.
    """
    total_dups = df.duplicated(keep=False).sum()
    id1_dups = df.duplicated(subset=[id_1]).sum()
    id2_dups = df.duplicated(subset=[id_2]).sum()
    other_dups = df.drop([id_1, id_2], axis=1).duplicated(keep=False).sum()

    print(f"Duplicated rows in dataframe: {total_dups}")
    print(f"Duplicated '{id_1}' in dataframe: {id1_dups}")
    print(f"Duplicated '{id_2}' in dataframe: {id2_dups}")
    print(
        f"Number of duplicate values present in dataframe "
        f"excluding '{id_1}' and '{id_2}': {other_dups}"
    )


def get_missing_df_info(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze and report missing values in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to analyze.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        - missing: DataFrame with columns that have missing values,
                   including their count and percentage.
        - no_missing: DataFrame with columns that have no missing values.

    Notes
    -----
    Also prints:
    - Locations of empty strings ('') and 'XNA' values.
    - Percentage of rows with any missing values.
    """
    print("Empty strings:", np.where(df.map(lambda x: x == "")))
    print("XNA strings:", np.where(df.map(lambda x: x == "XNA")))
    missing_row_pct = 100 * df.isnull().any(axis=1).mean()
    print("Percentage of total rows with missing values:", missing_row_pct)

    missing_number = df.isnull().sum()
    missing_percent = (missing_number / len(df)) * 100

    missing = pd.DataFrame(
        {"Missing number": missing_number, "Missing percentage": missing_percent}
    ).sort_values(by="Missing percentage", ascending=False)

    no_missing = missing[missing["Missing percentage"] == 0]
    missing = missing[missing["Missing percentage"] > 0]

    return missing, no_missing


def add_dup_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify and count duplicated rows in a DataFrame, ignoring missing values.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame in which to count duplicated rows.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the duplicated rows and their counts,
        sorted in descending order of duplication frequency.
    """
    filled_features = df.fillna("__NA__")
    dup_counts = (
        filled_features.groupby(filled_features.columns.tolist())
        .size()
        .reset_index(name="count")
    )
    dup_counts = dup_counts[dup_counts["count"] > 1]
    dup_counts.sort_values(by="count", ascending=False, inplace=True)
    dup_counts.replace("__NA__", np.nan, inplace=True)
    dup_counts = dup_counts.infer_objects(copy=False)

    return dup_counts


# Plotting functions for categorical and numerical variables

def percent_dist(data: pd.DataFrame, var: str) -> None:
    """
    Plot the percentage distribution of a categorical variable.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame containing the data.
    var : str
        The name of the categorical variable to plot.

    Returns
    -------
    None
        This function displays a bar plot and returns nothing.
    """
    p = data[var].value_counts(normalize=True) * 100
    ax = sns.barplot(x=p.index, y=p.values)

    ax.bar_label(
        ax.containers[0],
        labels=[f"{round(v, 0)}%" for v in ax.containers[0].datavalues],
    )

    plt.title(f"Percentage distribution of '{var}'")
    plt.xlabel(var)
    plt.ylabel("Percentage (%)")
    plt.tight_layout()
    plt.show()


def plot_categorical_distribution(df: pd.DataFrame, col: str) -> None:
    """
    Plot the distribution of a categorical column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the column.
    col : str
        Column name to plot.
    """
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x=col)
    plt.title(f"Distribution of {col}")
    plt.show()


def explore_numerical(var: str, missing_df: pd.DataFrame, df: pd.DataFrame) -> None:
    """
    Explore a numerical variable by displaying missing value information,
    plotting a boxplot by the 'TARGET' variable, and a histogram of distribution.

    Parameters
    ----------
    var : str
        The name of the numerical variable to explore.
    missing_df : pd.DataFrame
        DataFrame containing missing value statistics (indexed by column names).
    df : pd.DataFrame
        The main DataFrame containing the actual data.

    Returns
    -------
    None
        This function prints missing value information and displays plots.
    """
    print(f"Exploring {var}")
    if var in missing_df.index:
        print(missing_df.loc[var], "\n")
    else:
        print("No missing values")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.boxplot(data=df, x=var, hue="TARGET", showmeans=True, ax=axes[0]).set(
        title=f"Boxplot of {var} by TARGET", xlabel=var
    )

    plot_numerical(df, var, bins=80, ax=axes[1], xlabel=var)

    plt.tight_layout()
    plt.show()


def explore_category(var: str, missing_df: pd.DataFrame, df: pd.DataFrame) -> None:
    """
    Explore a categorical variable by displaying missing value information
    and plotting its distribution by the 'TARGET' variable.

    Parameters
    ----------
    var : str
        The name of the variable to explore.
    missing_df : pd.DataFrame
        DataFrame containing missing value statistics (indexed by column names).
    df : pd.DataFrame
        The main DataFrame containing the actual data.

    Returns
    -------
    None
        This function prints missing value information and displays a plot.
    """
    print(f"Exploring {var}")
    if var in missing_df.index:
        print(missing_df.loc[var], "\n")
    else:
        print("No missing values")

    plot_category(df, var)


def plot_numerical(
    data: pd.DataFrame, var: str, bins: int, ax: plt.Axes, xlabel: str
) -> None:
    """
    Plot a histogram showing the percentage distribution of a numerical variable
    across different classes of the 'TARGET' variable.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame containing the data.
    var : str
        The name of the numerical variable to plot.
    bins : int
        The number of bins to use in the histogram.
    ax : plt.Axes
        The matplotlib Axes object on which to draw the plot.
    xlabel : str
        The label for the x-axis.

    Returns
    -------
    None
        This function creates a plot and returns nothing.
    """
    counts = data.groupby([var, "TARGET"]).size().reset_index(name="count")
    counts["percentage"] = counts["count"].transform(lambda x: x / x.sum() * 100)

    sns.histplot(
        data=counts,
        x=var,
        weights="percentage",
        hue="TARGET",
        multiple="stack",
        bins=bins,
        kde=True,
        ax=ax,
    ).set(title=f"{var} with TARGET")

    ax.set_ylabel("Percentage (%)")
    ax.set_xlabel(xlabel)


def plot_category(data: pd.DataFrame, var: str) -> None:
    """
    Plot the percentage distribution of a categorical variable segmented by 'TARGET'.
    Also annotates each bar with its value and stroke ratio below the x-axis.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame containing the data.
    var : str
        The name of the categorical variable to plot.

    Returns
    -------
    None
        This function displays a bar plot and returns nothing.
    """
    counts = (
        data.groupby([var, "TARGET"], observed=False).size().reset_index(name="count")
    )
    counts["percentage"] = counts["count"].transform(lambda x: x / x.sum() * 100)

    total_per_category = (
        data.groupby(var, observed=False).size().reset_index(name="total_count")
    )
    target_counts = (
        data[data["TARGET"] == 1]
        .groupby(var, observed=False)
        .size()
        .reset_index(name="target_count")
    )
    target_ratio = total_per_category.merge(target_counts, on=var, how="left").fillna(0)
    target_ratio["ratio"] = target_ratio["target_count"] / target_ratio["total_count"]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=var, y="percentage", hue="TARGET", data=counts, ax=ax)
    ax.set_title(f"Percentage Distribution of {var} by TARGET", fontsize=14)
    ax.set_ylabel("Percentage (%)", fontsize=12)

    for patch in ax.patches:
        height = patch.get_height()
        if height > 0:
            ax.text(
                patch.get_x() + patch.get_width() / 2.0,
                height + 0.5,
                f"{height:.1f}%",
                ha="center",
                fontsize=11,
            )

    ax.set_ylim(0, max(counts["percentage"]) + 5)
    plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
    plt.tight_layout()


def plot_qq(variable: pd.Series, ax: plt.Axes) -> None:
    """
    Plot a Q-Q plot to assess the normality of a variable.

    Parameters
    ----------
    variable : pd.Series
        The numerical variable to be analyzed for normality.
    ax : plt.Axes
        The matplotlib axes object on which to plot the Q-Q plot.

    Returns
    -------
    None
        Displays the Q-Q plot on the provided axes.
    """
    stats.probplot(variable, dist="norm", plot=ax)
    ax.set_title(f"Q-Q Plot for {variable.name}")


def plot_multiple_qq(features: List[str], df: pd.DataFrame, n_cols: int = 4) -> None:
    """
    Plot multiple Q-Q plots for a list of numerical features in a DataFrame.

    Parameters
    ----------
    features : List[str]
        List of column names (features) to plot.
    df : pd.DataFrame
        The DataFrame containing the numerical features.
    n_cols : int, optional
        Number of columns for subplot layout (default is 4).

    Returns
    -------
    None
        Displays a grid of Q-Q plots for the selected features.
    """
    n = len(features)
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    axes = axes.flatten()

    for i, feature in enumerate(features):
        ax = axes[i]
        plot_qq(df[feature], ax)
        ax.set_title(feature)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()



# Statistical tests and analysis functions

def chi_squared_test(df: pd.DataFrame, var1: str, var2: str) -> None:
    """
    Perform a Chi-squared test of independence between two categorical variables.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the categorical variables.
    var1 : str
        The name of the first categorical variable.
    var2 : str
        The name of the second categorical variable.

    Returns
    -------
    None
        Prints the results of the test including the contingency table,
        chi-squared statistic, p-value, and degrees of freedom.
    """
    alpha = 0.05
    contingency_table = pd.crosstab(df[var1], df[var2])
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)

    print("Contingency table:\n", contingency_table)
    print(f"Chi-Squared Statistic: {chi2:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Degrees of Freedom: {dof}")

    if p_value < alpha:
        print(
            f"Reject the null hypothesis - There is a significant relationship "
            f"between '{var1}' and '{var2}'."
        )
    else:
        print(
            f"Fail to reject the null hypothesis - No significant relationship "
            f"between '{var1}' and '{var2}'."
        )


def mann_whitney_test(df: pd.DataFrame, var: str, c_var: str) -> Tuple[float, float]:
    """
    Perform the Mann-Whitney U test to assess whether two independent samples
    (split by a binary variable) have different distributions.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data.
    var : str
        The binary variable used to split the data into two groups.
    c_var : str
        The numerical variable for which distributions are compared.

    Returns
    -------
    Tuple[float, float]
        The lower and upper bounds of the 95% confidence interval for the Mann-Whitney U statistic.
    """
    alpha = 0.05
    n_iterations = 1000

    group_0 = df[df[var] == 0][c_var]
    group_1 = df[df[var] == 1][c_var]

    U_statistic, p_value = mannwhitneyu(group_0, group_1, alternative="two-sided")
    print(f"U-statistic: {U_statistic}")
    print(f"P-value: {p_value}")

    if p_value < alpha:
        print(
            f"Reject the null hypothesis - There is a relationship between '{var}' and '{c_var}'."
        )
    else:
        print(
            f"Fail to reject the null hypothesis - No significant relationship between '{var}' and '{c_var}'."
        )

    U_stats = []
    for _ in range(n_iterations):
        sample1 = resample(group_0)
        sample2 = resample(group_1)
        U_stat, _ = mannwhitneyu(sample1, sample2, alternative="two-sided")
        U_stats.append(U_stat)

    lower_bound = np.percentile(U_stats, 2.5)
    upper_bound = np.percentile(U_stats, 97.5)

    print(
        f"95% Confidence Interval for Mann-Whitney U statistic: ({lower_bound:.2f}, {upper_bound:.2f})"
    )

    return lower_bound, upper_bound


def mann_whitney_alt_greater(
    df: pd.DataFrame, var: str, c_var: str
) -> Tuple[float, float]:
    """
    Perform the Mann-Whitney U test (one-sided, alternative="greater") to evaluate
    if the distribution of group 1 is significantly greater than group 0.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data.
    var : str
        The binary variable used to split the data into two groups.
    c_var : str
        The continuous variable to compare between the two groups.

    Returns
    -------
    Tuple[float, float]
        The lower and upper bounds of the 95% confidence interval for
        the Mann-Whitney U statistic based on bootstrapping.
    """
    alpha = 0.05
    n_iterations = 1000

    group_0 = df[df[var] == 0][c_var]
    group_1 = df[df[var] == 1][c_var]

    U_statistic, p_value = mannwhitneyu(group_0, group_1, alternative="greater")
    print(f"U-statistic: {U_statistic}")
    print(f"P-value: {p_value}")

    if p_value < alpha:
        print(
            "Reject the null hypothesis - group 1 tends toward larger values "
            "than group 0 in the distribution of the variable."
        )
    else:
        print(
            "Fail to reject the null hypothesis - no evidence that group 1 "
            "tends toward larger values than group 0."
        )

    U_stats = []
    for _ in range(n_iterations):
        sample_0 = resample(group_0)
        sample_1 = resample(group_1)
        U_stat, _ = mannwhitneyu(sample_0, sample_1, alternative="greater")
        U_stats.append(U_stat)

    lower_bound = np.percentile(U_stats, 2.5)
    upper_bound = np.percentile(U_stats, 97.5)

    print(
        f"95% Confidence Interval for Mann-Whitney U statistic: "
        f"({lower_bound:.2f}, {upper_bound:.2f})"
    )

    return lower_bound, upper_bound


def evaluate_vif(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factors (VIF) for a set of features.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the features to check for multicollinearity.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing each feature, its corresponding VIF value,
        and a multicollinearity severity classification:
        - 'none' (≤ 1)
        - 'small' (1–5]
        - 'moderate' (5–10]
        - 'severe' (> 10)
    """
    vif_data = pd.DataFrame()
    vif_data["feature"] = df.columns
    vif_data["VIF"] = [
        variance_inflation_factor(df.values, i) for i in range(len(df.columns))
    ]

    conditions = [
        (vif_data["VIF"] <= 1.0),
        (vif_data["VIF"] > 1.0) & (vif_data["VIF"] <= 5.0),
        (vif_data["VIF"] > 5.0) & (vif_data["VIF"] <= 10.0),
        (vif_data["VIF"] > 10.0),
    ]
    values = ["none", "small", "moderate", "severe"]

    vif_data["multicollinearity_warning_strength"] = np.select(
        conditions, values, default="unknown"
    )

    return vif_data.sort_values(by="VIF").reset_index(drop=True)


# Data prepcrocessing and transformation functions

def cap_outliers(
    df: pd.DataFrame,
    columns: List[str],
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> pd.DataFrame:
    """
    Cap outliers in specified columns of a DataFrame by clipping values
    outside given quantile thresholds.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : List[str]
        List of column names to cap outliers for.
    lower_q : float, optional
        Lower quantile threshold (default is 0.01).
    upper_q : float, optional
        Upper quantile threshold (default is 0.99).

    Returns
    -------
    pd.DataFrame
        A copy of the DataFrame with outliers capped in the specified columns.
    """
    df_capped = df.copy()
    for col in columns:
        lower = df[col].quantile(lower_q)
        upper = df[col].quantile(upper_q)
        df_capped[col] = df[col].clip(lower=lower, upper=upper)
    return df_capped


def get_columns_with_outliers(
    df: pd.DataFrame, threshold: float = 0.01, k: float = 1.5
) -> List[str]:
    """
    Identify numeric columns in the DataFrame that have outliers based on
    the IQR method and a specified threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    threshold : float, optional
        Minimum ratio of outliers (default is 0.01).
    k : float, optional
        Multiplier for the IQR to define outlier boundaries (default is 1.5).

    Returns
    -------
    List[str]
        List of column names containing outliers exceeding the threshold.
    """
    outlier_columns = []
    numeric_cols = df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        if df[col].value_counts().count() > 2:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - k * iqr
            upper = q3 + k * iqr
            outliers = ((df[col] < lower) | (df[col] > upper)).sum()
            ratio = outliers / df[col].notnull().sum()
            if ratio > threshold:
                outlier_columns.append(col)
    return outlier_columns


def log_transform_column(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Apply log1p transformation to specified columns in the DataFrame,
    creating new columns with suffix '_log'.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : List[str]
        List of column names to log-transform.

    Returns
    -------
    pd.DataFrame
        DataFrame with new log-transformed columns added.
    """
    df_transformed = df.copy()
    for col in columns:
        df_transformed[f"{col}_log"] = np.log1p(df[col] + 1)
    return df_transformed


def suggest_log_transform_columns(
    df: pd.DataFrame,
    skew_threshold: float = 1.0,
    max_unique: int = 20,
) -> List[str]:
    """
    Suggest numeric columns for log transformation based on skewness and
    uniqueness criteria.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    skew_threshold : float, optional
        Minimum skewness threshold to suggest log transform (default is 1.0).
    max_unique : int, optional
        Maximum number of unique values allowed to exclude likely categorical
        variables (default is 20).

    Returns
    -------
    List[str]
        List of column names suggested for log transformation.
    """
    suggested = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if df[col].min() < 0:
            continue
        if df[col].nunique() <= max_unique:
            continue
        col_skew = skew(df[col].dropna())
        if col_skew > skew_threshold:
            suggested.append(col)

    return suggested


def plot_log_transformed_distributions(
    df: pd.DataFrame,
    columns: List[str],
) -> None:
    """
    Plot original and log1p-transformed distributions side-by-side for
    specified columns, skipping columns with non-positive values.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : List[str]
        List of column names to plot.

    Returns
    -------
    None
        Displays the plots.
    """
    for col in columns:
        if (df[col] + 1 <= 0).any():
            print(f"Skipping {col}: contains non-positive values.")
            continue

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(df[col].dropna(), kde=True, ax=axes[0], bins=30, color="skyblue")
        axes[0].set_title(f"Original: {col}")
        axes[0].set_xlabel(col)

        log_data = np.log1p(df[col].dropna())
        sns.histplot(log_data, kde=True, ax=axes[1], bins=30, color="orange")
        axes[1].set_title(f"Log-Transformed: {col}")
        axes[1].set_xlabel(f"log1p({col})")

        plt.tight_layout()
        plt.show()


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean DataFrame column names by replacing any character
    that is not a letter, number, or underscore with an underscore.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with original column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with cleaned column names.
    """
    df = df.copy()
    df.columns = [re.sub(r"[^\w_]", "_", col) for col in df.columns]
    return df

# Feature evaluation and selection functions

def evaluate_feature_counts(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: Union[pd.Series, list, pd.DataFrame],
    ranked_features: List[str],
    cv: StratifiedKFold,
    metric: str = "roc_auc",
    step: int = 50,
    patience: int = 10,
) -> pd.DataFrame:
    """
    Evaluate model performance (AUC by default) using increasing subsets of
    features, stopping early if performance doesn't improve for a given number
    of iterations.

    Parameters
    ----------
    model : BaseEstimator
        The sklearn-compatible model to evaluate.
    X : pd.DataFrame
        Feature dataset.
    y : Union[pd.Series, list, pd.DataFrame]
        Target variable.
    ranked_features : List[str]
        List of feature names ordered by importance/ranking.
    cv : StratifiedKFold
        Cross-validation strategy (e.g., StratifiedKFold).
    metric : str, optional
        Scoring metric for cross-validation (default is 'roc_auc').
    step : int, optional
        Number of features to increase at each iteration (default is 50).
    patience : int, optional
        Number of non-improving iterations to tolerate before stopping (default is 10).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['num_features', 'auc'] showing performance by
        number of features used.
    """
    results = []
    best_score = -float("inf")
    no_improve_count = 0

    for k in range(10, len(ranked_features) + 1, step):
        print(f"Evaluating top {k} features...")
        selected = ranked_features[:k]
        scores = cross_val_score(model, X[selected], y, cv=cv, scoring=metric)
        mean_score = scores.mean()

        results.append((k, mean_score))

        if mean_score > best_score:
            best_score = mean_score
            no_improve_count = 0
        else:
            no_improve_count += 1
            print(f"No improvement for {no_improve_count} iteration(s)")

        if no_improve_count >= patience:
            print(f"Stopping early after {patience} non-improving iterations.")
            break

    return pd.DataFrame(results, columns=["num_features", "auc"])


def plot_feature_count_vs_auc(
    results: List[Tuple[int, float]], title: str = "Feature Count vs AUC"
) -> None:
    """
    Plot the relationship between number of features and AUC scores.

    Parameters
    ----------
    results : List[Tuple[int, float]]
        List of tuples where each tuple contains (number_of_features, AUC_score).
    title : str, optional
        Title of the plot (default is "Feature Count vs AUC").

    Returns
    -------
    None
        Displays the plot.
    """
    ks, scores = zip(*results)
    plt.plot(ks, scores, marker="o")
    plt.xlabel("Number of Features")
    plt.ylabel("AUC")
    plt.title(title)
    plt.grid(True)
    plt.show()


def evaluate_feature_subset(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    features: List[str],
    class_weight_dict: Union[Dict[int, float], str],
) -> float:
    """
    Train an LGBMClassifier on a subset of features and evaluate it using ROC AUC score on validation data.

    Parameters:
    ----------
    X_train : pd.DataFrame
        Training feature data.
    y_train : pd.Series
        Training target labels.
    X_val : pd.DataFrame
        Validation feature data.
    y_val : pd.Series
        Validation target labels.
    features : List[str]
        List of feature names to use for training and validation.
    class_weight_dict : dict or 'balanced'
        Class weight dictionary or 'balanced' to handle class imbalance.

    Returns:
    -------
    float
        ROC AUC score of the model on the validation data.
    """
    model = LGBMClassifier(
        random_state=42,
        class_weight=class_weight_dict,
        objective="binary",
        metric="auc",
        verbose=-1,
    )
    model.fit(X_train[features], y_train)
    preds = model.predict_proba(X_val[features])[:, 1]
    return roc_auc_score(y_val, preds)


def custom_auc_score(
    estimator: BaseEstimator,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
) -> float:
    """
    Compute the ROC AUC score for a given estimator and dataset.

    Parameters
    ----------
    estimator : BaseEstimator
        A fitted classifier with either a predict_proba or decision_function method.
    X : pd.DataFrame or np.ndarray
        Feature matrix to predict on.
    y : pd.Series or np.ndarray
        True binary labels.

    Returns
    -------
    float
        ROC AUC score.
    """
    if hasattr(estimator, "predict_proba"):
        y_scores = estimator.predict_proba(X)[:, 1]
    else:
        y_scores = estimator.decision_function(X)

    return roc_auc_score(y, y_scores)


def custom_pr_auc_score(
    estimator: BaseEstimator,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
) -> float:
    """
    Compute the Average Precision (PR AUC) score for a given estimator and dataset.

    Parameters
    ----------
    estimator : BaseEstimator
        A fitted classifier with either a predict_proba or decision_function method.
    X : pd.DataFrame or np.ndarray
        Feature matrix to predict on.
    y : pd.Series or np.ndarray
        True binary labels.

    Returns
    -------
    float
        Average Precision (PR AUC) score.
    """
    if hasattr(estimator, "predict_proba"):
        y_scores = estimator.predict_proba(X)[:, 1]
    else:
        y_scores = estimator.decision_function(X)

    return average_precision_score(y, y_scores)


def get_results(
    models: Dict[str, Any], x: pd.DataFrame, y: pd.Series, cv: Any
) -> pd.DataFrame:
    """
    Evaluate multiple models on given data using cross-validation and return
    a DataFrame summarizing multiple scoring metrics.

    Parameters
    ----------
    models : dict
        Dictionary mapping model names to estimator objects.
    x : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    cv : cross-validation generator or int
        Cross-validation splitting strategy.

    Returns
    -------
    pd.DataFrame
        DataFrame containing mean ± std of each scoring metric per model,
        sorted by AUC in descending order.
    """
    scoring = {
        "auc": custom_auc_score,
        "PR_AUC": custom_pr_auc_score,
        "recall": make_scorer(recall_score, average="weighted"),
        "precision": make_scorer(precision_score, average="weighted", zero_division=0),
        "accuracy": make_scorer(accuracy_score),
        "weighted_f1": make_scorer(f1_score, average="weighted"),
        "MCC": make_scorer(matthews_corrcoef),
    }

    results = []
    for name, model in models.items():
        scores = cross_validate(model, x, y, cv=cv, scoring=scoring)
        result = {"Model": name}
        for metric in scoring.keys():
            mean_val = scores[f"test_{metric}"].mean()
            std_val = scores[f"test_{metric}"].std()
            result[metric.replace("_", " ").upper()] = f"{mean_val:.4f} ± {std_val:.4f}"
        results.append(result)

    results_df = pd.DataFrame(results)
    results_df.sort_values(by="AUC", ascending=False, inplace=True)
    return results_df


def get_results_auc(
    models: Dict[str, BaseEstimator],
    x: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
) -> pd.DataFrame:
    """
    Evaluate multiple models using cross-validation and return a summary of ROC AUC scores.

    Parameters
    ----------
    models : dict
        Dictionary with model names as keys and model instances as values.
    x : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    cv : StratifiedKFold
        Cross-validation splitting strategy.

    Returns
    -------
    pd.DataFrame
        DataFrame summarizing mean and standard deviation of ROC AUC scores per model.
    """
    scoring = {
        "auc": custom_auc_score,
    }

    results = []
    for name, model in models.items():
        scores = cross_validate(
            model, x, y, cv=cv, scoring=scoring, return_estimator=False
        )
        mean_auc = scores["test_auc"].mean()
        std_auc = scores["test_auc"].std()
        results.append(
            {
                "Model": name,
                "ROC_AUC_Mean": mean_auc,
                "ROC_AUC_Std": std_auc,
            }
        )

    return pd.DataFrame(results)


def format_cm_labels(cm: np.ndarray) -> np.ndarray:
    """
    Format confusion matrix labels for better visualization.

    Parameters
    ----------
    cm : np.ndarray
        2x2 confusion matrix.

    Returns
    -------
    np.ndarray
        2x2 array of formatted labels combining class names, counts, and percentages.
    """
    group_names = ["True Neg", "False Pos", "False Neg", "True Pos"]
    group_counts = [f"{value:0.0f}" for value in cm.flatten()]
    group_percentages = [f"{value:.2%}" for value in cm.flatten() / np.sum(cm)]
    labels = [
        f"{name}\n{count}\n{percent}"
        for name, count, percent in zip(group_names, group_counts, group_percentages)
    ]
    return np.array(labels).reshape(2, 2)


def bayes_hyperparam_tuning(
    model: BaseEstimator,
    x: pd.DataFrame,
    y: pd.Series,
    param_space: Dict[str, object],
) -> BaseEstimator:
    """
    Perform Bayesian hyperparameter tuning using BayesSearchCV, optimizing
    Matthews correlation coefficient as the evaluation metric.

    Parameters
    ----------
    model : BaseEstimator
        The machine learning model to tune.
    x : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    param_space : dict
        Dictionary defining the search space for hyperparameters.

    Returns
    -------
    BaseEstimator
        The model with the best hyperparameters found during the search.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    bayes_search = BayesSearchCV(
        estimator=model,
        search_spaces=param_space,
        n_iter=50,
        cv=cv,
        scoring="matthews_corrcoef",
        n_jobs=-1,
        verbose=1,
        random_state=42,
    )
    bayes_search.fit(x, y)
    print("Best Parameters:", bayes_search.best_params_)
    return bayes_search.best_estimator_


def plot_cdf_by_target(
    df: pd.DataFrame,
    feature: str,
    ax: Axes,
    target_col: str = "TARGET",
) -> None:
    """
    Plot the cumulative distribution functions (CDFs) of a numeric feature
    for two classes defined by the target column, and display the
    Kolmogorov-Smirnov (KS) statistic between them.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data.
    feature : str
        The feature/column name to plot the CDF for.
    ax : matplotlib.axes.Axes
        Matplotlib Axes object to plot on.
    target_col : str, optional
        Name of the target column defining two classes (default is 'TARGET').

    Returns
    -------
    None
        Displays the plot on the provided Axes.
    """
    data_0 = df[df[target_col] == 0][feature].dropna()
    data_1 = df[df[target_col] == 1][feature].dropna()

    x0 = np.sort(data_0)
    y0 = np.arange(1, len(x0) + 1) / len(x0)

    x1 = np.sort(data_1)
    y1 = np.arange(1, len(x1) + 1) / len(x1)

    all_x = np.sort(np.concatenate([x0, x1]))
    y0_interp = np.searchsorted(x0, all_x, side="right") / len(x0)
    y1_interp = np.searchsorted(x1, all_x, side="right") / len(x1)

    ks_stat = np.max(np.abs(y0_interp - y1_interp))
    ks_idx = np.argmax(np.abs(y0_interp - y1_interp))
    ks_x = all_x[ks_idx]

    ax.vlines(
        ks_x, y0_interp[ks_idx], y1_interp[ks_idx], color="black", linestyle="-", lw=2
    )
    ax.text(
        0.05,
        0.05,
        f"KS = {ks_stat:.3f}",
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="black"),
    )

    ax.plot(x0, y0, label="TARGET = 0", color="blue")
    ax.plot(x1, y1, label="TARGET = 1", color="orange")

    ax.set_title(f"Cumulative Distribution: {feature}", fontsize=12)
    ax.set_xlabel(feature)
    ax.set_ylabel("CDF")
    ax.legend()

