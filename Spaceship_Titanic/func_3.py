from typing import Tuple
import matplotlib.pylab as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import chi2_contingency, mannwhitneyu
from sklearn.utils import resample
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    recall_score,
    average_precision_score,
    matthews_corrcoef
)
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, make_scorer
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from skopt import BayesSearchCV
from statsmodels.stats.outliers_influence import variance_inflation_factor


def plot_qq(variable: pd.Series, ax: plt.Axes) -> None:
    """
    Plot a Q-Q plot to check the normality of a given variable.

    Args:
        variable (pd.Series): The variable to be analyzed for normality.
        ax (plt.Axes): The matplotlib axes on which to plot the Q-Q plot.

    Returns:
        None
    """
    stats.probplot(variable, dist="norm", plot=ax)
    ax.set_title(f"Q-Q Plot for {variable.name}")


def num_dist_ins(
    data: pd.DataFrame, var: str, bins: int, ax: plt.Axes, xlabel: str
) -> None:
    """
    This function creates a histogram showing the percentages of a variable (Series) in a DataFrame, in terms of the Transported variable (orange if the passenger was transported).
    Args:
    data (pd.DataFrame): The DataFrame containing the data.
    var (str): The name of the categorical variable to plot.
    bins (int): the number of bins to have in the histogram.
    ax (plt.Axes): The axes on which to plot the histogram.
    xlabel (str): The x label for the x axis.

    Returns:
    None
    """
    counts = data.groupby([var, "Transported"]).size().reset_index(name="count")
    counts["percentage"] = counts["count"].transform(lambda x: x / x.sum() * 100)
    sns.histplot(
        data=counts,
        x=var,
        weights="percentage",
        hue="Transported",
        multiple="stack",
        bins=bins,
        kde=True,
        ax=ax,
    ).set(title=f"{var} with Transported")
    ax.set_ylabel("Percentage (%)")
    ax.set_xlabel(f"{xlabel}")


def cat_dist_ins(data: pd.DataFrame, var: str, axes) -> None:
    """
    Plots the percentage distribution of a categorical variable by Transported (to be used for the Kaggle competiton 'Spaceship Titanic').
    Also annotates the stroke ratio per category below x-axis labels.

    Args:
    data (pd.DataFrame): The dataframe containing the data.
    var (str): The name of the categorical variable to plot.
    axes (matplotlib.axes.Axes): The axes object to plot on.

    Returns:
    None
    """
    counts = data.groupby([var, "Transported"]).size().reset_index(name="count")
    counts["percentage"] = counts["count"].transform(lambda x: x / x.sum() * 100)

    total_per_category = data.groupby(var).size().reset_index(name="total_count")
    transported_counts = (
        data[data["Transported"] == 1]
        .groupby(var)
        .size()
        .reset_index(name="transported_count")
    )
    transported_ratio = total_per_category.merge(
        transported_counts, on=var, how="left"
    ).fillna(0)
    transported_ratio["ratio"] = (
        transported_ratio["transported_count"] / transported_ratio["total_count"]
    )
    sns.barplot(x=var, y="percentage", hue="Transported", data=counts, ax=axes)
    axes.set_title(f"Percentage Distribution of {var} by transported", fontsize=14)
    axes.set_ylabel("Percentage (%)", fontsize=12)

    for p in axes.patches:
        height = p.get_height()
        if height > 0:
            axes.text(
                p.get_x() + p.get_width() / 2.0,
                height + 0.5,
                f"{height:.1f}%",
                ha="center",
                fontsize=11,
            )
    axes.set_ylim(0, max(counts["percentage"]) + 5)


def percent_dist(data: pd.DataFrame, var: str) -> None:
    """
    Plots the percentage distribution of a categorical variable in a DataFrame.

    Args:
    data (pd.DataFrame): The DataFrame containing the data.
    var (str): The name of the categorical variable to plot.

    Returns:
    None
    """
    p = data[var].value_counts(normalize=True) * 100
    ax = sns.barplot(x=p.index, y=p.values)
    ax.bar_label(
        ax.containers[0],
        labels=[f"{round(v, 0)}%" for v in ax.containers[0].datavalues],
    )
    plt.title(f"Percentage distribution of {var}")
    plt.xlabel(f"{var}")
    plt.ylabel("Percentage (%)")


def vif(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function creates and returns a dataframe for calculated variance inflation factors taken from an input dataframe's columns.

    Args:
    df (pd.DataFrame): this is the dataframe with variables to check for multicollinearity.

    Returns:
    vif_data: pd.Dataframe showing a list of the input variables, their respective variance inflation factors,
        and a scale of multicollinearity severity (none (<1), small (<5), moderate (5-10), or severe (> 10))
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
    vif_data["multicollinearity warning strength"] = np.select(
        conditions, values, default="Unknows"
    )
    return vif_data.sort_values(by="VIF")


def Mann_Whit(df: pd.DataFrame, var: str, c_var: str) -> Tuple[float, float]:
    """
    Perform the Mann-Whitney U test to determine if there is a significant difference
    in the annual income distribution between two groups defined by the binary variable 'var'.

    Args:
    df (pd.DataFrame): The dataframe containing the data.
     var (str): The numerical variable to evaluate.
    c_var (str): The binary variable to split the data into two groups.

    Returns:
    Tuple[float, float]: The lower and upper bounds of the 95% confidence interval for the Mann-Whitney U statistic.
    """
    alpha = 0.05
    n_iterations = 1000
    group_0 = df[df[var] == 0][c_var]
    group_1 = df[df[var] == 1][c_var]
    U_statistic, p_value = mannwhitneyu(group_0, group_1)
    print(f"U-statistic: {U_statistic}")
    print(f"P-value: {p_value}")
    if p_value < alpha:
        print(
            f"Reject the null hypothesis - There is a relationship between {var} and {c_var}"
        )
    else:
        print(
            f"Fail to reject the null hypothesis - There is no relationship between {var} and {c_var}"
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


def Mann_Whit_gr0_less(df: pd.DataFrame, var: str, c_var: str):
    """
    Perform the Mann-Whitney U test to determine if there is a significant difference
    in the annual income distribution between two groups defined by the binary variable 'var'.

    Args:
    df (pd.DataFrame): The dataframe containing the data.
    var (str): The numerical variable to evaluate.
    c_var (str): The binary variable to split the data into two groups.

    Returns:
    Tuple[float, float]: The lower and upper bounds of the 95% confidence interval for the Mann-Whitney U statistic.
    """
    alpha = 0.05
    n_iterations = 1000
    group_0 = df[df[var] == 0][c_var]
    group_1 = df[df[var] == 1][c_var]
    U_statistic, p_value = stats.mannwhitneyu(group_0, group_1, alternative="less")
    print(f"U-statistic: {U_statistic}")
    print(f"P-value: {p_value}")
    if p_value < alpha:
        print(
            f"Reject the null hypothesis - the distribution of those not transported tends towards smaller values than the distribution of those transported."
        )
    else:
        print(
            f"Fail to reject the null hypothesis - There is no evidence that the distribution of those not transported tends towards smaller values than the distribution of those transported."
        )

    U_stats = []
    for _ in range(n_iterations):
        sample1 = resample(group_0)
        sample2 = resample(group_1)
        U_stat, _ = stats.mannwhitneyu(sample1, sample2, alternative="less")
        U_stats.append(U_stat)
    lower_bound = np.percentile(U_stats, 2.5)
    upper_bound = np.percentile(U_stats, 97.5)
    print(
        f"95% Confidence Interval for Mann-Whitney U statistic: ({lower_bound:.2f}, {upper_bound:.2f})"
    )
    return lower_bound, upper_bound


def Mann_Whit_gr1_greater(
    df: pd.DataFrame, var: str, c_var: str
) -> Tuple[float, float]:
    """
    Perform the Mann-Whitney U test to determine if there is a significant difference
    in the annual income distribution between two groups defined by the binary variable 'var'.

    Parameters:
    df (pd.DataFrame): The dataframe containing the data.
    var (str): The numerical variable to evaluate.
    c_var (str): The binary variable to split the data into two groups.

    Returns:
    Tuple[float, float]: The lower and upper bounds of the 95% confidence interval for the Mann-Whitney U statistic.
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
            f"Reject the null hypothesis - the distribution of those transported tends towards larger values than the distribution of those not transported."
        )
    else:
        print(
            f"Fail to reject the null hypothesis - There is no evidence that the distribution of those transported tends towards larger values than the distribution of those not transported."
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


def chisq(df: pd.DataFrame, var1: str, var2: str) -> None:
    alpha = 0.05
    """
    This function performs a chi squared test with the TravelInsurance variable and another categorical variable using a 95% confidence interval, and prints the results. 

    Args: 
    df (pd.DataFrame): The dataframe with categorical variables.
    var1 (str): The first variable (series) in the dataframe of interest.
    var2(str): The second variable (series) in the dataframe of interest. 

    Returns: 
    None
    """
    contingency_table = pd.crosstab(df[var1], df[var2])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print("contingengy table \n", contingency_table)
    print(f"Chi-Squared Statistic: {chi2}")
    print(f"p-value: {p}")
    print(f"Degrees of Freedom: {dof}")
    if p < alpha:
        print(
            f"Reject the null hypothesis - There is a relationship between {var1} and {var2}"
        )
    else:
        print(
            f"Fail to reject the null hypothesis - There is no relationship between {var1} and {var2}"
        )


def get_results(
    models: dict, x: pd.DataFrame, y: pd.Series, cv: StratifiedKFold
) -> pd.DataFrame:
    """
    Evaluate multiple models using cross-validation and return a summary of performance metrics.

    Args:
        models (dict): A dictionary where keys are model names and values are model instances.
        x (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.
        cv (StratifiedKFold): Cross-validation strategy.

    Returns:
        pd.DataFrame: A DataFrame containing the mean and standard deviation of various performance metrics for each model.
    """
    scoring = {
        "recall": make_scorer(recall_score, average="weighted"),
        "accuracy": make_scorer(accuracy_score),
        "weighted_f1": make_scorer(f1_score, average="weighted"),
        "roc_auc": make_scorer(roc_auc_score),
        "average_precision": make_scorer(average_precision_score),
        "MCC": make_scorer(matthews_corrcoef),
        "PR_AUC": make_scorer(average_precision_score, average="macro"),
    }
    results = []
    for name, model in models.items():
        scores = cross_validate(
            model, x, y, cv=cv, scoring=scoring, return_estimator=True
        )

        result = {"Model": name}

        metrics = [
            "test_accuracy",
            "test_average_precision",
            "test_recall",
            "test_weighted_f1",
            "test_PR_AUC",
            "test_MCC",
        ]

        for metric in metrics:
            mean_val = scores[metric].mean()
            std_val = scores[metric].std()
            result[metric.replace("test_", "").replace("_", " ")] = (
                f"{mean_val:.4f} ± {std_val:.4f}"
            )

        results.append(result)

    results_df = pd.DataFrame(results)
    return results_df


def format_cm_labels(cm: np.ndarray) -> np.ndarray:
    """
    Format the confusion matrix labels for better visualization.

    Args:
    cm (np.ndarray): Confusion matrix.

    Returns:
    np.ndarray: Formatted labels for the confusion matrix.
    """
    group_names = ["True Neg", "False Pos", "False Neg", "True Pos"]
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cm.flatten() / np.sum(cm)]
    labels = [
        f"{v1}\n{v2}\n{v3}"
        for v1, v2, v3 in zip(group_names, group_counts, group_percentages)
    ]
    return np.asarray(labels).reshape(2, 2)


def bayes_hyperparam_tuning(
    model: BaseEstimator, x: pd.DataFrame, y: pd.Series, param_space: dict
) -> BaseEstimator:
    """
    Perform Bayesian hyperparameter tuning using BayesSearchCV, using Matthews correlation coefficient as an evaluation metric.

    Args:
        model (BaseEstimator): The machine learning model to tune.
        x (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.
        param_space (dict): Dictionary defining the search space for hyperparameters.

    Returns:
        BaseEstimator: The model with the best hyperparameters found during the search.
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
