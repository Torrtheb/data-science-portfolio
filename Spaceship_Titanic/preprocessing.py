import pandas as pd
import numpy as np


def add_indicators(x: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary missing value indicators for specific columns.

    Args:
        x (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with added missing value indicator columns.
    """
    missing_indicator_cols = [
        "RoomService",
        "FoodCourt",
        "ShoppingMall",
        "Spa",
        "VRDeck",
        "Age",
        "VIP",
    ]
    x = x.copy()
    for col in missing_indicator_cols:
        x[f"{col}_missing"] = x[col].isnull().astype(int)
    return x


def add_tot_spend(x: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the total spending across various services and add it as a new column.

    Args:
        x (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with 'tot_spend' column added.
    """
    x = x.copy()
    x["tot_spend"] = x[
        ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    ].sum(axis=1)
    return x


def binning(x: pd.DataFrame) -> pd.DataFrame:
    """
    Bin spending columns and total spending into labeled categories.

    Args:
        x (pd.DataFrame): Input DataFrame with spend-related columns.

    Returns:
        pd.DataFrame: DataFrame with new binned columns.
    """
    x = x.copy()
    billed_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    bins = [0, 1, 50, 500, 1000, 5000, np.inf]
    labels = ["0", "1-50", "50-500", "500-1000", "1000-5000", "5000+"]
    for var in billed_cols:
        x[f"{var}_binned"] = pd.cut(
            x[var], bins=bins, labels=labels, include_lowest=True
        ).astype(str)

    bins = [0, 1, 750, 1000, 2500, 5000, np.inf]
    labels = ["0", "1-750", "750-1000", "1000-2500", "2500-5000", "5000+"]
    x["tot_spend_binned"] = pd.cut(
        x["tot_spend"], bins=bins, labels=labels, include_lowest=True
    ).astype(str)
    return x


def group(x: pd.DataFrame) -> pd.DataFrame:
    """
    Create group ID and group size features from PassengerId.

    Args:
        x (pd.DataFrame): Input DataFrame with 'PassengerId' column.

    Returns:
        pd.DataFrame: DataFrame with 'group' and 'group_size' columns.
    """
    x = x.copy()
    x["group"] = x["PassengerId"].apply(lambda x: x[:4]).astype(int)
    x["group_size"] = x.groupby("group")["group"].transform("count")
    return x


def cryosleep(x: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing CryoSleep values based on total spending and convert to string.

    Args:
        x (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with updated 'CryoSleep' column.
    """
    x = x.copy()
    x.loc[x["CryoSleep"].isnull() & (x["tot_spend"] > 0), "CryoSleep"] = False
    x["CryoSleep"] = x["CryoSleep"].astype(str)
    return x


def cabin(x: pd.DataFrame) -> pd.DataFrame:
    """
    Extract deck and side information from 'Cabin' column.

    Args:
        x (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with 'cabin_deck' and 'cabin_side' columns.
    """
    x = x.copy()
    x["Cabin"] = x["Cabin"].fillna("0/0/0")
    x["cabin_deck"] = x["Cabin"].apply(lambda x: x[0]).astype(str)
    x["cabin_side"] = x["Cabin"].apply(lambda x: x[-1]).astype(str)
    return x


def vip(x: pd.DataFrame) -> pd.DataFrame:
    """
    Impute and convert 'VIP' column to binary integer format.

    Args:
        x (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with 'VIP' as integer.
    """
    x = x.copy()
    x["VIP"] = x["VIP"].fillna("False").astype(bool)
    x["VIP"] = x["VIP"].astype(int)
    return x


def name(x: pd.DataFrame) -> pd.DataFrame:
    """
    Split full names into first and last name columns.

    Args:
        x (pd.DataFrame): Input DataFrame with 'Name' column.

    Returns:
        pd.DataFrame: DataFrame with 'first_name' and 'last_name' columns.
    """
    x = x.copy()
    x["Name"] = x["Name"].fillna("No Name")
    x[["first_name", "last_name"]] = x["Name"].str.split(" ", n=1, expand=True)
    return x


def cat_missing(x: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing categorical values with 'Missing'.

    Args:
        x (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with filled categorical values.
    """
    x = x.copy()
    x["HomePlanet"] = x["HomePlanet"].fillna("Missing")
    x["CryoSleep"] = x["CryoSleep"].fillna("Missing").astype(str)
    x["Destination"] = x["Destination"].fillna("Missing")
    return x


def log_transform(x: pd.DataFrame) -> pd.DataFrame:
    """
    Apply log transformation to spending columns, handling NaNs and infinities.

    Args:
        x (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with log-transformed columns.
    """
    x = x.copy()
    billed_cols = [
        "RoomService",
        "FoodCourt",
        "ShoppingMall",
        "Spa",
        "VRDeck",
        "tot_spend",
    ]
    for var in billed_cols:
        x[f"{var}_log"] = np.log1p(x[var])
        x[f"{var}_log"] = (
            x[f"{var}_log"].replace([np.inf, -np.inf], 0).fillna(0).astype(int)
        )
    return x


def new_feat(x: pd.DataFrame) -> pd.DataFrame:
    """
    Generate additional engineered features.

    Args:
        x (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with new features.
    """
    x = x.copy()
    x["group_div_group_size"] = x["group"] / x["group_size"]
    x["group_div_group_size"] = (
        x["group_div_group_size"].replace([np.inf, -np.inf], 0).fillna(0)
    )
    x["group_mul_group_size"] = x["group"] * x["group_size"]
    x["group_mul_group_size"] = (
        x["group_mul_group_size"].replace([np.inf, -np.inf], 0).fillna(0)
    )

    return x


def preprocessing_initial(x: pd.DataFrame) -> pd.DataFrame:
    """
    Run a full pipeline of preprocessing functions in sequence.

    Args:
        x (pd.DataFrame): Raw input DataFrame.

    Returns:
        pd.DataFrame: Fully preprocessed DataFrame.
    """
    x = x.copy()
    x = add_tot_spend(x)
    x = binning(x)
    x = cryosleep(x)
    x = group(x)
    x = cabin(x)
    x = vip(x)
    x = name(x)
    x = cat_missing(x)
    return x


def preprocessing_final(x: pd.DataFrame) -> pd.DataFrame:
    """
    Run a full pipeline of preprocessing functions in sequence.

    Args:
        x (pd.DataFrame): Raw input DataFrame.

    Returns:
        pd.DataFrame: Fully preprocessed DataFrame.
    """
    x = x.copy()
    x = add_tot_spend(x)
    x = binning(x)
    x = cryosleep(x)
    x = group(x)
    x = cabin(x)
    x = vip(x)
    x = name(x)
    x = cat_missing(x)
    x = new_feat(x)
    x = log_transform(x)
    return x
