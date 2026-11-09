
import pandas as pd

from src.config import DATE_COLUMN, TARGET_COLUMN


LAG_DAYS = [1, 2, 3, 7, 14, 21, 28]
ROLLING_WINDOWS = [7, 14, 28]


def create_calendar_features(df):
    """Create calendar-based features from the date column."""

    df = df.copy()

    dates = pd.to_datetime(df[DATE_COLUMN])

    df["year"] = dates.dt.year
    df["month"] = dates.dt.month
    df["day"] = dates.dt.day
    df["day_of_week"] = dates.dt.dayofweek
    df["day_of_year"] = dates.dt.dayofyear
    df["week_of_year"] = dates.dt.isocalendar().week.astype(int)
    df["is_weekend"] = (dates.dt.dayofweek >= 5).astype(int)

    return df


def create_lag_features(df):
    """Create lag features using historical target values."""

    df = df.copy()

    for lag in LAG_DAYS:
        df[f"{TARGET_COLUMN}_lag_{lag}"] = df[TARGET_COLUMN].shift(lag)

    return df


def create_rolling_features(df):
    """Create leakage-safe rolling statistics."""

    df = df.copy()

    for window in ROLLING_WINDOWS:
        shifted_target = df[TARGET_COLUMN].shift(1)

        df[f"{TARGET_COLUMN}_rolling_mean_{window}"] = (
            shifted_target.rolling(window=window).mean()
        )

        df[f"{TARGET_COLUMN}_rolling_std_{window}"] = (
            shifted_target.rolling(window=window).std()
        )

    return df


def create_features(df, drop_missing=True):
    """
    Create all forecasting features.

    The input must contain the date and target columns,
    and must be sorted chronologically.
    """

    df = df.copy()

    if DATE_COLUMN not in df.columns:
        raise ValueError(f"Missing date column: {DATE_COLUMN}")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])

    if not df[DATE_COLUMN].is_monotonic_increasing:
        raise ValueError("Data must be sorted chronologically.")

    df = create_calendar_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)

    if drop_missing:
        df = df.dropna().copy()

    return df
