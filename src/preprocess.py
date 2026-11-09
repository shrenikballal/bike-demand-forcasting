
import pandas as pd

from src.config import RAW_DATA_PATH, DATE_COLUMN, TARGET_COLUMN


def load_raw_data(file_path=RAW_DATA_PATH):
    """
    Load the raw bike-sharing dataset.

    Parameters
    ----------
    file_path : Path
        Location of the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {file_path}"
        )

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError("The dataset is empty.")

    return df


def validate_data(df):
    """
    Validate the raw bike-sharing dataset.

    Checks required columns, missing values,
    duplicate records, dates, and target values.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    required_columns = [DATE_COLUMN, TARGET_COLUMN]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if df[required_columns].isnull().any().any():
        raise ValueError(
            "Missing values found in required columns."
        )

    if df.duplicated(subset=[DATE_COLUMN]).any():
        raise ValueError("Duplicate dates found in dataset.")

    try:
        dates = pd.to_datetime(df[DATE_COLUMN], errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid date values found.") from exc

    if not dates.is_monotonic_increasing:
        raise ValueError("Dates must be in chronological order.")

    if not pd.api.types.is_numeric_dtype(df[TARGET_COLUMN]):
        raise TypeError("Target column must be numeric.")

    if (df[TARGET_COLUMN] < 0).any():
        raise ValueError("Rental counts cannot be negative.")

    return True


def load_and_validate_data(file_path=RAW_DATA_PATH):
    """
    Load the raw dataset and validate it before use.
    """
    df = load_raw_data(file_path)
    validate_data(df)

    return df
