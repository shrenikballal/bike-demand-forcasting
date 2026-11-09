
import json
import joblib
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.config import (
    MODEL_PATH,
    MODEL_DIR,
    REPORT_DIR,
    METRICS_PATH,
    TARGET_COLUMN,
    DATE_COLUMN,
)

from src.preprocess import load_and_validate_data
from src.features import create_features


TEST_START_DATE = "2012-10-01"


def load_model_and_schema():
    """Load the trained model and its feature schema."""

    schema_path = MODEL_DIR / "feature_schema.json"

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. Run training first."
        )

    if not schema_path.exists():
        raise FileNotFoundError(
            f"Feature schema not found: {schema_path}"
        )

    model = joblib.load(MODEL_PATH)

    with open(schema_path, "r", encoding="utf-8") as file:
        schema = json.load(file)

    return model, schema


def prepare_test_data(feature_columns):
    """Recreate the chronological test dataset."""

    df = load_and_validate_data()
    feature_df = create_features(df)

    missing_features = [
        col for col in feature_columns
        if col not in feature_df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing model features: {missing_features}"
        )

    test_mask = feature_df[DATE_COLUMN] >= TEST_START_DATE

    X_test = feature_df.loc[
        test_mask, feature_columns
    ].copy()

    y_test = feature_df.loc[
        test_mask, TARGET_COLUMN
    ].copy()

    dates = feature_df.loc[
        test_mask, DATE_COLUMN
    ].copy()

    if X_test.empty:
        raise ValueError("Test dataset is empty.")

    return X_test, y_test, dates


def evaluate_model(model, X_test, y_test):
    """Calculate model evaluation metrics."""

    predictions = model.predict(X_test)

    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(
            mean_squared_error(y_test, predictions) ** 0.5
        ),
        "r2": float(r2_score(y_test, predictions)),
    }

    return metrics, predictions


def save_evaluation_results(dates, y_test, predictions, metrics):
    """Save metrics and individual prediction results."""

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame({
        DATE_COLUMN: dates.to_numpy(),
        "actual": y_test.to_numpy(),
        "predicted": predictions,
    })

    results_df["residual"] = (
        results_df["actual"] - results_df["predicted"]
    )

    results_df["absolute_error"] = (
        results_df["residual"].abs()
    )

    results_df.to_csv(
        REPORT_DIR / "predictions.csv",
        index=False
    )

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)


def main():
    """Execute the evaluation pipeline."""

    print("Loading saved model...")
    model, schema = load_model_and_schema()

    feature_columns = schema["feature_columns"]

    print("Preparing test data...")
    X_test, y_test, dates = prepare_test_data(feature_columns)

    print("Evaluating model...")
    metrics, predictions = evaluate_model(
        model, X_test, y_test
    )

    print("\nEvaluation results:")

    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.2f}")

    save_evaluation_results(
        dates, y_test, predictions, metrics
    )

    print("\nEvaluation results saved successfully.")
    print(f"Metrics: {METRICS_PATH}")
    print(f"Predictions: {REPORT_DIR / 'predictions.csv'}")


if __name__ == "__main__":
    main()
