
import json
import joblib
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import (
    MODEL_DIR,
    MODEL_PATH,
    REPORT_DIR,
    METRICS_PATH,
    TARGET_COLUMN,
    DATE_COLUMN,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
)

from src.preprocess import load_and_validate_data
from src.features import create_features


# Training configuration
TEST_START_DATE = "2012-10-01"

EXCLUDED_COLUMNS = [
    TARGET_COLUMN,
    "casual",
    "registered",
    "instant",
    DATE_COLUMN,
    "yr",
]


def prepare_training_data(df):
    """Generate features and prepare chronological train-test sets."""

    feature_df = create_features(df)

    numeric_columns = feature_df.select_dtypes(
        include="number"
    ).columns

    feature_columns = [
        col for col in numeric_columns
        if col not in EXCLUDED_COLUMNS
    ]

    X = feature_df[feature_columns].copy()
    y = feature_df[TARGET_COLUMN].copy()

    train_mask = feature_df[DATE_COLUMN] < TEST_START_DATE
    test_mask = feature_df[DATE_COLUMN] >= TEST_START_DATE

    X_train = X.loc[train_mask]
    X_test = X.loc[test_mask]

    y_train = y.loc[train_mask]
    y_test = y.loc[test_mask]

    if X_train.empty or X_test.empty:
        raise ValueError("Training or test dataset is empty.")

    return X_train, X_test, y_train, y_test, feature_columns


def train_model(X_train, y_train):
    """Train the Linear Regression model."""

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model."""

    predictions = model.predict(X_test)

    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
        "r2": float(r2_score(y_test, predictions)),
    }

    return metrics, predictions


def save_model(model, feature_columns):
    """Save model and feature schema."""

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    schema = {
        "feature_columns": feature_columns,
        "target_column": TARGET_COLUMN,
        "date_column": DATE_COLUMN,
        "model_name": "LinearRegression",
    }

    schema_path = MODEL_DIR / "feature_schema.json"

    with open(schema_path, "w", encoding="utf-8") as file:
        json.dump(schema, file, indent=4)

    return schema_path


def save_metrics(metrics):
    """Save evaluation metrics as JSON."""

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)


def main():
    """Execute the complete model training and experiment tracking pipeline."""

    # Configure MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    print("Loading dataset...")
    df = load_and_validate_data()

    print("Preparing features...")
    X_train, X_test, y_train, y_test, feature_columns = (
        prepare_training_data(df)
    )

    with mlflow.start_run(run_name="LinearRegression"):

        print("Training Linear Regression...")
        model = train_model(X_train, y_train)

        print("Evaluating model...")
        metrics, predictions = evaluate_model(
            model,
            X_test,
            y_test,
        )

        print("\nEvaluation results:")
        for metric, value in metrics.items():
            print(f"{metric.upper()}: {value:.2f}")

        # Log experiment parameters
        mlflow.log_params({
            "model_type": "LinearRegression",
            "test_start_date": TEST_START_DATE,
            "training_rows": len(X_train),
            "testing_rows": len(X_test),
            "feature_count": len(feature_columns),
        })

        # Log evaluation metrics
        mlflow.log_metrics(metrics)

        # Preserve existing project artifacts
        print("\nSaving model...")
        schema_path = save_model(model, feature_columns)
        save_metrics(metrics)

        # Log model and metadata to MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
        )

        mlflow.log_artifact(
            str(schema_path),
            artifact_path="metadata",
        )

        mlflow.log_artifact(
            str(METRICS_PATH),
            artifact_path="metadata",
        )

        print(f"\nModel saved to: {MODEL_PATH}")
        print(f"Feature schema saved to: {schema_path}")
        print(f"Metrics saved to: {METRICS_PATH}")

        print(f"MLflow experiment: {MLFLOW_EXPERIMENT_NAME}")
        print(f"MLflow run ID: {mlflow.active_run().info.run_id}")

    return model, metrics, predictions


if __name__ == "__main__":
    main()
