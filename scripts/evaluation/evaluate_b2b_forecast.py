
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


# --------------------------------------------------
# Paths and configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "b2b_monthly_model_demand.csv"
)

RESULTS_DIR = PROJECT_ROOT / "models"

VALIDATION_RESULTS_PATH = (
    RESULTS_DIR / "b2b_forecast_validation_results.csv"
)

TEST_RESULTS_PATH = (
    RESULTS_DIR / "b2b_forecast_comparison_results.csv"
)

PREDICTIONS_PATH = (
    RESULTS_DIR / "b2b_forecast_test_predictions.csv"
)

# Keep the original final test period untouched during model selection.
VALIDATION_END = pd.Timestamp("2024-06-01")
TEST_START = pd.Timestamp("2024-10-01")

# Exponential recency weighting:
# larger alpha gives more weight to recent months.
ALPHAS = [0.1, 0.3, 0.5, 0.7, 0.9]


# --------------------------------------------------
# Metrics and forecasting functions
# --------------------------------------------------

def calculate_wape(actual, predicted):
    """Weighted absolute percentage error, expressed as a percentage."""
    denominator = np.abs(actual).sum()

    if denominator == 0:
        return np.nan

    return np.abs(actual - predicted).sum() / denominator * 100


def calculate_metrics(actual, predicted):
    """Calculate MAE and WAPE for predictions."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "wape_percent": float(calculate_wape(actual, predicted)),
        "actual_total": int(actual.sum()),
        "predicted_total": round(float(predicted.sum()), 2),
        "observations": len(actual),
    }


def fit_weighted_forecast(training_data, alpha):
    """
    Produce one forecast per product model.

    More recent observations receive exponentially greater weight.
    Missing months are already represented by zero in the prepared data.
    """
    forecasts = {}

    for model_id, group in training_data.groupby("PRODUCT_MODEL_ID"):
        group = group.sort_values("month")
        values = group["order_frequency"].to_numpy(dtype=float)

        # Oldest observation receives the smallest weight.
        weights = alpha ** np.arange(len(values) - 1, -1, -1)

        forecasts[model_id] = float(
            np.average(values, weights=weights)
        )

    return pd.Series(forecasts, name="prediction")


def score_forecast(test_data, forecast_series):
    """Score predictions against held-out observations."""
    scored = test_data.copy()

    scored["prediction"] = (
        scored["PRODUCT_MODEL_ID"]
        .map(forecast_series)
        .fillna(0)
        .clip(lower=0)
    )

    metrics = calculate_metrics(
        scored["order_frequency"],
        scored["prediction"],
    )

    return scored, metrics


# --------------------------------------------------
# Main evaluation
# --------------------------------------------------

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Prepared dataset not found: {DATA_PATH}\n"
            "Run prepare_b2b.py first."
        )

    df = pd.read_csv(DATA_PATH, parse_dates=["month"])

    required = {
        "month",
        "PRODUCT_MODEL_ID",
        "order_frequency",
    }

    if not required.issubset(df.columns):
        raise ValueError(
            f"Dataset must contain these columns: {sorted(required)}"
        )

    if df.duplicated(["month", "PRODUCT_MODEL_ID"]).any():
        raise ValueError(
            "Duplicate month/product-model combinations were found."
        )

    if df[["month", "PRODUCT_MODEL_ID", "order_frequency"]].isna().any().any():
        raise ValueError("Required forecasting fields contain missing values.")

    if (df["order_frequency"] < 0).any():
        raise ValueError("Order frequency cannot be negative.")

    # Training data for selecting alpha.
    model_selection_train = df[df["month"] <= VALIDATION_END].copy()

    # Validation months: July, August, September.
    validation = df[
        (df["month"] > VALIDATION_END)
        & (df["month"] < TEST_START)
    ].copy()

    # Final test months: October, November, December.
    test = df[df["month"] >= TEST_START].copy()

    if model_selection_train.empty or validation.empty or test.empty:
        raise ValueError(
            "Training, validation, or test period is empty. "
            "Check the dataset's date range."
        )

    print("\nB2B RECENCY-WEIGHTED FORECAST EVALUATION")
    print("=" * 48)
    print(
        f"Model-selection training: "
        f"{model_selection_train['month'].min():%Y-%m} to "
        f"{model_selection_train['month'].max():%Y-%m}"
    )
    print(
        f"Validation: {validation['month'].min():%Y-%m} to "
        f"{validation['month'].max():%Y-%m}"
    )
    print(
        f"Final test: {test['month'].min():%Y-%m} to "
        f"{test['month'].max():%Y-%m}"
    )

    # 1. Select alpha using validation data only.
    validation_results = []

    for alpha in ALPHAS:
        forecast = fit_weighted_forecast(
            model_selection_train,
            alpha,
        )

        _, metrics = score_forecast(validation, forecast)

        validation_results.append({
            "alpha": alpha,
            **metrics,
        })

    validation_df = pd.DataFrame(validation_results)
    validation_df = validation_df.sort_values(
        ["mae", "wape_percent"]
    )

    best_alpha = float(validation_df.iloc[0]["alpha"])

    # 2. Compare selected forecast and original simple baselines
    # on the final test period.
    # All forecasts are fitted using January-September.
    final_training = df[df["month"] < TEST_START].copy()

    mean_forecast = (
        final_training.groupby("PRODUCT_MODEL_ID")["order_frequency"]
        .mean()
    )

    september_forecast = (
        final_training[final_training["month"] == pd.Timestamp("2024-09-01")]
        .set_index("PRODUCT_MODEL_ID")["order_frequency"]
    )

    selected_forecast = fit_weighted_forecast(
        final_training,
        best_alpha,
    )

    forecast_methods = [
        ("Training mean", mean_forecast),
        ("Last training month", september_forecast),
        (f"Recency weighted alpha={best_alpha}", selected_forecast),
    ]

    test_results = []
    prediction_frames = []

    for method_name, forecast in forecast_methods:
        scored, metrics = score_forecast(test, forecast)

        test_results.append({
            "model": method_name,
            **metrics,
        })

        scored["model"] = method_name
        prediction_frames.append(scored)

    test_results_df = pd.DataFrame(test_results).sort_values(
        ["mae", "wape_percent"]
    )

    # 3. Save all results.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    validation_df.to_csv(VALIDATION_RESULTS_PATH, index=False)
    test_results_df.to_csv(TEST_RESULTS_PATH, index=False)

    pd.concat(prediction_frames, ignore_index=True).to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    print("\nVALIDATION RESULTS — USED TO SELECT ALPHA")
    print(validation_df.to_string(index=False))

    print(f"\nSelected alpha: {best_alpha}")

    print("\nFINAL TEST RESULTS — OCTOBER TO DECEMBER")
    print(test_results_df.to_string(index=False))

    print("\nFiles saved:")
    print(VALIDATION_RESULTS_PATH)
    print(TEST_RESULTS_PATH)
    print(PREDICTIONS_PATH)

    print(
        "\nReminder: these forecasts predict order frequency, "
        "not physical units demanded."
    )
    print(
        "The final test covers only three months, so performance "
        "estimates have substantial uncertainty."
    )


if __name__ == "__main__":
    main()
