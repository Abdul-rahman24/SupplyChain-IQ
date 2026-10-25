
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

OUTPUT_DIR = PROJECT_ROOT / "models"
RESULTS_PATH = OUTPUT_DIR / "b2b_forecast_baseline_results.csv"

TRAIN_END = pd.Timestamp("2024-09-01")
TEST_START = pd.Timestamp("2024-10-01")


def calculate_wape(actual, predicted):
    """Calculate weighted absolute percentage error."""
    denominator = np.sum(np.abs(actual))

    if denominator == 0:
        return np.nan

    return np.sum(np.abs(actual - predicted)) / denominator * 100


def main():
    # 1. Load and validate the prepared dataset.
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Prepared dataset not found: {DATA_PATH}\n"
            "Run prepare_b2b.py first."
        )

    df = pd.read_csv(DATA_PATH, parse_dates=["month"])

    required_columns = {
        "month",
        "PRODUCT_MODEL_ID",
        "order_frequency",
    }

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Dataset must contain: {sorted(required_columns)}"
        )

    if df.duplicated(["month", "PRODUCT_MODEL_ID"]).any():
        raise ValueError(
            "More than one row exists per month and product model."
        )

    if (df["order_frequency"] < 0).any():
        raise ValueError("Order frequency cannot be negative.")

    # 2. Define the chronological training and test periods.
    train = df[df["month"] <= TRAIN_END].copy()
    test = df[df["month"] >= TEST_START].copy()

    if train.empty or test.empty:
        raise ValueError("Training or test period is empty.")

    test_months = sorted(test["month"].unique())

    print("\nB2B FORECAST BASELINE EVALUATION")
    print("=" * 42)
    print(f"Training period: {train['month'].min():%Y-%m} "
          f"to {train['month'].max():%Y-%m}")
    print(f"Test period: {test['month'].min():%Y-%m} "
          f"to {test['month'].max():%Y-%m}")
    print(f"Training rows: {len(train):,}")
    print(f"Test rows: {len(test):,}")

    # 3. Baseline A: average monthly frequency from training.
    mean_forecast = (
        train.groupby("PRODUCT_MODEL_ID")["order_frequency"]
        .mean()
        .rename("mean_forecast")
    )

    # 4. Baseline B: frequency in the most recent training month.
    last_month = train["month"].max()

    last_forecast = (
        train[train["month"] == last_month]
        .set_index("PRODUCT_MODEL_ID")["order_frequency"]
        .rename("last_month_forecast")
    )

    # 5. Evaluate each model on the same held-out observations.
    results = []

    for forecast_name, forecast_series in [
        ("Training mean", mean_forecast),
        ("Last training month", last_forecast),
    ]:
        evaluated = test.copy()

        evaluated["prediction"] = (
            evaluated["PRODUCT_MODEL_ID"]
            .map(forecast_series)
            .fillna(0)
            .clip(lower=0)
        )

        actual = evaluated["order_frequency"].to_numpy(dtype=float)
        predicted = evaluated["prediction"].to_numpy(dtype=float)

        mae = mean_absolute_error(actual, predicted)
        wape = calculate_wape(actual, predicted)

        results.append(
            {
                "model": forecast_name,
                "test_start": str(pd.Timestamp(test_months[0]).date()),
                "test_end": str(pd.Timestamp(test_months[-1]).date()),
                "test_observations": len(evaluated),
                "actual_total": int(actual.sum()),
                "predicted_total": round(float(predicted.sum()), 2),
                "mae": round(float(mae), 4),
                "wape_percent": (
                    round(float(wape), 2)
                    if not np.isnan(wape)
                    else np.nan
                ),
            }
        )

    results_df = pd.DataFrame(results)

    # 6. Save the results for reproducibility.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_PATH, index=False)

    print("\nEVALUATION RESULTS")
    print(results_df.to_string(index=False))

    print("\nInterpretation:")
    print("- Lower MAE is better.")
    print("- Lower WAPE is better when total actual frequency is nonzero.")
    print("- These are baseline results, not evidence that ML is necessary.")
    print(f"\nResults saved to:\n{RESULTS_PATH}")


if __name__ == "__main__":
    main()
