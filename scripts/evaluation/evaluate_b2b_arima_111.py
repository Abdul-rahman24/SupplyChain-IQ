
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import (
    ConvergenceWarning,
    ValueWarning,
)


# --------------------------------------------------
# Paths and configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "b2b_monthly_model_demand_all_years.csv"
)

HISTORY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "b2b_product_model_history.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "models"

SUMMARY_PATH = OUTPUT_DIR / "b2b_arima_evaluation_summary_111.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "b2b_arima_evaluation_predictions_111.csv"

ARIMA_ORDER = (1, 1, 1)

SPLITS = [
    {
        "name": "validation_2028",
        "cutoff": "2027-12-01",
        "start": "2028-01-01",
        "end": "2028-12-01",
    },
    {
        "name": "test_2029",
        "cutoff": "2028-12-01",
        "start": "2029-01-01",
        "end": "2029-12-01",
    },
]


def calculate_metrics(actual, predicted):
    """Calculate MAE and WAPE for evaluated forecasts."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    absolute_errors = np.abs(actual - predicted)
    actual_total = float(actual.sum())

    return {
        "MAE": float(absolute_errors.mean()),
        "WAPE_percent": (
            float(absolute_errors.sum() / actual_total * 100)
            if actual_total > 0
            else np.nan
        ),
        "actual_total": actual_total,
        "predicted_total": float(predicted.sum()),
        "total_error": float(predicted.sum() - actual_total),
        "evaluated_model_months": len(actual),
    }


def fit_and_forecast(series, horizon):
    """Fit a fixed-order ARIMA model and forecast future months."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", ValueWarning)

        fitted = ARIMA(
            series,
            order=ARIMA_ORDER,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit()

        forecast = np.asarray(
            fitted.forecast(steps=horizon),
            dtype=float,
        )

    if not np.isfinite(forecast).all():
        raise ValueError("ARIMA produced non-finite forecast values.")

    # Order frequency cannot be negative.
    return np.clip(forecast, 0, None)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    if not HISTORY_PATH.exists():
        raise FileNotFoundError(f"Model history file not found: {HISTORY_PATH}")

    data = pd.read_csv(
        DATA_PATH,
        dtype={"PRODUCT_MODEL_ID": "string"},
    )

    history = pd.read_csv(
        HISTORY_PATH,
        dtype={"PRODUCT_MODEL_ID": "string"},
    )

    data["month"] = pd.to_datetime(data["month"], errors="raise")
    history["first_observed_month"] = pd.to_datetime(
        history["first_observed_month"],
        errors="raise",
    )
    data["order_frequency"] = pd.to_numeric(
        data["order_frequency"],
        errors="raise",
    )

    if data.duplicated(["month", "PRODUCT_MODEL_ID"]).any():
        raise ValueError("Duplicate month/model combinations found.")

    if data["order_frequency"].isna().any():
        raise ValueError("Missing order-frequency values found.")

    if (
        (data["order_frequency"] < 0).any()
        or not np.isfinite(data["order_frequency"]).all()
    ):
        raise ValueError("Order frequency contains invalid values.")

    if history["PRODUCT_MODEL_ID"].duplicated().any():
        raise ValueError("Duplicate product models in history file.")

    first_seen = history.set_index(
        "PRODUCT_MODEL_ID"
    )["first_observed_month"]

    values = data.set_index(
        ["PRODUCT_MODEL_ID", "month"]
    )["order_frequency"]

    all_predictions = []
    summary_rows = []

    for split in SPLITS:
        split_name = split["name"]
        cutoff = pd.Timestamp(split["cutoff"])
        forecast_start = pd.Timestamp(split["start"])
        forecast_end = pd.Timestamp(split["end"])

        forecast_months = pd.date_range(
            forecast_start,
            forecast_end,
            freq="MS",
        )

        eligible_models = first_seen[
            first_seen <= cutoff
        ].index.tolist()

        split_predictions = []
        failed_models = []

        print(f"\n{split_name.upper()}")
        print("-" * 55)
        print(f"Training cutoff: {cutoff:%Y-%m}")
        print(f"Forecast period: {forecast_start:%Y-%m} to {forecast_end:%Y-%m}")
        print(f"Eligible product models: {len(eligible_models)}")

        for position, model_id in enumerate(eligible_models, start=1):
            first_month = first_seen.loc[model_id]

            training_months = pd.date_range(
                first_month,
                cutoff,
                freq="MS",
            )

            training_index = pd.MultiIndex.from_arrays(
                [
                    [model_id] * len(training_months),
                    training_months,
                ],
                names=["PRODUCT_MODEL_ID", "month"],
            )

            training_values = values.reindex(training_index)

            if training_values.isna().any():
                raise ValueError(
                    f"Missing training months for model {model_id}."
                )

            series = training_values.to_numpy(dtype=float)

            try:
                forecast = fit_and_forecast(
                    series,
                    len(forecast_months),
                )
                forecast_method = "ARIMA"
            except Exception as exc:
                # Keep the model-month coverage comparable.
                # Record fallback use explicitly instead of hiding failures.
                fallback = float(series.mean())
                forecast = np.full(
                    len(forecast_months),
                    max(0.0, fallback),
                )
                forecast_method = "training_mean_fallback"
                failed_models.append(
                    {
                        "PRODUCT_MODEL_ID": model_id,
                        "error": str(exc)[:300],
                    }
                )

            for month, prediction in zip(forecast_months, forecast):
                actual = values.get((model_id, month), np.nan)

                if pd.isna(actual):
                    raise ValueError(
                        f"Missing actual frequency for {model_id} "
                        f"in {month:%Y-%m}."
                    )

                split_predictions.append(
                    {
                        "split": split_name,
                        "PRODUCT_MODEL_ID": model_id,
                        "month": month,
                        "actual_order_frequency": float(actual),
                        "predicted_order_frequency": float(prediction),
                        "forecast_method": forecast_method,
                        "training_months": len(series),
                    }
                )

            if position % 25 == 0:
                print(
                    f"Processed {position}/{len(eligible_models)} models"
                )

        predictions = pd.DataFrame(split_predictions)
        all_predictions.append(predictions)

        metrics = calculate_metrics(
            predictions["actual_order_frequency"],
            predictions["predicted_order_frequency"],
        )

        summary_rows.append(
            {
                "split": split_name,
                "method": f"ARIMA{ARIMA_ORDER}",
                "eligible_models": len(eligible_models),
                "models_using_arima": int(
                    (predictions["forecast_method"] == "ARIMA")
                    .groupby(predictions["PRODUCT_MODEL_ID"])
                    .any()
                    .sum()
                ),
                "models_using_fallback": len(failed_models),
                **metrics,
            }
        )

        print(f"Models requiring fallback: {len(failed_models)}")
        print(f"MAE: {metrics['MAE']:.4f}")
        print(f"WAPE: {metrics['WAPE_percent']:.2f}%")
        print(f"Actual total: {metrics['actual_total']:.0f}")
        print(f"Predicted total: {metrics['predicted_total']:.2f}")

        if failed_models:
            print("Some models required the explicitly recorded mean fallback.")

    summary = pd.DataFrame(summary_rows)
    predictions = pd.concat(all_predictions, ignore_index=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary.to_csv(SUMMARY_PATH, index=False)
    predictions.to_csv(PREDICTIONS_PATH, index=False)

    print("\nARIMA EVALUATION COMPLETE")
    print("=" * 65)
    print(
        summary[
            [
                "split",
                "method",
                "eligible_models",
                "models_using_arima",
                "models_using_fallback",
                "MAE",
                "WAPE_percent",
                "actual_total",
                "predicted_total",
            ]
        ].round(3).to_string(index=False)
    )

    print("\nSaved files:")
    print(SUMMARY_PATH)
    print(PREDICTIONS_PATH)

    print(
        "\nNote: 2028 is validation; 2029 is the final test. "
        "Any training-mean fallbacks are counted and reported."
    )


if __name__ == "__main__":
    main()
