
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning


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

OUTPUT_PATH = PROJECT_ROOT / "models" / "b2b_arima_order_comparison.csv"

TRAIN_CUTOFF = pd.Timestamp("2027-12-01")
VALIDATION_START = pd.Timestamp("2028-01-01")
VALIDATION_END = pd.Timestamp("2028-12-01")

ORDERS = [
    (1, 0, 0),
    (0, 0, 1),
    (1, 0, 1),
    (1, 1, 1),
]


def calculate_metrics(actual, predicted):
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
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    if not HISTORY_PATH.exists():
        raise FileNotFoundError(f"Model history not found: {HISTORY_PATH}")

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
        history["first_observed_month"], errors="raise"
    )
    data["order_frequency"] = pd.to_numeric(
        data["order_frequency"], errors="raise"
    )

    if data.duplicated(["month", "PRODUCT_MODEL_ID"]).any():
        raise ValueError("Duplicate month/model rows found.")

    if data["order_frequency"].isna().any():
        raise ValueError("Missing order-frequency values found.")

    if (
        (data["order_frequency"] < 0).any()
        or not np.isfinite(data["order_frequency"]).all()
    ):
        raise ValueError("Invalid order-frequency values found.")

    first_seen = history.set_index(
        "PRODUCT_MODEL_ID"
    )["first_observed_month"]

    values = data.set_index(
        ["PRODUCT_MODEL_ID", "month"]
    )["order_frequency"]

    validation_months = pd.date_range(
        VALIDATION_START,
        VALIDATION_END,
        freq="MS",
    )

    eligible_models = first_seen[
        first_seen <= TRAIN_CUTOFF
    ].index.tolist()

    if not eligible_models:
        raise ValueError("No eligible product models.")

    print("B2B ARIMA ORDER TUNING")
    print("=" * 65)
    print(f"Training cutoff: {TRAIN_CUTOFF:%Y-%m}")
    print(
        f"Validation period: {VALIDATION_START:%Y-%m} "
        f"to {VALIDATION_END:%Y-%m}"
    )
    print(f"Eligible product models: {len(eligible_models)}")

    all_results = []

    for order in ORDERS:
        print(f"\nEvaluating ARIMA{order} ...")

        actual_values = []
        predicted_values = []
        models_with_warnings = 0
        models_with_errors = 0

        for position, model_id in enumerate(eligible_models, start=1):
            first_month = first_seen.loc[model_id]

            training_months = pd.date_range(
                first_month,
                TRAIN_CUTOFF,
                freq="MS",
            )

            training_index = pd.MultiIndex.from_arrays(
                [
                    [model_id] * len(training_months),
                    training_months,
                ],
                names=["PRODUCT_MODEL_ID", "month"],
            )

            training = values.reindex(training_index)

            if training.isna().any():
                raise ValueError(
                    f"Missing training observations for {model_id}."
                )

            series = training.to_numpy(dtype=float)

            try:
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter("always")

                    fitted = ARIMA(
                        series,
                        order=order,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    ).fit()

                if captured:
                    models_with_warnings += 1

                forecast = np.asarray(
                    fitted.forecast(steps=len(validation_months)),
                    dtype=float,
                )

                if not np.isfinite(forecast).all():
                    raise ValueError("Non-finite forecast.")

                forecast = np.clip(forecast, 0, None)

            except Exception:
                models_with_errors += 1

                # Explicitly use the training mean if this order fails.
                # Count the fallback rather than hiding model failures.
                forecast = np.full(
                    len(validation_months),
                    max(0.0, float(series.mean())),
                )

            validation_index = pd.MultiIndex.from_arrays(
                [
                    [model_id] * len(validation_months),
                    validation_months,
                ],
                names=["PRODUCT_MODEL_ID", "month"],
            )

            actual = values.reindex(validation_index)

            if actual.isna().any():
                raise ValueError(
                    f"Missing validation observations for {model_id}."
                )

            actual_values.extend(actual.to_numpy(dtype=float))
            predicted_values.extend(forecast)

            if position % 50 == 0:
                print(
                    f"  Processed {position}/{len(eligible_models)} models"
                )

        metrics = calculate_metrics(
            actual_values,
            predicted_values,
        )

        result = {
            "order": str(order),
            "p": order[0],
            "d": order[1],
            "q": order[2],
            "eligible_models": len(eligible_models),
            "models_with_warnings": models_with_warnings,
            "models_with_errors_using_fallback": models_with_errors,
            **metrics,
        }

        all_results.append(result)

        print(f"MAE: {metrics['MAE']:.4f}")
        print(f"WAPE: {metrics['WAPE_percent']:.2f}%")
        print(f"Models with warnings: {models_with_warnings}")
        print(f"Models requiring fallback: {models_with_errors}")

    results = pd.DataFrame(all_results).sort_values(
        ["WAPE_percent", "MAE"]
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    print("\nARIMA ORDER COMPARISON — 2028 VALIDATION")
    print("=" * 85)
    print(
        results[
            [
                "order",
                "MAE",
                "WAPE_percent",
                "actual_total",
                "predicted_total",
                "total_error",
                "models_with_warnings",
                "models_with_errors_using_fallback",
            ]
        ].round(3).to_string(index=False)
    )

    print(f"\nResults saved to:\n{OUTPUT_PATH}")
    print(
        "\nSelection rule: use 2028 validation to choose the order. "
        "Do not select the order based on 2029 test results."
    )


if __name__ == "__main__":
    main()
