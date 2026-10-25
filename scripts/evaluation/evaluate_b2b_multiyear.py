
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

DATA_FILE = DATA_DIR / "b2b_monthly_model_demand_all_years.csv"
HISTORY_FILE = DATA_DIR / "b2b_product_model_history.csv"

ALPHA = 0.9

SPLITS = [
    {
        "name": "validation_2028",
        "cutoff": "2027-12-01",
        "forecast_start": "2028-01-01",
        "forecast_end": "2028-12-01",
    },
    {
        "name": "test_2029",
        "cutoff": "2028-12-01",
        "forecast_start": "2029-01-01",
        "forecast_end": "2029-12-01",
    },
]


def weighted_recent_mean(values, alpha=ALPHA):
    """Return an exponentially recency-weighted average."""
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return 0.0

    # The latest observation receives the greatest weight.
    weights = alpha ** np.arange(len(values) - 1, -1, -1)
    return float(np.average(values, weights=weights))


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    absolute_errors = np.abs(actual - predicted)
    actual_total = float(actual.sum())
    predicted_total = float(predicted.sum())

    mae = float(absolute_errors.mean()) if len(actual) else np.nan
    wape = (
        float(absolute_errors.sum() / actual_total * 100)
        if actual_total > 0
        else np.nan
    )

    return {
        "MAE": mae,
        "WAPE_percent": wape,
        "actual_total": actual_total,
        "predicted_total": predicted_total,
        "total_error": predicted_total - actual_total,
        "evaluated_model_months": len(actual),
    }


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for path in [DATA_FILE, HISTORY_FILE]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    monthly = pd.read_csv(
        DATA_FILE,
        dtype={"PRODUCT_MODEL_ID": "string"},
    )
    history = pd.read_csv(
        HISTORY_FILE,
        dtype={"PRODUCT_MODEL_ID": "string"},
    )

    required_monthly = {
        "month",
        "PRODUCT_MODEL_ID",
        "order_frequency",
    }
    required_history = {
        "PRODUCT_MODEL_ID",
        "first_observed_month",
    }

    if not required_monthly.issubset(monthly.columns):
        raise ValueError(
            f"Monthly data must contain: {required_monthly}"
        )

    if not required_history.issubset(history.columns):
        raise ValueError(
            f"History data must contain: {required_history}"
        )

    monthly["month"] = pd.to_datetime(
        monthly["month"], errors="raise"
    )
    history["first_observed_month"] = pd.to_datetime(
        history["first_observed_month"], errors="raise"
    )
    monthly["order_frequency"] = pd.to_numeric(
        monthly["order_frequency"], errors="raise"
    )

    if monthly.duplicated(["month", "PRODUCT_MODEL_ID"]).any():
        raise ValueError("Duplicate month-model combinations found.")

    if monthly["order_frequency"].isna().any():
        raise ValueError("Missing order-frequency values found.")

    if (monthly["order_frequency"] < 0).any():
        raise ValueError("Negative order-frequency values found.")

    if history["PRODUCT_MODEL_ID"].duplicated().any():
        raise ValueError("Duplicate product models in history report.")

    first_seen = history.set_index(
        "PRODUCT_MODEL_ID"
    )["first_observed_month"]

    models_in_data = set(monthly["PRODUCT_MODEL_ID"])
    models_in_history = set(history["PRODUCT_MODEL_ID"])

    if models_in_data != models_in_history:
        raise ValueError(
            "Product models in monthly data and history report differ."
        )

    # Index the full monthly grid for reliable model/month lookups.
    indexed = monthly.set_index(
        ["PRODUCT_MODEL_ID", "month"]
    )["order_frequency"]

    all_predictions = []
    all_summary_rows = []

    for split in SPLITS:
        cutoff = pd.Timestamp(split["cutoff"])
        forecast_start = pd.Timestamp(split["forecast_start"])
        forecast_end = pd.Timestamp(split["forecast_end"])

        forecast_months = pd.date_range(
            forecast_start, forecast_end, freq="MS"
        )

        eligible_models = first_seen[
            first_seen <= cutoff
        ].index.tolist()

        if not eligible_models:
            raise ValueError(
                f"No eligible models for {split['name']}."
            )

        split_predictions = []

        for model_id in eligible_models:
            first_month = first_seen.loc[model_id]

            # Never include months before this model's first observation.
            training_months = pd.date_range(
                first_month, cutoff, freq="MS"
            )

            training_values = indexed.reindex(
                pd.MultiIndex.from_arrays(
                    [
                        [model_id] * len(training_months),
                        training_months,
                    ],
                    names=["PRODUCT_MODEL_ID", "month"],
                )
            )

            if training_values.isna().any():
                raise ValueError(
                    f"Incomplete training grid for model {model_id}."
                )

            values = training_values.to_numpy(dtype=float)

            mean_prediction = float(values.mean())
            last_prediction = float(values[-1])
            weighted_prediction = weighted_recent_mean(values)

            predictions_by_method = {
                "training_mean": mean_prediction,
                "last_month": last_prediction,
                "recency_weighted": weighted_prediction,
            }

            for month in forecast_months:
                actual_value = indexed.get((model_id, month), np.nan)

                if pd.isna(actual_value):
                    raise ValueError(
                        f"Missing actual value for model {model_id}, "
                        f"month {month:%Y-%m}."
                    )

                for method, prediction in predictions_by_method.items():
                    split_predictions.append(
                        {
                            "split": split["name"],
                            "method": method,
                            "PRODUCT_MODEL_ID": model_id,
                            "month": month,
                            "actual_order_frequency": float(actual_value),
                            "predicted_order_frequency": max(
                                0.0, prediction
                            ),
                            "training_months": len(values),
                        }
                    )

        prediction_df = pd.DataFrame(split_predictions)
        all_predictions.append(prediction_df)

        for method, group in prediction_df.groupby("method"):
            metrics = calculate_metrics(
                group["actual_order_frequency"],
                group["predicted_order_frequency"],
            )

            all_summary_rows.append(
                {
                    "split": split["name"],
                    "method": method,
                    "cutoff_month": cutoff,
                    "forecast_start": forecast_start,
                    "forecast_end": forecast_end,
                    "eligible_models": len(eligible_models),
                    **metrics,
                }
            )

        print(f"\n{split['name'].upper()}")
        print("-" * 55)
        print(f"Training cutoff: {cutoff:%Y-%m}")
        print(
            f"Forecast period: {forecast_start:%Y-%m} "
            f"to {forecast_end:%Y-%m}"
        )
        print(f"Eligible product models: {len(eligible_models)}")
        print(
            prediction_df.groupby("method")[
                ["actual_order_frequency", "predicted_order_frequency"]
            ].sum().round(2)
        )

    summary = pd.DataFrame(all_summary_rows)
    predictions = pd.concat(all_predictions, ignore_index=True)

    summary_path = MODELS_DIR / "b2b_multiyear_evaluation_summary.csv"
    predictions_path = MODELS_DIR / "b2b_multiyear_evaluation_predictions.csv"

    summary.to_csv(summary_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    print("\nMULTI-YEAR EVALUATION COMPLETE")
    print("=" * 65)
    print(
        summary[
            [
                "split",
                "method",
                "eligible_models",
                "MAE",
                "WAPE_percent",
                "actual_total",
                "predicted_total",
                "total_error",
            ]
        ].round(3).to_string(index=False)
    )

    print("\nSaved files:")
    print(summary_path)
    print(predictions_path)

    print(
        "\nInterpretation note: first observed month is not a verified "
        "product launch date. These results evaluate models with observed "
        "history, not cold-start forecasting for newly observed models."
    )


if __name__ == "__main__":
    main()
