
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

DATA_FILE = DATA_DIR / "b2b_monthly_model_demand_all_years.csv"
HISTORY_FILE = DATA_DIR / "b2b_product_model_history.csv"
BASELINE_FILE = MODELS_DIR / "b2b_multiyear_evaluation_summary.csv"

SPLITS = [
    ("validation_2028", "2027-12-01", "2028-01-01", "2028-12-01"),
    ("test_2029", "2028-12-01", "2029-01-01", "2029-12-01"),
]


def metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    errors = np.abs(actual - predicted)
    total = actual.sum()

    return {
        "MAE": errors.mean(),
        "WAPE_percent": errors.sum() / total * 100 if total else np.nan,
        "actual_total": total,
        "predicted_total": predicted.sum(),
        "total_error": predicted.sum() - total,
        "evaluated_model_months": len(actual),
    }


def main():
    for path in (DATA_FILE, HISTORY_FILE, BASELINE_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    data = pd.read_csv(DATA_FILE, dtype={"PRODUCT_MODEL_ID": "string"})
    history = pd.read_csv(
        HISTORY_FILE, dtype={"PRODUCT_MODEL_ID": "string"}
    )
    previous_results = pd.read_csv(BASELINE_FILE)

    data["month"] = pd.to_datetime(data["month"])
    history["first_observed_month"] = pd.to_datetime(
        history["first_observed_month"]
    )
    data["order_frequency"] = pd.to_numeric(
        data["order_frequency"], errors="raise"
    )

    values = data.set_index(
        ["PRODUCT_MODEL_ID", "month"]
    )["order_frequency"]

    first_seen = history.set_index(
        "PRODUCT_MODEL_ID"
    )["first_observed_month"]

    result_rows = []
    prediction_rows = []

    for split_name, cutoff_text, start_text, end_text in SPLITS:
        cutoff = pd.Timestamp(cutoff_text)
        forecast_months = pd.date_range(
            start_text, end_text, freq="MS"
        )

        eligible_models = first_seen[
            first_seen <= cutoff
        ].index.tolist()

        for model in eligible_models:
            first_month = first_seen.loc[model]
            training_months = pd.date_range(
                first_month, cutoff, freq="MS"
            )

            training_index = pd.MultiIndex.from_arrays(
                [[model] * len(training_months), training_months],
                names=["PRODUCT_MODEL_ID", "month"],
            )
            training_values = values.reindex(training_index)

            if training_values.isna().any():
                raise ValueError(
                    f"Missing training data for model {model}"
                )

            training_mean = float(training_values.mean())

            for month in forecast_months:
                actual = values.get((model, month), np.nan)
                if pd.isna(actual):
                    raise ValueError(
                        f"Missing actual for {model} in {month:%Y-%m}"
                    )

                prior_year_month = month - pd.DateOffset(years=1)

                # Use the same month from the previous year only if
                # it is not earlier than the model's first observation.
                if prior_year_month >= first_month:
                    prediction = values.get(
                        (model, prior_year_month), np.nan
                    )
                    method = "seasonal_naive"
                else:
                    # Explicit fallback for models without a full
                    # year of observed history.
                    prediction = training_mean
                    method = "seasonal_naive_mean_fallback"

                if pd.isna(prediction):
                    raise ValueError(
                        f"Missing seasonal value for {model}, "
                        f"{month:%Y-%m}"
                    )

                prediction_rows.append(
                    {
                        "split": split_name,
                        "PRODUCT_MODEL_ID": model,
                        "month": month,
                        "actual_order_frequency": float(actual),
                        "predicted_order_frequency": max(
                            0.0, float(prediction)
                        ),
                        "method": method,
                    }
                )

        split_predictions = pd.DataFrame(
            [row for row in prediction_rows if row["split"] == split_name]
        )

        combined_actual = split_predictions["actual_order_frequency"]
        combined_prediction = split_predictions[
            "predicted_order_frequency"
        ]

        combined = metrics(combined_actual, combined_prediction)
        result_rows.append(
            {
                "split": split_name,
                "method": "seasonal_naive_with_mean_fallback",
                "eligible_models": len(eligible_models),
                **combined,
            }
        )

        print(f"\n{split_name.upper()}")
        print("-" * 50)
        print(f"Eligible models: {len(eligible_models)}")
        print(
            split_predictions["method"]
            .value_counts()
            .to_string()
        )
        print(f"MAE: {combined['MAE']:.4f}")
        print(f"WAPE: {combined['WAPE_percent']:.2f}%")
        print(f"Actual total: {combined['actual_total']:.0f}")
        print(f"Predicted total: {combined['predicted_total']:.2f}")

    seasonal_results = pd.DataFrame(result_rows)
    seasonal_predictions = pd.DataFrame(prediction_rows)

    seasonal_results.to_csv(
        MODELS_DIR / "b2b_seasonal_evaluation_summary.csv",
        index=False,
    )
    seasonal_predictions.to_csv(
        MODELS_DIR / "b2b_seasonal_evaluation_predictions.csv",
        index=False,
    )

    comparison = pd.concat(
        [
            previous_results[
                previous_results["split"].isin(
                    ["validation_2028", "test_2029"]
                )
            ],
            seasonal_results,
        ],
        ignore_index=True,
    )

    comparison.to_csv(
        MODELS_DIR / "b2b_all_baselines_comparison.csv",
        index=False,
    )

    print("\nALL BASELINES — COMPARISON")
    print("=" * 75)
    print(
        comparison[
            [
                "split",
                "method",
                "MAE",
                "WAPE_percent",
                "actual_total",
                "predicted_total",
            ]
        ].round(3).to_string(index=False)
    )

    print("\nSaved:")
    print(MODELS_DIR / "b2b_seasonal_evaluation_summary.csv")
    print(MODELS_DIR / "b2b_seasonal_evaluation_predictions.csv")
    print(MODELS_DIR / "b2b_all_baselines_comparison.csv")


if __name__ == "__main__":
    main()
