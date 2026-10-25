
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"

INPUT_FILES = {
    "baseline": MODELS_DIR / "b2b_multiyear_evaluation_summary.csv",
    "arima_101": MODELS_DIR / "b2b_arima_evaluation_summary.csv",
    "arima_111": MODELS_DIR / "b2b_arima_evaluation_summary_111.csv",
}

OUTPUT_FILE = MODELS_DIR / "b2b_forecasting_model_comparison.csv"


def load_summary(path, method_label):
    """Load one evaluation report and standardize its method name."""
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation report not found: {path}\n"
            "Run the corresponding evaluation script first."
        )

    df = pd.read_csv(path)

    required_columns = {
        "split",
        "MAE",
        "WAPE_percent",
        "actual_total",
        "predicted_total",
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"{path.name} is missing columns: {sorted(missing)}"
        )

    df = df.copy()

    if method_label == "baseline":
        if "method" not in df.columns:
            raise ValueError(
                f"{path.name} must contain a 'method' column."
            )
        df["method"] = df["method"].astype(str)
    else:
        df["method"] = method_label

    return df


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    reports = []

    for label, path in INPUT_FILES.items():
        df = load_summary(path, label)

        if label == "baseline":
            reports.append(df)
        else:
            reports.append(df)

    comparison = pd.concat(reports, ignore_index=True, sort=False)

    # Keep only the metrics needed for model comparison.
    columns = [
        "split",
        "method",
        "eligible_models",
        "models_using_arima",
        "models_using_fallback",
        "MAE",
        "WAPE_percent",
        "actual_total",
        "predicted_total",
        "total_error",
    ]

    available_columns = [
        column for column in columns if column in comparison.columns
    ]
    comparison = comparison[available_columns]

    comparison["MAE"] = pd.to_numeric(
        comparison["MAE"], errors="coerce"
    )
    comparison["WAPE_percent"] = pd.to_numeric(
        comparison["WAPE_percent"], errors="coerce"
    )

    comparison = comparison.sort_values(
        ["split", "MAE", "WAPE_percent"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    comparison.to_csv(OUTPUT_FILE, index=False)

    for split_name, group in comparison.groupby("split", sort=False):
        print(f"\n{split_name.upper()}")
        print("-" * 80)

        display_columns = [
            column
            for column in [
                "method",
                "MAE",
                "WAPE_percent",
                "actual_total",
                "predicted_total",
            ]
            if column in group.columns
        ]

        print(group[display_columns].round(3).to_string(index=False))

        valid = group.dropna(subset=["MAE", "WAPE_percent"])
        if not valid.empty:
            best = valid.sort_values(
                ["MAE", "WAPE_percent"]
            ).iloc[0]

            print(
                "\nLowest-error method for this split: "
                f"{best['method']} "
                f"(MAE={best['MAE']:.4f}, "
                f"WAPE={best['WAPE_percent']:.2f}%)"
            )

    print("\nCOMPARISON COMPLETE")
    print(f"Saved report: {OUTPUT_FILE}")
    print(
        "\nImportant: use validation_2028 for model selection. "
        "Treat test_2029 as an already inspected test period, "
        "not a pristine holdout."
    )


if __name__ == "__main__":
    main()
