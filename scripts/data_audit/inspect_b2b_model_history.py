
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

YEARS = range(2024, 2030)

MODEL_COLUMN = "PRODUCT_MODEL_ID"
DATE_COLUMN = "REQUEST_DATE"
ORDER_COLUMN = "ORDER_ID"


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    yearly_observations = []

    for year in YEARS:
        file_path = RAW_DIR / f"{year}_full_anonymus_v2_final.csv"

        print(f"Reading {year}: {file_path.name}")

        df = pd.read_csv(
            file_path,
            sep=";",
            dtype={
                ORDER_COLUMN: "string",
                MODEL_COLUMN: "string",
            },
            usecols=[ORDER_COLUMN, DATE_COLUMN, MODEL_COLUMN],
        )

        df[DATE_COLUMN] = pd.to_datetime(
            df[DATE_COLUMN],
            errors="coerce",
        )

        df = df.dropna(
            subset=[ORDER_COLUMN, DATE_COLUMN, MODEL_COLUMN]
        )

        # Count each order only once per product model and date.
        df = df.drop_duplicates(
            subset=[ORDER_COLUMN, DATE_COLUMN, MODEL_COLUMN]
        )

        df["month"] = (
            df[DATE_COLUMN]
            .dt.to_period("M")
            .dt.to_timestamp()
        )

        monthly = (
            df.groupby(["month", MODEL_COLUMN])[ORDER_COLUMN]
            .nunique()
            .reset_index(name="order_frequency")
        )

        yearly_observations.append(monthly)

    observed = pd.concat(yearly_observations, ignore_index=True)

    # Combine observations into one monthly record per product model.
    observed = (
        observed.groupby(["month", MODEL_COLUMN], as_index=False)
        ["order_frequency"]
        .sum()
    )

    # Find each model's first and last observed month.
    history = (
        observed.groupby(MODEL_COLUMN)
        .agg(
            first_observed_month=("month", "min"),
            last_observed_month=("month", "max"),
            months_with_orders=("month", "nunique"),
            total_order_frequency=("order_frequency", "sum"),
        )
        .reset_index()
    )

    history["first_observed_year"] = (
        history["first_observed_month"].dt.year
    )

    history["last_observed_year"] = (
        history["last_observed_month"].dt.year
    )

    # Summarize when models first appear in the dataset.
    first_year_summary = (
        history.groupby("first_observed_year")
        .agg(
            product_models=(MODEL_COLUMN, "nunique"),
            total_order_frequency=("total_order_frequency", "sum"),
        )
        .reset_index()
        .sort_values("first_observed_year")
    )

    # Count models that are absent before their first observed month.
    all_months = pd.date_range(
        start=observed["month"].min(),
        end=observed["month"].max(),
        freq="MS",
    )

    observed_months = observed[
        ["month", MODEL_COLUMN]
    ].drop_duplicates()

    model_first_month = history.set_index(MODEL_COLUMN)[
        "first_observed_month"
    ]

    observed_months["first_observed_month"] = (
        observed_months[MODEL_COLUMN].map(model_first_month)
    )

    observed_months["before_first_observation"] = (
        observed_months["month"]
        < observed_months["first_observed_month"]
    )

    print("\nPRODUCT MODEL HISTORY SUMMARY")
    print("---------------------------------------------")
    print(f"Distinct product models: {len(history):,}")
    print(f"Observed monthly model records: {len(observed):,}")
    print(
        "Models first observed in 2024: "
        f"{(history['first_observed_year'] == 2024).sum():,}"
    )
    print(
        "Models first observed after 2024: "
        f"{(history['first_observed_year'] > 2024).sum():,}"
    )
    print(
        "Model-month records before a model's first observation "
        f"in the full grid: "
        f"{sum((len(all_months) - history['months_with_orders']) * 0):,}"
    )

    print("\nModels by first observed year:")
    print(first_year_summary.to_string(index=False))

    history_path = PROCESSED_DIR / "b2b_product_model_history.csv"
    summary_path = PROCESSED_DIR / "b2b_product_model_first_year_summary.csv"

    history.to_csv(history_path, index=False)
    first_year_summary.to_csv(summary_path, index=False)

    print("\nSaved reports:")
    print(history_path)
    print(summary_path)

    print(
        "\nNote: First observed month means the first month found in "
        "these files. It does not prove the product's actual launch date."
    )


if __name__ == "__main__":
    main()
