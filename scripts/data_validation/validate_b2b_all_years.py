
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "b2b_monthly_model_demand_all_years.csv"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "b2b_all_years_validation_report.csv"
)

REQUIRED_COLUMNS = [
    "month",
    "PRODUCT_MODEL_ID",
    "order_frequency",
]


def validate_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Combined dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH,
        dtype={"PRODUCT_MODEL_ID": "string"},
    )

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df["month"] = pd.to_datetime(
        df["month"],
        errors="raise",
    )

    if df[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("Required fields contain missing values.")

    if df["PRODUCT_MODEL_ID"].str.strip().eq("").any():
        raise ValueError("Blank product model identifiers found.")

    if not df["month"].dt.is_month_start.all():
        raise ValueError(
            "Some dates are not normalized to the first of the month."
        )

    if (df["month"].dt.day != 1).any():
        raise ValueError("Non-month-start dates detected.")

    if df.duplicated(
        ["month", "PRODUCT_MODEL_ID"]
    ).any():
        raise ValueError(
            "Duplicate month/product-model combinations found."
        )

    if not pd.api.types.is_numeric_dtype(df["order_frequency"]):
        raise ValueError("order_frequency is not numeric.")

    if not (df["order_frequency"] >= 0).all():
        raise ValueError("Negative order frequencies found.")

    if not (df["order_frequency"] % 1 == 0).all():
        raise ValueError("Non-integer order frequencies found.")

    months = pd.date_range(
        df["month"].min(),
        df["month"].max(),
        freq="MS",
    )

    models = sorted(df["PRODUCT_MODEL_ID"].unique())

    expected_rows = len(months) * len(models)

    if len(df) != expected_rows:
        raise ValueError(
            f"Incomplete monthly grid: found {len(df):,} rows, "
            f"expected {expected_rows:,}."
        )

    expected_combinations = pd.MultiIndex.from_product(
        [months, models],
        names=["month", "PRODUCT_MODEL_ID"],
    )

    actual_combinations = pd.MultiIndex.from_frame(
        df[["month", "PRODUCT_MODEL_ID"]]
    )

    missing_combinations = expected_combinations.difference(
        actual_combinations
    )

    if len(missing_combinations):
        raise ValueError(
            f"{len(missing_combinations):,} monthly combinations missing."
        )

    monthly_totals = (
        df.groupby("month")["order_frequency"]
        .sum()
        .rename("total_order_frequency")
    )

    report = pd.DataFrame({
        "month": monthly_totals.index,
        "total_order_frequency": monthly_totals.values,
    })

    report.to_csv(
        REPORT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nB2B MULTI-YEAR VALIDATION PASSED")
    print("-" * 45)
    print(f"Dataset: {DATA_PATH}")
    print(f"Rows: {len(df):,}")
    print(f"Distinct months: {df['month'].nunique():,}")
    print(f"Distinct product models: {len(models):,}")
    print(f"Expected grid rows: {expected_rows:,}")
    print(f"Missing combinations: {len(missing_combinations):,}")
    print(f"Total order frequency: {df['order_frequency'].sum():,}")
    print(
        "Zero-frequency combinations: "
        f"{(df['order_frequency'] == 0).sum():,}"
    )
    print(
        f"Monthly range: {df['month'].min():%Y-%m} "
        f"to {df['month'].max():%Y-%m}"
    )
    print(f"Validation report: {REPORT_PATH}")

    print("\nMonthly frequency totals:")
    print(report.to_string(index=False))


if __name__ == "__main__":
    validate_dataset()
