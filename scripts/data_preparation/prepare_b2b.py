
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# SupplyChain-IQ: Multi-year B2B data preparation
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "b2b_monthly_model_demand_all_years.csv"
)

YEARS = range(2024, 2030)

REQUIRED_COLUMNS = [
    "ORDER_ID",
    "REQUEST_DATE",
    "PRODUCT_MODEL_ID",
]


def read_yearly_file(path):
    """Read a semicolon-separated yearly CSV with common encodings."""
    errors = []

    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            df = pd.read_csv(
                path,
                sep=";",
                encoding=encoding,
                low_memory=False,
                dtype={
                    "ORDER_ID": "string",
                    "PRODUCT_MODEL_ID": "string",
                },
            )

            if len(df.columns) < 2:
                raise ValueError(
                    "Only one column detected; expected semicolon "
                    "separation."
                )

            return df, encoding

        except (
            UnicodeDecodeError,
            pd.errors.ParserError,
            ValueError,
        ) as exc:
            errors.append(f"{encoding}: {exc}")

    raise RuntimeError(
        f"Unable to read {path.name}: " + " | ".join(errors)
    )


def prepare_b2b_all_years():
    """Prepare monthly order-frequency observations from 2024–2029."""

    all_order_model_parts = []
    source_rows_total = 0
    duplicates_total = 0
    deduplicated_rows_total = 0
    per_file_report = []

    # --------------------------------------------------
    # 1. Read, validate and deduplicate each source file.
    # --------------------------------------------------
    for year in YEARS:
        source_path = (
            RAW_DIR / f"{year}_full_anonymus_v2_final.csv"
        )

        if not source_path.exists():
            raise FileNotFoundError(
                f"Required yearly file not found: {source_path}"
            )

        df, encoding = read_yearly_file(source_path)
        original_rows = len(df)

        missing_columns = [
            column
            for column in REQUIRED_COLUMNS
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"{source_path.name} is missing columns: "
                f"{missing_columns}"
            )

        required = df[REQUIRED_COLUMNS]

        if required.isna().any().any():
            raise ValueError(
                f"{source_path.name} contains missing required "
                "identifiers or request dates."
            )

        for column in ("ORDER_ID", "PRODUCT_MODEL_ID"):
            if df[column].str.strip().eq("").any():
                raise ValueError(
                    f"{source_path.name} contains blank {column} values."
                )

        df["REQUEST_DATE"] = pd.to_datetime(
            df["REQUEST_DATE"],
            errors="raise",
        )

        # Check that dates fall within the expected anonymized year.
        outside_year = df["REQUEST_DATE"].dt.year.ne(year)

        if outside_year.any():
            raise ValueError(
                f"{source_path.name} contains "
                f"{outside_year.sum():,} dates outside {year}."
            )

        exact_duplicates = int(df.duplicated().sum())
        df = df.drop_duplicates().copy()

        # Each order must have one request date.
        dates_per_order = (
            df.groupby("ORDER_ID")["REQUEST_DATE"].nunique()
        )

        if dates_per_order.gt(1).any():
            conflicting_orders = int(dates_per_order.gt(1).sum())

            raise ValueError(
                f"{source_path.name} has {conflicting_orders:,} orders "
                "with multiple request dates."
            )

        # One order contributes at most once per product model.
        order_model = df[
            ["ORDER_ID", "REQUEST_DATE", "PRODUCT_MODEL_ID"]
        ].drop_duplicates()

        order_model["month"] = (
            order_model["REQUEST_DATE"]
            .dt.to_period("M")
            .dt.to_timestamp()
        )

        source_rows_total += original_rows
        duplicates_total += exact_duplicates
        deduplicated_rows_total += len(df)

        all_order_model_parts.append(order_model)

        per_file_report.append({
            "year": year,
            "source_rows": original_rows,
            "exact_duplicates_removed": exact_duplicates,
            "rows_after_deduplication": len(df),
            "distinct_orders": df["ORDER_ID"].nunique(),
            "product_models": df["PRODUCT_MODEL_ID"].nunique(),
            "order_model_observations": len(order_model),
            "encoding": encoding,
        })

        print(
            f"{year}: {original_rows:,} source rows, "
            f"{exact_duplicates:,} exact duplicates removed, "
            f"{len(order_model):,} distinct order-model observations"
        )

    # --------------------------------------------------
    # 2. Combine order-model observations across all years.
    # --------------------------------------------------
    combined = pd.concat(
        all_order_model_parts,
        ignore_index=True,
    )

    # Do not silently remove cross-file duplicate transactions.
    # Check whether order IDs are unique across the combined data.
    order_dates = combined[
        ["ORDER_ID", "REQUEST_DATE"]
    ].drop_duplicates()

    order_date_counts = order_dates.groupby("ORDER_ID")[
        "REQUEST_DATE"
    ].nunique()

    if order_date_counts.gt(1).any():
        raise ValueError(
            "An ORDER_ID appears with different request dates across "
            "the combined datasets. Investigate before aggregation."
        )

    duplicate_order_model = combined.duplicated(
        ["ORDER_ID", "PRODUCT_MODEL_ID"],
        keep=False,
    )

    if duplicate_order_model.any():
        examples = combined.loc[
            duplicate_order_model,
            ["ORDER_ID", "REQUEST_DATE", "PRODUCT_MODEL_ID"],
        ].head(10)

        raise ValueError(
            "Duplicate order-model pairs exist across yearly files. "
            "These may indicate cross-file duplication or reused order "
            "IDs. Investigate before combining.\n"
            + examples.to_string(index=False)
        )

    # --------------------------------------------------
    # 3. Aggregate monthly order frequency by product model.
    # --------------------------------------------------
    monthly = (
        combined.groupby(["month", "PRODUCT_MODEL_ID"])
        .size()
        .rename("order_frequency")
        .reset_index()
    )

    start_month = combined["month"].min()
    end_month = combined["month"].max()

    months = pd.date_range(
        start=start_month,
        end=end_month,
        freq="MS",
    )

    product_models = sorted(
        combined["PRODUCT_MODEL_ID"].unique()
    )

    complete_grid = pd.MultiIndex.from_product(
        [months, product_models],
        names=["month", "PRODUCT_MODEL_ID"],
    ).to_frame(index=False)

    monthly = complete_grid.merge(
        monthly,
        on=["month", "PRODUCT_MODEL_ID"],
        how="left",
        validate="one_to_one",
    )

    monthly["order_frequency"] = (
        monthly["order_frequency"]
        .fillna(0)
        .astype("int64")
    )

    monthly = monthly.sort_values(
        ["month", "PRODUCT_MODEL_ID"]
    ).reset_index(drop=True)

    # --------------------------------------------------
    # 4. Validate output consistency.
    # --------------------------------------------------
    total_order_model_observations = len(combined)

    if int(monthly["order_frequency"].sum()) != (
        total_order_model_observations
    ):
        raise AssertionError(
            "Validation failed: monthly frequencies do not sum to "
            "the number of order-model observations."
        )

    expected_grid_size = len(months) * len(product_models)

    if len(monthly) != expected_grid_size:
        raise AssertionError(
            "Validation failed: the monthly product-model grid is "
            "incomplete."
        )

    # --------------------------------------------------
    # 5. Save to a new file, preserving existing outputs.
    # --------------------------------------------------
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    monthly.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    report_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "b2b_all_years_preparation_report.csv"
    )

    pd.DataFrame(per_file_report).to_csv(
        report_path,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------
    # 6. Print preparation report.
    # --------------------------------------------------
    print("\nMULTI-YEAR B2B DATA PREPARATION COMPLETE")
    print("-" * 50)
    print(f"Files processed: {len(per_file_report)}")
    print(f"Source rows: {source_rows_total:,}")
    print(f"Exact duplicate rows removed: {duplicates_total:,}")
    print(f"Rows after per-file deduplication: {deduplicated_rows_total:,}")
    print(f"Distinct orders: {combined['ORDER_ID'].nunique():,}")
    print(f"Product models: {len(product_models):,}")
    print(f"Months covered: {len(months):,}")
    print(f"Monthly observations: {len(monthly):,}")
    print(
        "Total order-model observations: "
        f"{monthly['order_frequency'].sum():,}"
    )
    print(
        "Zero-frequency combinations: "
        f"{(monthly['order_frequency'] == 0).sum():,}"
    )
    print(f"First month: {start_month:%Y-%m}")
    print(f"Last month: {end_month:%Y-%m}")
    print(f"\nCombined output:\n{OUTPUT_PATH}")
    print(f"Per-file report:\n{report_path}")
    print("\nSample records:")
    print(monthly.head(10).to_string(index=False))


if __name__ == "__main__":
    prepare_b2b_all_years()
