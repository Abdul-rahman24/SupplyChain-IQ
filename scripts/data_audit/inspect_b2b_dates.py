
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

SUMMARY_FILE = OUTPUT_DIR / "b2b_date_audit_summary.csv"
YEAR_COUNTS_FILE = OUTPUT_DIR / "b2b_date_year_distribution.csv"

YEARS = range(2024, 2030)
DATE_COLUMN = "REQUEST_DATE"

ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin1")


def read_csv(path):
    """Read a semicolon-separated source file."""
    errors = []

    for encoding in ENCODINGS:
        try:
            df = pd.read_csv(
                path,
                sep=";",
                encoding=encoding,
                low_memory=False,
                dtype={DATE_COLUMN: "string"},
            )

            if len(df.columns) < 2:
                raise ValueError("Unexpected single-column CSV.")

            return df

        except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
            errors.append(f"{encoding}: {exc}")

    raise RuntimeError(
        f"Could not read {path.name}: " + " | ".join(errors)
    )


def parse_request_dates(series):
    """Parse dates and retry ambiguous values using day-first parsing."""
    values = series.astype("string").str.strip()

    missing = values.isna() | values.eq("")
    parsed = pd.to_datetime(
        values.mask(missing),
        errors="coerce",
        format="mixed",
    )

    retry_mask = ~missing & parsed.isna()

    if retry_mask.any():
        retry = pd.to_datetime(
            values.loc[retry_mask],
            errors="coerce",
            format="mixed",
            dayfirst=True,
    )
        parsed.loc[retry_mask] = retry

    return parsed, missing


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summaries = []
    yearly_counts = []

    # Exact row hashes are used to detect matching complete records
    # across files without loading all six datasets simultaneously.
    seen_hashes = {}
    cross_file_matches = {}

    print("=" * 75)
    print("SUPPLYCHAIN-IQ | REQUEST_DATE AND CROSS-FILE AUDIT")
    print("=" * 75)

    for expected_year in YEARS:
        path = RAW_DIR / (
            f"{expected_year}_full_anonymus_v2_final.csv"
        )

        print(f"\n{'-' * 75}")
        print(f"File: {path.name}")

        if not path.exists():
            print("ERROR: File not found.")

            summaries.append({
                "filename": path.name,
                "expected_year": expected_year,
                "status": "missing_file",
            })
            continue

        try:
            df = read_csv(path)

            if DATE_COLUMN not in df.columns:
                raise ValueError(
                    f"Required date column {DATE_COLUMN!r} not found. "
                    f"Columns: {list(df.columns)}"
                )

            parsed, missing_mask = parse_request_dates(
                df[DATE_COLUMN]
            )

            invalid_mask = ~missing_mask & parsed.isna()
            valid = parsed.dropna()

            wrong_year_mask = parsed.notna() & (
                parsed.dt.year != expected_year
            )

            wrong_year_count = int(wrong_year_mask.sum())

            summary = {
                "filename": path.name,
                "expected_year": expected_year,
                "status": "success",
                "row_count": len(df),
                "missing_request_dates": int(missing_mask.sum()),
                "invalid_request_dates": int(invalid_mask.sum()),
                "valid_request_dates": int(parsed.notna().sum()),
                "min_request_date": (
                    str(valid.min()) if not valid.empty else None
                ),
                "max_request_date": (
                    str(valid.max()) if not valid.empty else None
                ),
                "rows_outside_filename_year": wrong_year_count,
                "distinct_actual_years": (
                    ", ".join(
                        map(
                            str,
                            sorted(valid.dt.year.unique().tolist()),
                        )
                    )
                    if not valid.empty
                    else ""
                ),
            }

            summaries.append(summary)

            print(f"Rows: {len(df):,}")
            print(f"Missing dates: {missing_mask.sum():,}")
            print(f"Invalid non-empty dates: {invalid_mask.sum():,}")
            print(f"Valid dates: {parsed.notna().sum():,}")

            if not valid.empty:
                print(f"Earliest date: {valid.min()}")
                print(f"Latest date: {valid.max()}")

            print(
                f"Rows outside expected year {expected_year}: "
                f"{wrong_year_count:,}"
            )

            # Record counts by actual calendar year and month.
            dated = pd.DataFrame({
                "actual_year": parsed.dt.year,
                "actual_month": parsed.dt.month,
            })

            distribution = (
                dated.dropna()
                .groupby(["actual_year", "actual_month"])
                .size()
                .reset_index(name="row_count")
            )

            for _, record in distribution.iterrows():
                yearly_counts.append({
                    "filename": path.name,
                    "expected_year": expected_year,
                    "actual_year": int(record["actual_year"]),
                    "actual_month": int(record["actual_month"]),
                    "row_count": int(record["row_count"]),
                })

            print("Record distribution by actual year:")
            if valid.empty:
                print("  No valid dates found.")
            else:
                for actual_year, count in valid.dt.year.value_counts().sort_index().items():
                    print(f"  {actual_year}: {count:,} rows")

            if invalid_mask.any():
                print("Examples of invalid non-empty date values:")
                print(
                    df.loc[invalid_mask, DATE_COLUMN]
                    .drop_duplicates()
                    .head(10)
                    .to_string(index=False)
                )

            # Compare complete records across files.
            # Normalize each field as text and hash the full row.
            normalized = df.fillna("").astype(str)
            row_hashes = pd.util.hash_pandas_object(
                normalized,
                index=False,
            )

            matches_for_file = 0

            for row_hash in row_hashes:
                key = int(row_hash)

                previous_years = seen_hashes.get(key, set())
                other_years = previous_years - {expected_year}

                if other_years:
                    matches_for_file += 1

                    for other_year in other_years:
                        pair = (min(other_year, expected_year),
                                max(other_year, expected_year))
                        cross_file_matches[pair] = (
                            cross_file_matches.get(pair, 0) + 1
                        )

                seen_hashes.setdefault(key, set()).add(expected_year)

            print(
                "Rows matching complete records in previously inspected "
                f"files (hash-based candidate count): {matches_for_file:,}"
            )

        except Exception as exc:
            print(f"ERROR: {type(exc).__name__}: {exc}")

            summaries.append({
                "filename": path.name,
                "expected_year": expected_year,
                "status": f"error: {type(exc).__name__}: {exc}",
            })

    pd.DataFrame(summaries).to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        yearly_counts,
        columns=[
            "filename",
            "expected_year",
            "actual_year",
            "actual_month",
            "row_count",
        ],
    ).to_csv(
        YEAR_COUNTS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\n{'=' * 75}")
    print("CROSS-FILE EXACT-RECORD CANDIDATES")
    print("=" * 75)

    if not cross_file_matches:
        print("No matching complete-row hash candidates detected.")
    else:
        for (year_a, year_b), count in sorted(cross_file_matches.items()):
            print(
                f"{year_a} vs {year_b}: "
                f"{count:,} matching row occurrences"
            )

    print(f"\nSummary saved to: {SUMMARY_FILE}")
    print(f"Monthly distribution saved to: {YEAR_COUNTS_FILE}")
    print(
        "\nNote: Matching row hashes are candidates for investigation, "
        "not definitive proof of duplicate records. A hash collision is "
        "theoretically possible."
    )
    print(
        "Date ranges and row matches alone cannot establish data "
        "provenance or prove that future-dated observations are genuine."
    )


if __name__ == "__main__":
    main()
