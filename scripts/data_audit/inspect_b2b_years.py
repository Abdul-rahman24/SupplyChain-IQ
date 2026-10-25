
from pathlib import Path
import hashlib
import json
import sys

import pandas as pd


# ---------------------------------------------------------
# SupplyChain-IQ: B2B yearly dataset audit
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "b2b_file_audit.csv"

YEARS = range(2024, 2030)

# Candidate names are matched case-insensitively.
DATE_CANDIDATES = {
    "date",
    "orderdate",
    "order_date",
    "orderdatekey",
    "order_date_key",
    "createdat",
    "created_at",
    "timestamp",
    "datetime",
}

ORDER_CANDIDATES = {
    "order",
    "orderid",
    "order_id",
    "ordernumber",
    "order_number",
    "orderno",
    "order_no",
}

ENCODING_CANDIDATES = ("utf-8-sig", "utf-8", "cp1252", "latin1")


def normalize_name(name):
    """Normalize column names for matching."""
    return "".join(
        character.lower()
        for character in str(name).strip()
        if character.isalnum()
    )


def find_matching_columns(columns, candidates):
    """Return columns whose normalized names match candidates."""
    normalized_candidates = {
        normalize_name(candidate) for candidate in candidates
    }

    return [
        column
        for column in columns
        if normalize_name(column) in normalized_candidates
    ]


def read_dataset(path):
    """Read a semicolon-separated CSV, trying common encodings."""
    errors = []

    for encoding in ENCODING_CANDIDATES:
        try:
            dataframe = pd.read_csv(
                path,
                sep=";",
                encoding=encoding,
                low_memory=False,
            )

            # A likely delimiter or encoding problem should be visible,
            # rather than silently accepting a one-column dataset.
            if len(dataframe.columns) < 2:
                raise ValueError(
                    "Only one column detected. Check the CSV delimiter."
                )

            return dataframe, encoding

        except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
            errors.append(f"{encoding}: {exc}")

    raise RuntimeError(
        f"Could not read {path.name}.\n" + "\n".join(errors)
    )


def file_sha256(path):
    """Calculate the raw file hash to detect byte-identical files."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def parse_dates(dataframe, column):
    """Parse a date column without changing the source data."""
    values = dataframe[column].astype("string").str.strip()

    # Handles common ISO dates and timestamps, as well as many
    # conventional day/month/year formats.
    parsed = pd.to_datetime(values, errors="coerce", format="mixed")

    # Retry failed values with day-first interpretation. This helps
    # identify dates written as DD/MM/YYYY.
    failed = parsed.isna() & values.notna() & values.ne("")

    if failed.any():
        retry = pd.to_datetime(
            values.loc[failed],
            errors="coerce",
            format="mixed",
            dayfirst=True,
        )
        parsed.loc[failed] = retry

    return parsed


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    audit_rows = []
    loaded_datasets = {}
    file_hashes = {}

    print("=" * 78)
    print("SUPPLYCHAIN-IQ | B2B YEARLY DATASET AUDIT")
    print("=" * 78)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Source folder: {RAW_DIR}")
    print(f"Output file: {OUTPUT_FILE}")

    # Phase 1: Load and inspect each available yearly file.
    for year in YEARS:
        path = RAW_DIR / f"{year}_full_anonymus_v2_final.csv"

        print("\n" + "-" * 78)
        print(f"YEAR {year}: {path.name}")

        if not path.exists():
            print("STATUS: FILE NOT FOUND")

            audit_rows.append({
                "filename": path.name,
                "expected_year": year,
                "file_found": False,
                "error": "File not found",
            })
            continue

        try:
            dataframe, encoding = read_dataset(path)
            loaded_datasets[year] = dataframe
            file_hashes[year] = file_sha256(path)

            columns = list(dataframe.columns)
            date_columns = find_matching_columns(
                columns, DATE_CANDIDATES
            )
            order_columns = find_matching_columns(
                columns, ORDER_CANDIDATES
            )

            exact_duplicates = int(
                dataframe.duplicated(keep="first").sum()
            )

            date_details = []
            date_min = None
            date_max = None
            date_valid_counts = {}

            for column in date_columns:
                parsed = parse_dates(dataframe, column)
                valid_dates = parsed.dropna()

                date_valid_counts[column] = int(len(valid_dates))

                if not valid_dates.empty:
                    date_details.append(
                        f"{column}: {valid_dates.min()} to "
                        f"{valid_dates.max()} "
                        f"({len(valid_dates)} parseable values)"
                    )

                    current_min = valid_dates.min()
                    current_max = valid_dates.max()

                    date_min = (
                        current_min
                        if date_min is None
                        else min(date_min, current_min)
                    )
                    date_max = (
                        current_max
                        if date_max is None
                        else max(date_max, current_max)
                    )

            order_details = []
            order_sets = {}

            for column in order_columns:
                # String comparison avoids merging identifiers such as
                # "00123" and "123" solely because of numeric parsing.
                identifiers = (
                    dataframe[column]
                    .astype("string")
                    .str.strip()
                )

                identifiers = identifiers[
                    identifiers.notna() & identifiers.ne("")
                ]

                unique_ids = set(identifiers.tolist())
                order_sets[column] = unique_ids

                order_details.append(
                    f"{column}: {len(unique_ids)} distinct non-empty IDs"
                )

            # A schema fingerprint ignores column order but preserves
            # column names and inferred data types.
            schema_description = sorted(
                (
                    str(column),
                    str(dataframe[column].dtype),
                )
                for column in columns
            )
            schema_fingerprint = hashlib.sha256(
                json.dumps(
                    schema_description,
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest()

            row = {
                "filename": path.name,
                "expected_year": year,
                "file_found": True,
                "encoding": encoding,
                "row_count": len(dataframe),
                "column_count": len(columns),
                "columns": " | ".join(map(str, columns)),
                "schema_fingerprint": schema_fingerprint,
                "raw_file_sha256": file_hashes[year],
                "exact_duplicate_rows": exact_duplicates,
                "rows_after_exact_dedup": (
                    len(dataframe) - exact_duplicates
                ),
                "missing_values_total": int(
                    dataframe.isna().sum().sum()
                ),
                "date_columns": " | ".join(map(str, date_columns)),
                "date_min": (
                    str(date_min) if date_min is not None else None
                ),
                "date_max": (
                    str(date_max) if date_max is not None else None
                ),
                "date_parseable_counts": json.dumps(
                    date_valid_counts
                ),
                "order_columns": " | ".join(map(str, order_columns)),
                "order_counts": json.dumps({
                    column: len(ids)
                    for column, ids in order_sets.items()
                }),
                "error": None,
            }

            audit_rows.append(row)

            print(f"Encoding: {encoding}")
            print(f"Rows: {len(dataframe):,}")
            print(f"Columns: {len(columns)}")
            print(f"Exact duplicate rows: {exact_duplicates:,}")
            print(
                f"Missing values: "
                f"{dataframe.isna().sum().sum():,}"
            )
            print("Column names and inferred data types:")

            for column in columns:
                print(
                    f"  - {column}: {dataframe[column].dtype} "
                    f"(missing={dataframe[column].isna().sum():,})"
                )

            print(
                "Date columns: "
                + (", ".join(map(str, date_columns)) or "None detected")
            )

            for detail in date_details:
                print(f"  - {detail}")

            if not date_details and date_columns:
                print("  - No parseable dates found in detected columns.")

            print(
                "Order columns: "
                + (", ".join(map(str, order_columns)) or "None detected")
            )

            for detail in order_details:
                print(f"  - {detail}")

            print(f"Raw SHA-256: {file_hashes[year]}")
            print(f"Schema fingerprint: {schema_fingerprint}")

        except Exception as exc:
            print(f"ERROR: {type(exc).__name__}: {exc}")

            audit_rows.append({
                "filename": path.name,
                "expected_year": year,
                "file_found": True,
                "error": f"{type(exc).__name__}: {exc}",
            })

    # Phase 2: Compare schemas, raw file hashes, and order IDs.
    print("\n" + "=" * 78)
    print("CROSS-FILE COMPARISON")
    print("=" * 78)

    successful_years = sorted(loaded_datasets)

    for index, year_a in enumerate(successful_years):
        dataframe_a = loaded_datasets[year_a]
        columns_a = set(map(str, dataframe_a.columns))

        for year_b in successful_years[index + 1:]:
            dataframe_b = loaded_datasets[year_b]
            columns_b = set(map(str, dataframe_b.columns))

            common_columns = sorted(columns_a & columns_b)
            only_a = sorted(columns_a - columns_b)
            only_b = sorted(columns_b - columns_a)

            print(f"\n{year_a} vs {year_b}")
            print(f"  Shared column names: {len(common_columns)}")
            print(f"  Only in {year_a}: {only_a or 'None'}")
            print(f"  Only in {year_b}: {only_b or 'None'}")

            if file_hashes[year_a] == file_hashes[year_b]:
                print("  WARNING: Raw files are byte-identical.")

            # Check order overlap when compatible order columns exist.
            order_columns_a = find_matching_columns(
                dataframe_a.columns, ORDER_CANDIDATES
            )
            order_columns_b = find_matching_columns(
                dataframe_b.columns, ORDER_CANDIDATES
            )

            for column_a in order_columns_a:
                matching_b = [
                    column_b
                    for column_b in order_columns_b
                    if normalize_name(column_a)
                    == normalize_name(column_b)
                ]

                for column_b in matching_b:
                    ids_a = set(
                        dataframe_a[column_a]
                        .astype("string")
                        .str.strip()
                        .dropna()
                    )
                    ids_b = set(
                        dataframe_b[column_b]
                        .astype("string")
                        .str.strip()
                        .dropna()
                    )

                    ids_a.discard("")
                    ids_b.discard("")

                    overlap = ids_a & ids_b

                    if ids_a and ids_b:
                        print(
                            f"  Order-ID overlap ({column_a}): "
                            f"{len(overlap):,} IDs "
                            f"({len(overlap) / len(ids_a | ids_b):.2%} "
                            "of the combined distinct-ID set)"
                        )

                        # Matching IDs may be legitimate across files;
                        # record the evidence, not an automatic verdict.
                        for audit_row in audit_rows:
                            if audit_row.get("expected_year") == year_a:
                                audit_row.setdefault(
                                    "cross_file_order_overlap", []
                                ).append(
                                    f"{year_b}:{column_a}/{column_b}:"
                                    f"{len(overlap)}"
                                )

    # Convert list-valued fields to CSV-friendly text.
    for row in audit_rows:
        if isinstance(row.get("cross_file_order_overlap"), list):
            row["cross_file_order_overlap"] = " | ".join(
                row["cross_file_order_overlap"]
            )

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 78)
    print("AUDIT COMPLETE")
    print("=" * 78)
    print(f"Files found and loaded: {len(successful_years)} / 6")
    print(f"Audit summary saved to:\n{OUTPUT_FILE}")

    if len(successful_years) < 6:
        print(
            "\nWARNING: Not all six datasets were successfully inspected. "
            "Review the missing-file or error entries before deciding "
            "whether to combine any files."
        )

    print(
        "\nImportant: Matching order IDs are a potential-overlap signal, "
        "not proof of duplicated transactions. Different years may reuse "
        "IDs, and files may contain legitimate shared orders."
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAudit interrupted by user.")
        sys.exit(130)
