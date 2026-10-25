
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "data" / "raw" / "DataCoSupplyChainDataset.csv"
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "dataco_ml_ready.csv"

# Load source data
df = pd.read_csv(
    INPUT_FILE,
    encoding="latin1",
    low_memory=False
)

# Columns selected because they may be available at order time.
# We deliberately exclude actual delivery outcomes and personal data.
feature_columns = [
    "Type",
    "Days for shipment (scheduled)",
    "Category Name",
    "Customer Segment",
    "Department Name",
    "Market",
    "Order Country",
    "Order Region",
    "Shipping Mode",
    "Order Item Quantity",
    "Order Item Product Price",
    "Order Item Discount Rate",
    "order date (DateOrders)",
]

target_column = "Late_delivery_risk"

required_columns = feature_columns + [target_column]
missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns are missing: {missing_columns}"
    )

# Select only the required columns
ml_df = df[required_columns].copy()

# Parse order date
ml_df["order date (DateOrders)"] = pd.to_datetime(
    ml_df["order date (DateOrders)"],
    errors="coerce"
)

# Remove rows with invalid dates or missing target labels
ml_df = ml_df.dropna(
    subset=["order date (DateOrders)", target_column]
)

# Create date features that could be known at order time
order_date = ml_df["order date (DateOrders)"]

ml_df["order_year"] = order_date.dt.year
ml_df["order_month"] = order_date.dt.month
ml_df["order_day_of_week"] = order_date.dt.dayofweek

# Remove raw timestamp after extracting its features
ml_df = ml_df.drop(columns=["order date (DateOrders)"])

# Ensure the target is binary
ml_df[target_column] = ml_df[target_column].astype(int)

if not set(ml_df[target_column].unique()).issubset({0, 1}):
    raise ValueError("Target contains values other than 0 and 1.")

# Save processed data
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ml_df.to_csv(OUTPUT_FILE, index=False)

print("--- DATA PREPARATION COMPLETE ---")
print("Original rows:", len(df))
print("Prepared rows:", len(ml_df))
print("Feature count:", len(ml_df.columns) - 1)
print("Target column:", target_column)
print("\nTarget distribution:")
print(ml_df[target_column].value_counts())
print("\nMissing values:")
print(ml_df.isnull().sum().sort_values(ascending=False).head(10))
print("\nSaved to:", OUTPUT_FILE)
