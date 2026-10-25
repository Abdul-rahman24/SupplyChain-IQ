
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "data" / "raw" / "DataCoSupplyChainDataset.csv"
OUTPUT_DIR = ROOT / "data" / "processed"

df = pd.read_csv(
    INPUT_FILE,
    encoding="latin1",
    low_memory=False
)

# Only use features that could plausibly be known at order time.
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
]

target = "Late_delivery_risk"
order_id = "Order Id"
date_col = "order date (DateOrders)"

# Parse order dates and derive calendar features.
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col, target, order_id]).copy()

df["order_year"] = df[date_col].dt.year
df["order_month"] = df[date_col].dt.month
df["order_day_of_week"] = df[date_col].dt.dayofweek

feature_columns += [
    "order_year",
    "order_month",
    "order_day_of_week",
]

# Sort distinct orders chronologically.
orders = (
    df.groupby(order_id)[date_col]
    .min()
    .sort_values()
)

cutoff = int(len(orders) * 0.80)
train_order_ids = set(orders.index[:cutoff])
test_order_ids = set(orders.index[cutoff:])

train = df[df[order_id].isin(train_order_ids)].copy()
test = df[df[order_id].isin(test_order_ids)].copy()

# Remove grouping/date identifiers from model-ready files.
train = train[feature_columns + [target]]
test = test[feature_columns + [target]]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
train.to_csv(OUTPUT_DIR / "dataco_train.csv", index=False)
test.to_csv(OUTPUT_DIR / "dataco_test.csv", index=False)

print("--- CHRONOLOGICAL SPLIT COMPLETE ---")
print("Distinct orders:", len(orders))
print("Training orders:", len(train_order_ids))
print("Testing orders:", len(test_order_ids))
print("Training rows:", len(train))
print("Testing rows:", len(test))
print("Training date range:",
      df[df[order_id].isin(train_order_ids)][date_col].min(),
      "to",
      df[df[order_id].isin(train_order_ids)][date_col].max())
print("Testing date range:",
      df[df[order_id].isin(test_order_ids)][date_col].min(),
      "to",
      df[df[order_id].isin(test_order_ids)][date_col].max())
print("Shared orders between splits:",
      len(train_order_ids.intersection(test_order_ids)))
print("Training target rate:", round(train[target].mean(), 4))
print("Testing target rate:", round(test[target].mean(), 4))
print("Saved files:")
print(OUTPUT_DIR / "dataco_train.csv")
print(OUTPUT_DIR / "dataco_test.csv")
