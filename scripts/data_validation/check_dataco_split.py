
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FILE = ROOT / "data" / "raw" / "DataCoSupplyChainDataset.csv"

df = pd.read_csv(FILE, encoding="latin1", low_memory=False)

# 1. Check how many rows and distinct orders we have
print("--- ORDER STRUCTURE ---")
print("Rows:", len(df))
print("Unique orders:", df["Order Id"].nunique())
print("Rows per order:")
print(df.groupby("Order Id").size().describe().round(2))

# 2. Check whether items in the same order have consistent labels
labels_per_order = df.groupby("Order Id")["Late_delivery_risk"].nunique()
print("\nOrders with conflicting target labels:", (labels_per_order > 1).sum())

# 3. Examine target against scheduled shipping duration
print("\n--- TARGET VS SCHEDULED SHIPPING DAYS ---")
print(pd.crosstab(
    df["Days for shipment (scheduled)"],
    df["Late_delivery_risk"],
    normalize="index"
).round(3))

# 4. Examine target against actual shipping duration for audit only
# Do not use actual duration as a model feature.
print("\n--- TARGET VS ACTUAL SHIPPING DAYS (AUDIT ONLY) ---")
print(pd.crosstab(
    df["Days for shipping (real)"],
    df["Late_delivery_risk"],
    normalize="index"
).round(3))

# 5. Check the order date range for a chronological split
order_dates = pd.to_datetime(
    df["order date (DateOrders)"],
    errors="coerce"
)
print("\n--- CHRONOLOGICAL SPLIT INFORMATION ---")
print("Earliest order:", order_dates.min())
print("Latest order:", order_dates.max())
print("Invalid order dates:", order_dates.isna().sum())

print("\nAudit complete.")
