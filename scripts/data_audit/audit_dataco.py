
from pathlib import Path
import pandas as pd

file_path = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "raw"
    / "DataCoSupplyChainDataset.csv"
)

df = pd.read_csv(
    file_path,
    encoding="latin1",
    low_memory=False
)

print("\n--- TARGET DISTRIBUTION ---")
print(df["Late_delivery_risk"].value_counts(dropna=False))
print("\nTarget percentages:")
print(
    (df["Late_delivery_risk"].value_counts(normalize=True) * 100)
    .round(2)
)

print("\n--- DELIVERY STATUS DISTRIBUTION ---")
print(df["Delivery Status"].value_counts(dropna=False))

print("\n--- TARGET VS DELIVERY STATUS ---")
print(pd.crosstab(
    df["Delivery Status"],
    df["Late_delivery_risk"],
    margins=True
))

print("\n--- DATE VALIDATION ---")
for column in [
    "order date (DateOrders)",
    "shipping date (DateOrders)"
]:
    parsed = pd.to_datetime(df[column], errors="coerce")
    print(f"\n{column}")
    print("Invalid or missing dates:", parsed.isna().sum())
    print("Earliest:", parsed.min())
    print("Latest:", parsed.max())

print("\n--- DUPLICATE CHECK ---")
print("Fully duplicated rows:", df.duplicated().sum())
print(
    "Duplicate Order Item Id values:",
    df["Order Item Id"].duplicated().sum()
)

print("\n--- POSSIBLE TARGET LEAKAGE CHECK ---")
print(
    "Real shipping days vs risk label:"
)
print(pd.crosstab(
    df["Days for shipping (real)"],
    df["Late_delivery_risk"],
    normalize="index"
).round(3))

print("\n--- AUDIT COMPLETE ---")
