
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data" / "raw" / "DataCoSupplyChainDataset.csv"
OUT = ROOT / "data" / "processed"

df = pd.read_csv(SOURCE, encoding="latin1", low_memory=False)

date_col = "order date (DateOrders)"
order_col = "Order Id"
target = "Late_delivery_risk"

features = [
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

df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col, order_col, target]).copy()

# Derive features using only the order date.
df["order_year"] = df[date_col].dt.year
df["order_month"] = df[date_col].dt.month
df["order_day_of_week"] = df[date_col].dt.dayofweek
features += ["order_year", "order_month", "order_day_of_week"]

# Assign every order to exactly one chronological partition.
orders = (
    df.groupby(order_col)[date_col]
    .min()
    .sort_values()
)

n = len(orders)
train_end = int(n * 0.64)
valid_end = int(n * 0.80)

train_ids = set(orders.index[:train_end])
valid_ids = set(orders.index[train_end:valid_end])

train = df[df[order_col].isin(train_ids)].copy()
valid = df[df[order_col].isin(valid_ids)].copy()

X_train, y_train = train[features], train[target]
X_valid, y_valid = valid[features], valid[target]

cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ]), cat_cols),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000)),
])

model.fit(X_train, y_train)
probabilities = model.predict_proba(X_valid)[:, 1]

print("--- CHRONOLOGICAL VALIDATION ---")
print("Training orders:", len(train_ids))
print("Validation orders:", len(valid_ids))
print("Training date range:",
      train[date_col].min(), "to", train[date_col].max())
print("Validation date range:",
      valid[date_col].min(), "to", valid[date_col].max())
print("Shared orders:",
      len(train_ids.intersection(valid_ids)))
print("Validation ROC-AUC:", round(roc_auc_score(y_valid, probabilities), 4))

print("\n--- THRESHOLD COMPARISON ---")
print("Threshold | Precision | Recall | F1")

for threshold in [0.30, 0.40, 0.50, 0.60]:
    predictions = (probabilities >= threshold).astype(int)
    print(
        f"{threshold:9.2f} | "
        f"{precision_score(y_valid, predictions, zero_division=0):9.4f} | "
        f"{recall_score(y_valid, predictions, zero_division=0):6.4f} | "
        f"{f1_score(y_valid, predictions, zero_division=0):.4f}"
    )

print("\nValidation complete. Test set was not used.")
