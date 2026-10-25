
from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression


# Project paths
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

TRAIN_FILE = DATA_DIR / "dataco_train.csv"
MODEL_FILE = MODEL_DIR / "dataco_logistic_pipeline.joblib"
CONFIG_FILE = MODEL_DIR / "dataco_model_config.json"


# Target and features
TARGET = "Late_delivery_risk"

CATEGORICAL_FEATURES = [
    "Type",
    "Category Name",
    "Customer Segment",
    "Department Name",
    "Market",
    "Order Country",
    "Order Region",
    "Shipping Mode",
]

NUMERIC_FEATURES = [
    "Days for shipment (scheduled)",
    "Order Item Quantity",
    "Order Item Product Price",
    "Order Item Discount Rate",
    "order_year",
    "order_month",
    "order_day_of_week",
]

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Load training data only
    train_df = pd.read_csv(TRAIN_FILE)

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    # Preprocess categorical and numerical features
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer([
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
    ])

    # Full preprocessing + model pipeline
    model_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])

    print("Training model on dataco_train.csv...")
    model_pipeline.fit(X_train, y_train)

    # Save the fitted pipeline
    joblib.dump(model_pipeline, MODEL_FILE)

    # Store the selected operating threshold separately.
    # 0.30 favors recall; we can revise this after business-cost analysis.
    config = {
        "model_name": "DataCo Logistic Regression",
        "target": TARGET,
        "threshold": 0.30,
        "features": FEATURES,
        "positive_class": 1,
        "note": (
            "Threshold 0.30 is provisional and favors detecting more "
            "late-delivery-risk cases at the cost of more false alarms."
        ),
    }

    CONFIG_FILE.write_text(json.dumps(config, indent=4), encoding="utf-8")

    print(f"Saved model: {MODEL_FILE}")
    print(f"Saved config: {CONFIG_FILE}")
    print(f"Training rows: {len(train_df):,}")
    print("Model pipeline saved successfully.")


if __name__ == "__main__":
    main()
