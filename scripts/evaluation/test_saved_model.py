
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

MODEL_FILE = MODEL_DIR / "dataco_logistic_pipeline.joblib"
CONFIG_FILE = MODEL_DIR / "dataco_model_config.json"
TEST_FILE = DATA_DIR / "dataco_test.csv"


def main():
    # Load the saved model and its configuration
    model = joblib.load(MODEL_FILE)

    with open(CONFIG_FILE, "r", encoding="utf-8") as file:
        config = json.load(file)

    test_df = pd.read_csv(TEST_FILE)
    features = config["features"]
    target = config["target"]

    X_test = test_df[features]
    y_test = test_df[target]

    # Predict probabilities, then apply the configured threshold
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= config["threshold"]).astype(int)

    print("Saved model loaded successfully.")
    print(f"Test rows: {len(test_df):,}")
    print(f"Configured threshold: {config['threshold']}")
    print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    # Show example predictions
    examples = X_test.head(5).copy()
    examples["Actual risk"] = y_test.head(5).values
    examples["Predicted probability"] = probabilities[:5].round(4)
    examples["Predicted risk"] = predictions[:5]

    print("\nExample predictions:")
    print(examples.to_string(index=False))


if __name__ == "__main__":
    main()
