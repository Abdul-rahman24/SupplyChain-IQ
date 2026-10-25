
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"

train = pd.read_csv(DATA / "dataco_train.csv")
test = pd.read_csv(DATA / "dataco_test.csv")

target = "Late_delivery_risk"
X_train = train.drop(columns=[target])
y_train = train[target]
X_test = test.drop(columns=[target])
y_test = test[target]

categorical = X_train.select_dtypes(
    include=["object"]
).columns.tolist()
numeric = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

preprocessor = ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), numeric),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000)),
])

# Fit only on the training period.
model.fit(X_train, y_train)
probabilities = model.predict_proba(X_test)[:, 1]

print("--- FINAL TEST ROC-AUC ---")
print(round(roc_auc_score(y_test, probabilities), 4))

for threshold in [0.30, 0.50]:
    predictions = (probabilities >= threshold).astype(int)

    print(f"\n--- TEST RESULTS: THRESHOLD {threshold:.2f} ---")
    print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
    print("Precision:", round(precision_score(y_test, predictions), 4))
    print("Recall:", round(recall_score(y_test, predictions), 4))
    print("F1-score:", round(f1_score(y_test, predictions), 4))
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification report:")
    print(classification_report(
        y_test, predictions, digits=4, zero_division=0
    ))

print("\nFinal evaluation complete.")
