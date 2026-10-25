
from pathlib import Path
import pandas as pd

# Locate the DataCo CSV relative to this script
project_dir = Path(__file__).resolve().parents[2]
file_path = (
    project_dir
    / "data"
    / "raw"
    / "DataCoSupplyChainDataset.csv"
)

print("Looking for:", file_path)

if not file_path.exists():
    print("File not found. Check the filename and folder.")
else:
    df = pd.read_csv(
        file_path,
        encoding="latin1",
        low_memory=False
    )

    print("\n--- DATASET OVERVIEW ---")
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    print("\n--- COLUMN NAMES ---")
    print(df.columns.tolist())

    print("\n--- FIRST 5 ROWS ---")
    print(df.head().to_string())

    print("\n--- MISSING VALUES (TOP 15) ---")
    print(df.isnull().sum().sort_values(
        ascending=False
    ).head(15))

    print("\n--- DATA TYPES ---")
    print(df.dtypes.to_string())
