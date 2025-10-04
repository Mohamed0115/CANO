# merge_catalogs.py
import pandas as pd

files = [
    "data/processed/kepler.csv",
    "data/processed/k2.csv",
    "data/processed/tess.csv"
]

dfs = [pd.read_csv(f) for f in files]
merged = pd.concat(dfs, ignore_index=True)

merged.to_csv("data/processed/all_catalogs.csv", index=False)
print("✅ Merged dataset saved: data/processed/all_catalogs.csv")
print("Shape:", merged.shape)
print("Label distribution:", merged['label'].value_counts().to_dict())
