# preprocess_k2.py
import pandas as pd, os
import numpy as np

os.makedirs("data/processed", exist_ok=True)
INPUT = "data/raw/k2_table.csv"
OUTPUT = "data/processed/k2.csv"

df = pd.read_csv(INPUT, comment="#")

rename_map = {
    "tic_id": "star_id",
    "pl_orbper": "orbital_period",
    "pl_trandurh": "transit_duration",
    "pl_trandep": "transit_depth",
    "pl_rade": "planet_radius",
    "st_rad": "star_radius",
    "st_teff": "star_temp",
    "st_logg": "star_logg"
}
df = df.rename(columns=rename_map)

# Encode labels
y = df["disposition"].str.upper()
df["label"] = y.apply(lambda v: 1 if v in ["CONFIRMED", "CANDIDATE"] else 0)

# Define schema
keep = ["star_id","orbital_period","transit_duration","transit_depth",
        "planet_radius","star_radius","star_temp","star_logg","label"]

# Add any missing columns as NaN
for col in keep:
    if col not in df.columns:
        df[col] = np.nan

df[keep].to_csv(OUTPUT, index=False)
print(f"✅ K2 saved to {OUTPUT}, shape {df.shape}")
