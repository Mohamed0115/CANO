# preprocess_kepler.py
import pandas as pd, os

os.makedirs("data/processed", exist_ok=True)
INPUT = "data/raw/kepler_table.csv"
OUTPUT = "data/processed/kepler.csv"

df = pd.read_csv(INPUT, comment="#")

# Rename to unified schema
rename_map = {
    "kepid": "star_id",
    "koi_period": "orbital_period",
    "koi_prad": "planet_radius",
    "koi_srad": "star_radius",
    "koi_steff": "star_temp",
    "koi_slogg": "star_logg"
}
df = df.rename(columns=rename_map)

# Encode labels
y = df["koi_disposition"].str.upper()
df["label"] = y.apply(lambda v: 1 if v in ["CONFIRMED", "CANDIDATE"] else 0)

keep = ["star_id","orbital_period","planet_radius","star_radius",
        "star_temp","star_logg","label"]
df[keep].to_csv(OUTPUT, index=False)
print(f"✅ Kepler saved to {OUTPUT}, shape {df.shape}")
