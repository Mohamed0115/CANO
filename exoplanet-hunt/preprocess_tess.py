# preprocess_tess.py
import pandas as pd, os

os.makedirs("data/processed", exist_ok=True)
INPUT = "data/raw/tess_table.csv"
OUTPUT = "data/processed/tess.csv"

df = pd.read_csv(INPUT, comment="#")

rename_map = {
    "tid": "star_id",
    "pl_orbper": "orbital_period",
    "pl_trandurh": "transit_duration",
    "pl_trandep": "transit_depth",
    "pl_rade": "planet_radius",
    "st_rad": "star_radius",
    "st_teff": "star_temp",
    "st_logg": "star_logg"
}
df = df.rename(columns=rename_map)

y = df["tfopwg_disp"].str.upper()
df["label"] = y.apply(lambda v: 1 if v in ["CP","PC","KP"] else 0)

keep = ["star_id","orbital_period","transit_duration","transit_depth",
        "planet_radius","star_radius","star_temp","star_logg","label"]
df[keep].to_csv(OUTPUT, index=False)
print(f"✅ TESS saved to {OUTPUT}, shape {df.shape}")
