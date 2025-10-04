# preprocess_catalog.py
import pandas as pd
import os

os.makedirs("data/processed", exist_ok=True)

# Replace this with the actual filename you downloaded
INPUT_FILE = "data/raw/tess_table.csv"
OUTPUT_FEATURES = "data/processed/features.csv"
OUTPUT_LABELS = "data/processed/labels.csv"

# Load the catalog (CSV from NASA Exoplanet Archive)
df = pd.read_csv(INPUT_FILE, comment="#")


print("Original columns:", df.columns.tolist()[:20], "...")  # show first 20 columns

# ---- STEP 1: Pick useful numeric features ----
# (Adjust column names based on dataset — Kepler/K2/TESS names differ a bit)
possible_features = [
    "pl_orbper", "pl_trandurh", "pl_trandep", "pl_rade",
    "st_rad", "st_teff", "st_logg"
]

# Keep only the features that actually exist in the file
features = [c for c in possible_features if c in df.columns]
print("Using features:", features)

X = df[features].copy()

# ---- STEP 2: Handle missing values ----
X = X.fillna(X.median())

# ---- STEP 3: Encode labels ----
# Depending on dataset, disposition column name differs
label_col_candidates = [
    "Disposition Using Kepler Data",
    "Archive Disposition",
    "TFOPWG Disposition",
    "tfopwg_disp"
]


label_col = None
for c in label_col_candidates:
    if c in df.columns:
        label_col = c
        break

if label_col is None:
    raise ValueError("No disposition column found. Please check your CSV headers.")

print("Using label column:", label_col)

# Map labels to binary (1 = planet, 0 = not planet)

y = df[label_col].astype(str).str.upper()
# y = y.str.lower()


planet_labels = ["CP", "PC", "KP"]
y_bin = y.apply(lambda v: 1 if v in planet_labels else 0)


# ---- STEP 4: Save processed files ----
X.to_csv(OUTPUT_FEATURES, index=False)
y_bin.to_csv(OUTPUT_LABELS, index=False, header=["label"])

print("Saved features to", OUTPUT_FEATURES)
print("Saved labels to", OUTPUT_LABELS)
print("Shape:", X.shape, "Labels:", y_bin.value_counts().to_dict())
