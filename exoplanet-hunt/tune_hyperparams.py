# tune_hyperparams.py
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

# ---- Load merged dataset ----
DATA_PATH = "data/processed/all_catalogs.csv"
df = pd.read_csv(DATA_PATH)
print(f"Loaded dataset: {df.shape} Label dist: {df['label'].value_counts().to_dict()}")

# Drop ID columns or any non-numeric fields
drop_cols = [c for c in df.columns if df[c].dtype == "object" or c == "label"]
X = df.drop(columns=drop_cols)
y = df["label"]

print("Final features:", X.columns.tolist())
print("Dtypes:\n", X.dtypes.value_counts())

# ---- Train/test split ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---- Define base model ----
model = lgb.LGBMClassifier(
    random_state=42,
    class_weight="balanced",
    n_estimators=2000,   # allow early stopping to cut this down
)

# ---- Smaller param grid ----
param_grid = {
    "learning_rate": [0.01, 0.05],
    "max_depth": [-1, 10],
    "num_leaves": [31, 63],
    "min_child_samples": [10, 20],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

# ---- Randomized Search ----
search = RandomizedSearchCV(
    model,
    param_distributions=param_grid,
    n_iter=40,         # fewer but smarter tries
    scoring="roc_auc",
    cv=3,
    n_jobs=-1,
    verbose=2,
    random_state=42
)

print("🚀 Starting randomized search with early stopping...")

# Important: pass eval_set + early stopping to .fit()
search.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],   # use held-out test set for stopping
    eval_metric="auc",
    early_stopping_rounds=50,
    verbose=False
)

print("\n✅ Best parameters:", search.best_params_)
print("✅ Best ROC AUC (CV):", search.best_score_)

# ---- Evaluate best model on test set ----
best_model = search.best_estimator_
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print("\n--- Final Test Set Evaluation ---")
print(classification_report(y_test, y_pred))
print("Test ROC AUC:", roc_auc_score(y_test, y_proba))

# ---- Save tuned model ----
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/lightgbm_best.pkl")
print("✅ Tuned model saved to models/lightgbm_best.pkl")

import json

# ---- Save tuned model ----
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/lightgbm_best.pkl")
print("✅ Tuned model saved to models/lightgbm_best.pkl")

# ---- Save best params separately ----
best_params_path = "models/best_params.json"
with open(best_params_path, "w") as f:
    json.dump(search.best_params_, f, indent=4)

print(f"✅ Best parameters saved to {best_params_path}")
