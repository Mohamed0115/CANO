# tune_threshold.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib, json
from sklearn.model_selection import train_test_split

# ---- Load dataset ----
df = pd.read_csv("data/processed/all_catalogs.csv")
X = df.drop(columns=[c for c in df.columns if df[c].dtype == "object" or c == "label"])
y = df["label"]

# ---- Load trained model ----
model = joblib.load("models/lightgbm_model.pkl")

# ---- Train/Test Split ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---- Predict probabilities ----
y_proba = model.predict_proba(X_test)[:, 1]

thresholds = np.arange(0.0, 1.01, 0.01)
precisions, recalls, f1s = [], [], []

for t in thresholds:
    y_pred = (y_proba >= t).astype(int)
    precisions.append(precision_score(y_test, y_pred, zero_division=0))
    recalls.append(recall_score(y_test, y_pred))
    f1s.append(f1_score(y_test, y_pred))

# ---- Find best threshold by F1 ----
best_idx = np.argmax(f1s)
best_threshold = thresholds[best_idx]

print(f"✅ Best Threshold (by F1): {best_threshold:.2f}")
print(f"Precision={precisions[best_idx]:.3f}, Recall={recalls[best_idx]:.3f}, F1={f1s[best_idx]:.3f}")

# ---- Save results ----
results = {
    "best_threshold": float(best_threshold),
    "precision": float(precisions[best_idx]),
    "recall": float(recalls[best_idx]),
    "f1": float(f1s[best_idx])
}
with open("models/best_threshold.json", "w") as f:
    json.dump(results, f, indent=2)

# ---- Plot curves ----
plt.figure()
plt.plot(thresholds, precisions, label="Precision")
plt.plot(thresholds, recalls, label="Recall")
plt.plot(thresholds, f1s, label="F1-score")
plt.axvline(best_threshold, color="red", linestyle="--", label=f"Best={best_threshold:.2f}")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Threshold Tuning")
plt.legend()
plt.tight_layout()
plt.savefig("threshold_tuning.png")
plt.show()
