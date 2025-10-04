# evaluate_model.py
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, f1_score
)
import matplotlib.pyplot as plt
import shap
import os
import json

# ---- Load dataset ----
DATA_PATH = "data/processed/all_catalogs.csv"
MODEL_PATH = "models/lightgbm_model.pkl"
df = pd.read_csv(DATA_PATH)
print("Loaded dataset:", df.shape)

# Drop ID and non-numeric fields
drop_cols = [c for c in df.columns if df[c].dtype == "object" or c == "label"]
X = df.drop(columns=drop_cols)
y = df["label"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Load trained model
model = joblib.load(MODEL_PATH)

# Predictions
y_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_proba > 0.5).astype(int)

# ---- Report ----
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))

# Save report
report = classification_report(y_test, y_pred, output_dict=True)
pd.DataFrame(report).to_csv("evaluation_report.csv")
print("✅ Saved evaluation_report.csv")

# ---- Confusion matrix, ROC, PR ----
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks([0,1], ["Not Planet", "Planet"])
plt.yticks([0,1], ["Not Planet", "Planet"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
plt.tight_layout()
plt.savefig("confusion_matrix_eval.png")
plt.close()
print("✅ Saved confusion_matrix_eval.png")

fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_proba):.3f}")
plt.plot([0,1],[0,1],"--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.close()
print("✅ Saved roc_curve.png")

prec, rec, thr = precision_recall_curve(y_test, y_proba)
plt.figure()
plt.plot(thr, prec[:-1], label="Precision")
plt.plot(thr, rec[:-1], label="Recall")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision-Recall vs Threshold")
plt.legend()
plt.tight_layout()
plt.savefig("precision_recall_curve.png")
plt.close()
print("✅ Saved precision_recall_curve.png")

# ---- Threshold search ----
thresholds = np.linspace(0, 1, 101)
f1_scores = []
recalls = []

for t in thresholds:
    preds = (y_proba > t).astype(int)
    f1_scores.append(f1_score(y_test, preds))
    recalls.append((y_test & preds).sum() / y_test.sum())

best_f1_idx = np.argmax(f1_scores)
best_f1_threshold = float(thresholds[best_f1_idx])

# First threshold where recall >= 0.9
high_recall_threshold = float(
    thresholds[np.where(np.array(recalls) >= 0.9)[0][0]]
) if np.any(np.array(recalls) >= 0.9) else 0.5

print(f"Best F1 threshold: {best_f1_threshold:.3f}")
print(f"High recall (>=0.9) threshold: {high_recall_threshold:.3f}")

os.makedirs("models", exist_ok=True)
with open("models/thresholds.json", "w") as f:
    json.dump({
        "best_f1_threshold": best_f1_threshold,
        "high_recall_threshold": high_recall_threshold
    }, f, indent=2)

print("✅ Saved thresholds.json")

# ---- SHAP values ----
print("Calculating SHAP values...")
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig("shap_summary.png")
plt.close()
print("✅ Saved shap_summary.png")

# ---- Plot F1 vs Threshold (with Precision & Recall) ----
from sklearn.metrics import precision_score, recall_score, f1_score

# Sweep thresholds
thresholds = np.linspace(0, 1, 101)
precisions, recalls, f1s, tags = [], [], [], []
for t in thresholds:
    preds = (y_proba >= t).astype(int)
    p = precision_score(y_test, preds, zero_division=0)
    r = recall_score(y_test, preds, zero_division=0)
    f = f1_score(y_test, preds, zero_division=0)

    precisions.append(p)
    recalls.append(r)
    f1s.append(f)

    # --- Tagging important thresholds ---
    if abs(t - best_f1_threshold) < 1e-6:
        tags.append("best_f1")
    elif abs(t - high_recall_threshold) < 1e-6:
        tags.append("high_recall")
    else:
        tags.append("")

# Save full threshold curve with tags
thr_df = pd.DataFrame({
    "threshold": thresholds,
    "precision": precisions,
    "recall": recalls,
    "f1": f1s,
    "tag": tags
})
thr_df.to_csv("threshold_curve.csv", index=False)
print("✅ Saved threshold_curve.csv with strategy tags")

# --- Plot Precision-Recall-F1 vs Threshold ---
plt.figure(figsize=(8, 6))
plt.plot(thresholds, precisions, label="Precision", linestyle="--")
plt.plot(thresholds, recalls, label="Recall", linestyle="-")
plt.plot(thresholds, f1s, label="F1-score", linestyle="-.")

plt.axvline(best_f1_threshold, color="red", linestyle=":", label=f"Best F1 ({best_f1_threshold:.2f})")
plt.axvline(high_recall_threshold, color="green", linestyle=":", label=f"High Recall ({high_recall_threshold:.2f})")

# Highlight dots
plt.scatter(best_f1_threshold,
            f1_score(y_test, (y_proba >= best_f1_threshold).astype(int), zero_division=0),
            color="red", s=80, zorder=5)
plt.scatter(high_recall_threshold,
            recall_score(y_test, (y_proba >= high_recall_threshold).astype(int), zero_division=0),
            color="green", s=80, zorder=5)

plt.title("Precision, Recall & F1 vs Threshold")
plt.xlabel("Decision Threshold")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("threshold_analysis.png")
plt.show()
print("✅ Saved threshold_analysis.png")


