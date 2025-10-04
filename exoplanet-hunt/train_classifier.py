# train_classifier.py
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# ---- Load merged dataset ----
DATA_PATH = "data/processed/all_catalogs.csv"
df = pd.read_csv(DATA_PATH)

print("Loaded dataset:", df.shape)
print("Label distribution:", df["label"].value_counts().to_dict())

# Drop ID columns or any non-numeric fields
drop_cols = [c for c in df.columns if df[c].dtype == "object" or c == "label"]
X = df.drop(columns=drop_cols)
y = df["label"]


# To check
print("Final features:", X.columns.tolist())
print("Dtypes:\n", X.dtypes.value_counts())


# ---- Train/test split ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---- Train LightGBM ----
model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=-1,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train, y_train)

# ---- Evaluate ----
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))

# ---- Feature Importances ----
importances = pd.Series(model.feature_importances_, index=X.columns)
plt.figure(figsize=(8, 5))
importances.sort_values().plot(kind="barh")
plt.title("Feature Importances (LightGBM)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importances.png")
plt.show()

# ---- Confusion Matrix ----
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Not Planet", "Planet"],
            yticklabels=["Not Planet", "Planet"])
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# ---- Save trained model ----
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/lightgbm_model.pkl")
print("✅ Model saved to models/lightgbm_model.pkl")
