# predict.py
import pandas as pd
import joblib
import argparse
import os

MODEL_PATH = "models/lightgbm_model.pkl"

def load_model(model_path=MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. "
                                "Run train_classifier.py first.")
    model = joblib.load(model_path)
    return model

def predict_file(input_csv, output_csv=None):
    model = load_model()
    X = pd.read_csv(input_csv)

    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]  # probability of being planet

    results = X.copy()
    results["prediction"] = preds
    results["probability"] = probs

    print("\n--- Predictions ---")
    print(results.head())

    if output_csv:
        results.to_csv(output_csv, index=False)
        print(f"✅ Results saved to {output_csv}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict exoplanet candidates using trained LightGBM model")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input CSV file (features)")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional path to save predictions CSV")
    args = parser.parse_args()

    predict_file(args.input, args.output)
