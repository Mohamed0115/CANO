# researcher.py
import streamlit as st
import pandas as pd
import joblib
import os
import shutil
import subprocess
import sys

repo_root = os.path.dirname(os.path.abspath(__file__))
default_dataset = os.path.join(repo_root, "data", "processed", "all_catalogs.csv")

def show_researcher():
    st.title("🔬 Researcher Dashboard")
    st.write("Upload datasets, retrain the model, and view statistics.")

    # --------------------------
    # Dataset Upload Section
    # --------------------------
    st.subheader("📂 Upload New Dataset then click \"Retrain below\"")
    uploaded_file = st.file_uploader("Upload a CSV file (catalog dataset)", type=["csv"])

    if uploaded_file:
        save_path = os.path.join("data", "processed", "uploaded_dataset.csv")
        os.makedirs("data/processed", exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"✅ File uploaded and saved as {save_path}")
        # Replace the default dataset with uploaded
        shutil.copy(save_path, "data/processed/all_catalogs.csv")

    # Fallback to default
    default_path = "data/processed/all_catalogs.csv"
    if os.path.exists(default_path):
        df = pd.read_csv(default_path)
        st.info(f"Using dataset: {default_path}, shape: {df.shape}")
        st.dataframe(df.head())
    else:
        st.error("❌ No dataset found. Please upload one.")
        return

    # --------------------------
    # Load trained model
    # --------------------------
    model_path = "models/lightgbm_model.pkl"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        st.success("✅ Trained LightGBM model loaded.")
    else:
        st.warning("⚠️ No trained model found yet. Retrain below.")

    # --------------------------
    # Evaluation Results
    # --------------------------
    st.subheader("📊 Model Evaluation Results")
    report_path = "evaluation_report.csv"
    if os.path.exists(report_path):
        report_df = pd.read_csv(report_path)
        st.dataframe(report_df)
        st.download_button(
            "⬇️ Download Report CSV",
            data=report_df.to_csv(index=False),
            file_name="evaluation_report.csv",
            mime="text/csv"
        )

    # Show plots if available
    plot_files = {
        "Confusion Matrix": "confusion_matrix_eval.png",
        "ROC Curve": "roc_curve.png",
        "Precision-Recall Curve": "precision_recall_curve.png",
        "SHAP Summary": "shap_summary.png"
    }

    cols = st.columns(2)
    i = 0
    for title, file in plot_files.items():
        if os.path.exists(file):
            with cols[i % 2]:
                st.markdown(f"**{title}**")
                st.image(file, use_container_width=True)
        i += 1

    # --------------------------
    # Retrain Model Section
    # --------------------------
    st.subheader("⚡ Retrain the Model with uploaded Dataset")
    if st.button("🔄 Run Training + Evaluation "):
        with st.spinner("Training and evaluating model... ⏳"):
            try:
                repo_root = os.path.dirname(os.path.abspath(__file__))
                train_script = os.path.join(repo_root, "train_classifier.py")
                eval_script = os.path.join(repo_root, "evaluate_model.py")

                # Step 1: Train
                result = subprocess.run(
                    [sys.executable, train_script],
                    capture_output=True, text=True, check=True
                )
                st.success("✅ Training completed")
                st.text_area("Training Log", result.stdout, height=200)

                # Step 2: Evaluate
                eval_result = subprocess.run(
                    [sys.executable, eval_script],
                    capture_output=True, text=True, check=True
                )
                st.success("📊 Evaluation complete")
                st.text_area("Evaluation Log", eval_result.stdout, height=200)

                # Step 3: Refresh plots
                st.subheader("📈 Updated Evaluation Plots")
                for pf in plot_files.values():
                    if os.path.exists(pf):
                        st.image(pf, caption=pf, use_column_width=True)

            except subprocess.CalledProcessError as e:
                st.error("❌ Training/Evaluation failed!")
                st.text_area("Error Log", e.stderr, height=300)

    # Back to Home Button
    if st.button("🏠 Back to Home"):
        st.session_state["mode"] = "home"
        st.rerun()
