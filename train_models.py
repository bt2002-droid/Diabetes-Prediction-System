"""
train_models.py
----------------
Trains and evaluates 4 machine learning models on the Pima Indians Diabetes
Dataset, then saves the trained models, the scaler, and all evaluation
results so app.py can load them without retraining.

Run with:
    python train_models.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# --------------------------------------------------------------------------
# 1. PATHS
# --------------------------------------------------------------------------
DATA_PATH = "diabetes.csv"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURE_NAMES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
TARGET_NAME = "Outcome"

# Columns where a value of 0 is medically impossible / represents a missing
# reading, not a genuine measurement of zero.
ZERO_INVALID_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n\n'{path}' was not found.\n"
            f"Please download the Pima Indians Diabetes Dataset (diabetes.csv) "
            f"and place it in the project root folder:\n\n"
            f"    diabetes-prediction/diabetes.csv\n\n"
            f"It must contain exactly these columns:\n"
            f"    {', '.join(FEATURE_NAMES + [TARGET_NAME])}\n"
        )
    df = pd.read_csv(path)

    missing_cols = set(FEATURE_NAMES + [TARGET_NAME]) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset is missing expected columns: {missing_cols}")

    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Duplicate check ---
    n_dupes = df.duplicated().sum()
    if n_dupes > 0:
        print(f"Found {n_dupes} duplicate rows -> removing them.")
        df = df.drop_duplicates()
    else:
        print("No duplicate rows found.")

    # --- Missing value check (NaNs) ---
    n_missing = df.isnull().sum().sum()
    print(f"Missing (NaN) values in dataset: {n_missing}")
    if n_missing > 0:
        df = df.dropna()
        print("Rows with NaN values dropped.")

    # --- Invalid zero handling ---
    # In this dataset, 0 in Glucose / BloodPressure / SkinThickness / Insulin
    # / BMI is not a real measurement, it means the value was not recorded.
    # We replace these zeros with the column median (computed AFTER removing
    # duplicates/NaNs, and BEFORE the train/test split is only for reporting;
    # the real replacement value used for scaling is fit on the training set
    # only, inside main(), to avoid data leakage).
    print("\nInvalid (zero) value counts before cleaning:")
    for col in ZERO_INVALID_COLS:
        n_zero = (df[col] == 0).sum()
        print(f"  {col}: {n_zero} zero values")

    return df


def replace_invalid_zeros_with_train_median(X_train, X_test, cols):
    """Fit median on training data only, apply to both train and test."""
    X_train = X_train.copy()
    X_test = X_test.copy()

    for col in cols:
        train_col = X_train[col].replace(0, np.nan)
        median_value = train_col.median()

        X_train[col] = X_train[col].replace(0, median_value)
        X_test[col] = X_test[col].replace(0, median_value)

    return X_train, X_test


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
    }, confusion_matrix(y_test, y_pred)


def main():
    print("=" * 60)
    print("STEP 1: Loading dataset")
    print("=" * 60)
    df = load_dataset(DATA_PATH)
    print(f"Dataset shape: {df.shape}")

    print("\n" + "=" * 60)
    print("STEP 2: Cleaning dataset")
    print("=" * 60)
    df = preprocess(df)

    print("\n" + "=" * 60)
    print("STEP 3: Splitting features and target")
    print("=" * 60)
    X = df[FEATURE_NAMES]
    y = df[TARGET_NAME]
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"Class balance:\n{y.value_counts()}")

    print("\n" + "=" * 60)
    print("STEP 4: Train-test split (80/20, stratified)")
    print("=" * 60)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    print("\n" + "=" * 60)
    print("STEP 5: Handling invalid zero values (median imputation)")
    print("=" * 60)
    X_train, X_test = replace_invalid_zeros_with_train_median(
        X_train, X_test, ZERO_INVALID_COLS
    )
    print("Invalid zeros replaced using TRAINING SET medians (no data leakage).")

    print("\n" + "=" * 60)
    print("STEP 6: Feature scaling (StandardScaler)")
    print("=" * 60)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Scaler fit on training data only, then applied to both sets.")

    print("\n" + "=" * 60)
    print("STEP 7: Training 4 models")
    print("=" * 60)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
    }

    # Logistic Regression and SVM are distance/gradient based -> use scaled data.
    # Tree-based models don't need scaling, but using scaled data does not hurt
    # them (splits are based on relative order), so we use the same scaled
    # features everywhere for a single consistent preprocessing pipeline.
    results = {}
    confusion_matrices = {}
    trained_models = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_scaled, y_train)
        metrics, cm = evaluate_model(model, X_test_scaled, y_test)
        results[name] = metrics
        confusion_matrices[name] = cm
        trained_models[name] = model
        print(f"  Accuracy={metrics['Accuracy']:.4f}  Precision={metrics['Precision']:.4f}  "
              f"Recall={metrics['Recall']:.4f}  F1={metrics['F1 Score']:.4f}")

    print("\n" + "=" * 60)
    print("STEP 8: Model comparison")
    print("=" * 60)
    results_df = pd.DataFrame(results).T
    results_df.index.name = "Model"
    print(results_df.round(4))

    best_model_name = results_df["F1 Score"].idxmax()
    print(f"\nBest performing model (highest F1 Score): {best_model_name}")

    print("\n" + "=" * 60)
    print("STEP 9: Saving models and preprocessing objects")
    print("=" * 60)

    file_map = {
        "Logistic Regression": "logistic_regression.pkl",
        "Decision Tree": "decision_tree.pkl",
        "Random Forest": "random_forest.pkl",
        "SVM": "svm.pkl",
    }
    for name, model in trained_models.items():
        path = os.path.join(MODELS_DIR, file_map[name])
        joblib.dump(model, path)
        print(f"Saved {name} -> {path}")

    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(FEATURE_NAMES, os.path.join(MODELS_DIR, "feature_names.pkl"))
    print("Saved scaler.pkl and feature_names.pkl")

    print("\n" + "=" * 60)
    print("STEP 10: Saving evaluation results")
    print("=" * 60)

    results_df.to_csv(os.path.join(MODELS_DIR, "metrics.csv"))
    print("Saved metrics.csv")

    joblib.dump(confusion_matrices, os.path.join(MODELS_DIR, "confusion_matrices.pkl"))
    print("Saved confusion_matrices.pkl")

    with open(os.path.join(MODELS_DIR, "best_model.json"), "w") as f:
        json.dump({"best_model": best_model_name}, f)
    print(f"Saved best_model.json -> {best_model_name}")

    # Feature importance (tree-based models only)
    importance_data = {}
    if hasattr(trained_models["Random Forest"], "feature_importances_"):
        importance_data["Random Forest"] = trained_models["Random Forest"].feature_importances_.tolist()
    if hasattr(trained_models["Decision Tree"], "feature_importances_"):
        importance_data["Decision Tree"] = trained_models["Decision Tree"].feature_importances_.tolist()

    importance_df = pd.DataFrame(importance_data, index=FEATURE_NAMES)
    importance_df.to_csv(os.path.join(MODELS_DIR, "feature_importance.csv"))
    print("Saved feature_importance.csv")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"All models and results saved inside the '{MODELS_DIR}/' folder.")
    print("You can now run:  streamlit run app.py")


if __name__ == "__main__":
    main()
