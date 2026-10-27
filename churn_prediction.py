"""
Customer Churn Prediction - Complete End-to-End Project
Author: Folasunkanmi
Repository: https://github.com/folasunkanmi/AI-machine-learning
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix, RocCurveDisplay
)
from sklearn.pipeline import Pipeline
import joblib

# Set style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (10, 6)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def generate_synthetic_churn_data(n_samples: int = 5000) -> pd.DataFrame:
    """Generate a realistic synthetic customer churn dataset."""
    print("Generating synthetic customer churn dataset...")

    tenure = np.random.randint(1, 73, n_samples)
    monthly_charges = np.round(np.random.uniform(20, 120, n_samples), 2)

    # Total charges roughly related to tenure * monthly + noise
    total_charges = np.round(tenure * monthly_charges * np.random.uniform(0.85, 1.15, n_samples), 2)

    contract = np.random.choice(["Month-to-month", "One year", "Two year"], n_samples, p=[0.55, 0.25, 0.20])
    payment_method = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        n_samples, p=[0.35, 0.20, 0.25, 0.20]
    )
    internet_service = np.random.choice(["DSL", "Fiber optic", "No"], n_samples, p=[0.35, 0.45, 0.20])
    online_security = np.random.choice(["Yes", "No"], n_samples, p=[0.35, 0.65])
    tech_support = np.random.choice(["Yes", "No"], n_samples, p=[0.30, 0.70])
    senior_citizen = np.random.choice([0, 1], n_samples, p=[0.84, 0.16])
    partner = np.random.choice(["Yes", "No"], n_samples, p=[0.48, 0.52])
    dependents = np.random.choice(["Yes", "No"], n_samples, p=[0.30, 0.70])

    # Create churn probability based on realistic rules
    churn_prob = np.zeros(n_samples)

    # Higher churn for month-to-month
    churn_prob += np.where(contract == "Month-to-month", 0.35, 0.0)
    churn_prob += np.where(contract == "One year", 0.12, 0.0)
    churn_prob += np.where(contract == "Two year", 0.05, 0.0)

    # Higher for fiber + high charges
    churn_prob += np.where(internet_service == "Fiber optic", 0.15, 0.0)
    churn_prob += np.where(monthly_charges > 80, 0.12, 0.0)

    # Lower when they have security / support
    churn_prob -= np.where(online_security == "Yes", 0.10, 0.0)
    churn_prob -= np.where(tech_support == "Yes", 0.08, 0.0)

    # Tenure effect (newer customers more likely to churn)
    churn_prob += np.where(tenure < 12, 0.20, 0.0)
    churn_prob += np.where(tenure > 48, -0.15, 0.0)

    # Senior citizens slightly higher
    churn_prob += np.where(senior_citizen == 1, 0.08, 0.0)

    # Clip and add noise
    churn_prob = np.clip(churn_prob + np.random.normal(0, 0.08, n_samples), 0.05, 0.85)
    churn = np.where(np.random.rand(n_samples) < churn_prob, "Yes", "No")

    df = pd.DataFrame({
        "tenure": tenure,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "contract_type": contract,
        "payment_method": payment_method,
        "internet_service": internet_service,
        "online_security": online_security,
        "tech_support": tech_support,
        "senior_citizen": senior_citizen,
        "partner": partner,
        "dependents": dependents,
        "churn": churn
    })

    print(f"Dataset created with {len(df)} rows and {df.shape[1]} columns.")
    print(f"Churn rate: {(df['churn'] == 'Yes').mean():.2%}")
    return df


def perform_eda(df: pd.DataFrame):
    """Basic Exploratory Data Analysis."""
    print("\n" + "="*60)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*60)

    print("\n--- Dataset Info ---")
    print(df.info())

    print("\n--- First 5 rows ---")
    print(df.head())

    print("\n--- Descriptive Statistics (Numerical) ---")
    print(df.describe().round(2))

    print("\n--- Churn Distribution ---")
    print(df["churn"].value_counts(normalize=True).round(3))

    print("\n--- Churn rate by Contract Type ---")
    print(df.groupby("contract_type")["churn"].apply(lambda x: (x == "Yes").mean()).round(3))

    print("\n--- Churn rate by Internet Service ---")
    print(df.groupby("internet_service")["churn"].apply(lambda x: (x == "Yes").mean()).round(3))

    # Optional: Save a couple of plots
    os.makedirs("plots", exist_ok=True)

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="churn", hue="churn", legend=False)
    plt.title("Churn Distribution")
    plt.savefig("plots/churn_distribution.png", dpi=120, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x="churn", y="monthly_charges")
    plt.title("Monthly Charges by Churn")
    plt.savefig("plots/monthly_charges_by_churn.png", dpi=120, bbox_inches="tight")
    plt.close()

    print("\nEDA plots saved to ./plots/")


def preprocess_data(df: pd.DataFrame):
    """Encode categoricals and split data."""
    print("\n" + "="*60)
    print("PREPROCESSING")
    print("="*60)

    df = df.copy()

    # Encode target
    le = LabelEncoder()
    df["churn"] = le.fit_transform(df["churn"])  # Yes=1, No=0

    # One-hot encode categoricals
    cat_cols = ["contract_type", "payment_method", "internet_service",
                "online_security", "tech_support", "partner", "dependents"]

    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    X = df.drop("churn", axis=1)
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set:     {X_test.shape[0]} samples")
    print(f"Number of features after encoding: {X_train.shape[1]}")

    return X_train, X_test, y_train, y_test, X.columns.tolist()


def train_and_evaluate(X_train, X_test, y_train, y_test, feature_names):
    """Train multiple models and select the best one."""
    print("\n" + "="*60)
    print("MODEL TRAINING & EVALUATION")
    print("="*60)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=10, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=RANDOM_STATE)
    }

    results = {}
    best_model = None
    best_score = 0
    best_name = ""

    for name, model in models.items():
        print(f"\nTraining {name}...")

        # Scale for Logistic Regression
        if name == "Logistic Regression":
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("model", model)
            ])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            y_proba = pipe.predict_proba(X_test)[:, 1]
            trained = pipe
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            trained = model

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        results[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": auc,
            "model": trained
        }

        print(f"  Accuracy : {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall   : {rec:.4f}")
        print(f"  F1-Score : {f1:.4f}")
        print(f"  ROC-AUC  : {auc:.4f}")

        # Select best by F1 (balanced for churn)
        if f1 > best_score:
            best_score = f1
            best_model = trained
            best_name = name

    print("\n" + "-"*60)
    print(f"Best model: {best_name} (F1 = {best_score:.4f})")
    print("-"*60)

    # Detailed report for best model
    print(f"\nClassification Report ({best_name}):")
    y_pred_best = best_model.predict(X_test)
    print(classification_report(y_test, y_pred_best, target_names=["No Churn", "Churn"]))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_best)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"])
    plt.title(f"Confusion Matrix - {best_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.savefig("plots/confusion_matrix.png", dpi=120, bbox_inches="tight")
    plt.close()

    # Feature importance (for tree models)
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(12)

        plt.figure(figsize=(10, 6))
        sns.barplot(x=feat_imp.values, y=feat_imp.index, palette="viridis")
        plt.title(f"Top Feature Importances - {best_name}")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig("plots/feature_importance.png", dpi=120, bbox_inches="tight")
        plt.close()
        print("\nTop Feature Importances:")
        print(feat_imp.round(4))

    return best_model, best_name, results


def save_model(model, name: str):
    """Save the best model."""
    os.makedirs("models", exist_ok=True)
    path = "models/best_churn_model.joblib"
    joblib.dump(model, path)
    print(f"\nBest model ({name}) saved to: {path}")


def main():
    print("="*60)
    print("  CUSTOMER CHURN PREDICTION PROJECT")
    print("  https://github.com/folasunkanmi/AI-machine-learning")
    print("="*60)

    # 1. Generate data
    df = generate_synthetic_churn_data(n_samples=5000)

    # 2. EDA
    perform_eda(df)

    # 3. Preprocess
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(df)

    # 4. Train & Evaluate
    best_model, best_name, results = train_and_evaluate(
        X_train, X_test, y_train, y_test, feature_names
    )

    # 5. Save model
    save_model(best_model, best_name)

    print("\n" + "="*60)
    print("Project completed successfully!")
    print("Check the 'plots/' folder for visualizations.")
    print("="*60)


if __name__ == "__main__":
    main()
