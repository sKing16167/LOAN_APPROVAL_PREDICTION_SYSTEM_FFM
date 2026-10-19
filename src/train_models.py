import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

from data_processor import engineer_features, encode_categoricals, FEATURE_COLUMNS

def train_and_evaluate():
    data_path = os.path.join("data", "loan_approval_dataset.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
        
    print(f"Loading raw data from {data_path}...")
    raw_df = pd.read_csv(data_path)
    
    # Feature engineering and encoding
    df_feat = engineer_features(raw_df)
    df_clean = encode_categoricals(df_feat)
    
    X = df_clean[FEATURE_COLUMNS]
    y = df_clean["loan_status"]
    
    # Stratified Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")
    
    # Feature Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=120, max_depth=12, min_samples_split=4, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=120, learning_rate=0.08, max_depth=5, random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, min_samples_split=6, random_state=42
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=1.0, random_state=42
        )
    }
    
    os.makedirs("models", exist_ok=True)
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Logistic Regression uses scaled features; Tree models use unscaled/scaled identically but unscaled keeps feature interpretability
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        
        # Feature importances
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_.tolist()
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0]).tolist()
        else:
            importances = [0.0] * len(FEATURE_COLUMNS)
            
        print(f"Results for {name}:")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall   : {rec:.4f}")
        print(f"  F1 Score : {f1:.4f}")
        print(f"  ROC-AUC  : {auc:.4f}")
        
        results[name] = {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(float(auc), 4),
            "confusion_matrix": cm,
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "feature_importances": dict(zip(FEATURE_COLUMNS, [round(x, 4) for x in importances]))
        }
        
        # Save model
        model_filename = f"{name.lower().replace(' ', '_')}.joblib"
        joblib.dump(model, os.path.join("models", model_filename))
        
    # Save scaler and metadata
    joblib.dump(scaler, os.path.join("models", "scaler.joblib"))
    
    metadata = {
        "feature_columns": FEATURE_COLUMNS,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "target_classes": ["Rejected (0)", "Approved (1)"],
        "metrics_summary": results
    }
    
    with open(os.path.join("models", "evaluation_metrics.json"), "w") as f:
        json.dump(metadata, f, indent=4)
        
    print("\nTraining completed successfully! Saved all artifacts to /models.")
    return metadata

if __name__ == "__main__":
    train_and_evaluate()
