#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
modelling.py — CatBoost classification + MLflow
- Load preprocessed train/test (hasil automate_Nama-siswa.py)
- Split train -> train/val
- Train CatBoost (MultiClass)
- Log metrics & artifacts ke MLflow
- Inference ke test (tanpa label) -> submission.csv
"""

import os
import numpy as np
import pandas as pd
import mlflow
import matplotlib.pyplot as plt

from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix
)

# ================== 0. Path & Setup ==================
# Sesuaikan kalau struktur folder berbeda
TRAIN_PATH = "../data_preprocessed/preprocessed_train.csv"
TEST_PATH  = "../data_preprocessed/preprocessed_test.csv"
TARGET_COL = "Credit_Score"
MLFLOW_URI = "http://127.0.0.1:5000/"
EXPERIMENT_NAME = "Credit_Scoring_Classification"

# ================== 1. Load Preprocessed Data ==================
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

# X, y dari train
X = train_df.drop(columns=[TARGET_COL])
y = train_df[TARGET_COL].astype(str)

# ================== 2. Train/Val Split ==================
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42, shuffle=True
)

# ================== 3. Siapkan kolom kategori untuk CatBoost ==================
# Treat semua kolom non-numerik sebagai kategori, kecuali kolom text khusus
cat_columns = X_tr.select_dtypes(exclude="number").columns.drop(["Type_of_Loan"], errors="ignore").tolist()
text_column = ["Type_of_Loan"] if "Type_of_Loan" in X_tr.columns else []

feature_names = X_tr.columns.tolist()
cat_features_idx  = [feature_names.index(c) for c in cat_columns]
text_features_idx = [feature_names.index(c) for c in text_column] if text_column else []

# Cast kategori/text ke string (aman & konsisten)
for c in cat_columns + text_column:
    X_tr[c]  = X_tr[c].astype(str)
    X_val[c] = X_val[c].astype(str)

# Pool untuk training & validation (punya label)
train_pool = Pool(
    X_tr, label=y_tr,
    cat_features=cat_features_idx,
    text_features=text_features_idx or None
)
valid_pool = Pool(
    X_val, label=y_val,
    cat_features=cat_features_idx,
    text_features=text_features_idx or None
)

# Pool untuk inference (tanpa label)
X_test_infer = test_df.drop(columns=[TARGET_COL], errors="ignore").copy()
for c in cat_columns + text_column:
    if c in X_test_infer.columns:
        X_test_infer[c] = X_test_infer[c].astype(str)
test_pool = Pool(
    X_test_infer,
    cat_features=cat_features_idx,
    text_features=text_features_idx or None
)

# ================== 4. MLflow Setup ==================
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# Autolog (universal, cocok untuk MLflow >= 2.5)
mlflow.autolog(log_models=True)

params = {
    "iterations": 800,
    "learning_rate": 0.05,
    "depth": 8,
    "loss_function": "MultiClass",
    "eval_metric": "Accuracy",
    "random_seed": 42,
    "verbose": 100,
    "od_type": "Iter",
    "od_wait": 50
}

# ================== 5. Train & Log ==================
with mlflow.start_run(run_name="CatBoost_CreditScore") as run:
    run_id = run.info.run_id
    print(f"🏃 View run at: {MLFLOW_URI}#/experiments/{mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id}/runs/{run_id}")

    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

    # ===== Validation metrics =====
    y_val_pred = model.predict(X_val).ravel().astype(str)
    acc  = accuracy_score(y_val, y_val_pred)
    f1w  = f1_score(y_val, y_val_pred, average="weighted")
    prec = precision_score(y_val, y_val_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_val, y_val_pred, average="weighted", zero_division=0)

    mlflow.log_metric("val_accuracy", acc)
    mlflow.log_metric("val_f1_weighted", f1w)
    mlflow.log_metric("val_precision_weighted", prec)
    mlflow.log_metric("val_recall_weighted", rec)

    print(f"\n✅ Model trained successfully!")
    print(f"Val Accuracy: {acc:.4f} | F1_w: {f1w:.4f} | Precision_w: {prec:.4f} | Recall_w: {rec:.4f}")

    # ===== Confusion matrix (validation) =====
    classes = np.unique(y)  # semua kelas dari y (train)
    cm = confusion_matrix(y_val, y_val_pred, labels=classes)

    fig = plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion Matrix (Validation)')
    plt.xticks(ticks=range(len(classes)), labels=classes, rotation=45, ha="right")
    plt.yticks(ticks=range(len(classes)), labels=classes)
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha='center', va='center')
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.tight_layout()
    cm_path = "confusion_matrix_val.png"
    fig.savefig(cm_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(cm_path)

    # ===== Feature importance =====
    imp = model.get_feature_importance(train_pool, type="FeatureImportance")
    fi = pd.DataFrame({"feature": feature_names, "importance": imp}).sort_values("importance", ascending=False)
    fi_path = "feature_importance.csv"
    fi.to_csv(fi_path, index=False)
    mlflow.log_artifact(fi_path)

    # ===== Simpan model eksplisit (autolog juga menyimpan) =====
    # NB: dengan mlflow.autolog, model sudah dilog otomatis; baris berikut opsional.
    # from mlflow.catboost import log_model
    # log_model(cb_model=model, artifact_path="model")

    # ================== 6. Inference pada test (tanpa label) ==================
    test_pred = model.predict(X_test_infer).ravel().astype(str)
    submission = test_df.copy()
    submission["Credit_Score_pred"] = test_pred

    # Sertakan ID kalau ada
    id_col = None
    for cand in ["Customer_ID", "ID", "id"]:
        if cand in submission.columns:
            id_col = cand
            break

    keep_cols = [id_col, "Credit_Score_pred"] if id_col else ["Credit_Score_pred"]
    submission_out = submission[keep_cols]
    sub_path = "submission.csv"
    submission_out.to_csv(sub_path, index=False)
    mlflow.log_artifact(sub_path)

print(f"\n📊 Check MLflow UI → {MLFLOW_URI}")