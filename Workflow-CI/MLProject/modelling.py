#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
modelling.py — CatBoost classification + MLflow (final attach-safe version)
"""

import os
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import mlflow
import matplotlib.pyplot as plt

from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
)

# ================== 0. CLI Args ==================
parser = argparse.ArgumentParser()
parser.add_argument("--train_path", type=str, required=True)
parser.add_argument("--test_path", type=str, required=True)
args = parser.parse_args()

BASE = Path(__file__).resolve().parent
TRAIN_PATH = (BASE / args.train_path).resolve() if not Path(args.train_path).is_absolute() else Path(args.train_path)
TEST_PATH  = (BASE / args.test_path ).resolve() if not Path(args.test_path ).is_absolute() else Path(args.test_path)
TARGET_COL = "Credit_Score"
EXPERIMENT_NAME = "Credit_Scoring_Classification"

print(f"[INFO] Current Working Dir : {Path.cwd()}")
print(f"[INFO] Train Path Resolved : {TRAIN_PATH}")
print(f"[INFO] Test Path Resolved  : {TEST_PATH}")

if not TRAIN_PATH.exists():
    raise FileNotFoundError(f"Train file not found: {TRAIN_PATH}")
if not TEST_PATH.exists():
    raise FileNotFoundError(f"Test file not found: {TEST_PATH}")

# ================== 1. Load Data ==================
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

if train_df[TARGET_COL].dtype != "object":
    train_df[TARGET_COL] = train_df[TARGET_COL].astype(str)

label_order = sorted(train_df[TARGET_COL].unique().tolist())
label_to_id = {lbl: i for i, lbl in enumerate(label_order)}
print(f"[INFO] Label mapping applied: {label_to_id}")

y_all = train_df[TARGET_COL].map(label_to_id)
X_all = train_df.drop(columns=[TARGET_COL])

X_tr, X_val, y_tr, y_val = train_test_split(
    X_all, y_all, test_size=0.2, stratify=y_all, random_state=42, shuffle=True
)

cat_columns = X_tr.select_dtypes(include=["object", "category"]).columns.tolist()
print(f"[INFO] Detected categorical columns: {cat_columns}")

for c in cat_columns:
    X_tr[c]  = X_tr[c].astype(str)
    X_val[c] = X_val[c].astype(str)

feature_names = X_tr.columns.tolist()
cat_idx = [feature_names.index(c) for c in cat_columns]

train_pool = Pool(X_tr, label=y_tr, cat_features=cat_idx)
valid_pool = Pool(X_val, label=y_val, cat_features=cat_idx)

X_test_infer = test_df.drop(columns=[TARGET_COL], errors="ignore").copy()
for c in cat_columns:
    if c in X_test_infer.columns:
        X_test_infer[c] = X_test_infer[c].astype(str)
test_pool = Pool(X_test_infer, cat_features=cat_idx)

# ================== 2. MLflow Setup ==================
# ================== 2. MLflow Setup ==================
mlruns_root = (BASE.parent / "mlruns").as_posix()
mlflow.set_tracking_uri(f"file://{mlruns_root}")

env_run_id = os.getenv("MLFLOW_RUN_ID")

if env_run_id:
    print(f"[INFO] Attached to MLflow Project run: {env_run_id}")
    # DO NOT call set_experiment() or start_run() — run is already active
else:
    print("[INFO] Running standalone — setting experiment and starting run.")
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.start_run(run_name="Manual_CreditScore")

mlflow.sklearn.autolog(disable=False, log_models=False)

# ================== 3. Attach-safe MLflow Context ==================
env_run_id = os.getenv("MLFLOW_RUN_ID")

if env_run_id:
    print(f"[INFO] Attached to MLflow Project run: {env_run_id} (already active by MLflow)")
else:
    print("[INFO] Running standalone (no MLFLOW_RUN_ID set) — manual/local mode.")
    # Uncomment jika ingin logging saat menjalankan manual:
    # mlflow.start_run(run_name="Manual_CreditScore")

# ================== 4. Training & Logging ==================
params = {
    "iterations": 400,
    "depth": 6,
    "learning_rate": 0.1,
    "loss_function": "MultiClass",
    "eval_metric": "Accuracy",
    "random_seed": 42,
    "verbose": 100,
    "od_type": "Iter",
    "od_wait": 50
}

mlflow.log_params(params)
mlflow.log_param("cat_features", ",".join(cat_columns))
mlflow.log_param("label_mapping", str(label_to_id))

print("\n🏃 Training CatBoost…")
model = CatBoostClassifier(**params)
model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

# ===== Validation metrics =====
id_to_label = {v: k for k, v in label_to_id.items()}
y_val_pred_ids = model.predict(X_val).ravel().astype(int)
y_val_true_lbl = y_val.map(id_to_label)
y_val_pred_lbl = pd.Series(y_val_pred_ids).map(id_to_label)

acc  = accuracy_score(y_val_true_lbl, y_val_pred_lbl)
f1w  = f1_score(y_val_true_lbl, y_val_pred_lbl, average="weighted")
prec = precision_score(y_val_true_lbl, y_val_pred_lbl, average="weighted", zero_division=0)
rec  = recall_score(y_val_true_lbl, y_val_pred_lbl, average="weighted", zero_division=0)

mlflow.log_metric("val_accuracy", acc)
mlflow.log_metric("val_f1_weighted", f1w)
mlflow.log_metric("val_precision_weighted", prec)
mlflow.log_metric("val_recall_weighted", rec)

print(f"✅ Val Accuracy: {acc:.4f} | F1_w: {f1w:.4f} | Precision_w: {prec:.4f} | Recall_w: {rec:.4f}")

# ===== Confusion Matrix =====
classes = [id_to_label[i] for i in range(len(label_to_id))]
cm = confusion_matrix(y_val_true_lbl, y_val_pred_lbl, labels=classes)

fig = plt.figure(figsize=(6,5))
plt.imshow(cm, interpolation='nearest')
plt.title('Confusion Matrix (Validation)')
plt.xticks(ticks=range(len(classes)), labels=classes, rotation=45, ha="right")
plt.yticks(ticks=range(len(classes)), labels=classes)
for (i, j), v in np.ndenumerate(cm):
    plt.text(j, i, str(v), ha='center', va='center')
plt.xlabel('Predicted'); plt.ylabel('True'); plt.tight_layout()
cm_path = (BASE / "confusion_matrix_val.png")
fig.savefig(cm_path, dpi=140, bbox_inches="tight")
plt.close(fig)
mlflow.log_artifact(str(cm_path))

# ===== Feature Importance =====
imp = model.get_feature_importance(train_pool, type="FeatureImportance")
fi = pd.DataFrame({"feature": feature_names, "importance": imp}).sort_values("importance", ascending=False)
fi_path = (BASE / "feature_importance.csv")
fi.to_csv(fi_path, index=False)
mlflow.log_artifact(str(fi_path))

# ===== Save Model =====
mlflow.catboost.log_model(
    cb_model=model,
    artifact_path="model",
    registered_model_name=None
)

# ===== Inference =====
test_pred_ids = model.predict(X_test_infer).ravel().astype(int)
test_pred_lbl = pd.Series(test_pred_ids).map(id_to_label)
sub = test_df.copy()
sub["Credit_Score_pred"] = test_pred_lbl
for cand in ["Customer_ID", "ID", "id"]:
    if cand in sub.columns:
        sub = sub[[cand, "Credit_Score_pred"]]
        break
sub_path = (BASE / "submission.csv")
sub.to_csv(sub_path, index=False)
mlflow.log_artifact(str(sub_path))

print("\n[INFO] Artifacts logged: confusion_matrix_val.png, feature_importance.csv, submission.csv")
print("[INFO] ✅ Model logged successfully")