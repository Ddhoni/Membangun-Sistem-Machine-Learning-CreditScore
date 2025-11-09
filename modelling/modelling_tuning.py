import dagshub
import mlflow
import mlflow.catboost
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, cohen_kappa_score
)
from sklearn.model_selection import train_test_split

# ================== 1️⃣ LOAD DATA ==================
train_path = "../data_preprocessed/preprocessed_train.csv"
test_path  = "../data_preprocessed/preprocessed_test.csv"
TARGET_COL = "Credit_Score"

train_df = pd.read_csv(train_path)
test_df  = pd.read_csv(test_path)

# Drop kolom yang tidak relevan (optional, tergantung preprocessing)
X = train_df.drop(columns=[TARGET_COL])
y = train_df[TARGET_COL].astype(str)

# Split menjadi train/val agar validasi punya label
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42, shuffle=True
)

# ================== 2️⃣ DAGSHUB CONFIG ==================
# Pastikan kamu sudah punya token DagsHub dan sudah export:
# export MLFLOW_TRACKING_USERNAME=abiyamf
# export MLFLOW_TRACKING_PASSWORD=<your_personal_token>

dagshub.init(repo_owner="Ddhoni", repo_name="my-first-repo", mlflow=True)
mlflow.set_experiment("Credit_Scoring_Tuning_CatBoost")

# ================== 3️⃣ CATBOOST FEATURE SETUP ==================
cat_columns = X_train.select_dtypes(exclude="number").columns.drop(["Type_of_Loan"], errors="ignore").tolist()
text_column = ["Type_of_Loan"] if "Type_of_Loan" in X_train.columns else []

feature_names = X_train.columns.tolist()
cat_features_idx  = [feature_names.index(c) for c in cat_columns]
text_features_idx = [feature_names.index(c) for c in text_column] if text_column else []

for c in cat_columns + text_column:
    X_train[c] = X_train[c].astype(str)
    X_val[c]   = X_val[c].astype(str)

train_pool = Pool(X_train, label=y_train, cat_features=cat_features_idx, text_features=text_features_idx or None)
val_pool   = Pool(X_val,   label=y_val,   cat_features=cat_features_idx, text_features=text_features_idx or None)

# ================== 4️⃣ HYPERPARAMETER SEARCH ==================
learning_rates = [0.05, 0.1, 0.2]
depths = [6, 8, 10]
l2_leaf_regs = [1, 3, 5]

best_score = -np.inf
best_params = {}
best_model = None

# ================== 5️⃣ MANUAL LOGGING LOOP ==================
for lr in learning_rates:
    for depth in depths:
        for reg in l2_leaf_regs:
            with mlflow.start_run(run_name=f"catboost_lr{lr}_d{depth}_reg{reg}"):
                params = {
                    "iterations": 500,
                    "learning_rate": lr,
                    "depth": depth,
                    "l2_leaf_reg": reg,
                    "loss_function": "MultiClass",
                    "eval_metric": "Accuracy",
                    "random_seed": 42,
                    "verbose": 100,
                    "od_type": "Iter",
                    "od_wait": 50
                }

                model = CatBoostClassifier(**params)
                model.fit(train_pool, eval_set=val_pool, use_best_model=True, verbose=False)

                # === PREDIKSI ===
                y_pred = model.predict(X_val).ravel().astype(str)
                y_pred_proba = model.predict_proba(X_val)

                # === METRICS ===
                acc  = accuracy_score(y_val, y_pred)
                f1w  = f1_score(y_val, y_pred, average="weighted")
                prec = precision_score(y_val, y_pred, average="weighted", zero_division=0)
                rec  = recall_score(y_val, y_pred, average="weighted", zero_division=0)
                try:
                    roc  = roc_auc_score(pd.get_dummies(y_val), y_pred_proba, average="weighted", multi_class="ovr")
                except Exception:
                    roc = np.nan
                kappa = cohen_kappa_score(y_val, y_pred)

                # === MANUAL LOGGING ===
                mlflow.log_params(params)
                mlflow.log_metrics({
                    "accuracy": acc,
                    "f1_weighted": f1w,
                    "precision_weighted": prec,
                    "recall_weighted": rec,
                    "roc_auc_weighted": roc,  # ➕ Extra metric 1
                    "cohen_kappa": kappa      # ➕ Extra metric 2
                })

                # Simpan model terbaik (berdasarkan ROC-AUC)
                if roc > best_score:
                    best_score = roc
                    best_params = params
                    best_model = model

                    mlflow.catboost.log_model(
                        cb_model=model,
                        artifact_path="best_model",
                        input_example=X_val.iloc[:1]
                    )

print("\n✅ Hyperparameter tuning selesai!")
print(f"🏆 Best ROC-AUC: {best_score:.4f}")
print(f"📊 Best Params: {best_params}")
print(f"🔗 Lihat hasil di DagsHub: https://dagshub.com/abiyamf/my-first-repo/experiments")