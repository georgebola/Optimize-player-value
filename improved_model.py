"""
Improved Player Salary Estimation Pipeline
Improvements over baseline Random Forest:
  1. Log-transform salary target (fixes right skew)
  2. Feature engineering: per-90 stats, contract length in years
  3. Season-based train/test split (train 18-19+19-20, test 20-21)
  4. Hyperparameter tuning via RandomizedSearchCV
  5. XGBoost + LightGBM trained alongside RF, blended via stacking meta-learner
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
df_1819 = pd.read_excel("processed_dataset/18-19.xlsx")
df_1920 = pd.read_excel("processed_dataset/19-20.xlsx")
df_2021 = pd.read_excel("processed_dataset/20-21.xlsx")

df_1819["Season"] = "18-19"
df_1920["Season"] = "19-20"
df_2021["Season"] = "20-21"

all_data = pd.concat([df_1819, df_1920, df_2021], ignore_index=True)
print(f"Total records across all seasons: {len(all_data)}")

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def engineer_features(df):
    d = df.copy()

    # Per-90 normalised stats (avoid div-by-zero)
    min90 = d["Min"].replace(0, np.nan) / 90
    for col in ["Gls", "Ast", "SoT", "TklW", "Int", "Clr", "Carries",
                "Dribble_Att", "Pass_Att", "Blocks", "Press"]:
        if col in d.columns:
            d[f"{col}_p90"] = d[col] / min90

    # Contract length: parse years from LENGTH string (e.g. "3 years")
    def parse_length(val):
        try:
            return float(str(val).split()[0])
        except Exception:
            return np.nan

    d["Contract_Years"] = d["LENGTH"].apply(parse_length)
    d["Contract_Years"] = d["Contract_Years"].fillna(d["Contract_Years"].median())

    # Age brackets
    d["Age_sq"] = d["Current_Age"] ** 2

    # Minutes share (proxy for starter vs rotation)
    max_min = d["Min"].max()
    d["Min_share"] = d["Min"] / max_min

    return d


all_data = engineer_features(all_data)

# ─────────────────────────────────────────────
# 3. SEASON-BASED TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
train_df = all_data[all_data["Season"].isin(["18-19", "19-20"])].copy()
test_df  = all_data[all_data["Season"] == "20-21"].copy()

print(f"Train size: {len(train_df)}  |  Test size: {len(test_df)}")

FEATURE_COLS = [
    "Current_Age", "Age_sq", "POS", "grade_value",
    "Starts", "Min", "Min_share",
    "Gls", "Ast", "CrdY", "CrdR", "SoT", "G_Sh",
    "Pass_Att", "Cmp_per", "TklW", "Blocks", "Int", "Clr",
    "Dribble_Att", "Dribble_Succ_per", "Carries", "Targ", "Rec_per",
    "League_num", "Club_num", "Contract_Years",
    # per-90 features
    "Gls_p90", "Ast_p90", "SoT_p90", "TklW_p90", "Int_p90",
    "Clr_p90", "Carries_p90", "Dribble_Att_p90", "Pass_Att_p90",
]
# Keep only cols that exist
FEATURE_COLS = [c for c in FEATURE_COLS if c in all_data.columns]

X_train = train_df[FEATURE_COLS].fillna(0).values
X_test  = test_df[FEATURE_COLS].fillna(0).values

# Log-transform salary target (improvement #3)
y_train_raw = train_df["WEEKLY_GROSS"].values
y_test_raw  = test_df["WEEKLY_GROSS"].values
y_train = np.log1p(y_train_raw)
y_test  = np.log1p(y_test_raw)

print(f"Features used: {len(FEATURE_COLS)}")
print(f"Feature list: {FEATURE_COLS}")

# ─────────────────────────────────────────────
# 4. HYPERPARAMETER TUNING — RANDOM FOREST
# ─────────────────────────────────────────────
print("\n--- Tuning Random Forest ---")

rf_param_dist = {
    "n_estimators":     [200, 300, 500, 800],
    "max_depth":        [None, 10, 20, 30],
    "min_samples_split":[2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features":     ["sqrt", "log2", None],
}

rf_base = RandomForestRegressor(random_state=2, n_jobs=-1)
rf_search = RandomizedSearchCV(
    rf_base, rf_param_dist, n_iter=20, scoring="r2",
    cv=5, random_state=42, n_jobs=-1, verbose=0
)
rf_search.fit(X_train, y_train)
best_rf = rf_search.best_estimator_
print(f"Best RF params: {rf_search.best_params_}")
print(f"Best RF CV R²: {rf_search.best_score_:.4f}")

# ─────────────────────────────────────────────
# 5a. XGBOOST
# ─────────────────────────────────────────────
print("\n--- Training XGBoost ---")

xgb_param_dist = {
    "n_estimators":     [200, 400, 600],
    "max_depth":        [4, 6, 8],
    "learning_rate":    [0.01, 0.05, 0.1],
    "subsample":        [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.85, 1.0],
    "min_child_weight": [1, 3, 5],
}

xgb_base = XGBRegressor(random_state=2, n_jobs=-1, verbosity=0)
xgb_search = RandomizedSearchCV(
    xgb_base, xgb_param_dist, n_iter=15, scoring="r2",
    cv=5, random_state=42, n_jobs=-1, verbose=0
)
xgb_search.fit(X_train, y_train)
best_xgb = xgb_search.best_estimator_
print(f"Best XGB params: {xgb_search.best_params_}")
print(f"Best XGB CV R²: {xgb_search.best_score_:.4f}")

# ─────────────────────────────────────────────
# 5b. LIGHTGBM
# ─────────────────────────────────────────────
print("\n--- Training LightGBM ---")

lgb_param_dist = {
    "n_estimators":     [200, 400],
    "max_depth":        [6, 8],
    "learning_rate":    [0.05, 0.1],
    "subsample":        [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "num_leaves":       [31, 63],
    "min_child_samples":[10, 20],
}

lgb_base = LGBMRegressor(random_state=2, n_jobs=1, verbose=-1)
lgb_search = RandomizedSearchCV(
    lgb_base, lgb_param_dist, n_iter=15, scoring="r2",
    cv=5, random_state=42, n_jobs=1, verbose=0
)
lgb_search.fit(X_train, y_train)
best_lgb = lgb_search.best_estimator_
print(f"Best LGB params: {lgb_search.best_params_}")
print(f"Best LGB CV R²: {lgb_search.best_score_:.4f}")

# ─────────────────────────────────────────────
# 5c. STACKING ENSEMBLE
# ─────────────────────────────────────────────
print("\n--- Building Stacking Ensemble ---")

stack = StackingRegressor(
    estimators=[
        ("rf",  best_rf),
        ("xgb", best_xgb),
        ("lgb", best_lgb),
    ],
    final_estimator=Ridge(alpha=1.0),
    cv=5,
    n_jobs=-1,
)
stack.fit(X_train, y_train)
print("Stacking ensemble trained.")

# ─────────────────────────────────────────────
# 6. EVALUATE ON HELD-OUT TEST SET (20-21)
# ─────────────────────────────────────────────
def evaluate(name, model, X, y_log, y_raw):
    pred_log = model.predict(X)
    pred_raw = np.expm1(pred_log)
    r2  = r2_score(y_log, pred_log)
    mae = mean_absolute_error(y_raw, pred_raw)
    rmse = np.sqrt(mean_squared_error(y_raw, pred_raw))
    sape = np.mean(np.abs(y_raw - pred_raw) / ((y_raw + pred_raw) / 2))
    return {"Model": name, "R²": round(r2, 4), "MAE (£/wk)": round(mae, 0),
            "RMSE (£/wk)": round(rmse, 0), "Mean SAPE": round(sape, 4),
            "pred_raw": pred_raw}

results = []
preds   = {}

baseline_rf = RandomForestRegressor(
    max_features=None, n_estimators=200, max_depth=None,
    min_samples_split=2, min_samples_leaf=1,
    criterion="squared_error", random_state=2, n_jobs=-1
)
baseline_rf.fit(X_train, y_train)

for name, model in [
    ("Baseline RF (original)",   baseline_rf),
    ("Tuned RF",                 best_rf),
    ("XGBoost",                  best_xgb),
    ("LightGBM",                 best_lgb),
    ("Stacking Ensemble",        stack),
]:
    r = evaluate(name, model, X_test, y_test, y_test_raw)
    preds[name] = r.pop("pred_raw")
    results.append(r)

results_df = pd.DataFrame(results)
print("\n=== MODEL COMPARISON ON TEST SET (20-21 season) ===")
print(results_df.to_string(index=False))

# ─────────────────────────────────────────────
# 7. VALUATION LABELS FOR BEST MODEL
# ─────────────────────────────────────────────
SAPE_THRESHOLD = 0.2937

def label_valuation(actual, pred, threshold=SAPE_THRESHOLD):
    sape = abs(actual - pred) / ((actual + pred) / 2)
    if sape <= threshold:
        return "Normal"
    return "Underestimation" if pred > actual else "Overestimation"

best_model_name = results_df.sort_values("MAE (£/wk)").iloc[0]["Model"]
print(f"\nBest model by MAE: {best_model_name}")

best_preds = preds[best_model_name]
output_df = test_df[["Player", "Current_Age", "Season", "WEEKLY_GROSS"]].copy()
output_df["Best_Model_Predicted"] = np.round(best_preds, 2)
output_df["Gap_(Pred-Actual)"]    = np.round(best_preds - output_df["WEEKLY_GROSS"], 2)
output_df["SAPE"] = output_df.apply(
    lambda r: abs(r["WEEKLY_GROSS"] - r["Best_Model_Predicted"]) /
              ((r["WEEKLY_GROSS"] + r["Best_Model_Predicted"]) / 2), axis=1
)
output_df["Valuation"] = output_df.apply(
    lambda r: label_valuation(r["WEEKLY_GROSS"], r["Best_Model_Predicted"]), axis=1
)

# Add individual model predictions
for name, pred_arr in preds.items():
    output_df[name.replace(" ", "_")] = np.round(pred_arr, 2)

print("\nValuation distribution (best model on test set):")
print(output_df["Valuation"].value_counts())

# ─────────────────────────────────────────────
# 8. FEATURE IMPORTANCE (Tuned RF)
# ─────────────────────────────────────────────
fi_df = pd.DataFrame({
    "Feature":   FEATURE_COLS,
    "Importance": best_rf.feature_importances_
}).sort_values("Importance", ascending=False)

print("\nTop 15 Feature Importances (Tuned RF):")
print(fi_df.head(15).to_string(index=False))

# ─────────────────────────────────────────────
# 9. SAVE ALL RESULTS TO EXCEL
# ─────────────────────────────────────────────
print("\nSaving results to Improved_Model_Results.xlsx ...")

with pd.ExcelWriter("Improved_Model_Results.xlsx", engine="openpyxl") as writer:

    # Sheet 1: Model comparison
    results_df.to_excel(writer, sheet_name="Model_Comparison", index=False)

    # Sheet 2: Test set predictions (all models)
    output_df.to_excel(writer, sheet_name="Test_Predictions", index=False)

    # Sheet 3: Feature importance
    fi_df.to_excel(writer, sheet_name="Feature_Importance", index=False)

    # Sheet 4: Training inputs summary
    train_summary = train_df[FEATURE_COLS + ["WEEKLY_GROSS", "Player", "Season"]].copy()
    train_summary.insert(0, "Split", "Train")
    test_summary  = test_df[FEATURE_COLS + ["WEEKLY_GROSS", "Player", "Season"]].copy()
    test_summary.insert(0, "Split", "Test")
    pd.concat([train_summary, test_summary]).to_excel(writer, sheet_name="All_Inputs", index=False)

    # Sheet 5: Valuation summary by model
    val_rows = []
    for name, pred_arr in preds.items():
        labels = [label_valuation(a, p) for a, p in zip(y_test_raw, pred_arr)]
        from collections import Counter
        c = Counter(labels)
        val_rows.append({
            "Model": name,
            "Normal": c.get("Normal", 0),
            "Underestimation": c.get("Underestimation", 0),
            "Overestimation": c.get("Overestimation", 0),
        })
    pd.DataFrame(val_rows).to_excel(writer, sheet_name="Valuation_Summary", index=False)

print("Done — Improved_Model_Results.xlsx written.")

# Save best model
joblib.dump(stack, "trained_models/stacking_ensemble.pkl")
joblib.dump(best_rf, "trained_models/tuned_RF_model.pkl")
joblib.dump(best_xgb, "trained_models/xgb_model.pkl")
joblib.dump(best_lgb, "trained_models/lgb_model.pkl")
print("Trained models saved to trained_models/")
