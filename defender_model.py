"""
Defender-only salary estimation — Baseline RF (POS=1).
Trains on 18-19 + 19-20, tests on 20-21 holdout.
Prints all metrics needed for comparison with full-dataset RF.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from collections import Counter

# ── 1. LOAD & FILTER TO DEFENDERS ───────────────────────────────────────────
df_1819 = pd.read_excel("processed_dataset/18-19.xlsx"); df_1819["Season"] = "18-19"
df_1920 = pd.read_excel("processed_dataset/19-20.xlsx"); df_1920["Season"] = "19-20"
df_2021 = pd.read_excel("processed_dataset/20-21.xlsx"); df_2021["Season"] = "20-21"
all_data = pd.concat([df_1819, df_1920, df_2021], ignore_index=True)

# Filter defenders only (POS=1)
all_data = all_data[all_data["POS"] == 1].copy()
print(f"Defenders — Total rows: {len(all_data)}")
print(all_data["Season"].value_counts().sort_index())

# ── 2. FEATURE ENGINEERING ───────────────────────────────────────────────────
def engineer_features(df):
    d = df.copy()
    min90 = d["Min"].replace(0, np.nan) / 90
    for col in ["Gls","Ast","SoT","TklW","Int","Clr","Carries",
                "Dribble_Att","Pass_Att","Blocks","Press"]:
        if col in d.columns:
            d[f"{col}_p90"] = d[col] / min90
    def parse_length(val):
        try: return float(str(val).split()[0])
        except: return np.nan
    d["Contract_Years"] = d["LENGTH"].apply(parse_length)
    d["Contract_Years"] = d["Contract_Years"].fillna(d["Contract_Years"].median())
    d["Age_sq"]    = d["Current_Age"] ** 2
    d["Min_share"] = d["Min"] / d["Min"].max()
    return d

all_data = engineer_features(all_data)

FEATURE_COLS = [
    "Current_Age","Age_sq","POS","grade_value",
    "Starts","Min","Min_share",
    "Gls","Ast","CrdY","CrdR","SoT","G_Sh",
    "Pass_Att","Cmp_per","TklW","Blocks","Int","Clr",
    "Dribble_Att","Dribble_Succ_per","Carries","Targ","Rec_per",
    "League_num","Club_num","Contract_Years",
    "Gls_p90","Ast_p90","SoT_p90","TklW_p90","Int_p90",
    "Clr_p90","Carries_p90","Dribble_Att_p90","Pass_Att_p90",
]
FEATURE_COLS = [c for c in FEATURE_COLS if c in all_data.columns]

# ── 3. SEASON-BASED TRAIN/TEST SPLIT ────────────────────────────────────────
train_df = all_data[all_data["Season"].isin(["18-19","19-20"])].copy()
test_df  = all_data[all_data["Season"] == "20-21"].copy()
print(f"\nTrain: {len(train_df)} | Test: {len(test_df)}")

X_train   = train_df[FEATURE_COLS].fillna(0).values
X_test    = test_df[FEATURE_COLS].fillna(0).values
y_train_raw = train_df["WEEKLY_GROSS"].values
y_test_raw  = test_df["WEEKLY_GROSS"].values
y_train = np.log1p(y_train_raw)
y_test  = np.log1p(y_test_raw)

# ── 4. BASELINE RF ───────────────────────────────────────────────────────────
print("\n--- Training Baseline RF ---")
rf = RandomForestRegressor(
    max_features=None, n_estimators=200, max_depth=None,
    min_samples_split=2, min_samples_leaf=1,
    criterion="squared_error", random_state=2, n_jobs=-1
)
rf.fit(X_train, y_train)

pred_log = rf.predict(X_test)
pred_raw = np.expm1(pred_log)

r2   = r2_score(y_test, pred_log)
mae  = mean_absolute_error(y_test_raw, pred_raw)
rmse = np.sqrt(mean_squared_error(y_test_raw, pred_raw))
sape = np.mean(np.abs(y_test_raw - pred_raw) / ((y_test_raw + pred_raw) / 2))

print("\n=== DEFENDER BASELINE RF — TEST SET METRICS (2020-21) ===")
print(f"  R²            : {r2:.4f}")
print(f"  MAE (£/wk)    : £{mae:,.0f}")
print(f"  RMSE (£/wk)   : £{rmse:,.0f}")
print(f"  Mean SAPE     : {sape:.4f}  ({sape*100:.2f}%)")

# ── 5. VALUATION LABELS ──────────────────────────────────────────────────────
SAPE_THRESHOLD = 0.2937

def label_valuation(actual, pred):
    s = abs(actual - pred) / ((actual + pred) / 2)
    if s <= SAPE_THRESHOLD: return "Normal"
    return "Underestimation" if pred > actual else "Overestimation"

output_df = test_df[["Player","Current_Age","Season","WEEKLY_GROSS"]].copy()
output_df["Predicted"]         = np.round(pred_raw, 2)
output_df["Gap_(Pred-Actual)"] = np.round(pred_raw - output_df["WEEKLY_GROSS"], 2)
output_df["SAPE"] = output_df.apply(
    lambda r: abs(r["WEEKLY_GROSS"]-r["Predicted"])/((r["WEEKLY_GROSS"]+r["Predicted"])/2), axis=1)
output_df["Valuation"] = output_df.apply(
    lambda r: label_valuation(r["WEEKLY_GROSS"], r["Predicted"]), axis=1)

print("\nValuation distribution:")
print(output_df["Valuation"].value_counts())

# ── 6. TOP UNDER / OVER VALUED ───────────────────────────────────────────────
top_under = (output_df[output_df["Valuation"]=="Underestimation"]
             .sort_values("Gap_(Pred-Actual)", ascending=False).head(10))
top_over  = (output_df[output_df["Valuation"]=="Overestimation"]
             .sort_values("Gap_(Pred-Actual)").head(10))

print("\nTop 10 undervalued defenders (model thinks they deserve more):")
print(top_under[["Player","Current_Age","WEEKLY_GROSS","Predicted","Gap_(Pred-Actual)"]].to_string(index=False))

print("\nTop 10 overvalued defenders (model thinks they earn too much):")
print(top_over[["Player","Current_Age","WEEKLY_GROSS","Predicted","Gap_(Pred-Actual)"]].to_string(index=False))

# ── 7. FEATURE IMPORTANCE ────────────────────────────────────────────────────
fi_df = pd.DataFrame({
    "Feature":    FEATURE_COLS,
    "Importance": rf.feature_importances_
}).sort_values("Importance", ascending=False)

print("\nTop 10 features:")
print(fi_df.head(10).to_string(index=False))

# ── 8. SAVE TO EXCEL ─────────────────────────────────────────────────────────
print("\nSaving to Defender_Model_Results.xlsx ...")
perf_df = pd.DataFrame([{
    "Model": "Baseline RF (Defenders only)",
    "R²":    round(r2, 4),
    "MAE (£/wk)":  round(mae, 0),
    "RMSE (£/wk)": round(rmse, 0),
    "Mean SAPE":   round(sape, 4),
}])

with pd.ExcelWriter("Defender_Model_Results.xlsx", engine="openpyxl") as writer:
    perf_df.to_excel(writer,   sheet_name="Model_Comparison",   index=False)
    output_df.to_excel(writer, sheet_name="Test_Predictions",   index=False)
    fi_df.to_excel(writer,     sheet_name="Feature_Importance", index=False)

    train_s = train_df[FEATURE_COLS+["WEEKLY_GROSS","Player","Season"]].copy()
    train_s.insert(0,"Split","Train")
    test_s  = test_df[FEATURE_COLS+["WEEKLY_GROSS","Player","Season"]].copy()
    test_s.insert(0,"Split","Test")
    pd.concat([train_s, test_s]).to_excel(writer, sheet_name="All_Inputs", index=False)

print("Done.")
