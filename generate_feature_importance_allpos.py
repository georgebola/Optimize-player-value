"""
Regenerate feature_importance.png from the all-position Baseline RF
(same params used for the final model: all positions, train 18-19+19-20).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor

# ── 1. LOAD ALL DATA (no position filter) ────────────────────────────────────
df_1819 = pd.read_excel("processed_dataset/18-19.xlsx"); df_1819["Season"] = "18-19"
df_1920 = pd.read_excel("processed_dataset/19-20.xlsx"); df_1920["Season"] = "19-20"
df_2021 = pd.read_excel("processed_dataset/20-21.xlsx"); df_2021["Season"] = "20-21"
all_data = pd.concat([df_1819, df_1920, df_2021], ignore_index=True)
print(f"All positions — Total rows: {len(all_data)}")

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

# ── 3. TRAIN ON 18-19 + 19-20 (all positions) ────────────────────────────────
train_df = all_data[all_data["Season"].isin(["18-19","19-20"])].copy()
X_train  = train_df[FEATURE_COLS].fillna(0).values
y_train  = np.log1p(train_df["WEEKLY_GROSS"].values)

print(f"Training on {len(train_df)} rows (all positions, 18-19 + 19-20)")

rf = RandomForestRegressor(
    max_features=None, n_estimators=200, max_depth=None,
    min_samples_split=2, min_samples_leaf=1,
    criterion="squared_error", random_state=2, n_jobs=-1
)
rf.fit(X_train, y_train)
print("Training complete.")

# ── 4. FEATURE IMPORTANCE BAR CHART ─────────────────────────────────────────
fi_df = pd.DataFrame({
    "Feature":    FEATURE_COLS,
    "Importance": rf.feature_importances_
}).sort_values("Importance", ascending=False).head(15).reset_index(drop=True)

print("\nTop 15 features (all-position model):")
print(fi_df.to_string(index=False))

# Friendly display names
name_map = {
    "Targ":               "Targets Received",
    "grade_value":        "Grade Value",
    "League_num":         "League",
    "Club_num":           "Club",
    "Age_sq":             "Age²",
    "Current_Age":        "Current Age",
    "Min":                "Minutes Played",
    "Min_share":          "Minutes Share",
    "Contract_Years":     "Contract Years",
    "Pass_Att":           "Pass Attempts",
    "Pass_Att_p90":       "Pass Att p90",
    "Carries":            "Carries",
    "Carries_p90":        "Carries p90",
    "Rec_per":            "Receive %",
    "Cmp_per":            "Pass Completion %",
    "Starts":             "Starts",
    "Gls":                "Goals",
    "Ast":                "Assists",
    "TklW":               "Tackles Won",
    "TklW_p90":           "Tackles Won p90",
    "Int":                "Interceptions",
    "Int_p90":            "Interceptions p90",
    "Clr":                "Clearances",
    "Clr_p90":            "Clearances p90",
    "SoT":                "Shots on Target",
    "SoT_p90":            "Shots on Target p90",
    "Gls_p90":            "Goals p90",
    "Ast_p90":            "Assists p90",
    "Dribble_Att":        "Dribble Attempts",
    "Dribble_Att_p90":    "Dribble Att p90",
    "Dribble_Succ_per":   "Dribble Success %",
    "Blocks":             "Blocks",
    "CrdY":               "Yellow Cards",
    "CrdR":               "Red Cards",
    "G_Sh":               "Goals per Shot",
    "POS":                "Position",
}
fi_df["Label"] = fi_df["Feature"].map(name_map).fillna(fi_df["Feature"])

# Colour coding: red=top 2 (market/reputation), orange=3-5 (context/role), blue=6-15
colors = (
    ["#e74c3c"] * 2 +
    ["#e67e22"] * 3 +
    ["#3498db"] * 10
)[:len(fi_df)]

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(
    fi_df["Label"][::-1],
    fi_df["Importance"][::-1] * 100,
    color=colors[::-1],
    edgecolor="white", linewidth=0.5
)

for bar, val in zip(bars, fi_df["Importance"][::-1] * 100):
    ax.text(val + 0.15, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", ha="left", fontsize=9, color="#2c3e50")

ax.set_xlabel("Feature Importance (%)", fontsize=11)
ax.set_title(
    "Top 15 Features — All-Position Baseline RF\n"
    "(Trained on 18-19 + 19-20, All Positions)",
    fontsize=13, fontweight="bold", pad=14
)
ax.set_xlim(0, fi_df["Importance"].max() * 100 * 1.18)
ax.axvline(0, color="grey", linewidth=0.5)
ax.tick_params(axis="y", labelsize=10)

from matplotlib.patches import Patch
legend_handles = [
    Patch(facecolor="#e74c3c", label="Market & Reputation (ranks 1–2)"),
    Patch(facecolor="#e67e22", label="Context & Role (ranks 3–5)"),
    Patch(facecolor="#3498db", label="Performance Stats (ranks 6–15)"),
]
ax.legend(handles=legend_handles, loc="lower right", fontsize=9, framealpha=0.85)

plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
print("\nSaved feature_importance.png (all-position Baseline RF)")
