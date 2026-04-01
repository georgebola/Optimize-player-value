"""
Marcos Llorente — actual salary (last 5 seasons, web data) vs
model-estimated salary (baseline RF, trained on 18-19 + 19-20, predicting 20-21).

Actual salary sources:
  - 2020-21 / 2021-22: Capology / SalarySport — £40,385/wk (old Atletico contract)
  - 2022-23 onward: new contract signed Aug 2022, €8M/yr ≈ £130,000/wk
  - 2023-24: SalarySport — £136,437/wk
  - 2024-25: SalarySport — £138,114/wk
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

# ── 1. Train baseline RF and get model estimate for 2020-21 ─────────────────
df_1819 = pd.read_excel("processed_dataset/18-19.xlsx"); df_1819["Season"] = "18-19"
df_1920 = pd.read_excel("processed_dataset/19-20.xlsx"); df_1920["Season"] = "19-20"
df_2021 = pd.read_excel("processed_dataset/20-21.xlsx"); df_2021["Season"] = "20-21"
all_data = pd.concat([df_1819, df_1920, df_2021], ignore_index=True)

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

train_df = all_data[all_data["Season"].isin(["18-19","19-20"])].copy()
rf = RandomForestRegressor(
    max_features=None, n_estimators=200, max_depth=None,
    min_samples_split=2, min_samples_leaf=1,
    criterion="squared_error", random_state=2, n_jobs=-1
)
rf.fit(train_df[FEATURE_COLS].fillna(0).values,
       np.log1p(train_df["WEEKLY_GROSS"].values))

# Predict on the 2020-21 Llorente row
llorente_row = all_data[
    (all_data["Player"] == "Marcos Llorente") &
    (all_data["Season"] == "20-21")
]
model_estimate = np.expm1(
    rf.predict(llorente_row[FEATURE_COLS].fillna(0).values)[0]
)
print(f"Model estimate (20-21): £{model_estimate:,.0f}/wk")

# ── 2. Real salary data (from web sources) ──────────────────────────────────
seasons = ["20-21", "21-22", "22-23", "23-24", "24-25"]
actual_salary = [
    40_385,    # Capology / dataset — old Atletico contract
    40_385,    # Capology — old contract still in place
    130_000,   # New contract signed Aug 2022: €8M/yr ≈ £130,000/wk
    136_437,   # SalarySport
    138_114,   # SalarySport
]

# ── 3. Plot ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

# Actual salary bars
bars = ax.bar(seasons, actual_salary,
              color="#1f77b4", alpha=0.8, width=0.5,
              label="Actual salary (market)", zorder=2)

# Model estimate — dashed horizontal line (based on 2020-21 stats)
ax.axhline(y=model_estimate, color="#d62728", linewidth=2.5,
           linestyle="--", label=f"Model estimate (2020-21 stats): £{model_estimate:,.0f}/wk",
           zorder=3)

# Annotate the gap in 20-21
gap = model_estimate - actual_salary[0]
ax.annotate(
    f"Gap: £{gap:,.0f}/wk\n(market undervaluation)",
    xy=("20-21", actual_salary[0]),
    xytext=(0.18, 0.38),
    textcoords="axes fraction",
    arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.5),
    fontsize=9, color="#d62728", fontweight="bold"
)

# Annotate market correction
ax.annotate(
    "Market correction:\nnew contract signed",
    xy=("22-23", actual_salary[2]),
    xytext=(0.58, 0.72),
    textcoords="axes fraction",
    arrowprops=dict(arrowstyle="->", color="green", lw=1.5),
    fontsize=9, color="green", fontweight="bold"
)

ax.set_title("Marcos Llorente — Actual vs Model-Estimated Weekly Salary\n"
             "(Model trained on 18-19 & 19-20, estimated on 20-21 stats)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Season", fontsize=11)
ax.set_ylabel("Weekly Salary (£)", fontsize=11)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
ax.legend(fontsize=10, loc="upper left")
ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=1)
ax.set_ylim(0, model_estimate * 1.2)

# Source note
fig.text(0.5, -0.02,
         "Sources: Capology, SalarySport, SalaryLeaks | Model: Baseline Random Forest",
         ha="center", fontsize=8, color="grey")

plt.tight_layout()
plt.savefig("marcos_llorente_salary.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved to marcos_llorente_salary.png")
