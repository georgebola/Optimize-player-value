"""
Plot actual vs estimated salary across all available seasons
for the top 3 undervalued players (highest predicted-vs-actual gap in 20-21).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.ensemble import RandomForestRegressor

# ── Load & tag seasons ──────────────────────────────────────────────────────
df_1819 = pd.read_excel("processed_dataset/18-19.xlsx"); df_1819["Season"] = "18-19"
df_1920 = pd.read_excel("processed_dataset/19-20.xlsx"); df_1920["Season"] = "19-20"
df_2021 = pd.read_excel("processed_dataset/20-21.xlsx"); df_2021["Season"] = "20-21"
all_data = pd.concat([df_1819, df_1920, df_2021], ignore_index=True)

# ── Feature engineering (mirrors improved_model.py) ─────────────────────────
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

# ── Train baseline RF on 18-19 + 19-20, predict on all seasons ──────────────
train_df = all_data[all_data["Season"].isin(["18-19","19-20"])].copy()
X_train  = train_df[FEATURE_COLS].fillna(0).values
y_train  = np.log1p(train_df["WEEKLY_GROSS"].values)

rf = RandomForestRegressor(
    max_features=None, n_estimators=200, max_depth=None,
    min_samples_split=2, min_samples_leaf=1,
    criterion="squared_error", random_state=2, n_jobs=-1
)
rf.fit(X_train, y_train)

# Predict for every row in the full dataset
all_data["Predicted"] = np.expm1(rf.predict(all_data[FEATURE_COLS].fillna(0).values))

# ── Top 3 undervalued players (by gap in 20-21 holdout) ─────────────────────
test_df = all_data[all_data["Season"] == "20-21"].copy()
test_df["Gap"] = test_df["Predicted"] - test_df["WEEKLY_GROSS"]
top3_players = (
    test_df[test_df["Gap"] > 0]
    .sort_values("Gap", ascending=False)
    .head(3)["Player"]
    .tolist()
)
print("Top 3 undervalued players:", top3_players)

# ── Plot ─────────────────────────────────────────────────────────────────────
SEASON_ORDER = ["18-19", "19-20", "20-21"]
colors = {"Actual salary": "#1f77b4", "Estimated salary": "#ff7f0e"}

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Actual vs Estimated Weekly Salary — Top 3 Undervalued Players",
             fontsize=14, fontweight="bold", y=1.02)

for ax, player in zip(axes, top3_players):
    pdata = (
        all_data[all_data["Player"] == player][["Season","WEEKLY_GROSS","Predicted"]]
        .drop_duplicates("Season")
        .set_index("Season")
        .reindex(SEASON_ORDER)
        .dropna(how="all")
        .reset_index()
    )

    seasons = pdata["Season"].tolist()
    actual  = pdata["WEEKLY_GROSS"].tolist()
    pred    = pdata["Predicted"].tolist()

    ax.bar(seasons, actual, color=colors["Actual salary"],
           alpha=0.75, label="Actual salary", width=0.4, zorder=2)
    ax.plot(seasons, pred, color=colors["Estimated salary"],
            marker="o", linewidth=2.5, markersize=8,
            label="Estimated salary", zorder=3)

    ax.set_title(player, fontsize=12, fontweight="bold")
    ax.set_xlabel("Season", fontsize=10)
    ax.set_ylabel("Weekly Salary (£)", fontsize=10)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"£{x:,.0f}")
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=1)
    ax.set_ylim(bottom=0)

    # Annotate the 20-21 gap
    last_season = seasons[-1]
    if last_season == "20-21":
        a = actual[-1]; p = pred[-1]
        gap = p - a
        if gap > 0:
            ax.annotate(
                f"Gap: £{gap:,.0f}/wk",
                xy=(last_season, max(a, p)),
                xytext=(0, 10), textcoords="offset points",
                ha="center", fontsize=8, color="#d62728",
                fontweight="bold"
            )

plt.tight_layout()
plt.savefig("top3_undervalued_salary_plot.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot saved to top3_undervalued_salary_plot.png")
