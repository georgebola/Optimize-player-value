# Player Value Estimation — Model Documentation

## Overview

This project builds a machine learning system that estimates what a football player's weekly salary *should* be based on their on-pitch performance, age, contract situation, league, and club. By comparing the model's estimate to what a player is actually earning, we can flag whether a player is **overvalued**, **undervalued**, or **fairly valued** — giving us a structured, data-driven signal for transfer and contract decisions.

---

## The Core Question We Are Answering

> *Given what a player actually does on the pitch, what should the market be paying them?*

The gap between what the model says a player is worth and what they are currently earning is the key output. That gap is what drives our buy / hold / sell decisions.

---

## Data

- **Seasons covered:** 2018-19, 2019-20, 2020-21
- **Total records:** 4,021 player-seasons
- **Target variable:** `WEEKLY_GROSS` — the player's actual weekly gross salary in £
- **Features:** 36 inputs covering age, position, performance stats (goals, assists, tackles, passes, carries, etc.), league, club, and contract length

---

## The Valuation Labels

Every player in the output receives one of three labels based on the **Symmetric Absolute Percentage Error (SAPE)** between the model's predicted salary and their actual salary:

| Label | Meaning | SAPE threshold |
|---|---|---|
| **Underestimation** | Model predicts more than they earn → player is **undervalued by the market** | SAPE > 29.37% and predicted > actual |
| **Normal** | Model and actual salary are broadly in line | SAPE ≤ 29.37% |
| **Overestimation** | Model predicts less than they earn → player is **overvalued by the market** | SAPE > 29.37% and predicted < actual |

The 29.37% threshold was established in the original research as the boundary that best separates meaningful mispricing from normal prediction noise.

---

## Buy / Hold / Sell Decision Framework

| Valuation Label | Signal | Action |
|---|---|---|
| **Underestimation** | Player outperforms their wage — the market hasn't priced in their output | **Buy / Extend contract** — you can acquire or retain this player at below-market value for their contribution |
| **Normal** | Player is fairly compensated relative to output | **Hold** — no mispricing to exploit; reassess at next contract milestone |
| **Overestimation** | Player earns more than their output justifies | **Sell / Let contract expire** — offloading frees budget without losing proportional output |

**Important:** The model should be used as one input among others. Factors like squad depth, positional need, age trajectory, and transfer market availability must also be considered. A player labelled "Underestimation" who is 34 years old needs a different conversation to one who is 21.

---

## Key Output Columns in `Improved_Model_Results.xlsx`

The file contains five sheets:

### Sheet: `Test_Predictions` (primary decision sheet)

| Column | What it means |
|---|---|
| `Player` | Player name |
| `Current_Age` | Age at time of prediction |
| `WEEKLY_GROSS` | Actual weekly salary (£) |
| `Best_Model_Predicted` | Stacking Ensemble's salary estimate (£) |
| `Gap_(Pred-Actual)` | Positive = model thinks player should earn more (undervalued). Negative = model thinks player earns too much (overvalued) |
| `SAPE` | Symmetric error — how far off the estimate is as a proportion |
| `Valuation` | **Underestimation / Normal / Overestimation** — the decision label |

### Sheet: `Model_Comparison`
All five models' R², MAE, RMSE, and SAPE on the held-out 2020-21 test season.

### Sheet: `Feature_Importance`
Which inputs drove the model's predictions most — useful for understanding *why* a player received a particular valuation.

### Sheet: `Valuation_Summary`
How each model distributes players across the three labels — useful for cross-checking.

---

## Model Development: From Baseline to Final

### Case 1 — Baseline Random Forest (original)

**What it was:**
A single Random Forest trained on 24 raw features with default hyperparameters (`n_estimators=200`, no depth limit, `min_samples_leaf=1`). Training and testing used all three seasons mixed together — no temporal separation.

**Features:** Age, position, grade value, starts, minutes, goals, assists, cards, shots, pass accuracy, tackles, blocks, interceptions, clearances, dribbles, carries, targets received, reception rate, league, club (24 features total).

**Results:**
| Metric | Score |
|---|---|
| R² (test) | 0.541 |
| MAE | £28,643 / week |
| RMSE | £58,578 / week |
| Mean SAPE | 0.604 |

**Problems identified:**
1. Salary data is heavily right-skewed (a few top earners pulling the distribution). The model was trying to predict raw £ values directly, making it hard to handle the full range.
2. Raw stats like goals and tackles don't account for how much a player plays — a striker with 10 goals in 500 minutes is very different from one with 10 goals in 3,000 minutes.
3. No contract length feature, even though longer contracts are correlated with higher wages.
4. Hyperparameters were never tuned — the model used defaults designed for general use, not this specific problem.
5. Temporal data leakage: training and test data were mixed across seasons, so the model could "see the future" during cross-validation.

---

### Case 2 — Tuned Random Forest

**What changed:**
- Hyperparameters searched via `RandomizedSearchCV` over 20 combinations (5-fold CV)
- Best parameters found: `n_estimators=200`, `max_depth=30`, `min_samples_split=5`, `min_samples_leaf=4`
- Applied the same log-transform and feature engineering as described below

**Results:**
| Metric | Score |
|---|---|
| R² (test) | 0.533 |
| MAE | £28,945 / week |
| RMSE | £59,605 / week |
| Mean SAPE | 0.607 |

**Observation:** Tuning the RF alone did not help — it actually performed slightly worse than the baseline on this test set. Random Forests are relatively robust to hyperparameter choice at this data scale, and the architecture of the model itself was the limiting factor, not the parameters. This told us we needed different model families, not just better-tuned versions of the same one.

---

### Case 3 — XGBoost (Gradient Boosting)

**What changed:**
- Replaced Random Forest with XGBoost, a gradient boosting model that builds trees sequentially, each one correcting the errors of the last
- Tuned with `RandomizedSearchCV` over 15 combinations: learning rate, tree depth, subsampling, column sampling, minimum child weight
- Best parameters: `n_estimators=400`, `max_depth=6`, `learning_rate=0.05`, `subsample=0.85`, `colsample_bytree=0.85`

**Why gradient boosting helps here:** Unlike Random Forest (which averages many independent trees), XGBoost focuses each new tree on the hardest-to-predict players from the previous round. This is particularly useful for salary data where the high earners are outliers that a bagging model tends to underfit.

**Results:**
| Metric | Score |
|---|---|
| R² (test) | 0.611 |
| MAE | £25,526 / week |
| RMSE | £51,138 / week |
| Mean SAPE | 0.558 |

**Improvement over baseline:** MAE down £3,117/week (−10.9%), RMSE down £7,440/week.

---

### Case 4 — LightGBM

**What changed:**
- Trained LightGBM alongside XGBoost — a gradient boosting framework optimised for speed and handling of large feature sets, using leaf-wise tree growth instead of level-wise
- Tuned with `RandomizedSearchCV` over 15 combinations
- Best parameters: `n_estimators=400`, `max_depth=6`, `learning_rate=0.05`, `num_leaves=31`, `min_child_samples=10`

**Results:**
| Metric | Score |
|---|---|
| R² (test) | 0.614 |
| MAE | £25,375 / week |
| RMSE | £49,832 / week |
| Mean SAPE | 0.554 |

**Improvement over baseline:** MAE down £3,268/week (−11.4%). Slightly better than XGBoost on all metrics.

---

### Case 5 — Stacking Ensemble (chosen model)

**What it is:**
A meta-learning approach that combines the predictions of all three tuned models (RF, XGBoost, LightGBM) as inputs to a Ridge regression meta-learner. Rather than picking one winner, the stacking layer learns *how much to trust each model* for different types of players.

**Why stacking works here:**
- RF captures broad patterns well but struggles with extremes
- XGBoost corrects the hard cases but can overfit noisier players
- LightGBM is the fastest at adapting to the data distribution
- The Ridge meta-learner finds the optimal blend of these three signals, reducing the variance of any single model's errors

**Architecture:**
```
  Player features
        │
  ┌─────┴─────┐
  RF   XGB   LGB    ← base models (5-fold CV predictions)
  │     │     │
  └─────┴─────┘
        │
   Ridge meta-learner
        │
  Final salary estimate
```

**Results:**
| Metric | Score |
|---|---|
| R² (test) | **0.629** |
| MAE | **£24,665 / week** |
| RMSE | **£47,888 / week** |
| Mean SAPE | **0.543** |

**Improvement over baseline:**
| Metric | Baseline RF | Stacking Ensemble | Improvement |
|---|---|---|---|
| R² | 0.541 | 0.629 | +16.3% |
| MAE | £28,643 | £24,665 | **−13.9%** |
| RMSE | £58,578 | £47,888 | **−18.3%** |
| Mean SAPE | 0.604 | 0.543 | −10.1% |

**Valuation distribution on 2020-21 test season (1,581 players):**
- Underestimation: 584 players (36.9%)
- Normal: 551 players (34.8%)
- Overestimation: 446 players (28.2%)

---

## Engineering Improvements Applied Across Cases 3-5

These changes were applied consistently once identified and contributed to the gains across all three improved models:

### 1. Log-transform on salary target
Weekly salaries range from ~£500 to £350,000+ — a highly skewed distribution. Predicting on the log scale (`log(salary + 1)`) compresses this range, making the learning problem more uniform. Predictions are then converted back (`exp(prediction) - 1`) for interpretation.

### 2. Per-90 normalised statistics
A player's raw goal tally is misleading without context. A player who scores 8 goals in 900 minutes is different from one who scores 8 in 2,700 minutes. We created per-90-minute versions of 10 key stats: goals, assists, shots on target, tackles won, interceptions, clearances, carries, dribble attempts, pass attempts, and blocks.

### 3. Age-squared term
The relationship between age and salary is non-linear — earnings peak in the late 20s and decline. Adding `Age²` as a feature allows tree models to capture this curve without needing to bin ages manually.

### 4. Contract length
Parsed from the `LENGTH` field (e.g., "3 years") into a numeric `Contract_Years` feature. Longer contracts are associated with higher wages and represent the club's confidence in the player.

### 5. Minutes share
`Min / max(Min)` — a normalised proxy for whether the player is a starter (high share) or a rotation/squad player (low share). This separates players whose raw counting stats look similar but who have very different roles.

### 6. Temporal train/test split (no data leakage)
The original model trained and evaluated on mixed seasons. We corrected this by training only on 2018-19 and 2019-20, then evaluating exclusively on 2020-21. This reflects the real-world use case: you train on past seasons and predict on a future one you haven't seen.

---

## Top Features Driving the Model (from Tuned RF)

| Feature | Importance | Interpretation |
|---|---|---|
| `Targ` (targets received) | 26.2% | High-demand players — those who receive the ball frequently — earn more |
| `grade_value` | 18.6% | Market valuation / transfer value proxy — strongly correlated with wages |
| `League_num` | 9.8% | Top leagues (Premier League, etc.) pay significantly more |
| `Age_sq` | 4.1% | Non-linear age effect captured by the squared term |
| `Club_num` | 3.9% | Elite clubs pay a premium independent of individual stats |
| `Current_Age` | 3.7% | Peak earnings in late 20s |
| `Pass_Att` | 2.5% | Volume of involvement — midfielders and CBs rank here |

---

## The Model We Use Going Forward: Stacking Ensemble

**File saved:** `trained_models/stacking_ensemble.pkl`

We move forward with the **Stacking Ensemble** as the production model. It achieves the best performance on every metric and is the most robust — by combining three different model families, it is less sensitive to any single model's blind spots.

### Immediate next goal

The model currently explains **62.9% of salary variance** (R² = 0.629) with a mean absolute error of **£24,665/week**. The remaining error comes primarily from:

1. **Reputation and brand value** — players like Ronaldo or Beckham earn a premium that no performance metric captures
2. **Injury history** — not in the dataset but affects contract terms
3. **Agent leverage and negotiation** — non-measurable
4. **Transfer fee paid** — clubs amortise large transfer fees through higher wages

The goal for the next iteration is to push R² above 0.70 and MAE below £20,000/week by incorporating:
- Transfer fee data as a feature
- Player reputation proxies (international caps, previous club tier)
- Multi-season performance trends (not just the current season snapshot)
- Position-specific sub-models (a goalkeeper's relevant features are entirely different from a striker's)

---

## How to Use the Output File for Decisions

1. Open `Improved_Model_Results.xlsx`
2. Go to the **`Test_Predictions`** sheet
3. Sort by `Gap_(Pred-Actual)` descending to find the most undervalued players (model thinks they should earn far more than they do)
4. Sort ascending to find the most overvalued
5. Cross-reference the `SAPE` column — a large gap is more meaningful when SAPE is also high
6. Use the `Valuation` column as the headline flag for each player
7. Check `Feature_Importance` to understand *why* the model values a player the way it does before making a final call
