# Predicting Chess Upset Anomalies
### A Comparative Analysis of Elo-Based Baselines and Non-Linear Gradient-Boosted Trees
**Author:** Justin Lee  
**Course:** UCSD DSC 148: Introduction to Data Mining  
**Core Stack:** `Python`, `XGBoost`, `LightGBM`, `Scikit-Learn`, `Pandas`

---

## Project Overview
Traditional competitive chess forecasting models rely heavily on historical, pre-game Elo ratings to estimate win probabilities via logistic functions. While these methods are effective for long-term skill tracking, they fail to forecast individual, short-term match anomalies—specifically **upset anomalies**, where a lower-rated player defeats a heavily favored opponent. 

This repository containing the codebase `dsc148-final-project` implements an end-to-end data mining pipeline that reframes chess forecasting as an environment-aware anomaly detection task. By engineering real-time behavioral heuristics (e.g., tactical development tempo, early queen sorties, and king safety indices) alongside environmental constraints (e.g., Blitz vs. Classical clock structures), our gradient-boosted tree architectures break the predictive paralysis of traditional linear baselines.

---

## Dataset & Predictive Task
The modeling pipeline ingests a subset of **100,000 standard competitive matches** sourced from the open-source Lichess public database. 

### Target Formulation
Our binary target variable is formulated as:

$$\text{is\_upset} = \begin{cases} 1 & \text{if the lower-rated player wins} \\ 0 & \text{if the higher-rated player wins or the match is a draw} \end{cases}$$

Due to the natural layout of competitive sports matching, this creates a distinct **minority class imbalance of 34.48%**. To ensure zero data leakage, data partitions are strictly split using a stratified 70/15/15 distribution before any downstream feature transformations:

* **Full Corpus:** 100,000 games (Normal: 65,519 | Upset: 34,481)
* **Training Partition (`X_train`):** 70,000 games (Normal: 45,863 | Upset: 24,137)
* **Validation Partition (`X_val`):** 15,000 games (Normal: 9,828 | Upset: 5,172)
* **Testing Partition (`X_test`):** 15,000 games (Normal: 9,828 | Upset: 5,172)

---

## Feature Engineering & Preprocessing
Raw move sequences (`white_moves`, `black_moves`) were dropped after extracting domain-specific tactical proxies calculated at a snapshot of match ply 24:
* **Tactical Heuristics:** `white_castled`/`black_castled` (king safety), `white_developed`/`black_developed` (quantified piece tempo), and `white_queen_moved`/`black_queen_moved` (early strategic rushing/tilt flags).
* **Environmental Context:** `game_elo` (mean rating tier), `base_time`, `increment`, and one-hot encoded `speed_category` structures.
* **High-Cardinality Target Encoding:** Dense text indicators (`opening_name`, `opening_eco`) were mapped into continuous probability space using a cross-validated **m-estimate smoothed target encoder ($m=20$)** calculated *strictly* inside the training split.

---

## Modeling Performance Matrix

To evaluate predictive performance, we established a rigorous benchmark comparing a standard linear baseline against our gradient-boosted decision trees (GBDTs). 

### Default vs. Optimized Threshold Performance

| Architecture Model | Threshold ($t$) | Global Accuracy | Target Precision | Target Recall | Balanced $F_1$-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 0.50 | 65.52% | 0.00% | 0.00% | 0.00% | 0.6455 |
| **Advanced LightGBM (Default)** | 0.50 | 64.91% | 47.10% | 14.31% | 21.95% | 0.6400 |
| **Advanced XGBoost (Default)** | 0.50 | 64.79% | 46.62% | 14.68% | 22.32% | 0.6383 |
| **LightGBM (Optimized Final)** | **0.2374** | **49.75%** | **39.75%** | **88.65%** | **54.88%** | **0.6400** |
| **XGBoost (Optimized Final)** | **0.2293** | **48.90%** | **39.45%** | **90.08%** | **54.87%** | **0.6383** |

### Key Insights
1. **Baseline Paralysis:** Restricted to a single metric (`abs_rating_diff`), the parametric linear baseline fails to clear the default 0.50 probability threshold for a minority class anomaly, predicting normal outcomes for 100% of cases and achieving a **0.00% target recall**.
2. **Threshold Shift:** Calibrating the decision boundaries down into the lower-bound quartile ($\sim 0.23$) effectively counteracts the minority class imbalance. This intentionally trades a fraction of global accuracy to lift target recall up to **$\sim 89-90\%$** on validation splits.
3. **Generalization Performance:** Evaluating our optimized champion LightGBM framework against the locked testing split achieved an unbiased **test $F_1$-score of 54.47%** and a **test recall of 88.71%**, proving robust generalization.

---

## Feature Ablation Analysis
A systematic ablation study was conducted by freezing the optimal threshold ($t=0.2374$) and stripping column subsets to track signal variances:

| Trial Configuration | Column Width | Optimized $F_1$-Score | Target Precision | Target Recall |
| :--- | :---: | :---: | :---: | :---: |
| **1. Full Feature Space** | 18 | 54.81% | 39.94% | 87.36% |
| **2. No Tactical Heuristics** | 12 | 54.78% | 39.80% | 87.82% |
| **3. No Environmental Controls** | 13 | 54.75% | 40.39% | 84.98% |
| **4. No Opening Information** | 16 | **55.43%** | **40.89%** | **86.00%** |

*Discovery:* Stripping the high-cardinality chess opening book data (**Trial 4**) resulted in a performance *increase* to **55.43%**. This proves that specific opening strings add high-dimensional noise, confirming that competitive chess chokes are fundamentally driven by active tactical blunders and time pressure rather than theoretical preparation.

---

## Operational Limitations
While optimizing the decision boundary mathematically maximizes the balanced $F_1$-score, a post-hoc deployment audit exposes a significant real-world constraint:

* **Alert Volume:** Out of 15,000 unseen testing matches, the model triggers an upset warning for **77.8% of all games** (11,675 total predictions).
* **False Positive Rate:** Within that pool of issued alerts, the system is incorrect **60.7% of the time** (7,087 false alarms).

In a live production setting (e.g., a streaming broadcast overlay flagging high-stakes chess matches), this behavior induces severe **user alarm fatigue**. Future iterations of this work will pivot away from a symmetric $F_1$-metric to optimize an $F_\beta$-score (specifically $F_{0.5}$), re-weighting the optimization loop to value precision twice as heavily as recall and minimizing system false alarms.