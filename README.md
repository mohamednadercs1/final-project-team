# 🎬 Movie Success Prediction

Predicting whether a movie will be a commercial success (revenue ≥ $100M) from its characteristics, TMDB metadata, and external critic reception.

## 🎯 Business Objective
Help studios evaluate new releases by estimating commercial success *before* a movie hits theaters.

**ML task:** Binary classification — `success = 1` if revenue ≥ $100M, else `0`.

## 📦 Data Sources
| Source | Method | What we used |
|---|---|---|
| TMDB 5000 Movies & Credits (Kaggle) | CSV | budget, revenue, genres, cast, crew, release date, runtime |
| TMDB API | API | extra metadata (e.g. collection/sequel info, certification) |
| Critic/audience scores (Metacritic / OMDb) | Web scraping | `critic_score`, `audience_score` |

## 🗂️ Repo Structure
```
├── data/
│   ├── raw/                  # original CSVs
│   ├── scraped_scores.csv    # scraping output
│   └── final_dataset.csv     # merged + engineered features
├── notebooks/                # cleaning, EDA
├── src/
│   ├── scraping.py
│   ├── cleaning.py
│   ├── features.py
│   └── modeling.py
├── results/
│   ├── metrics.csv
│   └── figures/
└── README.md
```

## 🔄 Pipeline
1. **Scraping** – critic & audience scores, cached, polite delays, missing movies handled.
2. **Cleaning** – parse JSON columns (`ast.literal_eval`), treat `budget=0` / `revenue=0` as missing, drop duplicates, fix dates.
3. **Feature engineering** – director, star count, crew size, release month/season, genre one-hot, sequel flag, `success` label.
4. **EDA** – success distribution, budget vs revenue, best genres/months, critic score vs success.
5. **Modeling** – Logistic Regression, Random Forest, XGBoost vs. a naive baseline.

## 🧪 Modeling Setup
- **Split:** time-based — train on movies released **before 2010**, test on **2010 and later** (more realistic than a random split).
- **Class imbalance:** `class_weight="balanced"` (LR/RF) and `scale_pos_weight` (XGBoost).
- **Tuning:** `RandomizedSearchCV` with 5-fold stratified CV, optimizing ROC-AUC (train set only).
- **Preprocessing inside `sklearn.Pipeline`** so imputation/scaling are fit only on training folds (no leakage).
- **Metrics:** Accuracy, Precision, Recall, F1, ROC-AUC. Accuracy alone is misleading with imbalanced classes.

### Three feature sets compared
| Set | Features | Purpose |
|---|---|---|
| **A – Pre-release** | budget, runtime, genres, month, cast/crew, director history, sequel flag | Realistic "before release" prediction |
| **B – Pre-release + Critics** | A + critic & audience scores | Do critics add predictive value? |
| **C – Post-release (leaky)** | B + `vote_count`, `vote_average`, `popularity` | Shows how data leakage inflates scores |

## 📊 Results
> Fill in after running `python modeling.py` on the real dataset (see `results/metrics.csv`).

| Feature set | Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| A | Baseline | | | | | |
| A | Logistic Regression | | | | | |
| A | Random Forest | | | | | |
| A | XGBoost | | | | | |
| B | … | | | | | |

**Key findings** (update with real numbers):
- Best model: _XGBoost / RF_ with ROC-AUC = _X_
- Adding critic scores improved ROC-AUC from _X_ to _Y_
- Top drivers: budget, sequel flag, director track record, ...
- Set C reaches near-perfect scores purely due to leakage — which is why it is **not** used as the final model.

Figures: `results/figures/` (ROC curves, confusion matrix, feature importance, SHAP, feature-set comparison).

## ⚠️ Assumptions & Limitations
- Critic scores may only be available shortly before/at release (review embargo) — Set B assumes they are.
- Revenue is **not inflation-adjusted**; $100M in 1995 ≠ $100M in 2015.
- Movies with missing budget/revenue were handled by flagging/imputation or removal, which may bias the sample.
- Scraped scores depend on title/ID matching quality; unmatched movies have missing scores.
- Dataset covers up to ~2016 only.

## ▶️ How to Run
```bash
pip install -r requirements.txt
python src/modeling.py
```
`requirements.txt`: `pandas numpy scikit-learn xgboost shap matplotlib seaborn`

## 👥 Team
| Role | Member |
|---|---|
| Web scraping | _Person 1_ |
| Data cleaning & parsing | _Person 2_ |
| Merging & feature engineering | _Person 3_ |
| EDA & visualization | _Person 4_ |
| Modeling, evaluation & README | _Person 5_ |
