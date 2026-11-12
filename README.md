# 🎬 Movie Success Prediction

Predicting whether a movie will be a commercial success (revenue ≥ $100M) from its characteristics, TMDB metadata, external datasets, and scraped critic reception.

## 🎯 Business Objective
Help studios evaluate new releases by estimating commercial success *before* a movie hits theaters.

**ML task:** Binary classification — `success = 1` if revenue ≥ $100M, else `0`.

## 📦 Data Sources
| Source | Method | Used for |
|---|---|---|
| TMDB 5000 Movies & Credits (Kaggle) | CSV | budget, revenue, genres, cast, crew, release date, runtime |
| External datasets (IMDb / MovieLens) | Download + matching | extra ratings / metadata |
| Critic & audience scores | Web scraping (ethical, rate-limited) | `critic_score`, `audience_score` |

## 🗂️ Repo Structure
```
├── data/
│   ├── raw/                      # original CSVs
│   └── final_dataset.csv         # merged + engineered features (Member 4)
├── notebooks/
│   ├── 01_cleaning_eda.ipynb             # Member 1
│   ├── 02_external_data.ipynb            # Member 2
│   ├── 03_scraping_validation.ipynb      # Member 3
│   ├── 04_feature_engineering.ipynb      # Member 4
│   └── 05_modeling_evaluation.ipynb      # Member 5
├── scripts/
│   ├── load_external_data.py     # Member 2
│   ├── scrape_scores.py          # Member 3
│   └── merge_data.py             # Member 4
├── src/
│   ├── modeling.py               # Member 5: train / tune / evaluate
│   └── make_report.py            # Member 5: fills report + README from results
├── report/                       # one section per member + final_report.md
├── results/                      # metrics.csv, summary.json, figures/
├── docs/linkedin_post.md
└── requirements.txt
```

## 🔄 Pipeline
1. **Cleaning & EDA** – parse JSON columns, treat `budget=0` / `revenue=0` as missing, drop duplicates, fix dates.
2. **External data** – match IMDb / MovieLens records to TMDB 5000.
3. **Scraping** – critic & audience scores (robots.txt respected, delays, cache).
4. **Features & merging** – director, star count, crew size, release month/season, genre one-hot, sequel flag, `success` label → `data/final_dataset.csv`.
5. **Modeling** – Logistic Regression, Random Forest, XGBoost vs. a naive baseline.

## 🧪 Modeling Setup
- **Split:** time-based — train on movies released **before 2010**, test on **2010 and later** (more realistic than a random split).
- **Class imbalance:** `class_weight="balanced"` (LR/RF) and `scale_pos_weight` (XGBoost).
- **Tuning:** `RandomizedSearchCV`, 5-fold stratified CV, optimizing ROC-AUC (training data only).
- **Preprocessing inside `sklearn.Pipeline`** so imputation/scaling are fit only on training folds.
- **Metrics:** Accuracy, Precision, Recall, F1, ROC-AUC (accuracy alone is misleading with imbalanced classes).

### Three feature sets compared
| Set | Features | Purpose |
|---|---|---|
| **A – Pre-release** | budget, runtime, genres, month, cast/crew, director history, sequel flag | Realistic "before release" prediction |
| **B – Pre-release + Critics** | A + critic & audience scores | Do critics add predictive value? |
| **C – Post-release (leaky)** | B + `vote_count`, `vote_average`, `popularity` | Shows how data leakage inflates scores |

## 📊 Results
<!-- RESULTS:START -->
_Results will appear here after running:_
```bash
python src/modeling.py
python src/make_report.py
```
<!-- RESULTS:END -->

Figures are saved in `results/figures/` (ROC curves, confusion matrix, feature importance, SHAP, feature-set comparison).

## ⚠️ Assumptions & Limitations
- Critic scores may only be available shortly before/at release (review embargo) — Set B assumes they are.
- Revenue is **not inflation-adjusted**; $100M in 1995 ≠ $100M in 2015.
- Missing budget/revenue handled by flagging, imputation, or removal, which may bias the sample.
- Scraped scores depend on title/ID matching quality; unmatched movies have missing scores.
- Dataset covers movies up to ~2016 only; the test period is therefore small.

## ▶️ How to Run
```bash
pip install -r requirements.txt
python src/modeling.py        # needs data/final_dataset.csv
python src/make_report.py     # fills report/ and README with the real numbers
```
Or open `notebooks/05_modeling_evaluation.ipynb`.

## 👥 Team
| Member | Role |
|---|---|
| 1 | Data cleaning & EDA |
| 2 | External datasets integration |
| 3 | Web scraping & ethics |
| 4 | Feature engineering & merging |
| 5 | Modeling, evaluation, final report & README |
