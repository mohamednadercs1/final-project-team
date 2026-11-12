# Modeling, Results & Conclusion

## 1. Objective
Predict whether a movie earns at least $100M in revenue, using only information that is realistically available before release, and test whether external critic reception adds predictive value.

## 2. Methodology
- **Target:** `success = 1` if revenue ≥ $100M, otherwise 0. The classes are imbalanced (successes are the minority).
- **Time-based split:** models are trained on movies released before 2010 and tested on movies from 2010 onward. This mimics real use (predicting future movies from past ones) and avoids the optimistic bias of a random split.
- **Models:** a naive baseline (always predict "flop"), Logistic Regression, Random Forest and XGBoost.
- **Imbalance handling:** `class_weight="balanced"` for Logistic Regression and Random Forest, `scale_pos_weight` for XGBoost.
- **Tuning:** `RandomizedSearchCV` (10 iterations, 5-fold stratified CV, scoring = ROC-AUC) on the training set only. Imputation and scaling live inside a `sklearn.Pipeline`, so they are fitted on training folds only.
- **Metrics:** Accuracy, Precision, Recall, F1 and ROC-AUC on the held-out test set. Accuracy alone is misleading because a model that always predicts "flop" already scores well.
- **Three feature sets:**
  - **A – Pre-release:** budget, runtime, genres, release month, cast/crew features, director track record, sequel flag.
  - **B – Pre-release + critics:** A plus scraped critic and audience scores.
  - **C – Post-release (leaky):** B plus `vote_count`, `vote_average`, `popularity`. These are only known after release and are included to *demonstrate* data leakage, not as a deployable model.

## 3. Results
<!-- AUTO:START -->
_This section is generated automatically. After `data/final_dataset.csv` exists, run:_
```bash
python src/modeling.py
python src/make_report.py
```
<!-- AUTO:END -->

## 4. Limitations
- Critic scores may only exist close to release; Set B assumes they are available beforehand.
- Revenue is not adjusted for inflation, so the $100M threshold is easier to reach in later years.
- The test period (2010 onward) contains relatively few movies, so metrics have noticeable variance.
- Scraped scores depend on title/ID matching quality; unmatched movies have missing values (median-imputed).
- Director track-record features must only use *earlier* movies; any slip would leak future information.

## 5. Conclusion & Recommendations
A studio can use the pre-release model (Set A/B) to screen projects early, focusing on the strongest drivers listed above. Post-release signals (votes, popularity) should never be used for pre-release decisions, because they leak the outcome. Future work: inflation-adjusted revenue or ROI as the target, richer cast popularity features, and probability calibration for budget decisions.
