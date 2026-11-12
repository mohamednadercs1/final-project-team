"""
Movie Success Prediction - Task 5: Modeling & Evaluation
Input : data/final_dataset.csv   (output of Person 3)
Output: results/metrics.csv, results/figures/*.png

Expected columns in final_dataset.csv (names can be changed in CONFIG):
  movie_id, title, release_year, revenue, success,
  budget, budget_missing, runtime, is_sequel, release_month, num_stars,
  crew_size, director_avg_revenue, genre_* (one-hot),
  critic_score, audience_score,
  vote_average, vote_count, popularity   (post-release / leaky)
"""
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,
                             accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ----------------------------- CONFIG ---------------------------------
DATA_PATH = "data/final_dataset.csv"
OUT_DIR = "results"
FIG_DIR = f"{OUT_DIR}/figures"
TARGET = "success"
YEAR_COL = "release_year"
SPLIT_YEAR = 2010            # train: < 2010 , test: >= 2010
RANDOM_STATE = 42

BASE_FEATURES = ["budget", "budget_missing", "runtime", "is_sequel",
                 "release_month", "num_stars", "crew_size",
                 "director_avg_revenue"]
CRITIC_FEATURES = ["critic_score", "audience_score"]
LEAKY_FEATURES = ["vote_average", "vote_count", "popularity"]
# ----------------------------------------------------------------------

os.makedirs(FIG_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")


def get_feature_sets(df):
    genres = [c for c in df.columns if c.startswith("genre_")]
    base = [c for c in BASE_FEATURES + genres if c in df.columns]
    critics = [c for c in CRITIC_FEATURES if c in df.columns]
    leaky = [c for c in LEAKY_FEATURES if c in df.columns]
    return {
        "A_pre_release": base,
        "B_pre_release+critics": base + critics,
        "C_post_release(leaky)": base + critics + leaky,
    }


def build_models(pos_weight):
    return {
        "Baseline (always fail)": (
            Pipeline([("m", DummyClassifier(strategy="most_frequent"))]), None),
        "Logistic Regression": (
            Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler()),
                      ("m", LogisticRegression(max_iter=2000,
                                               class_weight="balanced"))]),
            {"m__C": [0.01, 0.1, 1, 10]}),
        "Random Forest": (
            Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("m", RandomForestClassifier(class_weight="balanced",
                                                   random_state=RANDOM_STATE,
                                                   n_jobs=-1))]),
            {"m__n_estimators": [200, 400],
             "m__max_depth": [5, 8, 12, None],
             "m__min_samples_leaf": [1, 3, 5]}),
        "XGBoost": (
            Pipeline([("m", XGBClassifier(scale_pos_weight=pos_weight,
                                          eval_metric="logloss",
                                          random_state=RANDOM_STATE,
                                          n_jobs=-1))]),
            {"m__n_estimators": [200, 400],
             "m__max_depth": [3, 4, 6],
             "m__learning_rate": [0.03, 0.05, 0.1],
             "m__subsample": [0.8, 1.0],
             "m__colsample_bytree": [0.8, 1.0]}),
    }


def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, proba),
    }


def main():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=[TARGET, YEAR_COL]).copy()
    train_df = df[df[YEAR_COL] < SPLIT_YEAR]
    test_df = df[df[YEAR_COL] >= SPLIT_YEAR]
    y_train, y_test = train_df[TARGET].astype(int), test_df[TARGET].astype(int)
    print(f"Train: {len(train_df)} ({y_train.mean():.1%} success) | "
          f"Test: {len(test_df)} ({y_test.mean():.1%} success)")

    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    rows, fitted = [], {}
    for fs_name, feats in get_feature_sets(df).items():
        X_train, X_test = train_df[feats], test_df[feats]
        for m_name, (pipe, grid) in build_models(pos_weight).items():
            if grid is None:
                best = pipe.fit(X_train, y_train)
            else:
                search = RandomizedSearchCV(
                    pipe, grid, n_iter=10, scoring="roc_auc", cv=cv,
                    random_state=RANDOM_STATE, n_jobs=-1)
                best = search.fit(X_train, y_train).best_estimator_
            res = evaluate(best, X_test, y_test)
            rows.append({"Feature set": fs_name, "Model": m_name, **res})
            fitted[(fs_name, m_name)] = (best, feats)
            print(f"[{fs_name}] {m_name}: F1={res['F1']:.3f} "
                  f"AUC={res['ROC-AUC']:.3f}")

    results = pd.DataFrame(rows).round(3)
    results.to_csv(f"{OUT_DIR}/metrics.csv", index=False)
    print("\n", results.to_string(index=False))

    # ---- 1) Do critic scores help? (AUC comparison, tuned models) ----
    plt.figure(figsize=(9, 5))
    sns.barplot(data=results[results.Model != "Baseline (always fail)"],
                x="Model", y="ROC-AUC", hue="Feature set")
    plt.ylim(0.5, 1.0)
    plt.title("Does critic reception improve prediction? (ROC-AUC, test set)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/featureset_comparison.png", dpi=200)
    plt.close()

    # ---- 2) Pick best honest model: best AUC on B (pre-release+critics) ----
    honest = results[results["Feature set"] == "B_pre_release+critics"]
    best_name = honest.sort_values("ROC-AUC", ascending=False).iloc[0]["Model"]
    best_model, feats = fitted[("B_pre_release+critics", best_name)]
    X_test = test_df[feats]
    print(f"\nBest model (set B): {best_name}")

    # ROC curves for all models in set B
    fig, ax = plt.subplots(figsize=(7, 6))
    for m_name in ["Logistic Regression", "Random Forest", "XGBoost"]:
        mdl, f = fitted[("B_pre_release+critics", m_name)]
        RocCurveDisplay.from_estimator(mdl, test_df[f], y_test, name=m_name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_title("ROC Curves (pre-release + critic features)")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/roc_curves.png", dpi=200)
    plt.close()

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_estimator(
        best_model, X_test, y_test, display_labels=["Flop", "Success"],
        cmap="Blues", ax=ax)
    ax.set_title(f"Confusion Matrix - {best_name}")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/confusion_matrix.png", dpi=200)
    plt.close()

    # ---- 3) Feature importance ----
    final = best_model.named_steps["m"]
    if hasattr(final, "feature_importances_"):
        imp = pd.Series(final.feature_importances_, index=feats)
    else:
        imp = pd.Series(np.abs(final.coef_[0]), index=feats)
    imp = imp.sort_values(ascending=False).head(15)[::-1]
    plt.figure(figsize=(8, 6))
    imp.plot(kind="barh", color="#e50914")
    plt.title(f"Top 15 Features - {best_name}")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/feature_importance.png", dpi=200)
    plt.close()

    # ---- 4) SHAP (tree models only) ----
    if best_name in ("Random Forest", "XGBoost"):
        try:
            import shap
            Xt = X_test.copy()
            if "imp" in best_model.named_steps:
                Xt = pd.DataFrame(best_model.named_steps["imp"].transform(Xt),
                                  columns=feats)
            sv = shap.TreeExplainer(final).shap_values(Xt)
            if isinstance(sv, list):          # older RF API
                sv = sv[1]
            elif getattr(sv, "ndim", 2) == 3:  # newer RF API
                sv = sv[:, :, 1]
            shap.summary_plot(sv, Xt, show=False, max_display=15)
            plt.tight_layout()
            plt.savefig(f"{FIG_DIR}/shap_summary.png", dpi=200,
                        bbox_inches="tight")
            plt.close()
        except ImportError:
            print("shap not installed -> skipping SHAP plot")

    print(f"\nDone. See {OUT_DIR}/metrics.csv and {FIG_DIR}/")


if __name__ == "__main__":
    main()
