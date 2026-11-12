"""
Movie Success Prediction - Task 5 (Member 5): Modeling & Evaluation

Run from anywhere:  python src/modeling.py
Input : data/final_dataset.csv   (produced by Member 4, scripts/merge_data.py)
Output: results/metrics.csv, results/summary.json, results/figures/*.png

Column names are auto-mapped (see ALIASES) and missing columns are reported,
so small naming differences in final_dataset.csv will not break the pipeline.
"""
import json
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,
                             accuracy_score, classification_report, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ----------------------------- CONFIG ---------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "final_dataset.csv"
OUT_DIR = ROOT / "results"
FIG_DIR = OUT_DIR / "figures"
TARGET = "success"
YEAR_COL = "release_year"
SPLIT_YEAR = 2010            # train: < 2010 , test: >= 2010
SUCCESS_THRESHOLD = 100_000_000
RANDOM_STATE = 42
QUICK = bool(os.environ.get("QUICK"))   # QUICK=1 -> fast smoke test (less tuning)
N_ITER = 2 if QUICK else 10

BASE_FEATURES = ["budget", "budget_missing", "runtime", "is_sequel",
                 "release_month", "num_stars", "crew_size",
                 "director_avg_revenue"]
CRITIC_FEATURES = ["critic_score", "audience_score"]
LEAKY_FEATURES = ["vote_average", "vote_count", "popularity"]

# alternative column names -> standard name
ALIASES = {
    "release_year": ["year", "release_yr", "releaseyear"],
    "release_month": ["month"],
    "success": ["is_success", "target", "label"],
    "critic_score": ["metascore", "critic", "tomatometer", "metacritic_score"],
    "audience_score": ["audience", "user_score", "imdb_rating"],
    "is_sequel": ["is_franchise", "in_collection", "has_collection"],
    "num_stars": ["n_stars", "star_count", "famous_actors"],
    "crew_size": ["n_crew", "crew_count"],
    "director_avg_revenue": ["director_mean_revenue", "director_avg_rev"],
    "budget_missing": ["budget_is_missing", "budget_missing_flag"],
}
# ----------------------------------------------------------------------

sns.set_theme(style="whitegrid")
FS_A, FS_B, FS_C = "A_pre_release", "B_pre_release+critics", "C_post_release(leaky)"
BASELINE = "Baseline (always fail)"


def load_data(path=DATA_PATH):
    """Load final_dataset.csv, standardise column names, derive what is missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Member 4 must upload data/final_dataset.csv "
            "(output of scripts/merge_data.py) before modeling can run.")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    for std, alts in ALIASES.items():
        if std not in df.columns:
            for a in alts:
                if a in df.columns:
                    df = df.rename(columns={a: std})
                    break
    if "release_date" in df.columns:
        d = pd.to_datetime(df["release_date"], errors="coerce")
        if YEAR_COL not in df.columns:
            df[YEAR_COL] = d.dt.year
        if "release_month" not in df.columns:
            df["release_month"] = d.dt.month
    if TARGET not in df.columns and "revenue" in df.columns:
        df[TARGET] = (df["revenue"] >= SUCCESS_THRESHOLD).astype(int)
    if "budget_missing" not in df.columns and "budget" in df.columns:
        df["budget_missing"] = (df["budget"].isna() | (df["budget"] == 0)).astype(int)
    for c in (TARGET, YEAR_COL):
        if c not in df.columns:
            raise ValueError(f"Required column '{c}' not found. Columns: {list(df.columns)}")
    df = df.dropna(subset=[TARGET, YEAR_COL]).copy()
    df[TARGET] = df[TARGET].astype(int)
    return df


def get_feature_sets(df):
    genres = [c for c in df.columns if c.startswith("genre_")]
    base = [c for c in BASE_FEATURES + genres if c in df.columns]
    critics = [c for c in CRITIC_FEATURES if c in df.columns]
    leaky = [c for c in LEAKY_FEATURES if c in df.columns]
    missing = [c for c in BASE_FEATURES + CRITIC_FEATURES + LEAKY_FEATURES
               if c not in df.columns]
    if missing:
        print(f"[warning] expected columns not found (skipped): {missing}")
    if len(base) < 3:
        raise ValueError("Fewer than 3 base features found - check column names.")
    sets = {FS_A: base}
    if critics:
        sets[FS_B] = base + critics
    if leaky:
        sets[FS_C] = base + critics + leaky
    return sets


def split_by_year(df):
    train, test = df[df[YEAR_COL] < SPLIT_YEAR], df[df[YEAR_COL] >= SPLIT_YEAR]
    if train[TARGET].nunique() < 2 or test[TARGET].nunique() < 2:
        raise ValueError("Train or test set has a single class - adjust SPLIT_YEAR.")
    return train, test


def build_models(pos_weight):
    return {
        BASELINE: (Pipeline([("m", DummyClassifier(strategy="most_frequent"))]), None),
        "Logistic Regression": (
            Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler()),
                      ("m", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
            {"m__C": [0.01, 0.1, 1, 10]}),
        "Random Forest": (
            Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("m", RandomForestClassifier(class_weight="balanced",
                                                   random_state=RANDOM_STATE, n_jobs=-1))]),
            {"m__n_estimators": [200, 400], "m__max_depth": [5, 8, 12, None],
             "m__min_samples_leaf": [1, 3, 5]}),
        "XGBoost": (
            Pipeline([("m", XGBClassifier(scale_pos_weight=pos_weight, eval_metric="logloss",
                                          random_state=RANDOM_STATE, n_jobs=-1))]),
            {"m__n_estimators": [200, 400], "m__max_depth": [3, 4, 6],
             "m__learning_rate": [0.03, 0.05, 0.1], "m__subsample": [0.8, 1.0],
             "m__colsample_bytree": [0.8, 1.0]}),
    }


def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {"Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1": f1_score(y_test, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, proba)}


def run_experiments(train_df, test_df, feature_sets, verbose=True):
    """Train + tune every model on every feature set. Returns (results_df, fitted)."""
    y_train, y_test = train_df[TARGET], test_df[TARGET]
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows, fitted = [], {}
    for fs_name, feats in feature_sets.items():
        for m_name, (pipe, grid) in build_models(pos_weight).items():
            if grid is None:
                best = pipe.fit(train_df[feats], y_train)
            else:
                search = RandomizedSearchCV(pipe, grid, n_iter=N_ITER, scoring="roc_auc",
                                            cv=cv, random_state=RANDOM_STATE, n_jobs=-1)
                best = search.fit(train_df[feats], y_train).best_estimator_
            res = evaluate(best, test_df[feats], y_test)
            rows.append({"Feature set": fs_name, "Model": m_name, **res})
            fitted[(fs_name, m_name)] = (best, feats)
            if verbose:
                print(f"[{fs_name}] {m_name}: F1={res['F1']:.3f} AUC={res['ROC-AUC']:.3f}")
    return pd.DataFrame(rows).round(3), fitted


def pick_best(results, fitted):
    """Best model by ROC-AUC on the most complete *honest* set (B if it exists, else A)."""
    fs = FS_B if any(k[0] == FS_B for k in fitted) else FS_A
    sub = results[(results["Feature set"] == fs) & (results.Model != BASELINE)]
    name = sub.sort_values("ROC-AUC", ascending=False).iloc[0]["Model"]
    return fs, name


def get_importances(model, feats):
    final = model.named_steps["m"]
    if hasattr(final, "feature_importances_"):
        return pd.Series(final.feature_importances_, index=feats).sort_values(ascending=False)
    return pd.Series(np.abs(final.coef_[0]), index=feats).sort_values(ascending=False)


def _finish(name, show, save=True):
    plt.tight_layout()
    if save and name:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(FIG_DIR / name, dpi=200, bbox_inches="tight")
    plt.show() if show else plt.close()


def make_plots(results, fitted, test_df, show=False, save=True):
    """Draw all figures. show=True displays inline (notebook); save=True writes PNGs."""
    y_test = test_df[TARGET]
    fs, best_name = pick_best(results, fitted)
    best_model, feats = fitted[(fs, best_name)]

    plt.figure(figsize=(9, 5))
    sns.barplot(data=results[results.Model != BASELINE], x="Model", y="ROC-AUC", hue="Feature set")
    plt.ylim(0.5, 1.0)
    plt.xticks(rotation=15)
    plt.title("Does critic reception improve prediction? (ROC-AUC, test set)")
    _finish("featureset_comparison.png", show, save)

    fig, ax = plt.subplots(figsize=(7, 6))
    for m in ["Logistic Regression", "Random Forest", "XGBoost"]:
        mdl, f = fitted[(fs, m)]
        RocCurveDisplay.from_estimator(mdl, test_df[f], y_test, name=m, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_title(f"ROC Curves ({fs})")
    _finish("roc_curves.png", show, save)

    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_estimator(best_model, test_df[feats], y_test,
                                          display_labels=["Flop", "Success"], cmap="Blues", ax=ax)
    ax.set_title(f"Confusion Matrix - {best_name}")
    _finish("confusion_matrix.png", show, save)

    imp = get_importances(best_model, feats).head(15)[::-1]
    plt.figure(figsize=(8, 6))
    imp.plot(kind="barh", color="#e50914")
    plt.title(f"Top 15 Features - {best_name}")
    _finish("feature_importance.png", show, save)

    if best_name in ("Random Forest", "XGBoost"):
        try:
            import shap
            Xt = test_df[feats].copy()
            if "imp" in best_model.named_steps:
                Xt = pd.DataFrame(best_model.named_steps["imp"].transform(Xt), columns=feats)
            sv = shap.TreeExplainer(best_model.named_steps["m"]).shap_values(Xt)
            if isinstance(sv, list):
                sv = sv[1]
            elif getattr(sv, "ndim", 2) == 3:
                sv = sv[:, :, 1]
            shap.summary_plot(sv, Xt, show=False, max_display=15)
            _finish("shap_summary.png", show, save)
        except ImportError:
            print("shap not installed -> skipping SHAP plot")


def save_outputs(df, train_df, test_df, sets, results, fitted, show_plots=False):
    """Write metrics.csv, summary.json and figures from already-trained models."""
    OUT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_DIR / "metrics.csv", index=False)
    fs, best_name = pick_best(results, fitted)
    best_model, feats = fitted[(fs, best_name)]
    make_plots(results, fitted, test_df, show=show_plots, save=True)
    summary = {
        "n_total": int(len(df)), "n_train": int(len(train_df)), "n_test": int(len(test_df)),
        "train_success_rate": round(float(train_df[TARGET].mean()), 3),
        "test_success_rate": round(float(test_df[TARGET].mean()), 3),
        "split_year": SPLIT_YEAR, "threshold": SUCCESS_THRESHOLD,
        "best_model": best_name, "best_feature_set": fs,
        "top_features": list(get_importances(best_model, feats).head(8).index),
        "feature_counts": {k: len(v) for k, v in sets.items()},
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))


def main():
    df = load_data()
    train_df, test_df = split_by_year(df)
    print(f"Train: {len(train_df)} ({train_df[TARGET].mean():.1%} success) | "
          f"Test: {len(test_df)} ({test_df[TARGET].mean():.1%} success)")
    sets = get_feature_sets(df)
    results, fitted = run_experiments(train_df, test_df, sets)
    print("\n", results.to_string(index=False))
    fs, best_name = pick_best(results, fitted)
    best_model, feats = fitted[(fs, best_name)]
    print(f"\nBest model ({fs}): {best_name}")
    print(classification_report(test_df[TARGET], best_model.predict(test_df[feats]),
                                target_names=["Flop", "Success"]))
    save_outputs(df, train_df, test_df, sets, results, fitted)
    print(f"\nDone. See {OUT_DIR}/metrics.csv, summary.json and {FIG_DIR}/")


if __name__ == "__main__":
    main()
