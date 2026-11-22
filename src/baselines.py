"""Step 6 - Baselines, ablations and leakage-gap sensitivity.

Answers three questions the main results table cannot:
  1. How much of the performance comes from the budget alone?            (feature-set ablation)
  2. Do the director / cast / studio history features actually help?      (paired bootstrap CI)
  3. How much does the "greenlight gap" inflate the history features?     (lag sweep)

All models are scored on the same temporal test set; every PR-AUC has a 95% bootstrap CI and every
"full minus ablation" difference has a paired-bootstrap CI (if the CI contains 0 -> no evidence).

Run:
    python baselines.py                          # ROI target, LogReg + XGBoost
    python baselines.py --target revenue
    python baselines.py --models "Logistic Regression" LightGBM --lags 365 545 730
    python baselines.py --skip-lag               # ablation only
"""
import argparse
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

import config
from features import HISTORY_COLS, FeatureBuilder, temporal_split
from history_lagged import add_history_features_lagged
from train_models import TARGET_COLS, make_models, paired_bootstrap_delta


# ------------------------------------------------------------------ helpers
def bootstrap_ci(y, p, n=1000, seed=config.SEED):
    """95% percentile CI of PR-AUC over resampled test films."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() != y[idx].max():
            vals.append(average_precision_score(y[idx], p[idx]))
    return tuple(np.percentile(vals, [2.5, 97.5]))


def fit_predict(model, Xtr, ytr, Xte):
    return clone(model).fit(Xtr, ytr).predict_proba(Xte)[:, 1]


def forward_chain_pr_auc(model, X, y, n_splits=4):
    """Mean PR-AUC of an expanding-window (time-respecting) CV on the training period."""
    scores = []
    for tr, va in TimeSeriesSplit(n_splits).split(X):          # X is already sorted by release date
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
            continue
        p = clone(model).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[va])[:, 1]
        scores.append(average_precision_score(y[va], p))
    return float(np.mean(scores)) if scores else np.nan


def feature_sets(cols):
    hist = [c for c in cols if c in HISTORY_COLS]
    genre = [c for c in cols if c.startswith("genre_")]
    return {
        "budget only": ["log_budget"],
        "budget + genre": ["log_budget", "log_budget_to_genre_ratio", *genre],
        "full - history": [c for c in cols if c not in hist],
        "history only": hist,
        "full greenlight": list(cols),
    }


def significance(lo, hi):
    return "yes" if lo > 0 else ("NO (worse)" if hi < 0 else "no (CI includes 0)")


# ------------------------------------------------------------------ 1+2: ablation
def run_ablation(train, test, y_col, model_names, n_boot):
    builder = FeatureBuilder().fit(train)
    Xtr, Xte = builder.transform(train), builder.transform(test)
    ytr, yte = train[y_col].to_numpy(), test[y_col].to_numpy()
    sets = feature_sets(builder.greenlight_features_)
    models = make_models(ytr)
    base = float(yte.mean())

    rows = [{"model": "(always predict success)", "features": "-", "n_feat": 0, "TimeCV PR-AUC": np.nan,
             "Test PR-AUC": base, "PR-AUC 95% CI": "-", "Test ROC-AUC": 0.5,
             "full - this [95% CI]": "-", "full better?": "-"}]
    for mname in model_names:
        preds = {}
        for sname, cols in sets.items():
            preds[sname] = fit_predict(models[mname], Xtr[cols], ytr, Xte[cols])
        for sname, cols in sets.items():
            p = preds[sname]
            lo, hi = bootstrap_ci(yte, p, n_boot)
            row = {"model": mname, "features": sname, "n_feat": len(cols),
                   "TimeCV PR-AUC": forward_chain_pr_auc(models[mname], Xtr[cols], ytr),
                   "Test PR-AUC": average_precision_score(yte, p),
                   "PR-AUC 95% CI": f"[{lo:.3f}, {hi:.3f}]", "Test ROC-AUC": roc_auc_score(yte, p),
                   "full - this [95% CI]": "-", "full better?": "-"}
            if sname != "full greenlight":
                dlo, dhi = paired_bootstrap_delta(yte, p, preds["full greenlight"], n=n_boot)
                d = average_precision_score(yte, preds["full greenlight"]) - average_precision_score(yte, p)
                row["full - this [95% CI]"] = f"{d:+.3f} [{dlo:+.3f}, {dhi:+.3f}]"
                row["full better?"] = significance(dlo, dhi)
            rows.append(row)
    return pd.DataFrame(rows).set_index(["model", "features"]), base


def plot_ablation(res, base, path):
    r = res.drop(index="(always predict success)", level=0)
    labels = [f"{m} | {f}" for m, f in r.index]
    pr = r["Test PR-AUC"].to_numpy(float)
    ci = np.array([[float(x) for x in s.strip("[]").split(",")] for s in r["PR-AUC 95% CI"]])
    fig, ax = plt.subplots(figsize=(8, 0.38 * len(r) + 1.5))
    ypos = np.arange(len(r))
    ax.barh(ypos, pr, xerr=[pr - ci[:, 0], ci[:, 1] - pr], color="#4c78a8", capsize=3)
    ax.axvline(base, color="grey", ls="--", lw=1, label=f"always-success baseline ({base:.2f})")
    ax.set_yticks(ypos, labels, fontsize=8); ax.invert_yaxis()
    ax.set_xlim(max(0, min(ci[:, 0].min(), base) - 0.05), ci[:, 1].max() + 0.03)
    ax.set(xlabel="Test PR-AUC (95% bootstrap CI)", title="Feature-set ablation"); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


# ------------------------------------------------------------------ 3: greenlight-gap sensitivity
def run_lag_sweep(df, lags, y_col, model_names, n_boot):
    """Re-compute the history features with a release-date lag and compare the full greenlight model.

    The test films are identical in every row (same chronological split), so the comparison with the
    current (lag-free) features is paired.
    """
    no_hist = df.drop(columns=[c for c in HISTORY_COLS if c in df.columns])
    variants = [("current (no lag)", df)] + [(f"lag {lag} d", add_history_features_lagged(no_hist, lag)) for lag in lags]
    out, ref = [], {}
    for label, frame in variants:
        train, test = temporal_split(frame)
        builder = FeatureBuilder().fit(train)
        Xtr, Xte = builder.transform(train), builder.transform(test)
        gl = builder.greenlight_features_
        ytr, yte = train[y_col].to_numpy(), test[y_col].to_numpy()
        models = make_models(ytr)
        for mname in model_names:
            p = fit_predict(models[mname], Xtr[gl], ytr, Xte[gl])
            lo, hi = bootstrap_ci(yte, p, n_boot)
            row = {"model": mname, "history": label, "Test PR-AUC": average_precision_score(yte, p),
                   "PR-AUC 95% CI": f"[{lo:.3f}, {hi:.3f}]", "Test ROC-AUC": roc_auc_score(yte, p),
                   "lag - current [95% CI]": "-"}
            if label.startswith("current"):
                ref[mname] = (yte, p)
            else:
                y0, p0 = ref[mname]
                dlo, dhi = paired_bootstrap_delta(y0, p0, p, n=n_boot)
                row["lag - current [95% CI]"] = (f"{average_precision_score(yte, p) - average_precision_score(y0, p0):+.3f} "
                                                 f"[{dlo:+.3f}, {dhi:+.3f}]")
            out.append(row)
    return pd.DataFrame(out).set_index(["model", "history"])


# ------------------------------------------------------------------ main
def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target", choices=["roi", "revenue"], default="roi")
    ap.add_argument("--models", nargs="+", default=["Logistic Regression", "XGBoost"])
    ap.add_argument("--lags", nargs="+", type=int, default=[365, 545, 730], help="history lags in days")
    ap.add_argument("--boot", type=int, default=1000, help="bootstrap resamples")
    ap.add_argument("--skip-lag", action="store_true")
    args = ap.parse_args()

    out_dir = config.REPORTS_DIR; out_dir.mkdir(exist_ok=True)
    df = pd.read_pickle(config.MODELING_FRAME)
    y_col = TARGET_COLS[args.target]
    train, test = temporal_split(df)
    print(f"Target {y_col}: train {len(train)} / test {len(test)} films, test base rate {test[y_col].mean():.3f}")

    res, base = run_ablation(train, test, y_col, args.models, args.boot)
    print("\n== Feature-set ablation (temporal test) ==")
    print(res.round(3).to_string())
    res.round(4).to_csv(out_dir / f"ablation_{args.target}.csv")
    plot_ablation(res, base, out_dir / f"ablation_{args.target}.png")

    if not args.skip_lag:
        lag = run_lag_sweep(df, args.lags, y_col, args.models, args.boot)
        print("\n== Greenlight-gap sensitivity: history features computed with a release-date lag ==")
        print(lag.round(3).to_string())
        print("\nIf 'lag - current' is ~0 with a CI around 0, the original leakage was harmless; "
              "if it is clearly negative, the old numbers were inflated.")
        lag.round(4).to_csv(out_dir / f"lag_sweep_{args.target}.csv")
    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
