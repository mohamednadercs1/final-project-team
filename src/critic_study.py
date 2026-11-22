"""Step 7 - Do critic scores add predictive value beyond greenlight information?

Improves on the quick comparison inside train_models.py:
  * prints the coverage of critic scores (overall / train / test) and checks whether the covered
    films differ from the uncovered ones (selection bias from a partial scrape),
  * compares, on the SAME covered test films:
        greenlight (trained on all train films)       <- fair, no sample-size handicap
        greenlight (trained on covered train films)   <- same training sample as the critic model
        + critic score
        budget + critic score
        critic score only
  * reports PR-AUC / ROC-AUC with 95% bootstrap CIs and a paired CI of (+critic) - (greenlight),
  * refuses to draw conclusions when the test sample is too small.

Caveat to keep in the report: OMDb / Rotten Tomatoes scores fetched today are a snapshot taken
AFTER release, so this is a "post-screening" model, not a pre-release one.

Run:
    python critic_study.py                       # ROI target
    python critic_study.py --target revenue --models "Logistic Regression" LightGBM XGBoost
"""
import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score

import config
from baselines import bootstrap_ci, significance
from features import FeatureBuilder, temporal_split
from train_models import TARGET_COLS, make_models, paired_bootstrap_delta

MIN_TRAIN, MIN_TEST = 150, 100


def coverage_report(df, train, test, y_col):
    cov = df["critic_score"].notna()
    print("== Critic-score coverage ==")
    print(f"all films : {cov.mean():.0%} of {len(df)}")
    print(f"train     : {train['critic_score'].notna().sum()} of {len(train)}")
    print(f"test      : {test['critic_score'].notna().sum()} of {len(test)}")
    by_year = df.assign(year=df["release_date"].dt.year // 5 * 5).groupby("year")["critic_score"].apply(lambda s: s.notna().mean())
    print("coverage by 5-year bucket:", {int(k): f"{v:.0%}" for k, v in by_year.items()})

    t = test.assign(covered=test["critic_score"].notna())
    if t["covered"].nunique() == 2:
        cmp = t.groupby("covered").agg(films=("id", "size"), hit_rate=(y_col, "mean"),
                                       median_budget_musd=("budget", lambda s: s.median() / 1e6))
        print("\nTest films with vs without a critic score (selection-bias check):")
        print(cmp.round(3).to_string())
        gap = abs(cmp.loc[True, "hit_rate"] - cmp.loc[False, "hit_rate"])
        if gap > 0.08:
            print(f"WARNING: hit rate differs by {gap:.2f} between covered and uncovered films -> "
                  "scrape more (or scrape randomly) before trusting the comparison.")


def run(df, y_col, model_names, n_boot):
    if "critic_score" not in df or df["critic_score"].notna().sum() == 0:
        print("No critic_score column / no scores yet. Run scraper.py and rebuild the frame (features.py).")
        return None
    train, test = temporal_split(df)
    coverage_report(df, train, test, y_col)
    mtr, mte = train["critic_score"].notna().to_numpy(), test["critic_score"].notna().to_numpy()
    if mtr.sum() < MIN_TRAIN or mte.sum() < MIN_TEST:
        print(f"\nNot enough covered films yet (train {mtr.sum()}/{MIN_TRAIN}, test {mte.sum()}/{MIN_TEST}). "
              "Keep scraping - prioritise the most recent films (the test period) first.")
        return None

    builder = FeatureBuilder().fit(train)
    Xtr, Xte = builder.transform(train), builder.transform(test)
    gl = builder.greenlight_features_
    ytr, yte = train[y_col].to_numpy(), test[y_col].to_numpy()
    y_a, y_b = ytr[mtr], yte[mte]
    if len(np.unique(y_b)) < 2:
        print("Covered test films contain one class only; cannot evaluate.")
        return None
    models = make_models(ytr)
    Xte_c = Xte.loc[mte]
    print(f"\n== Critic comparison on the same {int(mte.sum())} covered test films "
          f"(base rate {y_b.mean():.2f}; train subset {int(mtr.sum())}) ==")

    def predict(mname, cols, subset_train):
        Xa, ya = (Xtr.loc[mtr, cols], y_a) if subset_train else (Xtr[cols], ytr)
        return clone(models[mname]).fit(Xa, ya).predict_proba(Xte_c[cols])[:, 1]

    rows = []
    for mname in model_names:
        variants = {
            "greenlight (all train)": predict(mname, gl, False),
            "greenlight (covered train)": predict(mname, gl, True),
            "greenlight + critic": predict(mname, gl + ["critic_score"], True),
            "budget + critic": predict(mname, ["log_budget", "critic_score"], True),
            "critic only": predict(mname, ["critic_score"], True),
        }
        ref_name = "greenlight (covered train)"
        for vname, p in variants.items():
            lo, hi = bootstrap_ci(y_b, p, n_boot)
            row = {"model": mname, "features": vname, "PR-AUC": average_precision_score(y_b, p),
                   "PR-AUC 95% CI": f"[{lo:.3f}, {hi:.3f}]", "ROC-AUC": roc_auc_score(y_b, p),
                   "vs greenlight(covered) [95% CI]": "-", "critic helps?": "-"}
            if vname in ("greenlight + critic", "budget + critic", "critic only"):
                dlo, dhi = paired_bootstrap_delta(y_b, variants[ref_name], p, n=n_boot)
                d = average_precision_score(y_b, p) - average_precision_score(y_b, variants[ref_name])
                row["vs greenlight(covered) [95% CI]"] = f"{d:+.3f} [{dlo:+.3f}, {dhi:+.3f}]"
                if vname == "greenlight + critic":
                    row["critic helps?"] = significance(dlo, dhi)
            rows.append(row)
    res = pd.DataFrame(rows).set_index(["model", "features"])
    print(res.round(3).to_string())
    print("\nRead 'critic helps?' only on the 'greenlight + critic' rows: 'yes' needs the whole CI above 0.")
    return res


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target", choices=["roi", "revenue"], default="roi")
    ap.add_argument("--models", nargs="+", default=["Logistic Regression", "XGBoost"])
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    out_dir = config.REPORTS_DIR; out_dir.mkdir(exist_ok=True)
    res = run(pd.read_pickle(config.MODELING_FRAME), TARGET_COLS[args.target], args.models, args.boot)
    if res is not None:
        res.round(4).to_csv(out_dir / f"critic_study_{args.target}.csv")
        print(f"Saved -> {out_dir / f'critic_study_{args.target}.csv'}")


if __name__ == "__main__":
    main()
