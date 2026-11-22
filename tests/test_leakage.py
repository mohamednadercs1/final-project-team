"""Leakage tests: a film's features must not depend on its own outcome or on later films' outcomes.

Idea (a "perturbation test"): change the outcome of ONE film to an absurd value and re-build the
features. Every film that is not allowed to know that outcome must get *identical* features.

Run:  pytest -q tests/test_leakage.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from features import HISTORY_COLS, VALUES, FeatureBuilder, add_history_features, add_targets  # noqa: E402
from history_lagged import add_history_features_lagged  # noqa: E402

LAG = 365
POST_RELEASE = ["popularity", "vote_average", "vote_count", "vote", "revenue", "roi",
                "log_roi", "log_rev", "success_roi", "success_rev", "tier"]


# ------------------------------------------------------------------ synthetic data
def make_frame(n=300, seed=0, unique_dates=True):
    rng = np.random.default_rng(seed)
    days = rng.choice(6000, n, replace=not unique_dates)
    dates = pd.Timestamp("2000-01-01") + pd.to_timedelta(days, unit="D")
    directors = [f"dir{i}" for i in range(25)] + [None]
    actors = [f"actor{i}" for i in range(60)]
    studios = [f"studio{i}" for i in range(15)]
    df = pd.DataFrame({
        "id": np.arange(n),
        "release_date": dates,
        "director": [directors[i] for i in rng.integers(0, len(directors), n)],
        "cast_top3": [list(rng.choice(actors, 3, replace=False)) for _ in range(n)],
        "studios_list": [list(rng.choice(studios, rng.integers(1, 3), replace=False)) for _ in range(n)],
        "log_roi": rng.normal(0.5, 1, n),
        "log_rev": rng.normal(17, 1.5, n),
        "vote": np.where(rng.random(n) < 0.1, np.nan, rng.normal(6.2, 0.8, n)),
    })
    return df.sort_values(["release_date", "id"]).reset_index(drop=True)


def perturb(df, k):
    out = df.copy()
    for v in VALUES:
        out.loc[k, v] = 1e4
    return out


def assert_same(a, b, mask, label):
    pd.testing.assert_frame_equal(a.loc[mask, HISTORY_COLS].reset_index(drop=True),
                                  b.loc[mask, HISTORY_COLS].reset_index(drop=True),
                                  check_dtype=False, obj=label)


# ------------------------------------------------------------------ original implementation
def test_history_never_sees_own_or_later_outcomes():
    """Films released on/before film k (unique dates) must be unaffected by k's outcome."""
    df = make_frame()
    base = add_history_features(df)
    changed_later = 0
    for k in range(0, 120, 8):                       # early films -> many later films share people
        alt = add_history_features(perturb(df, k))
        not_allowed = df["release_date"] <= df.loc[k, "release_date"]
        assert_same(base, alt, not_allowed, f"film {k}")
        later = ~not_allowed
        changed_later += int((base.loc[later, HISTORY_COLS].fillna(-1).to_numpy()
                              != alt.loc[later, HISTORY_COLS].fillna(-1).to_numpy()).any(axis=1).sum())
    assert changed_later > 0, "perturbation had no effect anywhere: the test itself is broken"


# ------------------------------------------------------------------ same-day ties
def _tie_frame():
    """Two films by the same director/studio released the same day (index 0 then 1)."""
    df = make_frame(60, seed=1)
    df.loc[1, "release_date"] = df.loc[0, "release_date"]
    for col in ("director",):
        df.loc[1, col] = df.loc[0, col] = "dir_tie"
    df.at[0, "studios_list"] = ["studio_tie"]
    df.at[1, "studios_list"] = ["studio_tie"]
    return df.sort_values(["release_date", "id"]).reset_index(drop=True)


@pytest.mark.parametrize("fn, kwargs", [
    pytest.param(add_history_features, {}, marks=pytest.mark.xfail(
        reason="KNOWN ISSUE: same-day releases leak into each other (ordering by movie_idx). "
               "Use add_history_features_lagged.", strict=False)),
    pytest.param(add_history_features_lagged, {"lag_days": LAG}),
])
def test_same_day_release_does_not_leak(fn, kwargs):
    df = _tie_frame()
    tie = df.index[df["director"] == "dir_tie"].tolist()
    assert len(tie) == 2
    a, b = tie
    base = fn(df, **kwargs)
    alt = fn(perturb(df, a), **kwargs)
    # film b is released the same day as a: it must not see a's outcome
    assert_same(base, alt, df.index == b, "same-day film")


# ------------------------------------------------------------------ lagged implementation
def test_lagged_respects_the_lag_window():
    df = make_frame()
    base = add_history_features_lagged(df, LAG)
    changed_after_lag = 0
    for k in range(0, 120, 8):
        alt = add_history_features_lagged(perturb(df, k), LAG)
        horizon = df.loc[k, "release_date"] + pd.Timedelta(days=LAG)
        not_allowed = df["release_date"] < horizon          # includes k itself
        assert_same(base, alt, not_allowed, f"film {k}")
        allowed = ~not_allowed
        changed_after_lag += int((base.loc[allowed, HISTORY_COLS].fillna(-1).to_numpy()
                                  != alt.loc[allowed, HISTORY_COLS].fillna(-1).to_numpy()).any(axis=1).sum())
    assert changed_after_lag > 0, "perturbation had no effect anywhere: the test itself is broken"


def test_lagged_matches_original_when_lag_is_one_day():
    """Sanity check of the new code: with unique dates and a 1-day lag both versions agree."""
    df = make_frame()
    pd.testing.assert_frame_equal(add_history_features(df)[HISTORY_COLS],
                                  add_history_features_lagged(df, 1)[HISTORY_COLS],
                                  check_dtype=False, check_exact=False, rtol=1e-9, atol=1e-9)


def test_lagged_rejects_zero_lag():
    with pytest.raises(ValueError):
        add_history_features_lagged(make_frame(30), 0)


# ------------------------------------------------------------------ feature builder
def test_feature_matrix_has_no_post_release_columns():
    df = make_frame(400, seed=2)
    rng = np.random.default_rng(2)
    df["budget"] = rng.lognormal(16.5, 1, len(df))
    df["revenue"] = df["budget"] * np.exp(df["log_roi"])
    df["runtime"] = rng.normal(105, 15, len(df))
    df["original_language"] = "en"
    df["is_sequel"] = rng.integers(0, 2, len(df))
    df["genres_list"] = [list(rng.choice(["Action", "Drama", "Comedy"], 2, replace=False)) for _ in range(len(df))]
    df["popularity"] = rng.random(len(df)); df["vote_average"] = rng.normal(6, 1, len(df))
    df["vote_count"] = rng.integers(1, 5000, len(df))
    d = add_history_features(add_targets(df.drop(columns=["log_roi", "log_rev", "vote"])))
    X = FeatureBuilder().fit(d).transform(d)
    leaked = sorted(set(X.columns) & set(POST_RELEASE))
    assert not leaked, f"post-release columns in the feature matrix: {leaked}"
