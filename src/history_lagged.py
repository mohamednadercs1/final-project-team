"""Time-lagged "creative power" history features (drop-in for features.add_history_features).

Why: a film is greenlit 12-24 months before release. The original features use EVERY earlier
release, including films that came out a few weeks before, whose final box office was not known
when the greenlight decision was made. Here a film only sees films released at least
`lag_days` days before it, which also removes same-day ties.

Usage (features.py, end of build_modeling_frame):
    from history_lagged import add_history_features_lagged
    return add_history_features_lagged(df, lag_days=config.HISTORY_LAG_DAYS)   # e.g. 365
"""
import numpy as np
import pandas as pd

from features import GROUPS, HISTORY_COLS, VALUES, _long_table


def add_history_features_lagged(df, lag_days=365):
    """Mean log-ROI / log-revenue / vote of each person's films released <= (release_date - lag_days)."""
    if lag_days < 1:
        raise ValueError("lag_days must be >= 1 (0 would let same-day releases leak)")
    d = df.reset_index(drop=True)
    d = d.drop(columns=[c for c in HISTORY_COLS if c in d.columns])
    lag = pd.Timedelta(days=lag_days)

    for group in GROUPS:
        long = _long_table(d, group)
        frames = []
        for _, g in long.groupby("person", sort=False):
            g = g.sort_values("release_date", kind="mergesort")
            dates = g["release_date"].to_numpy()
            cutoff = (g["release_date"] - lag).to_numpy()
            k = np.searchsorted(dates, cutoff, side="right")      # films released on/before the cutoff
            out = {"movie_idx": g["movie_idx"].to_numpy(), f"{group}_n_prior": k.astype(float)}
            for v in VALUES:
                vals = g[v].to_numpy(dtype=float)
                valid = ~np.isnan(vals)
                csum = np.concatenate([[0.0], np.cumsum(np.where(valid, vals, 0.0))])
                ccnt = np.concatenate([[0], np.cumsum(valid)])
                cnt = ccnt[k]
                with np.errstate(invalid="ignore", divide="ignore"):
                    out[f"{group}_prior_{v}"] = np.where(cnt > 0, csum[k] / np.maximum(cnt, 1), np.nan)
            frames.append(pd.DataFrame(out))
        cols = [f"{group}_prior_{v}" for v in VALUES] + [f"{group}_n_prior"]
        if frames:
            agg = pd.concat(frames).groupby("movie_idx")[cols].mean()
            d = d.join(agg)
        else:
            for c in cols:
                d[c] = np.nan
        d[f"{group}_n_prior"] = d[f"{group}_n_prior"].fillna(0)
    return d
