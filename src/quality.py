"""Input-quality layer: missing, stale and unusual weather observations.

Used by the scenario / live-weather paths so the model is never silently fed bad data.
"""
import numpy as np
import pandas as pd

from . import config as C

MAX_GAP_WEEKS = 2          # linear interpolation allowed for gaps up to this length
MAX_STALE_DAYS = 10        # a "latest" observation older than this is flagged stale


def repair_history(h: pd.DataFrame, max_gap: int = MAX_GAP_WEEKS) -> tuple[pd.DataFrame, list]:
    """Fill short gaps (<= max_gap consecutive weeks) in the weather columns by linear interpolation.
    Returns (repaired_frame, notes). Longer gaps are left as NaN so downstream validation rejects them."""
    h = h.sort_values("date").reset_index(drop=True).copy()
    notes = []
    for c in C.WEATHER_COLS:
        s = h[c].astype(float)
        n_na = int(s.isna().sum())
        if n_na == 0:
            continue
        filled = s.interpolate(limit=max_gap, limit_area="inside")
        # interpolate(limit=...) fills the first `max_gap` NaNs of a longer gap too; undo for gaps > max_gap
        na = s.isna()
        run_id = (na != na.shift()).cumsum()
        run_len = na.groupby(run_id).transform("sum")
        filled[na & (run_len > max_gap)] = np.nan
        notes.append(f"{c}: {n_na - int(filled.isna().sum())} of {n_na} missing week(s) interpolated"
                     + (f", {int(filled.isna().sum())} left missing (gap > {max_gap} weeks)" if filled.isna().any() else ""))
        h[c] = filled
    return h, notes


def unusual_observations(h: pd.DataFrame, ref: pd.DataFrame, lo_q=0.001, hi_q=0.999) -> list:
    """Warn about values outside the range seen in the training data (model would extrapolate)."""
    out = []
    for c in C.WEATHER_COLS:
        lo, hi = ref[c].quantile(lo_q), ref[c].quantile(hi_q)
        bad = h[(h[c] < lo) | (h[c] > hi)]
        for _, r in bad.iterrows():
            out.append(f"{pd.Timestamp(r['date']).date()}: {c}={r[c]:.1f} is outside the training range [{lo:.1f}, {hi:.1f}]")
    return out


def freshness(latest_obs_date, now=None, max_stale_days: int = MAX_STALE_DAYS) -> dict:
    now = pd.Timestamp.now().normalize() if now is None else pd.Timestamp(now).normalize()
    age = int((now - pd.Timestamp(latest_obs_date).normalize()).days)
    return {"latest_observation": str(pd.Timestamp(latest_obs_date).date()), "age_days": age, "stale": age > max_stale_days}


def data_quality_report(df: pd.DataFrame) -> dict:
    """Dataset-level checks (duplicates, weekly continuity, ranges, missing values)."""
    d = df.sort_values(["ward_id", "date"])
    gaps = d.groupby("ward_id")["date"].apply(lambda s: int((s.diff().dropna() != pd.Timedelta(weeks=1)).sum()))
    return {
        "rows": int(len(d)), "wards": int(d.ward_id.nunique()), "date_min": str(d.date.min().date()), "date_max": str(d.date.max().date()),
        "duplicate_ward_weeks": int(d.duplicated(["ward_id", "date"]).sum()),
        "missing_cells": {c: int(v) for c, v in d[C.WEATHER_COLS + ["dengue_cases"]].isna().sum().items()},
        "wards_with_calendar_gaps": int((gaps > 0).sum()),
        "negative_values": int((d[["rainfall_mm", "dengue_cases"]] < 0).sum().sum()),
        "humidity_out_of_range": int((~d.relative_humidity_pct.between(0, 100)).sum()),
        "zero_case_share": float((d.dengue_cases == 0).mean()),
    }
