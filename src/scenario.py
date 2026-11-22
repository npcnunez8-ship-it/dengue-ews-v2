"""What-if / operational inference: score a ward from a user-supplied weather history.

Uses the *same* feature code (features.add_weather_features) and the *same* serialized bundle as
training and the batch dashboard, so a scenario can never drift from the trained feature definitions.
"""
import numpy as np
import pandas as pd

from . import config as C
from .features import REQUIRED, add_weather_features
from .predict import load_bundle, predict_frame

N_HISTORY = max(C.LAGS) + 1      # 9 weekly rows are needed for lag-8 / roll-8 features
RAW = ["date", "city", "ward_id", *C.STATIC_COLS, *C.WEATHER_COLS]
BOUNDS = {"rainfall_mm": (0, 1000), "relative_humidity_pct": (0, 100), "avg_temp_c": (-5, 50)}


def ward_history(df: pd.DataFrame, ward: str, date, n: int = N_HISTORY) -> pd.DataFrame:
    h = df[(df.ward_id == ward) & (df.date <= pd.Timestamp(date))].sort_values("date").tail(n)
    return h[RAW].reset_index(drop=True)


def validate_history(h: pd.DataFrame) -> pd.DataFrame:
    """Reject incomplete / physically implausible input with a clear message."""
    miss = [c for c in RAW if c not in h.columns]
    if miss:
        raise ValueError(f"Missing columns: {miss}")
    if len(h) < N_HISTORY:
        raise ValueError(f"Need {N_HISTORY} consecutive weekly rows (lags go back {max(C.LAGS)} weeks); got {len(h)}.")
    h = h.sort_values("date").tail(N_HISTORY).reset_index(drop=True)
    if h[C.WEATHER_COLS].isna().any().any():
        raise ValueError("Weather history contains missing values; fill or re-fetch them before scoring.")
    for c, (lo, hi) in BOUNDS.items():
        if not h[c].between(lo, hi).all():
            raise ValueError(f"{c} outside plausible range [{lo}, {hi}].")
    if not (h["date"].diff().dropna() == pd.Timedelta(weeks=1)).all():
        raise ValueError("History must be consecutive weekly observations.")
    return h


def apply_scenario(h, rain_mult=1.0, temp_shift=0.0, hum_shift=0.0, weeks=None):
    """Scale rainfall / shift temperature & humidity on the last `weeks` rows (None = all)."""
    h = h.copy()
    idx = h.index if weeks is None else h.index[-int(weeks):]
    h.loc[idx, "rainfall_mm"] = (h.loc[idx, "rainfall_mm"] * rain_mult).clip(0)
    h.loc[idx, "avg_temp_c"] = h.loc[idx, "avg_temp_c"] + temp_shift
    h.loc[idx, "relative_humidity_pct"] = (h.loc[idx, "relative_humidity_pct"] + hum_shift).clip(0, 100)
    return h


def predict_history(h: pd.DataFrame, window=C.MAIN_WINDOW, bundle=None) -> dict:
    h = validate_history(h)
    b = bundle or load_bundle()
    h = h.assign(dengue_cases=0)                      # placeholder: weather-only model never reads it
    last = add_weather_features(h[REQUIRED]).iloc[[-1]]
    r = predict_frame(last, window, b).iloc[0]
    if pd.isna(r.pred):
        raise ValueError("Could not compute features from the supplied history.")
    return {"window": window, "pred": float(r.pred), "lo": float(r.lo), "hi": float(r.hi), "tier": r.tier, "alert": bool(r.alert),
            "cutpoints": b["windows"][window]["cutpoints"]}


def all_horizons(row: pd.DataFrame, bundle=None) -> pd.DataFrame:
    """Multi-week forecast (week 2/3/4 + the 2-4 week total) with 80% intervals for one ward-week."""
    b = bundle or load_bundle()
    out = []
    for w in C.WINDOWS:
        r = predict_frame(row, w, b).iloc[0]
        out.append({"window": w, "pred": r.pred, "lo": r.lo, "hi": r.hi, "tier": r.tier, "alert": bool(r.alert)})
    return pd.DataFrame(out)
