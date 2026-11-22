"""Ward-level weather lag / rolling feature engineering (leak-free, calendar-aligned).

Every feature at forecast date t uses only information observed at or before t.
Lags / rolling windows / targets are aligned on (ward_id, calendar date), NOT on row position:
a missing week therefore yields NaN features for the rows that depend on it instead of silently
pulling a value from a different week.  Duplicate ward-week records are rejected.
"""
import numpy as np
import pandas as pd

from .config import (DATA_PATH, LAGS, ROLL_WINDOWS, STATIC_COLS, WEATHER_COLS)

SHORT = {"avg_temp_c": "temp", "rainfall_mm": "rain", "relative_humidity_pct": "hum"}
REQUIRED = ["date", "city", "ward_id", *STATIC_COLS, *WEATHER_COLS, "dengue_cases"]
WEEK = pd.Timedelta(weeks=1)


def check_unique_ward_weeks(df: pd.DataFrame) -> None:
    """Raise if any (ward_id, date) appears more than once (calendar alignment needs a unique key)."""
    dup = df.duplicated(["ward_id", "date"], keep=False)
    if dup.any():
        ex = df.loc[dup, ["ward_id", "date"]].drop_duplicates().head(3).astype(str).values.tolist()
        raise ValueError(f"{int(df.duplicated(['ward_id', 'date']).sum())} duplicate ward-week record(s), e.g. {ex}")


def value_at_offset(df: pd.DataFrame, values: pd.Series, weeks: int) -> np.ndarray:
    """For every row (ward w, date t) return `values` observed for the SAME ward at date t + `weeks` weeks.

    weeks < 0 -> lag (past), weeks > 0 -> lead (future target).  Rows whose partner week is absent get NaN.
    This is the single place where time shifting happens; nothing in the project shifts by row position.
    """
    key = pd.MultiIndex.from_arrays([df["ward_id"].to_numpy(), df["date"].to_numpy()])
    src = pd.Series(np.asarray(values, dtype=float), index=key)
    if not src.index.is_unique:
        raise ValueError("duplicate ward-week records: calendar alignment is ambiguous")
    want = pd.MultiIndex.from_arrays([df["ward_id"].to_numpy(), (df["date"] + weeks * WEEK).to_numpy()])
    return src.reindex(want).to_numpy()


def load_raw(path=DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    check_unique_ward_weeks(df)
    return df.sort_values(["ward_id", "date"]).reset_index(drop=True)


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Weather lags (1-8 calendar weeks), trailing rolling sums/means (window ends at t), seasonality."""
    check_unique_ward_weeks(df)
    df = df.sort_values(["ward_id", "date"]).reset_index(drop=True)
    new = {}
    for col in WEATHER_COLS:
        s = SHORT[col]
        cur = df[col].to_numpy(dtype=float)
        lags = {0: cur}
        for k in range(1, max(max(LAGS), max(ROLL_WINDOWS) - 1) + 1):
            lags[k] = value_at_offset(df, df[col], -k)
        for k in LAGS:
            new[f"{s}_lag_{k}w"] = lags[k]
        for w in ROLL_WINDOWS:
            stack = np.vstack([lags[k] for k in range(w)])          # weeks t, t-1, ..., t-w+1
            # any missing week inside the window -> NaN (no silent partial windows)
            new[f"{s}_roll{w}w_" + ("sum" if col == "rainfall_mm" else "mean")] = (
                stack.sum(axis=0) if col == "rainfall_mm" else stack.mean(axis=0))
    woy = df["date"].dt.isocalendar().week.astype(int)
    new["week_sin"] = np.sin(2 * np.pi * woy / 52.0)
    new["week_cos"] = np.cos(2 * np.pi * woy / 52.0)
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


def add_case_history(df: pd.DataFrame) -> pd.DataFrame:
    """Optional autoregressive features (observed cases up to t). Kept separate so
    weather-only vs weather+history can be compared honestly."""
    new = {f"cases_lag_{k}w": value_at_offset(df, df["dengue_cases"], -k) for k in (1, 2, 3, 4)}
    stack = np.vstack([df["dengue_cases"].to_numpy(float)] + [new[f"cases_lag_{k}w"] for k in (1, 2, 3)])
    new["cases_roll4w_mean"] = stack.mean(axis=0)
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


def add_target(df: pd.DataFrame, horizons=(2, 3, 4)) -> pd.Series:
    """Future cases summed over the given weeks-ahead, aligned by calendar date.
    NaN unless EVERY requested future week is observed."""
    out = np.zeros(len(df))
    for h in horizons:
        out = out + value_at_offset(df, df["dengue_cases"], h)
    return pd.Series(out, index=df.index)


def shifted_target(df: pd.DataFrame, y: pd.Series, weeks: int) -> pd.Series:
    """`y` of the same ward `weeks` weeks earlier (negative offset) - e.g. seasonal-naive reference = 52."""
    return pd.Series(value_at_offset(df, y, -weeks), index=df.index)


def feature_sets(df: pd.DataFrame) -> dict:
    """Named feature configurations for the ablation study."""
    wx = [c for c in df.columns if any(c.startswith(f"{s}_") for s in SHORT.values())]
    cur = list(WEATHER_COLS)
    season = ["week_sin", "week_cos"]
    hist = [c for c in df.columns if c.startswith("cases_")]
    return {
        "weather_only": cur + wx + STATIC_COLS + season,
        "weather_no_static": cur + wx + season,
        "weather_plus_history": cur + wx + STATIC_COLS + season + hist,
    }


def feature_groups(cols) -> dict:
    """Group columns into human-readable drivers for explanations."""
    groups = {"Rainfall (lagged)": [], "Rainfall (current)": [], "Temperature": [],
              "Humidity": [], "Ward vulnerability": [], "Season": [], "Recent cases": []}
    for c in cols:
        if c == "rainfall_mm":
            groups["Rainfall (current)"].append(c)
        elif c.startswith("rain_"):
            groups["Rainfall (lagged)"].append(c)
        elif c.startswith("temp_") or c == "avg_temp_c":
            groups["Temperature"].append(c)
        elif c.startswith("hum_") or c == "relative_humidity_pct":
            groups["Humidity"].append(c)
        elif c in STATIC_COLS:
            groups["Ward vulnerability"].append(c)
        elif c.startswith("week_"):
            groups["Season"].append(c)
        elif c.startswith("cases_"):
            groups["Recent cases"].append(c)
    return {k: v for k, v in groups.items() if v}


def build_dataset(path=DATA_PATH, with_history=True) -> pd.DataFrame:
    df = add_weather_features(load_raw(path))
    return add_case_history(df) if with_history else df
