"""Regression tests for the calendar-date alignment of lags, rolling windows and targets.

Each test is written so that it FAILS if the corresponding bug is re-introduced (see tools/mutation_check.py,
which re-introduces the bugs one by one and verifies this suite goes red).
"""
import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.features import (add_case_history, add_target, add_weather_features, check_unique_ward_weeks, load_raw,
                          shifted_target, value_at_offset)

WEEK = pd.Timedelta(weeks=1)


def _toy(n=14, wards=("A", "B"), drop=None):
    """Two wards x n weekly rows with distinct, easily recognisable values: rain = 100*ward_idx + week_idx."""
    rows = []
    for wi, w in enumerate(wards):
        for k in range(n):
            rows.append({"date": pd.Timestamp("2024-01-01") + k * WEEK, "city": "X", "ward_id": w,
                         "population_density_sqkm": 1.0, "water_logging_risk_index": .5, "sanitation_score": .5,
                         "avg_temp_c": 20.0 + k, "rainfall_mm": 100.0 * wi + k, "relative_humidity_pct": 50.0 + k,
                         "dengue_cases": 10 * wi + k})
    d = pd.DataFrame(rows)
    if drop is not None:                                   # drop (ward, week_idx) rows -> calendar gap
        for w, k in drop:
            d = d[~((d.ward_id == w) & (d.date == pd.Timestamp("2024-01-01") + k * WEEK))]
    return d.reset_index(drop=True)


def _ref(d, ward, date, col):
    """Independent reference: look the value up by (ward, calendar date)."""
    m = d[(d.ward_id == ward) & (d.date == date)]
    return float(m[col].iloc[0]) if len(m) else np.nan


def test_lags_match_calendar_reference_with_and_without_gaps():
    for drop in (None, [("A", 5)], [("A", 5), ("A", 6), ("B", 9)]):
        d = _toy(drop=drop)
        f = add_weather_features(d)
        for k in C.LAGS:
            for col, s in (("rainfall_mm", "rain"), ("avg_temp_c", "temp"), ("relative_humidity_pct", "hum")):
                exp = np.array([_ref(d, r.ward_id, r.date - k * WEEK, col) for r in f.itertuples()])
                assert np.allclose(f[f"{s}_lag_{k}w"].to_numpy(), exp, equal_nan=True), (drop, s, k)


def test_missing_week_yields_nan_not_a_misaligned_value():
    d = _toy(drop=[("A", 5)])                               # ward A has no data for week 5
    f = add_weather_features(d).set_index(["ward_id", "date"])
    t = lambda k: pd.Timestamp("2024-01-01") + k * WEEK
    assert np.isnan(f.loc[("A", t(6)), "rain_lag_1w"])      # the lag points at the missing week 5
    assert f.loc[("A", t(7)), "rain_lag_1w"] == 6.0         # week 6 is present
    assert np.isnan(f.loc[("A", t(7)), "rain_lag_2w"])      # points at the missing week 5 (positional shift(2) would return week 4's value)
    assert np.isnan(f.loc[("A", t(8)), "rain_lag_3w"])      # positional shift(3) would silently return week 4's value
    assert f.loc[("A", t(8)), "rain_lag_4w"] == 4.0         # week 4 is present and correctly 4 calendar weeks back


def test_rolling_windows_are_calendar_based_and_nan_when_a_week_is_missing():
    d = _toy(drop=[("A", 5)])
    f = add_weather_features(d).set_index(["ward_id", "date"])
    t = lambda k: pd.Timestamp("2024-01-01") + k * WEEK
    assert f.loc[("B", t(10)), "rain_roll4w_sum"] == sum(100 + k for k in (7, 8, 9, 10))
    assert f.loc[("B", t(10)), "rain_roll3w_sum"] == sum(100 + k for k in (8, 9, 10))
    assert np.isnan(f.loc[("A", t(7)), "rain_roll4w_sum"])  # window 4..7 contains the missing week 5
    assert f.loc[("A", t(9)), "rain_roll4w_sum"] == sum((6, 7, 8, 9))
    assert f.loc[("A", t(10)), "temp_roll4w_mean"] == np.mean([20 + k for k in (7, 8, 9, 10)])


def test_target_is_calendar_aligned_and_nan_if_any_horizon_week_missing():
    d = _toy(drop=[("A", 8)])
    y = add_target(d, (2, 3, 4))
    for r in d.itertuples():
        vals = [_ref(d, r.ward_id, r.date + h * WEEK, "dengue_cases") for h in (2, 3, 4)]
        exp = np.nan if np.isnan(vals).any() else sum(vals)
        got = y[r.Index]
        assert (np.isnan(got) and np.isnan(exp)) or got == exp
    a = d[(d.ward_id == "A") & (d.date == pd.Timestamp("2024-01-01") + 5 * WEEK)].index[0]
    assert np.isnan(y[a])                                    # t+3 = week 8 is missing


def test_shifted_target_for_seasonal_naive_uses_calendar_weeks():
    d = _toy(n=60, drop=[("A", 30)])
    y = add_target(d, (2,))
    s = shifted_target(d, y, 52)
    for r in d.itertuples():
        exp = _ref(d.assign(yy=y), r.ward_id, r.date - 52 * WEEK, "yy")
        assert (np.isnan(s[r.Index]) and np.isnan(exp)) or s[r.Index] == exp


def test_duplicate_ward_week_records_are_rejected(tmp_path):
    d = _toy()
    dup = pd.concat([d, d.iloc[[3]]])
    with pytest.raises(ValueError, match="duplicate"):
        check_unique_ward_weeks(dup)
    with pytest.raises(ValueError, match="duplicate"):
        add_weather_features(dup)
    with pytest.raises(ValueError, match="duplicate"):
        value_at_offset(dup, dup["rainfall_mm"], -1)
    p = tmp_path / "dup.csv"; dup.to_csv(p, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        load_raw(p)


def test_features_never_use_future_weather_or_cases():
    """Changing ANY value after the cut-off must leave every feature at or before the cut-off unchanged."""
    d = _toy(n=30)
    cut = pd.Timestamp("2024-01-01") + 15 * WEEK
    for build in (add_weather_features, lambda x: add_case_history(x)):
        a = build(d)
        mod = d.copy()
        for c in ("rainfall_mm", "avg_temp_c", "relative_humidity_pct", "dengue_cases"):
            mod.loc[mod.date > cut, c] = 9999
        b = build(mod)
        cols = [c for c in a.columns if c.startswith(("rain_", "temp_", "hum_", "cases_"))]
        m = a.date <= cut
        pd.testing.assert_frame_equal(a.loc[m, cols], b.loc[m, cols])


def test_lag_direction_is_past_not_future():
    d = _toy(n=20)
    f = add_weather_features(d)
    x = f[f.ward_id == "A"].reset_index(drop=True)
    assert (x.rain_lag_2w.dropna() < x.rainfall_mm[x.rain_lag_2w.notna()]).all()    # rain increases with time in the toy data


def test_real_dataset_is_clean_and_lags_equal_positional_shift_only_because_it_is_gapless():
    raw = load_raw()
    assert not raw.duplicated(["ward_id", "date"]).any()
    gaps = raw.groupby("ward_id").date.apply(lambda s: (s.diff().dropna() != WEEK).sum())
    assert gaps.sum() == 0
    f = add_weather_features(raw)
    w = f[f.ward_id == f.ward_id.iloc[0]].reset_index(drop=True)
    assert np.allclose(w.rain_lag_4w[4:], w.rainfall_mm.shift(4)[4:])
