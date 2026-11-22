"""Portable serialization, input-quality, live-weather (mocked network), geo and alert-threshold tests."""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
import pytest

from src import config as C
from src import live_weather as LW
from src import model_io
from src.evaluate import alert_metrics, calibration_by_level, core_metrics, pick_alert_cut
from src.geo import city_summary, ward_coordinates
from src.quality import data_quality_report, freshness, repair_history, unusual_observations
from src.features import load_raw


# ---------- model_io ----------
def test_bundle_contains_no_pandas_objects_and_records_versions():
    raw = joblib.load(model_io.BUNDLE_PATH)
    assert isinstance(raw["climatology"], dict) and isinstance(raw["static_ref"], dict)   # plain containers -> no pyarrow dependency
    assert raw["meta"]["versions"]["scikit_learn"]
    b = model_io.read_bundle()
    assert isinstance(b["climatology"], pd.DataFrame) and set(b["features"]) <= set(b["climatology"].columns)
    for w in b["windows"].values():
        assert w["alert_cut"] <= w["cutpoints"][1] + 1e-9 and w["alert_cut"] > 0


def test_ensure_bundle_rebuilds_when_unreadable(tmp_path, monkeypatch):
    bad = tmp_path / "bundle.joblib"
    bad.write_bytes(b"not a pickle")
    calls = []
    good = model_io.read_bundle()
    monkeypatch.setattr("src.train.main", lambda bundle_only=False: calls.append(bundle_only))
    monkeypatch.setattr(model_io, "read_bundle", lambda path=bad, _n=[0]: (_ for _ in ()).throw(RuntimeError("x")) if not calls else good)
    assert model_io.ensure_bundle(bad, log=lambda *_: None) is good and calls == [True]


def test_version_mismatch_is_detected(tmp_path):
    raw = joblib.load(model_io.BUNDLE_PATH)
    raw["meta"]["versions"]["scikit_learn"] = "0.1.0"
    p = tmp_path / "old.joblib"
    joblib.dump(raw, p)
    with pytest.raises(RuntimeError):
        model_io.read_bundle(p)


# ---------- quality ----------
def _hist(n=9):
    return pd.DataFrame({"date": pd.date_range("2025-06-02", periods=n, freq="7D"), "rainfall_mm": 20.0, "avg_temp_c": 28.0, "relative_humidity_pct": 80.0})


def test_repair_fills_short_gaps_only():
    h = _hist(); h.loc[3, "rainfall_mm"] = np.nan
    r, notes = repair_history(h)
    assert r.rainfall_mm.notna().all() and notes
    h = _hist(); h.loc[2:5, "rainfall_mm"] = np.nan                      # 4-week gap > 2 -> must stay missing
    r, _ = repair_history(h)
    assert r.rainfall_mm.iloc[2:6].isna().all()


def test_unusual_and_freshness_and_dataset_report():
    ref = load_raw()
    h = _hist(); h.loc[4, "rainfall_mm"] = 900
    assert len(unusual_observations(h, ref)) == 1
    assert freshness("2026-11-01", now="2026-11-22")["stale"] and not freshness("2026-11-20", now="2026-11-22")["stale"]
    q = data_quality_report(ref)
    assert q["duplicate_ward_weeks"] == 0 and q["wards_with_calendar_gaps"] == 0 and sum(q["missing_cells"].values()) == 0


# ---------- live weather (no network used) ----------
NOW = pd.Timestamp("2026-11-22 10:00")    # a Sunday


def _payload(days=71, miss=None):
    dates = pd.date_range(end=NOW.normalize(), periods=days, freq="D")          # last element = today
    rain = [float(i % 5) for i in range(days)]
    if miss is not None:
        rain[miss] = None
    return {"daily": {"time": [d.strftime("%Y-%m-%d") for d in dates], "temperature_2m_mean": [28.0] * days,
                      "precipitation_sum": rain, "relative_humidity_2m_mean": [80.0] * days}}


def test_live_weekly_uses_only_complete_observed_weeks(tmp_path):
    calls = []
    res = LW.fetch_city_weather("Dhaka North", now=NOW, cache_dir=tmp_path, fetch=lambda *a: calls.append(1) or _payload())
    assert res.ok and res.source == "network" and calls == [1]
    assert res.weekly.date.dt.day_of_week.eq(0).all()                              # Monday-start weeks
    assert res.weekly.date.max() + pd.Timedelta(days=6) < NOW.normalize()      # no partial / future week
    assert len(res.weekly.tail(9)) == 9
    assert (res.daily.kind == "forecast").sum() == 1                          # today is not treated as observed


def test_live_cache_minimises_requests_and_survives_outage(tmp_path):
    calls = []
    fetch = lambda *a: calls.append(1) or _payload()
    LW.fetch_city_weather("Gazipur", now=NOW, cache_dir=tmp_path, fetch=fetch)
    r2 = LW.fetch_city_weather("Gazipur", now=NOW, cache_dir=tmp_path, fetch=fetch)
    assert r2.source == "cache" and calls == [1]                               # second call: zero API requests
    f = next(tmp_path.glob("*.json")); old = time.time() - 48 * 3600; os.utime(f, (old, old))     # expire cache
    def boom(*a): raise OSError("network down")
    r3 = LW.fetch_city_weather("Gazipur", now=NOW, cache_dir=tmp_path, fetch=boom)
    assert r3.ok and r3.source == "stale-cache" and r3.freshness["stale"] is True
    r4 = LW.fetch_city_weather("Chattogram", now=NOW, cache_dir=tmp_path, fetch=boom)
    assert not r4.ok and "keeps working" in r4.message                         # never raises into the UI


def test_live_offline_flag_and_bad_payload(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVE_WEATHER_OFFLINE", "1")
    called = []
    r = LW.fetch_city_weather("Dhaka South", now=NOW, cache_dir=tmp_path, fetch=lambda *a: called.append(1) or _payload())
    assert not r.ok and not called
    monkeypatch.delenv("LIVE_WEATHER_OFFLINE")
    r = LW.fetch_city_weather("Dhaka South", now=NOW, cache_dir=tmp_path, fetch=lambda *a: {"error": True, "reason": "bad"})
    assert not r.ok


def test_live_history_scores_with_same_model(tmp_path):
    from src.predict import load_features
    from src.scenario import predict_history
    df = load_features()
    res = LW.fetch_city_weather("Dhaka North", now=NOW, cache_dir=tmp_path, fetch=lambda *a: _payload(miss=40))
    hist, notes = LW.history_for_scoring(res, df[df.city == "Dhaka North"].iloc[-1])
    out = predict_history(hist)
    assert out["tier"] in C.RISK_TIERS and out["pred"] >= 0


# ---------- geo ----------
def test_no_ward_coordinates_are_invented(tmp_path, monkeypatch):
    assert not C.WARD_COORD_PATH.exists() or ward_coordinates() is not None
    monkeypatch.setattr(C, "WARD_COORD_PATH", tmp_path / "nope.csv")
    assert ward_coordinates() is None
    p = tmp_path / "w.csv"; p.write_text("ward_id,lat,lon\nX,10.0,10.0\n")
    monkeypatch.setattr(C, "WARD_COORD_PATH", p)
    with pytest.raises(ValueError):
        ward_coordinates()


def test_city_summary_matches_snapshot():
    snap = pd.DataFrame({"city": ["Dhaka North"] * 3, "ward_id": list("abc"), "pred": [1.0, 30.0, 50.0],
                         "tier": ["Low", "High", "Critical"], "alert": [False, True, True]})
    s = city_summary(snap).iloc[0]
    assert s.high_critical_wards == 2 and s.watch_flag_wards == 2 and s.wards == 3


# ---------- alert threshold / extra metrics ----------
def test_alert_threshold_respects_false_alarm_cap_on_validation():
    rng = np.random.default_rng(0)
    y = rng.poisson(8, 3000).astype(float); p = y + rng.normal(0, 3, 3000)
    cut, grid = pick_alert_cut(y, p, 15.0)
    chosen = next(r for r in grid if abs(r["alert_cut"] - cut) < 1e-9)
    assert chosen["false_alarm_rate"] <= C.ALERT_MAX_FALSE_ALARM or chosen["factor"] == 1.0
    assert cut <= 15.0
    m = alert_metrics(y >= 15, p, cut)
    assert 0 <= m["recall"] <= 1


def test_published_metrics_include_prauc_alert_and_calibration():
    m = json.loads((C.REPORT_DIR / "metrics.json").read_text())
    for w, t in m["test_by_window"].items():
        assert 0 < t["HC_PR_AUC"] <= 1 and "ALERT_recall" in t and "ALERT_false_alarm_rate" in t
    cal = pd.read_csv(C.REPORT_DIR / "interval_calibration_test.csv")
    assert len(cal) == 5 and cal.coverage.between(0, 1).all()
    assert (C.REPORT_DIR / "pr_curve_test.csv").exists() and (C.REPORT_DIR / "horizon_strategy_validation.csv").exists()
