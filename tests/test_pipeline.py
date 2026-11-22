import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C
from src.features import add_target, add_weather_features, build_dataset, load_raw
from src.predict import explain_row, load_bundle, predict_frame, predict_one
from src.risk import fit_cutpoints, to_tier
from src.train import split_masks


@pytest.fixture(scope="module")
def df():
    return build_dataset(with_history=False)


def test_lags_are_per_ward_and_correct(df):
    for w in df.ward_id.unique()[:5]:
        x = df[df.ward_id == w].reset_index(drop=True)      # this dataset has no calendar gaps -> row shift == calendar shift
        assert np.allclose(x.rain_lag_3w[3:], x.rainfall_mm.shift(3)[3:])
        assert x.rain_lag_3w[:3].isna().all()        # no leakage across wards
    assert np.allclose(df.groupby("ward_id").rain_roll4w_sum.apply(lambda s: s.isna().sum()), 3)


def test_features_do_not_use_future_data():
    raw = load_raw()
    a = add_weather_features(raw)
    mod = raw.copy()
    cut = pd.Timestamp("2024-01-01")
    for c in ["rainfall_mm", "avg_temp_c", "relative_humidity_pct", "dengue_cases"]:
        mod.loc[mod.date > cut, c] = 9999
    b = add_weather_features(mod)
    cols = [c for c in a.columns if c.startswith(("rain_", "temp_", "hum_"))]
    m = a.date <= cut
    pd.testing.assert_frame_equal(a.loc[m, cols], b.loc[m, cols])


def test_target_matches_definition():
    raw = load_raw()
    y = add_target(raw)
    x = raw[raw.ward_id == "CCC_Ward_01"].reset_index(drop=True)
    exp = x.dengue_cases.shift(-2) + x.dengue_cases.shift(-3) + x.dengue_cases.shift(-4)
    assert np.allclose(y[raw.ward_id == "CCC_Ward_01"].values, exp.values, equal_nan=True)


def test_chronological_split_has_purge_gap(df):
    y = add_target(df)
    tr, va, ts = split_masks(df, y)
    assert df.loc[tr, "date"].max() < df.loc[va, "date"].min() < df.loc[va, "date"].max() < df.loc[ts, "date"].min()
    # train target window must end before validation start
    assert df.loc[tr, "date"].max() + pd.Timedelta(weeks=4) <= df.loc[va, "date"].min()
    assert not (tr & va).any() and not (va & ts).any()


def test_risk_tiers_monotonic():
    cuts = fit_cutpoints(np.arange(100), C.TIER_QUANTILES)
    tiers = to_tier([0, cuts[0], cuts[1], cuts[2] + 5], cuts)
    assert tiers == ["Low", "Medium", "High", "Critical"]


def test_predictions_valid_and_fast(df):
    p = predict_frame(df.tail(2000))
    ok = p.pred.notna()
    assert ok.any() and (p.pred[ok] >= 0).all()
    # every interval must contain its own point prediction (this was previously a vacuous `| True` assertion)
    assert (p.lo[ok] <= p.pred[ok] + 1e-9).all() and (p.pred[ok] <= p.hi[ok] + 1e-9).all() and (p.lo[ok] <= p.hi[ok]).all()
    assert set(p.tier[ok]) <= set(C.RISK_TIERS)


def test_predict_one_and_explain(df):
    out = predict_one(df, "DNCC_Ward_01", "2025-08-04")
    assert out["risk_tier"] in C.RISK_TIERS and out["predicted_cases"] >= 0
    ex = explain_row(df[(df.ward_id == "DNCC_Ward_01") & (df.date == "2025-08-04")])
    assert {"driver", "effect_cases"} <= set(ex.columns) and len(ex) >= 5


def test_rain_signal_monotone(df):
    """Sanity: heavy rain 3 weeks ago should not lower predicted risk for a summer week."""
    b = load_bundle()
    row = df[(df.ward_id == "DNCC_Ward_01") & (df.date == "2025-08-04")].copy()
    wet, dry = row.copy(), row.copy()
    for c in [c for c in b["features"] if c.startswith("rain_")]:
        dry[c] = 0.0
    assert predict_frame(wet).pred.iloc[0] >= predict_frame(dry).pred.iloc[0]


def test_metrics_report_schema_and_baselines_beaten():
    import json
    m = json.loads((C.REPORT_DIR / "metrics.json").read_text())
    t = m["test_by_window"][C.MAIN_WINDOW]
    for k in ("MAE", "RMSE", "R2", "HC_precision", "HC_recall", "HC_F1", "HC_false_alarm_rate",
              "interval_coverage", "interval_width", "n", "baselines"):
        assert k in t
    assert set(t["baselines"]) == {"Mean baseline", "Persistence (cases_t x horizons)", "Seasonal-naive (52w earlier)"}
    assert all(t["MAE"] < b["MAE"] for b in t["baselines"].values())   # model must beat every baseline on identical samples
    assert t["cutpoints"] == sorted(t["cutpoints"])


def test_slice_reports_exist_and_city_counts_add_up():
    import json
    m = json.loads((C.REPORT_DIR / "metrics.json").read_text())
    city = pd.read_csv(C.REPORT_DIR / "test_by_city.csv")
    assert city.n.sum() == m["test_by_window"][C.MAIN_WINDOW]["n"] == len(pd.read_csv(C.REPORT_DIR / "test_predictions.csv"))
    assert pd.read_csv(C.REPORT_DIR / "test_by_season.csv").n.sum() == city.n.sum()


def test_evaluation_samples_identical_across_models(df):
    """Seasonal-naive needs the target 52w earlier; ML models are scored on exactly the same rows."""
    y = add_target(df)
    tr, va, ts = split_masks(df, y)
    sn = y.groupby(df.ward_id, sort=False).shift(52)
    ts2 = ts & sn.notna()
    assert ts2.sum() == len(pd.read_csv(C.REPORT_DIR / "test_predictions.csv"))
