"""Escalation thresholds, interval containment/calibration and test-set independence."""
import json

import numpy as np
import pandas as pd
import pytest

from src import calibration as K
from src import config as C
from src.predict import load_features, load_bundle, predict_frame
from src.scenario import all_horizons

CUTS = [10.0, 27.0, 41.0]


def _fake_oof(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.gamma(1.0, 9.0, n)
    y = rng.poisson(np.maximum(p * 1.1, 0.05))
    return pd.DataFrame({"y": y.astype(float), "p": p, "fold": rng.integers(0, 5, n)})


def test_every_interval_contains_its_point_prediction_for_any_input():
    rng = np.random.default_rng(1)
    o = _fake_oof()
    for mode in ("scaled_global", "mondrian_asym"):
        cal = K.fit_interval(o.p, o.y, CUTS, 0.10, mode)
        p = np.concatenate([rng.uniform(0, 120, 5000), [0.0, 1e-9, 27.0, 41.0, 500.0]])
        lo, hi = K.apply_interval(p, CUTS, cal)
        assert (lo <= p + 1e-9).all() and (p <= hi + 1e-9).all() and (lo >= 0).all()


def test_shipped_batch_predictions_have_ordered_intervals_for_every_window():
    df = load_features()
    sample = df.dropna(subset=load_bundle()["features"]).sample(3000, random_state=0)
    for w in C.WINDOWS:
        r = predict_frame(sample, w)
        assert (r.lo <= r.pred + 1e-9).all() and (r.pred <= r.hi + 1e-9).all(), w
    row = sample.iloc[[0]]
    t = all_horizons(row)
    assert ((t.lo <= t.pred + 1e-9) & (t.pred <= t.hi + 1e-9)).all()


def test_published_test_predictions_intervals_contain_point_and_tiers_are_monotone():
    t = pd.read_csv(C.REPORT_DIR / "test_predictions.csv")
    assert ((t.lo <= t.pred + 1e-9) & (t.pred <= t.hi + 1e-9)).all()
    assert ((t.lo_original <= t.pred + 1e-9) & (t.pred <= t.hi_original + 1e-9)).all()
    order = {k: i for i, k in enumerate(C.RISK_TIERS)}
    s = t.assign(i=t.tier.map(order)).sort_values("pred")
    assert s.i.is_monotonic_increasing


def test_escalation_respects_false_alarm_caps_and_is_ordered():
    o = _fake_oof()
    thr, grid = K.select_escalation(o.y, o.p, CUTS)
    assert thr["watch"] <= thr["high"] < thr["critical"]
    for lvl in ("watch", "high", "critical"):
        cut = CUTS[2] if lvl == "critical" else CUTS[1]
        m = K._flag_metrics((o.y >= cut).values, (o.p >= thr[lvl]).values)
        if thr[lvl] > min(C.ESC_FACTORS) * cut + 1e-9:        # chosen factor is not the floor of the grid
            assert m["false_alarm_rate"] <= C.ESC_FAR_CAPS[lvl] + 1e-12, lvl
    # not degenerate: must not flag everything
    assert (o.p >= thr["high"]).mean() < 0.5


def test_escalation_never_lowers_recall_and_tier_index_is_monotone():
    o = _fake_oof()
    thr, _ = K.select_escalation(o.y, o.p, CUTS)
    a = o.y >= CUTS[1]
    assert (o.p >= thr["high"])[a].mean() >= (o.p >= CUTS[1])[a].mean()
    grid = np.linspace(0, 100, 2001)
    assert (np.diff(K.tier_index(grid, CUTS, thr)) >= 0).all()


def test_oof_calibration_never_touches_the_test_period():
    b = K.fold_bounds()
    assert max(e for _, e in b) <= pd.Timestamp(C.VAL_END)
    assert all(s <= e for s, e in b)
    m = json.loads((C.REPORT_DIR / "metrics.json").read_text())
    test_start = pd.Timestamp(m["split"]["test"][0])
    assert max(e for _, e in b) < test_start


def test_improved_vs_original_is_reported_side_by_side_and_actually_improves_detection():
    m = json.loads((C.REPORT_DIR / "metrics.json").read_text())
    for w in C.WINDOWS:
        o, n = m["test_by_window_original_method"][w], m["test_by_window"][w]
        assert o["MAE"] == pytest.approx(n["MAE"])             # same model, same point forecasts: only the decision rule differs
        assert o["HC_PR_AUC"] == pytest.approx(n["HC_PR_AUC"])  # ranking quality unchanged -> honest: operating-point change
        assert n["HC_FN"] <= o["HC_FN"] and n["HC_recall"] >= o["HC_recall"]
        assert n["HC_flagged_share"] < 0.5                      # not "everything is High"
        assert n["interval_contains_point"] and o["interval_contains_point"]
    main = m["test_by_window"][C.MAIN_WINDOW]
    assert main["interval_coverage_actual_critical"] > m["test_by_window_original_method"][C.MAIN_WINDOW]["interval_coverage_actual_critical"]


def test_reported_numbers_are_internally_consistent():
    m = json.loads((C.REPORT_DIR / "metrics.json").read_text())["test_by_window"][C.MAIN_WINDOW]
    assert m["HC_TP"] + m["HC_FN"] == round(m["HC_prevalence"] * m["n"])
    assert m["HC_recall"] == pytest.approx(m["HC_TP"] / (m["HC_TP"] + m["HC_FN"]))
    cm = np.array(m["tier_confusion"]); assert cm.sum() == m["n"]
    assert cm[2:, :].sum() == m["HC_TP"] + m["HC_FN"]
    tp = pd.read_csv(C.REPORT_DIR / "test_predictions.csv")
    assert (tp.tier.isin(["High", "Critical"]) & tp.actual_tier.isin(["High", "Critical"])).sum() == m["HC_TP"]


def test_persistence_alternative_is_reported_and_not_better_than_adopted_rule():
    t = pd.read_csv(C.REPORT_DIR / "escalation_persistence_check.csv").set_index("rule")
    assert t.loc["single week (adopted)", "recall"] >= t.loc["two consecutive weeks", "recall"]
    assert (t.false_alarm_rate <= C.ESC_FAR_CAPS["high"] + 1e-12).all()
