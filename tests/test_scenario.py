import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.predict import load_bundle, load_features, predict_frame
from src.scenario import N_HISTORY, all_horizons, apply_scenario, predict_history, validate_history, ward_history

DF = load_features()
B = load_bundle()
WARD, DATE = "DNCC_Ward_01", pd.Timestamp("2025-08-04")


def test_scenario_matches_batch_inference():
    """Scoring from a raw 9-week history must equal the batch pipeline (train/serve feature parity)."""
    batch = predict_frame(DF[(DF.ward_id == WARD) & (DF.date == DATE)], C.MAIN_WINDOW, B).iloc[0]
    one = predict_history(ward_history(DF, WARD, DATE), C.MAIN_WINDOW, B)
    assert one["pred"] == pytest.approx(batch.pred, rel=1e-9) and one["tier"] == batch.tier


def test_heavier_rain_raises_monsoon_risk_and_dry_lowers_it():
    h = ward_history(DF, WARD, DATE)
    base = predict_history(h, bundle=B)["pred"]
    assert predict_history(apply_scenario(h, 2.0, weeks=5), bundle=B)["pred"] >= base
    assert predict_history(apply_scenario(h, 0.0, weeks=5), bundle=B)["pred"] < base


@pytest.mark.parametrize("bad", ["short", "nan", "neg", "gap"])
def test_invalid_history_is_rejected(bad):
    h = ward_history(DF, WARD, DATE)
    if bad == "short": h = h.tail(5)
    if bad == "nan": h.loc[3, "rainfall_mm"] = np.nan
    if bad == "neg": h.loc[3, "rainfall_mm"] = -4
    if bad == "gap": h = h.drop(index=4)
    with pytest.raises(ValueError):
        validate_history(h)


def test_all_horizons_has_every_window_with_ordered_intervals():
    mh = all_horizons(DF[(DF.ward_id == WARD) & (DF.date == DATE)], B)
    assert list(mh.window) == list(C.WINDOWS)
    assert (mh.lo <= mh.pred + 1e-9).all() and (mh.pred <= mh.hi + 1e-9).all()
    assert set(mh.tier) <= set(C.RISK_TIERS)


def test_placebo_collapses_and_audit_report_is_consistent():
    import json
    a = json.loads((C.REPORT_DIR / "audit.json").read_text())
    assert a["placebo_shuffled_target_val"]["R2"] < 0.15          # leak-free pipeline cannot learn a shuffled target
    assert a["peak_past_lag_weeks"] in (2, 3, 4)                    # 2-4 week breeding clock
