"""Fast environment + pipeline self-check (a few seconds).  Run:  python -m src.selfcheck
Exit code 0 = everything needed to run the app and trust its numbers is in place."""
import json
import sys
import time

import numpy as np
import pandas as pd

from . import config as C


def _run(checks):
    ok_all = True
    for name, fn in checks:
        t = time.time()
        try:
            detail = fn()
            print(f"  PASS  {name}" + (f"  ({detail})" if detail else "") + f"  [{time.time() - t:.1f}s]")
        except Exception as e:  # noqa: BLE001
            ok_all = False
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    return ok_all


def main() -> int:
    from .features import add_target, build_dataset, load_raw
    from .model_io import read_bundle, versions
    from .predict import load_bundle, predict_frame
    from .quality import data_quality_report
    print("dengue-ews selfcheck  |", versions())
    state = {}

    def data():
        raw = load_raw()                                   # raises on duplicate ward-weeks / missing columns
        q = data_quality_report(raw)
        assert q["duplicate_ward_weeks"] == 0 and q["wards_with_calendar_gaps"] == 0, q
        state["raw"] = raw
        return f"{q['rows']} rows, {q['wards']} wards, {q['date_min']}..{q['date_max']}"

    def lags():
        df = build_dataset(with_history=False); state["df"] = df
        w = df[df.ward_id == df.ward_id.iloc[0]].set_index("date")
        t = w.index[20]
        for k in (1, 3, 8):
            assert w.loc[t, f"rain_lag_{k}w"] == w.loc[t - pd.Timedelta(weeks=k), "rainfall_mm"]
        y = add_target(state["raw"])
        assert y.notna().sum() > 0
        return "calendar-aligned lags/rolls verified on a sample ward"

    def bundle():
        note = "shipped bundle loads as-is"
        try:
            b = read_bundle()
        except Exception as e:  # noqa: BLE001  (different numpy/scikit-learn build -> designed auto-rebuild path)
            from .model_io import ensure_bundle
            b = ensure_bundle(log=lambda *_: None)
            note = f"shipped bundle unusable here ({type(e).__name__}); rebuilt from the CSV"
        assert set(b["windows"]) == set(C.WINDOWS)
        return f"{note}; scikit-learn {b['meta']['versions']['scikit_learn']}, {len(b['windows'])} windows"

    def inference():
        df = state["df"]; b = load_bundle()
        t0 = time.time(); p = predict_frame(df.dropna(subset=b["features"]).tail(5000))
        ms = (time.time() - t0) * 1000
        assert (p.lo <= p.pred + 1e-9).all() and (p.pred <= p.hi + 1e-9).all()
        assert set(p.tier.dropna()) <= set(C.RISK_TIERS)
        return f"5000 ward-weeks scored in {ms:.0f} ms, intervals contain their forecasts"

    def reports():
        m = json.loads((C.REPORT_DIR / "metrics.json").read_text())
        t, o = m["test_by_window"][C.MAIN_WINDOW], m["test_by_window_original_method"][C.MAIN_WINDOW]
        assert t["HC_FN"] <= o["HC_FN"]
        return f"test MAE {t['MAE']:.2f}; High/Critical missed {o['HC_FN']} -> {t['HC_FN']}"

    ok = _run([("dataset integrity", data), ("lag pipeline alignment", lags), ("model bundle loads", bundle),
               ("fast inference", inference), ("published reports", reports)])
    print("SELFCHECK " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
