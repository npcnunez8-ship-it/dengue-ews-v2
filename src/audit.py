"""Leakage / validity audit + chronological hyper-parameter check. Run: python -m src.audit
Writes reports/audit.json and reports/hpo_hgb.csv. Nothing here touches the TEST period."""
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor as HGB

from . import config as C
from .features import add_target, build_dataset, feature_sets
from .train import SEED, metrics, split_masks


def _setup():
    df = build_dataset(with_history=False)
    y = add_target(df, C.WINDOWS[C.MAIN_WINDOW])
    cols = feature_sets(df)["weather_only"]
    tr, va, ts = split_masks(df, y)
    ok = df[cols].notna().all(axis=1)
    return df, y, cols, tr & ok, va & ok


def _fit(cols, df, y, tr, **kw):
    p = dict(loss="poisson", max_iter=300, learning_rate=0.05, max_leaf_nodes=24, min_samples_leaf=30,
             l2_regularization=1.0, random_state=SEED); p.update(kw)
    return HGB(**p).fit(df.loc[tr, cols], y[tr])


def leakage_audit():
    df, y, cols, tr, va = _setup()
    rep, yv = {}, y[va]
    g = df.groupby("ward_id", sort=False)
    # 1) causal direction: past rain should explain cases; FUTURE rain must not exceed it
    past = {k: float(np.nanmean(g.apply(lambda x: x.rainfall_mm.shift(k).corr(x.dengue_cases), include_groups=False))) for k in range(0, 7)}
    fut = {k: float(np.nanmean(g.apply(lambda x: x.rainfall_mm.shift(-k).corr(x.dengue_cases), include_groups=False))) for k in range(1, 5)}
    rep["corr_cases_vs_past_rain_lag"] = past
    rep["corr_cases_vs_future_rain_lead"] = fut
    rep["peak_past_lag_weeks"] = max(past, key=past.get)
    # 2) permutation placebo: shuffle the target within each ward -> a leak-free pipeline must collapse to ~0 R2
    rng = np.random.default_rng(SEED)
    yp = y.copy()
    for _, ix in df.groupby("ward_id").groups.items():
        yp.loc[ix] = rng.permutation(y.loc[ix].values)
    m = _fit(cols, df, yp, tr & yp.notna())
    rep["placebo_shuffled_target_val"] = metrics(yv, m.predict(df.loc[va, cols]))
    # 3) how much is explained without any weather? (ward attributes + season only)
    sub = list(C.STATIC_COLS) + ["week_sin", "week_cos"]
    rep["static_plus_season_only_val"] = metrics(yv, _fit(sub, df, y, tr).predict(df.loc[va, sub]))
    m_full = _fit(cols, df, y, tr)
    rep["full_weather_model_val"] = metrics(yv, m_full.predict(df.loc[va, cols]))
    # 4) ward-by-week-of-year climatology (train only) as a strong seasonal reference
    wk = df["date"].dt.isocalendar().week.astype(int).clip(upper=52)
    clim = y[tr].groupby([df.loc[tr, "ward_id"], wk[tr]]).mean()
    pc = pd.Series(list(zip(df.loc[va, "ward_id"], wk[va]))).map(clim).fillna(y[tr].mean()).values
    rep["ward_week_climatology_val"] = metrics(yv, pc)
    # 5) synthetic-data signature: residual autocorrelation (real surveillance data is usually clustered)
    res = (yv - np.maximum(m_full.predict(df.loc[va, cols]), 0))
    r = res.groupby(df.loc[va, "ward_id"]).apply(lambda s: s.autocorr(1))
    rep["val_residual_lag1_autocorr_mean"] = float(r.mean())
    rep["val_R2_flag"] = "R2 above 0.9 is unusually high for real surveillance data -> treat as synthetic-data upper bound"
    return rep


def horizon_strategy_comparison():
    """Direct vs pooled vs derived multi-horizon strategies, VALIDATION only (never the test period).

    Targets: cases in a single week h in {2,3,4} ahead.
      direct  : one HistGB-Poisson per horizon (what the app ships)
      pooled  : one model with the horizon as an extra feature (stacked rows)
      derived : the weeks-2-4 window model's prediction divided by 3 (no horizon-specific information)
    """
    df, y_main, cols, _, _ = _setup()
    rows = []
    stack_tr, stack_va = [], []
    for h in (2, 3, 4):
        y = add_target(df, (h,))
        tr, va, _ = split_masks(df, y)
        ok = df[cols].notna().all(axis=1)
        tr, va = tr & ok, va & ok
        stack_tr.append(df.loc[tr, cols].assign(horizon=h, _y=y[tr], _id=df.index[tr]))
        stack_va.append(df.loc[va, cols].assign(horizon=h, _y=y[va], _id=df.index[va]))
    tr_all, va_all = pd.concat(stack_tr), pd.concat(stack_va)
    from sklearn.ensemble import HistGradientBoostingRegressor as _H
    kw = dict(loss="poisson", max_iter=300, learning_rate=0.05, max_leaf_nodes=24, min_samples_leaf=30, l2_regularization=1.0, random_state=SEED)
    pooled = _H(**kw).fit(tr_all[cols + ["horizon"]], tr_all["_y"])
    window_model = _fit(cols, df, y_main, split_masks(df, y_main)[0] & df[cols].notna().all(axis=1))
    for h in (2, 3, 4):
        yv = va_all[va_all.horizon == h]
        direct = _H(**kw).fit(stack_tr[h - 2][cols], stack_tr[h - 2]["_y"])
        pd_direct = direct.predict(yv[cols])
        pd_pooled = pooled.predict(yv[cols + ["horizon"]])
        pd_derived = window_model.predict(yv[cols]) / 3.0
        for name, pr in (("direct (per-horizon)", pd_direct), ("pooled (+horizon feature)", pd_pooled), ("derived (window/3)", pd_derived)):
            rows.append({"horizon_weeks": h, "strategy": name, "n": len(yv), **metrics(yv["_y"], pr)})
    t = pd.DataFrame(rows)
    return t


def tune_hgb(n_iter=12):
    df, y, cols, tr, va = _setup()
    rng = np.random.default_rng(SEED)
    rows = [dict(name="default", max_leaf_nodes=24, learning_rate=0.05, max_iter=300, min_samples_leaf=30, l2_regularization=1.0)]
    for i in range(n_iter):
        rows.append(dict(name=f"rs{i}", max_leaf_nodes=int(rng.choice([8, 12, 16, 24, 32])), learning_rate=float(rng.choice([0.03, 0.05, 0.08])),
                         max_iter=int(rng.choice([200, 300, 500])), min_samples_leaf=int(rng.choice([20, 30, 60, 100])),
                         l2_regularization=float(rng.choice([0.0, 1.0, 5.0, 10.0]))))
    out = []
    for r in rows:
        kw = {k: v for k, v in r.items() if k != "name"}
        t = time.time(); m = _fit(cols, df, y, tr, **kw)
        out.append({**r, **metrics(y[va], m.predict(df.loc[va, cols])), "fit_s": round(time.time() - t, 1)})
    res = pd.DataFrame(out).sort_values("MAE").reset_index(drop=True)
    return res


if __name__ == "__main__":
    a = leakage_audit()
    st = horizon_strategy_comparison()
    st.to_csv(C.REPORT_DIR / "horizon_strategy_validation.csv", index=False)
    a["horizon_strategy_val_mean_MAE"] = st.groupby("strategy").MAE.mean().round(4).to_dict()
    a["horizon_strategy_selected"] = st.groupby("strategy").MAE.mean().idxmin()
    h = tune_hgb()
    d = float(h.loc[h.name == "default", "MAE"].iloc[0]); b = float(h.MAE.iloc[0])
    a["hpo"] = {"default_val_MAE": d, "best_val_MAE": b, "best_name": h.name.iloc[0], "rel_gain": (d - b) / d,
                "adopt_threshold": 0.02, "adopted": bool((d - b) / d > 0.02)}
    h.to_csv(C.REPORT_DIR / "hpo_hgb.csv", index=False)
    (C.REPORT_DIR / "audit.json").write_text(json.dumps(a, indent=2))
    print(json.dumps(a, indent=1)); print(h.head(5).to_string())
