"""Train / validate / select / serialize. Run:  python -m src.train"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import cohen_kappa_score
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from . import calibration as K
from .features import add_target, build_dataset, feature_groups, feature_sets, shifted_target, value_at_offset
from .risk import fit_cutpoints, to_tier_idx
from .evaluate import (slice_report, core_metrics, SEASONS, season_of, pick_alert_cut, pr_curve_frame,
                       calibration_by_level)
from .model_io import save_bundle, versions

try:  # optional, comparison only (not needed at inference)
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None

SEED = 42
ALPHA = 0.10   # 80% prediction interval (quantiles 0.10 / 0.90)


def make_model(name):
    if name == "Mean baseline":
        return DummyRegressor(strategy="mean")
    if name == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    if name == "ExtraTrees":
        return ExtraTreesRegressor(n_estimators=200, min_samples_leaf=3, max_features=0.5,
                                   n_jobs=-1, random_state=SEED)
    if name == "HistGB (squared)":
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=24,
                                             min_samples_leaf=30, l2_regularization=1.0, random_state=SEED)
    if name == "HistGB (Poisson)":
        return HistGradientBoostingRegressor(loss="poisson", max_iter=300, learning_rate=0.05,
                                             max_leaf_nodes=24, min_samples_leaf=30,
                                             l2_regularization=1.0, random_state=SEED)
    if name == "RandomForest":
        return RandomForestRegressor(n_estimators=200, min_samples_leaf=3, max_features=0.5, n_jobs=-1, random_state=SEED)
    if name == "XGBoost (Poisson)":
        return XGBRegressor(objective="count:poisson", n_estimators=400, learning_rate=0.04, max_depth=5,
                            subsample=0.8, colsample_bytree=0.7, min_child_weight=5, n_jobs=2, random_state=SEED)
    raise KeyError(name)


def model_list():
    base = ["Ridge", "RandomForest", "ExtraTrees", "HistGB (squared)", "HistGB (Poisson)"]
    return base + (["XGBoost (Poisson)"] if XGBRegressor is not None else [])


def metrics(y, p):
    p = np.maximum(p, 0)
    return {"MAE": float(mean_absolute_error(y, p)),
            "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "R2": float(r2_score(y, p))}


def split_masks(df, target):
    gap = pd.Timedelta(weeks=C.MAX_HORIZON)
    te, ve = pd.Timestamp(C.TRAIN_END), pd.Timestamp(C.VAL_END)
    ok = target.notna()
    tr = ok & (df["date"] <= te - gap)
    va = ok & (df["date"] > te) & (df["date"] <= ve - gap)
    ts = ok & (df["date"] > ve)
    return tr, va, ts


def grouped_importance(model, X, y, groups, repeats=3):
    rng = np.random.default_rng(SEED)
    base = mean_absolute_error(y, np.maximum(model.predict(X), 0))
    out = {}
    for g, cols in groups.items():
        deltas = []
        for _ in range(repeats):
            Xp = X.copy()
            idx = rng.permutation(len(X))
            Xp[cols] = X[cols].values[idx]
            deltas.append(mean_absolute_error(y, np.maximum(model.predict(Xp), 0)) - base)
        out[g] = float(np.mean(deltas))
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def fit_quantiles(X, y):
    lo = HistGradientBoostingRegressor(loss="quantile", quantile=ALPHA / 2, max_iter=200, learning_rate=0.05,
                                       max_leaf_nodes=16, min_samples_leaf=30, random_state=SEED).fit(X, y)
    hi = HistGradientBoostingRegressor(loss="quantile", quantile=1 - ALPHA / 2, max_iter=200, learning_rate=0.05,
                                       max_leaf_nodes=16, min_samples_leaf=30, random_state=SEED).fit(X, y)
    return lo, hi


def cqr_qhat(lo, hi, X, y):
    """Conformalized quantile regression (Romano et al. 2019) calibration constant."""
    s = np.maximum(lo.predict(X) - y, y - hi.predict(X))
    n = len(y)
    level = min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n)
    return float(np.quantile(s, level))


def interval(lo, hi, X, qhat):
    l, h = lo.predict(X) - qhat, hi.predict(X) + qhat
    return np.maximum(l, 0), np.maximum(h, 0)


TIER_CANDIDATES = [(0.50, 0.80, 0.95), (0.60, 0.85, 0.95), (0.70, 0.90, 0.97), (0.60, 0.80, 0.90)]
INTERVAL_TARGET = 1 - ALPHA


def scaled_q(p, y):
    """Locally-scaled split-conformal constant: |y-p| / sqrt(p+1) (count-like heteroscedasticity)."""
    s = np.abs(y - p) / np.sqrt(p + 1.0)
    n = len(y)
    return float(np.quantile(s, min(1.0, np.ceil((n + 1) * INTERVAL_TARGET) / n)))


def scaled_interval(p, q):
    h = q * np.sqrt(p + 1.0)
    return np.maximum(p - h, 0), p + h


def choose_interval_method(m_tr, lo, hi, Xv, yv, dates_v):
    """Chronological check inside validation: calibrate on the first half, test on the later half."""
    order = np.argsort(dates_v.values, kind="stable")
    half = len(order) // 2
    a, b = order[:half], order[half:]
    p = np.maximum(m_tr.predict(Xv), 0)
    res = {}
    qc = cqr_qhat(lo, hi, Xv.iloc[a], yv.iloc[a])
    l, h = interval(lo, hi, Xv.iloc[b], qc)
    res["cqr"] = (float(((yv.iloc[b] >= l) & (yv.iloc[b] <= h)).mean()), float((h - l).mean()))
    qs = scaled_q(p[a], yv.iloc[a].values)
    l, h = scaled_interval(p[b], qs)
    res["scaled"] = (float(((yv.iloc[b].values >= l) & (yv.iloc[b].values <= h)).mean()), float((h - l).mean()))
    ok = {k: v for k, v in res.items() if v[0] >= INTERVAL_TARGET - 0.03}
    pick = min(ok, key=lambda k: ok[k][1]) if ok else "cqr"
    return pick, res


def window_interval(bw, X):
    """ORIGINAL interval method (validation-calibrated CQR / scaled), with the containment fix lo <= point <= hi."""
    p = np.maximum(bw["model"].predict(X), 0)
    if bw["interval_method"] == "scaled":
        l, h = scaled_interval(p, bw["qhat"])
    else:
        l, h = interval(bw["lo"], bw["hi"], X, bw["qhat"])
    return p, np.minimum(l, p), np.maximum(h, p)


def pick_tiers(y_tr, y_va, p_va):
    best, rows = None, []
    for qs in TIER_CANDIDATES:
        cuts = fit_cutpoints(y_tr, qs)
        kap = cohen_kappa_score(to_tier_idx(y_va, cuts), to_tier_idx(p_va, cuts), weights="quadratic")
        rows.append({"quantiles": list(qs), "cutpoints": cuts, "val_quadratic_kappa": float(kap)})
        if best is None or kap > best[0]:
            best = (kap, qs)
    return best[1], rows


def _finish_bundle(bundle, df, cols, trva, tr, va, split_info):
    """Seasonal-normal reference (train+val only), metadata, portable serialization."""
    wk = df["date"].dt.isocalendar().week.astype(int).clip(upper=52)
    bundle["climatology"] = df.loc[trva, cols].groupby(wk[trva]).mean()
    bundle["static_ref"] = df.loc[trva, cols].mean()
    bundle["meta"] = {"trained_on": f"{split_info['train'][0]}..{split_info['val'][1]}", "seed": SEED}
    save_bundle(bundle)


def main(bundle_only=False):
    """Full experiment (default) or `bundle_only`: refit + recalibrate the already-selected model only
    (no model comparison, no test scoring, no report files) - used to rebuild the bundle on a new machine."""
    t0 = time.time()
    df = build_dataset(with_history=True)
    fsets = feature_sets(df)
    y_main = add_target(df, C.WINDOWS[C.MAIN_WINDOW])
    tr, va, ts = split_masks(df, y_main)
    fs_all = sorted(set(sum(fsets.values(), [])))
    okf = df[fs_all].notna().all(axis=1)
    # identical evaluation samples for ML models AND baselines: seasonal-naive needs the target 52 weeks earlier
    snaive_main = shifted_target(df, y_main, 52)
    ev = okf & snaive_main.notna()
    tr, va, ts = tr & okf, va & ev, ts & ev
    print(f"rows train/val/test = {tr.sum()}/{va.sum()}/{ts.sum()}")
    d_ = lambda m: [str(df.loc[m, 'date'].min().date()), str(df.loc[m, 'date'].max().date()), int(m.sum())]
    split_info = {"train": d_(tr), "val": d_(va), "test": d_(ts), "purge_gap_weeks": C.MAX_HORIZON,
                  "note": "val/test evaluation rows additionally require a seasonal-naive reference (target 52 weeks earlier)"}

    if bundle_only:
        try:
            sel = json.loads((C.REPORT_DIR / "metrics.json").read_text())["selected"]
            model_name, feat_name, use_hist = sel["model"], sel["features"], False
        except Exception:  # noqa: BLE001
            model_name, feat_name, use_hist = "HistGB (Poisson)", "weather_only", False
        cols = fsets[feat_name]
        if model_name.startswith("XGBoost") and XGBRegressor is None:
            model_name = "HistGB (Poisson)"
    else:
        # ---- 1. Baselines + model x feature-set comparison (validation only, identical samples) ----
        nh = len(C.WINDOWS[C.MAIN_WINDOW])
        rows = []
        def add(feat, model, y, p):
            r = metrics(y, p); rows.append({"features": feat, "model": model, "n_val": int(len(y)), **r})
            print(f"{feat:22s} {model:20s} MAE={r['MAE']:.3f} RMSE={r['RMSE']:.3f} R2={r['R2']:.3f}")
        yv = y_main[va]
        add("-", "Mean baseline", yv, np.full(len(yv), y_main[tr].mean()))
        add("-", "Persistence (cases_t x horizons)", yv, df.loc[va, "dengue_cases"].values * nh)
        add("-", "Seasonal-naive (52w earlier)", yv, snaive_main[va].values)
        for fname, cols in fsets.items():
            for mname in model_list():
                if mname == "RandomForest" and fname != "weather_only":
                    continue
                m = make_model(mname).fit(df.loc[tr, cols], y_main[tr])
                add(fname, mname, yv, m.predict(df.loc[va, cols]))
        comp = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)

        ml = comp[comp.features != "-"]
        def pick(fname):
            c = ml[ml.features == fname]
            top, core = c.iloc[0], c[c.model != "XGBoost (Poisson)"].iloc[0]
            # parsimony: XGBoost adds a runtime dependency, so it must beat the best sklearn model by >1%
            return top if (top.model != "XGBoost (Poisson)" or top.MAE < 0.99 * core.MAE) else core
        wo, wh = pick("weather_only"), pick("weather_plus_history")
        use_hist = wh.MAE < 0.97 * wo.MAE      # history only if >3% better on validation
        best = wh if use_hist else wo
        feat_name, model_name = best.features, best.model
        cols = fsets[feat_name]
        print(f"\nSELECTED: {model_name} on {feat_name} (val MAE {best.MAE:.3f}); history adopted={use_hist}")


    # ---- 2. Per-window bundles ----
    bundle = {"features": cols, "feature_set": feat_name, "model_name": model_name,
              "windows": {}, "alpha": ALPHA, "groups": feature_groups(cols)}
    test_report, test_report_orig, window_val, tier_sel, interval_sel, esc_sel, slices, slices_orig, alert_sel = {}, {}, {}, {}, {}, {}, {}, {}, {}
    extra = {}                                   # main-window diagnostic tables written to reports/
    fit_reg = lambda X_, y_: make_model(model_name).fit(X_, y_)
    for wname, hz in C.WINDOWS.items():
        y = add_target(df, hz)
        sn = shifted_target(df, y, 52)
        trw, vaw, tsw = (m & y.notna() for m in (tr, va, ts))
        m_tr = make_model(model_name).fit(df.loc[trw, cols], y[trw])
        lo, hi = fit_quantiles(df.loc[trw, cols], y[trw])
        Xv, yvw = df.loc[vaw, cols], y[vaw]
        pv = np.maximum(m_tr.predict(Xv), 0)
        window_val[wname] = metrics(yvw, pv)
        meth, meth_res = choose_interval_method(m_tr, lo, hi, Xv, yvw, df.loc[vaw, "date"])
        qhat = cqr_qhat(lo, hi, Xv, yvw) if meth == "cqr" else scaled_q(pv, yvw.values)
        interval_sel[wname] = {"original_method_chosen": meth, "val_second_half_(coverage,width)": meth_res}
        _, tier_rows = pick_tiers(y[trw].values, yvw.values, pv)           # sensitivity table only - NOT used to choose
        qs = C.TIER_QUANTILES
        cuts = fit_cutpoints(y[trw].values, qs)
        tier_sel[wname] = {"fixed_quantiles": list(qs), "validation_kappa_sensitivity": tier_rows,
                           "note": "tier quantiles are fixed policy parameters; kappa table is a sensitivity check only"}
        alert_cut, alert_rows = pick_alert_cut(yvw.values, pv, cuts[1])      # ORIGINAL watch flag (single validation season)
        alert_sel[wname] = {"chosen_alert_cut": alert_cut, "high_cut": cuts[1], "validation_grid": alert_rows}

        # ---- NEW: chronological OOF calibration of High/Critical thresholds and intervals (pre-test data only) ----
        okw = y.notna() & df[cols].notna().all(axis=1)
        oof = K.oof_predictions(df, cols, y, fit_reg, okw)
        esc, esc_grid = K.select_escalation(oof.y, oof.p, cuts)
        iv_mode, iv_res = K.choose_interval(oof, cuts, ALPHA)
        cal = K.fit_interval(oof.p, oof.y, cuts, ALPHA, iv_mode)
        esc_sel[wname] = {"thresholds_cases": esc, "far_caps": C.ESC_FAR_CAPS, "oof_rows": int(len(oof)),
                          "oof_folds": int(oof.fold.nunique()), "interval_mode": iv_mode, "interval_candidates_heldout": iv_res}

        tv = trw | vaw
        m_f = make_model(model_name).fit(df.loc[tv, cols], y[tv])
        lo_f, hi_f = fit_quantiles(df.loc[tv, cols], y[tv])
        bw = {"horizons": hz, "model": m_f, "lo": lo_f, "hi": hi_f, "qhat": qhat,
              "interval_method": meth, "cutpoints": cuts, "tier_quantiles": list(qs), "alert_cut": alert_cut,
              "esc": esc, "interval_cal": cal, "interval_mode": iv_mode}
        bundle["windows"][wname] = bw
        if bundle_only:
            continue
        # ---- test (touched once per window, after ALL selection) ----
        Xt, yt = df.loc[tsw, cols], y[tsw]
        pt, l_o, h_o = window_interval(bw, Xt)                    # ORIGINAL interval method
        l_n, h_n = K.apply_interval(pt, cuts, cal)                # NEW calibrated interval
        meta = df.loc[tsw, ["date", "city", "ward_id"]]
        rep_o = slice_report(meta, yt, pt, l_o, h_o, cuts, alert_cut)
        rep = slice_report(meta, yt, pt, l_n, h_n, cuts, esc["watch"], esc["high"], esc["critical"])
        a_t = to_tier_idx(yt, cuts)
        p_o, p_n = to_tier_idx(pt, cuts), K.tier_index(pt, cuts, esc)
        base = {"Mean baseline": np.full(len(yt), y[trw].mean()),
                "Persistence (cases_t x horizons)": df.loc[tsw, "dengue_cases"].values * len(hz),
                "Seasonal-naive (52w earlier)": sn[tsw].values}
        base_m = {k: core_metrics(yt, v, cuts=cuts) for k, v in base.items()}
        for r_, p_t_ in ((rep_o, p_o), (rep, p_n)):
            r_["overall"].update({"tier_exact_accuracy": float((a_t == p_t_).mean()), "tier_within_one": float((np.abs(a_t - p_t_) <= 1).mean()),
                                  "tier_confusion": pd.crosstab(pd.Series(a_t, name="a"), pd.Series(p_t_, name="p"))
                                  .reindex(index=range(4), columns=range(4), fill_value=0).values.tolist(),
                                  "cutpoints": cuts, "tier_quantiles": list(qs)})
        rep["overall"].update({"baselines": base_m, "interval_method": iv_mode, "escalation_thresholds": esc})
        rep_o["overall"].update({"baselines": base_m, "interval_method": meth})
        test_report[wname], test_report_orig[wname] = rep["overall"], rep_o["overall"]
        slices[wname], slices_orig[wname] = rep, rep_o
        o, o0 = rep["overall"], rep_o["overall"]
        print(f"[{wname}] TEST n={o['n']} MAE={o['MAE']:.3f} RMSE={o['RMSE']:.3f} R2={o['R2']:.3f} | HC recall orig->new "
              f"{o0['HC_recall']:.2f}->{o['HC_recall']:.2f} FN {o0['HC_FN']}->{o['HC_FN']} FAR {o0['HC_false_alarm_rate']:.3f}->{o['HC_false_alarm_rate']:.3f} "
              f"| cov orig->new {o0['interval_coverage']:.3f}->{o['interval_coverage']:.3f} | crit-cov {o0['interval_coverage_actual_critical']:.2f}->{o['interval_coverage_actual_critical']:.2f}")
        if wname == C.MAIN_WINDOW:
            tdf = meta.copy()
            tdf["actual"], tdf["pred"], tdf["lo"], tdf["hi"] = yt.values, pt, l_n, h_n
            tdf["lo_original"], tdf["hi_original"] = l_o, h_o
            tdf["tier_original"] = [C.RISK_TIERS[i] for i in p_o]
            tdf["tier"] = [C.RISK_TIERS[i] for i in p_n]
            tdf["actual_tier"] = [C.RISK_TIERS[i] for i in a_t]
            tdf["watch"] = pt >= esc["watch"]
            tdf["persistence"], tdf["seasonal_naive"] = base["Persistence (cases_t x horizons)"], base["Seasonal-naive (52w earlier)"]
            tdf.to_csv(C.REPORT_DIR / "test_predictions.csv", index=False)
            pr_curve_frame(yt, pt, cuts[1]).to_csv(C.REPORT_DIR / "pr_curve_test.csv", index=False)
            calibration_by_level(yt, pt, l_n, h_n).to_csv(C.REPORT_DIR / "interval_calibration_test.csv", index=False)
            pd.DataFrame(alert_rows).to_csv(C.REPORT_DIR / "alert_threshold_validation.csv", index=False)
            for k in ("city", "season", "ward", "actual_tier", "pred_tier"):
                rep[k].to_csv(C.REPORT_DIR / f"test_by_{k}.csv", index=False)
            for k in ("season", "actual_tier"):
                rep_o[k].to_csv(C.REPORT_DIR / f"test_by_{k}_original_method.csv", index=False)
            groups = feature_groups(cols)
            imp = grouped_importance(m_tr, Xv.reset_index(drop=True), yvw.reset_index(drop=True), groups)
            # --- diagnostics: OOF grid, per-fold stability, nested check, scorer alternatives, missed-outbreak analysis ---
            esc_grid.to_csv(C.REPORT_DIR / "escalation_oof_grid.csv", index=False)
            fold_rows = []
            for kf, g in oof.groupby("fold"):
                for lvl, cut in (("watch", cuts[1]), ("high", cuts[1]), ("critical", cuts[2])):
                    fold_rows.append({"fold": int(kf), "first_date": str(g.date.min().date()), "level": lvl, "threshold_cases": esc[lvl],
                                      "n_pos": int((g.y >= cut).sum()), **{k: v for k, v in K._flag_metrics((g.y >= cut).values, (g.p >= esc[lvl]).values).items()
                                                                           if k in ("recall", "precision", "false_alarm_rate", "fn")}})
            pd.DataFrame(fold_rows).to_csv(C.REPORT_DIR / "escalation_oof_by_fold.csv", index=False)
            K.nested_check(oof, cuts).to_csv(C.REPORT_DIR / "escalation_nested_check.csv", index=False)
            K.persistence_check(oof, cuts).to_csv(C.REPORT_DIR / "escalation_persistence_check.csv", index=False)
            hq = cuts[1]
            alt = K.scorer_alternatives(df, cols, y, okw, hq, {
                "Poisson regression (shipped)": (fit_reg, lambda m, X: np.maximum(m.predict(X), 0), lambda v: v),
                "HistGB classifier P(y>=High)": (lambda X_, y_: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=24, min_samples_leaf=30,
                                                                                               l2_regularization=1.0, random_state=SEED).fit(X_, y_),
                                                 lambda m, X: m.predict_proba(X)[:, 1], lambda v: (v >= hq).astype(int)),
                "HistGB 70th-percentile quantile": (lambda X_, y_: HistGradientBoostingRegressor(loss="quantile", quantile=0.7, max_iter=200, learning_rate=0.05,
                                                                                                 max_leaf_nodes=16, min_samples_leaf=30, random_state=SEED).fit(X_, y_),
                                                    lambda m, X: m.predict(X), lambda v: v)})
            base_rec = float(alt.loc[0, "recall_at_FAR5"])
            alt["gain_recall_at_FAR5_vs_shipped"] = alt["recall_at_FAR5"] - base_rec
            alt["adopted"] = (alt.index == 0) | (alt["gain_recall_at_FAR5_vs_shipped"] >= 0.03)   # pre-set rule: >= +3 points
            alt.to_csv(C.REPORT_DIR / "risk_scorer_alternatives_oof.csv", index=False)
            miss = tdf.copy()
            miss["prev_week_target"] = shifted_target(df, y, 1)[tsw].values
            miss["month"] = miss.date.dt.month
            miss["pred_over_actual"] = miss.pred / miss.actual.clip(lower=1)
            miss["onset_week"] = miss.prev_week_target < cuts[1]
            fn_o = miss[(miss.actual >= cuts[1]) & (miss.pred < cuts[1])]
            fn_n = miss[(miss.actual >= cuts[1]) & (miss.pred < esc["high"])]
            ana = []
            for lab, fn in (("original rule (pred >= High cut)", fn_o), ("escalated High+ (pred >= escalated threshold)", fn_n)):
                ana.append({"rule": lab, "missed_High_or_Critical": int(len(fn)), "of_actual_HC": int((miss.actual >= cuts[1]).sum()),
                            "median_pred_over_actual": float(fn.pred_over_actual.median()), "share_in_onset_week": float(fn.onset_week.mean()),
                            "share_caught_by_watch_flag": float((fn.pred >= esc["watch"]).mean()),
                            "months": json.dumps({int(k): int(v) for k, v in fn.month.value_counts().sort_index().items()}),
                            "top_wards": json.dumps(fn.ward_id.value_counts().head(5).to_dict()),
                            "by_city": json.dumps(fn.city.value_counts().to_dict())})
            pd.DataFrame(ana).to_csv(C.REPORT_DIR / "missed_outbreaks_analysis.csv", index=False)
            extra["interval_oof"] = iv_res

    if bundle_only:
        _finish_bundle(bundle, df, cols, tr | va, tr, va, split_info)
        print(f"bundle rebuilt in {time.time()-t0:.0f}s ({model_name} on {feat_name})")
        return

    # ---- 3. Rolling-origin chronological CV (folds end before the final test period) ----
    folds = []
    for k, (sdt, edt) in enumerate(K.fold_bounds()):
        if sdt.year < 2021:
            continue
        trn = y_main.notna() & okf & (df.date <= sdt - pd.Timedelta(weeks=4))
        tst = y_main.notna() & ev & (df.date >= sdt) & (df.date <= edt)
        m = make_model(model_name).fit(df.loc[trn, cols], y_main[trn])
        pp = m.predict(df.loc[tst, cols])
        folds.append({"fold": f"train<{sdt.year} -> test {sdt.date()}..{edt.date()}", "n": int(tst.sum()), **metrics(y_main[tst], pp),
                      "persistence_MAE": metrics(y_main[tst], df.loc[tst, "dengue_cases"].values * nh)["MAE"],
                      "seasonal_naive_MAE": metrics(y_main[tst], snaive_main[tst].values)["MAE"]})
    cv = pd.DataFrame(folds)
    print(cv.round(3))

    _finish_bundle(bundle, df, cols, tr | va, tr, va, split_info)

    # ---- 5. Reports ----
    comp.to_csv(C.REPORT_DIR / "model_comparison_validation.csv", index=False)
    cv.to_csv(C.REPORT_DIR / "rolling_origin_cv.csv", index=False)
    cmp_rows = []
    for wname in C.WINDOWS:
        for label, rep_ in (("original", test_report_orig[wname]), ("improved", test_report[wname])):
            cmp_rows.append({"window": wname, "method": label, **{k: rep_.get(k) for k in (
                "n", "MAE", "RMSE", "R2", "HC_PR_AUC", "HC_decision_cut", "HC_recall", "HC_precision", "HC_FN", "HC_FP", "HC_false_alarm_rate", "HC_flagged_share",
                "CRIT_decision_cut", "CRIT_recall", "CRIT_precision", "CRIT_FN", "n_actual_critical", "interval_coverage", "interval_width",
                "interval_coverage_actual_critical", "tier_exact_accuracy", "tier_within_one")}})
    pd.DataFrame(cmp_rows).to_csv(C.REPORT_DIR / "risk_comparison_original_vs_improved.csv", index=False)
    summary = {"split": split_info, "selected": {"model": model_name, "features": feat_name,
               "history_adopted": bool(use_hist), "n_features": len(cols)},
               "validation_by_window": window_val, "test_by_window": test_report, "test_by_window_original_method": test_report_orig,
               "interval_method_selection": interval_sel, "escalation_selection": esc_sel, "alert_threshold_selection_original": alert_sel,
               "environment": versions(), "tier_threshold_selection": tier_sel,
               "city_flags": slices[C.MAIN_WINDOW]["city"][["city", "relative_MAE", "unreliable_flag"]].to_dict("records"),
               "grouped_importance_val_MAE_increase": imp, "train_seconds": round(time.time() - t0, 1)}
    (C.REPORT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train / validate / serialize the dengue early-warning model")
    ap.add_argument("--bundle-only", action="store_true",
                    help="only refit + recalibrate the already selected model (fast; no reports)")
    main(bundle_only=ap.parse_args().bundle_only)
