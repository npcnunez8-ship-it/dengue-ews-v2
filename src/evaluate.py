"""Slice-wise honest evaluation: city, ward, season, baselines, risk-tier classification, intervals."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, mean_absolute_error, precision_recall_curve, r2_score

from .config import ALERT_BETA as C_BETA, ALERT_FACTORS, ALERT_MAX_FALSE_ALARM
from .risk import to_tier_idx
from . import calibration as K

# Seasons by forecast-origin month (monsoon Jun-Sep, post-monsoon Oct-Nov, dry Dec-May)
SEASONS = {"Monsoon (Jun-Sep)": [6, 7, 8, 9], "Post-monsoon (Oct-Nov)": [10, 11],
           "Dry (Dec-May)": [12, 1, 2, 3, 4, 5]}


def season_of(months: pd.Series) -> pd.Series:
    inv = {m: k for k, v in SEASONS.items() for m in v}
    return months.map(inv)


def core_metrics(y, p, lo=None, hi=None, cuts=None, alert_cut=None, hc_cut=None, crit_cut=None):
    """Error + interval + outbreak-detection metrics.
    hc_cut / crit_cut: decision thresholds (predicted cases) for flagging High+ / Critical; default = the tier
    cut-points themselves (the ORIGINAL rule).  Actual High/Critical are always defined by the cut-points."""
    y, p = np.asarray(y, float), np.maximum(np.asarray(p, float), 0)
    out = {"n": int(len(y)), "mean_actual": float(y.mean()), "MAE": float(np.abs(y - p).mean()),
           "RMSE": float(np.sqrt(((y - p) ** 2).mean()))}
    out["R2"] = float(r2_score(y, p)) if len(y) > 2 and np.var(y) > 0 else None
    out["relative_MAE"] = out["MAE"] / out["mean_actual"] if out["mean_actual"] > 0 else None
    if lo is not None:
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        inside = (y >= lo) & (y <= hi)
        out["interval_coverage"] = float(inside.mean())
        out["interval_width"] = float((hi - lo).mean())
        out["interval_contains_point"] = bool(((lo <= p + 1e-9) & (p <= hi + 1e-9)).all())
        if cuts is not None:
            crit = y >= cuts[2]
            out["interval_coverage_actual_critical"] = float(inside[crit].mean()) if crit.any() else None
            out["n_actual_critical"] = int(crit.sum())
    if cuts is not None:
        hc_cut = cuts[1] if hc_cut is None else hc_cut
        crit_cut = cuts[2] if crit_cut is None else crit_cut
        a, q = y >= cuts[1], p >= hc_cut                    # actual High/Critical vs flagged High/Critical
        tp, fp, fn = int((a & q).sum()), int((~a & q).sum()), int((a & ~q).sum())
        tn = int((~a & ~q).sum())
        prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        out.update({"HC_decision_cut": float(hc_cut), "HC_prevalence": float(a.mean()), "HC_TP": tp, "HC_FP": fp, "HC_FN": fn,
                    "HC_precision": prec if tp + fp else None, "HC_recall": rec if tp + fn else None,
                    "HC_F1": 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0,
                    "HC_false_alarm_rate": fp / max(fp + tn, 1), "HC_flagged_share": float(q.mean())})
        ca, cq = y >= cuts[2], p >= crit_cut                # Critical only
        ctp, cfp, cfn = int((ca & cq).sum()), int((~ca & cq).sum()), int((ca & ~cq).sum())
        out.update({"CRIT_decision_cut": float(crit_cut), "CRIT_prevalence": float(ca.mean()), "CRIT_TP": ctp, "CRIT_FP": cfp, "CRIT_FN": cfn,
                    "CRIT_recall": ctp / (ctp + cfn) if ctp + cfn else None, "CRIT_precision": ctp / (ctp + cfp) if ctp + cfp else None,
                    "CRIT_false_alarm_rate": cfp / max(int((~ca).sum()), 1)})
        # threshold-free ranking quality of the predicted case count for High/Critical outbreaks
        out["HC_PR_AUC"] = float(average_precision_score(a, p)) if 0 < a.sum() < len(a) else None
        if alert_cut is not None:
            out.update({f"ALERT_{k}": v for k, v in alert_metrics(a, p, alert_cut).items()})
    return out


def slice_report(meta: pd.DataFrame, y, p, lo, hi, cuts, alert_cut=None, hc_cut=None, crit_cut=None):
    """meta needs columns date, city, ward_id. Returns dict of DataFrames (city, season, ward, actual_tier, pred_tier)."""
    d = meta.reset_index(drop=True).copy()
    d["y"], d["p"], d["lo"], d["hi"] = np.asarray(y, float), np.asarray(p, float), np.asarray(lo, float), np.asarray(hi, float)
    d["season"] = season_of(d["date"].dt.month)
    kw = dict(cuts=cuts, alert_cut=alert_cut, hc_cut=hc_cut, crit_cut=crit_cut)
    rows = lambda g, name: [{name: k, **core_metrics(x.y, x.p, x.lo, x.hi, **kw)} for k, x in g]
    city = pd.DataFrame(rows(d.groupby("city"), "city"))
    season = pd.DataFrame(rows(d.groupby("season"), "season"))
    ward = pd.DataFrame(rows(d.groupby(["city", "ward_id"]), "ward")).assign(ward=lambda t: t.ward.map(lambda k: k[1]))
    overall = core_metrics(d.y, d.p, d.lo, d.hi, **kw)
    d["actual_tier"] = to_tier_idx(d.y, cuts)
    names = ["Low", "Medium", "High", "Critical"]
    at = pd.DataFrame(rows(d.groupby("actual_tier"), "actual_tier")); at["actual_tier"] = at.actual_tier.map(lambda i: names[int(i)])
    thr = {"high": cuts[1] if hc_cut is None else hc_cut, "critical": cuts[2] if crit_cut is None else crit_cut}
    d["pred_tier"] = K.tier_index(d.p, cuts, thr)
    pt = pd.DataFrame(rows(d.groupby("pred_tier"), "pred_tier")); pt["pred_tier"] = pt.pred_tier.map(lambda i: names[int(i)])
    # flag cities whose relative error is >25% above the overall relative error
    city["unreliable_flag"] = city.relative_MAE > 1.25 * overall["relative_MAE"]
    return {"overall": overall, "city": city, "season": season, "ward": ward, "actual_tier": at, "pred_tier": pt}


def alert_metrics(actual_hc, p, alert_cut):
    """Precision / recall / F2 / false-alarm rate of the flag `predicted cases >= alert_cut`."""
    q = np.asarray(p, float) >= alert_cut
    a = np.asarray(actual_hc, bool)
    tp, fp, fn, tn = int((a & q).sum()), int((~a & q).sum()), int((a & ~q).sum()), int((~a & ~q).sum())
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    b2 = C_BETA ** 2
    f = (1 + b2) * prec * rec / (b2 * prec + rec) if (b2 * prec + rec) > 0 else 0.0
    return {"precision": prec if tp + fp else None, "recall": rec if tp + fn else None, "F2": f,
            "false_alarm_rate": fp / max(fp + tn, 1), "alert_cut": float(alert_cut)}


def pick_alert_cut(y_val, p_val, high_cut):
    """Choose the sensitive alert threshold on VALIDATION: maximise F2 s.t. false-alarm rate <= cap."""
    a = np.asarray(y_val, float) >= high_cut
    rows = []
    for f in ALERT_FACTORS:
        m = alert_metrics(a, np.maximum(p_val, 0), f * high_cut)
        rows.append({"factor": f, **m})
    ok = [r for r in rows if r["false_alarm_rate"] <= ALERT_MAX_FALSE_ALARM]
    best = max(ok, key=lambda r: (r["F2"], r["factor"])) if ok else next(r for r in rows if r["factor"] == 1.0)
    return best["alert_cut"], rows


def pr_curve_frame(y, p, high_cut):
    a = np.asarray(y, float) >= high_cut
    pr, rc, th = precision_recall_curve(a, np.maximum(p, 0))
    return pd.DataFrame({"threshold_cases": np.append(th, np.nan), "precision": pr, "recall": rc})


def calibration_by_level(y, p, lo, hi, n_bins=5):
    """Interval coverage within quantile bins of the predicted level (is the interval calibrated at every risk level?)."""
    d = pd.DataFrame({"y": np.asarray(y, float), "p": np.asarray(p, float), "lo": np.asarray(lo, float), "hi": np.asarray(hi, float)})
    d["bin"] = pd.qcut(d.p.rank(method="first"), n_bins, labels=False)
    g = d.groupby("bin")
    return pd.DataFrame({"n": g.size(), "mean_pred": g.p.mean(), "mean_actual": g.y.mean(),
                         "coverage": g.apply(lambda x: ((x.y >= x.lo) & (x.y <= x.hi)).mean(), include_groups=False),
                         "width": g.apply(lambda x: (x.hi - x.lo).mean(), include_groups=False)}).reset_index()
