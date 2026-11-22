"""Chronological out-of-fold (OOF) calibration of (a) the High/Critical decision thresholds and
(b) the prediction intervals.  Nothing here reads the TEST period.

Why: a point forecast of "expected cases" regresses toward the mean, so at outbreak levels it under-predicts
(observed on OOF data: mean actual > mean predicted in every predicted-level bin above 5 cases).  Flagging
`pred >= High cut` therefore misses many true High weeks.  Instead of tuning on a single validation season we
pool several chronological OOF seasons (2019-2024H1) and choose, for each tier, the *lowest* decision threshold
whose false-alarm rate stays within a stated budget (config.ESC_FAR_CAPS).  This is a cost-sensitive operating
point, not a change of the epidemiological definition of High/Critical (the cut-points are unchanged).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from . import config as C


# --------------------------------------------------------------------------------------- OOF forecasts
def fold_bounds():
    last = pd.Timestamp(C.VAL_END) - pd.Timedelta(weeks=C.MAX_HORIZON)
    return [(pd.Timestamp(s), last if e is None else pd.Timestamp(e)) for s, e in C.OOF_FOLDS]


def oof_predictions(df, cols, y, fit, ok):
    """Expanding-window OOF point forecasts.  `fit(X, y)` -> fitted estimator; `ok` marks usable rows.
    Training rows of a fold end MAX_HORIZON weeks before the fold's first forecast date (purge gap)."""
    gap = pd.Timedelta(weeks=C.MAX_HORIZON)
    parts = []
    for k, (s, e) in enumerate(fold_bounds()):
        trm = ok & (df["date"] <= s - gap)
        tsm = ok & (df["date"] >= s) & (df["date"] <= e)
        if trm.sum() == 0 or tsm.sum() == 0:
            continue
        m = fit(df.loc[trm, cols], y[trm])
        o = df.loc[tsm, ["date", "city", "ward_id"]].copy()
        o["y"], o["p"] = y[tsm].to_numpy(float), np.maximum(m.predict(df.loc[tsm, cols]), 0)
        o["fold"] = k
        parts.append(o)
    out = pd.concat(parts)
    assert out["date"].max() <= pd.Timestamp(C.VAL_END), "OOF calibration must not touch the test period"
    return out


# --------------------------------------------------------------------------------------- thresholds
def _flag_metrics(pos, flag):
    tp, fp, fn, tn = int((pos & flag).sum()), int((~pos & flag).sum()), int((pos & ~flag).sum()), int((~pos & ~flag).sum())
    return {"recall": tp / max(tp + fn, 1), "precision": tp / max(tp + fp, 1) if tp + fp else None,
            "false_alarm_rate": fp / max(fp + tn, 1), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def select_escalation(y, p, cuts, caps=None, factors=None):
    """Lowest decision threshold per level with pooled-OOF false-alarm rate <= cap.
    watch / high: target y >= High cut ; critical: target y >= Critical cut.  Returns (thresholds, grid rows)."""
    caps, factors = caps or C.ESC_FAR_CAPS, factors or C.ESC_FACTORS
    y, p = np.asarray(y, float), np.asarray(p, float)
    rows, thr = [], {}
    for level, cut in (("watch", cuts[1]), ("high", cuts[1]), ("critical", cuts[2])):
        pos = y >= cut
        chosen = None
        for f in sorted(factors):                       # ascending threshold -> descending recall
            m = _flag_metrics(pos, p >= f * cut)
            rows.append({"level": level, "factor": f, "threshold_cases": f * cut, **m})
            if chosen is None and m["false_alarm_rate"] <= caps[level]:
                chosen = f
        chosen = chosen if chosen is not None else 1.0
        thr[level] = float(chosen * cut)
    # ordering guard: watch <= high < critical, high >= Medium cut
    thr["high"] = max(thr["high"], thr["watch"], cuts[0] + 1.0)
    thr["critical"] = max(thr["critical"], thr["high"] + 1.0)
    return thr, pd.DataFrame(rows)


def tier_index(p, cuts, thr):
    """0 Low / 1 Medium / 2 High / 3 Critical using escalated High/Critical decision thresholds."""
    p = np.asarray(p, float)
    return np.where(p >= thr["critical"], 3, np.where(p >= thr["high"], 2, np.where(p >= cuts[0], 1, 0)))


# --------------------------------------------------------------------------------------- intervals
def _level(n, alpha):
    return min(1.0, np.ceil((n + 1) * (1 - alpha)) / max(n, 1))


def _groups(p, cuts, merge_map):
    g = np.searchsorted(np.asarray(cuts), np.asarray(p, float), side="right")      # 0..3 point tier
    return np.array([merge_map[i] for i in g])


def fit_interval(p, y, cuts, alpha, mode):
    """mode: 'scaled_global' (symmetric, one constant) | 'mondrian_asym' (per predicted-tier, separate lower/upper).
    Score = residual / sqrt(p + 1) (count-like heteroscedasticity)."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    sc = np.sqrt(p + 1.0)
    if mode == "scaled_global":
        q = float(np.quantile(np.abs(y - p) / sc, _level(len(y), alpha)))
        return {"mode": mode, "map": [0, 0, 0, 0], "q_lo": [q], "q_hi": [q], "alpha": alpha}
    gid = np.searchsorted(np.asarray(cuts), p, side="right")
    merge = [0]
    for t in range(1, 4):                      # a sparse predicted tier shares the interval constant of the tier below
        merge.append(merge[-1] if (gid == t).sum() < C.INTERVAL_MIN_GROUP else merge[-1] + 1)
    uniq = sorted(set(merge))
    g = np.array([merge[i] for i in gid]); q_lo, q_hi = [], []
    for k in range(len(uniq)):
        m = g == k
        lvl = _level(int(m.sum()), alpha / 2)
        q_lo.append(max(float(np.quantile(((p - y) / sc)[m], lvl)), 0.0))
        q_hi.append(max(float(np.quantile(((y - p) / sc)[m], lvl)), 0.0))
    return {"mode": mode, "map": merge, "q_lo": q_lo, "q_hi": q_hi, "alpha": alpha}


def apply_interval(p, cuts, cal):
    """Interval that ALWAYS contains the point forecast (q >= 0 by construction)."""
    p = np.maximum(np.asarray(p, float), 0)
    sc = np.sqrt(p + 1.0)
    g = _groups(p, cuts, cal["map"])
    qlo, qhi = np.asarray(cal["q_lo"])[g], np.asarray(cal["q_hi"])[g]
    return np.maximum(p - qlo * sc, 0), p + qhi * sc


def _cov_by_pred_tier(y, p, lo, hi, cuts, min_n=30):
    t = np.searchsorted(np.asarray(cuts), p, side="right")
    return {int(k): float(((y >= lo) & (y <= hi))[t == k].mean()) for k in range(4) if (t == k).sum() >= min_n}


def choose_interval(oof, cuts, alpha):
    """Calibrate on earlier OOF folds, check on the last two (chronological hold-out inside the pre-test period)."""
    last = oof["fold"].max()
    cal, chk = oof[oof["fold"] < last - 1], oof[oof["fold"] >= last - 1]
    res = {}
    for mode in ("scaled_global", "mondrian_asym"):
        c = fit_interval(cal.p, cal.y, cuts, alpha, mode)
        lo, hi = apply_interval(chk.p.values, cuts, c)
        cov = _cov_by_pred_tier(chk.y.values, chk.p.values, lo, hi, cuts)
        res[mode] = {"coverage": float(((chk.y.values >= lo) & (chk.y.values <= hi)).mean()), "width": float((hi - lo).mean()),
                     "coverage_by_pred_tier": cov, "mean_abs_dev_from_nominal": float(np.mean([abs(v - (1 - alpha)) for v in cov.values()]))}
    best = min(res, key=lambda m: (round(res[m]["mean_abs_dev_from_nominal"], 2), res[m]["width"]))
    return best, res


# --------------------------------------------------------------------------------------- diagnostics
def recall_at_far(pos, score, far):
    neg = np.sort(np.asarray(score)[~pos]); thr = neg[int(np.ceil((1 - far) * len(neg))) - 1]
    return float(((score > thr) & pos).sum() / pos.sum())


def scorer_alternatives(df, cols, y, ok, high_cut, fits):
    """Do other scorers rank High/Critical weeks better than the Poisson regression?  OOF folds 2019-2023 only.
    `fits` = {name: (fit(X, y_target) -> model, scorer(model, X) -> scores, target_fn)}."""
    gap = pd.Timedelta(weeks=C.MAX_HORIZON)
    folds = [f for f in fold_bounds() if f[1].year <= 2023 and f[1] - f[0] > pd.Timedelta(days=300)]
    store = {k: [] for k in fits}; truth = []
    for s, e in folds:
        trm, tsm = ok & (df["date"] <= s - gap), ok & (df["date"] >= s) & (df["date"] <= e)
        truth.append((y[tsm] >= high_cut).to_numpy())
        for name, (fit, score, target) in fits.items():
            m = fit(df.loc[trm, cols], target(y[trm]))
            store[name].append(score(m, df.loc[tsm, cols]))
    pos = np.concatenate(truth); rows = []
    for name, parts in store.items():
        s = np.concatenate(parts)
        rows.append({"scorer": name, "oof_n": int(len(pos)), "oof_pos": int(pos.sum()), "PR_AUC": float(average_precision_score(pos, s)),
                     "recall_at_FAR5": recall_at_far(pos, s, 0.05), "recall_at_FAR10": recall_at_far(pos, s, 0.10)})
    return pd.DataFrame(rows)


def nested_check(oof, cuts):
    """Would thresholds chosen on OLDER seasons have held on a NEWER one?  select on folds < last-1, evaluate on the last two."""
    last = oof["fold"].max()
    cal, chk = oof[oof["fold"] < last - 1], oof[oof["fold"] >= last - 1]
    thr, _ = select_escalation(cal.y, cal.p, cuts)
    rows = []
    for level, cut in (("watch", cuts[1]), ("high", cuts[1]), ("critical", cuts[2])):
        pos = chk.y.values >= cut
        rows.append({"level": level, "threshold_cases": thr[level], "far_cap": C.ESC_FAR_CAPS[level],
                     "holdout_pos": int(pos.sum()), **_flag_metrics(pos, chk.p.values >= thr[level])})
    return pd.DataFrame(rows)


def persistence_check(oof, cuts, far_cap=None):
    """Alternative escalation rule tested and compared at the SAME false-alarm budget: flag High only when the forecast
    exceeds the threshold in two consecutive weeks (calendar-aligned).  Reported, not adopted unless it recalls more."""
    from .features import value_at_offset
    far_cap = C.ESC_FAR_CAPS["high"] if far_cap is None else far_cap
    o = oof.reset_index(drop=True)
    prev = value_at_offset(o, o["p"], -1)
    pos = o["y"].to_numpy() >= cuts[1]
    rows = []
    for rule in ("single week (adopted)", "two consecutive weeks"):
        best = None
        for f in sorted(C.ESC_FACTORS):
            flag = o["p"].to_numpy() >= f * cuts[1]
            if rule.startswith("two"):
                flag = flag & (prev >= f * cuts[1])
            m = _flag_metrics(pos, flag)
            if m["false_alarm_rate"] <= far_cap:
                best = {"rule": rule, "factor": f, "threshold_cases": f * cuts[1], **m}
                break
        rows.append(best or {"rule": rule})
    return pd.DataFrame(rows)
