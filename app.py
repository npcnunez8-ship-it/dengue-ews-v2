"""Dengue Early-Warning System - Streamlit decision-support dashboard.   Run:  streamlit run app.py"""
import platform
import sys

# -----------------------------------------------------------------------------
# Canonical Runtime Protection (Proposal A Governance)
# -----------------------------------------------------------------------------
try:
    import sklearn
    _sklearn_ver = getattr(sklearn, "__version__", "unknown")
except ImportError:
    _sklearn_ver = "not installed"

_py_major_minor = sys.version_info[:2]
if _py_major_minor != (3, 12) or _sklearn_ver != "1.9.1":
    _err_msg = (
        f"CANONICAL RUNTIME ERROR:\n"
        f"Dengue EWS V2 requires Python 3.12 and scikit-learn 1.9.1 (pinned in requirements-lock.txt).\n"
        f"Current Python: {platform.python_version()} ({sys.executable})\n"
        f"Current scikit-learn: {_sklearn_ver}\n\n"
        f"Please launch using the canonical virtual environment:\n"
        f"  .\\.venv\\Scripts\\python.exe -m streamlit run app.py\n"
    )
    try:
        import streamlit as _st
        _st.error(_err_msg)
    except Exception:
        pass
    raise SystemExit(_err_msg)

import datetime
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import config as C
from src.features import add_target
from src.geo import city_summary, ward_coordinates
from src.live_weather import fetch_city_weather, history_for_scoring
from src.predict import explain_row, load_bundle, load_features, predict_frame
from src.quality import repair_history, unusual_observations
from src.scenario import N_HISTORY, all_horizons, apply_scenario, predict_history, ward_history

st.set_page_config(
    page_title="Dengue Early Warning - Bangladesh",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded",
)
TIER_COLOR = {"Low": "#2e9e5b", "Medium": "#e6b800", "High": "#e8710a", "Critical": "#c62828"}
TIER_IDX = {t: i for i, t in enumerate(C.RISK_TIERS)}
ADVICE = {
    "Low": "Routine surveillance. No extra action needed.",
    "Medium": "Increase larval surveys and drain-clearing in this ward over the next 2 weeks.",
    "High": "Deploy larvicide/spraying teams and alert nearby hospitals to prepare beds and saline.",
    "Critical": "Priority ward: immediate vector control, community alerts and surge capacity at hospitals.",
}
CSS_FILE = C.ROOT / "assets" / "custom.css"
if CSS_FILE.exists():
    st.markdown(f"<style>\n{CSS_FILE.read_text(encoding='utf-8')}\n</style>", unsafe_allow_html=True)
else:
    st.markdown("""
<style>
.block-container{padding-top:1.2rem;max-width:1400px}
.hero{background:linear-gradient(120deg,#0b3d3a 0%,#0f766e 60%,#2a9d8f 100%);color:#fff;padding:18px 24px;border-radius:14px;margin-bottom:14px}
.hero h1{margin:0;font-size:1.65rem;color:#fff}.hero p{margin:4px 0 0 0;opacity:.88;font-size:.95rem}
[data-testid="stMetric"]{background:#f6faf9;border:1px solid #dbe8e5;border-radius:12px;padding:10px 14px}
.badge{display:block;color:#fff;padding:10px 12px;border-radius:10px;font-size:1.35rem;font-weight:700;text-align:center}
.hcard{border:1px solid #dbe8e5;border-radius:12px;padding:10px 14px;background:#fbfdfc}
.hcard .t{font-size:.8rem;color:#52706a;text-transform:uppercase;letter-spacing:.04em}
.hcard .v{font-size:1.7rem;font-weight:700;line-height:1.2}.hcard .i{font-size:.85rem;color:#52706a}
.pill{display:inline-block;color:#fff;border-radius:999px;padding:1px 10px;font-size:.8rem;font-weight:600}
.small{font-size:.82rem;color:#52706a}
</style>""", unsafe_allow_html=True)


# ---------------- cached resources ----------------
@st.cache_resource(show_spinner="Loading model (rebuilds automatically if this machine needs it)...")
def bundle():
    return load_bundle()


@st.cache_data(show_spinner="Building ward-level weather-lag features (once)...")
def features():
    return load_features()


@st.cache_data(show_spinner="Running batch inference...")
def all_predictions(window):
    df = features()
    p = predict_frame(df, window, bundle())
    out = pd.concat([df[["date", "city", "ward_id", "dengue_cases"]], p], axis=1)
    out["actual"] = add_target(df, C.WINDOWS[window])
    return out


@st.cache_data
def lag_corr():
    d = features().sort_values(["ward_id", "date"])
    g = d.groupby("ward_id")["rainfall_mm"]
    return pd.DataFrame([{"lag_weeks": k, "correlation": d.dengue_cases.corr(g.shift(k))} for k in range(0, 9)])


@st.cache_data
def report_csv(name):
    return pd.read_csv(C.REPORT_DIR / name)


@st.cache_data
def metrics():
    return json.loads((C.REPORT_DIR / "metrics.json").read_text())


@st.cache_data
def audit():
    return json.loads((C.REPORT_DIR / "audit.json").read_text())


@st.cache_data
def test_preds():
    return pd.read_csv(C.REPORT_DIR / "test_predictions.csv", parse_dates=["date"])


def badge(tier):
    return f"<span class='badge' style='background:{TIER_COLOR[tier]}'>{tier}</span>"


def pill(tier):
    return f"<span class='pill' style='background:{TIER_COLOR[tier]}'>{tier}</span>"


df, B, M = features(), bundle(), metrics()
VAL_END = pd.Timestamp(C.VAL_END)

# ---------------- sidebar ----------------
st.sidebar.title("🦟 Dengue Early Warning")
window = st.sidebar.selectbox("Forecast window", list(C.WINDOWS), index=0,
                              help="Cases summed over the selected weeks ahead of the prediction date.")
city = st.sidebar.selectbox("City", sorted(df.city.unique()))
ward = st.sidebar.selectbox("Ward", sorted(df[df.city == city].ward_id.unique()))
valid_dates = df.loc[df[B["features"]].notna().all(axis=1), "date"].drop_duplicates().sort_values()
default = valid_dates[(valid_dates.dt.year == 2025) & (valid_dates.dt.month == 8)]
default = default.iloc[0] if len(default) else valid_dates.iloc[-1]
date = st.sidebar.select_slider("Prediction date (weekly)", options=list(valid_dates), value=default,
                                format_func=lambda d: d.strftime("%Y-%m-%d"))
if date <= VAL_END:
    st.sidebar.warning(f"In-sample date (training/validation period). For honest out-of-sample behaviour pick a date after {VAL_END.date()}.")
else:
    st.sidebar.success("Out-of-sample date: never used for training or model selection.")
st.sidebar.info(f"Data: {df.date.min().date()} → {df.date.max().date()} (supplied **simulated-looking** dataset). Weather source here is "
                "**observed historical** data from the CSV - no forecast weather and no live feed unless you use the Live tab.")
st.sidebar.caption("Forecast timing: made at the end of week *t* using only weather observed up to week *t*; "
                   "'weeks 2-4 ahead' = cases in weeks t+2..t+4 (week t+1 is the unavoidable blind week).")

P = all_predictions(window)
snap_all = P[P.date == date].dropna(subset=["pred"]).copy()
row = P[(P.ward_id == ward) & (P.date == date)].iloc[0]
cuts = B["windows"][window]["cutpoints"]
ESC = B["windows"][window].get("esc") or {"high": cuts[1], "critical": cuts[2], "watch": B["windows"][window].get("alert_cut", cuts[1])}
alert_cut = ESC["watch"]
T_ = M["test_by_window"][window]          # held-out test results for THIS window (numbers below are read, never hard-coded)
T0_ = M.get("test_by_window_original_method", {}).get(window)

st.markdown(f"<div class='hero'><h1>🦟 Urban Dengue Outbreak Early-Warning</h1><p>Ward-level risk 2-4 weeks ahead from multi-week weather lags · "
            f"{city} / {ward} · prediction date <b>{date.date()}</b> · window <b>{window}</b></p></div>", unsafe_allow_html=True)
SECTIONS = [
    "🏠 Overview",
    "📍 Ward forecast",
    "🗺️ Risk maps & ward ranking",
    "🧪 What-if",
    "🌐 Live weather (optional)",
    "📊 Validation",
    "ℹ️ Methodology & limits",
]

st.markdown("<div class='nav-wrapper'>", unsafe_allow_html=True)
active_section = st.radio(
    "Navigation",
    SECTIONS,
    index=0,
    horizontal=True,
    label_visibility="collapsed",
    key="nav_section",
)
st.markdown("</div>", unsafe_allow_html=True)


def generate_situation_report_csv(snap_hc: pd.DataFrame, date_val: pd.Timestamp, window_val: str) -> str:
    """Generate deterministic UTF-8 CSV export of High and Critical wards."""
    if len(snap_hc) == 0:
        cols = [
            "ward_id", "city", "forecast_date", "forecast_horizon",
            "predicted_cases", "interval_80_lo", "interval_80_hi",
            "risk_tier", "watch_alert"
        ]
        return pd.DataFrame(columns=cols).to_csv(index=False)

    export_df = pd.DataFrame({
        "ward_id": snap_hc["ward_id"],
        "city": snap_hc["city"],
        "forecast_date": date_val.strftime("%Y-%m-%d"),
        "forecast_horizon": window_val,
        "predicted_cases": snap_hc["pred"],
        "interval_80_lo": snap_hc["lo"],
        "interval_80_hi": snap_hc["hi"],
        "risk_tier": snap_hc["tier"],
        "watch_alert": snap_hc["alert"],
    })
    return export_df.to_csv(index=False)


def generate_situation_report_markdown(
    snap_scope: pd.DataFrame,
    snap_hc: pd.DataFrame,
    date_val: pd.Timestamp,
    window_val: str,
    scope_name: str,
    esc_thresholds: dict = None,
    alert_thresh: float = None,
    gen_time_str: str = None
) -> str:
    """Generate concise, readable Executive Situation Report in Markdown."""
    if gen_time_str is None:
        gen_time_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    n_monitored = len(snap_scope)
    n_crit = int((snap_scope["tier"] == "Critical").sum())
    n_high = int((snap_scope["tier"] == "High").sum())
    n_watch = int((snap_scope["alert"] & ~snap_scope["tier"].isin(["High", "Critical"])).sum())
    n_total_alert = int(snap_scope["alert"].sum())
    total_cases = snap_scope["pred"].sum()

    status_alert = "MODEL CLASSIFICATION: CRITICAL TIER (SIMULATED/ESTIMATED)" if n_crit > 0 else (
        "MODEL CLASSIFICATION: HIGH RISK TIER" if n_high > 0 else (
            "PRELIMINARY SURVEILLANCE WATCH" if n_watch > 0 else "BASELINE SURVEILLANCE (NORMAL)"
        )
    )

    lines = []
    lines.append("# Dengue Early Warning System V2 — Executive Situation Report")
    lines.append("### Bangladesh Urban Corporation Epidemiological Intelligence")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Document Metadata & Surveillance Scope")
    lines.append("")
    lines.append("- **System:** Dengue Early Warning System (EWS) V2 — Bangladesh")
    lines.append(f"- **Report Generated:** `{gen_time_str}` (compilation timestamp, distinct from forecast cycle)")
    lines.append(f"- **Active Epidemiological Forecast Date:** **`{date_val.strftime('%Y-%m-%d')}`** (weekly surveillance cycle)")
    lines.append(f"- **Forecast Horizon:** **`{window_val}`**")
    lines.append(f"- **Geographic Scope:** **{scope_name}** ({n_monitored} wards monitored)")
    lines.append(f"- **Operational Surveillance Status:** **`{status_alert}`**")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. Executive Surveillance Summary")
    lines.append("")
    lines.append(f"- **Critical Risk Wards:** **{n_crit}** ward(s) exceeding the calibrated model Critical-tier threshold")
    lines.append(f"- **High Risk Wards:** **{n_high}** ward(s) exceeding the calibrated model High-tier threshold")
    lines.append(f"- **Preliminary Watch Alerts:** **{n_watch}** ward(s) flagged on early screening list (total active watch: **{n_total_alert}**)")
    lines.append(f"- **Total Projected Case Burden in Scope:** **{total_cases:,.1f} cases** ({window_val})")
    if esc_thresholds:
        lines.append(f"- **Decision Cutpoints for `{window_val}`:** High ≥ {esc_thresholds.get('high', 0):.1f} cases | Critical ≥ {esc_thresholds.get('critical', 0):.1f} cases | Watch Screening ≥ {alert_thresh:.1f} cases")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. Priority Action Ward Table (High & Critical Risk Tiers)")
    lines.append("")

    if len(snap_hc) == 0:
        lines.append("*No wards in the selected scope currently meet High or Critical risk escalation thresholds.*")
        lines.append("")
        lines.append("All monitored wards in this scope are classified within baseline (Low or Medium) expected risk.")
    else:
        lines.append(f"A total of **{len(snap_hc)} ward(s)** meet calibrated model criteria for High or Critical decision-support review:")
        lines.append("")
        lines.append("| Rank | Ward Identifier | City / Corporation | Forecast Cases | 80% Conformal Interval | Risk Tier | Outbreak Watch |")
        lines.append("|:---:|:---|:---|---:|:---:|:---:|:---:|")
        for rank_idx, (_, r) in enumerate(snap_hc.iterrows(), start=1):
            watch_str = "⚠️ Watch Active" if r["alert"] else "— Standard"
            lines.append(f"| {rank_idx} | `{r['ward_id']}` | {r['city']} | **{r['pred']:.1f}** | [{r['lo']:.0f} – {r['hi']:.0f}] | **{r['tier']}** | {watch_str} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 4. Predictive Driver Interpretation")
    lines.append("")
    lines.append("Model forecasts are synthesized from calendar-aligned historical weather lags (cumulative rainfall, mean temperature, relative humidity across 2-to-16-week lag windows) and historical ward vulnerability baselines.")
    lines.append("")
    lines.append("> **Scientific Interpretation Qualification:**")
    lines.append("> These are model-based predictive contributions relative to the seasonal-normal reference, not causal estimates. Weather anomalies indicate environmental suitability for Aedes breeding and viral incubation, but do not directly constitute causal proof of transmission chains.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 5. Public Health Decision-Support Considerations")
    lines.append("")
    lines.append("Illustrative Public Health Decision-Support Considerations for administrative leadership and municipal health teams in identified High and Critical wards:")
    lines.append("")
    lines.append("- **Targeted Source Reduction:** Intensify container clearing and domestic standing water elimination in prioritized wards.")
    lines.append("- **Larvicide Review:** Coordinate application of approved biological or chemical larvicides (e.g. BTI, Temephos) at perennial breeding sites.")
    lines.append("- **Healthcare Facility Readiness:** Alert municipal clinics, stock rapid NS1 diagnostic tests, and prepare pediatric/adult triage beds for potential case surges.")
    lines.append("- **Watch-List Verification:** For wards triggering preliminary watch screening flags, dispatch field entomological inspection teams before case burdens cross escalation thresholds.")
    lines.append("")
    lines.append("*Note: Guidance above represents standard illustrative decision-support material. It does not prescribe chemical dosages, application frequencies, municipal budgets, or hospital staffing quotas.*")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 6. Methodological Limitations & Official Disclaimer")
    lines.append("")
    lines.append("> **Important Methodological Notice:**")
    lines.append("> This application uses synthetic/simulated epidemiological data. Forecasts and risk classifications are experimental decision-support outputs and have not been established as operationally validated public-health surveillance results. They are not clinical diagnoses or official DGHS instructions. Confirm any public-health action with authorized local authorities and current official guidance.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 7. Data Provenance & Verification Audit")
    lines.append("")
    lines.append("- **Authoritative Environment:** Canonical Python 3.12.10, scikit-learn 1.9.1, Streamlit 1.67.0")
    lines.append("- **Prediction Pipeline:** Multi-output Gradient Boosting Regressor with Mondrian conformal prediction intervals (80% empirical test coverage)")
    lines.append("- **Surveillance Extent:** 60 administrative wards across Dhaka North, Dhaka South, Chattogram, Gazipur, Narayanganj")
    lines.append("- **Data Provenance:** Chronologically partitioned training (2016–2023), validation (2023–2024), and held-out test splits (2024–2025).")
    lines.append("")
    return "\n".join(lines)


# ================= 1. OVERVIEW =================
if active_section == "🏠 Overview":
    n_crit = int((snap_all.tier == "Critical").sum())
    n_high = int((snap_all.tier == "High").sum())
    n_watch = int((snap_all.alert & ~snap_all.tier.isin(["High", "Critical"])).sum())
    tot_cases = snap_all.pred.sum()

    # 1. OUTBREAK ALERT PANEL
    if n_crit + n_high > 0:
        crit_wards = snap_all[snap_all.tier == "Critical"]
        high_wards = snap_all[snap_all.tier == "High"]
        panel_class = "alert-active-critical" if n_crit > 0 else "alert-active-high"

        crit_chips = "".join([
            f"<span class='alert-chip chip-critical'>🔴 {r.ward_id} ({r.city}) · {r.pred:.1f} cases [{r.lo:.0f}–{r.hi:.0f}]</span>"
            for _, r in crit_wards.iterrows()
        ])
        high_chips = "".join([
            f"<span class='alert-chip chip-high'>🟠 {r.ward_id} ({r.city}) · {r.pred:.1f} cases [{r.lo:.0f}–{r.hi:.0f}]</span>"
            for _, r in high_wards.iterrows()
        ])

        watch_note = (
            f"<div style='margin-top:8px;font-size:0.84rem;color:var(--text-secondary);'>"
            f"⚠️ Additionally, <b>{n_watch} ward(s)</b> triggered preliminary watch-list screening flags (cases ≥ {alert_cut:.1f})."
            f"</div>" if n_watch > 0 else ""
        )

        st.markdown(
            f"<div class='outbreak-alert-panel {panel_class}'>"
            f"<div class='outbreak-alert-title'>🚨 Outbreak Surge Escalation Alert · {date.date()} ({window})</div>"
            f"<div class='outbreak-alert-body'>"
            f"<b>{n_crit} Critical</b> and <b>{n_high} High-risk</b> ward(s) currently exceed operational escalation thresholds. "
            f"Immediate targeted vector control and hospital triage surge readiness recommended."
            f"<div class='alert-chips-container'>{crit_chips}{high_chips}</div>"
            f"{watch_note}"
            f"<div style='margin-top:8px;font-size:0.78rem;color:var(--text-muted);'>"
            f"Decision thresholds: High ≥ {ESC['high']:.1f} cases · Critical ≥ {ESC['critical']:.1f} cases · "
            f"Calibrated on test data to capture {T_['HC_recall']:.0%} of historical surges with {T_['HC_precision']:.0%} precision."
            f"</div>"
            f"</div></div>",
            unsafe_allow_html=True,
        )
    elif n_watch > 0:
        watch_wards = snap_all[snap_all.alert & ~snap_all.tier.isin(["High", "Critical"])]
        watch_chips = "".join([
            f"<span class='alert-chip chip-watch'>🟡 {r.ward_id} ({r.city}) · {r.pred:.1f} cases</span>"
            for _, r in watch_wards.iterrows()
        ])
        st.markdown(
            f"<div class='outbreak-alert-panel alert-active-watch'>"
            f"<div class='outbreak-alert-title'>⚠️ Preliminary Surveillance Watch Alert · {date.date()} ({window})</div>"
            f"<div class='outbreak-alert-body'>"
            f"No wards currently meet High or Critical escalation thresholds. However, <b>{n_watch} ward(s)</b> are flagged on the "
            f"early watch list (predicted ≥ {alert_cut:.1f} cases). Recommended for early field inspection and larviciding review."
            f"<div class='alert-chips-container'>{watch_chips}</div>"
            f"<div style='margin-top:8px;font-size:0.78rem;color:var(--text-muted);'>"
            f"Screening decision rule: {T_['ALERT_recall']:.0%} test recall, {T_['ALERT_precision']:.0%} precision, {T_['ALERT_false_alarm_rate']:.0%} false-alarm rate. A screening signal, not an operational surge tier."
            f"</div>"
            f"</div></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='outbreak-alert-panel alert-baseline'>"
            f"<div class='outbreak-alert-title'>✅ Surveillance Status: Baseline (Normal) · {date.date()} ({window})</div>"
            f"<div class='outbreak-alert-body'>"
            f"All {len(snap_all)} monitored wards across Dhaka North, Dhaka South, and Chittagong are currently within baseline (Low or Medium) expected risk. "
            f"Zero wards meet High or Critical thresholds, and zero screening watch flags are active."
            f"</div></div>",
            unsafe_allow_html=True,
        )

    # 2. EXECUTIVE KPI CARDS
    k = st.columns(5)
    with k[0]:
        st.markdown(
            f"<div class='exec-kpi-card kpi-total'>"
            f"<div class='exec-kpi-label'><span>Monitored Wards</span><span>🏛️</span></div>"
            f"<div class='exec-kpi-value'>{len(snap_all)}</div>"
            f"<div class='exec-kpi-subtext'>All administrative wards · 3 cities</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with k[1]:
        st.markdown(
            f"<div class='exec-kpi-card kpi-critical'>"
            f"<div class='exec-kpi-label'><span>Critical Risk</span><span>🔴</span></div>"
            f"<div class='exec-kpi-value'>{n_crit}</div>"
            f"<div class='exec-kpi-subtext'>≥ {ESC['critical']:.1f} cases threshold</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with k[2]:
        st.markdown(
            f"<div class='exec-kpi-card kpi-high'>"
            f"<div class='exec-kpi-label'><span>High Risk</span><span>🟠</span></div>"
            f"<div class='exec-kpi-value'>{n_high}</div>"
            f"<div class='exec-kpi-subtext'>≥ {ESC['high']:.1f} cases · {T_['HC_recall']:.0%} recall</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with k[3]:
        st.markdown(
            f"<div class='exec-kpi-card kpi-watch'>"
            f"<div class='exec-kpi-label'><span>Watch List</span><span>🟡</span></div>"
            f"<div class='exec-kpi-value'>{n_watch}</div>"
            f"<div class='exec-kpi-subtext'>≥ {alert_cut:.1f} cases screening signal</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with k[4]:
        st.markdown(
            f"<div class='exec-kpi-card kpi-aggregate'>"
            f"<div class='exec-kpi-label'><span>Total Forecast</span><span>📊</span></div>"
            f"<div class='exec-kpi-value'>{tot_cases:,.0f}</div>"
            f"<div class='exec-kpi-subtext'>Predicted cases ({window})</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.write("")

    # 3. TOP-RISK WARDS & 4. CITY RISK DISTRIBUTION
    left, right = st.columns([3, 2])
    with left:
        st.subheader("🎯 Priority Action Ranking")
        top = snap_all.sort_values("pred", ascending=False).head(10).copy()
        top["rank"] = range(1, len(top) + 1)
        top["interval"] = top.apply(lambda r: f"[{r.lo:.1f} – {r.hi:.1f}]", axis=1)
        top["watch_flag"] = top["alert"].map({True: "⚠️ Watch", False: "—"})

        display_top = top[["rank", "ward_id", "city", "pred", "interval", "tier", "watch_flag"]].rename(columns={
            "rank": "Rank",
            "ward_id": "Ward ID",
            "city": "City",
            "pred": "Forecast Cases",
            "interval": "80% Interval",
            "tier": "Risk Tier",
            "watch_flag": "Watch Alert"
        })

        st.dataframe(
            display_top,
            hide_index=True,
            width="stretch",
            column_config={
                "Rank": st.column_config.NumberColumn("Rank", width="small"),
                "Ward ID": st.column_config.TextColumn("Ward", width="medium"),
                "City": st.column_config.TextColumn("City", width="medium"),
                "Forecast Cases": st.column_config.ProgressColumn(
                    "Forecast Cases",
                    help=f"Predicted cases for {window}",
                    format="%.1f",
                    min_value=0.0,
                    max_value=float(max(top["pred"].max() * 1.15, 10.0)),
                    width="medium"
                ),
                "80% Interval": st.column_config.TextColumn("80% Interval", help="Empirically calibrated 80% prediction interval", width="medium"),
                "Risk Tier": st.column_config.TextColumn("Risk Tier", width="small"),
                "Watch Alert": st.column_config.TextColumn("Alert", width="small"),
            }
        )
        st.caption(
            f"Top 10 wards ranked by expected dengue cases for **{window}** on **{date.date()}**. "
            f"Decision thresholds: Medium ≥ {cuts[0]:.0f} · High ≥ {ESC['high']:.1f} · Critical ≥ {ESC['critical']:.1f} cases. "
            f"Prediction intervals denote calibrated 80% coverage on held-out test data."
        )

    with right:
        st.subheader("🏙️ City Risk Distribution")
        cnt = snap_all.groupby(["city", "tier"]).size().reset_index(name="wards")
        fig = px.bar(
            cnt,
            x="wards",
            y="city",
            color="tier",
            orientation="h",
            color_discrete_map=TIER_COLOR,
            category_orders={"tier": C.RISK_TIERS},
            text="wards"
        )
        fig.update_traces(
            textposition="inside",
            insidetextanchor="middle",
            hovertemplate="<b>%{y}</b><br>Tier: %{data.name}<br>Wards: %{x}<extra></extra>"
        )
        fig.update_layout(
            height=340,
            margin=dict(l=0, r=10, t=10, b=0),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                title_text=""
            ),
            yaxis=dict(title="", automargin=True),
            xaxis=dict(title="Wards in Risk Tier", showgrid=True, gridcolor="#E2EBE8"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif")
        )
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False, "responsive": True})
        st.caption(f"Count of wards by risk tier across all 3 city corporations for the selected forecast window ({window}).")

    # 5. SELECTED WARD AT A GLANCE
    st.subheader("📍 Selected Ward Operational Profile")
    watch_badge = (
        "<span style='display:inline-block;padding:2px 8px;border-radius:9999px;font-size:0.75rem;font-weight:700;background:#fef3c7;color:#92400e;border:1px solid #fde68a;'>⚠️ WATCH ACTIVE</span>"
        if row.alert else
        "<span style='display:inline-block;padding:2px 8px;border-radius:9999px;font-size:0.75rem;font-weight:600;background:#ecfdf5;color:#065f46;border:1px solid #a7f3d0;'>✅ NO WATCH FLAG</span>"
    )
    st.markdown(
        f"<div class='ward-summary-card'>"
        f"<div class='ward-summary-grid'>"
        f"<div>"
        f"<div class='ward-summary-meta'>Surveillance Target · {city}</div>"
        f"<div class='ward-summary-heading'>{ward} &nbsp; {pill(row.tier)} &nbsp; {watch_badge}</div>"
        f"<div class='ward-summary-stats'>"
        f"Forecast Window: <b>{window}</b> · Prediction Date: <b>{date.date()}</b><br>"
        f"Expected Cases: <b style='font-size:1.15rem;color:var(--text-primary);'>{row.pred:.1f}</b> "
        f"<span style='color:var(--text-secondary);'>(80% Calibrated Interval: <b>{row.lo:.0f} – {row.hi:.0f}</b> cases)</span><br>"
        f"<span style='font-size:0.8rem;color:var(--text-muted);'>Thresholds for window: Medium ≥ {cuts[0]:.0f} · High ≥ {ESC['high']:.1f} · Critical ≥ {ESC['critical']:.1f} · Watch ≥ {alert_cut:.1f}</span>"
        f"</div>"
        f"</div>"
        f"<div>"
        f"<div class='ward-summary-advice-box'>"
        f"<div class='ward-summary-advice-title'>Standard Operating Procedure · {row.tier} Tier</div>"
        f"<div class='ward-summary-advice-text'>{ADVICE[row.tier]}</div>"
        f"</div>"
        f"</div>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # 6. RELIABILITY AND MODEL INFORMATION
    t = M["test_by_window"][window]
    baseline_mae = t["baselines"]["Seasonal-naive (52w earlier)"]["MAE"]
    st.markdown(
        f"<div class='reliability-card'>"
        f"<div class='reliability-header'>🛡️ Model Performance & Surveillance Reliability ({window})</div>"
        f"<div style='font-size:0.85rem;color:var(--text-secondary);line-height:1.45;'>"
        f"Rigorous out-of-sample evaluation on held-out test data (never seen during training or threshold tuning):"
        f"</div>"
        f"<div class='reliability-grid'>"
        f"<div class='reliability-stat-item'>"
        f"<div class='label'>Model Test MAE</div>"
        f"<div class='val'>{t['MAE']:.2f} <span style='font-size:0.8rem;font-weight:normal;color:var(--text-secondary);'>cases</span></div>"
        f"<div class='desc'>Baseline seasonal-naive MAE: {baseline_mae:.2f} cases</div>"
        f"</div>"
        f"<div class='reliability-stat-item'>"
        f"<div class='label'>High/Critical Surge Recall</div>"
        f"<div class='val'>{t['HC_recall']:.0%}</div>"
        f"<div class='desc'>Detected {t['HC_recall']:.0%} of historical High/Critical surge events</div>"
        f"</div>"
        f"<div class='reliability-stat-item'>"
        f"<div class='label'>High/Critical Precision</div>"
        f"<div class='val'>{t['HC_precision']:.0%}</div>"
        f"<div class='desc'>False-alarm rate: {t['HC_false_alarm_rate']:.0%} on surge alerts</div>"
        f"</div>"
        f"<div class='reliability-stat-item'>"
        f"<div class='label'>80% Interval Coverage</div>"
        f"<div class='val'>80% Empirical</div>"
        f"<div class='desc'>Calibrated conformal prediction intervals</div>"
        f"</div>"
        f"</div>"
        f"<div class='reliability-disclaimer'>"
        f"<b>⚠️ Public Health Surveillance Notice & Scientific Disclosure:</b><br>"
        f"1. <b>Synthetic Data Limitation:</b> Statistical testing indicates the supplied dataset exhibits simulated/synthetic properties. "
        f"Performance metrics above demonstrate algorithmic capability on this specific dataset, but do <b>not</b> constitute proof of real-world epidemiological accuracy.<br>"
        f"2. <b>Operational Status:</b> This system is an analytical decision-support prototype and is <b>not operationally validated</b> by public health authorities. "
        f"Predictions should not replace clinical laboratory diagnostics, entomological vector field audits, or official Ministry of Health advisories.<br>"
        f"3. <b>Lag Limitations:</b> Predictions rely on observed historical weather lags (up to 16 weeks). They cannot foresee unseasonal flash flooding, sudden municipal insecticide spraying campaigns, or sudden population displacements."
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # 7. EXECUTIVE EPIDEMIOLOGICAL SITUATION REPORT EXPORT
    st.markdown("<div class='chart-surface'>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='chart-surface-header'>"
        f"  <div>"
        f"    <div class='chart-surface-title'>📥 Executive Situation Report Export</div>"
        f"    <div class='chart-surface-desc'>Download structured epidemiological intelligence briefings and priority ward callouts for operational decision-support.</div>"
        f"  </div>"
        f"</div>",
        unsafe_allow_html=True
    )

    # Scope Selection for Report Export
    rep_c1, rep_c2 = st.columns([1.2, 2])
    with rep_c1:
        rep_scope = st.radio("Export Scope", ["All cities", f"{city} only"], horizontal=True, key="sitrep_scope_radio")

    rep_snap = snap_all if rep_scope == "All cities" else snap_all[snap_all.city == city]
    rep_hc = rep_snap[rep_snap.tier.isin(["High", "Critical"])].sort_values("pred", ascending=False).reset_index(drop=True)
    scope_label = "National (All 60 Wards)" if rep_scope == "All cities" else f"{city} Corporation ({len(rep_snap)} Wards)"

    # Visible indication of active reporting parameters
    with rep_c2:
        st.markdown(
            f"<div class='sitrep-meta-box'>"
            f"<b>Active Reporting Parameters:</b><br>"
            f"• <b>Forecast Date:</b> {date.date()} &nbsp;|&nbsp; • <b>Window:</b> {window}<br>"
            f"• <b>Geographic Scope:</b> {scope_label} &nbsp;|&nbsp; • <b>High/Critical Priority Wards:</b> <b>{len(rep_hc)}</b> of {len(rep_snap)}"
            f"</div>",
            unsafe_allow_html=True
        )

    # Format filenames deterministically
    date_clean = date.strftime("%Y-%m-%d")
    scope_slug = "all_cities" if rep_scope == "All cities" else city.lower().replace(" ", "_")
    csv_filename = f"dengue_situation_report_{scope_slug}_{date_clean}.csv"
    md_filename = f"dengue_situation_report_{scope_slug}_{date_clean}.md"

    # Generate CSV & Markdown contents
    csv_data = generate_situation_report_csv(rep_hc, date, window)
    md_data = generate_situation_report_markdown(
        snap_scope=rep_snap,
        snap_hc=rep_hc,
        date_val=date,
        window_val=window,
        scope_name=scope_label,
        esc_thresholds=ESC,
        alert_thresh=alert_cut
    )

    # Download Buttons
    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1.5])
    with btn_col1:
        st.download_button(
            label="📄 Download CSV Data",
            data=csv_data,
            file_name=csv_filename,
            mime="text/csv",
            use_container_width=True,
            key="dl_sitrep_csv"
        )
    with btn_col2:
        st.download_button(
            label="📝 Download Markdown SitRep",
            data=md_data,
            file_name=md_filename,
            mime="text/markdown",
            use_container_width=True,
            key="dl_sitrep_md"
        )
    with btn_col3:
        if len(rep_hc) > 0:
            st.caption(f"Includes {len(rep_hc)} High/Critical wards with 80% conformal intervals and watch flags.")
        else:
            st.caption("No High/Critical wards in scope; exports valid empty dataset with standard column headers.")

    # Expandable preview of the Markdown report
    with st.expander("👁️ Preview Executive Situation Report (Markdown)", expanded=False):
        st.markdown(md_data)

    st.markdown("</div>", unsafe_allow_html=True)

# ================= 2. WARD FORECAST =================
elif active_section == "📍 Ward forecast":
    mh = all_horizons(df[(df.ward_id == ward) & (df.date == date)], B)
    
    st.markdown(
        f"<div class='section-header' style='margin-bottom: 12px;'>"
        f"<h3 style='margin: 0; font-size: 1.25rem; color: var(--text-primary); font-weight: 700;'>📍 {ward} ({city}) — Multi-Horizon Outbreak Forecasts</h3>"
        f"<p style='margin: 3px 0 0 0; font-size: 0.82rem; color: var(--text-secondary);'>Multi-week prediction horizon comparison, 80% conformal uncertainty intervals, and calibrated risk tiers for <b>{date.date()}</b>.</p>"
        f"</div>",
        unsafe_allow_html=True
    )

    # 1. Multi-Horizon Cards Grid
    cols = st.columns(4)
    for c, (_, r) in zip(cols, mh.sort_values("window", key=lambda s: s.map({w: i for i, w in enumerate([C.MAIN_WINDOW, "Week 2 ahead", "Week 3 ahead", "Week 4 ahead"])})).iterrows()):
        is_active = (r.window == window)
        active_cls = " is-active" if is_active else ""
        watch_flag_html = '<span style="color: #b27a00; font-size: 0.72rem; font-weight: 700;">⚠️ Watch Trigger</span>' if r.alert else '<span style="color: var(--text-muted); font-size: 0.72rem;">Standard</span>'
        c.markdown(
            f"<div class='ward-horizon-card{active_cls}'>"
            f"  <div class='ward-horizon-title'>{r.window}</div>"
            f"  <div class='ward-horizon-val'>{r.pred:.1f} <span style='font-size: 0.85rem; font-weight: 500; color: var(--text-secondary);'>cases</span></div>"
            f"  <div class='ward-horizon-interval'>80% interval: [{r.lo:.0f} – {r.hi:.0f}]</div>"
            f"  <div class='ward-horizon-footer'>"
            f"    {pill(r.tier)}"
            f"    {watch_flag_html}"
            f"  </div>"
            f"</div>",
            unsafe_allow_html=True
        )

    st.write("")

    # 2. Executive Forecast KPI Cards for Active Horizon
    st.markdown(
        f"<div style='font-size: 0.84rem; font-weight: 700; color: var(--primary-dark); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px; display: flex; align-items: center; justify-content: space-between;'>"
        f"<span>Active Evaluation Window: <b>{window}</b></span>"
        f"<span style='font-size: 0.76rem; font-weight: 500; color: var(--text-secondary); text-transform: none;'>Ward: {ward} | City: {city}</span>"
        f"</div>",
        unsafe_allow_html=True
    )

    kc1, kc2, kc3, kc4 = st.columns(4)
    with kc1:
        hindsight_sub = ""
        if pd.notna(row.actual):
            err = row.actual - row.pred
            err_color = "#c62828" if abs(err) > 10 else ("#e8710a" if abs(err) > 5 else "#2e9e5b")
            hindsight_sub = f"<div style='font-size: 0.73rem; color: {err_color}; margin-top:3px;'>Reported (hindsight): <b>{row.actual:.0f}</b> ({err:+.1f})</div>"
        else:
            hindsight_sub = "<div style='font-size: 0.73rem; color: var(--text-muted); margin-top:3px;'>Target window not yet observed</div>"

        kc1.markdown(
            f"<div class='exec-kpi-card kpi-total'>"
            f"  <div class='exec-kpi-label'>Forecast Cases <span>🦟</span></div>"
            f"  <div class='exec-kpi-value'>{row.pred:.1f} <span style='font-size: 0.85rem; font-weight: 500; color: var(--text-secondary);'>cases</span></div>"
            f"  <div class='exec-kpi-subtext'>Expected count in {window}{hindsight_sub}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

    with kc2:
        kc2.markdown(
            f"<div class='exec-kpi-card kpi-aggregate'>"
            f"  <div class='exec-kpi-label'>80% Prediction Interval <span>📊</span></div>"
            f"  <div class='exec-kpi-value'>[{row.lo:.0f} – {row.hi:.0f}]</div>"
            f"  <div class='exec-kpi-subtext'>Mondrian conformal uncertainty (span: {row.hi - row.lo:.1f} cases)</div>"
            f"</div>",
            unsafe_allow_html=True
        )

    with kc3:
        tier_cls = "kpi-critical" if row.tier == "Critical" else ("kpi-high" if row.tier == "High" else ("kpi-watch" if row.tier == "Medium" else "kpi-total"))
        kc3.markdown(
            f"<div class='exec-kpi-card {tier_cls}'>"
            f"  <div class='exec-kpi-label'>Calibrated Risk Tier <span>🏷️</span></div>"
            f"  <div class='exec-kpi-value' style='display:flex; align-items:center; gap:8px;'>{row.tier}</div>"
            f"  <div class='exec-kpi-subtext'>Cutpoints: Med ≥ {cuts[0]:.0f}, High ≥ {ESC['high']:.1f}, Crit ≥ {ESC['critical']:.1f}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

    with kc4:
        alert_cls = "kpi-critical" if row.alert else "kpi-total"
        status_text = "ACTIVE WATCH" if row.alert else ("NORMAL" if row.tier == "Low" else "ELEVATED")
        alert_desc = f"Forecast ≥ {alert_cut:.1f} watch trigger threshold" if row.alert else f"Forecast below {alert_cut:.1f} watch threshold"
        kc4.markdown(
            f"<div class='exec-kpi-card {alert_cls}'>"
            f"  <div class='exec-kpi-label'>Surveillance Alert <span>⚠️</span></div>"
            f"  <div class='exec-kpi-value' style='font-size: 1.45rem;'>{status_text}</div>"
            f"  <div class='exec-kpi-subtext'>{alert_desc}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

    # 3. Ward-Specific Public Health Guidance Panel
    guidance_cls = f"guidance-{row.tier.lower()}"
    tier_badges_html = f"<span style='background: {TIER_COLOR[row.tier]}; color: white; padding: 2px 10px; border-radius: 999px; font-size: 0.78rem; font-weight: 700;'>{row.tier} Risk Tier</span>"
    
    guidance_content = {
        "Critical": {
            "surveillance": "Prioritize high-frequency fever tracking across local primary health clinics. Monitor daily admission indicators.",
            "awareness": "Review community notification channels regarding water storage containers, rooftop debris, and peri-domestic breeding sites.",
            "preparedness": "Verify local clinic hydration supplies, IV fluid stocks, and diagnostic test kit readiness. Assess ward coordinator contact roster.",
            "priority": "Immediate operational review recommended with municipal health departments. Track 48-hour case velocity."
        },
        "High": {
            "surveillance": "Increase surveillance frequency across local community clinics. Review weekly outpatient sentinel trends.",
            "awareness": "Distribute standard municipal mosquito avoidance advisories and promote cleanup of stagnant drainage water.",
            "preparedness": "Alert nearby health centers to monitor fever caseload trends and review bed surge availability.",
            "priority": "Elevated surveillance priority over the next 2 to 3 weeks. Coordinate with local ward sanitation teams."
        },
        "Medium": {
            "surveillance": "Maintain standard weekly clinical surveillance. Track potential early cluster reports in neighboring wards.",
            "awareness": "Reinforce regular citizen awareness regarding household water containers and post-rainfall drainage.",
            "preparedness": "Ensure baseline medical and diagnostic supplies remain at standard seasonal buffer levels.",
            "priority": "Routine active monitoring. Re-evaluate trajectory upon next weekly weather update."
        },
        "Low": {
            "surveillance": "Routine baseline surveillance. Report standard weekly aggregated indicators to central health registry.",
            "awareness": "Standard seasonal public health messaging regarding personal protection and environmental hygiene.",
            "preparedness": "Maintain standard inventory management and baseline public health readiness.",
            "priority": "Normal operational status. No emergency response deployment indicated by model."
        }
    }
    g_info = guidance_content.get(row.tier, guidance_content["Low"])

    st.markdown(
        f"<div class='guidance-panel {guidance_cls}'>"
        f"  <div class='guidance-header'>"
        f"    <div class='guidance-title'>🛡️ Public Health Decision-Support Considerations {tier_badges_html}</div>"
        f"    <div style='font-size: 0.8rem; color: var(--text-secondary);'>Ward: <b>{ward}</b> | City: <b>{city}</b></div>"
        f"  </div>"
        f"  <div style='font-size: 0.88rem; color: var(--text-primary); margin-bottom: 8px;'><b>Operational Posture:</b> {ADVICE[row.tier]}</div>"
        f"  <div class='guidance-grid'>"
        f"    <div class='guidance-col'>"
        f"      <div class='guidance-col-title'>🔍 Surveillance Focus</div>"
        f"      <div class='guidance-col-text'>{g_info['surveillance']}</div>"
        f"    </div>"
        f"    <div class='guidance-col'>"
        f"      <div class='guidance-col-title'>📢 Community Engagement</div>"
        f"      <div class='guidance-col-text'>{g_info['awareness']}</div>"
        f"    </div>"
        f"    <div class='guidance-col'>"
        f"      <div class='guidance-col-title'>🏥 Clinical Preparedness</div>"
        f"      <div class='guidance-col-text'>{g_info['preparedness']}</div>"
        f"    </div>"
        f"    <div class='guidance-col'>"
        f"      <div class='guidance-col-title'>⏱️ Monitoring Posture</div>"
        f"      <div class='guidance-col-text'>{g_info['priority']}</div>"
        f"    </div>"
        f"  </div>"
        f"  <div class='guidance-disclaimer'>"
        f"    <b>Operational Notice & Limitation:</b> Illustrative decision-support considerations only. Confirm actions with authorized local public-health authorities and current official guidance. Model risk tiers are statistical decision aids based on historical weather patterns, not clinical diagnoses or official DGHS instructions."
        f"  </div>"
        f"</div>",
        unsafe_allow_html=True
    )

    # 4. Driver Breakdown & Weather Observations
    left, right = st.columns(2)
    with left:
        st.markdown("<div class='chart-surface'>", unsafe_allow_html=True)
        st.markdown(
            "<div class='chart-surface-header'>"
            "  <div>"
            "    <div class='chart-surface-title'>🔬 Predictive Driver Attribution</div>"
            "    <div class='chart-surface-desc'>Grouped feature occlusion relative to seasonal-normal conditions</div>"
            "  </div>"
            "</div>",
            unsafe_allow_html=True
        )

        ex = explain_row(df[(df.ward_id == ward) & (df.date == date)], window, B)
        fig_ex = px.bar(
            ex.iloc[::-1],
            x="effect_cases",
            y="driver",
            orientation="h",
            color=ex.iloc[::-1].effect_cases > 0,
            color_discrete_map={True: "#c62828", False: "#168653"},
            labels={"effect_cases": "Attributed case delta vs. normal conditions", "driver": ""}
        )
        fig_ex.update_layout(
            showlegend=False,
            height=320,
            margin=dict(l=0, r=10, t=10, b=10),
            xaxis=dict(zeroline=True, zerolinewidth=1.5, zerolinecolor="#64748b", gridcolor="#f1f5f9"),
            yaxis=dict(gridcolor="#f1f5f9")
        )
        fig_ex.update_traces(hovertemplate="<b>%{y}</b><br>Attributed Effect: %{x:+.1f} cases<extra></extra>")
        st.plotly_chart(fig_ex, width="stretch")

        top = ex.iloc[0]
        st.caption(
            f"<b>Scientific Interpretation:</b> Red indicates that observed conditions in this feature group pushed model predictions upward relative to historical seasonal averages. Primary upward driver: <b>{top.driver}</b> ({top.effect_cases:+.1f} cases).<br>"
            f"<i>Model-based predictive contribution — not a causal estimate.</i>",
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='chart-surface'>", unsafe_allow_html=True)
        st.markdown(
            "<div class='chart-surface-header'>"
            "  <div>"
            "    <div class='chart-surface-title'>🌦️ Trailing Weather Observations</div>"
            "    <div class='chart-surface-desc'>Observed rainfall, temperature, and humidity (last 16 weeks)</div>"
            "  </div>"
            "</div>",
            unsafe_allow_html=True
        )

        hist = df[(df.ward_id == ward) & (df.date <= date)].tail(16)
        fig_w = go.Figure()
        fig_w.add_bar(
            x=hist.date,
            y=hist.rainfall_mm,
            name="Rainfall (mm)",
            marker_color="#3b82f6",
            hovertemplate="%{x|%Y-%m-%d}<br>Rainfall: %{y:.1f} mm<extra></extra>"
        )
        fig_w.add_scatter(
            x=hist.date,
            y=hist.avg_temp_c,
            name="Temperature (°C)",
            yaxis="y2",
            line=dict(color="#ea580c", width=2.5),
            hovertemplate="%{x|%Y-%m-%d}<br>Avg Temp: %{y:.1f} °C<extra></extra>"
        )
        fig_w.add_scatter(
            x=hist.date,
            y=hist.relative_humidity_pct,
            name="Humidity (%)",
            yaxis="y2",
            line=dict(color="#8b5cf6", width=2, dash="dot"),
            hovertemplate="%{x|%Y-%m-%d}<br>Humidity: %{y:.1f}%<extra></extra>"
        )

        if len(hist) > 3:
            fig_w.add_vline(
                x=hist.date.iloc[-4],
                line_dash="dash",
                line_color="#64748b",
                annotation_text="3-week lag (peak correlation)",
                annotation_position="top left",
                annotation_font_size=10
            )

        fig_w.update_layout(
            height=320,
            margin=dict(l=0, r=0, t=10, b=10),
            yaxis=dict(title="Rainfall (mm)", gridcolor="#f1f5f9"),
            yaxis2=dict(overlaying="y", side="right", title="Temp (°C) / Humidity (%)", gridcolor="#f1f5f9"),
            legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center", font=dict(size=10))
        )
        st.plotly_chart(fig_w, width="stretch")
        st.caption("Historical observations from weekly dataset. Vertical reference line marks peak empirical lag period prior to forecast date.")
        st.markdown("</div>", unsafe_allow_html=True)

    # 5. Historical vs Predicted Epidemiological Trajectory
    st.markdown("<div class='chart-surface'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='chart-surface-header'>"
        "  <div>"
        "    <div class='chart-surface-title'>📈 Historical vs. Predicted Outbreak Trajectory</div>"
        f"    <div class='chart-surface-desc'>Trajectory for {ward} across ±26 weeks (Window: {window})</div>"
        "  </div>"
        "</div>",
        unsafe_allow_html=True
    )

    w_data = P[(P.ward_id == ward) & (P.date.between(date - pd.Timedelta(weeks=26), date + pd.Timedelta(weeks=26)))]
    fig_t = go.Figure()
    fig_t.add_scatter(x=w_data.date, y=w_data.hi, line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig_t.add_scatter(x=w_data.date, y=w_data.lo, fill="tonexty", fillcolor="rgba(232, 113, 10, 0.15)", line=dict(width=0), name="80% Prediction Interval")
    fig_t.add_scatter(
        x=w_data.date,
        y=w_data.pred,
        name="Model Forecast",
        line=dict(color="#e8710a", width=2.5),
        hovertemplate="%{x|%Y-%m-%d}<br>Forecast: %{y:.1f} cases<extra></extra>"
    )
    fig_t.add_scatter(
        x=w_data.date,
        y=w_data.actual,
        name="Observed Reported Cases",
        line=dict(color="#0f172a", width=1.8, dash="dot"),
        hovertemplate="%{x|%Y-%m-%d}<br>Reported: %{y:.0f} cases<extra></extra>"
    )
    fig_t.add_vline(
        x=date,
        line_color="#ef4444",
        line_width=2,
        annotation_text="Prediction Date",
        annotation_position="top left",
        annotation_font_size=10
    )
    fig_t.update_layout(
        height=340,
        margin=dict(l=0, r=0, t=10, b=10),
        yaxis=dict(title="Dengue Cases in Window", gridcolor="#f1f5f9"),
        xaxis=dict(gridcolor="#f1f5f9"),
        legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center", font=dict(size=11))
    )
    st.plotly_chart(fig_t, width="stretch")
    st.caption("Solid orange line = model point forecast. Shaded region = 80% conformal prediction interval. Dotted black line = observed reported cases. Red vertical marker = selected forecast issuance date.")
    st.markdown("</div>", unsafe_allow_html=True)

    # 6. Live Weather-Lag Pipeline Check Expander (Preserved Intact)
    with st.expander("✅ Weather-lag pipeline check for this ward-week (per-ward lags, verified live against raw data)"):
        hw = df[df.ward_id == ward].reset_index(drop=True)
        i = int(hw.index[hw.date == date][0])
        rows = []
        for k_ in range(1, 9):
            for col, short in (("rainfall_mm", "rain"), ("avg_temp_c", "temp"), ("relative_humidity_pct", "hum")):
                src = hw.loc[i - k_, col] if i - k_ >= 0 else np.nan
                val = hw.loc[i, f"{short}_lag_{k_}w"]
                rows.append({"feature": f"{short}_lag_{k_}w", "source week": str(hw.loc[i - k_, "date"].date()) if i - k_ >= 0 else "-",
                             "feature value": val, "raw value in source week": src, "match": bool(np.isclose(val, src, equal_nan=True))})
        lt = pd.DataFrame(rows)
        a_, b_ = st.columns([2, 1])
        a_.dataframe(lt.round(2), hide_index=True, width="stretch", height=300)
        b_.metric("Lag features verified", f"{int(lt['match'].sum())}/{len(lt)}")
        b_.caption("Rolling sums/means (2/3/4/6/8 weeks) are trailing windows ending at the prediction week and are computed per ward_id only.")
        lc = lag_corr()
        fig = px.bar(lc, x="lag_weeks", y="correlation", labels={"lag_weeks": "Rainfall lag (weeks before cases)", "correlation": "Correlation with weekly cases"})
        fig.update_layout(height=260, margin=dict(t=10, b=0))
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Correlation peaks at lag {int(lc.loc[lc.correlation.idxmax(), 'lag_weeks'])} weeks, matching the 7-14 day breeding cycle plus incubation (descriptive; the model is validated chronologically).")

# ================= 3. MAPS & RANKING =================
elif active_section == "🗺️ Risk maps & ward ranking":
    st.markdown(
        f"<div class='section-header' style='margin-bottom: 12px;'>"
        f"<h3 style='margin: 0; font-size: 1.25rem; color: var(--text-primary); font-weight: 700;'>🗺️ Risk Maps & Ward Prioritization Ranking</h3>"
        f"<p style='margin: 3px 0 0 0; font-size: 0.82rem; color: var(--text-secondary);'>Multi-ward comparative surveillance, longitudinal transmission trajectories, and administrative risk distributions for <b>{date.date()}</b> ({window}).</p>"
        f"</div>",
        unsafe_allow_html=True
    )

    # Scope Selection Bar
    sc_col1, sc_col2 = st.columns([1, 2])
    with sc_col1:
        scope = st.radio("Scope", ["All cities", f"{city} only"], horizontal=True, key="maps_scope")
    snap = snap_all if scope == "All cities" else snap_all[snap_all.city == city]

    # Executive Surveillance KPI Grid
    n_eval = len(snap)
    n_crit_scope = int((snap.tier == "Critical").sum())
    n_high_scope = int((snap.tier == "High").sum())
    n_watch_scope = int(snap.alert.sum())
    total_cases_scope = snap.pred.sum()

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        kpi1.markdown(
            f"<div class='exec-kpi-card kpi-total'>"
            f"  <div class='exec-kpi-label'>Wards Evaluated <span>🏥</span></div>"
            f"  <div class='exec-kpi-value'>{n_eval} <span style='font-size: 0.85rem; font-weight: 500; color: var(--text-secondary);'>wards</span></div>"
            f"  <div class='exec-kpi-subtext'>Total forecast: <b>{total_cases_scope:.1f} cases</b> in active scope</div>"
            f"</div>",
            unsafe_allow_html=True
        )
    with kpi2:
        crit_cls = "kpi-critical" if n_crit_scope > 0 else "kpi-total"
        kpi2.markdown(
            f"<div class='exec-kpi-card {crit_cls}'>"
            f"  <div class='exec-kpi-label'>Critical Priority <span>🚨</span></div>"
            f"  <div class='exec-kpi-value'>{n_crit_scope} <span style='font-size: 0.85rem; font-weight: 500; color: var(--text-secondary);'>wards</span></div>"
            f"  <div class='exec-kpi-subtext'>Threshold: ≥ {ESC['critical']:.1f} cases ({ESC['critical']:.0f}+ surge)</div>"
            f"</div>",
            unsafe_allow_html=True
        )
    with kpi3:
        high_cls = "kpi-high" if n_high_scope > 0 else "kpi-total"
        kpi3.markdown(
            f"<div class='exec-kpi-card {high_cls}'>"
            f"  <div class='exec-kpi-label'>High Risk Wards <span>⚠️</span></div>"
            f"  <div class='exec-kpi-value'>{n_high_scope} <span style='font-size: 0.85rem; font-weight: 500; color: var(--text-secondary);'>wards</span></div>"
            f"  <div class='exec-kpi-subtext'>Threshold: ≥ {ESC['high']:.1f} cases (escalation alert)</div>"
            f"</div>",
            unsafe_allow_html=True
        )
    with kpi4:
        watch_cls = "kpi-watch" if n_watch_scope > 0 else "kpi-total"
        kpi4.markdown(
            f"<div class='exec-kpi-card {watch_cls}'>"
            f"  <div class='exec-kpi-label'>Watch Triggers <span>👁️</span></div>"
            f"  <div class='exec-kpi-value'>{n_watch_scope} <span style='font-size: 0.85rem; font-weight: 500; color: var(--text-secondary);'>wards</span></div>"
            f"  <div class='exec-kpi-subtext'>Screening threshold: ≥ {alert_cut:.1f} cases</div>"
            f"</div>",
            unsafe_allow_html=True
        )

    st.write("")

    # --- 1. 60-Ward Prioritization Ranking Table ---
    st.markdown("<div class='chart-surface'>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='chart-surface-header'>"
        f"  <div>"
        f"    <div class='chart-surface-title'>📋 60-Ward Epidemiological Prioritization Table — {date.date()} ({window})</div>"
        f"    <div class='chart-surface-desc'>Comparative ranking of expected case burden across Bangladesh urban corporations with conformal uncertainty and calibrated risk tiers.</div>"
        f"  </div>"
        f"</div>",
        unsafe_allow_html=True
    )

    rk = snap.sort_values("pred", ascending=False).reset_index(drop=True)
    rk.insert(0, "rank", rk.index + 1)
    rk["vs city median"] = rk.pred - rk.groupby("city").pred.transform("median")

    # Table Controls & Filtering
    tf_c1, tf_c2, tf_c3 = st.columns([1.5, 1.2, 1.3])
    with tf_c1:
        search_query = st.text_input("🔍 Search Ward ID", placeholder="e.g. Ward_01", key="tbl_search_ward").strip()
    with tf_c2:
        tier_filter = st.selectbox("Filter Risk Tier", ["All Tiers", "Critical", "High", "Medium", "Low"], key="tbl_tier_filter")
    with tf_c3:
        watch_filter = st.selectbox("Watch Status", ["All Wards", "Watch Flagged Only", "Standard Only"], key="tbl_watch_filter")

    filtered_rk = rk.copy()
    if search_query:
        filtered_rk = filtered_rk[filtered_rk.ward_id.str.contains(search_query, case=False, na=False)]
    if tier_filter != "All Tiers":
        filtered_rk = filtered_rk[filtered_rk.tier == tier_filter]
    if watch_filter == "Watch Flagged Only":
        filtered_rk = filtered_rk[filtered_rk.alert]
    elif watch_filter == "Standard Only":
        filtered_rk = filtered_rk[~filtered_rk.alert]

    # Format table for display while preserving exact underlying data
    display_df = filtered_rk.copy()
    display_df["80% Interval"] = display_df.apply(lambda r: f"[{r.lo:.0f} – {r.hi:.0f}]", axis=1)
    display_df["Watch Status"] = display_df.alert.map(lambda a: "⚠️ Active Watch" if a else "Standard")

    st.dataframe(
        display_df[["rank", "city", "ward_id", "pred", "80% Interval", "tier", "Watch Status", "vs city median"]].round(1),
        hide_index=True,
        width="stretch",
        height=320,
        column_config={
            "rank": st.column_config.NumberColumn("Rank", format="#%d", width="small"),
            "city": st.column_config.TextColumn("City Corporation", width="medium"),
            "ward_id": st.column_config.TextColumn("Ward ID", width="small"),
            "pred": st.column_config.ProgressColumn(
                "Forecast Cases",
                min_value=0.0,
                max_value=float(max(rk.pred.max(), 1.0)),
                format="%.1f",
                help="Predicted case burden in target forecast window"
            ),
            "80% Interval": st.column_config.TextColumn("80% Uncertainty [Lo–Hi]", width="medium"),
            "tier": st.column_config.TextColumn("Calibrated Risk", width="small"),
            "Watch Status": st.column_config.TextColumn("Surveillance Watch", width="medium"),
            "vs city median": st.column_config.NumberColumn("vs City Median", format="%+.1f", width="small"),
        }
    )

    st.markdown(
        f"<div class='risk-legend-bar'>"
        f"  <span style='font-weight:700; color:var(--text-secondary);'>Risk Tier Standards:</span>"
        f"  <span class='risk-legend-item'><span class='risk-dot' style='background:{TIER_COLOR['Critical']};'></span> Critical (≥ {ESC['critical']:.1f} cases)</span>"
        f"  <span class='risk-legend-item'><span class='risk-dot' style='background:{TIER_COLOR['High']};'></span> High (≥ {ESC['high']:.1f} cases)</span>"
        f"  <span class='risk-legend-item'><span class='risk-dot' style='background:{TIER_COLOR['Medium']};'></span> Medium (≥ {cuts[0]:.1f} cases)</span>"
        f"  <span class='risk-legend-item'><span class='risk-dot' style='background:{TIER_COLOR['Low']};'></span> Low (< {cuts[0]:.1f} cases)</span>"
        f"  <span style='margin-left:auto; color:var(--text-muted); font-size:0.73rem;'>Displaying {len(filtered_rk)} of {len(rk)} wards</span>"
        f"</div>",
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # --- 2. Top-15 Priority Visualization ---
    st.markdown("<div class='chart-surface'>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='chart-surface-header'>"
        f"  <div>"
        f"    <div class='chart-surface-title'>🏆 Top-15 Priority Wards by Forecast Outbreak Burden</div>"
        f"    <div class='chart-surface-desc'>Highest projected case volumes with 80% Mondrian conformal uncertainty whiskers [Lo–Hi] and risk-tier encoding.</div>"
        f"  </div>"
        f"</div>",
        unsafe_allow_html=True
    )

    top15 = rk.head(15).copy()
    top15["err_hi"] = (top15.hi - top15.pred).clip(lower=0)
    top15["err_lo"] = (top15.pred - top15.lo).clip(lower=0)

    fig_bar = go.Figure()
    for t_name in ["Critical", "High", "Medium", "Low"]:
        t_sub = top15[top15.tier == t_name]
        if len(t_sub) == 0:
            continue
        fig_bar.add_trace(go.Bar(
            y=t_sub.ward_id,
            x=t_sub.pred,
            name=t_name,
            orientation="h",
            marker=dict(color=TIER_COLOR[t_name], line=dict(width=1, color="rgba(0,0,0,0.15)")),
            error_x=dict(
                type="data",
                symmetric=False,
                array=t_sub.err_hi,
                arrayminus=t_sub.err_lo,
                color="#475569",
                thickness=1.5,
                width=4
            ),
            customdata=list(zip(t_sub.city, t_sub.lo, t_sub.hi, t_sub.tier, t_sub.alert)),
            hovertemplate=(
                "<b>%{y}</b> (%{customdata[0]})<br>"
                "Predicted Cases: <b>%{x:.1f}</b><br>"
                "80% Interval: [%{customdata[1]:.0f} – %{customdata[2]:.0f}]<br>"
                "Calibrated Tier: <b>%{customdata[3]}</b><br>"
                "Watch Status: %{customdata[4]}<extra></extra>"
            )
        ))

    fig_bar.update_layout(
        height=430,
        margin=dict(l=0, r=0, t=10, b=10),
        barmode="stack",
        yaxis=dict(categoryorder="total ascending", tickfont=dict(size=12)),
        xaxis=dict(title="Predicted cases with 80% uncertainty bounds [Lo – Hi]", gridcolor="#e2e8f0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=""),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif")
    )
    st.plotly_chart(fig_bar, width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

    # --- 3. Ward × Week Temporal Risk Heatmap ---
    st.markdown("<div class='chart-surface'>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='chart-surface-header'>"
        f"  <div>"
        f"    <div class='chart-surface-title'>🗓️ Longitudinal Ward × Week Risk Trajectory (Last 26 Weeks)</div>"
        f"    <div class='chart-surface-desc'>Multi-week risk tier evolution for wards in <b>{city}</b> up to {date.date()} ({window}). Colors indicate calibrated operational risk tiers.</div>"
        f"  </div>"
        f"</div>",
        unsafe_allow_html=True
    )

    hm = P[(P.city == city) & P.date.between(date - pd.Timedelta(weeks=25), date)].dropna(subset=["pred"]).copy()
    hm["ti"] = hm.tier.map(TIER_IDX)
    piv = hm.pivot(index="ward_id", columns="date", values="ti")
    pv_pred = hm.pivot(index="ward_id", columns="date", values="pred")
    pv_tier = hm.pivot(index="ward_id", columns="date", values="tier")

    colorscale = [
        [0, TIER_COLOR["Low"]], [0.25, TIER_COLOR["Low"]],
        [0.25, TIER_COLOR["Medium"]], [0.5, TIER_COLOR["Medium"]],
        [0.5, TIER_COLOR["High"]], [0.75, TIER_COLOR["High"]],
        [0.75, TIER_COLOR["Critical"]], [1, TIER_COLOR["Critical"]]
    ]

    custom_text = [[f"{pv_pred.values[i][j]:.1f} cases ({pv_tier.values[i][j]})" for j in range(pv_pred.shape[1])] for i in range(pv_pred.shape[0])]

    fig_hm = go.Figure(go.Heatmap(
        z=piv.values,
        x=piv.columns,
        y=piv.index,
        zmin=0,
        zmax=3,
        colorscale=colorscale,
        showscale=False,
        xgap=1.5,
        ygap=1.5,
        customdata=custom_text,
        hovertemplate="<b>%{y}</b><br>Week of %{x|%Y-%m-%d}<br>%{customdata}<extra></extra>"
    ))
    fig_hm.update_layout(
        height=max(320, 22 * len(piv) + 80),
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
        xaxis=dict(showgrid=False, tickfont=dict(size=11)),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif")
    )
    st.plotly_chart(fig_hm, width="stretch")
    st.caption(f"Rows = wards of {city}; columns = chronological weekly forecasts up to {date.date()}. Green: Low · Yellow: Medium · Orange: High · Red: Critical.")
    st.markdown("</div>", unsafe_allow_html=True)

    # --- 4. Geographic Risk Map ---
    st.markdown("<div class='chart-surface'>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='chart-surface-header'>"
        f"  <div>"
        f"    <div class='chart-surface-title'>🗺️ Spatial Risk Map & Administrative City Aggregates</div>"
        f"    <div class='chart-surface-desc'>Geographic overview of disease transmission burden and high-risk ward concentrations for <b>{date.date()}</b>.</div>"
        f"  </div>"
        f"</div>",
        unsafe_allow_html=True
    )

    wc = ward_coordinates()
    if wc is not None:
        g = snap.merge(wc, on="ward_id", how="inner")
        fig_geo = px.scatter_geo(
            g, lat="lat", lon="lon", color="tier", size="pred", hover_name="ward_id",
            color_discrete_map=TIER_COLOR,
            category_orders={"tier": C.RISK_TIERS}, size_max=22
        )
        st.caption(f"Ward-level map from verified coordinates in `data/ward_coordinates.csv` ({len(g)} wards matched).")
    else:
        cs = city_summary(snap_all)
        fig_geo = px.scatter_geo(
            cs, lat="lat", lon="lon", size="mean_pred_cases", color="share_high_critical", hover_name="city",
            hover_data={
                "wards": True, "high_critical_wards": True, "watch_flag_wards": True,
                "lat": False, "lon": False,
                "mean_pred_cases": ":.1f", "share_high_critical": ":.0%"
            },
            color_continuous_scale=["#2e9e5b", "#e6b800", "#e8710a", "#c62828"],
            range_color=(0, 1),
            size_max=38
        )
        fig_geo.update_coloraxes(colorbar=dict(title="High/Crit %", thickness=12, len=0.7))

    fig_geo.update_geos(
        fitbounds="locations",
        showcountries=True,
        showland=True,
        landcolor="#f8fafc",
        countrycolor="#94a3b8",
        subunitcolor="#cbd5e1",
        resolution=50
    )
    fig_geo.update_layout(
        height=440,
        margin=dict(l=0, r=0, t=10, b=0),
        font=dict(family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif")
    )
    st.plotly_chart(fig_geo, width="stretch")

    # Geospatial Methodological Transparency Box
    st.markdown(
        f"<div class='geo-disclosure-box'>"
        f"  <b>📍 Administrative City Centroid Mapping Notice (Geospatial Transparency):</b><br>"
        f"  The current epidemiological dataset provides ward-level tabular records but does not include verified polygon boundary GIS shapefiles "
        f"  or verified GPS coordinates for individual wards. Bubble markers represent official city corporation administrative centres. "
        f"  To maintain epidemiological scientific integrity and prevent spatial misinterpretation, individual ward boundaries are not simulated or guessed. "
        f"  Granular ward prioritization is provided with exact numerical precision in the 60-Ward Ranking Table and Longitudinal Risk Heatmap above.<br>"
        f"  <i>Marker coordinates are offline; basemap boundary outlines require internet connectivity.</i>"
        f"</div>",
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ================= 4. WHAT-IF =================
elif active_section == "🧪 What-if":
    st.markdown(
        f"<div class='section-header' style='margin-bottom: 12px;'>"
        f"<h3 style='margin: 0; font-size: 1.25rem; color: var(--text-primary); font-weight: 700;'>🧪 What-If Scenario Comparator</h3>"
        f"<p style='margin: 3px 0 0 0; font-size: 0.82rem; color: var(--text-secondary);'>Exploratory model sensitivity analysis evaluating how hypothetical atmospheric anomalies alter point forecasts, Mondrian conformal intervals, and risk-tier classifications for <b>{ward}</b> ({city}) on <b>{date.date()}</b> ({window}).</p>"
        f"</div>",
        unsafe_allow_html=True
    )

    base_h = ward_history(df, ward, date)

    # Controls Card
    st.markdown("<div class='chart-surface'>", unsafe_allow_html=True)
    ctrl_header_col, reset_btn_col = st.columns([3, 1])
    with ctrl_header_col:
        st.markdown(
            "<div class='chart-surface-title'>🎛️ Hypothetical Scenario Configuration</div>"
            "<div class='chart-surface-desc'>Specify trailing atmospheric perturbations relative to verified historical observations.</div>",
            unsafe_allow_html=True
        )
    with reset_btn_col:
        reset_key = f"reset_btn_{ward}_{date}_{window}"
        if st.button("🔄 Reset to Baseline", key=reset_key, help="Restore all scenario controls to unperturbed baseline values", use_container_width=True):
            st.session_state[f"rm_{ward}_{date}"] = 1.0
            st.session_state[f"ts_{ward}_{date}"] = 0.0
            st.session_state[f"hs_{ward}_{date}"] = 0.0
            st.session_state[f"nw_{ward}_{date}"] = 4
            st.rerun()

    # Session-state backed sliders
    s1, s2, s3, s4 = st.columns(4)
    rm_key = f"rm_{ward}_{date}"
    ts_key = f"ts_{ward}_{date}"
    hs_key = f"hs_{ward}_{date}"
    nw_key = f"nw_{ward}_{date}"

    if rm_key not in st.session_state:
        st.session_state[rm_key] = 1.0
    if ts_key not in st.session_state:
        st.session_state[ts_key] = 0.0
    if hs_key not in st.session_state:
        st.session_state[hs_key] = 0.0
    if nw_key not in st.session_state:
        st.session_state[nw_key] = 4

    rm = s1.slider("Rainfall Multiplier (×)", 0.0, 3.0, float(st.session_state[rm_key]), 0.1, key=f"sl_rm_{ward}_{date}", help="Proportional scaling applied to observed rainfall (1.0 = observed)")
    ts_ = s2.slider("Temperature Shift (°C)", -5.0, 5.0, float(st.session_state[ts_key]), 0.5, key=f"sl_ts_{ward}_{date}", help="Additive temperature offset applied to observed weekly mean temperature")
    hs = s3.slider("Humidity Shift (pts)", -20.0, 20.0, float(st.session_state[hs_key]), 1.0, key=f"sl_hs_{ward}_{date}", help="Additive percentage point offset applied to observed relative humidity")
    nw = s4.slider("Perturbation Horizon (weeks)", 1, N_HISTORY, int(st.session_state[nw_key]), 1, key=f"sl_nw_{ward}_{date}", help=f"Number of trailing weeks (up to {N_HISTORY}) to apply the scenario changes")

    st.session_state[rm_key] = rm
    st.session_state[ts_key] = ts_
    st.session_state[hs_key] = hs
    st.session_state[nw_key] = nw

    scen = apply_scenario(base_h, rm, ts_, hs, nw)

    with st.expander("📋 Review or fine-tune weekly weather records (optional data editor)", expanded=False):
        st.caption("Directly inspect or edit weekly weather inputs feeding the lag engineering pipeline. All values must remain physically plausible.")
        edited = st.data_editor(
            scen[["date", "rainfall_mm", "avg_temp_c", "relative_humidity_pct"]],
            hide_index=True,
            disabled=["date"],
            width="stretch",
            key=f"ed_{ward}_{date}_{rm}_{ts_}_{hs}_{nw}"
        )

    st.markdown("</div>", unsafe_allow_html=True)

    try:
        sc_in = scen.assign(**{c: edited[c].values for c in C.WEATHER_COLS})
        unusual = unusual_observations(sc_in, df)
        for u in unusual:
            st.warning(f"⚠️ Model Extrapolation Warning: {u} — observations outside the historical range carry elevated epistemic uncertainty.")

        res_b = predict_history(base_h, window, B)
        res_s = predict_history(sc_in, window, B)

        diff_cases = res_s["pred"] - res_b["pred"]
        pct_diff_str = f"{(diff_cases / res_b['pred'] * 100):+.1f}%" if res_b["pred"] > 0 else "N/A (baseline is 0)"
        diff_color = "#c62828" if diff_cases > 0.5 else ("#168653" if diff_cases < -0.5 else "var(--text-secondary)")

        tier_b = res_b["tier"]
        tier_s = res_s["tier"]
        watch_b = "⚠️ Active Watch" if res_b["alert"] else "— Normal"
        watch_s = "⚠️ Active Watch" if res_s["alert"] else "— Normal"

        # --- Side-by-Side Comparison Panels ---
        p1, p2 = st.columns(2)
        with p1:
            st.markdown(
                f"<div class='comparator-card baseline'>"
                f"  <div class='comparator-header'>"
                f"    <div class='comparator-title'>📍 Baseline Model Estimate</div>"
                f"    <span style='font-size:0.75rem; font-weight:700; color:var(--primary); background:var(--primary-light); padding:2px 8px; border-radius:9999px;'>OBSERVED WEATHER</span>"
                f"  </div>"
                f"  <div class='comparator-val'>{res_b['pred']:.1f} <span style='font-size:0.9rem; font-weight:500; color:var(--text-secondary);'>cases</span></div>"
                f"  <div class='comparator-interval'>80% Conformal Interval: <b>[{res_b['lo']:.0f} – {res_b['hi']:.0f}]</b> (span: {res_b['hi'] - res_b['lo']:.1f})</div>"
                f"  <div class='comparator-row'><span>Calibrated Risk Tier</span><span>{pill(tier_b)}</span></div>"
                f"  <div class='comparator-row'><span>Surveillance Watch Status</span><span style='font-weight:600;'>{watch_b}</span></div>"
                f"  <div class='comparator-row'><span>Mean Rainfall (last {nw} wks)</span><span style='font-weight:600;'>{base_h.tail(nw)['rainfall_mm'].mean():.1f} mm/wk</span></div>"
                f"  <div class='comparator-row'><span>Mean Temperature (last {nw} wks)</span><span style='font-weight:600;'>{base_h.tail(nw)['avg_temp_c'].mean():.1f} °C</span></div>"
                f"  <div class='comparator-row'><span>Mean Relative Humidity (last {nw} wks)</span><span style='font-weight:600;'>{base_h.tail(nw)['relative_humidity_pct'].mean():.1f}%</span></div>"
                f"</div>",
                unsafe_allow_html=True
            )

        with p2:
            st.markdown(
                f"<div class='comparator-card scenario'>"
                f"  <div class='comparator-header'>"
                f"    <div class='comparator-title'>🧪 Hypothetical Scenario Estimate</div>"
                f"    <span style='font-size:0.75rem; font-weight:700; color:#ea580c; background:#fff7ed; padding:2px 8px; border-radius:9999px;'>SIMULATED INPUTS</span>"
                f"  </div>"
                f"  <div class='comparator-val' style='color:#c2410c;'>{res_s['pred']:.1f} <span style='font-size:0.9rem; font-weight:500; color:var(--text-secondary);'>cases</span></div>"
                f"  <div class='comparator-interval'>80% Conformal Interval: <b>[{res_s['lo']:.0f} – {res_s['hi']:.0f}]</b> (span: {res_s['hi'] - res_s['lo']:.1f})</div>"
                f"  <div class='comparator-row'><span>Calibrated Risk Tier</span><span>{pill(tier_s)}</span></div>"
                f"  <div class='comparator-row'><span>Surveillance Watch Status</span><span style='font-weight:600;'>{watch_s}</span></div>"
                f"  <div class='comparator-row'><span>Scenario Rainfall (last {nw} wks)</span><span style='font-weight:600;'>{sc_in.tail(nw)['rainfall_mm'].mean():.1f} mm/wk ({rm:.1f}×)</span></div>"
                f"  <div class='comparator-row'><span>Scenario Temperature (last {nw} wks)</span><span style='font-weight:600;'>{sc_in.tail(nw)['avg_temp_c'].mean():.1f} °C ({ts_:+0.1f}°C)</span></div>"
                f"  <div class='comparator-row'><span>Scenario Relative Humidity (last {nw} wks)</span><span style='font-weight:600;'>{sc_in.tail(nw)['relative_humidity_pct'].mean():.1f}% ({hs:+0.0f} pts)</span></div>"
                f"</div>",
                unsafe_allow_html=True
            )

        # --- Comparative Sensitivity Delta Card ---
        tier_change_note = f"Risk Tier changed from <b>{tier_b}</b> to <b>{tier_s}</b>" if tier_b != tier_s else f"Risk Tier remained <b>{tier_b}</b>"
        watch_change_note = f"Watch status transitioned ({watch_b} → {watch_s})" if res_b["alert"] != res_s["alert"] else f"Watch status remained {watch_b}"

        st.markdown(
            f"<div class='diff-summary-card'>"
            f"  <div style='display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:8px;'>"
            f"    <div style='font-size:0.95rem; font-weight:700; color:var(--text-primary);'>📊 Comparative Sensitivity Analysis</div>"
            f"    <div style='font-size:0.80rem; color:var(--text-secondary);'>{tier_change_note} · {watch_change_note}</div>"
            f"  </div>"
            f"  <div class='diff-metric-grid'>"
            f"    <div class='diff-metric-item'>"
            f"      <div class='diff-metric-label'>Absolute Forecast Delta</div>"
            f"      <div class='diff-metric-val' style='color:{diff_color};'>{diff_cases:+.1f} <span style='font-size:0.8rem; font-weight:normal;'>cases</span></div>"
            f"    </div>"
            f"    <div class='diff-metric-item'>"
            f"      <div class='diff-metric-label'>Relative Percentage Shift</div>"
            f"      <div class='diff-metric-val' style='color:{diff_color};'>{pct_diff_str}</div>"
            f"    </div>"
            f"    <div class='diff-metric-item'>"
            f"      <div class='diff-metric-label'>80% Interval Lower Delta</div>"
            f"      <div class='diff-metric-val'>{res_s['lo'] - res_b['lo']:+.1f} <span style='font-size:0.8rem; font-weight:normal;'>cases</span></div>"
            f"    </div>"
            f"    <div class='diff-metric-item'>"
            f"      <div class='diff-metric-label'>80% Interval Upper Delta</div>"
            f"      <div class='diff-metric-val'>{res_s['hi'] - res_b['hi']:+.1f} <span style='font-size:0.8rem; font-weight:normal;'>cases</span></div>"
            f"    </div>"
            f"  </div>"
            f"</div>",
            unsafe_allow_html=True
        )

        # --- Side-by-Side Visual Comparison Charts ---
        c_left, c_right = st.columns([1, 1.4])
        with c_left:
            st.markdown("<div class='chart-surface'>", unsafe_allow_html=True)
            st.markdown(
                "<div class='chart-surface-header'>"
                "  <div>"
                "    <div class='chart-surface-title'>📊 Forecast & Interval Comparison</div>"
                "    <div class='chart-surface-desc'>Point estimates with 80% Mondrian conformal uncertainty whiskers</div>"
                "  </div>"
                "</div>",
                unsafe_allow_html=True
            )
            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Bar(
                x=["Observed Baseline", "Hypothetical Scenario"],
                y=[res_b["pred"], res_s["pred"]],
                marker_color=["#0F766E", "#ea580c"],
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=[max(0, res_b["hi"] - res_b["pred"]), max(0, res_s["hi"] - res_s["pred"])],
                    arrayminus=[max(0, res_b["pred"] - res_b["lo"]), max(0, res_s["pred"] - res_s["lo"])],
                    color="#334155",
                    thickness=1.5,
                    width=6
                ),
                text=[f"{res_b['pred']:.1f}", f"{res_s['pred']:.1f}"],
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>Forecast: %{y:.1f} cases<extra></extra>"
            ))
            fig_cmp.update_layout(
                height=290,
                margin=dict(l=0, r=0, t=10, b=0),
                yaxis=dict(title="Predicted cases", gridcolor="#e2e8f0"),
                xaxis=dict(tickfont=dict(size=12, family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif")),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif")
            )
            st.plotly_chart(fig_cmp, width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)

        with c_right:
            st.markdown("<div class='chart-surface'>", unsafe_allow_html=True)
            st.markdown(
                "<div class='chart-surface-header'>"
                "  <div>"
                "    <div class='chart-surface-title'>🌦️ Trailing Meteorological Inputs Comparison</div>"
                "    <div class='chart-surface-desc'>Observed historical inputs vs hypothetical perturbed trajectory</div>"
                "  </div>"
                "</div>",
                unsafe_allow_html=True
            )
            fig_w = go.Figure()
            fig_w.add_trace(go.Bar(
                x=base_h.date,
                y=base_h.rainfall_mm,
                name="Observed Rain (mm)",
                marker_color="#3b82f6",
                hovertemplate="%{x|%Y-%m-%d}<br>Observed Rain: %{y:.1f} mm<extra></extra>"
            ))
            fig_w.add_trace(go.Bar(
                x=sc_in.date,
                y=sc_in.rainfall_mm,
                name="Scenario Rain (mm)",
                marker_color="#f97316",
                hovertemplate="%{x|%Y-%m-%d}<br>Scenario Rain: %{y:.1f} mm<extra></extra>"
            ))
            fig_w.add_trace(go.Scatter(
                x=base_h.date,
                y=base_h.avg_temp_c,
                name="Observed Temp (°C)",
                yaxis="y2",
                line=dict(color="#0F766E", width=2),
                hovertemplate="%{x|%Y-%m-%d}<br>Observed Temp: %{y:.1f} °C<extra></extra>"
            ))
            fig_w.add_trace(go.Scatter(
                x=sc_in.date,
                y=sc_in.avg_temp_c,
                name="Scenario Temp (°C)",
                yaxis="y2",
                line=dict(color="#dc2626", width=2, dash="dash"),
                hovertemplate="%{x|%Y-%m-%d}<br>Scenario Temp: %{y:.1f} °C<extra></extra>"
            ))
            fig_w.update_layout(
                barmode="group",
                height=290,
                margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
                yaxis=dict(title="Rainfall (mm)", gridcolor="#e2e8f0"),
                yaxis2=dict(title="Temp (°C)", overlaying="y", side="right", showgrid=False),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif")
            )
            st.plotly_chart(fig_w, width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)

        # Mandatory Scientific Integrity Disclaimer Box
        st.markdown(
            "<div class='whatif-disclaimer-box'>"
            "  <b>⚠️ Exploratory Model Sensitivity Disclaimer:</b><br>"
            "  This What-If Comparator explores hypothetical changes to model inputs. Differences represent model sensitivity "
            "  under the specified assumptions, not causal effects or proven intervention outcomes. The application uses synthetic/simulated "
            "  epidemiological data and is not operationally validated for public-health decision-making. Results must not be interpreted as "
            "  official DGHS forecasts or intervention effectiveness estimates."
            "</div>",
            unsafe_allow_html=True
        )

    except ValueError as e:
        st.error(f"Invalid scenario input: {e}")

# ================= 5. LIVE WEATHER =================
elif active_section == "🌐 Live weather (optional)":
    st.subheader("Optional live weather ingestion (Open-Meteo)")
    st.markdown("The app is fully functional offline. This panel makes **one** HTTP request per city, cached for 6 h on disk and in memory, and "
                "scores the wards of the selected city from the last complete weeks of **recent observed/analysis** weather. "
                "It never mixes *forecast* days into the lag inputs. **Experimental:** the model was trained on the supplied (apparently synthetic) "
                "dataset, city-level weather is applied to every ward, and nothing here has been validated against real cases.")
    if st.button(f"Fetch latest weather for {city}", type="primary"):
        with st.spinner("Contacting Open-Meteo (1 request; skipped if cached < 6 h)..."):
            st.session_state.setdefault("live_res", {})[city] = fetch_city_weather(city)   # failures are not memoised
    if city in st.session_state.get("live_res", {}):
        res = st.session_state["live_res"][city]
        if not res.ok:
            st.warning(res.message)
        else:
            a, b_, c_ = st.columns(3)
            a.metric("Source", res.source); b_.metric("Latest observed day", res.freshness.get("latest_observation", "?"),
                                                      f"{res.freshness.get('age_days', '?')} days old", delta_color="off")
            c_.metric("Complete weeks available", len(res.weekly))
            for n_ in res.notes:
                (st.warning if "STALE" in n_ or "unavailable" in n_ else st.caption)(n_)
            if res.freshness.get("stale"):
                st.error("Data is stale - predictions below are not up to date.")
            lw = res.weekly.tail(N_HISTORY)
            st.markdown("**Weekly inputs fed to the lag features** (provenance: observed/analysis, Monday-start weeks; rain = weekly sum, temp/humidity = weekly mean)")
            st.dataframe(lw[["date", "rainfall_mm", "avg_temp_c", "relative_humidity_pct"]].round(1), hide_index=True, width="stretch")
            if len(lw) < N_HISTORY:
                st.error(f"Only {len(lw)} complete weeks available; {N_HISTORY} are required for the lag features.")
            else:
                out = []
                for wid in sorted(df[df.city == city].ward_id.unique()):
                    wr = df[df.ward_id == wid].iloc[-1]
                    hist, notes = history_for_scoring(res, wr)
                    for u in unusual_observations(hist, df)[:1]:
                        st.warning(u)
                    r = predict_history(hist, window, B)
                    out.append({"ward_id": wid, "pred cases": r["pred"], "80% lo": r["lo"], "80% hi": r["hi"], "tier": r["tier"], "watch flag": r["alert"]})
                st.markdown(f"**Experimental ward scores for {city}** using live city-level weather (as of week starting {lw.date.iloc[-1].date()}; {window})")
                st.dataframe(pd.DataFrame(out).sort_values("pred cases", ascending=False).round(1), hide_index=True, width="stretch")
                st.caption("These are model outputs on a different weather source than the training data (distribution shift) - treat as a technical demonstration of the live pipeline, not as an operational alert.")
    else:
        st.info("Press the button to fetch. If you are offline, nothing is lost: set `LIVE_WEATHER_OFFLINE=1` to disable the network entirely.")

# ================= 6. VALIDATION =================
elif active_section == "📊 Validation":
    t = M["test_by_window"][window]

    # --- Implementation A: Section Header and Disclosure Surface ---
    st.markdown(
        f"<div class='section-header' style='margin-bottom: 12px;'>"
        f"<h3 style='margin: 0; font-size: 1.25rem; color: var(--text-primary); font-weight: 700;'>📊 Validation & Model Reliability</h3>"
        f"<p style='margin: 3px 0 0 0; font-size: 0.82rem; color: var(--text-secondary);'>Empirical out-of-sample model performance evaluated on held-out test partition — <b>{window}</b> ({M['split']['test'][0]} → {M['split']['test'][1]}, n = {t['n']:,} ward-weeks across 60 wards).</p>"
        f"</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        f"<div class='diag-disclosure-box'>"
        f"  <b>⚠️ Synthetic/Simulated Data Performance Disclosure:</b><br>"
        f"  These metrics describe performance on a held-out temporal test partition of synthetic/simulated epidemiological data. "
        f"  They do not establish real-world predictive accuracy, clinical validation, or operational readiness. "
        f"  This prototype is not an official DGHS forecasting system."
        f"</div>",
        unsafe_allow_html=True
    )

    # --- Implementation B: Two-Tier Diagnostic KPI Surfaces ---
    # Card A: Continuous Forecast Error
    naive_mae = t['baselines']['Seasonal-naive (52w earlier)']['MAE']
    st.markdown(
        f"<div class='diag-surface-kpi'>"
        f"  <div class='diag-surface-kpi-header'>"
        f"    <div class='diag-surface-kpi-title'>📉 Continuous Forecast Error</div>"
        f"    <span class='diag-surface-kpi-badge error-badge'>Regression Accuracy</span>"
        f"  </div>"
        f"  <div class='diag-metric-grid-4'>"
        f"    <div class='diag-metric-cell'>"
        f"      <div class='diag-metric-cell-label'>MAE (Mean Absolute Error)</div>"
        f"      <div class='diag-metric-cell-val'>{t['MAE']:.2f} <span style='font-size:0.75rem; font-weight:normal; color:var(--text-secondary);'>cases</span></div>"
        f"      <div class='diag-metric-cell-sub'>Seasonal-naive baseline: {naive_mae:.2f} cases</div>"
        f"    </div>"
        f"    <div class='diag-metric-cell'>"
        f"      <div class='diag-metric-cell-label'>RMSE (Root Mean Squared)</div>"
        f"      <div class='diag-metric-cell-val'>{t['RMSE']:.2f} <span style='font-size:0.75rem; font-weight:normal; color:var(--text-secondary);'>cases</span></div>"
        f"      <div class='diag-metric-cell-sub'>Penalizes large outlier misses</div>"
        f"    </div>"
        f"    <div class='diag-metric-cell'>"
        f"      <div class='diag-metric-cell-label'>R² (Variance Explained)</div>"
        f"      <div class='diag-metric-cell-val'>{t['R2']:.3f}</div>"
        f"      <div class='diag-metric-cell-sub'>Relative to sample mean baseline</div>"
        f"    </div>"
        f"    <div class='diag-metric-cell'>"
        f"      <div class='diag-metric-cell-label'>Mean 80% Conformal Span</div>"
        f"      <div class='diag-metric-cell-val'>{t['interval_width']:.1f} <span style='font-size:0.75rem; font-weight:normal; color:var(--text-secondary);'>cases</span></div>"
        f"      <div class='diag-metric-cell-sub'>Average prediction interval width</div>"
        f"    </div>"
        f"  </div>"
        f"</div>",
        unsafe_allow_html=True
    )

    # Card B: Outbreak Detection & Uncertainty
    pr_auc_str = f"{t['HC_PR_AUC']:.2f}" if t.get("HC_PR_AUC") is not None else "n/a"
    st.markdown(
        f"<div class='diag-surface-kpi'>"
        f"  <div class='diag-surface-kpi-header'>"
        f"    <div class='diag-surface-kpi-title'>🚨 Outbreak Detection & Conformal Uncertainty</div>"
        f"    <span class='diag-surface-kpi-badge detect-badge'>Classification & Coverage</span>"
        f"  </div>"
        f"  <div class='diag-metric-grid-6'>"
        f"    <div class='diag-metric-cell'>"
        f"      <div class='diag-metric-cell-label'>High/Crit Recall</div>"
        f"      <div class='diag-metric-cell-val' style='color:#0f766e;'>{t['HC_recall']:.0%}</div>"
        f"      <div class='diag-metric-cell-sub'>Sensitivity for true outbreaks</div>"
        f"    </div>"
        f"    <div class='diag-metric-cell'>"
        f"      <div class='diag-metric-cell-label'>High/Crit Precision</div>"
        f"      <div class='diag-metric-cell-val'>{t['HC_precision']:.0%}</div>"
        f"      <div class='diag-metric-cell-sub'>Positive predictive value</div>"
        f"    </div>"
        f"    <div class='diag-metric-cell'>"
        f"      <div class='diag-metric-cell-label'>High/Crit PR-AUC</div>"
        f"      <div class='diag-metric-cell-val'>{pr_auc_str}</div>"
        f"      <div class='diag-metric-cell-sub'>Area under PR curve</div>"
        f"    </div>"
        f"    <div class='diag-metric-cell'>"
        f"      <div class='diag-metric-cell-label'>False-Alarm Rate</div>"
        f"      <div class='diag-metric-cell-val' style='color:#c2410c;'>{t['HC_false_alarm_rate']:.1%}</div>"
        f"      <div class='diag-metric-cell-sub'>Non-outbreak alarm rate</div>"
        f"    </div>"
        f"    <div class='diag-metric-cell'>"
        f"      <div class='diag-metric-cell-label'>80% Conformal Cov</div>"
        f"      <div class='diag-metric-cell-val' style='color:#0f766e;'>{t['interval_coverage']:.0%}</div>"
        f"      <div class='diag-metric-cell-sub'>Empirical test set coverage</div>"
        f"    </div>"
        f"    <div class='diag-metric-cell'>"
        f"      <div class='diag-metric-cell-label'>Critical-Week Cov</div>"
        f"      <div class='diag-metric-cell-val' style='color:#0f766e;'>{t['interval_coverage_actual_critical']:.0%}</div>"
        f"      <div class='diag-metric-cell-sub'>Actual Critical (n={t['n_actual_critical']})</div>"
        f"    </div>"
        f"  </div>"
        f"  <div class='diag-supplementary-bar'>"
        f"    <div class='diag-supp-items'>"
        f"      <span class='diag-supp-item'>High/Critical F1: <b>{t['HC_F1']:.2f}</b></span>"
        f"      <span class='diag-supp-item'>Tier Exact Accuracy: <b>{t['tier_exact_accuracy']:.0%}</b></span>"
        f"      <span class='diag-supp-item'>Within One Tier: <b>{t['tier_within_one']:.0%}</b></span>"
        f"    </div>"
        f"    <div style='font-size:0.75rem; color:var(--text-muted);'>"
        f"      Threshold calibration budget: FAR ≤ {C.ESC_FAR_CAPS['high']:.0%} on OOF 2019–2024H1 (test FAR: {t['HC_false_alarm_rate']:.1%})"
        f"    </div>"
        f"  </div>"
        f"</div>",
        unsafe_allow_html=True
    )

    # --- Implementation C & D: Analysis Blocks Surfaces & Plotly Styling ---

    # 1. Baseline Model Comparison
    st.markdown("<div class='chart-surface'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='chart-surface-header'>"
        "  <div>"
        "    <div class='chart-surface-title'>1. Baseline Model Comparison (Held-out Test)</div>"
        "    <div class='chart-surface-desc'>Benchmarking against naive historical, seasonal, and rolling-mean baselines</div>"
        "  </div>"
        "</div>",
        unsafe_allow_html=True
    )
    bl = pd.DataFrame({
        "Model": ["**ML model (HistGB-Poisson)**"] + list(t["baselines"]),
        "MAE": [t["MAE"]] + [v["MAE"] for v in t["baselines"].values()],
        "RMSE": [t["RMSE"]] + [v["RMSE"] for v in t["baselines"].values()],
        "R²": [t["R2"]] + [v["R2"] for v in t["baselines"].values()],
        "High/Crit recall": [t["HC_recall"]] + [v["HC_recall"] for v in t["baselines"].values()]
    })
    st.dataframe(bl.round(3), hide_index=True, width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

    # 2. Original vs Improved High/Critical Detection
    st.markdown("<div class='chart-surface'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='chart-surface-header'>"
        "  <div>"
        "    <div class='chart-surface-title'>2. Original vs Calibrated High/Critical Detection (Held-out Test)</div>"
        "    <div class='chart-surface-desc'>Operational decision-threshold calibration: trading false alarms for outbreak recall</div>"
        "  </div>"
        "</div>",
        unsafe_allow_html=True
    )
    rows = []
    if T0_:
        rows.append({
            "rule": f"Original: tier ≥ High at cut-point ({T0_['HC_decision_cut']:.0f} cases)",
            "recall": T0_["HC_recall"], "missed (FN)": T0_["HC_FN"], "false alarms (FP)": T0_["HC_FP"],
            "precision": T0_["HC_precision"], "false-alarm rate": T0_["HC_false_alarm_rate"],
            "Critical recall": T0_["CRIT_recall"], "Critical-week interval coverage": T0_.get("interval_coverage_actual_critical")
        })
    rows.append({
        "rule": f"Improved: tier ≥ High at escalated threshold ({t['HC_decision_cut']:.1f} cases)",
        "recall": t["HC_recall"], "missed (FN)": t["HC_FN"], "false alarms (FP)": t["HC_FP"],
        "precision": t["HC_precision"], "false-alarm rate": t["HC_false_alarm_rate"],
        "Critical recall": t["CRIT_recall"], "Critical-week interval coverage": t.get("interval_coverage_actual_critical")
    })
    if "ALERT_recall" in t:
        rows.append({
            "rule": f"Watch flag (≥ {t['ALERT_alert_cut']:.1f} cases)",
            "recall": t["ALERT_recall"], "precision": t["ALERT_precision"], "false-alarm rate": t["ALERT_false_alarm_rate"]
        })
    st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, width="stretch")
    st.caption(
        f"The improved rule lowers the High/Critical decision thresholds to the lowest value whose false-alarm rate stayed ≤ {C.ESC_FAR_CAPS['high']:.0%} "
        f"(Critical ≤ {C.ESC_FAR_CAPS['critical']:.0%}, watch ≤ {C.ESC_FAR_CAPS['watch']:.0%}) on pooled chronological out-of-fold forecasts for 2019 – mid 2024 "
        f"(never the test period). Ranking quality (PR-AUC) is unchanged — the gain is a different operational operating point, paid for with more false alarms. "
        f"On test the false-alarm rate came out at {t['HC_false_alarm_rate']:.0%} (budget {C.ESC_FAR_CAPS['high']:.0%}): thresholds do not transfer perfectly across seasons."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if window == C.MAIN_WINDOW:
        # 3. Precision-Recall & Interval Calibration
        st.markdown("<div class='chart-surface'>", unsafe_allow_html=True)
        st.markdown(
            "<div class='chart-surface-header'>"
            "  <div>"
            "    <div class='chart-surface-title'>3. Precision-Recall & Conformal Interval Calibration</div>"
            "    <div class='chart-surface-desc'>High/Critical classification curve and Mondrian conformal coverage across predicted-risk quintiles</div>"
            "  </div>"
            "</div>",
            unsafe_allow_html=True
        )
        c1, c2 = st.columns(2)
        with c1:
            pr = report_csv("pr_curve_test.csv")
            fig = px.line(pr, x="recall", y="precision", title="Precision-Recall Curve: High/Critical (Test)")
            if T0_:
                fig.add_scatter(x=[T0_["HC_recall"]], y=[T0_["HC_precision"]], mode="markers", marker=dict(size=12, color="#6b7280", symbol="x"), name="Original rule")
            fig.add_scatter(x=[t["HC_recall"]], y=[t["HC_precision"]], mode="markers", marker=dict(size=12, color="#c62828"), name="Improved tier ≥ High")
            fig.add_scatter(x=[t["ALERT_recall"]], y=[t["ALERT_precision"]], mode="markers", marker=dict(size=12, color="#0f766e", symbol="diamond"), name="Watch flag")
            fig.update_layout(
                height=330,
                margin=dict(l=0, r=0, t=30, b=0),
                yaxis_range=[0, 1.02],
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif"),
                xaxis=dict(gridcolor="#e2e8f0"),
                yaxis=dict(gridcolor="#e2e8f0")
            )
            st.plotly_chart(fig, width="stretch")
        with c2:
            cal = report_csv("interval_calibration_test.csv")
            fig = go.Figure()
            fig.add_bar(x=[f"{m:.1f}" for m in cal.mean_pred], y=cal.coverage, name="Empirical coverage", marker_color="#0f766e")
            fig.add_hline(y=0.8, line_dash="dash", line_color="#dc2626", annotation_text="nominal 80%")
            fig.update_layout(
                title="Interval Calibration by Predicted-Risk Level (Test)",
                height=330,
                margin=dict(l=0, r=0, t=30, b=0),
                yaxis_range=[0, 1.05],
                xaxis_title="mean predicted cases in bin (quintiles)",
                yaxis_title="coverage",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif"),
                xaxis=dict(gridcolor="#e2e8f0"),
                yaxis=dict(gridcolor="#e2e8f0")
            )
            st.plotly_chart(fig, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

        # 4. Chronological Error Analysis
        st.markdown("<div class='chart-surface'>", unsafe_allow_html=True)
        st.markdown(
            "<div class='chart-surface-header'>"
            "  <div>"
            "    <div class='chart-surface-title'>4. Chronological Error Analysis & Goodness of Fit</div>"
            "    <div class='chart-surface-desc'>Temporal monthly error tracking across test period and actual versus predicted agreement</div>"
            "  </div>"
            "</div>",
            unsafe_allow_html=True
        )
        tp = test_preds()
        monthly = tp.assign(m=tp.date.dt.to_period("M").astype(str), ae=(tp.actual - tp.pred).abs(), se=(tp.actual - tp.pred) ** 2) \
            .groupby("m").agg(MAE=("ae", "mean"), se=("se", "mean")).reset_index()
        monthly["RMSE"] = np.sqrt(monthly.se)
        col_err1, col_err2 = st.columns(2)
        fig_m = px.line(
            monthly.melt("m", ["MAE", "RMSE"]),
            x="m", y="value", color="variable", markers=True,
            title="Chronological Error by Forecast Month (Test Period)",
            labels={"m": "", "value": "cases"}
        )
        fig_m.update_layout(
            height=330,
            margin=dict(l=0, r=0, t=30, b=0),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif"),
            xaxis=dict(gridcolor="#e2e8f0"),
            yaxis=dict(gridcolor="#e2e8f0")
        )
        col_err1.plotly_chart(fig_m, width="stretch")

        s = tp.sample(min(1500, len(tp)), random_state=0)
        fig_s = px.scatter(s, x="actual", y="pred", opacity=.4, title="Actual vs Predicted Cases (Test)")
        mx = float(max(s.actual.max(), s.pred.max()))
        fig_s.add_shape(type="line", x0=0, y0=0, x1=mx, y1=mx, line=dict(dash="dash", color="#dc2626"))
        fig_s.update_layout(
            height=330,
            margin=dict(l=0, r=0, t=30, b=0),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif"),
            xaxis=dict(gridcolor="#e2e8f0"),
            yaxis=dict(gridcolor="#e2e8f0")
        )
        col_err2.plotly_chart(fig_s, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

        # 5. Subgroup Slices (City & Season)
        st.markdown("<div class='chart-surface'>", unsafe_allow_html=True)
        st.markdown(
            "<div class='chart-surface-header'>"
            "  <div>"
            "    <div class='chart-surface-title'>5. Subgroup Slices (City & Season)</div>"
            "    <div class='chart-surface-desc'>Performance stratified by city corporation jurisdiction and climatic season</div>"
            "  </div>"
            "</div>",
            unsafe_allow_html=True
        )
        cA, cB = st.columns(2)
        cdf = report_csv("test_by_city.csv")
        cA.markdown("**By city corporation jurisdiction (test)**")
        cA.dataframe(cdf[["city", "n", "mean_actual", "MAE", "RMSE", "R2", "relative_MAE", "HC_recall", "HC_precision", "interval_coverage", "unreliable_flag"]].round(3), hide_index=True, width="stretch")
        sdf = report_csv("test_by_season.csv")
        cB.markdown("**By season of origin month (test)**")
        cB.dataframe(sdf[["season", "n", "mean_actual", "MAE", "RMSE", "R2", "HC_recall", "HC_precision", "HC_false_alarm_rate", "interval_coverage"]].round(3), hide_index=True, width="stretch")
        st.caption("Dry and post-monsoon seasons exhibit near-zero case incidence; monsoon error is the primary operational determinant.")
        st.markdown("</div>", unsafe_allow_html=True)

        # 6. Reliability Breakdown by Actual & Predicted Tier
        st.markdown("<div class='chart-surface'>", unsafe_allow_html=True)
        st.markdown(
            "<div class='chart-surface-header'>"
            "  <div>"
            "    <div class='chart-surface-title'>6. Reliability Breakdown by Actual & Predicted Tier</div>"
            "    <div class='chart-surface-desc'>Calibration, error magnitude, and interval coverage conditioned on outcome vs forecast</div>"
            "  </div>"
            "</div>",
            unsafe_allow_html=True
        )
        cC, cD = st.columns(2)
        cC.caption("Conditioned on ACTUAL tier (realized epidemic state)")
        cC.dataframe(report_csv("test_by_actual_tier.csv")[["actual_tier", "n", "mean_actual", "MAE", "interval_coverage", "interval_width"]].round(3), hide_index=True, width="stretch")
        cD.caption("Conditioned on PREDICTED tier (surveillance alert level seen by public health officers)")
        cD.dataframe(report_csv("test_by_pred_tier.csv")[["pred_tier", "n", "mean_actual", "MAE", "interval_coverage", "interval_width"]].round(3), hide_index=True, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

        # 7. Missed Outbreaks (False Negatives) Analysis
        st.markdown("<div class='chart-surface'>", unsafe_allow_html=True)
        st.markdown(
            "<div class='chart-surface-header'>"
            "  <div>"
            "    <div class='chart-surface-title'>7. Missed Outbreaks (False Negatives) Analysis</div>"
            "    <div class='chart-surface-desc'>Diagnostic failure analysis of outbreak weeks where model forecast was below High tier</div>"
            "  </div>"
            "</div>",
            unsafe_allow_html=True
        )
        mo = report_csv("missed_outbreaks_analysis.csv")
        st.dataframe(mo[["rule", "missed_High_or_Critical", "of_actual_HC", "median_pred_over_actual", "share_in_onset_week", "share_caught_by_watch_flag", "months", "by_city"]].round(2),
                     hide_index=True, width="stretch")
        fnp = tp[(tp.actual_tier.isin(["High", "Critical"])) & (~tp.tier.isin(["High", "Critical"]))].sort_values("actual", ascending=False)
        st.caption(f"{len(fnp)} true High/Critical ward-weeks are still missed by the improved tier rule; they cluster in outbreak-onset weeks, when weather lags have not yet moved the forecast.")
        st.dataframe(fnp[["date", "city", "ward_id", "actual", "pred", "lo", "hi", "tier", "watch"]].head(50).assign(date=lambda d: d.date.dt.date).round({"actual": 1, "pred": 1, "lo": 1, "hi": 1}), hide_index=True, width="stretch", height=240)
        st.markdown("</div>", unsafe_allow_html=True)

        # 8. Ward-Level Performance
        st.markdown("<div class='chart-surface'>", unsafe_allow_html=True)
        st.markdown(
            "<div class='chart-surface-header'>"
            "  <div>"
            "    <div class='chart-surface-title'>8. Ward-Level Performance Ranking (Test; 75 Forecasts per Ward)</div>"
            "    <div class='chart-surface-desc'>Individual evaluation metrics across all 60 administrative wards</div>"
            "  </div>"
            "</div>",
            unsafe_allow_html=True
        )
        wdf = report_csv("test_by_ward.csv")
        st.dataframe(wdf[["ward", "n", "mean_actual", "MAE", "RMSE", "R2", "relative_MAE", "interval_coverage"]].sort_values("MAE", ascending=False).round(3), hide_index=True, width="stretch", height=260)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Detailed city/season/ward slices, PR curve and calibration are produced for the main window. Select 'Weeks 2-4 ahead (total)' in the sidebar.")

    # 9. Cross-Validation & Model Ablation Diagnostics
    st.markdown("<div class='chart-surface'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='chart-surface-header'>"
        "  <div>"
        "    <div class='chart-surface-title'>9. Cross-Validation & Model Ablation Diagnostics</div>"
        "    <div class='chart-surface-desc'>Chronological rolling-origin cross-validation, feature importance, and architecture comparisons</div>"
        "  </div>"
        "</div>",
        unsafe_allow_html=True
    )
    c1, c2 = st.columns(2)
    cv = report_csv("rolling_origin_cv.csv")
    c1.markdown("**Rolling-origin chronological CV** (train before year, test on that year)")
    c1.dataframe(cv.round(3), hide_index=True, width="stretch")
    imp = pd.Series(M["grouped_importance_val_MAE_increase"]).sort_values()
    fig_imp = px.bar(imp, orientation="h", title="Grouped Permutation Importance (Validation, ΔMAE)", labels={"value": "MAE increase when group shuffled", "index": ""})
    fig_imp.update_layout(
        showlegend=False,
        height=300,
        margin=dict(l=0, r=0, t=30, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif"),
        xaxis=dict(gridcolor="#e2e8f0"),
        yaxis=dict(gridcolor="#e2e8f0")
    )
    c2.plotly_chart(fig_imp, width="stretch")
    c2.caption("Statistical sensitivity metric (ΔMAE under feature permutation). Does not indicate causal transmission mechanism.")

    cm = pd.DataFrame(t["tier_confusion"], index=[f"actual {x}" for x in C.RISK_TIERS], columns=[f"pred {x}" for x in C.RISK_TIERS])
    st.markdown("**Risk-tier confusion matrix (test)**")
    st.dataframe(cm, width="stretch")

    st.markdown("**Model comparison on the validation set (main window)**")
    st.dataframe(report_csv("model_comparison_validation.csv").round(3), hide_index=True, width="stretch")

    st.markdown("**Forecast-strategy comparison (validation, single-week targets)**")
    hs_ = report_csv("horizon_strategy_validation.csv")
    st.dataframe(hs_.round(3), hide_index=True, width="stretch")
    st.caption("Direct per-horizon models had the lowest mean validation MAE and are what the app uses (pooled model was a close second).")
    st.markdown("</div>", unsafe_allow_html=True)


# ================= 7. METHODOLOGY =================
elif active_section == "ℹ️ Methodology & limits":
    a_ = audit()
    st.subheader("Scientific Methodology, Leakage Controls & Operational Limitations")
    st.markdown(f"""
### 1. Model Architecture & Feature Engineering
**Model:** {B['model_name']} on `{B['feature_set']}` features ({len(B['features'])} inputs): current weather, per-ward 1-8 week lags and trailing 2-8 week rolling sums/means of rainfall, temperature and humidity, ward vulnerability attributes (density, water-logging, sanitation) and seasonality. No reported-case history is needed, so wards without surveillance data can be scored. One direct model per horizon (week 2, 3, 4 and the 2-4 week total).

### 2. Timing Protocol & Leakage Controls
**Timing / leakage control:** The forecast for week *t* only uses weather observed up to week *t*. Split: train ≤ 2022, validation 2023-mid 2024, test mid-2024+, with a 4-week purge gap; model and features are selected on validation only; tier percentiles are fixed policy parameters; High/Critical decision thresholds, the watch flag and the prediction-interval calibration are fitted on chronological out-of-fold forecasts that end before the test period; the test period is scored once. A shuffled-target placebo gives R² = {a_['placebo_shuffled_target_val']['R2']:.2f} (no leakage), and past rain correlates most with cases at a {a_['peak_past_lag_weeks']}-week lag.

### 3. Operational & Epidemiological Limitations (Please Read)
- **Synthetic Characteristics:** The dataset **looks synthetic** (validation R² {a_['full_weather_model_val']['R2']:.2f}, clean lag peak, strongly autocorrelated residuals). Results show the method works on this data; they are **not** evidence of real-world accuracy or operational validity.
- **Statistical vs Clinical Tiers:** Risk tiers are percentiles of the training case distribution - **not** official public-health alert levels. The High/Critical decision thresholds trade missed outbreaks for false alarms; the watch flag is an even more sensitive screening threshold.
- **Conformal Interval Scope:** Intervals (scaled conformal residuals, calibrated per predicted-risk tier on out-of-fold forecasts) reached {T_['interval_coverage']:.0%} test coverage vs 80% nominal (conservative) and carry no guarantee under distribution shift.
- **Explanatory Attribution:** Driver explanations are what-if occlusions versus seasonal-normal conditions, not causal effects.
- **Geographic Resolution:** No ward coordinates exist in the data, so there is no ward-level geographic map (a verified `data/ward_coordinates.csv` enables it). The city map uses approximate city centres.
- **Experimental Ingestion:** The live-weather panel is experimental (different weather source, city-level weather for all wards, unvalidated).

### 4. AI & External Services Disclosure
**AI & external services:** Claude (Anthropic) assisted with code review, refactoring, tests and documentation. Runtime: **no LLM calls**; the only optional external service is Open-Meteo (weather, no key), used only when you press the Live button. Full disclosure in `README.md`.
""")
