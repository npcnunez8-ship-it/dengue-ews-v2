"""OPTIONAL live-weather ingestion (Open-Meteo, free, no API key). The app works fully offline without it.

Design goals: minimal calls, safe failure, explicit provenance.
  * ONE HTTP request per city per cache window (default 6 h) covering the last ~10 weeks of daily data
    (`past_days`), cached on disk as JSON -> 5 cities = <= 5 requests per refresh, ~20/day worst case.
  * Never raises into the UI: returns a result object with `ok`, `message`, `source`, `freshness`.
  * Every value carries a provenance label: past days = "observed/analysis" (model reanalysis, NOT station
    measurements); days after today = "forecast". Only COMPLETE past weeks are fed to the lag features;
    forecast values are never mixed into lag inputs.
  * Config via environment variables: OPEN_METEO_BASE_URL, OPEN_METEO_API_KEY (optional, commercial plan),
    LIVE_WEATHER_CACHE_HOURS, LIVE_WEATHER_TIMEOUT_S, LIVE_WEATHER_OFFLINE=1 (never touch the network).
The weekly aggregation (rain = weekly SUM, temperature/humidity = weekly MEAN, weeks start on Monday) mirrors
how the supplied weekly dataset is used by the model; the dataset does not document its own aggregation, so
this is an ASSUMPTION that is stated in the dashboard.
"""
import hashlib
import json
import os
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import config as C
from .quality import MAX_STALE_DAYS, freshness, repair_history

DAILY_VARS = "temperature_2m_mean,precipitation_sum,relative_humidity_2m_mean"
PAST_DAYS = 70    # 10 weeks: enough for 9 complete weeks even when the current week is partial


def _cfg():
    return {
        "base": os.environ.get("OPEN_METEO_BASE_URL", "https://api.open-meteo.com/v1/forecast"),
        "key": os.environ.get("OPEN_METEO_API_KEY", ""),
        "ttl_h": float(os.environ.get("LIVE_WEATHER_CACHE_HOURS", "6")),
        "timeout": float(os.environ.get("LIVE_WEATHER_TIMEOUT_S", "8")),
        "offline": os.environ.get("LIVE_WEATHER_OFFLINE", "0") == "1",
        "cooldown_min": float(os.environ.get("LIVE_WEATHER_COOLDOWN_MIN", "10")),   # no retry for N min after a failure / HTTP 429
    }


@dataclass
class LiveResult:
    ok: bool
    message: str
    daily: pd.DataFrame | None = None        # date, temp, rain, humidity, kind in {"observed", "forecast"}
    weekly: pd.DataFrame | None = None       # complete weeks only (Mon-Sun, all 7 days observed)
    source: str = "none"                     # "network" | "cache" | "stale-cache" | "none"
    fetched_at: str | None = None
    freshness: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)


def _cache_path(lat, lon, cache_dir=None):
    d = Path(cache_dir or C.LIVE_CACHE_DIR)
    d.mkdir(parents=True, exist_ok=True)
    return d / (hashlib.md5(f"{lat:.2f},{lon:.2f}".encode()).hexdigest()[:12] + ".json")


def _request(lat, lon, cfg):
    q = {"latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}", "daily": DAILY_VARS, "past_days": PAST_DAYS,
         "forecast_days": 1, "timezone": "Asia/Dhaka"}
    if cfg["key"]:
        q["apikey"] = cfg["key"]
    url = cfg["base"] + "?" + urllib.parse.urlencode(q)
    with urllib.request.urlopen(url, timeout=cfg["timeout"]) as r:       # noqa: S310 (fixed https endpoint)
        return json.loads(r.read().decode())


def _parse(payload: dict, today: pd.Timestamp) -> pd.DataFrame:
    d = payload["daily"]
    out = pd.DataFrame({"date": pd.to_datetime(d["time"]), "avg_temp_c": d["temperature_2m_mean"],
                        "rainfall_mm": d["precipitation_sum"], "relative_humidity_pct": d["relative_humidity_2m_mean"]})
    out["kind"] = ["observed" if t < today else "forecast" for t in out["date"]]   # today & later are not complete days
    return out


def to_weekly(daily: pd.DataFrame, today=None) -> pd.DataFrame:
    """Aggregate to Monday-start weeks; keep only weeks whose 7 days are all complete observed days."""
    today = pd.Timestamp.now().normalize() if today is None else pd.Timestamp(today).normalize()
    obs = daily[(daily.kind == "observed") & (daily.date < today)].copy()
    obs["week"] = obs.date - pd.to_timedelta(obs.date.dt.day_of_week, unit="D")
    g = obs.groupby("week")
    wk = pd.DataFrame({"avg_temp_c": g.avg_temp_c.mean(), "rainfall_mm": g.rainfall_mm.sum(min_count=7),
                       "relative_humidity_pct": g.relative_humidity_pct.mean(), "n_days": g.size(),
                       "n_valid": g.apply(lambda x: int(x[["avg_temp_c", "rainfall_mm", "relative_humidity_pct"]].notna().all(axis=1).sum()),
                                          include_groups=False)}).reset_index().rename(columns={"week": "date"})
    return wk[(wk.n_days == 7) & (wk.n_valid >= 6)].reset_index(drop=True)      # tolerate one missing day, not more


def fetch_city_weather(city: str, now=None, cache_dir=None, fetch=None) -> LiveResult:
    """Return recent weekly weather for a city. `fetch` is injectable for tests."""
    cfg = _cfg()
    lat, lon = C.CITY_CENTRES[city]
    now = pd.Timestamp.now() if now is None else pd.Timestamp(now)
    today = now.normalize()
    cp = _cache_path(lat, lon, cache_dir)
    cached = None
    if cp.exists():
        try:
            cached = json.loads(cp.read_text())
        except Exception:  # noqa: BLE001
            cached = None
    age_h = (time.time() - cp.stat().st_mtime) / 3600 if cached else None
    if cached and age_h is not None and age_h <= cfg["ttl_h"]:
        return _build(cached, "cache", today, cp, notes=[f"served from cache ({age_h:.1f} h old, ttl {cfg['ttl_h']:g} h): no API call"])
    if cfg["offline"]:
        return _stale_or_fail(cached, today, cp, "LIVE_WEATHER_OFFLINE=1 (network disabled)")
    fail = cp.with_suffix(".fail")
    if fail.exists() and (time.time() - fail.stat().st_mtime) / 60 < cfg["cooldown_min"]:      # back off after timeouts / rate limits
        return _stale_or_fail(cached, today, cp, f"cooling down {cfg['cooldown_min']:g} min after the previous failed request ({fail.read_text()[:80]})")
    try:
        payload = (fetch or _request)(lat, lon, cfg)
        if "daily" not in payload:
            raise ValueError(payload.get("reason", "response without 'daily' block"))
        cp.write_text(json.dumps(payload))
        fail.unlink(missing_ok=True)
        return _build(payload, "network", today, cp, notes=["1 API request made"])
    except Exception as e:  # noqa: BLE001 (never propagate network problems into the UI)
        reason = f"{type(e).__name__}: {e}"
        try:
            fail.write_text(reason)                       # timeouts, HTTP 429 (rate limit) etc. -> cooldown
        except OSError:
            pass
        return _stale_or_fail(cached, today, cp, reason)


def _build(payload, source, today, cp, notes):
    try:
        daily = _parse(payload, today)
        weekly = to_weekly(daily, today)
    except Exception as e:  # noqa: BLE001
        return LiveResult(False, f"Could not parse weather payload ({e})", source="none")
    if weekly.empty:
        return LiveResult(False, "No complete weeks in the response.", daily=daily, source=source)
    latest_day = daily.loc[daily.kind == "observed", "date"].max()
    fr = freshness(latest_day, today)
    fetched = pd.Timestamp(cp.stat().st_mtime, unit="s").isoformat(timespec="minutes") if cp.exists() else None
    if fr["stale"]:
        notes.append(f"STALE: latest observed day is {fr['age_days']} days old (limit {MAX_STALE_DAYS}).")
    return LiveResult(True, "ok", daily, weekly, source, fetched, fr, notes)


def _stale_or_fail(cached, today, cp, reason):
    if cached:
        r = _build(cached, "stale-cache", today, cp, notes=[f"live request unavailable ({reason}); using last cached response"])
        r.freshness["stale"] = True
        return r
    return LiveResult(False, f"Live weather unavailable and no cache: {reason}. The app keeps working with the bundled dataset.")


def history_for_scoring(res: LiveResult, ward_row: pd.Series, n_weeks: int = 9):
    """Build the 9-week raw history (+ ward attributes) expected by src.scenario from complete observed weeks.
    City-level weather is applied to the ward (the dataset has no ward coordinates) - stated in the UI."""
    wk = res.weekly.tail(n_weeks).copy()
    wk, notes = repair_history(wk.drop(columns=["n_days", "n_valid"]))
    for c in ("city", "ward_id", *C.STATIC_COLS):
        wk[c] = ward_row[c]
    return wk, notes
