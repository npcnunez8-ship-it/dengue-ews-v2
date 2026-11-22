"""Geographic layers - only from verified inputs.

* ward level: requires data/ward_coordinates.csv (ward_id, lat, lon) supplied by the team from a verified source.
  Missing file -> `ward_coordinates()` returns None and the dashboard says so; nothing is invented.
* city level: approximate city-centre coordinates from config (display only).
"""
import pandas as pd

from . import config as C


def ward_coordinates() -> pd.DataFrame | None:
    p = C.WARD_COORD_PATH
    if not p.exists():
        return None
    d = pd.read_csv(p)
    if not {"ward_id", "lat", "lon"} <= set(d.columns):
        raise ValueError("ward_coordinates.csv must have columns ward_id,lat,lon")
    bad = d[~(d.lat.between(20.5, 26.7) & d.lon.between(88.0, 92.7))]      # Bangladesh bounding box
    if len(bad):
        raise ValueError(f"{len(bad)} coordinate(s) fall outside Bangladesh: {bad.ward_id.tolist()[:5]}")
    return d[["ward_id", "lat", "lon"]].drop_duplicates("ward_id")


def city_summary(snap: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a ward snapshot (date, city, ward_id, pred, tier, alert) to one row per city with coordinates."""
    g = snap.groupby("city")
    out = pd.DataFrame({"wards": g.ward_id.nunique(), "total_pred_cases": g.pred.sum(), "mean_pred_cases": g.pred.mean(),
                        "high_critical_wards": g.tier.apply(lambda s: int(s.isin(["High", "Critical"]).sum())),
                        "watch_flag_wards": g.alert.sum().astype(int)}).reset_index()
    out["lat"] = out.city.map(lambda c: C.CITY_CENTRES[c][0])
    out["lon"] = out.city.map(lambda c: C.CITY_CENTRES[c][1])
    out["share_high_critical"] = out.high_critical_wards / out.wards
    return out
