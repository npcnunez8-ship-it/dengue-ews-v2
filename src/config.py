"""Central configuration (single source of truth for paths, splits, tiers)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "bangladesh_dengue_weekly.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

# Chronological split (weekly data, Mondays). A purge gap equal to the
# longest forecast horizon keeps target windows from crossing boundaries.
TRAIN_END = "2022-12-31"   # train: forecast dates <= TRAIN_END - gap
VAL_END = "2024-06-30"     # val: (TRAIN_END, VAL_END - gap]; test: > VAL_END
MAX_HORIZON = 4

# Forecast windows = which weeks-ahead are summed into the target.
WINDOWS = {
    "Weeks 2-4 ahead (total)": (2, 3, 4),
    "Week 2 ahead": (2,),
    "Week 3 ahead": (3,),
    "Week 4 ahead": (4,),
}
MAIN_WINDOW = "Weeks 2-4 ahead (total)"

RISK_TIERS = ["Low", "Medium", "High", "Critical"]
# Tier definitions are FIXED policy parameters (not tuned): Medium / High / Critical start at the 70th / 90th / 97th
# percentile of the TRAINING-period target of the same forecast window.  (An earlier version chose these quantiles by
# validation kappa; that choice proved unstable - numerically negligible float differences (<1e-12) flipped it - so
# it is now only reported as a sensitivity table.)  The numeric cut-points are fit from training data only (train.py).
TIER_QUANTILES = (0.70, 0.90, 0.97)

WEATHER_COLS = ["avg_temp_c", "rainfall_mm", "relative_humidity_pct"]
STATIC_COLS = ["population_density_sqkm", "water_logging_risk_index", "sanitation_score"]
LAGS = list(range(1, 9))       # weekly lags (covers 2-4 week breeding clock + margin)
ROLL_WINDOWS = [2, 3, 4, 6, 8]

# ---- Operational early-warning flag (sensitive operating point, chosen on VALIDATION only) ----
ALERT_MAX_FALSE_ALARM = 0.10   # cap on validation false-alarm rate when choosing the alert threshold
ALERT_BETA = 2.0               # F-beta with beta=2 weights recall twice as much as precision
ALERT_FACTORS = [round(x, 2) for x in [0.40 + 0.05 * i for i in range(13)]]   # alert cut = factor x High cut

# ---- Geography ----
# The dataset has NO ward coordinates and the ward ids (e.g. DNCC_Ward_01) are not mapped to real,
# verifiable ward polygons, so we never fabricate ward positions.  Two honest layers instead:
#  1) APPROXIMATE city-centre coordinates (+-5 km, public geographic knowledge) used ONLY for the
#     city-level map and as the weather-lookup point of the optional live-weather layer.
#  2) OPTIONAL data/ward_coordinates.csv (columns ward_id,lat,lon) -> real ward-level heatmap as soon
#     as verified coordinates are supplied.  The file is intentionally NOT shipped.
CITY_CENTRES = {
    "Dhaka North": (23.82, 90.39),
    "Dhaka South": (23.71, 90.41),
    "Chattogram": (22.36, 91.78),
    "Gazipur": (23.99, 90.42),
    "Narayanganj": (23.62, 90.50),
}
WARD_COORD_PATH = ROOT / "data" / "ward_coordinates.csv"
LIVE_CACHE_DIR = ROOT / ".cache" / "live_weather"

# ---- Risk-escalation + interval calibration (chronological, pre-test only) ----
# Expanding-window out-of-fold (OOF) forecasts: each fold is predicted by a model trained only on data that
# ends >= MAX_HORIZON weeks before the fold starts.  The last fold stops at VAL_END - purge gap, so the
# final TEST period (> VAL_END) is never used for any threshold, interval or model-selection decision.
OOF_FOLDS = [("2019-01-01", "2019-12-31"), ("2020-01-01", "2020-12-31"), ("2021-01-01", "2021-12-31"),
             ("2022-01-01", "2022-12-31"), ("2023-01-01", "2023-12-31"), ("2024-01-01", None)]   # None -> VAL_END - gap
# Operational false-alarm budgets (share of NOT-outbreak ward-weeks that may be flagged), enforced on pooled OOF:
ESC_FAR_CAPS = {"watch": 0.10, "high": 0.05, "critical": 0.02}
ESC_FACTORS = [round(0.40 + 0.025 * i, 3) for i in range(25)]    # decision threshold = factor x tier cut-point (<= 1.0)
INTERVAL_MIN_GROUP = 100       # Mondrian interval groups smaller than this are merged with the group below
