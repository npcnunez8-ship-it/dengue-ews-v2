# 🦟 Dengue Early-Warning System (Dhaka North/South · Chattogram · Gazipur · Narayanganj)

Ward-level dengue outbreak risk **2-4 weeks ahead** from multi-week weather lags, with Low / Medium / High / Critical tiers, a watch flag, 80% prediction intervals,
driver explanations, a what-if simulator and a chronological-validation dashboard (EDU AI Engineering Hackathon 2026, "AI for Urban Dengue Outbreak").
Runs **fully offline**: no API key, no LLM calls. An optional live-weather panel (Open-Meteo) exists but is never required.

> **Read this first.** The supplied dataset looks **synthetic** (see [Limitations](#limitations-and-honesty-notes)). All metrics below demonstrate that the *method* works on this
> data; they are **not** evidence of real-world dengue-forecasting accuracy or of operational validity.

## Quick start
```bash
git clone <YOUR_REPO_URL> && cd <repo-folder>
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt      # or: pip install -r requirements-lock.txt  (exact tested versions)
python -m src.selfcheck              # optional, ~3 s: data integrity, lag alignment, model load, inference, reports
streamlit run app.py                 # opens http://localhost:8501
```
The trained model (`models/dengue_bundle.joblib`) is committed and locked to the canonical `.venv` (`scikit-learn==1.9.1`).

> **Environment Governance Rule:** Always use the canonical `.venv` environment (`scikit-learn 1.9.1`). Global Python or unpinned environments must not be used. Automatic retraining of `models/dengue_bundle.joblib` is disabled by default to protect canonical model baselines. If rebuilding is genuinely required, explicit opt-in via `ALLOW_BUNDLE_REBUILD=1` is required.

**Windows PowerShell Canonical Commands:**
```powershell
# 1. Pipeline and data self-check
.\.venv\Scripts\python.exe -m src.selfcheck

# 2. Run full test suite (54 unit tests)
.\.venv\Scripts\python.exe -m pytest -q tests

# 3. Launch Streamlit dashboard
.\.venv\Scripts\python.exe -m streamlit run app.py

# 4. Explicit Model Bundle Rebuild (ONLY when intentionally retrained)
$env:ALLOW_BUNDLE_REBUILD="1"
.\.venv\Scripts\python.exe -m src.train --bundle-only
```

**Verified environments**:
* `requirements-lock.txt` — **Canonical Baseline**: Python 3.12, numpy 2.5.5, pandas 3.1.1, scikit-learn 1.9.1, streamlit 1.67.0. All canonical predictions and reported metrics evaluate against this runtime.
* Historical Reference (Archival Phase 3): Python 3.12, numpy 1.26.4, pandas 2.2.2, scikit-learn 1.5.0.

## Canonical Prediction Baselines (`.venv` · scikit-learn 1.9.1)
Primary horizon (Weeks 2–4 ahead total):
* `DNCC_Ward_01` (2025-08-04): Pred **28.42** | Tier **High** | Alert **True** | Interval `[15.42, 48.09]`
* `DSCC_Ward_04` (2025-08-04): Pred **35.64** | Tier **Critical** | Alert **True** | Interval `[21.13, 57.58]`
* `CCC_Ward_03`  (2025-08-04): Pred **37.21** | Tier **Critical** | Alert **True** | Interval `[22.40, 59.62]`
* `DNCC_Ward_01` (2025-02-03): Pred **0.07** | Tier **Low** | Alert **False** | Interval `[0.00, 1.63]`
* `DSCC_Ward_04` (2025-02-03): Pred **0.04** | Tier **Low** | Alert **False** | Interval `[0.00, 1.58]`
* `CCC_Ward_03`  (2025-02-03): Pred **0.08** | Tier **Low** | Alert **False** | Interval `[0.00, 1.64]`

*(Archival historical reference values under scikit-learn 1.5.0: 27.70, 34.94, 36.41, 0.08, 0.08, 0.08).*

## Commands
| Task | Canonical Command (Windows PowerShell) | Standard POSIX Command |
|---|---|---|
| Launch dashboard | `.\.venv\Scripts\python.exe -m streamlit run app.py` | `streamlit run app.py` |
| Self-check (environment + pipeline) | `.\.venv\Scripts\python.exe -m src.selfcheck` | `python -m src.selfcheck` |
| Run tests (54) | `.\.venv\Scripts\python.exe -m pytest -q tests` | `python -m pytest -q tests` |
| Prove tests not vacuous | `.\.venv\Scripts\python.exe tools/mutation_check.py` | `python tools/mutation_check.py` |
| Full retrain + all reports (~3.5 min) | `$env:ALLOW_BUNDLE_REBUILD="1"; .\.venv\Scripts\python.exe -m src.train` | `ALLOW_BUNDLE_REBUILD=1 python -m src.train` |
| Rebuild only model bundle (~75 s) | `$env:ALLOW_BUNDLE_REBUILD="1"; .\.venv\Scripts\python.exe -m src.train --bundle-only` | `ALLOW_BUNDLE_REBUILD=1 python -m src.train --bundle-only` |
| Leakage audit (~40 s) | `.\.venv\Scripts\python.exe -m src.audit` | `python -m src.audit` |
| Offline guarantee for live panel | `$env:LIVE_WEATHER_OFFLINE="1"; .\.venv\Scripts\python.exe -m streamlit run app.py` | `LIVE_WEATHER_OFFLINE=1 streamlit run app.py` |

Optional environment variables (live panel only): `OPEN_METEO_BASE_URL`, `OPEN_METEO_API_KEY`, `LIVE_WEATHER_CACHE_HOURS` (6), `LIVE_WEATHER_TIMEOUT_S` (8), `LIVE_WEATHER_COOLDOWN_MIN` (10), `LIVE_WEATHER_OFFLINE`.

## Dashboard (7 tabs)
1. **Overview** - banner with the number of Critical / High wards, watch-only wards, top-10 priority wards, tier mix by city.
2. **Ward forecast** - separate **week 2 / 3 / 4** and 2-4 week-total forecasts with 80% intervals and tiers, driver breakdown, observed weather, historical vs predicted trend, live *lag-pipeline check*.
3. **Risk maps & ranking** - ward ranking, ward×week risk heatmap, city-level map (ward-level map only if verified coordinates are supplied).
4. **What-if** - edit the last 9 weeks of weather and re-score through the same lag code and model.
5. **Live weather (optional)** - recent observed weather for a city (1 cached request, cooldown after failures, stale-cache fallback).
6. **Validation** - original-vs-improved High/Critical detection, PR curve, interval calibration by risk tier, **false-negative panel**, error by month / city / season / ward / actual tier, tier confusion, rolling-origin CV, model comparison.
7. **Methodology & limits.**

## Project layout
```
app.py                  Streamlit dashboard
src/features.py         calendar-aligned weather lags 1-8w / rolling 2-8w / targets (value_at_offset); duplicate ward-weeks rejected
src/train.py            chronological split, baselines, model comparison, per-window calibration, test scoring, original-vs-improved reports
src/calibration.py      chronological out-of-fold forecasts -> High/Critical/watch decision thresholds + per-tier prediction intervals
src/model_io.py         portable bundle (no pandas objects), version check, automatic rebuild
src/predict.py          fast batch inference + grouped driver explanations
src/scenario.py         what-if scoring from a raw 9-week weather history (same features + bundle)
src/quality.py          missing / unusual / stale input handling, dataset quality report
src/live_weather.py     optional Open-Meteo ingestion: cache, cooldown, freshness, provenance labels
src/geo.py              city summary + optional verified ward coordinates (nothing invented)
src/evaluate.py         city / ward / season / actual-tier / predicted-tier slices, High-Critical + Critical metrics incl. PR-AUC, interval coverage
src/audit.py            leakage audit, horizon-strategy comparison, chronological hyper-parameter check
src/selfcheck.py        fast environment + pipeline check
tests/                  54 tests (alignment with gaps/duplicates, future leakage, interval containment, escalation caps, test-set independence, bundle, live weather, docs consistency)
tools/mutation_check.py re-introduces 10 known bugs into a copy of the project and verifies the test-suite fails
notebooks/              executed notebook (outputs regenerated)      legacy/  the team's original notebook (untouched archive)
models/ reports/        serialized model, metrics.json and CSVs; reports/baseline_shipped_zip = results shipped in the uploaded ZIP, kept for comparison
```

## Methodology in brief
- **Weather lag pipeline:** `avg_temp_c`, `rainfall_mm`, `relative_humidity_pct` → current value, per-`ward_id` lags 1-8 **calendar** weeks, trailing 2/3/4/6/8-week rolling sums (rain) or means (temp, humidity), week-of-year sine/cosine, plus ward attributes (density, water-logging, sanitation): 47 features. A missing week gives NaN features for every row that depends on it - never a value from another week.
- **Targets:** reported cases in weeks t+2…t+4 (total) and each of weeks 2, 3, 4 separately (one direct model each).
- **Model:** HistGradientBoosting with Poisson loss, selected on chronological validation against Ridge, RandomForest, ExtraTrees, HistGB-squared (and XGBoost if installed).
- **Validation:** train ≤ 2022-11, validation 2023-01…2024-05, test 2024-07…2025-12, 4-week purge gap, identical rows for every model and baseline, test scored once.
- **Risk tiers** (fixed policy: Medium / High / Critical start at the 70th / 90th / 97th percentile of the training-period window total → 10 / 27 / 41 cases for weeks 2-4) with **decision thresholds** for High / Critical / watch chosen on pooled chronological out-of-fold forecasts under stated false-alarm budgets; **intervals** = scaled conformal residuals calibrated per predicted-risk tier.

## Results (out-of-sample test, forecasts 2024-07 → 2025-12, 4,500 ward-weeks, 60 wards, 5 cities)
Window = cases in weeks 2-4 ahead. Every method is scored on identical rows. "Original" = the rule shipped in the uploaded ZIP, re-run with the same code/data (same model, same point forecasts); "improved" = this upgrade.

| Model | MAE | RMSE | R² | High/Critical recall |
|---|---|---|---|---|
| **HistGB-Poisson, weather-only (ours)** | **2.64** | **4.92** | **0.81** | 71% original rule → 91% improved rule |
| Seasonal-naive (same window 52 weeks earlier) | 4.03 | 8.12 | 0.49 | 66% |
| Persistence (current cases × 3) | 6.22 | 10.75 | 0.11 | 33% |
| Mean baseline | 9.57 | 11.38 | 0.00 | 0% |

**Original vs improved risk detection (same model, same test rows; 432 actual High/Critical ward-weeks, 53 actual Critical):**

| Metric | Original | Improved |
|---|---|---|
| High/Critical decision threshold (cases in 3 weeks) | ≥ 27 | ≥ 22.3 |
| Missed High/Critical ward-weeks (false negatives) | 127 | 40 |
| High/Critical recall | 71% | 91% |
| High/Critical precision | 55% | 44% |
| False-alarm rate (non-outbreak ward-weeks flagged) | 6.2% | 12.3% |
| Share of all ward-weeks flagged High/Critical (true prevalence 9.6%) | 12.4% | 19.8% |
| Critical recall (missed / 53) | 42% (31) | 72% (15) |
| Critical precision | 24% | 14% |
| 80% interval coverage on weeks that turned out Critical | 47% | 94% |
| 80% interval coverage overall / mean width | 95% / 13.3 | 91% / 11.9 |
| High/Critical PR-AUC (ranking quality) | 0.66 | 0.66 (unchanged) |
| Tier exact accuracy | 86% | 81% |

**How to read this honestly.** The model and its ranking quality are unchanged (identical MAE, RMSE, PR-AUC). The improvement is a **different operating point**: a lower, cost-sensitive decision threshold chosen on
chronological out-of-fold forecasts (2019 → mid-2024) with a false-alarm budget of 5% (High) / 2% (Critical); the test period played no part in choosing it. It finds 87 more of the 432 true High/Critical ward-weeks and
costs 249 additional false alarms (about 2.9 extra false alarms per extra outbreak week found). **The 5% false-alarm budget was not preserved on the test period (12.3%)**: test-year predictions were
relatively higher than the realised outbreaks (monsoon actual/predicted case ratio 0.94 in 2024 and 0.82 in 2025, versus roughly 0.9-1.3 in the 2019-2023 out-of-fold seasons), i.e. thresholds do not transfer perfectly between seasons. A nested check (thresholds chosen on 2019-2022 folds, scored on 2023 - mid-2024) held its budget (High: 2.1% false alarms, recall 87%), so the shift is a property of the test seasons, not a bug.
Critical alerts are noisy (38 true vs 242 false Critical flags on test). 40 true High/Critical ward-weeks are still missed; 65% of them occur in the first week of an outbreak, before weather lags have moved the forecast (`reports/missed_outbreaks_analysis.csv`).
The original 127 misses all lie in the monsoon (all 432 actual High/Critical weeks do); 95% of them (121) would have been caught by the existing sensitive watch flag (predicted ≥ 17.6 cases; test recall 99%, false-alarm rate 20%).

Other findings (all in `reports/`): single-week models (weeks 2 / 3 / 4): High/Critical recall 49% / 47% / 43% (original) → 78% / 79% / 72% (improved), R² 0.71 / 0.71 / 0.63; monsoon MAE 5.5 (R² 0.52); rolling-origin CV R² 0.51-0.91 (the 0.51 fold is Jan-May 2024, almost no cases); other High/Critical scorers on out-of-fold data (classifier, 70th-percentile quantile model) rank outbreaks about the same (PR-AUC 0.82 / 0.83 / 0.84; recall at 5% false alarms 0.81 / 0.80 / 0.82) and were not adopted under the pre-set rule (≥ +3 points).
Ablations: ward attributes + season alone → validation MAE 2.56; adding lagged weather → 1.86; case history did not help; the hyper-parameter search gained 1.3% (not adopted, bar 2%).

## Limitations and honesty notes
- **Synthetic-looking data.** Validation R² 0.91, a smooth lag-3 correlation peak and strongly autocorrelated residuals (0.55) are typical of simulated data. Provenance of the CSV is not documented. Results show the method works here; they are not medically or operationally validated.
- **Tiers are dataset-relative percentiles, not official public-health alert levels.** The decision thresholds trade missed outbreaks against false alarms; any real deployment must set that trade-off with public-health officers.
- **Weather in the dataset is ward-level and historical** (each ward has its own weekly weather). The optional live panel uses **city-level** weather for every ward and a different source and aggregation (the CSV does not document how weekly values were built; we assume weekly sum for rain, weekly mean for temperature/humidity, Monday-start). It was **not tested against the real Open-Meteo API** (no network access during the upgrade); it is unit-tested with mocked responses. Forecast days are never used as lag inputs.
- **What the application predicts:** expected reported cases in weeks t+2…t+4 for a ward, given weather observed through week t and the ward's static attributes, plus a risk tier. **What it cannot establish:** causal effects of rain, real-world accuracy, forecasts from *forecast* weather, ward geography, or case counts beyond what the supplied surveillance numbers represent.
- Results are sensitive to numerically negligible changes (feature differences < 1e-12 moved validation MAE by 1.7%), so re-running on another machine may change last digits; the reported numbers are the committed ones.
- **No ward-level geographic heatmap:** the data have no ward coordinates; the city-level map uses approximate centres; add `data/ward_coordinates.csv` (`ward_id,lat,lon`) for a ward map.
- The dashboard is a historical backtest: no live case feed. It was exercised through Streamlit's `AppTest` harness and a running server health check; **it was not viewed in a real browser** during this upgrade.
- Not run: installation from a public GitHub clone (no repository exists in this environment), Python versions other than 3.12.

## AI assistants & external services disclosure
- **AI assistants used:** Claude (Anthropic) - reviewed the original notebook; wrote/refactored the code in `src/`, the Streamlit app, tests, notebook cells and documentation across several working sessions, including this upgrade (calendar-aligned feature pipeline, out-of-fold threshold and interval calibration, regression tests, mutation check, documentation). All reported numbers were produced by running the code. *(Team: add any other assistant you actually used - ChatGPT, Gemini, Copilot, etc.)*
- **External APIs at runtime:** none by default. Optional: **Open-Meteo** (free, no key, CC BY 4.0, attribution required) is called only when a user presses *Fetch latest weather* in the Live tab; one request per city, cached 6 h, 10-minute cooldown after a failure or rate limit. Disable with `LIVE_WEATHER_OFFLINE=1`.
- **LLM calls at runtime:** none. **Data:** `data/bangladesh_dengue_weekly.csv` (hackathon-provided; byte-identical to the supplied CSV export).
- **Code/ideas from elsewhere:** no code copied. Ideas consulted: DengAI competition solutions (rolling weather windows), Mosqlimate / Johansson et al. PNAS 2019 (probabilistic arbovirus forecasting, coverage and baselines), conformal prediction (Romano et al. 2019; Mondrian conformal). *(Team: list any repo/tutorial code you copy, with URL and licence.)*
- **Libraries:** numpy, pandas, scikit-learn, joblib, pyarrow, streamlit, plotly, pytest (+ jupyter, matplotlib, seaborn for the notebook only).

## Team
*(Add team member names, roles and GitHub handles here. The rubric's Git-history criterion needs distributed commits - not something this ZIP can provide.)*
