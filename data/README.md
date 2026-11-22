# data/
* `bangladesh_dengue_weekly.csv` - hackathon-provided weekly ward dataset (60 wards in 5 cities, 522 Monday-start weeks each, 2016-01-04 → 2025-12-29, 31,320 rows; byte-identical to the supplied CSV export).
  Columns: `date, city, ward_id, population_density_sqkm, water_logging_risk_index, sanitation_score, avg_temp_c, max_temp_c, min_temp_c, rainfall_mm, relative_humidity_pct, dengue_cases`.
  * **Provenance is undocumented and the data look synthetic** (very high predictability, clean lag-3 rain correlation, strongly autocorrelated residuals). Treat every result as a demonstration of the method.
  * **Weather is ward-level and historical**: each ward has its own weekly weather in this file. (The optional live API layer is city-level only.)
  * Integrity checks (`python -m src.selfcheck`): no duplicate `(ward_id, date)` rows, no calendar gaps, no missing values. The loader rejects duplicates; lags/rolls/targets are aligned by calendar date, so any gap or deleted week would produce NaN rather than a misaligned value.
  * The dataset does not say how weekly values were aggregated (sum vs mean) - the live-weather layer assumes weekly sum for rain and weekly mean for temperature/humidity.
* `ward_coordinates.csv` - **optional, not shipped.** Columns `ward_id,lat,lon`, from a verified source (official ward boundaries/centroids).
  When present, the dashboard's map tab shows a ward-level geographic risk map. Coordinates outside Bangladesh are rejected. We do not invent ward locations or boundaries.
