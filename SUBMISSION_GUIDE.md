# Step-by-step submission guide

## A. Run it locally
1. Install Python 3.10+ and Git. Unzip the project; open a terminal in the folder.
2. `python -m venv .venv` → activate (`source .venv/bin/activate` or `.venv\Scripts\activate`) → `pip install -r requirements.txt` (exact tested versions: `requirements-lock.txt`). Optional: `python -m src.selfcheck` (≈3 s).
3. **Dashboard:** `streamlit run app.py` (browser opens at http://localhost:8501).
4. **Tests:** `python -m pytest -q tests` (expect 54 passed).
5. **Notebook:** `pip install -r requirements-notebook.txt`, then `jupyter notebook notebooks/dengue_ews_pipeline.ipynb` → Kernel → Restart & Run All (~1 min; it reads the committed reports; set `RETRAIN = True` in section 5 to retrain, ~4 min, which rewrites `models/` and `reports/`).

## B. Try different wards
Sidebar: choose a window, city, ward, and date. Use dates **after 2024-07** (green "out-of-sample" box) for honest forecasts.
Good demos: `DNCC_Ward_01` / `DSCC_Ward_04` / `CCC_Ward_03` on **2025-08-04** (High/Critical, monsoon) versus any ward in **2025-02-03** (Low).
Check the other tabs: Overview (priority wards), Risk maps & ranking (ward×week heatmap), What-if, Validation (original-vs-improved High/Critical detection, false-negative panel, MAE/RMSE by month, PR curve, interval calibration), Methodology & limits.

## C. GitHub + genuine distributed commits (10 pts)
1. Each member creates/uses a GitHub account; one member creates a **public** repo (no README/gitignore), adds the others as collaborators.
2. Everyone: `git clone <url>`. First person pushes the project: `git add . && git commit -m "Initial project" && git push`.
3. Split real work so everyone commits at different hours through the window (ends **Nov 22, 2:00 PM BST**), e.g.:
   - Member A: README/AI-disclosure team section, requirements check on a clean machine.
   - Member B: review/tweak dashboard text, tier advice wording, screenshots.
   - Member C: run tests/notebook, add any extra analysis, fix issues found.
   Commit small and often: `git pull` → edit → `git add -A` → `git commit -m "what you did"` → `git push`.
   Do **not** rewrite history or fake authorship; the judges check that all members contributed over the 24 hours.
4. Fill in the team names and any other AI tools you used in `README.md` (the disclosure must be complete and truthful).

## D. Verify from a clean clone (do this before recording)
```bash
git clone <url> test-clone && cd test-clone && python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && streamlit run app.py
```

## E. Record the uncut ≤3:00 demo (20 pts; no slides, no cuts)
Use OBS or the OS recorder with the microphone on. Rehearse twice. Warm the app up first (open once, click through each tab) so nothing loads slowly. Suggested script:
- 0:00-0:20 Problem: rain → 2-4 week lag → hospital spikes. Show the **Overview** tab (KPIs, top-10 wards, tier mix).
- 0:20-1:15 **Ward forecast**: pick DNCC_Ward_01 on 2025-08-04: week 2/3/4 cards with intervals and tiers, drivers (rainfall lag), observed weather, then open the *lag-pipeline check* expander (say: lags are per ward, verified against raw weather).
- 1:15-1:45 **Risk maps & ranking**: ranking, ward×week heatmap; say the dataset has no ward coordinates so the geographic map is city-level.
- 1:45-2:15 **What-if**: rainfall ×2 on the last 4 weeks → tier rises; mention same model, same lag code.
- 2:15-2:50 **Validation**: MAE 2.64 vs 4.03 seasonal-naive; the original-vs-improved table (missed High/Critical 127 → 40, false-alarm rate 6.2% → 12.3%); honest caveats (synthetic-looking data, backtest only, false-alarm budget not preserved on test).
- 2:50-2:58 Offline, joblib model, tests, no API/LLM cost. Stop before 3:00.
If you want to show the Live tab, only do it with internet and be honest that it is experimental; otherwise skip it.

## F. Submit
Submit the public repo link and the video link/file via the submission form on the brief (open the PDF's "Submission Link"). Before submitting, confirm: repo public, README commands work, `ARCHITECTURE.md` at repo root, video ≤3:00, AI disclosure filled. Submit before Nov 22, 2:00 PM BST (earlier is safer).
