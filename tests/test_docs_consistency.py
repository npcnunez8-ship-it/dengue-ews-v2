"""Documentation must quote the numbers in reports/metrics.json (no stale metrics in README / ARCHITECTURE / checklist)."""
import json

import pandas as pd
import pytest

from src import config as C

ROOT = C.ROOT
M = json.loads((C.REPORT_DIR / "metrics.json").read_text())
T, O = M["test_by_window"][C.MAIN_WINDOW], M["test_by_window_original_method"][C.MAIN_WINDOW]


def _tokens():
    pct = lambda v: f"{v:.0%}"
    return {
        "test MAE": f"{T['MAE']:.2f}", "test RMSE": f"{T['RMSE']:.2f}", "test R2": f"{T['R2']:.2f}",
        "seasonal-naive MAE": f"{T['baselines']['Seasonal-naive (52w earlier)']['MAE']:.2f}",
        "persistence MAE": f"{T['baselines']['Persistence (cases_t x horizons)']['MAE']:.2f}",
        "missed original": str(O["HC_FN"]), "missed improved": str(T["HC_FN"]),
        "recall original": pct(O["HC_recall"]), "recall improved": pct(T["HC_recall"]),
        "precision original": pct(O["HC_precision"]), "precision improved": pct(T["HC_precision"]),
        "FAR original": f"{O['HC_false_alarm_rate']:.1%}", "FAR improved": f"{T['HC_false_alarm_rate']:.1%}",
        "critical coverage original": pct(O["interval_coverage_actual_critical"]), "critical coverage improved": pct(T["interval_coverage_actual_critical"]),
        "decision threshold": f"{T['HC_decision_cut']:.1f}", "PR-AUC": f"{T['HC_PR_AUC']:.2f}",
        "n test": f"{T['n']:,}",
    }


@pytest.mark.parametrize("doc", ["README.md", "ARCHITECTURE.md"])
def test_headline_numbers_in_docs_match_reports(doc):
    text = (ROOT / doc).read_text()
    missing = {k: v for k, v in _tokens().items() if v not in text}
    assert not missing, f"{doc} does not contain current report values: {missing}"


def test_documented_test_count_matches_collected_tests():
    import subprocess, sys, re
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", str(ROOT / "tests")],
                         capture_output=True, text=True, cwd=ROOT).stdout
    n = int(re.search(r"(\d+) tests? collected", out).group(1))
    assert f"({n})" in (ROOT / "README.md").read_text() or f"{n} tests" in (ROOT / "README.md").read_text()


def test_docs_do_not_claim_stale_artifacts():
    for doc in ("README.md", "ARCHITECTURE.md", "RUBRIC_CHECKLIST.md", "SUBMISSION_GUIDE.md"):
        text = (ROOT / doc).read_text()
        for stale in ("32 passed", "recall 99% but", "Test coverage 94%", "2.621", "alpha=", "groupby('ward_id')`, never across wards"):
            assert stale not in text, (doc, stale)
