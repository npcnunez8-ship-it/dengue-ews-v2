"""Prove the regression tests are not vacuous: re-introduce each known bug into a COPY of the project and
check that the test-suite goes red.  Usage:  python tools/mutation_check.py   (writes reports/mutation_check.json)"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MUTATIONS = [
    ("positional lags (row shift instead of calendar date)", "src/features.py",
     "lags[k] = value_at_offset(df, df[col], -k)", "lags[k] = df.groupby('ward_id')[col].shift(k).to_numpy()"),
    ("future leakage in lags (looks forward in time)", "src/features.py",
     "lags[k] = value_at_offset(df, df[col], -k)", "lags[k] = value_at_offset(df, df[col], +k)"),
    ("partial rolling windows (missing weeks silently ignored)", "src/features.py",
     "stack.sum(axis=0) if col == \"rainfall_mm\" else stack.mean(axis=0))", "np.nansum(stack, axis=0) if col == \"rainfall_mm\" else np.nanmean(stack, axis=0))"),
    ("positional forecast target", "src/features.py",
     "out = out + value_at_offset(df, df[\"dengue_cases\"], h)", "out = out + df.groupby('ward_id')['dengue_cases'].shift(-h).to_numpy()"),
    ("duplicate ward-weeks accepted", "src/features.py",
     "    dup = df.duplicated([\"ward_id\", \"date\"], keep=False)\n    if dup.any():", "    dup = df.duplicated([\"ward_id\", \"date\"], keep=False)\n    if False:"),
    ("duplicate check removed from value_at_offset too", "src/features.py",
     "    if not src.index.is_unique:\n        raise ValueError(\"duplicate ward-week records: calendar alignment is ambiguous\")", "    pass"),
    ("interval no longer contains the point forecast (calibration module)", "src/calibration.py",
     "return np.maximum(p - qlo * sc, 0), p + qhi * sc", "return np.maximum(p - qlo * sc, 0) + 1000, p + qhi * sc"),
    ("interval no longer contains the point forecast (batch inference)", "src/predict.py",
     "out.loc[ok, \"pred\"], out.loc[ok, \"lo\"], out.loc[ok, \"hi\"] = p, lo, hi", "out.loc[ok, \"pred\"], out.loc[ok, \"lo\"], out.loc[ok, \"hi\"] = p, lo + 1000, hi"),
    ("escalation ignores the false-alarm budget", "src/calibration.py",
     "if chosen is None and m[\"false_alarm_rate\"] <= caps[level]:", "if chosen is None:"),
    ("calibration folds leak into the test period", "src/calibration.py",
     "last = pd.Timestamp(C.VAL_END) - pd.Timedelta(weeks=C.MAX_HORIZON)", "last = pd.Timestamp(C.VAL_END) + pd.Timedelta(weeks=60)"),
]


def run_tests(proj: Path) -> tuple[bool, str]:
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "tests"], cwd=proj, capture_output=True, text=True)
    tail = [ln for ln in r.stdout.strip().splitlines() if ln.strip()][-3:]
    return r.returncode == 0, " | ".join(tail)


def main():
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "base"
        shutil.copytree(ROOT, base, ignore=shutil.ignore_patterns(".venv", "__pycache__", ".pytest_cache", ".cache", "*.zip"))
        ok, msg = run_tests(base)
        print(f"baseline (unmodified): {'PASS' if ok else 'FAIL'}  {msg}")
        if not ok:
            sys.exit("baseline suite must be green before mutation testing")
        for name, rel, old, new in MUTATIONS:
            work = Path(tmp) / "m"
            if work.exists():
                shutil.rmtree(work)
            shutil.copytree(base, work)
            f = work / rel
            src = f.read_text()
            assert old in src, f"mutation target not found: {name}"
            f.write_text(src.replace(old, new))
            passed, msg = run_tests(work)
            results.append({"bug_reintroduced": name, "file": rel, "suite_detects_it": not passed, "pytest_tail": msg})
            print(f"{'DETECTED ' if not passed else 'MISSED   '} {name}  [{msg[-90:]}]")
    out = {"baseline_green": True, "mutations": results, "detected": sum(r["suite_detects_it"] for r in results), "total": len(results)}
    (ROOT / "reports" / "mutation_check.json").write_text(json.dumps(out, indent=2))
    print(f"\n{out['detected']}/{out['total']} reintroduced bugs detected")
    sys.exit(0 if out["detected"] == out["total"] else 1)


if __name__ == "__main__":
    main()
