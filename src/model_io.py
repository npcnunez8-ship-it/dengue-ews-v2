"""Portable model serialization.

Problems this module fixes (seen when the project was cloned on other machines):
  * the old bundle pickled pandas objects whose internals need `pyarrow`/the same pandas build;
  * scikit-learn estimators pickled with another minor version raise InconsistentVersionWarning or fail.
Strategy: (1) the bundle only contains sklearn estimators + plain python/numpy containers (no pandas
objects); (2) the library versions used for training are stored next to it; (3) `ensure_bundle()` loads
it and, if the file is missing / unreadable / built with another scikit-learn, transparently rebuilds it
from the CSV (`python -m src.train --bundle-only`, ~30-60 s) so the app always starts.
"""
import platform
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from . import config as C

BUNDLE_PATH = C.MODEL_DIR / "dengue_bundle.joblib"


def versions() -> dict:
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__}


def _pack_frame(df: pd.DataFrame) -> dict:
    return {"index": [int(i) for i in df.index], "columns": [str(c) for c in df.columns], "values": df.to_numpy(float)}


def _unpack_frame(d: dict) -> pd.DataFrame:
    return pd.DataFrame(d["values"], index=d["index"], columns=d["columns"])


def _pack_series(s: pd.Series) -> dict:
    return {"index": [str(i) for i in s.index], "values": s.to_numpy(float)}


def _unpack_series(d: dict) -> pd.Series:
    return pd.Series(d["values"], index=d["index"])


def save_bundle(bundle: dict, path: Path = BUNDLE_PATH) -> None:
    b = dict(bundle)
    b["climatology"] = _pack_frame(bundle["climatology"])
    b["static_ref"] = _pack_series(bundle["static_ref"])
    b["meta"] = {**bundle.get("meta", {}), "versions": versions()}
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(b, path, compress=3)


def read_bundle(path: Path = BUNDLE_PATH) -> dict:
    """Strict load: raises on any version inconsistency so the caller can rebuild."""
    from sklearn.exceptions import InconsistentVersionWarning
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        b = joblib.load(path)
    if isinstance(b["climatology"], dict):          # packed format
        b["climatology"] = _unpack_frame(b["climatology"])
        b["static_ref"] = _unpack_series(b["static_ref"])
    saved = b.get("meta", {}).get("versions", {}).get("scikit_learn")
    if saved is None or saved.split(".")[:2] != sklearn.__version__.split(".")[:2]:
        raise RuntimeError(f"bundle built with scikit-learn {saved}, installed {sklearn.__version__}")
    return b


def ensure_bundle(path: Path = BUNDLE_PATH, log=print) -> dict:
    """Load the bundle; rebuild it from the CSV when it cannot be used on this machine.
    
    Production bundle (models/dengue_bundle.joblib) will NOT automatically retrain or
    overwrite without explicit authorization via ALLOW_BUNDLE_REBUILD=1.
    """
    import os
    try:
        return read_bundle(path)
    except Exception as e:   # noqa: BLE001  (any failure -> rebuild)
        is_protected_bundle = (Path(path).resolve() == BUNDLE_PATH.resolve())
        opt_in = os.environ.get("ALLOW_BUNDLE_REBUILD", "").strip().lower() in ("1", "true", "yes")

        if is_protected_bundle and not opt_in:
            raise RuntimeError(
                f"[model_io] Protected model bundle '{path.name}' is incompatible or unreadable in current environment ({type(e).__name__}: {e}).\n"
                f"Automatic retraining/overwriting of models/{BUNDLE_PATH.name} is disabled by default to protect canonical model baselines.\n\n"
                f"Action options:\n"
                f"1. Run under the canonical .venv (Python 3.12, scikit-learn 1.9.1):\n"
                f"   .\\.venv\\Scripts\\python.exe -m streamlit run app.py\n"
                f"2. If rebuilding is genuinely required, explicitly opt in via ALLOW_BUNDLE_REBUILD=1:\n"
                f"   Windows PowerShell: $env:ALLOW_BUNDLE_REBUILD='1'; python -m src.train --bundle-only\n"
                f"   Linux/macOS:        ALLOW_BUNDLE_REBUILD=1 python -m src.train --bundle-only\n"
            ) from e

        log(f"[model_io] Rebuilding bundle for {path.name} ({type(e).__name__}: {e}) from data/ ...")
        from .train import main as train_main
        train_main(bundle_only=True)
        return read_bundle(path)
