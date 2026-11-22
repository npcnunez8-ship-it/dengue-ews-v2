"""Risk tiers from predicted case counts."""
import numpy as np
from .config import RISK_TIERS


def fit_cutpoints(train_target: np.ndarray, quantiles) -> list:
    """Cut-points (Medium/High/Critical start) from TRAINING target distribution only."""
    cuts = [float(np.quantile(train_target, q)) for q in quantiles]
    for i in range(1, len(cuts)):           # strictly increasing
        cuts[i] = max(cuts[i], cuts[i - 1] + 1.0)
    return cuts


def to_tier_idx(values, cuts) -> np.ndarray:
    return np.searchsorted(np.asarray(cuts), np.asarray(values), side="right")


def to_tier(values, cuts) -> list:
    return [RISK_TIERS[i] for i in to_tier_idx(values, cuts)]
