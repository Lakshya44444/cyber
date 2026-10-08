"""Split-conformal alert thresholds (the "false-alert budget").

We want: P(alert | genuine) <= alpha. Take the scores of genuine (non-scam)
cases in a held-out calibration set and use their finite-sample-corrected
(1 - alpha) quantile as the threshold (Angelopoulos & Bates, 2021).
Mondrian version: one threshold per subgroup (age band, language), so the
budget holds inside each group, not just on average.
"""
import numpy as np


def conformal_threshold(neg_scores, alpha: float) -> float:
    s = np.sort(np.asarray(neg_scores))
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return float("inf")  # not enough calibration data for this alpha
    return float(s[k - 1])


def mondrian_thresholds(scores, y, groups, alpha):
    out = {}
    for g in np.unique(groups):
        m = (groups == g) & (y == 0)
        out[g] = conformal_threshold(scores[m], alpha)
    return out


def apply_mondrian(scores, groups, thr: dict):
    return np.array([s > thr[g] for s, g in zip(scores, groups)])
