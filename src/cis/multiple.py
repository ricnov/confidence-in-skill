"""Multiple-testing procedures on a vector of one-sided p-values."""
from __future__ import annotations

import numpy as np

__all__ = ["bh_q", "bh_reject", "holm_reject", "bh_thresholds"]


def bh_q(p) -> np.ndarray:
    """Benjamini–Hochberg q-values (Storey's adjusted p-values), same order as the input."""
    p = np.asarray(p, dtype=float)
    m = p.size
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(ranked, 1.0)
    return out


def bh_reject(p, alpha: float) -> int:
    """Number of rejections of the BH step-up procedure at false discovery rate `alpha`."""
    p = np.sort(np.asarray(p, dtype=float))
    m = p.size
    ok = np.where(p <= np.arange(1, m + 1) * alpha / m)[0]
    return 0 if ok.size == 0 else int(ok.max() + 1)


def holm_reject(p, alpha: float) -> int:
    """Number of rejections of Holm's step-down procedure at family-wise error rate `alpha`."""
    p = np.sort(np.asarray(p, dtype=float))
    m = p.size
    for i, pi in enumerate(p):
        if pi > alpha / (m - i):
            return i
    return m


def bh_thresholds(m: int, alpha: float, ranks: int = 3) -> list[float]:
    """The first `ranks` BH thresholds k*alpha/m — what the top-ranked p-values must beat."""
    return [k * alpha / m for k in range(1, ranks + 1)]
