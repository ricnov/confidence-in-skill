"""cis — confidence in skill is one minus a p-value.

Replication and multiple-testing analysis of the Time-Varying Skill framework of
Muralidhar (2026), *The Journal of Performance Measurement*, Summer 2026, pp. 10–24.
"""
from .core import (
    tracking_error, k_from_annualised, cis_from_k, k_from_cis,
    log_excess, t_statistic, p_value, skill_from_returns,
)
from .multiple import bh_q, bh_reject, holm_reject, bh_thresholds
from .appendix import load_published, model_consistent_from_appendix, implied_drag, summary

__version__ = "0.1.0"
__all__ = [
    "tracking_error", "k_from_annualised", "cis_from_k", "k_from_cis", "log_excess",
    "t_statistic", "p_value", "skill_from_returns", "bh_q", "bh_reject", "holm_reject",
    "bh_thresholds", "load_published", "model_consistent_from_appendix", "implied_drag", "summary",
]
