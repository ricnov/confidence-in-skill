"""Figure 1 of the paper: null distributions of the count above 90% and of the largest t.

    python paper/make_figure.py data/panel_returns.csv [B]      (or the author's workbook .xlsx)

Runs on data/panel_returns.csv (or the author's workbook). Runs the year-cluster wild bootstrap with the
seed used in the paper (B = 50,000, seed 20261011), so the draws are the paper's, and overlays the
distributions that would obtain if the 47 funds were independent (Student-t draws, each fund's
own degrees of freedom). Writes paper/fig_null.pdf and paper/fig_null.png.
"""
import sys
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cis.panel import load_panel, log_excess_matrix
from cis.bootstrap import wild_bootstrap

if len(sys.argv) < 2:
    sys.exit(__doc__)
OUT = Path(__file__).resolve().parent
B = int(sys.argv[2]) if len(sys.argv) > 2 else 50000

panel, _ = load_panel(sys.argv[1])
mat = log_excess_matrix(panel)
boot = wild_bootstrap(mat, B=B, seed=20261011)
T, t_obs, n_obs = boot["T"], boot["t"], boot["n"]
m = t_obs.size

counts = (stats.norm.cdf(T) > 0.9).sum(axis=1)
maxt = T.max(axis=1)
rng = np.random.default_rng(20261012)
S = stats.t.rvs(df=n_obs - 1, size=(20000, m), random_state=rng)
counts_ind = (stats.norm.cdf(S) > 0.9).sum(axis=1)
maxt_ind = S.max(axis=1)
obs_count = int((stats.norm.cdf(t_obs) > 0.9).sum())
obs_max = float(t_obs.max())
top_fund = boot["funds"][int(np.argmax(t_obs))]

BLUE, GRAY, INK, MUTED = "#2a78d6", "#c9ccd1", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8))

ax = axes[0]
bins = np.arange(-0.5, 36.5, 1)
ax.hist(counts, bins=bins, density=True, color=GRAY, edgecolor="white", linewidth=0.5, label="year-cluster bootstrap")
h_ind, _ = np.histogram(counts_ind, bins=bins, density=True)
ax.step(bins[:-1] + 0.5, h_ind, where="mid", color=MUTED, linewidth=1.2, linestyle="--", label="independent funds")
ax.axvline(obs_count, color=BLUE, linewidth=2)
ax.text(obs_count + 0.6, ax.get_ylim()[1] * 0.92, "observed: %d" % obs_count, color=INK, fontsize=8.5, va="top")
ax.set_xlabel("funds above 90% confidence under the null")
ax.set_ylabel("probability")
ax.set_xlim(-0.5, 35.5)
ax.legend(frameon=False, fontsize=8, loc="center right")

ax = axes[1]
bins = np.linspace(1.0, 6.0, 51)
ax.hist(maxt, bins=bins, density=True, color=GRAY, edgecolor="white", linewidth=0.5)
h_ind, _ = np.histogram(maxt_ind, bins=bins, density=True)
ax.step(0.5 * (bins[:-1] + bins[1:]), h_ind, where="mid", color=MUTED, linewidth=1.2, linestyle="--")
ax.axvline(obs_max, color=BLUE, linewidth=2)
ax.text(obs_max + 0.08, ax.get_ylim()[1] * 0.92, "%s: %.2f" % (top_fund, obs_max), color=INK, fontsize=8.5, va="top")
ax.set_xlabel("largest $t$ of the %d funds under the null" % m)
ax.set_ylabel("density")
ax.set_xlim(1.0, 6.0)
fig.tight_layout(w_pad=2.0)
fig.savefig(OUT / "fig_null.pdf")
fig.savefig(OUT / "fig_null.png", dpi=200)
print("count 95/99:", np.percentile(counts, [95, 99]), "| max t 95/99:", np.percentile(maxt, [95, 99]).round(2),
      "| P(count >= obs) = %.4f, P(max >= obs) = %.4f" % ((counts >= obs_count).mean(), (maxt >= obs_max).mean()))
