"""Figure 1 of the paper: null distributions of the count above 90% and of the largest t.

    python paper/make_figure.py data/panel_returns.csv [B] [--raw]      (or the author's workbook .xlsx)

Runs on the cleaned panel of 42 programmes (--raw keeps all 47 rows). Runs the year-cluster wild bootstrap with the
seed used in the paper (B = 50,000, seed 20261011), so the draws are the paper's, and overlays the
distributions that would obtain if the programmes were independent (Student-t draws, each programme's
own degrees of freedom). Writes paper/fig_null.pdf and paper/fig_null.png in matplotlib's default style.
"""
import sys
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cis.panel import load_panel, clean_panel, log_excess_matrix
from cis.bootstrap import wild_bootstrap

args = [a for a in sys.argv[1:] if not a.startswith("--")]
if not args:
    sys.exit(__doc__)
OUT = Path(__file__).resolve().parent
B = int(args[1]) if len(args) > 1 else 50000

panel, _ = load_panel(args[0])
if "--raw" not in sys.argv:
    panel = clean_panel(panel)
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

plt.rcdefaults()
fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.5))

ax = axes[0]
bins = np.arange(-0.5, 36.5, 1)
ax.hist(counts, bins=bins, density=True, color="C0", label="year-cluster bootstrap")
h_ind, _ = np.histogram(counts_ind, bins=bins, density=True)
ax.step(bins[:-1] + 0.5, h_ind, where="mid", color="k", linestyle="--", linewidth=1.0, label="independent")
ax.axvline(obs_count, color="C3", linewidth=1.5, label="observed: %d" % obs_count)
ax.set_xlabel("programmes above 90% under the null")
ax.set_ylabel("probability")
ax.set_xlim(-0.5, 35.5)
ax.legend(loc="upper right", fontsize=7.5)

ax = axes[1]
bins = np.linspace(1.0, 6.0, 51)
ax.hist(maxt, bins=bins, density=True, color="C0", label="year-cluster bootstrap")
h_ind, _ = np.histogram(maxt_ind, bins=bins, density=True)
ax.step(0.5 * (bins[:-1] + bins[1:]), h_ind, where="mid", color="k", linestyle="--", linewidth=1.0, label="independent")
ax.axvline(obs_max, color="C3", linewidth=1.5, label="observed: %s, %.2f" % (top_fund, obs_max))
ax.set_xlabel("largest $t$ under the null")
ax.set_ylabel("density")
ax.set_xlim(1.0, 6.0)
ax.legend(loc="upper right", fontsize=7.5)
fig.tight_layout()
fig.savefig(OUT / "fig_null.pdf")
fig.savefig(OUT / "fig_null.png", dpi=200)
print("count 95/99:", np.percentile(counts, [95, 99]), "| max t 95/99:", np.percentile(maxt, [95, 99]).round(2),
      "| P(count >= obs) = %.4f, P(max >= obs) = %.4f" % ((counts >= obs_count).mean(), (maxt >= obs_max).mean()))
