"""
plot_ncep-air_ranks.py
=======================
Two stacked subplots showing per-slice rank retained by tsvdmii for
ncep-air data, comparing dct (top) and eye (bottom) mtypes.

X-axis: slice index
Y-axis: rank retained

Algorithm: tsvdmii, tol=0.01, perm_mode=0123, 64 threads

Run from the project root:
    python scripts/plot_ncep-air_ranks.py

Output: plots/ncep-air_ranks.pdf
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams.update({
    'font.size':        8,
    'axes.titlesize':   8,
    'axes.labelsize':   8,
    'xtick.labelsize':  7,
    'ytick.labelsize':  7,
    'legend.fontsize':  7,
})

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RANKS_DIR = os.path.join(os.environ["SCRATCH"], "pystarm", "ranks")
OUTFILE   = "plots/ncep-air_ranks.pdf"
FIG_SIZE  = (8.0, 8.0)
TOL       = "0.01"
THREADS   = 64
PERM_MODE = "0123"

SERIES = [
    {"mtype": "dct", "label": "tsvdmii-dct", "color": "#4C72B0"},
    {"mtype": "eye", "label": "tsvdmii-eye", "color": "#DD8452"},
]

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=FIG_SIZE, sharex=False)

for ax, cfg in zip(axes, SERIES):
    dirname  = f"ncep-air_tsvdmii_{TOL}_{cfg['mtype']}_{PERM_MODE}_{THREADS}_ranks"
    npz_path = os.path.join(RANKS_DIR, dirname, "ranks.npz")

    data         = np.load(npz_path)
    slice_ranks  = data["slice_ranks"]
    nslices      = len(slice_ranks)

    if nslices <= 2000:
        ax.bar(range(nslices), slice_ranks, color=cfg["color"], width=1.0)
    else:
        ax.plot(range(nslices), slice_ranks, color=cfg["color"], linewidth=0.5)

    ax.set_title(f"{cfg['label']} — tol={TOL}")
    ax.set_xlabel("slice index")
    ax.set_ylabel("rank retained")
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.grid(True, axis="y")

plt.suptitle("ncep-air-4 — Per-slice Rank Retained by tsvdmii (perm_mode=0123, 64 threads)",
             fontsize=9)
plt.tight_layout()
plt.savefig(OUTFILE, bbox_inches='tight')
plt.close()
print(f"Saved: {os.path.abspath(OUTFILE)}")
