"""
plot_ncep-air_ttm_scaling_4vs6.py
==================================
Grouped stacked bar chart comparing compress TTM strong scaling for
ncep-air (4-way tensor) vs ncep-air-6 (6-way tensor).

Each group on the x-axis corresponds to a thread count.
Within each group:
  - Left bar:  ncep-air   (4-way) — 2 stacked TTMs (mode 2, mode 3)
  - Right bar: ncep-air-6 (6-way) — 4 stacked TTMs (mode 2, mode 3, mode 4, mode 5)

If data for a thread count is missing, the bar is absent.

Algorithm: tsvdmii-dct, tol=0.01, perm_mode=0123 (4-way) / 012345 (6-way)

Run from the project root:
    python scripts/plot_ncep-air_ttm_scaling_4vs6.py

Output: plots/ncep-air_ttm_scaling_4vs6.pdf
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
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

LOGDIR   = os.path.join(os.environ["SCRATCH"], "pystarm", "logs")
OUTFILE  = "plots/ncep-air_ttm_scaling_4vs6.pdf"
TITLE    = "NCEP Air — Compress TTM Strong Scaling: 4-way vs 6-way (tsvdmii-dct, tol=0.01)"
FIG_SIZE = (8.0, 6.0)

ALG = "tsvdmii"
TOL = "0.01"

DATASETS = {
    "4-way": {"dname": "ncep-air",   "perm_mode": "0123",   "modes": [2, 3]},
    "6-way": {"dname": "ncep-air-6", "perm_mode": "012345", "modes": [2, 3, 4, 5]},
}

THREADS = [8, 16, 32, 64]

# One color per TTM mode — shared across both datasets for consistency
MODE_COLORS = {
    2: "#4C72B0",  # blue
    3: "#DD8452",  # orange
    4: "#55A868",  # green
    5: "#C44E52",  # red
}

# ---------------------------------------------------------------------------
# Parse log files
# ---------------------------------------------------------------------------

def parse_ttm_times(logfile):
    """Return dict {mode: time} for compress TTM lines. Returns {} if file missing."""
    if not os.path.exists(logfile):
        return {}
    times = {}
    pattern = re.compile(
        r'^\[tsvdm_II_compress\] Time for TTM on mode (\d+) :\s+([\d.e+\-]+)',
        re.MULTILINE
    )
    with open(logfile) as f:
        content = f.read()
    for m in pattern.finditer(content):
        times[int(m.group(1))] = float(m.group(2))
    return times

# data[dataset_label][threads] = {mode: time}
data = {}
for label, cfg in DATASETS.items():
    data[label] = {}
    for t in THREADS:
        fname = f"{cfg['dname']}_{ALG}_{TOL}_dct_{cfg['perm_mode']}_{t}"
        data[label][t] = parse_ttm_times(os.path.join(LOGDIR, fname))

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=FIG_SIZE)

bar_width   = 0.35
n_threads   = len(THREADS)
x           = np.arange(n_threads)
offsets     = {"4-way": x - bar_width / 2, "6-way": x + bar_width / 2}

for label, cfg in DATASETS.items():
    for i, t in enumerate(THREADS):
        times = data[label][t]
        if not times:
            continue  # missing data — leave bar absent
        bottom = 0.0
        for mode in cfg["modes"]:
            val = times.get(mode, 0.0)
            ax.bar(offsets[label][i], val, bar_width,
                   bottom=bottom, color=MODE_COLORS[mode],
                   edgecolor="white", linewidth=0.5)
            bottom += val

ax.set_xticks(x)
ax.set_xticklabels([str(t) for t in THREADS])
ax.set_xlabel("number of threads")
ax.set_ylabel("time (seconds)")
ax.set_title(TITLE)
ax.grid(True, axis="y")

# --- Legend ---
# Algorithm/dataset indicators
dataset_handles = [
    mpatches.Patch(facecolor="grey", hatch=None,  edgecolor="black", label="4-way (ncep-air)"),
    mpatches.Patch(facecolor="grey", hatch="//",  edgecolor="black", label="6-way (ncep-air-6)"),
]
# TTM mode colors
mode_handles = [
    mpatches.Patch(color=MODE_COLORS[m], label=f"TTM mode {m}") for m in sorted(MODE_COLORS)
]
ax.legend(handles=dataset_handles + mode_handles, loc="upper right")

plt.tight_layout()
plt.savefig(OUTFILE)
plt.close()
print(f"Saved: {os.path.abspath(OUTFILE)}")
