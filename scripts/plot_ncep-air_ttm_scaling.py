"""
plot_ncep-air_ttm_scaling.py
=============================
Grouped bar chart of compress TTM timings per mode for the NCEP-Air dataset,
across thread counts (strong scaling).

Each group on the x-axis corresponds to a thread count.
Within each group, one bar per TTM mode (mode 2, mode 3).

Algorithm: tsvdmii-dct, perm_mode=0123, tol=0.01

Run from the project root:
    python scripts/plot_ncep-air_ttm_scaling.py

Output: plots/ncep-air_ttm_scaling.pdf
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

LOGDIR    = os.path.join(os.environ["SCRATCH"], "pystarm", "logs")
OUTFILE   = "plots/ncep-air_ttm_scaling.pdf"
TITLE     = "NCEP Air — Compress TTM Strong Scaling (tsvdmii-dct, tol=0.01)"
FIG_SIZE  = (8.0, 6.0)

DNAME     = "ncep-air"
ALG       = "tsvdmii"
TOL       = "0.01"
MTYPE     = "dct"
PERM_MODE = "0123"
THREADS   = [8, 16, 32, 64]

# TTM modes present in ncep-air with perm_mode=0123
MODES = [2, 3]

COLORS = {
    2: "#4C72B0",  # blue
    3: "#DD8452",  # orange
}

# ---------------------------------------------------------------------------
# Parse log files
# ---------------------------------------------------------------------------

def parse_ttm_times(logfile):
    """Return dict {mode: time} for compress TTM lines in a log file."""
    times = {}
    pattern = re.compile(
        r'^\[tsvdm_II_compress\] Time for TTM on mode (\d+) :\s+([\d.e+\-]+)',
        re.MULTILINE
    )
    with open(logfile) as f:
        content = f.read()
    for m in pattern.finditer(content):
        mode = int(m.group(1))
        time = float(m.group(2))
        times[mode] = time
    return times

# {threads: {mode: time}}
data = {}
for t in THREADS:
    fname = f"{DNAME}_{ALG}_{TOL}_{MTYPE}_{PERM_MODE}_{t}"
    fpath = os.path.join(LOGDIR, fname)
    data[t] = parse_ttm_times(fpath)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=FIG_SIZE)

n_threads = len(THREADS)
n_modes   = len(MODES)
bar_width = 0.35
group_width = bar_width * n_modes
x = np.arange(n_threads)

for i, mode in enumerate(MODES):
    offsets = x - group_width / 2 + i * bar_width + bar_width / 2
    values  = [data[t].get(mode, 0.0) for t in THREADS]
    ax.bar(offsets, values, bar_width,
           color=COLORS[mode], label=f"TTM mode {mode}")

ax.set_xticks(x)
ax.set_xticklabels([str(t) for t in THREADS])
ax.set_xlabel("number of threads")
ax.set_ylabel("time (seconds)")
ax.set_title(TITLE)
ax.legend()
ax.grid(True, axis="y")

plt.tight_layout()
plt.savefig(OUTFILE)
plt.close()
print(f"Saved: {os.path.abspath(OUTFILE)}")
