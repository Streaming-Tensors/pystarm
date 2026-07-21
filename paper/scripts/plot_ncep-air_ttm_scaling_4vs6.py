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

Input:  scripts/benchmark_ttm.csv  (batched variant, averaged across runs)

Run from the project root:
    python scripts/plot_ncep-air_ttm_scaling_4vs6.py

Output: plots/ncep-air_ttm_scaling_4vs6.pdf
"""

import os
import pandas as pd
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

CSV_PATH = "scripts/benchmark_ttm_alcf-aurora.csv"
OUTFILE  = "plots/ncep-air_ttm_scaling_4vs6.pdf"
TITLE    = "TTM strong scaling: ncep-air-4 vs ncep-air-6"
FIG_SIZE = (3.33, 2.8)

DATASETS = {
    "4-way": {"dname": "ncep-air",   "modes": [2, 3]},
    "6-way": {"dname": "ncep-air-6", "modes": [2, 3, 4, 5]},
}

THREADS = [1, 2, 4, 8, 16, 32, 64]

MODE_COLORS = {
    2: "tab:blue",
    3: "tab:orange",
    4: "tab:green",
    5: "tab:red",
}

# ---------------------------------------------------------------------------
# Load and aggregate (batched variant only, mean across runs)
# ---------------------------------------------------------------------------

df = pd.read_csv(CSV_PATH)
df = df[(df["time_sec"] > 0) & (df["ttm_variant"] == "batched")]

agg = (df.groupby(["dname", "mode", "threads"])["time_sec"]
         .mean()
         .reset_index()
         .rename(columns={"time_sec": "mean_time"}))

# data[dataset_label][threads] = {mode: mean_time}
data = {}
for label, cfg in DATASETS.items():
    data[label] = {}
    sub = agg[agg["dname"] == cfg["dname"]]
    for t in THREADS:
        row = sub[sub["threads"] == t]
        if row.empty:
            data[label][t] = {}
        else:
            data[label][t] = dict(zip(row["mode"], row["mean_time"]))

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=FIG_SIZE)

bar_width   = 0.35
n_threads   = len(THREADS)
x           = np.arange(n_threads)
offsets     = {"4-way": x - bar_width / 2, "6-way": x + bar_width / 2}

# Track total bar heights for speedup annotations
total_heights = {label: {} for label in DATASETS}

for label, cfg in DATASETS.items():
    for i, t in enumerate(THREADS):
        times = data[label][t]
        if not times:
            continue  # missing data — leave bar absent
        bottom = 0.0
        for mode in cfg["modes"]:
            val = times.get(mode, 0.0)
            hatch = "//" if label == "6-way" else None
            ax.bar(offsets[label][i], val, bar_width,
                   bottom=bottom, color=MODE_COLORS[mode],
                   hatch=hatch, edgecolor="white", linewidth=0.5)
            bottom += val
        total_heights[label][t] = bottom

# Speedup annotations relative to 1 thread
for label in DATASETS:
    baseline = total_heights[label].get(1, None)
    if baseline is None:
        continue
    for i, t in enumerate(THREADS):
        if t == 1 or t not in total_heights[label]:
            continue
        speedup = baseline / total_heights[label][t]
        ax.text(offsets[label][i], total_heights[label][t],
                f"{speedup:.1f}×",
                ha='center', va='bottom', fontsize=5, rotation=90)

ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels([str(t) for t in THREADS])
ax.set_xlabel("number of threads")
ax.set_ylabel("time (seconds)")
ax.set_title(TITLE)
ax.grid(True, axis="y")

# --- Legend outside the plot (below) ---
dataset_handles = [
    mpatches.Patch(facecolor="grey", hatch=None,  edgecolor="black", label="ncep-air-4"),
    mpatches.Patch(facecolor="grey", hatch="//",  edgecolor="black", label="ncep-air-6"),
]
mode_handles = [
    mpatches.Patch(color=MODE_COLORS[m], label=f"TTM mode {m + 1}") for m in sorted(MODE_COLORS)
]
fig.legend(handles=dataset_handles + mode_handles,
           loc="lower center", bbox_to_anchor=(0.5, 0), ncol=3, fontsize=7)

plt.tight_layout(rect=[0, 0.18, 1, 1])
plt.savefig(OUTFILE, bbox_inches='tight')
plt.close()
print(f"Saved: {os.path.abspath(OUTFILE)}")
