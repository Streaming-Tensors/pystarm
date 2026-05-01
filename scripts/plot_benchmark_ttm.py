"""
plot_benchmark_ttm.py
=====================
Answers Q1: Does batched GEMM outperform loop GEMM for TTM?

One PDF per dataset (ncep-air, ncep-air-6, cfd).
Each figure has one subplot per TTM mode.
Each subplot: three lines (batched, loop, parfor), x = thread count (log2), y = mean time (s).
Runs with time_sec == -1 are excluded. Multiple runs are averaged.

Input:  scripts/benchmark_ttm_<machine>.csv
Output: plots/benchmark_ttm_ncep-air.pdf
        plots/benchmark_ttm_ncep-air-6.pdf
        plots/benchmark_ttm_cfd.pdf

Run from the project root:
    python scripts/plot_benchmark_ttm.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    'font.size':       8,
    'axes.titlesize':  8,
    'axes.labelsize':  8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
})

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CSV_PATH = "scripts/benchmark_ttm_alcf-aurora.csv"
OUTDIR   = "plots"

DATASETS = {
    "ncep-air":   {"modes": [2, 3],       "outfile": "plots/benchmark_ttm_ncep-air.pdf"},
    "ncep-air-6": {"modes": [2, 3, 4, 5], "outfile": "plots/benchmark_ttm_ncep-air-6.pdf"},
    "cfd":        {"modes": [2, 3, 4],    "outfile": "plots/benchmark_ttm_cfd.pdf"},
    "xray":       {"modes": [2],          "outfile": "plots/benchmark_ttm_xray.pdf"},
}

# Mode dimension sizes (0-based mode index → size). Omit entries where size is unknown.
MODE_DIMS = {
    "ncep-air":   {2: 17,  3: 14612},
    "ncep-air-6": {2: 17,  3: 4,    4: 365, 5: 10},
    "xray":       {2: 400},
}

DISPLAY_NAMES = {"ncep-air": "ncep-air-4", "ncep-air-6": "ncep-air-6", "cfd": "cfd", "xray": "xray"}

# ACM two-column: one column ~ 3.33 inches wide
COL_WIDTH = 3.33
ROW_HEIGHT = 1.5

COLORS = {"batched": "tab:blue", "loop": "tab:orange", "parfor": "tab:green"}
LSTYLE = {"batched": "-",       "loop": "--",      "parfor": ":"}
MARKER = {"batched": "o",       "loop": "s",       "parfor": "^"}

# ---------------------------------------------------------------------------
# Load and aggregate
# ---------------------------------------------------------------------------

df = pd.read_csv(CSV_PATH)
df = df[df["time_sec"] > 0]

agg = (df.groupby(["dname", "mode", "ttm_variant", "threads"])["time_sec"]
         .mean()
         .reset_index()
         .rename(columns={"time_sec": "mean_time"}))

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

for dname, cfg in DATASETS.items():
    modes = cfg["modes"]
    n_panels = len(modes)

    fig, axes = plt.subplots(n_panels, 1,
                             figsize=(COL_WIDTH, ROW_HEIGHT * n_panels),
                             sharey=False)
    if n_panels == 1:
        axes = [axes]

    for ax, mode in zip(axes, modes):
        for variant in ["batched", "loop", "parfor"]:
            sub = agg[
                (agg["dname"] == dname) &
                (agg["mode"]  == mode)  &
                (agg["ttm_variant"] == variant)
            ].sort_values("threads")

            if sub.empty:
                continue

            ax.plot(sub["threads"], sub["mean_time"],
                    color=COLORS[variant],
                    linestyle=LSTYLE[variant],
                    marker=MARKER[variant],
                    markersize=3,
                    linewidth=1.2,
                    label=variant)

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        threads_present = sorted(agg.loc[
            (agg["dname"] == dname) & (agg["mode"] == mode), "threads"
        ].unique())
        ax.set_xticks(threads_present)
        ax.set_xticklabels([str(t) for t in threads_present])
        ax.set_xlabel("threads")
        ax.set_ylabel("time (s)")
        mode_1based = mode + 1
        dim = MODE_DIMS.get(dname, {}).get(mode)
        title = f"mode {mode_1based} (dimension={dim})" if dim is not None else f"mode {mode_1based}"
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.4)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(f"TTM Benchmarking — {DISPLAY_NAMES[dname]}", fontsize=8, y=1.01)
    plt.tight_layout()
    plt.savefig(cfg["outfile"], bbox_inches="tight")
    plt.close()
    print(f"Saved: {cfg['outfile']}")
