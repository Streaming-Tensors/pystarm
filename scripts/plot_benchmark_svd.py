"""
plot_benchmark_svd.py
=====================
Answers Q2: Does parfor+sequential SVD outperform loop+parallel-MKL SVD?

One PDF per dataset (ncep-air-6, cfd).
Each figure: two lines (parfor, seq), x = thread count (log2), y = mean time (s).
Runs with time_sec == -1 are excluded. Multiple runs are averaged.

Input:  scripts/benchmark_svd.csv
Output: plots/benchmark_svd_ncep-air-6.pdf
        plots/benchmark_svd_cfd.pdf

Run from the project root:
    python scripts/plot_benchmark_svd.py
"""

import pandas as pd
import matplotlib
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

CSV_PATH = "scripts/benchmark_svd.csv"

DATASETS = {
    "ncep-air-6": "plots/benchmark_svd_ncep-air-6.pdf",
    "cfd":        "plots/benchmark_svd_cfd.pdf",
}

COL_WIDTH  = 3.33
ROW_HEIGHT = 1.5

COLORS = {"parfor": "#4C72B0", "seq": "#DD8452"}
LSTYLE = {"parfor": "-",       "seq": "--"}
MARKER = {"parfor": "o",       "seq": "s"}

LABELS = {"parfor": "parfor+seq SVD", "seq": "loop+parallel SVD"}

# ---------------------------------------------------------------------------
# Load and aggregate
# ---------------------------------------------------------------------------

df = pd.read_csv(CSV_PATH)
df = df[df["time_sec"] > 0]

agg = (df.groupby(["dname", "svd_variant", "threads"])["time_sec"]
         .mean()
         .reset_index()
         .rename(columns={"time_sec": "mean_time"}))

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

for dname, outfile in DATASETS.items():
    fig, ax = plt.subplots(1, 1, figsize=(COL_WIDTH, ROW_HEIGHT * 2))

    for variant in ["parfor", "seq"]:
        sub = agg[
            (agg["dname"] == dname) &
            (agg["svd_variant"] == variant)
        ].sort_values("threads")

        if sub.empty:
            continue

        ax.plot(sub["threads"], sub["mean_time"],
                color=COLORS[variant],
                linestyle=LSTYLE[variant],
                marker=MARKER[variant],
                markersize=3,
                linewidth=1.2,
                label=LABELS[variant])

    threads_present = sorted(agg.loc[agg["dname"] == dname, "threads"].unique())
    ax.set_xscale("log", base=2)
    ax.set_xticks(threads_present)
    ax.set_xticklabels([str(t) for t in threads_present])
    ax.set_xlabel("threads")
    ax.set_ylabel("time (s)")
    ax.set_title(dname)
    ax.grid(True, axis="y", alpha=0.4)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.08))

    fig.suptitle("SVD: parfor+seq vs loop+parallel", fontsize=8, y=1.01)
    plt.tight_layout()
    plt.savefig(outfile, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outfile}")
