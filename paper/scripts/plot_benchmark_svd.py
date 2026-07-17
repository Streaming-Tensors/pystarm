"""
plot_benchmark_svd.py
=====================
Answers Q2: Does parfor+sequential SVD outperform loop+parallel-MKL SVD?

One PDF with one subplot per dataset (ncep-air-6, cfd, xray), stacked vertically.
Each subplot: two lines (parfor, seq), x = thread count (log2), y = mean time (s).
Runs with time_sec == -1 are excluded. Multiple runs are averaged.

Input:  scripts/benchmark_svd_<machine>.csv
Output: plots/benchmark_svd.pdf

Run from the project root:
    python scripts/plot_benchmark_svd.py
"""

import pandas as pd
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

CSV_PATH = "scripts/benchmark_svd_alcf-aurora.csv"
OUTFILE  = "plots/benchmark_svd.pdf"

DATASETS = ["ncep-air-6", "cfd", "xray"]

DISPLAY_NAMES = {
    "ncep-air-6": "ncep-air-6 (248,200 slices of 73×144)",
    "cfd":        "cfd (6,072 slices of 253×253)",
    "xray":       "xray (400 slices of 300×400)",
}

COL_WIDTH  = 3.33
ROW_HEIGHT = 1.5

COLORS = {"parfor": "tab:blue", "seq": "tab:orange"}
LSTYLE = {"parfor": "-",        "seq": "--"}
MARKER = {"parfor": "o",        "seq": "s"}

LABELS = {"parfor": "parallel slices, sequential SVD",
          "seq":    "sequential slices, parallel SVD"}

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

n_panels = len(DATASETS)
fig, axes = plt.subplots(n_panels, 1,
                         figsize=(COL_WIDTH, ROW_HEIGHT * n_panels),
                         sharey=False)

for ax, dname in zip(axes, DATASETS):
    parfor_sub = None
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

        if variant == "parfor":
            parfor_sub = sub

    # Annotate speedup on parfor line relative to 1-thread time
    if parfor_sub is not None and not parfor_sub.empty:
        base = parfor_sub[parfor_sub["threads"] == 1]["mean_time"]
        if not base.empty:
            base_time = base.values[0]
            for _, row in parfor_sub.iterrows():
                speedup = base_time / row["mean_time"]
                ax.annotate(f"{speedup:.1f}x",
                            xy=(row["threads"], row["mean_time"]),
                            xytext=(0, 5), textcoords="offset points",
                            ha="center", fontsize=6, color="tab:blue")

    threads_present = sorted(agg.loc[agg["dname"] == dname, "threads"].unique())
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(threads_present)
    ax.set_xticklabels([str(t) for t in threads_present])
    ax.set_xlabel("threads")
    ax.set_ylabel("time (s)")
    ax.set_title(DISPLAY_NAMES[dname])
    ax.grid(True, axis="y", alpha=0.4)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=1,
           bbox_to_anchor=(0.5, -0.08))

fig.suptitle("Slicewise SVD Benchmarking", fontsize=8, y=1.01)
plt.tight_layout()
plt.savefig(OUTFILE, bbox_inches="tight")
plt.close()
print(f"Saved: {OUTFILE}")
