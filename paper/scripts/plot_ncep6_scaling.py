"""
plot_ncep6_scaling.py
=====================
Strong scaling grouped stacked bar chart for the NCEP-Air-6 dataset
(6-way tensor with time reshaped into tod x doy x year).

Each group on the x-axis corresponds to a thread count. Within each group:
  - Left bar:  tsvdmi  (fixed rank k)
  - Right bar: tsvdmii (tolerance tol)

Each bar is stacked with timing components. Shared components (compress TTM,
reconstruct matmul, reconstruct TTM) use the same color across both algorithms.
tsvdmii bars are hatched to visually distinguish them from tsvdmi bars.

X-axis: thread count
Y-axis: runtime (seconds)

Run from the project root:
    python scripts/plot_ncep6_scaling.py

Output: plots/ncep6_scaling.pdf
"""

import numpy as np
import pandas as pd
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
# Config — edit these to adjust the plot without touching the rest of the script
# ---------------------------------------------------------------------------

PERM_MODE = "012345"
DNAME     = "ncep-air-6"

K   = 1     # rank parameter for tsvdmi
TOL = 0.01  # tolerance parameter for tsvdmii

# Thread counts to include on x-axis (must exist in the CSV)
THREADS = [1, 2, 4, 8, 16, 32, 64]

CSV_FILE = "scripts/experiments_alcf-aurora.csv"
OUTFILE  = "plots/ncep6_scaling.pdf"
TITLE    = "ncep-air-6 — Strong Scaling"

FIG_SIZE = (3.33, 3.0)

# Hatch pattern applied to tsvdmii bars to distinguish from tsvdmi
TSVDMII_HATCH = "//"

# ---------------------------------------------------------------------------
# Colors — shared components use the same color in both algorithms' bars
# ---------------------------------------------------------------------------

COLORS = {
    "compress_ttm":    "tab:blue",
    "slicewise_svd":   "tab:orange",
    "svdvals":         "tab:green",
    "thresholds":      "tab:red",
    "svdks":           "tab:purple",
    "reconstruct_mul": "tab:brown",
    "reconstruct_ttm": "tab:pink",
}

LABELS = {
    "compress_ttm":    "transform TTM",
    "slicewise_svd":   "slicewise SVD",
    "svdvals":         "slicewise SVD (values)",
    "thresholds":      "thresholds",
    "svdks":           "slicewise SVD (vectors)",
    "reconstruct_mul": "reconstruct matmul",
    "reconstruct_ttm": "reconstruct TTM",
}

# Stack order for each algorithm
TSVDMI_COMPONENTS = [
    ("compress_ttm",  "time_compress_ttm_total"),
    ("slicewise_svd", "time_slicewise_svd"),
]

TSVDMII_COMPONENTS = [
    ("compress_ttm", "time_compress_ttm_total"),
    ("svdvals",      "time_slicewise_svdvals"),
    ("thresholds",   "time_thresholds"),
    ("svdks",        "time_slicewise_svdks"),
]

# ---------------------------------------------------------------------------
# Load and filter
# ---------------------------------------------------------------------------

data = pd.read_csv(CSV_FILE, dtype={"perm_mode_f": str})
data = data[
    (data["dname_f"]       == DNAME)     &
    (data["perm_mode_f"]   == PERM_MODE) &
    (data["mtype_f"]       == "dct")     &
    (data["complete"]      == True)      &
    (data["omp_threads_f"].isin([str(t) for t in THREADS]))
]

time_cols = [
    "time_compress_ttm_total", "time_slicewise_svd",
    "time_slicewise_svdvals", "time_thresholds", "time_slicewise_svdks",
]

df_tsvdmi  = (data[(data["alg"] == "tsvdmi")  & (data["k"]   == K  )]
              .groupby("omp_threads_f")[time_cols].mean())
df_tsvdmii = (data[(data["alg"] == "tsvdmii") & (data["tol"] == TOL)]
              .groupby("omp_threads_f")[time_cols].mean())

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=FIG_SIZE)

bar_width = 0.35
x = np.arange(len(THREADS))

legend_handles = {}

def plot_stacked_bars(ax, df, components, x_positions, width, hatch=None):
    """Draw stacked bars for one algorithm. Returns total bar heights (one per thread)."""
    bottoms = np.zeros(len(THREADS))
    for key, col in components:
        values = np.array([
            df.loc[str(t), col] if str(t) in df.index else 0.0
            for t in THREADS
        ], dtype=float)
        ax.bar(x_positions, values, width, bottom=bottoms,
               color=COLORS[key], hatch=hatch,
               edgecolor="white", linewidth=0.5)
        bottoms += values
        if key not in legend_handles:
            legend_handles[key] = mpatches.Patch(color=COLORS[key], label=LABELS[key])
    return bottoms

heights_tsvdmi  = plot_stacked_bars(ax, df_tsvdmi,  TSVDMI_COMPONENTS,  x - bar_width / 2, bar_width)
heights_tsvdmii = plot_stacked_bars(ax, df_tsvdmii, TSVDMII_COMPONENTS, x + bar_width / 2, bar_width,
                                    hatch=TSVDMII_HATCH)

# --- Speedup annotations relative to 1-thread runtime ---
for heights, x_positions in [(heights_tsvdmi, x - bar_width / 2),
                              (heights_tsvdmii, x + bar_width / 2)]:
    baseline = next((h for h in heights if h > 0), None)
    for i, h in enumerate(heights):
        if h <= 0 or h == baseline:
            continue
        speedup = baseline / h
        ax.text(x_positions[i], h, f"{speedup:.1f}×",
                ha='center', va='bottom', fontsize=5, rotation=90)

ax.set_xticks(x)
ax.set_xticklabels([str(t) for t in THREADS])
ax.set_xlabel("number of threads")
ax.set_ylabel("runtime (seconds)")
ax.set_yscale("log")
ax.set_title(TITLE)
ax.grid(True, axis="y")

alg_handles = [
    mpatches.Patch(facecolor="grey", hatch=None,          edgecolor="black", label=f"t-SVDM-I (k={K})"),
    mpatches.Patch(facecolor="grey", hatch=TSVDMII_HATCH, edgecolor="black", label=f"t-SVDM-II (tol={TOL})"),
]
component_handles = list(legend_handles.values())
fig.legend(handles=alg_handles + component_handles,
           loc="lower center", bbox_to_anchor=(0.5, 0), ncol=2, fontsize=7)

plt.tight_layout(rect=[0, 0.18, 1, 1])
plt.savefig(OUTFILE, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTFILE}")
