"""
plot_cfd_scaling.py
===================
Strong scaling grouped stacked bar chart for the CFD dataset.

Each group on the x-axis corresponds to a thread count. Within each group:
  - Left bar:  tsvdmi  (fixed rank k)
  - Right bar: tsvdmii (tolerance tol)

Each bar is stacked with timing components. Shared components (compress TTM,
reconstruct matmul, reconstruct TTM) use the same color across both algorithms.
tsvdmii bars are hatched to visually distinguish them from tsvdmi bars.

X-axis: thread count
Y-axis: runtime (seconds)

Run from the project root:
    python scripts/plot_cfd_scaling.py

Output: plots/cfd_scaling.pdf
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

PERM_MODE = "01234"
DNAME     = "cfd"

K   = 20   # rank parameter for tsvdmi
TOL = 0.1  # tolerance parameter for tsvdmii

# Thread counts to include on x-axis (must exist in the CSV)
THREADS = [8, 16, 32, 64]

CSV_FILE = "scripts/experiments.csv"   # path relative to project root
OUTFILE  = "plots/cfd_scaling.pdf"     # path relative to project root
TITLE    = "CFD — Strong Scaling"

FIG_SIZE = (3.33, 3.5)

# Hatch pattern applied to tsvdmii bars to distinguish from tsvdmi
TSVDMII_HATCH = "//"

# ---------------------------------------------------------------------------
# Colors — shared components use the same color in both algorithms' bars
# ---------------------------------------------------------------------------

COLORS = {
    "compress_ttm":    "#4C72B0",  # blue   — shared
    "slicewise_svd":   "#DD8452",  # orange — tsvdmi only
    "svdvals":         "#55A868",  # green  — tsvdmii only
    "thresholds":      "#C44E52",  # red    — tsvdmii only
    "svdks":           "#8172B2",  # purple — tsvdmii only
    "reconstruct_mul": "#937860",  # brown  — shared
    "reconstruct_ttm": "#DA8BC3",  # pink   — shared
}

LABELS = {
    "compress_ttm":    "compress TTM",
    "slicewise_svd":   "slicewise SVD",
    "svdvals":         "slicewise SVDvals",
    "thresholds":      "thresholds",
    "svdks":           "slicewise SVDks",
    "reconstruct_mul": "reconstruct matmul",
    "reconstruct_ttm": "reconstruct TTM",
}

# Stack order for each algorithm
TSVDMI_COMPONENTS = [
    ("compress_ttm",    "time_compress_ttm_total"),
    ("slicewise_svd",   "time_slicewise_svd"),
]

TSVDMII_COMPONENTS = [
    ("compress_ttm",    "time_compress_ttm_total"),
    ("svdvals",         "time_slicewise_svdvals"),
    ("thresholds",      "time_thresholds"),
    ("svdks",           "time_slicewise_svdks"),
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
    (data["omp_threads_f"].isin(THREADS))
]

df_tsvdmi  = data[(data["alg"] == "tsvdmi")  & (data["k"]   == K  )].set_index("omp_threads_f")
df_tsvdmii = data[(data["alg"] == "tsvdmii") & (data["tol"] == TOL)].set_index("omp_threads_f")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=FIG_SIZE)

bar_width = 0.35
x = np.arange(len(THREADS))

# Tracks which component keys have already been added to the legend
legend_handles = {}

def plot_stacked_bars(ax, df, components, x_positions, width, hatch=None):
    """Draw stacked bars for one algorithm. Returns nothing; updates legend_handles."""
    bottoms = np.zeros(len(THREADS))
    for key, col in components:
        values = np.array([
            df.loc[t, col] if t in df.index else 0.0
            for t in THREADS
        ], dtype=float)
        bars = ax.bar(x_positions, values, width, bottom=bottoms,
                      color=COLORS[key], hatch=hatch,
                      edgecolor="white", linewidth=0.5)
        bottoms += values
        # Only add to legend once per component key (shared components appear in both bars)
        if key not in legend_handles:
            legend_handles[key] = mpatches.Patch(color=COLORS[key], label=LABELS[key])

plot_stacked_bars(ax, df_tsvdmi,  TSVDMI_COMPONENTS,  x - bar_width / 2, bar_width)
plot_stacked_bars(ax, df_tsvdmii, TSVDMII_COMPONENTS, x + bar_width / 2, bar_width,
                  hatch=TSVDMII_HATCH)

# --- x-axis: thread count labels at group centers ---
ax.set_xticks(x)
ax.set_xticklabels([str(t) for t in THREADS])
ax.set_xlabel("number of threads")
ax.set_ylabel("runtime (seconds)")
ax.set_title(TITLE)
ax.grid(True, axis="y")

# --- Legend: component colors + algorithm indicators ---
# Add tsvdmi / tsvdmii distinguisher patches at the top of the legend
alg_handles = [
    mpatches.Patch(facecolor="grey", hatch=None,          edgecolor="black", label=f"tsvdmi  (k={K})"),
    mpatches.Patch(facecolor="grey", hatch=TSVDMII_HATCH, edgecolor="black", label=f"tsvdmii (tol={TOL})"),
]
component_handles = list(legend_handles.values())
ax.legend(handles=alg_handles + component_handles, loc="upper right", fontsize=8)

plt.tight_layout()
plt.savefig(OUTFILE)
plt.close()
print(f"Saved: {OUTFILE}")
