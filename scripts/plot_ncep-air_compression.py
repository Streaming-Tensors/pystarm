"""
plot_ncep-air_compression.py
=============================
Plots compression ratio vs relative error for the NCEP-Air dataset,
comparing tsvdmi-dct, tsvdmii-dct, and EOF algorithms.

X-axis: relative error (linear scale)
Y-axis: compression ratio (log scale, plain number format)

tsvdmi data points are annotated with their rank parameter k.
tsvdmii annotations are commented out but can be re-enabled.

Run from the project root:
    python scripts/plot_ncep-air_compression.py

Output: plots/ncep-air_compression.pdf
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.gridspec import GridSpec
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

# Perm mode to filter by. Change to "3012" to plot a different permutation.
PERM_MODE = "0123"

# Number of OMP threads to filter by. Compression ratio and relative error are
# thread-independent, so we pick one thread count to avoid duplicate data points.
OMP_NUM_THREADS = 64

DNAME    = "ncep-air"
CSV_FILE = "scripts/experiments.csv"   # path relative to project root
OUTFILE  = "plots/ncep-air_compression.pdf" # path relative to project root
TITLE    = "NCEP Air — Compression Ratio vs Relative Error"

# Figure size in inches (width, height)
FIG_SIZE = (8.0, 6.0)

# Font size for annotations on data points
ANNOTATION_FONTSIZE = 6

# Offset (in points) for annotations relative to their data point
ANNOTATION_OFFSET = (5, 5)

# Toggle annotations per series
ANNOTATE_TSVDMI      = False
ANNOTATE_TSVDMII     = False
ANNOTATE_TSVDMII_EYE = False
ANNOTATE_EOF         = False

# ---------------------------------------------------------------------------
# Load and filter
# ---------------------------------------------------------------------------

data = pd.read_csv(CSV_FILE, dtype={"perm_mode_f": str})

# Keep only complete runs for this dataset, perm mode, and transform type.
# mtype_f == "dct": use DCT transform only (change to "eye" for identity, "hosvd" for HOSVD).
data = data[
    (data["dname_f"]       == DNAME)           &
    (data["perm_mode_f"]   == PERM_MODE)       &
    (data["mtype_f"]       == "dct")           &
    (data["omp_threads_f"] == OMP_NUM_THREADS) &
    (data["complete"]      == True)
]

# Split by algorithm and sort by relative error so line segments connect left-to-right
data_tsvdmi  = data[data["alg"] == "tsvdmi" ].sort_values("relative_err")
data_tsvdmii = data[data["alg"] == "tsvdmii"].sort_values("relative_err")

# tsvdmii-eye and EOF do not fit the dct filter above — load separately
all_data = pd.read_csv(CSV_FILE, dtype={"perm_mode_f": str})
data_eof = all_data[
    (all_data["dname_f"]       == DNAME)           &
    (all_data["alg"]           == "eof")           &
    (all_data["omp_threads_f"] == OMP_NUM_THREADS) &
    (all_data["complete"]      == True)
].sort_values("relative_err")

data_tsvdmii_eye = all_data[
    (all_data["dname_f"]       == DNAME)           &
    (all_data["alg"]           == "tsvdmii")       &
    (all_data["mtype_f"]       == "eye")           &
    (all_data["perm_mode_f"]   == PERM_MODE)       &
    (all_data["omp_threads_f"] == OMP_NUM_THREADS) &
    (all_data["complete"]      == True)
].sort_values("relative_err")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=FIG_SIZE)
gs  = GridSpec(nrows=1, ncols=1)
ax  = fig.add_subplot(gs[0, 0])

if not data_tsvdmi.empty:
    ax.plot(data_tsvdmi["relative_err"], data_tsvdmi["compression_ratio"],
            marker="s", label="tsvdmi-dct")
    if ANNOTATE_TSVDMI:
        for _, row in data_tsvdmi.iterrows():
            ax.annotate(f"k={int(row['k'])}", (row["relative_err"], row["compression_ratio"]),
                        textcoords="offset points", xytext=ANNOTATION_OFFSET,
                        fontsize=ANNOTATION_FONTSIZE)

if not data_tsvdmii.empty:
    ax.plot(data_tsvdmii["relative_err"], data_tsvdmii["compression_ratio"],
            marker="x", label="tsvdmii-dct")
    if ANNOTATE_TSVDMII:
        for _, row in data_tsvdmii.iterrows():
            ax.annotate(f"tol={row['tol']}", (row["relative_err"], row["compression_ratio"]),
                        textcoords="offset points", xytext=ANNOTATION_OFFSET,
                        fontsize=ANNOTATION_FONTSIZE)

if not data_tsvdmii_eye.empty:
    ax.plot(data_tsvdmii_eye["relative_err"], data_tsvdmii_eye["compression_ratio"],
            marker="o", label="tsvdmii-eye")
    if ANNOTATE_TSVDMII_EYE:
        for _, row in data_tsvdmii_eye.iterrows():
            ax.annotate(f"tol={row['tol']}", (row["relative_err"], row["compression_ratio"]),
                        textcoords="offset points", xytext=ANNOTATION_OFFSET,
                        fontsize=ANNOTATION_FONTSIZE)

if not data_eof.empty:
    ax.plot(data_eof["relative_err"], data_eof["compression_ratio"],
            marker="^", label="eof")
    if ANNOTATE_EOF:
        for _, row in data_eof.iterrows():
            ax.annotate(f"tol={row['tol']}", (row["relative_err"], row["compression_ratio"]),
                        textcoords="offset points", xytext=ANNOTATION_OFFSET,
                        fontsize=ANNOTATION_FONTSIZE)

# Y-axis log scale helps spread out compression ratios that span orders of magnitude.
# Switch to "linear" if the data range is narrow.
ax.set_yscale("log")
# ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
# ax.yaxis.get_major_formatter().set_scientific(False)

ax.set_xlabel("relative error")
ax.set_ylabel("compression ratio")
ax.grid(True)
ax.legend()
ax.set_title(TITLE)

plt.tight_layout()
plt.savefig(OUTFILE)
plt.close()
print(f"Saved: {os.path.abspath(OUTFILE)}")
