"""
plot_ncep6_compression.py
=========================
Plots compression ratio vs relative error for the NCEP-Air-6 dataset
(6-way tensor with time reshaped into tod x doy x year),
comparing tsvdmi-dct and tsvdmii-dct algorithms.

X-axis: relative error (linear scale)
Y-axis: compression ratio (log scale)

tsvdmi data points are annotated with their rank parameter k.

Run from the project root:
    python scripts/plot_ncep6_compression.py

Output: plots/ncep6_compression.pdf
"""

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

PERM_MODE = "012345"

# Number of OMP threads to filter by. Compression ratio and relative error are
# thread-independent, so we pick one thread count to avoid duplicate data points.
OMP_NUM_THREADS = 64

DNAME    = "ncep-air-6"
CSV_FILE = "scripts/experiments.csv"    # path relative to project root
OUTFILE  = "plots/ncep6_compression.pdf"  # path relative to project root
TITLE    = "NCEP Air 6-way"

# Figure size in inches (width, height)
FIG_SIZE = (3.33, 3.0)

# Font size for k= annotations on tsvdmi data points
ANNOTATION_FONTSIZE = 6

# Offset (in points) for annotations relative to their data point
ANNOTATION_OFFSET = (5, 5)

# ---------------------------------------------------------------------------
# Load and filter
# ---------------------------------------------------------------------------

data = pd.read_csv(CSV_FILE, dtype={"perm_mode_f": str})

data = data[
    (data["dname_f"]       == DNAME)            &
    (data["perm_mode_f"]   == PERM_MODE)        &
    (data["mtype_f"]       == "dct")            &
    (data["omp_threads_f"] == OMP_NUM_THREADS)  &
    (data["complete"]      == True)
]

data_tsvdmi  = data[data["alg"] == "tsvdmi" ].sort_values("relative_err")
data_tsvdmii = data[data["alg"] == "tsvdmii"].sort_values("relative_err")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=FIG_SIZE)
gs  = GridSpec(nrows=1, ncols=1)
ax  = fig.add_subplot(gs[0, 0])

if not data_tsvdmi.empty:
    ax.plot(data_tsvdmi["relative_err"], data_tsvdmi["compression_ratio"],
            marker="s", label="tsvdmi-dct")
    for _, row in data_tsvdmi.iterrows():
        ax.annotate(f"k={int(row['k'])}", (row["relative_err"], row["compression_ratio"]),
                    textcoords="offset points", xytext=ANNOTATION_OFFSET,
                    fontsize=ANNOTATION_FONTSIZE)

if not data_tsvdmii.empty:
    ax.plot(data_tsvdmii["relative_err"], data_tsvdmii["compression_ratio"],
            marker="x", label="tsvdmii-dct")
    # Annotate each point with its tolerance parameter tol
    for _, row in data_tsvdmii.iterrows():
        ax.annotate(f"tol={row['tol']}", (row["relative_err"], row["compression_ratio"]),
                    textcoords="offset points", xytext=ANNOTATION_OFFSET,
                    fontsize=ANNOTATION_FONTSIZE)

ax.set_yscale("log")
ax.set_xlabel("relative error")
ax.set_ylabel("compression ratio")
ax.grid(True)
ax.legend()
ax.set_title(TITLE)

plt.savefig(OUTFILE, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTFILE}")
