"""
plot_cfd_compression.py
=======================
Plots compression ratio vs relative error for the CFD dataset,
comparing tsvdmi-dct and tsvdmii-dct algorithms.

X-axis: relative error (linear scale)
Y-axis: compression ratio (log scale)

tsvdmi data points are annotated with their rank parameter k.
tsvdmii annotations are commented out (hurt readability) but can be re-enabled.

Run from the project root:
    python scripts/plot_cfd_compression.py

Output: plots/cfd_compression.pdf
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib
import matplotlib.ticker
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

# Perm mode to filter by. CFD experiments are run with a single perm mode.
# Change this if experiments for a different perm mode are added later.
PERM_MODE = "01234"

# Number of OMP threads to filter by. Compression ratio and relative error are
# thread-independent, so we pick one thread count to avoid duplicate data points.
OMP_NUM_THREADS = 64

DNAME    = "cfd"
CSV_FILE = "scripts/experiments_nersc-perlmutter-cpu.csv"
OUTFILE  = "plots/cfd_compression.pdf"
TITLE    = "cfd: compression ratio"

# Figure size in inches (width, height)
FIG_SIZE = (3.33, 2.0)


# ---------------------------------------------------------------------------
# Load and filter
# ---------------------------------------------------------------------------

data = pd.read_csv(CSV_FILE, dtype={"perm_mode_f": str})

# Keep only complete runs for this dataset, perm mode, and transform type.
# mtype_f == "dct": use DCT transform only (change to "eye" for identity, "hosvd" for HOSVD).
data = data[
    (data["dname_f"]       == DNAME)          &
    (data["perm_mode_f"]   == PERM_MODE)      &
    (data["mtype_f"]       == "dct")          &
    (data["omp_threads_f"] == OMP_NUM_THREADS) &
    (data["complete"]      == True)
]

# Split by algorithm and sort by relative error so line segments connect left-to-right
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
            marker="s", label="t-SVDM-I-DCT")

if not data_tsvdmii.empty:
    ax.plot(data_tsvdmii["relative_err"], data_tsvdmii["compression_ratio"],
            marker="x", label="t-SVDM-II-DCT")

# Y-axis log scale helps spread out compression ratios that span orders of magnitude.
# Switch to "linear" if the data range is narrow.
ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel("relative error")
ax.set_ylabel("compression ratio")
ax.grid(True)
ax.legend()
ax.set_title(TITLE)

plt.tight_layout()
plt.savefig(OUTFILE, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTFILE}")
