"""
plot_xray_compression.py
========================
Plots compression ratio vs relative error for the xray dataset,
comparing tsvdmi-dct and tsvdmii-dct algorithms.

X-axis: relative error (linear scale)
Y-axis: compression ratio (log scale)

tsvdmi data points are annotated with their rank parameter k.

Run from the project root:
    python scripts/plot_xray_compression.py

Output: plots/xray_compression.pdf
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
# Config
# ---------------------------------------------------------------------------

PERM_MODE       = "012"
OMP_NUM_THREADS = 64
DNAME           = "xray"
CSV_FILE        = "scripts/experiments_nersc-perlmutter-cpu.csv"
OUTFILE         = "plots/xray_compression.pdf"
TITLE           = "xray: compression ratio"

FIG_SIZE             = (3.33, 2.0)

# ---------------------------------------------------------------------------
# Load and filter
# ---------------------------------------------------------------------------

data = pd.read_csv(CSV_FILE, dtype={"perm_mode_f": str})

data = data[
    (data["dname_f"]       == DNAME)           &
    (data["perm_mode_f"]   == PERM_MODE)       &
    (data["mtype_f"]       == "dct")           &
    (data["omp_threads_f"] == OMP_NUM_THREADS) &
    (data["complete"]      == True)
]

data_tsvdmi  = data[data["alg"] == "tsvdmi" ].sort_values("relative_err")
data_tsvdmii = data[data["alg"] == "tsvdmii"].sort_values("relative_err")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=FIG_SIZE)
ax  = fig.add_subplot(GridSpec(nrows=1, ncols=1)[0, 0])

if not data_tsvdmi.empty:
    ax.plot(data_tsvdmi["relative_err"], data_tsvdmi["compression_ratio"],
            marker="s", label="t-SVDM-I-DCT")

if not data_tsvdmii.empty:
    ax.plot(data_tsvdmii["relative_err"], data_tsvdmii["compression_ratio"],
            marker="x", label="t-SVDM-II-DCT")

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
