"""
plot_ncep_dct_vs_eye.py
=======================
Plots compression ratio vs relative error for the NCEP-Air dataset,
comparing tsvdmii-dct and tsvdmii-eye transforms.

X-axis: relative error (linear scale)
Y-axis: compression ratio (log scale)

Run from the project root:
    python scripts/plot_ncep_dct_vs_eye.py

Output: plots/ncep_dct_vs_eye.pdf
"""

import pandas as pd
import matplotlib.pyplot as plt
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
# Config
# ---------------------------------------------------------------------------

PERM_MODE       = "0123"
OMP_NUM_THREADS = 64
DNAME           = "ncep-air"
CSV_FILE        = "scripts/experiments_nersc-perlmutter-cpu.csv"
OUTFILE         = "plots/ncep_dct_vs_eye.pdf"
TITLE           = "NCEP Air — DCT vs Identity (tsvdmii)"
FIG_SIZE        = (3.33, 3.0)

# ---------------------------------------------------------------------------
# Load and filter
# ---------------------------------------------------------------------------

data = pd.read_csv(CSV_FILE, dtype={"perm_mode_f": str})
data = data[
    (data["dname_f"]       == DNAME)            &
    (data["perm_mode_f"]   == PERM_MODE)        &
    (data["omp_threads_f"] == OMP_NUM_THREADS)  &
    (data["complete"]      == True)
]

data_tsvdmii_dct = data[(data["alg"] == "tsvdmii") & (data["mtype_f"] == "dct")].sort_values("relative_err")
data_tsvdmii_eye = data[(data["alg"] == "tsvdmii") & (data["mtype_f"] == "eye")].sort_values("relative_err")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=FIG_SIZE)
gs  = GridSpec(nrows=1, ncols=1)
ax  = fig.add_subplot(gs[0, 0])

if not data_tsvdmii_dct.empty:
    ax.plot(data_tsvdmii_dct["relative_err"], data_tsvdmii_dct["compression_ratio"],
            marker="x", label="tsvdmii-dct")

if not data_tsvdmii_eye.empty:
    ax.plot(data_tsvdmii_eye["relative_err"], data_tsvdmii_eye["compression_ratio"],
            marker="s", label="tsvdmii-eye")


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
