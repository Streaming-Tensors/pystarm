"""
plot_ncep-air_compression_4vs6.py
===================================
Plots compression ratio vs relative error for ncep-air (4-way) and
ncep-air-6 (6-way) datasets, both using tsvdmii-dct at 64 threads.

X-axis: relative error (linear scale)
Y-axis: compression ratio (log scale)

Run from the project root:
    python scripts/plot_ncep-air_compression_4vs6.py

Output: plots/ncep-air_compression_4vs6.pdf
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
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

OMP_NUM_THREADS = 64
CSV_FILE        = "scripts/experiments_nersc-perlmutter-cpu.csv"
OUTFILE         = "plots/ncep-air_compression_4vs6.pdf"
TITLE           = "ncep-air: compression ratio (4-way vs 6-way)"
FIG_SIZE        = (3.5, 2.8)

ANNOTATE_4WAY = False
ANNOTATE_6WAY = False

ANNOTATION_FONTSIZE = 6
ANNOTATION_OFFSET   = (5, 5)

# ---------------------------------------------------------------------------
# Load and filter
# ---------------------------------------------------------------------------

data = pd.read_csv(CSV_FILE, dtype={"perm_mode_f": str})
data = data[
    (data["alg"]           == "tsvdmii")       &
    (data["mtype_f"]       == "dct")           &
    (data["omp_threads_f"] == OMP_NUM_THREADS) &
    (data["complete"]      == True)
]

data_4way = data[(data["dname_f"] == "ncep-air"  ) & (data["perm_mode_f"] == "0123"  )].sort_values("relative_err")
data_6way = data[(data["dname_f"] == "ncep-air-6") & (data["perm_mode_f"] == "012345")].sort_values("relative_err")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=FIG_SIZE)

if not data_4way.empty:
    ax.plot(data_4way["relative_err"], data_4way["compression_ratio"],
            marker="s", label="tsvdmii-dct 4-way")
    if ANNOTATE_4WAY:
        for _, row in data_4way.iterrows():
            ax.annotate(f"tol={row['tol']}", (row["relative_err"], row["compression_ratio"]),
                        textcoords="offset points", xytext=ANNOTATION_OFFSET,
                        fontsize=ANNOTATION_FONTSIZE)

if not data_6way.empty:
    ax.plot(data_6way["relative_err"], data_6way["compression_ratio"],
            marker="^", label="tsvdmii-dct 6-way")
    if ANNOTATE_6WAY:
        for _, row in data_6way.iterrows():
            ax.annotate(f"tol={row['tol']}", (row["relative_err"], row["compression_ratio"]),
                        textcoords="offset points", xytext=ANNOTATION_OFFSET,
                        fontsize=ANNOTATION_FONTSIZE)

ax.set_yscale("log")
ax.set_xlim(left=0)
ax.set_xlabel("relative error")
ax.set_ylabel("compression ratio")
ax.set_title(TITLE)
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.savefig(OUTFILE, bbox_inches='tight')
plt.close()
print(f"Saved: {os.path.abspath(OUTFILE)}")
