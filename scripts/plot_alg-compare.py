import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
matplotlib.rcParams.update(matplotlib.rcParamsDefault)

# --- Perm mode to use for each dataset ---
PERM_MODE = {
    "traffic-gray":  "120",
    "traffic-color": "0123",
    "dcmall":        "021",
}

# --- Human-readable title for each dataset ---
TITLE = {
    "traffic-gray":  "Traffic Grayscale, perm-mode=(1,2,0)",
    "traffic-color": "Traffic Color, perm-mode=(0,1,2,3)",
    "dcmall":        "DC Mall Hyperspectral, perm-mode=(0,2,1)",
}

parser = argparse.ArgumentParser()
parser.add_argument("--dname", required=True, choices=list(PERM_MODE.keys()),
                    help="Dataset name to plot")
args = parser.parse_args()

dname    = args.dname
perm     = PERM_MODE[dname]
title    = TITLE[dname]
outfile  = f"plots/{dname}_alg-compare.pdf"

csv_file = 'experiments.csv'
data = pd.read_csv(csv_file, dtype={"perm_mode_f": str})

data = data[(data["dname_f"] == dname) & (data["perm_mode_f"] == perm)]
data_tsvdmii = data[(data["mtype_f"] == "dct") & (data["alg"] == "tsvdmii")]
data_tsvdmi  = data[(data["mtype_f"] == "dct") & (data["alg"] == "tsvdmi")]
data_hosvd   = data[(data["mtype_f"] == "eye") & (data["alg"] == "hosvd")]

fig = plt.figure(figsize=(6, 6))
gs  = GridSpec(nrows=1, ncols=1)
ax  = fig.add_subplot(gs[0, 0])

ax.plot(data_tsvdmii['relative_err'], data_tsvdmii['compression_ratio'], marker='x', label="tsvdmii-dct")
ax.plot(data_tsvdmi['relative_err'],  data_tsvdmi['compression_ratio'],  marker='s', label="tsvdmi-dct")
ax.plot(data_hosvd['relative_err'],   data_hosvd['compression_ratio'],   marker='o', label="hosvd")

ax.set_yscale('log', base=2)
ax.set_xlabel("relative error")
ax.set_ylabel("compression ratio")
ax.grid(True)
ax.legend()
ax.set_title(title)

plt.savefig(outfile)
plt.close()
print(f"Saved: {outfile}")
