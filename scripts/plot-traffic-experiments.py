import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import MO, TU, WE, TH, FR, SA, SU
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as ticker
import matplotlib
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
import seaborn as sns

csv_file = 'experiments.csv'  # Replace with your actual CSV file path
data = pd.read_csv(csv_file, dtype={"perm_mode_f": str})
#print(data.columns.values.tolist())
# Grayscale video experiment, with the exact permutation mode suggested by tsvdm authors
data = data[(data["dname_f"] == "traffic-gray") & (data["perm_mode_f"] == "120")]
data_tsvdmii = data[(data["mtype_f"] == "dct") & (data["alg"] == "tsvdmii")]
data_tsvdmi = data[(data["mtype_f"] == "dct") & (data["alg"] == "tsvdmi")]
data_hosvd = data[(data["mtype_f"] == "eye") & (data["alg"] == "hosvd")]

fig = plt.figure(figsize=(6, 6))
gs = GridSpec(nrows=1, ncols=1)
ax = fig.add_subplot(gs[0,0])
ax.plot(data_tsvdmii['relative_err'], data_tsvdmii['compression_ratio'], marker='x', label="tsvdmii-dct")
ax.plot(data_tsvdmi['relative_err'], data_tsvdmi['compression_ratio'], marker='s', label="tsvdmi-dct")
ax.plot(data_hosvd['relative_err'], data_hosvd['compression_ratio'], marker='o', label="hosvd")
ax.set_yscale('log', base=2)
#ax.set_xscale('log', base=2)
ax.set_xlabel("relative error")
ax.set_ylabel("compression ratio")
ax.grid(True)
ax.legend()
ax.set_title("traffic-grayscale, perm-mode=(1,2,0)")
plt.savefig("plots/traffic-gray_alg-compare.pdf")
plt.close()
