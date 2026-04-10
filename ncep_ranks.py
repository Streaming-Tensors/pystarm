"""
ncep_ranks.py — Run tsvdmii on NCEP air temperature data and analyze per-slice ranks.

For each run, saves ranks.npz (ranks + metadata), histogram.pdf, and barplot.pdf
to /pscratch/sd/t/taufique/pystarm/{run_name}/, using the same naming convention as
experiments.sh: {dname}_tsvdmii_{tol}_{mtype}_{perm_mode}_{threads}_ranks.

Skips the compression experiment if ranks.npz already exists in the output directory,
and goes straight to plotting from the saved data.

Usage:
    python ncep_ranks.py -dname ncep-air   -mtype dct -tol 0.01 -perm-mode 0123
    python ncep_ranks.py -dname ncep-air-6 -mtype dct -tol 0.01 -perm-mode 012345
"""

import argparse
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.fft import dct
import pystarm
import pyttb as ttb
from experiments import read_ncep_air, read_ncep_air_6
from alg import tsvdm_II_compress


DATA_DIR   = "/global/cfs/cdirs/m4293/taufique/NCEP-NCAR/pressure"
YEAR_START = 1948
YEAR_END   = 1957
SCRATCH    = "/pscratch/sd/t/taufique/pystarm/ranks"


def run_name(dname, mtype, tol, perm_mode, threads):
    perm_str = "".join(str(p) for p in perm_mode)
    return f"{dname}_tsvdmii_{tol}_{mtype}_{perm_str}_{threads}_ranks"


def compress_and_save(arr, mtype, tol, perm_mode, ranks_file):
    ttm_modes = list(range(2, arr.ndim))

    Ms  = []
    MTs = []
    if mtype == "dct":
        for mode in ttm_modes:
            n   = arr.shape[mode]
            DF  = np.asfortranarray(dct(np.eye(n, dtype=np.float64), axis=0, norm="ortho"))
            DFT = np.asfortranarray(DF.T)
            Ms.append(pystarm.Matrix(DF,  DF.shape[0],  DF.shape[1]))
            MTs.append(pystarm.Matrix(DFT, DFT.shape[0], DFT.shape[1]))
    elif mtype == "eye":
        ttm_modes = []
    elif mtype == "hosvd":
        ttb_tensor = ttb.tensor(arr)
        hosvd      = ttb.hosvd(ttb_tensor, tol=0)
        Us         = hosvd.factor_matrices
        for mode in ttm_modes:
            U  = np.asfortranarray(Us[mode])
            UT = np.asfortranarray(Us[mode].T)
            Ms.append(pystarm.Matrix(U,  U.shape[0],  U.shape[1]))
            MTs.append(pystarm.Matrix(UT, UT.shape[0], UT.shape[1]))

    A = pystarm.Tensor(arr, len(arr.shape), arr.shape)

    print("Running tsvdmii...")
    t0 = time.perf_counter()
    (U_hat, S_hat, VT_hat) = tsvdm_II_compress(A, Ms, ttm_modes, tol, verbose=True)
    t1 = time.perf_counter()
    print(f"Compression time: {t1 - t0:.2f}s")

    slice_ranks = np.array(U_hat.slice_ranks)

    np.savez(ranks_file,
             slice_ranks  = slice_ranks,
             mtype        = np.array(mtype),
             tol          = np.array(tol),
             perm_mode    = np.array(perm_mode),
             tensor_shape = np.array(arr.shape),
             year_start   = np.array(YEAR_START),
             year_end     = np.array(YEAR_END))
    print(f"Saved ranks to {ranks_file}")

    # U_hat.clear()
    # S_hat.clear()
    # VT_hat.clear()
    # A.clear()
    # for m in Ms:
        # m.clear()
    # for m in MTs:
        # m.clear()

    return slice_ranks


def plot_histogram(slice_ranks, dname, mtype, tol, perm_mode, outpath):
    perm_str = "".join(str(p) for p in perm_mode)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(slice_ranks, bins=50, color='steelblue', edgecolor='white')
    ax.set_xlabel("Rank")
    ax.set_ylabel("Number of slices")
    ax.set_title(f"Per-slice rank distribution — {dname}, mtype={mtype}, tol={tol}, perm={perm_str}")
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print(f"Saved histogram to {outpath}")


def plot_barplot(slice_ranks, dname, mtype, tol, perm_mode, outpath):
    perm_str = "".join(str(p) for p in perm_mode)
    nslices  = len(slice_ranks)
    fig, ax  = plt.subplots(figsize=(12, 4))
    if nslices <= 2000:
        ax.bar(range(nslices), slice_ranks, color='steelblue', width=1.0)
    else:
        ax.plot(range(nslices), slice_ranks, color='steelblue', linewidth=0.5)
    ax.set_xlabel("Slice index")
    ax.set_ylabel("Rank")
    ax.set_title(f"Per-slice rank — {dname}, mtype={mtype}, tol={tol}, perm={perm_str}")
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print(f"Saved barplot to {outpath}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dname",     type=str,   default="ncep-air",  help="Dataset name: ncep-air or ncep-air-6")
    parser.add_argument("-mtype",     type=str,   required=True,       help="Transform type: dct, eye, hosvd")
    parser.add_argument("-tol",       type=float, required=True,       help="Energy tolerance for tsvdmii")
    parser.add_argument("-perm-mode", type=str,   default=None,        help="Mode permutation string (default: 0123 for ncep-air, 012345 for ncep-air-6)")
    parser.add_argument("-outdir",    type=str,   default=SCRATCH,     help="Output root directory")
    args = parser.parse_args()

    dname   = args.dname
    mtype   = args.mtype
    tol     = args.tol
    outdir  = args.outdir
    threads = os.environ.get("OMP_NUM_THREADS", "1")

    if args.perm_mode is not None:
        perm_mode = tuple(int(c) for c in args.perm_mode)
    elif dname == "ncep-air-6":
        perm_mode = (0, 1, 2, 3, 4, 5)
    else:
        perm_mode = (0, 1, 2, 3)

    if dname not in ("ncep-air", "ncep-air-6"):
        raise ValueError(f"Unknown dname: {dname}. Must be ncep-air or ncep-air-6.")

    run_dir    = os.path.join(outdir, run_name(dname, mtype, tol, perm_mode, threads))
    ranks_file = os.path.join(run_dir, "ranks.npz")

    if os.path.exists(ranks_file):
        print(f"ranks.npz found in {run_dir} — skipping compression, loading saved ranks.")
        data        = np.load(ranks_file, allow_pickle=True)
        slice_ranks = data["slice_ranks"]
    else:
        os.makedirs(run_dir, exist_ok=True)

        print(f"Loading {dname} data...")
        if dname == "ncep-air-6":
            arr = read_ncep_air_6(DATA_DIR, "air", YEAR_START, YEAR_END)
        else:
            arr = read_ncep_air(DATA_DIR, "air", YEAR_START, YEAR_END)

        if len(perm_mode) != arr.ndim:
            raise ValueError(f"perm_mode length {len(perm_mode)} != tensor ndim {arr.ndim}")
        arr = np.transpose(arr, perm_mode)
        x = np.zeros(arr.shape, dtype=np.float64, order='F')
        for i in range(arr.shape[-1]):
            x[..., i] = arr[..., i]
        arr = x

        print("Tensor shape:", arr.shape)

        slice_ranks = compress_and_save(arr, mtype, tol, perm_mode, ranks_file)

    print(f"\nSlice rank summary:")
    print(f"  nslices : {len(slice_ranks)}")
    print(f"  min     : {slice_ranks.min()}")
    print(f"  max     : {slice_ranks.max()}")
    print(f"  mean    : {slice_ranks.mean():.2f}")
    print(f"  median  : {int(np.median(slice_ranks))}")

    plot_histogram(slice_ranks, dname, mtype, tol, perm_mode, os.path.join(run_dir, "histogram.pdf"))
    plot_barplot(  slice_ranks, dname, mtype, tol, perm_mode, os.path.join(run_dir, "barplot.pdf"))


if __name__ == "__main__":
    main()
