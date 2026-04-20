"""
ncep_eof.py — Read NCEP air temperature data, compress with EOF (randomized SVD)
at a given tolerance, reconstruct, and save the reconstructed tensor to disk.

The tensor (73, 144, 17, time) is unfolded to a 2D matrix (73*144*17, time),
randomized SVD is applied at k_max, then truncated to k* modes that satisfy
the energy criterion matching tol, and the reconstruction is folded back.

Output files written to --outdir/{run_name}/:
    reconstruction.npy   reconstructed tensor, shape (lat,lon,level,time),
                         float64, Fortran order
    meta.json            provenance, parameters, rank, compression ratio, error, timing

Usage:
    python ncep_eof.py \
        --data-dir /global/cfs/cdirs/m4293/taufique/NCEP-NCAR/pressure \
        --year-start 1948 --year-end 1957 \
        --tol 0.01 \
        --k-max 1000 \
        --outdir /pscratch/sd/t/taufique/pystarm/extremes
"""

import argparse
import json
import os
import time
import numpy as np
from sklearn.utils.extmath import randomized_svd
from experiments import read_ncep_air


def make_run_name(tol, k_max, threads):
    return f"ncep-air_eof_{tol}_{k_max}_{threads}"


def main():
    parser = argparse.ArgumentParser(
        description="Compress and reconstruct NCEP air temperature with EOF (randomized SVD)."
    )
    parser.add_argument("--data-dir",   type=str,   required=True,
                        help="Directory containing air.{year}.nc files")
    parser.add_argument("--year-start", type=int,   default=1948,
                        help="First year to load inclusive (default: 1948)")
    parser.add_argument("--year-end",   type=int,   default=1957,
                        help="Last year to load inclusive (default: 1957)")
    parser.add_argument("--tol",        type=float, required=True,
                        help="Energy tolerance, e.g. 0.01")
    parser.add_argument("--k-max",      type=int,   default=1000,
                        help="Max rank for randomized SVD (default: 1000)")
    parser.add_argument("--outdir",     type=str,   required=True,
                        help="Output base directory (run subdirectory created inside)")
    args = parser.parse_args()

    threads = os.environ.get("OMP_NUM_THREADS", "1")
    rname   = make_run_name(args.tol, args.k_max, threads)
    run_dir = os.path.join(args.outdir, rname)

    os.makedirs(run_dir, exist_ok=True)
    out_npy  = os.path.join(run_dir, "reconstruction.npy")
    out_meta = os.path.join(run_dir, "meta.json")

    print(f"Run name : {rname}")
    print(f"Output   : {run_dir}")

    # --- Load data ---
    print(f"\nLoading NCEP air data: {args.year_start}–{args.year_end}")
    t0 = time.perf_counter()
    arr = read_ncep_air(args.data_dir, "air", args.year_start, args.year_end)
    t1 = time.perf_counter()
    load_time = t1 - t0
    print(f"Load time : {load_time:.1f}s  shape: {arr.shape}  dtype: {arr.dtype}")

    orig_shape    = arr.shape   # (73, 144, 17, time)
    original_size = arr.size

    # --- Unfold to 2D matrix (space*level, time) ---
    # Reshape with Fortran order to be consistent with the column-major layout
    # used throughout this codebase.
    lat, lon, level, ntime = orig_shape
    nspace = lat * lon * level   # 178416
    A2d = arr.reshape((nspace, ntime), order='F')
    print(f"Unfolded shape    : {A2d.shape}")

    # --- Randomized SVD ---
    print(f"\nRunning randomized SVD (k_max={args.k_max})...")
    t0 = time.perf_counter()
    U, s, VT = randomized_svd(A2d, n_components=args.k_max, random_state=0)
    t1 = time.perf_counter()
    svd_time = t1 - t0
    print(f"SVD time          : {svd_time:.1f}s")
    print(f"U shape           : {U.shape}")
    print(f"s shape           : {s.shape}")
    print(f"VT shape          : {VT.shape}")

    # --- Determine k* from energy criterion matching tol ---
    # Retain smallest k* such that sum(s[:k*]^2) / sum(s^2) >= 1 - tol^2
    energy_total      = np.sum(s ** 2)
    energy_cumulative = np.cumsum(s ** 2)
    energy_threshold  = (1.0 - args.tol ** 2) * energy_total
    k_star            = int(np.searchsorted(energy_cumulative, energy_threshold) + 1)
    k_star            = min(k_star, args.k_max)
    print(f"k* (effective rank): {k_star}  (out of k_max={args.k_max})")

    # --- Truncate to k* ---
    U  = U[:,  :k_star]
    s  = s[    :k_star]
    VT = VT[:k_star, :]

    compressed_size   = U.size + s.size + VT.size
    compression_ratio = original_size / compressed_size

    # --- Reconstruct ---
    print("\nReconstructing...")
    t0 = time.perf_counter()
    # Use (U * s) @ VT to avoid creating a large diagonal matrix
    A2d_reconst = (U * s) @ VT                                    # (nspace, ntime)
    arr_reconst = A2d_reconst.reshape(orig_shape, order='F')      # (73, 144, 17, time)
    # Make Fortran-contiguous copy, consistent with ncep_tsvdmii.py
    x = np.zeros(orig_shape, dtype=np.float64, order='F')
    for i in range(orig_shape[-1]):
        x[..., i] = arr_reconst[..., i]
    arr_reconst = x

    t1 = time.perf_counter()
    reconstruct_time = t1 - t0
    print(f"Reconstruct time  : {reconstruct_time:.1f}s")
    print(f"Compression ratio : {compression_ratio:.4f}x")

    # --- Compute global relative error ---
    rel_error = np.linalg.norm(arr - arr_reconst) / np.linalg.norm(arr)
    print(f"Relative error    : {rel_error:.6f}")

    # --- Save reconstruction ---
    print(f"\nSaving reconstruction to {out_npy} ...")
    t0 = time.perf_counter()
    np.save(out_npy, arr_reconst)
    t1 = time.perf_counter()
    save_time = t1 - t0
    print(f"Save time         : {save_time:.1f}s")

    # --- Save metadata ---
    meta = {
        "year_start":         args.year_start,
        "year_end":           args.year_end,
        "data_dir":           args.data_dir,
        "variable":           "air",
        "tol":                args.tol,
        "k_max":              args.k_max,
        "k_star":             k_star,
        "omp_threads":        threads,
        "tensor_shape":       list(orig_shape),
        "dtype":              "float64",
        "original_size":      original_size,
        "compressed_size":    compressed_size,
        "compression_ratio":  compression_ratio,
        "rel_error":          rel_error,
        "load_time_s":        load_time,
        "svd_time_s":         svd_time,
        "reconstruct_time_s": reconstruct_time,
        "save_time_s":        save_time,
    }
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {out_meta}")

    print("\nDone.")


if __name__ == "__main__":
    main()
