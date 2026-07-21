"""
ncep_tsvdmii.py — Read NCEP air temperature data, compress with tsvdmii at a
given tolerance, reconstruct, and save the reconstructed tensor to disk.

Output files written to --outdir/{run_name}/:
    reconstruction.npy   reconstructed tensor, shape (lat,lon,level,time),
                         float64, Fortran order
    meta.json            provenance, parameters, compression ratio, error, timing

Usage:
    python ncep_tsvdmii.py \
        --data-dir /global/cfs/cdirs/m4293/taufique/NCEP-NCAR/pressure \
        --year-start 1948 --year-end 1957 \
        --tol 0.01 \
        --mtype dct \
        --perm-mode 0123 \
        --outdir /pscratch/sd/t/taufique/pystarm/extremes
"""

import argparse
import json
import os
import sys
import time
import numpy as np
from scipy.fft import dct
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import pystarm
from experiments import read_ncep_air
from pystarm.alg import tsvdm_II_compress, tsvdm_II_reconstruct


def make_run_name(tol, mtype, perm_mode, threads):
    perm_str = "".join(str(p) for p in perm_mode)
    return f"ncep-air_tsvdmii_{tol}_{mtype}_{perm_str}_{threads}"


def main():
    parser = argparse.ArgumentParser(
        description="Compress and reconstruct NCEP air temperature with tsvdmii."
    )
    parser.add_argument("--data-dir",   type=str,   required=True,
                        help="Directory containing air.{year}.nc files")
    parser.add_argument("--year-start", type=int,   default=1948,
                        help="First year to load inclusive (default: 1948)")
    parser.add_argument("--year-end",   type=int,   default=1957,
                        help="Last year to load inclusive (default: 1957)")
    parser.add_argument("--tol",        type=float, required=True,
                        help="Energy tolerance for tsvdmii, e.g. 0.01")
    parser.add_argument("--mtype",      type=str,   default="dct",
                        choices=["dct", "eye"],
                        help="Transform type (default: dct)")
    parser.add_argument("--perm-mode",  type=str,   default="0123",
                        help="Mode permutation string (default: 0123)")
    parser.add_argument("--outdir",     type=str,   required=True,
                        help="Output base directory (run subdirectory created inside)")
    args = parser.parse_args()

    perm_mode = tuple(int(c) for c in args.perm_mode)
    threads   = os.environ.get("OMP_NUM_THREADS", "1")
    rname     = make_run_name(args.tol, args.mtype, perm_mode, threads)
    run_dir   = os.path.join(args.outdir, rname)

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

    # --- Permute and make Fortran-contiguous copy ---
    arr = np.transpose(arr, perm_mode)
    x = np.zeros(arr.shape, dtype=np.float64, order='F')
    for i in range(arr.shape[-1]):
        x[..., i] = arr[..., i]
    arr = x

    orig_shape    = arr.shape
    original_size = arr.size

    # --- Center by subtracting temporal mean ---
    # mean = arr.mean(axis=-1)        # shape (73, 144, 17)
    # arr  = arr - mean[..., None]

    # --- Diagnostic: energy split between mean and anomalies ---
    # norm_mean     = np.linalg.norm(mean[..., None] * np.ones_like(arr))
    # norm_anomaly  = np.linalg.norm(arr)
    # norm_total    = np.linalg.norm(arr + mean[..., None])
    # print(f"\nEnergy diagnostics:")
    # print(f"  norm(raw total)  : {norm_total:.4f}")
    # print(f"  norm(mean field) : {norm_mean:.4f}  ({100*norm_mean/norm_total:.2f}% of total)")
    # print(f"  norm(anomaly)    : {norm_anomaly:.4f}  ({100*norm_anomaly/norm_total:.2f}% of total)")

    # --- Build transform matrices ---
    ttm_modes = list(range(2, arr.ndim))
    Ms  = []
    MTs = []

    if args.mtype == "dct":
        for mode in ttm_modes:
            n   = arr.shape[mode]
            DF  = np.asfortranarray(dct(np.eye(n, dtype=np.float64), axis=0, norm="ortho"))
            DFT = np.asfortranarray(DF.T)
            Ms.append(pystarm.Matrix(DF,  DF.shape[0], DF.shape[1]))
            MTs.append(pystarm.Matrix(DFT, DFT.shape[0], DFT.shape[1]))
    elif args.mtype == "eye":
        ttm_modes = []  # identity transform is a no-op; skip TTM entirely

    # --- Compress ---
    A = pystarm.Tensor(arr, len(orig_shape), orig_shape)

    print(f"\nRunning tsvdmii (tol={args.tol}, mtype={args.mtype})...")
    t0 = time.perf_counter()
    (U_hat, S_hat, VT_hat) = tsvdm_II_compress(A, Ms, ttm_modes, args.tol, verbose=True)
    t1 = time.perf_counter()
    compress_time = t1 - t0

    compressed_size   = U_hat.getbuflen() + S_hat.getbuflen() + VT_hat.getbuflen()
    compression_ratio = original_size / compressed_size
    print(f"Compress time     : {compress_time:.1f}s")
    print(f"Compression ratio : {compression_ratio:.4f}x")

    # --- Reconstruct ---
    print("\nReconstructing...")
    t0 = time.perf_counter()
    Atilde = tsvdm_II_reconstruct(U_hat, S_hat, VT_hat, MTs, ttm_modes, orig_shape, verbose=True)
    t1 = time.perf_counter()
    reconstruct_time = t1 - t0
    print(f"Reconstruct time  : {reconstruct_time:.1f}s")

    arr_reconst = np.frombuffer(Atilde, dtype=np.float64).reshape(Atilde.getdims(), order='F', copy=False)

    # --- Compute relative error (should be ≈ tol) ---
    # rel_error_centered = np.linalg.norm(arr - arr_reconst) / np.linalg.norm(arr)
    # print(f"Relative error (centered)   : {rel_error_centered:.6f}  (tol={args.tol})")

    # --- Compressed size breakdown ---
    print(f"Compressed size breakdown:")
    print(f"  U_hat  : {U_hat.getbuflen()}")
    print(f"  S_hat  : {S_hat.getbuflen()}")
    print(f"  VT_hat : {VT_hat.getbuflen()}")

    # --- Add mean back to reconstruction ---
    # arr_reconst = arr_reconst + mean[..., None]

    # --- Compute global relative error ---
    # arr_orig  = arr + mean[..., None]
    rel_error = np.linalg.norm(arr - arr_reconst) / np.linalg.norm(arr)
    print(f"Relative error              : {rel_error:.6f}")

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
        "mtype":              args.mtype,
        "perm_mode":          list(perm_mode),
        "omp_threads":        threads,
        "tensor_shape":       list(orig_shape),
        "dtype":              "float64",
        "original_size":      original_size,
        "compressed_size":    compressed_size,
        "compression_ratio":  compression_ratio,
        "rel_error":          rel_error,
        "load_time_s":        load_time,
        "compress_time_s":    compress_time,
        "reconstruct_time_s": reconstruct_time,
        "save_time_s":        save_time,
    }
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {out_meta}")

    print("\nDone.")


if __name__ == "__main__":
    main()
