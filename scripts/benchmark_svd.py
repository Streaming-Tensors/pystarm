"""
benchmark_svd.py — Benchmark slicewise_svd (parfor + sequential dgesvd) vs
slicewise_svd_seq (sequential loop + MKL-threaded dgesvd).

Usage:
    python scripts/benchmark_svd.py \
        --dname ncep-air \
        --dfile /path/to/data \
        --outcsv scripts/benchmark_svd.csv \
        --nruns 5

Thread count is read from OMP_NUM_THREADS environment variable.
Run via benchmark_svd.sh to sweep over thread counts automatically.
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
from scipy.fft import dct

# Allow importing from the code root (experiments.py lives there)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pystarm
from experiments import (
    read_ncep_air, read_ncep_air_6, read_cfd_data
)


# ---------------------------------------------------------------------------
# CSV upsert helpers
# ---------------------------------------------------------------------------

COLUMNS = ['dname', 'svd_variant', 'threads', 'run_id', 'time_sec']
KEY_COLS = ('dname', 'svd_variant', 'threads', 'run_id')


def load_csv(path):
    """Load existing CSV into a dict keyed by KEY_COLS tuple."""
    rows = {}
    if not os.path.exists(path):
        return rows
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            key = tuple(row[c] for c in KEY_COLS)
            rows[key] = row
    return rows


def save_csv(path, rows):
    """Write all rows to CSV."""
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows.values())


def upsert(rows, record):
    """Insert or update a record in the rows dict."""
    key = tuple(str(record[c]) for c in KEY_COLS)
    rows[key] = {c: str(record[c]) for c in COLUMNS}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(dname, dfile):
    if dname == 'ncep-air':
        arr = read_ncep_air(dfile, 'air', 1948, 1957)
    elif dname == 'ncep-air-6':
        arr = read_ncep_air_6(dfile, 'air', 1948, 1957)
    elif dname == 'cfd':
        arr = read_cfd_data(dfile)
    else:
        raise ValueError(f"Unknown dname: {dname}")

    # Ensure Fortran-order float64
    x = np.zeros(arr.shape, dtype=np.float64, order='F')
    for i in range(arr.shape[-1]):
        x[..., i] = arr[..., i]
    return x


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dname',  required=True,  help='Dataset name: ncep-air, ncep-air-6, cfd')
    parser.add_argument('--dfile',  required=True,  help='Path to data directory')
    parser.add_argument('--outcsv', default=os.path.join(os.path.dirname(__file__), 'benchmark_svd.csv'),
                        help='Output CSV path')
    parser.add_argument('--nruns',  type=int, default=5, help='Number of timed repetitions')
    args = parser.parse_args()

    threads = int(os.environ.get('OMP_NUM_THREADS', 1))

    print(f"dname   : {args.dname}")
    print(f"dfile   : {args.dfile}")
    print(f"threads : {threads}")
    print(f"nruns   : {args.nruns}")

    # --- Load data ---
    print("Loading data...")
    arr = load_data(args.dname, args.dfile)
    print(f"Tensor shape: {arr.shape}")

    # --- Build DCT transform matrices ---
    ttm_modes = list(range(2, arr.ndim))
    Ms = []
    for mode in ttm_modes:
        n  = arr.shape[mode]
        DF = np.asfortranarray(dct(np.eye(n, dtype=np.float64), axis=0, norm='ortho'))
        Ms.append(pystarm.Matrix(DF, DF.shape[0], DF.shape[1]))

    # --- Convert to pystarm Tensor ---
    A = pystarm.Tensor(arr, arr.ndim, list(arr.shape))

    # --- Apply TTM to get A_hat (not benchmarked) ---
    print("Applying TTM transforms...")
    A_hat = None
    for i, mode in enumerate(ttm_modes):
        if i == 0:
            A_hat_temp = pystarm.ttm(A, Ms[i], mode)
        else:
            A_hat_temp = pystarm.ttm(A_hat, Ms[i], mode)
        if A_hat is not None:
            A_hat.clear()
        A_hat = A_hat_temp
    A.clear()
    for M in Ms:
        M.clear()

    if A_hat is None:
        raise RuntimeError("No TTM modes applied — tensor must have ndim >= 3")

    print(f"A_hat shape: {A_hat.getdims()}")

    # --- Benchmark ---
    variants = [
        ('parfor', pystarm.slicewise_svd),
        ('seq',    pystarm.slicewise_svd_seq),
    ]

    rows = load_csv(args.outcsv)

    for variant_name, fn in variants:
        print(f"\nBenchmarking slicewise_svd variant='{variant_name}'...")
        for run_id in range(1, args.nruns + 1):
            # Write sentinel -1 before the run; overwritten with actual time on success.
            # If the run crashes, -1 remains and identifies the failed run.
            upsert(rows, {
                'dname':       args.dname,
                'svd_variant': variant_name,
                'threads':     threads,
                'run_id':      run_id,
                'time_sec':    -1,
            })
            save_csv(args.outcsv, rows)

            t0 = time.perf_counter()
            U, S, Vt = fn(A_hat)
            t1 = time.perf_counter()
            U.clear(); S.clear(); Vt.clear()

            elapsed = t1 - t0
            print(f"  run {run_id}: {elapsed:.4f}s")

            upsert(rows, {
                'dname':       args.dname,
                'svd_variant': variant_name,
                'threads':     threads,
                'run_id':      run_id,
                'time_sec':    elapsed,
            })
            save_csv(args.outcsv, rows)

    A_hat.clear()
    print(f"\nResults written to {args.outcsv}")


if __name__ == '__main__':
    main()
