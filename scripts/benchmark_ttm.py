"""
benchmark_ttm.py — Benchmark pystarm.ttm (batched BLAS) vs pystarm.ttm_loop
(serial loop) for each TTM mode on real datasets.

For each mode, earlier modes are pre-applied first so the input tensor matches
what the real compress pipeline would see at that point.

Usage:
    python scripts/benchmark_ttm.py \
        --dname ncep-air \
        --dfile /path/to/data \
        --outcsv scripts/benchmark_ttm.csv \
        --nruns 5

Thread count is read from OMP_NUM_THREADS environment variable.
Run via benchmark_ttm.sh to sweep over thread counts automatically.
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
from scipy.fft import dct

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pystarm
from experiments import (
    read_ncep_air, read_ncep_air_6, read_cfd_data
)


# ---------------------------------------------------------------------------
# CSV upsert helpers
# ---------------------------------------------------------------------------

COLUMNS = ['dname', 'mode', 'ttm_variant', 'threads', 'run_id', 'time_sec']
KEY_COLS = ('dname', 'mode', 'ttm_variant', 'threads', 'run_id')


def load_csv(path):
    rows = {}
    if not os.path.exists(path):
        return rows
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            key = tuple(row[c] for c in KEY_COLS)
            rows[key] = row
    return rows


def save_csv(path, rows):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows.values())


def upsert(rows, record):
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
    parser.add_argument('--outcsv', default=os.path.join(os.path.dirname(__file__), 'benchmark_ttm.csv'),
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
    A_input = pystarm.Tensor(arr, arr.ndim, list(arr.shape))
    print(f"Tensor shape: {arr.shape}")

    # --- Build DCT transform matrices ---
    ttm_modes = list(range(2, arr.ndim))
    Ms = []
    for mode in ttm_modes:
        n  = arr.shape[mode]
        DF = np.asfortranarray(dct(np.eye(n, dtype=np.float64), axis=0, norm='ortho'))
        Ms.append(pystarm.Matrix(DF, DF.shape[0], DF.shape[1]))

    rows = load_csv(args.outcsv)

    variants = [
        ('batched', pystarm.ttm),
        ('loop',    pystarm.ttm_loop),
    ]

    for variant_name, fn in variants:
        print(f"\n=== variant='{variant_name}' ===")
        for run_id in range(1, args.nruns + 1):
            print(f"\n--- run {run_id}:  ---")
            A_hat = None
            for i, mode in enumerate(ttm_modes):
                upsert(rows, {
                    'dname':       args.dname,
                    'mode':        mode,
                    'ttm_variant': variant_name,
                    'threads':     threads,
                    'run_id':      run_id,
                    'time_sec':    -1,
                })
                save_csv(args.outcsv, rows)

                t0 = time.perf_counter()
                if i == 0:
                    A_hat_temp = fn(A_input, Ms[i], mode)
                else:
                    A_hat_temp = fn(A_hat, Ms[i], mode)
                t1 = time.perf_counter()
                if A_hat is not None:
                    A_hat.clear()
                A_hat = A_hat_temp

                elapsed = t1 - t0
                print(f"    Mode {mode}:{elapsed:.4f}s")

                upsert(rows, {
                    'dname':       args.dname,
                    'mode':        mode,
                    'ttm_variant': variant_name,
                    'threads':     threads,
                    'run_id':      run_id,
                    'time_sec':    elapsed,
                })
                save_csv(args.outcsv, rows)

            if A_hat is not None:
                A_hat.clear()

    save_csv(args.outcsv, rows)
    print(f"\nResults written to {args.outcsv}")


if __name__ == '__main__':
    main()
