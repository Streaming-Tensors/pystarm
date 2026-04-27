#!/bin/bash
# benchmark_svd.sh — Benchmark slicewise_svd (parfor) vs slicewise_svd_seq
# (sequential MKL) across datasets and thread counts.
# Results are upserted into benchmark_svd_${MACHINE}.csv.

export MKL_DYNAMIC=FALSE

# --- Machine ---
MACHINE="nersc-perlmutter-cpu"

# --- Output ---
OUTCSV="scripts/benchmark_svd_${MACHINE}.csv"

# --- Datasets ---
#DATASETS=("ncep-air-6" "cfd")
DATASETS=("xray")

# --- Data paths (update for each machine) ---
DFILE_NCEP_AIR_6="/global/cfs/cdirs/m4293/taufique/NCEP-NCAR/pressure"
DFILE_CFD="/global/cfs/cdirs/m4293/starMData/"
DFILE_XRAY="/global/cfs/cdirs/m4293/xray/z3d_movo.npy"

# --- Thread counts ---
THREADS=(64 32 16 8 4 2 1)

# --- Number of runs ---
# Default applies to all thread counts unless a per-thread override is set.
# To override for a specific thread count, set NRUNS_<N> (e.g. NRUNS_1=1).
# Leave empty to use NRUNS_DEFAULT.
NRUNS_DEFAULT=5
NRUNS_64=3
NRUNS_32=3
NRUNS_16=3
NRUNS_8=3
NRUNS_4=3
NRUNS_2=1
NRUNS_1=1

for DNAME in "${DATASETS[@]}"; do

    if [ "$DNAME" == "ncep-air-6" ]; then
        DFILE="$DFILE_NCEP_AIR_6"
    elif [ "$DNAME" == "cfd" ]; then
        DFILE="$DFILE_CFD"
    elif [ "$DNAME" == "xray" ]; then
        DFILE="$DFILE_XRAY"
    fi

    for NTHREADS in "${THREADS[@]}"; do
        export OMP_NUM_THREADS=$NTHREADS
        export MKL_NUM_THREADS=$NTHREADS

        VAR_THREAD="NRUNS_${NTHREADS}"
        NRUNS="${!VAR_THREAD:-$NRUNS_DEFAULT}"

        echo "=== DNAME=$DNAME NTHREADS=$NTHREADS NRUNS=$NRUNS ==="

        python scripts/benchmark_svd.py \
            --dname $DNAME \
            --dfile $DFILE \
            --outcsv $OUTCSV \
            --nruns $NRUNS
    done

done

echo "Done. Results in $OUTCSV"
