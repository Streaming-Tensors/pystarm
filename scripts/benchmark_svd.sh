#!/bin/bash
# benchmark_svd.sh — Benchmark slicewise_svd (parfor) vs slicewise_svd_seq
# (sequential MKL) across datasets and thread counts.
# Results are upserted into benchmark_svd.csv.

export MKL_DYNAMIC=FALSE

OUTCSV="scripts/benchmark_svd.csv"
NRUNS=5

DATASETS=("ncep-air-6" "cfd")
THREADS=(64 32 16 8 4 2 1)

for DNAME in "${DATASETS[@]}"; do

    if [ "$DNAME" == "ncep-air-6" ]; then
        DFILE="/global/cfs/cdirs/m4293/taufique/NCEP-NCAR/pressure"
    elif [ "$DNAME" == "cfd" ]; then
        DFILE="/global/cfs/cdirs/m4293/starMData/"
    fi

    for NTHREADS in "${THREADS[@]}"; do
        export OMP_NUM_THREADS=$NTHREADS
        export MKL_NUM_THREADS=$NTHREADS
        echo "=== DNAME=$DNAME NTHREADS=$NTHREADS ==="

        python scripts/benchmark_svd.py \
            --dname $DNAME \
            --dfile $DFILE \
            --outcsv $OUTCSV \
            --nruns $NRUNS
    done

done

echo "Done. Results in $OUTCSV"
