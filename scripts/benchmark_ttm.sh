#!/bin/bash
# benchmark_ttm.sh — Benchmark pystarm.ttm (batched BLAS) vs pystarm.ttm_loop
# (serial loop) across datasets and thread counts.
# Results are upserted into benchmark_ttm.csv.

export MKL_DYNAMIC=FALSE

OUTCSV="scripts/benchmark_ttm.csv"
NRUNS=5

#DATASETS=("ncep-air" "ncep-air-6" "cfd")
DATASETS=("ncep-air-6" "cfd")
#DATASETS=("cfd")
#THREADS=(64)
THREADS=(64 32 16 8 4 2 1)

for NTHREADS in "${THREADS[@]}"; do
    export OMP_NUM_THREADS=$NTHREADS
    export MKL_NUM_THREADS=$NTHREADS

    for DNAME in "${DATASETS[@]}"; do

        if [ "$DNAME" == "ncep-air" ]; then
            DFILE="/global/cfs/cdirs/m4293/taufique/NCEP-NCAR/pressure"
        elif [ "$DNAME" == "ncep-air-6" ]; then
            DFILE="/global/cfs/cdirs/m4293/taufique/NCEP-NCAR/pressure"
        elif [ "$DNAME" == "cfd" ]; then
            DFILE="/global/cfs/cdirs/m4293/starMData/"
        fi

        echo "=== DNAME=$DNAME NTHREADS=$NTHREADS ==="

        python scripts/benchmark_ttm.py \
            --dname $DNAME \
            --dfile $DFILE \
            --outcsv $OUTCSV \
            --nruns $NRUNS
    done

done

echo "Done. Results in $OUTCSV"
