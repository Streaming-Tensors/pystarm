#!/bin/bash

# ncep-air_extreme.sh
# Run the full extreme event analysis pipeline for NCEP air temperature:
#   1. Compress + reconstruct with tsvdmii
#   2. Compress + reconstruct with EOF
#   3. Compute extreme error maps and produce cartopy plots

# --- Settings ---
NTHREADS=64
export OMP_NUM_THREADS=$NTHREADS
export MKL_NUM_THREADS=$NTHREADS
export MKL_DYNAMIC=FALSE

DATA_DIR=/global/cfs/cdirs/m4293/taufique/NCEP-NCAR/pressure
YEAR_START=1948
YEAR_END=1957
TOL_TSVDMII=0.008
TOL_EOF=0.01
MTYPE=dct
PERM_MODE=0123
K_MAX=1000
LEVEL_HPA=850
METRIC=median
OUTDIR=/pscratch/sd/t/taufique/pystarm/extremes

TSVDMII_DIR=${OUTDIR}/ncep-air_tsvdmii_${TOL_TSVDMII}_${MTYPE}_${PERM_MODE}_${NTHREADS}
EOF_DIR=${OUTDIR}/ncep-air_eof_${TOL_EOF}_${K_MAX}_${NTHREADS}

# --- Step 1: tsvdmii reconstruction ---
echo "=== Step 1: tsvdmii reconstruction ==="
python scripts/ncep/ncep-air_tsvdmii.py \
    --data-dir   $DATA_DIR \
    --year-start $YEAR_START \
    --year-end   $YEAR_END \
    --tol        $TOL_TSVDMII \
    --mtype      $MTYPE \
    --perm-mode  $PERM_MODE \
    --outdir     $OUTDIR

# --- Step 2: EOF reconstruction ---
echo "=== Step 2: EOF reconstruction ==="
python scripts/ncep/ncep-air_eof.py \
    --data-dir   $DATA_DIR \
    --year-start $YEAR_START \
    --year-end   $YEAR_END \
    --tol        $TOL_EOF \
    --k-max      $K_MAX \
    --outdir     $OUTDIR

# --- Step 3: Extreme error maps and cartopy plots ---
echo "=== Step 3: Extreme error maps and plots ==="
python scripts/ncep/ncep-air_extremes.py \
    --data-dir    $DATA_DIR \
    --year-start  $YEAR_START \
    --year-end    $YEAR_END \
    --tsvdmii-dir $TSVDMII_DIR \
    --eof-dir     $EOF_DIR \
    --level-hpa   $LEVEL_HPA \
    --metric      $METRIC \
    --outdir      $OUTDIR

echo "=== Done. Output in $OUTDIR ==="
