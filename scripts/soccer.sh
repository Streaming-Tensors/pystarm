#!/bin/bash

export OMP_NUM_THREADS=64
OUTPUT_DIR="$SCRATCH/pystarm"
OUTPUT_PREFIX=""

MTYPES=("eye")
#MTYPES=("dct" "eye" "hosvd")

mkdir -p $OUTPUT_DIR

#for ALG in "tsvdmi" "tsvdmii"; do
for ALG in "tsvdmii"; do

    if [ "$ALG" == "tsvdmi" ]; then
        for K in 5 10 20 40 80 160 320 640 1280; do
            for MTYPE in "${MTYPES[@]}"; do
                LOGFILE="${OUTPUT_DIR}/${OUTPUT_PREFIX}${ALG}_${K}_${MTYPE}_${OMP_NUM_THREADS}"
                echo "Running ALG=$ALG K=$K MTYPE=$MTYPE -> $LOGFILE"
                python soccer.py -alg $ALG -mtype $MTYPE -k $K > $LOGFILE 2>&1
            done
        done

    elif [ "$ALG" == "tsvdmii" ]; then
        #for TOL in 0.0001 0.001 0.01 0.025 0.050 0.1 0.25 0.5; do
        for TOL in 0.1 0.25 0.5; do
            for MTYPE in "${MTYPES[@]}"; do
                LOGFILE="${OUTPUT_DIR}/${OUTPUT_PREFIX}${ALG}_${TOL}_${MTYPE}_${OMP_NUM_THREADS}"
                echo "Running ALG=$ALG TOL=$TOL MTYPE=$MTYPE -> $LOGFILE"
                python soccer.py -alg $ALG -mtype $MTYPE -tol $TOL > $LOGFILE 2>&1
            done
        done
    fi

done
