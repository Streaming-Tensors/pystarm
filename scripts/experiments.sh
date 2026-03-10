#!/bin/bash

export OMP_NUM_THREADS=64
OUTPUT_DIR="$SCRATCH/pystarm"
OUTPUT_PREFIX=""

#MTYPES=("eye")
MTYPES=("dct" "eye" "hosvd")

mkdir -p $OUTPUT_DIR

for ALG in "tsvdmi" "tsvdmii"; do
#for ALG in "tsvdmii"; do

    for DNAME in "traffic"; do
    #for DNAME in "soccer" "traffic" "cfd"; do

        if [ "$DNAME" == "soccer" ]; then
            DFILE="data/iniesta.mp4"
            PERM_MODES=("012")
        elif [ "$DNAME" == "traffic" ]; then
            DFILE="data/traffic.bin"
            PERM_MODES=("0123" "0321")
        elif [ "$DNAME" == "cfd" ]; then
            DFILE="/global/cfs/cdirs/m4293/starMData/"
            PERM_MODES=("01234")
        fi

        for PERM_MODE in "${PERM_MODES[@]}"; do

            if [ "$DNAME" == "soccer" ]; then
                if [ "$PERM_MODE" == "012" ]; then
                    K_VALUES=(5 10 20 40 80 160 320 640 1280)
                fi
            elif [ "$DNAME" == "traffic" ]; then
                if [ "$PERM_MODE" == "0123" ] || [ "$PERM_MODE" == "0321" ]; then
                    K_VALUES=(5 10 20 40)
                fi
            elif [ "$DNAME" == "cfd" ]; then
                if [ "$PERM_MODE" == "01234" ]; then
                    K_VALUES=(5 10 20 40 80)
                fi
            fi

            if [ "$ALG" == "tsvdmi" ]; then
                for K in "${K_VALUES[@]}"; do
                    for MTYPE in "${MTYPES[@]}"; do
                        LOGFILE="${OUTPUT_DIR}/${OUTPUT_PREFIX}${DNAME}_${ALG}_${K}_${MTYPE}_${PERM_MODE}_${OMP_NUM_THREADS}"
                        echo "Running DNAME=$DNAME ALG=$ALG K=$K MTYPE=$MTYPE PERM_MODE=$PERM_MODE -> $LOGFILE"
                        python experiments.py -alg $ALG -mtype $MTYPE -k $K -dname $DNAME -dfile $DFILE -perm-mode $PERM_MODE > $LOGFILE 2>&1
                    done
                done

            elif [ "$ALG" == "tsvdmii" ]; then
                for TOL in 0.0001 0.001 0.01 0.025 0.050 0.1 0.25 0.5; do
                #for TOL in 0.1 0.25 0.5; do
                    for MTYPE in "${MTYPES[@]}"; do
                        LOGFILE="${OUTPUT_DIR}/${OUTPUT_PREFIX}${DNAME}_${ALG}_${TOL}_${MTYPE}_${PERM_MODE}_${OMP_NUM_THREADS}"
                        echo "Running DNAME=$DNAME ALG=$ALG TOL=$TOL MTYPE=$MTYPE PERM_MODE=$PERM_MODE -> $LOGFILE"
                        python experiments.py -alg $ALG -mtype $MTYPE -tol $TOL -dname $DNAME -dfile $DFILE -perm-mode $PERM_MODE > $LOGFILE 2>&1
                    done
                done
            fi

        done
    done

done
