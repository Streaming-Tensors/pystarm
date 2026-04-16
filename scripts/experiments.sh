#!/bin/bash

# --- Switches ---
RUN_PYTHON=true
RUN_MATLAB=false

NTHREADS=32
#export OMP_NUM_THREADS=64
export MKL_NUM_THREADS=$NTHREADS
export OMP_NUM_THREADS=$NTHREADS
export MKL_DYNAMIC=FALSE

OUTPUT_DIR="$SCRATCH/pystarm/logs"
OUTPUT_PREFIX=""

# Path to MATLAB Tensor Toolbox — update this before running MATLAB experiments
TENSOR_TOOLBOX_PATH="/global/homes/t/taufique/Codes/tensor_toolbox-v3.8"

#MTYPES=("eye")
#MTYPES=("dct" "eye" "hosvd")
MTYPES=("dct")

mkdir -p $OUTPUT_DIR

#for DNAME in "soccer" "traffic-color" "traffic-gray" "dcmall" "cfd" "ncep-air" "ncep-air-6" "ncep-slp"; do
for DNAME in "ncep-air-6"; do

    # --- Dataset definitions (shared by Python and MATLAB) ---
    if [ "$DNAME" == "soccer" ]; then
        DFILE="data/iniesta.mp4"
        PERM_MODES=("012")
        K_VALUES=(5 10 20 40 80 160 320 640 1280)
    elif [ "$DNAME" == "traffic-color" ]; then
        DFILE="data/traffic.bin"
        PERM_MODES=("0123" "0321")
        K_VALUES=(5 10 20 40)
    elif [ "$DNAME" == "traffic-gray" ]; then
        DFILE="data/traffic.bin"
        #PERM_MODES=("012" "021" "120")
        PERM_MODES=("120")
        K_VALUES=(5 10 20 40)
    elif [ "$DNAME" == "cfd" ]; then
        DFILE="/global/cfs/cdirs/m4293/starMData/"
        PERM_MODES=("01234")
        K_VALUES=(5 10 20 40 80)
    elif [ "$DNAME" == "dcmall" ]; then
        DFILE="$CFS/m4293/HSI/Hyperspectral_Project/dc.tif"
        PERM_MODES=("021")
        K_VALUES=(5 10 20 40 80 120 160 200 250)
    elif [ "$DNAME" == "ncep-air" ]; then
        DFILE="/global/cfs/cdirs/m4293/taufique/NCEP-NCAR/pressure"
        PERM_MODES=("0123")
        K_VALUES=(5 10 20 40)
        K_MAX=1000
    elif [ "$DNAME" == "ncep-air-6" ]; then
        DFILE="/global/cfs/cdirs/m4293/taufique/NCEP-NCAR/pressure"
        PERM_MODES=("012345")
        K_VALUES=(5 10 20 40)
    elif [ "$DNAME" == "ncep-slp" ]; then
        DFILE="/global/cfs/cdirs/m4293/taufique/NCEP-NCAR/surface"
        PERM_MODES=("012")
        K_VALUES=(5 10 20 40)
        K_MAX=1000
    fi

    for PERM_MODE in "${PERM_MODES[@]}"; do

        # --- Python experiments ---
        if [ "$RUN_PYTHON" == "true" ]; then
            #for ALG in "tsvdmi" "tsvdmii" "eof"; do
            #for ALG in "tsvdmii"; do
            for ALG in "tsvdmi"; do
            #for ALG in "eof"; do

                if [ "$ALG" == "tsvdmi" ]; then
                    for K in "${K_VALUES[@]}"; do
                        for MTYPE in "${MTYPES[@]}"; do
                            LOGFILE="${OUTPUT_DIR}/${OUTPUT_PREFIX}${DNAME}_${ALG}_${K}_${MTYPE}_${PERM_MODE}_${OMP_NUM_THREADS}"
                            echo "Running DNAME=$DNAME ALG=$ALG K=$K MTYPE=$MTYPE PERM_MODE=$PERM_MODE -> $LOGFILE"
                            python experiments.py -alg $ALG -mtype $MTYPE -k $K -dname $DNAME -dfile $DFILE -perm-mode $PERM_MODE > $LOGFILE 2>&1
                        done
                    done

                elif [ "$ALG" == "tsvdmii" ]; then
                    for TOL in 0.0001 0.001 0.01 0.025 0.050 0.1 0.25; do
                    #for TOL in 0.01; do
                    #for TOL in 0.1; do
                        for MTYPE in "${MTYPES[@]}"; do
                            LOGFILE="${OUTPUT_DIR}/${OUTPUT_PREFIX}${DNAME}_${ALG}_${TOL}_${MTYPE}_${PERM_MODE}_${OMP_NUM_THREADS}"
                            echo "Running DNAME=$DNAME ALG=$ALG TOL=$TOL MTYPE=$MTYPE PERM_MODE=$PERM_MODE -> $LOGFILE"
                            python experiments.py -alg $ALG -mtype $MTYPE -tol $TOL -dname $DNAME -dfile $DFILE -perm-mode $PERM_MODE > $LOGFILE 2>&1
                        done
                    done

                elif [ "$ALG" == "eof" ]; then
                    for TOL in 0.0001 0.001 0.01 0.025 0.050 0.1 0.25; do
                        LOGFILE="${OUTPUT_DIR}/${OUTPUT_PREFIX}${DNAME}_${ALG}_${TOL}_none_none_${OMP_NUM_THREADS}"
                        echo "Running DNAME=$DNAME ALG=$ALG TOL=$TOL K_MAX=${K_MAX:-1000} -> $LOGFILE"
                        python experiments.py -alg $ALG -tol $TOL -k-max ${K_MAX:-1000} -dname $DNAME -dfile $DFILE > $LOGFILE 2>&1
                    done
                fi

            done
        fi

        # --- MATLAB HOSVD experiments ---
        if [ "$RUN_MATLAB" == "true" ]; then
            #for TOL in 0.0001 0.001 0.01 0.025 0.050 0.1 0.25 0.5; do
            for TOL in 0.01; do
                LOGFILE="${OUTPUT_DIR}/${DNAME}_hosvd_${TOL}_eye_${PERM_MODE}_1"
                echo "Running MATLAB DNAME=$DNAME TOL=$TOL PERM_MODE=$PERM_MODE -> $LOGFILE"
                matlab -nodisplay -nosplash -r \
                    "addpath('${TENSOR_TOOLBOX_PATH}'); alg_str='hosvd'; tol=${TOL}; perm_str='${PERM_MODE}'; dname_str='${DNAME}'; dfile_str='${DFILE}'; run('scripts/hosvd_experiment.m'); exit" \
                    > $LOGFILE 2>&1
            done

            for P in 0.125 0.175 0.225 0.275 0.325 0.375 0.425 0.475; do
                LOGFILE="${OUTPUT_DIR}/${DNAME}_hosvd-proportional_${P}_eye_${PERM_MODE}_1"
                echo "Running MATLAB hosvd-proportional DNAME=$DNAME P=$P PERM_MODE=$PERM_MODE -> $LOGFILE"
                matlab -nodisplay -nosplash -r \
                    "addpath('${TENSOR_TOOLBOX_PATH}'); alg_str='hosvd-proportional'; p=${P}; perm_str='${PERM_MODE}'; dname_str='${DNAME}'; dfile_str='${DFILE}'; run('scripts/hosvd_experiment.m'); exit" \
                    > $LOGFILE 2>&1
            done
        fi

    done

done
