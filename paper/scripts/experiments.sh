#!/bin/bash

# --- Switches ---
RUN_PYTHON=true
RUN_MATLAB=false

# --- Machine ---
MACHINE="nersc-perlmutter-cpu"
#MACHINE="alcf-aurora"

# --- Thread counts ---
THREADS=(64 32 16 8 4 2 1)

export MKL_DYNAMIC=FALSE

# --- Output ---
if [ "$MACHINE" == "nersc-perlmutter-cpu" ]; then
    OUTPUT_DIR="$SCRATCH/pystarm/logs"
elif [ "$MACHINE" == "alcf-aurora" ]; then
    OUTPUT_DIR="/lus/flare/projects/DFTCalc2/pystarm/logs"
fi
OUTPUT_PREFIX=""

# --- Data paths ---
if [ "$MACHINE" == "nersc-perlmutter-cpu" ]; then
    DFILE_SOCCER="data/iniesta.mp4"
    DFILE_TRAFFIC="data/traffic.bin"
    DFILE_DCMALL="$CFS/m4293/HSI/Hyperspectral_Project/dc.tif"
    DFILE_CFD="/global/cfs/cdirs/m4293/starMData/"
    DFILE_NCEP_AIR="/global/cfs/cdirs/m4293/taufique/NCEP-NCAR/pressure"
    DFILE_NCEP_AIR_6="/global/cfs/cdirs/m4293/taufique/NCEP-NCAR/pressure"
    DFILE_XRAY="/global/cfs/cdirs/m4293/xray/z3d_movo.npy"
elif [ "$MACHINE" == "alcf-aurora" ]; then
    DFILE_SOCCER="data/iniesta.mp4"
    DFILE_TRAFFIC="data/traffic.bin"
    DFILE_DCMALL=""
    DFILE_CFD="/lus/flare/projects/DFTCalc2/starM-HPC/starMData/cfd"
    DFILE_NCEP_AIR="/lus/flare/projects/DFTCalc2/starM-HPC/starMData/ncep/pressure"
    DFILE_NCEP_AIR_6="/lus/flare/projects/DFTCalc2/starM-HPC/starMData/ncep/pressure"
    DFILE_XRAY="/lus/flare/projects/DFTCalc2/starM-HPC/starMData/xray/z3d_movo.npy"
fi

# Path to MATLAB Tensor Toolbox — update this before running MATLAB experiments
TENSOR_TOOLBOX_PATH="/global/homes/t/taufique/Codes/tensor_toolbox-v3.8"

#MTYPES=("eye")
#MTYPES=("dct" "eye" "hosvd")
MTYPES=("dct")

NUM_RUNS=3

mkdir -p $OUTPUT_DIR

for NTHREADS in "${THREADS[@]}"; do
    export OMP_NUM_THREADS=$NTHREADS
    export MKL_NUM_THREADS=$NTHREADS

    #for DNAME in "soccer" "traffic-color" "traffic-gray" "dcmall" "cfd" "ncep-air" "ncep-air-6" "xray"; do
    for DNAME in "ncep-air" "ncep-air-6" "cfd" "xray"; do

        # --- Dataset definitions (shared by Python and MATLAB) ---
        if [ "$DNAME" == "soccer" ]; then
            DFILE="$DFILE_SOCCER"
            PERM_MODES=("012")
            K_VALUES=(5 10 20 40 80 160 320 640 1280)
        elif [ "$DNAME" == "traffic-color" ]; then
            DFILE="$DFILE_TRAFFIC"
            PERM_MODES=("0123" "0321")
            K_VALUES=(5 10 20 40)
        elif [ "$DNAME" == "traffic-gray" ]; then
            DFILE="$DFILE_TRAFFIC"
            #PERM_MODES=("012" "021" "120")
            PERM_MODES=("120")
            K_VALUES=(5 10 20 40)
        elif [ "$DNAME" == "cfd" ]; then
            DFILE="$DFILE_CFD"
            PERM_MODES=("01234")
            K_VALUES=(1 2 3 4 5 10 20 40 80 160)
        elif [ "$DNAME" == "dcmall" ]; then
            DFILE="$DFILE_DCMALL"
            PERM_MODES=("021")
            K_VALUES=(5 10 20 40 80 120 160 200 250)
        elif [ "$DNAME" == "ncep-air" ]; then
            DFILE="$DFILE_NCEP_AIR"
            PERM_MODES=("0123")
            K_VALUES=(1 2 3 4 5 10 20 40)
            K_MAX=1000
        elif [ "$DNAME" == "ncep-air-6" ]; then
            DFILE="$DFILE_NCEP_AIR_6"
            PERM_MODES=("012345")
            K_VALUES=(1 2 3 4 5 10 20 40)
        elif [ "$DNAME" == "xray" ]; then
            DFILE="$DFILE_XRAY"
            PERM_MODES=("012")
            K_VALUES=(5 10 20 40 80 100 150 200 250)
        fi

        for PERM_MODE in "${PERM_MODES[@]}"; do

            # --- Python experiments ---
            if [ "$RUN_PYTHON" == "true" ]; then
                for ALG in "tsvdmi" "tsvdmii"; do
                #for ALG in "tsvdmii"; do
                #for ALG in "tsvdmi"; do
                #for ALG in "eof"; do

                    if [ "$ALG" == "tsvdmi" ]; then
                        for K in "${K_VALUES[@]}"; do
                            for MTYPE in "${MTYPES[@]}"; do
                                for RUN_ID in $(seq 1 $NUM_RUNS); do
                                    LOGFILE="${OUTPUT_DIR}/${OUTPUT_PREFIX}${DNAME}_${ALG}_${K}_${MTYPE}_${PERM_MODE}_${OMP_NUM_THREADS}_run${RUN_ID}"
                                    echo "Running DNAME=$DNAME ALG=$ALG K=$K MTYPE=$MTYPE PERM_MODE=$PERM_MODE THREADS=$OMP_NUM_THREADS RUN=$RUN_ID -> $LOGFILE"
                                    python experiments.py -alg $ALG -mtype $MTYPE -k $K -dname $DNAME -dfile $DFILE -perm-mode $PERM_MODE > $LOGFILE 2>&1
                                done
                            done
                        done

                    elif [ "$ALG" == "tsvdmii" ]; then
                        for TOL in 0.0001 0.001 0.01 0.025 0.050 0.1 0.25; do
                        #for TOL in 0.002 0.003 0.005 0.007; do
                        #for TOL in 0.008; do
                        #for TOL in 0.1; do
                        #for TOL in 0.0082 0.0085 0.0088 0.009 0.0092 0.0095 0.0098; do
                            for MTYPE in "${MTYPES[@]}"; do
                                for RUN_ID in $(seq 1 $NUM_RUNS); do
                                    LOGFILE="${OUTPUT_DIR}/${OUTPUT_PREFIX}${DNAME}_${ALG}_${TOL}_${MTYPE}_${PERM_MODE}_${OMP_NUM_THREADS}_run${RUN_ID}"
                                    echo "Running DNAME=$DNAME ALG=$ALG TOL=$TOL MTYPE=$MTYPE PERM_MODE=$PERM_MODE THREADS=$OMP_NUM_THREADS RUN=$RUN_ID -> $LOGFILE"
                                    python experiments.py -alg $ALG -mtype $MTYPE -tol $TOL -dname $DNAME -dfile $DFILE -perm-mode $PERM_MODE > $LOGFILE 2>&1
                                done
                            done
                        done

                    elif [ "$ALG" == "eof" ]; then
                        for TOL in 0.0001 0.001 0.01 0.025 0.050 0.1 0.25; do
                            for RUN_ID in $(seq 1 $NUM_RUNS); do
                                LOGFILE="${OUTPUT_DIR}/${OUTPUT_PREFIX}${DNAME}_${ALG}_${TOL}_none_none_${OMP_NUM_THREADS}_run${RUN_ID}"
                                echo "Running DNAME=$DNAME ALG=$ALG TOL=$TOL K_MAX=${K_MAX:-1000} THREADS=$OMP_NUM_THREADS RUN=$RUN_ID -> $LOGFILE"
                                python experiments.py -alg $ALG -tol $TOL -k-max ${K_MAX:-1000} -dname $DNAME -dfile $DFILE > $LOGFILE 2>&1
                            done
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

done
