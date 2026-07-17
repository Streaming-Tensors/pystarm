#!/bin/bash
#PBS -N benchmark_svd
#PBS -A DFTCalc2
#PBS -q prod
#PBS -l select=1
#PBS -l walltime=12:00:00
#PBS -l filesystems=home:flare

cd /home/mth/Codes/pystarm
source scripts/alcf-env-setup.sh
bash scripts/benchmark_svd.sh
