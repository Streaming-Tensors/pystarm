#!/bin/bash
#PBS -N experiments
#PBS -A DFTCalc2
#PBS -q prod
#PBS -l select=1
#PBS -l walltime=12:00:00
#PBS -l filesystems=home:flare

cd /home/mth/Codes/pystarm
source scripts/alcf-env-setup.sh
bash scripts/experiments.sh
