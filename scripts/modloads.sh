# This is made for the Bebop systems at ANL. 
# Loads all necessary modules for compiling and running
# Test scripts.

#! /usr/bin/bash

module load miniforge
module load gcc
module load intel-oneapi-mkl

conda activate starm-env

# Export some necessary parameters

export MKL_NUM_THREADS=36
export OMP_NUM_THREADS=36
