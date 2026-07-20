module load python
conda activate pystarm
export MKLROOT=/global/common/software/nersc9/intel/oneapi/mkl/2024.1
export MKL_NUM_THREADS=64
export OMP_NUM_THREADS=64
export MKL_DYNAMIC=FALSE
