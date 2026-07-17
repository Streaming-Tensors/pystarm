#!/bin/bash
#SBATCH --job-name=benchmark_svd
#SBATCH --account=m4293
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=23:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

bash scripts/benchmark_svd.sh
