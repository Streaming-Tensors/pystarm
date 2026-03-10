#!/bin/bash
# asan-run-prep.sh
# Sets up the environment for running Python scripts that use the ASAN build of pystarm.
# Must be sourced (not executed) so the variables are set in the current shell:
#   source scripts/asan-run-prep.sh

# Selects 64-bit integer interface for dynamic MKL (matches -DMKL_ILP64 used at compile time)
export MKL_INTERFACE_LAYER=ILP64

# Selects GNU OpenMP threading backend for dynamic MKL (matches -fopenmp / libgomp used at compile time)
export MKL_THREADING_LAYER=GNU

# Adds MKL's dynamic library directory to the runtime library search path
# so the OS dynamic loader can find libmkl_rt.so.2 when pystarm_asan is imported
export LD_LIBRARY_PATH=$MKLROOT/lib/intel64:$LD_LIBRARY_PATH

# Ensures the ASAN runtime is loaded before Python, MKL, and OpenMP
# ASAN must be the first allocator in the process to intercept all malloc/free calls
# Find the path on your system with: gcc -print-file-name=libasan.so
export LD_PRELOAD=/usr/lib64/gcc/x86_64-suse-linux/14/libasan.so

# Limits OpenMP to 4 threads for ASAN runs to keep output manageable
# In an OpenMP parallel region, multiple threads may hit the same bug simultaneously,
# each triggering their own ASAN error report — fewer threads means fewer concurrent
# reports and easier to read output
export OMP_NUM_THREADS=4

# Tunes ASAN runtime behavior:
# quarantine_size_mb=1  : shrinks the freed-memory quarantine from the default (~256 MB) to 1 MB,
#                         so ASAN recycles chunks sooner and detects overflows closer to where they happen
# malloc_context_size=30: captures 30 stack frames in allocation/free records (default is 30, made explicit)
#                         giving a fuller call stack when an error is detected
export ASAN_OPTIONS=quarantine_size_mb=1:malloc_context_size=30
