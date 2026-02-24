## Following: https://pybind11.readthedocs.io/en/stable/compiling.html#building-manually
## Comments generated with claude-code

## Base compiler flags shared between production and ASAN builds
# -Wall      : enable all compiler warnings
# -shared    : produce a shared library (.so) instead of an executable
# -std=c++11 : use C++11 standard
# -fPIC      : position independent code, required for shared libraries
# -fopenmp   : enable OpenMP for multi-threaded parallelism
CFLAGS_BASE = -Wall -shared -std=c++11 -fPIC -fopenmp

## Production compiler flags
# -O3 : maximum optimization level for best runtime performance
CFLAGS = $(CFLAGS_BASE) -O3

## ASAN compiler flags
# -O0                    : disable optimizations so ASAN reports map exactly to source lines
# -fsanitize=address     : instruments every memory read/write for overflow/use-after-free detection
# -g                     : includes debug symbols so ASAN reports show file names and line numbers
# -fno-omit-frame-pointer: keeps stack frame pointers so ASAN can unwind the call stack accurately
CFLAGS_ASAN = $(CFLAGS_BASE) -O0 -fsanitize=address -g -fno-omit-frame-pointer

## Production MKL static linking flags
# -m64                      : target 64-bit architecture, must match MKL_ILP64
# --start-group/--end-group : resolves circular dependencies between the three MKL static libraries
# libmkl_intel_ilp64.a      : MKL core routines with 64-bit integer interface
# libmkl_gnu_thread.a       : MKL threading layer using GNU OpenMP runtime (matches -fopenmp)
# libmkl_core.a             : MKL computational kernels
# -lgomp                    : GNU OpenMP runtime, must match the threading layer above
# -lpthread                 : POSIX threads, required by OpenMP and MKL
# -lm                       : standard math library
# -ldl                      : dynamic loading library, required by MKL at runtime
LIB = -m64 -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a \
	  ${MKLROOT}/lib/intel64/libmkl_gnu_thread.a \
	  ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lgomp \
	  -lpthread -lm -ldl

## ASAN MKL dynamic linking flags
# -L${MKLROOT}/lib/intel64 : tells the linker where to find MKL's dynamic libraries
# -lmkl_rt                 : Intel's single dynamic library bundling all of MKL,
#                            replaces the three static .a files and their --start-group grouping.
#                            Dynamic MKL is used here to avoid ASAN false positives from MKL's
#                            internal allocations which are invisible to the static build's ASAN
# -lgomp                   : GNU OpenMP runtime, must match the threading layer above
# -lpthread                 : POSIX threads, required by OpenMP and MKL
# -lm                       : standard math library
# -ldl                      : dynamic loading library, required by MKL at runtime
# -fsanitize=address        : links the ASAN runtime that provides checking logic and error reporting
LIB_ASAN = -m64 -L${MKLROOT}/lib/intel64 -lmkl_rt -lgomp -lpthread -lm -ldl \
           -fsanitize=address

## Source files
SRC = cpp/starm.cpp

## Output filename for the production shared library
# python3-config --extension-suffix appends the platform-specific suffix automatically
TARGET = pystarm$(shell python3-config --extension-suffix)

## Output filename for the ASAN shared library — separate from production so both can coexist
TARGET_ASAN = pystarm_asan$(shell python3-config --extension-suffix)

## Default target — builds the production shared library
# Running plain "make" only builds TARGET, nothing changes for other developers
all: $(TARGET)

## Production build rule
# $(CC)                                   : the C compiler
# $(shell python3 -m pybind11 --includes) : adds pybind11 and Python header include paths
# -DMKL_ILP64                             : tells MKL to use 64-bit integers for array indices,
#                                           required for large problem sizes
# -m64                                    : targets 64-bit architecture, must match MKL_ILP64
# -I${MKLROOT}/include                    : adds MKL header files needed by the C++ source
# $<                                      : the first dependency (SRC)
# -o $@                                   : output to the target filename
$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(shell python3 -m pybind11 --includes) -DMKL_ILP64 -m64 -I${MKLROOT}/include $< -o $@ $(LIB)

## ASAN target — builds the ASAN instrumented shared library
# Run with: make asan
# Before running any Python script with the ASAN build, set the following environment variables:
#   export MKL_INTERFACE_LAYER=ILP64   (selects 64-bit integer interface for dynamic MKL)
#   export MKL_THREADING_LAYER=GNU     (selects GNU OpenMP threading backend for dynamic MKL)
#   export LD_PRELOAD=<path to libasan.so>  (ensures ASAN runtime is loaded before everything else;
#                                            find the path with: gcc -print-file-name=libasan.so)
asan: $(TARGET_ASAN)

## ASAN build rule — identical to production except uses CFLAGS_ASAN and LIB_ASAN
$(TARGET_ASAN): $(SRC)
	$(CC) $(CFLAGS_ASAN) $(shell python3 -m pybind11 --includes) -DMKL_ILP64 -m64 -I${MKLROOT}/include $< -o $@ $(LIB_ASAN)

## Removes all build outputs including both production and ASAN shared libraries
clean:
	rm -f $(TARGET)
	rm -f *.so
	rm -rf build

## .PHONY declares targets that are not real files so Make always runs them
## regardless of whether a file with the same name exists in the directory
.PHONY: all asan clean
