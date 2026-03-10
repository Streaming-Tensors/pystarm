# pystarm — Project Guide for Claude

## What is pystarm

pystarm is a Python/C++ library that wraps matrix and tensor operations via pybind11 and Intel MKL. It is used to implement and benchmark t-SVDM (Tensor SVD with Matrix Transform) algorithms for tensor compression.

## Algorithms

Two variants of t-SVDM are implemented:

- **tsvdm-I** (`tsvdmi`): Fixed-rank compression. Takes a rank parameter `k`.
- **tsvdm-II** (`tsvdmii`): Tolerance-based compression. Takes an error tolerance `tol`.

Both variants follow the same structure: apply a transformation (TTM) on all modes except the first two, perform a slicewise SVD, then reconstruct via inverse TTM.

## Key Conventions

- All data is **64-bit float (float64)**.
- All arrays and tensors are in **Fortran (column-major) order**.
- pystarm objects (`Tensor`, `Matrix`) must be **explicitly freed** by calling `.clear()` — ownership is on the C++ side.
- `np.frombuffer(pystarm_obj, dtype=np.float64)` gives a read-only 1D view of the internal buffer. Use `.getdims()` to get the shape and reshape with `order='F'`.
- `slicewise_svd` / `slicewise_svdx` treat `dims[0]` and `dims[1]` as matrix dimensions and `dims[2:]` as slice dimensions. They always return a **3D tensor**, collapsing all modes ≥ 2 into a single `nslices` dimension.

## Directory Structure

```
cpp/               C++ source code
  matrix.hpp/cpp   Matrix class
  tensor.hpp/cpp   Tensor class
  ops.cpp          Matrix and tensor operations (TTM, slicewise SVD, etc.)
  starm.cpp        pybind11 Python bindings
alg.py             tsvdm-I and tsvdm-II compress and reconstruct functions
experiments.py     Unified experiment script for soccer, traffic, and CFD datasets
parse_logs.py      Parses experiment log files into a CSV
scripts/
  experiments.sh   Bash script to batch-run experiments
  asan-run-prep.sh Sets up environment for running with AddressSanitizer
known_issues/      Known bugs and debugging notes
test.py            Unit tests and usage examples for pystarm
```

## Datasets

| Name    | Format         | Shape           | Reader function       |
|---------|----------------|-----------------|-----------------------|
| soccer  | .mp4 (video)   | H × W × T      | `read_soccer_data`    |
| traffic | .bin (binary)  | H × W × 3 × T  | `read_traffic_data`   |
| cfd     | .h5 (HDF5)     | 5-way tensor   | `read_cfd_data`       |

### traffic.bin binary format
Header: `uint32 height`, `uint32 width`, `float64 fps`
Body: float64 values in MATLAB column-major order, shape `(H, W, 3, T)`

## experiments.py

Unified script for running experiments on any dataset.

```bash
python experiments.py -alg <tsvdmi|tsvdmii> -mtype <dct|eye|hosvd> \
    -k <rank> -tol <tolerance> \
    -dname <soccer|traffic|cfd> -dfile <path> \
    -perm-mode <digit string e.g. 0123>
```

- `-perm-mode`: digit string passed to `np.transpose` to reorder modes so transformation modes are last. e.g. `"0123"` → `(0,1,2,3)`.
- `ttm_modes = list(range(2, arr.ndim))` — transformation is applied on all modes except the first two.
- After permutation, a Fortran-order copy is made slice-by-slice along the last axis.
- For `mtype=eye`, no TTM is applied (identity transform is a no-op).
- For `mtype=hosvd`, HOSVD is run once via pyttb to get all factor matrices, then applied per mode.

## Building

```bash
make all       # standard build
make asan      # build with AddressSanitizer (produces pystarm_asan.*.so)
```

Set `MKLROOT` if not already set:
```bash
export MKLROOT=/global/common/software/nersc9/intel/oneapi/mkl/2024.1
```

Built and tested with GNU compiler on NERSC Perlmutter. Intel compiler build has unresolved errors.

## AddressSanitizer

To run with ASan:
1. `source scripts/asan-run-prep.sh`
2. `make asan`
3. In `soccer.py` (or any experiment script), uncomment the `importlib` block at the top to load `pystarm_asan.*.so` and comment out `import pystarm`.
4. Run and redirect output: `python soccer.py ... > asan_out.txt 2>&1`

See `known_issues/README.md` for known bugs and debugging notes.

## Known Issues

See `known_issues/README.md`. Key issue: `tsvdmi` with `mtype=eye` on soccer data crashes under multi-threaded execution. Does not crash single-threaded, with DCT, with tsvdmii, or on traffic data.
