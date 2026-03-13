# pystarm — Project Guide for Claude

## What is pystarm

pystarm is a Python/C++ library that wraps matrix and tensor operations via pybind11 and Intel MKL. It is used to implement and benchmark t-SVDM (Tensor SVD with Matrix Transform) algorithms for tensor compression.

## Reference Papers

This work closely follows these two papers:

- **PNAS publication**: "Tensor-tensor algebra for optimal representation and compression of multiway data" — Misha E. Kilmer, Lior Horesh, Haim Avron, and Elizabeth Newman. https://www.pnas.org/doi/epub/10.1073/pnas.2015851118
- **ArXiv tech report**: "Tensor-Tensor Products for Optimal Representation and Compression" — Misha Kilmer, Lior Horesh, Haim Avron, and Elizabeth Newman. https://arxiv.org/abs/2001.00046

## Paper Summary

### Core Idea

The central question of the papers is: is it better to compress data as a matrix (2D) or as a tensor (multi-dimensional array)? The papers prove that treating data as a tensor and compressing via t-SVDM is provably at least as good as, and often strictly better than, matrix SVD.

### The t-SVDM Framework

Any invertible matrix **M** defines a tensor-tensor product (called ?M). The compression proceeds as:
1. Apply M along the transformation modes of the tensor (go to transform domain)
2. Perform independent matrix SVDs on each frontal slice in that domain
3. Apply M⁻¹ to reconstruct (inverse transform)

The original t-product (Kilmer & Martin, 2011) is the special case where M is the DFT matrix. The papers generalize this to any invertible M, with the key requirement that M must be a **non-zero multiple of a unitary (orthogonal) matrix** for the Eckart-Young optimality theorem to hold.

### Choices of M

| M | Notes |
|---|---|
| DFT | Original t-product, complex arithmetic |
| DCT | Real-valued, good for spatially/temporally smooth data |
| Wavelet | Orthogonal wavelet matrix |
| HOSVD factor Z | M = Z^T recovers HOSVD as a special case |
| Random orthogonal | No benefit — used as a negative baseline in experiments |
| Identity (eye) | No inter-slice mixing, weakest compression |

A good M concentrates the energy of the tensor into fewer, larger singular values in the transform domain (faster decay), enabling better compression at the same error level.

### Energy

The **energy** of a tensor is its squared Frobenius norm — the sum of squares of all its entries. By Corollary 6 of the paper, this equals the sum of squared singular values across all slices in the transform domain. Singular values therefore directly partition the total energy, and retaining the largest ones retains the most energy.

### Algorithms

**t-SVDM / tsvdmi** (Algorithm 2 in paper): Fix a global rank **k**. Transform, truncate every slice to rank k, inverse transform. Eckart-Young optimal: best rank-k approximation in Frobenius norm. Weakness: treats all slices equally — may over-allocate for easy slices and under-allocate for hard ones.

**t-SVDMII / tsvdmii** (Algorithm 3 in paper): Fix an energy level **γ** (e.g. 0.99 = retain 99% of energy). Compute all singular values across all slices globally, sort them, find a threshold τ such that retained values cover energy γ. Each slice keeps a different number of singular values (multi-rank ρ). Strictly better compression than t-SVDM: same or smaller error, smaller or equal storage (Theorem 17).

### Key Theoretical Results

- **Eckart-Young for t-SVDM** (Theorem 10): rank-k t-SVDM is the best rank-k approximation under ?M — analogous to truncated matrix SVD.
- **t-SVDMII is also optimal** (Theorem 11): multi-rank ρ approximation is best possible at that multi-rank.
- **Tensors beat matrices** (Theorem 15): t-SVDM error ≤ matrix SVD error for the same truncation parameter k. Strict inequality is possible.
- **t-rank ≤ matrix rank** (Theorem 13): A good M can reveal t-rank << matrix rank.
- **HOSVD is a special case** (Theorem 18 & 19): HOSVD is a specific instance of ?M, and truncated HOSVD is provably suboptimal compared to truncated t-SVDM.

### Extension to Higher-Order Tensors

For 4-way and higher tensors, different transforms M, B are applied along different modes, and slicewise SVDs are performed in the multi-transformed domain (Algorithm 5). This is what `experiments.py` implements — `ttm_modes = list(range(2, arr.ndim))` applies transforms on all modes beyond the first two.

### Numerical Results

Tested on Yale face data, traffic video (120×160×120), and hyperspectral images. In all cases t-SVDMII with DCT or wavelet outperforms matrix SVD and HOSVD at the same compression ratio. The traffic video experiment is directly relevant to this codebase.

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
