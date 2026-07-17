# pystarm

pystarm is a Python library for tensor decomposition via Star-M transformation
(t-SVDM). It provides efficient parallel implementations of the t-SVDM-I and
t-SVDM-II algorithms with a C++ core backed by Intel MKL.

## Reference paper

The algorithms implemented in this library are based on the following t-SVDM
papers:

- PNAS: "Tensor-tensor algebra for optimal representation and compression of
  multiway data" — https://www.pnas.org/doi/epub/10.1073/pnas.2015851118
- ArXiv: "Tensor-Tensor Products for Optimal Representation and Compression"
  — https://arxiv.org/abs/2001.00046

This library and its parallel implementation are described in our ICPP 2026 paper:
https://arxiv.org/abs/2605.16058

## Prerequisites

- Python 3.9 or higher
- Intel MKL (with `MKLROOT` set to its installation path)
- GNU C++ compiler with OpenMP support
- pybind11 (install via `pip install pybind11`)

Other compilers may work with appropriate adjustments to the Makefile, but only
GCC has been tested.

## Build and install

It might be better to use a virtual environment (`venv` or `conda env`) to
avoid dependency conflicts.

Set the MKL installation path:

```bash
export MKLROOT=/path/to/mkl
```

Build the C++ extension:

```bash
make all
```

Install the package:

```bash
pip install .
```

For active development, use an editable install so code changes are reflected
immediately without reinstalling:

```bash
pip install -e .
```

For NERSC Perlmutter and ALCF Aurora, see the environment setup scripts at
[paper/scripts/nersc-env-setup.sh](paper/scripts/nersc-env-setup.sh) and
[paper/scripts/alcf-env-setup.sh](paper/scripts/alcf-env-setup.sh).

## Usage

pystarm expects all numpy arrays to satisfy three requirements:

- `float64` dtype
- Fortran (column-major) order
- Contiguous memory layout

Non-contiguous buffers will cause a segmentation fault at runtime. If your
array is a view, a slice, or does not satisfy any of the above, create a new
array that meets all three requirements before wrapping it with a pystarm
`Tensor` or `Matrix`.

### Thread control

For good performance, set the OpenMP and MKL thread counts to the same value
and disable MKL's dynamic thread adjustment:

```bash
export OMP_NUM_THREADS=<N>
export MKL_NUM_THREADS=<N>
export MKL_DYNAMIC=FALSE
```

where `<N>` is the number of cores you want to use.

### API reference

Data structures (wrap contiguous Fortran-order float64 numpy arrays):

- `pystarm.Tensor(arr, ndim, shape)` — wrap an n-D numpy array as a pystarm tensor
- `pystarm.Matrix(arr, nrow, ncol)` — wrap a 2-D numpy array as a pystarm matrix

Main algorithms:

- `pystarm.tsvdm_I_compress(A, Ms, ttm_modes, k)` — fixed-rank compression. Returns `(U, S, VT)`.
- `pystarm.tsvdm_I_reconstruct(U, S, VT, Minvs, ttm_modes, orig_shape)` — reconstruct from fixed-rank factors.
- `pystarm.tsvdm_II_compress(A, Ms, ttm_modes, tol)` — tolerance-based compression. Returns `(U, S, VT)`.
- `pystarm.tsvdm_II_reconstruct(U, S, VT, Minvs, ttm_modes, orig_shape)` — reconstruct from tolerance-based factors.

Where:
- `A` is a `pystarm.Tensor`
- `Ms` / `Minvs` are lists of `pystarm.Matrix` — forward and inverse transforms for each mode
- `ttm_modes` is a list of ints — which modes the transforms apply to
- `k` is the fixed slice rank; `tol` is the relative error tolerance
- Returned `U`, `S`, `VT` are pystarm objects — call `.clear()` to release their memory when no longer needed

### Example

The following example compresses a random 4D tensor using tsvdmii and
reconstructs it:

```python
import numpy as np
from scipy.fft import dct
import pystarm

# Create a random 4D tensor in Fortran order
shape = (50, 60, 40, 30)
arr = np.asfortranarray(np.random.randn(*shape))
A = pystarm.Tensor(arr, len(shape), shape)

# Build DCT transform matrices for the last two modes
ttm_modes = [2, 3]                                            # Modes 2 and 3 (last two dims) will be transformed
Ms, Minvs = [], []                                            # Lists to hold forward and inverse transforms
for mode in ttm_modes:
    n = shape[mode]                                           # Size of this mode
    # DCT of identity gives the DCT matrix. norm="ortho" makes it orthonormal,
    # which is required for the tsvdmii algorithm to be optimal.
    M = np.asfortranarray(dct(np.eye(n, dtype=np.float64), axis=0, norm="ortho"))
    Minv = np.asfortranarray(M.T)                             # DCT matrix is orthonormal, so inverse = transpose
    Ms.append(pystarm.Matrix(M, M.shape[0], M.shape[1]))      # Wrap in pystarm.Matrix
    Minvs.append(pystarm.Matrix(Minv, Minv.shape[0], Minv.shape[1]))

# Compress with tsvdmii (relative error tolerance of 1%)
tol = 0.01
U, S, VT = pystarm.tsvdm_II_compress(A, Ms, ttm_modes, tol)   # Compress: returns U, S, V^T factors
A_reconst = pystarm.tsvdm_II_reconstruct(U, S, VT,            # Reconstruct: apply inverse transforms
                                          Minvs, ttm_modes, shape)

# np.frombuffer returns a 1D view of the underlying buffer.
# Reshape with the original shape and order='F' to get the multi-dimensional tensor back.
arr_reconst = np.frombuffer(A_reconst, dtype=np.float64).reshape(shape, order='F')
err = np.linalg.norm(arr - arr_reconst) / np.linalg.norm(arr)
print(f"Relative error: {err:.4f} (tolerance was {tol})")

# Cleanup: pystarm objects returned by C++ operations must be explicitly cleared
# to free their underlying memory. Objects constructed from numpy buffers (A, Ms,
# Minvs) should NOT be cleared — their memory is owned by numpy.
U.clear()
S.clear()
VT.clear()
A_reconst.clear()
```

The example above uses `tsvdm_II_compress` (tolerance-based) and
`tsvdm_II_reconstruct`. A fixed-rank variant, `tsvdm_I_compress` and
`tsvdm_I_reconstruct`, is also available — it takes a rank `k` instead of a
tolerance.

## Citation

If you use pystarm in your research, please cite our paper:

```bibtex
@article{hussain2026high,
  title={High-Performance Star-M SVD for Big Data Compression},
  author={Hussain, Md Taufique and Ballard, Grey and Devarakonda, Aditya and Eswar, Srinivas and Pesricha, Naman and Rao, Vishwas},
  journal={arXiv preprint arXiv:2605.16058},
  year={2026}
}
```
