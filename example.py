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
