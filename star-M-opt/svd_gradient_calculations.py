'''
These are helper functions for calculating SVD gradients in objective_func.py.

You can find the derivations for these gradient calculations in the supplementary materials of
OPTIMAL MATRIX-MIMETIC TENSOR ALGEBRAS VIA VARIABLE PROJECTION (2025) by Elizabeth Newman and
Katherine Keegan.

Reference implementation: https://github.com/elizabethnewman/star-M-opt, file
`tensorSVD/facewiseSVDJacobian.m`, which in turn follows
https://j-towns.github.io/papers/svd-derivative.pdf.

WHAT THESE FUNCTIONS COMPUTE
----------------------------
Let A be an m x n x ... tensor with slicewise economy SVD A = U * S * VT (facewise), so that
U is m x k x ..., S is the compact k x nslices matrix of singular values, and VT is k x n x ...
Each of the three functions below is the *adjoint* of the Jacobian of one factor with respect
to A, applied to a cotangent (an incoming gradient) of that factor's shape:

    calc_U_gradient (U, S, VT, dU)  implements MATLAB's JacU.AT,  dU  has the shape of U
    calc_S_gradient (U, S, VT, dS)  implements MATLAB's JacS.AT,  dS  has the shape k x k x ...
    calc_VT_gradient(U, S, VT, dVT) implements MATLAB's JacV.AT,  dVT has the shape of VT

All three return a tensor with the shape of A.

The defining property, and the one the tests check, is the adjoint identity

    <d(factor)/dA [x], y>_F  ==  <x, calc_*_gradient(..., y)>_F

for every direction x of A's shape and every cotangent y.

TRUNCATION CAVEAT -- READ THIS BEFORE TRUSTING THE OPTIMIZER
------------------------------------------------------------
The identity above holds only when k == min(m, n), i.e. for the full economy SVD. For a
truncated SVD (k < min(m, n)) the U and VT adjoints are NOT the true adjoints: the
derivative of a truncated singular vector has contributions from the discarded singular
directions i > k, which the formula's F matrix (restricted to the retained k x k block)
leaves out. Measured relative errors against central differences in that regime run from
about 0.1 to 1.5 and do not shrink as the finite-difference step is refined, which is how
you tell this apart from discretisation noise. The singular-value adjoint
(calc_S_gradient) stays exact under truncation, because singular values do not depend on
the discarded subspace to first order.

This limitation is inherited from the reference, not introduced here.
facewiseSVDJacobian.m states "assume matrices are full and full-rank" in its header and
its own CheckAdjoint tests only exercise the economy case, yet
objectiveFunctions/objFctnLowRank.m feeds it truncated factors anyway. Since
objective_func.py calls slicewise_svdx(A_hat, k) with k < min(m, n), the star-M optimizer
is in exactly that regime and its M-gradient is correspondingly approximate. Both
behaviours are pinned by tests in tests/test.py (SVDGradientTestCase).

SIGNATURE CONVENTION
--------------------
All three take the uniform signature (U, S, VT, dX). The previous code had three different
signatures, none of which matched its call site in objective_func.py. `calc_S_gradient` does
not use S, but keeps it so that every call site reads the same; the rank k is read off
U.getdims() rather than being passed in, so it cannot disagree with the factors.

TRANSPOSE CONVENTION
--------------------
The MATLAB reference carries V; pystarm's slicewise_svd/slicewise_svdx return VT. Every
formula below has therefore been rewritten in terms of VT. In particular MATLAB's
`y ./ reshape(s,[1,k,...])` (divide column j of an n x k cotangent by s_j) becomes, for a
k x n cotangent, `diag(1/s) * y`, i.e. a LEFT multiply by the reciprocal diagonal, which is
slicewise_matmul(..., inv_flag=True, ten_left=False).

MEMORY
------
pystarm does not garbage collect: every tensor returned by a kernel is a C++ malloc that must
be released with .clear(). These functions clear every intermediate they allocate and return a
single live tensor that the caller owns and must clear. Tensors that wrap a NumPy buffer (only
F, built in get_F) are never cleared, since the buffer belongs to NumPy.
'''
import numpy as np
import pystarm
from pystarm import slicewise_matmul, hadamard_pointwise
import math


def _nslices(T):
    '''Number of frontal slices of a pystarm Tensor, i.e. the product of modes 3 and up.'''
    return math.prod(T.getdims()[2:])


def get_F(S, k, n_slices):
    '''
    Build the k x k x n_slices tensor of reciprocal squared-singular-value differences.

    MATLAB (facewiseSVDJacobian.m):
        s      = facewiseDiag(S);      % k x 1 x nslices (a column of diagonals)
        s2     = s.^2;
        F      = -s2 + tran(s2);       % broadcasts to k x k x nslices
        idx    = (F ~= 0);
        F(idx) = 1 ./ F(idx);

    `s` is a k x 1 column, not a 1 x k row: that is why the Jacobians have to
    `reshape(s,[1,k,...])` before dividing by it. So -s2 varies down the rows (index i) and
    tran(s2) varies across the columns (index j), giving F(i,j,l) = s(j,l)^2 - s(i,l)^2,
    which is the convention of Townsend's note that the file cites. After the guarded
    reciprocal,

        F[i,j,l] = 1 / (s_j^2 - s_i^2)   when s_i^2 != s_j^2
        F[i,j,l] = 0                     otherwise (which includes the whole diagonal)

    Getting this orientation backwards flips the sign of the U and V gradients, which the
    finite-difference tests in tests/test.py catch immediately.

    REPEATED SINGULAR VALUES. The reciprocal is guarded exactly as the MATLAB reference
    guards it: entries where the two squared singular values are bit-for-bit equal are left
    at zero rather than becoming +/-Inf. This is a convention, not a derivative. When two
    singular values coincide the corresponding singular vectors are not unique, so U and VT
    are not differentiable functions of A there and no finite value would be correct; zero
    is chosen because it keeps the result finite and leaves the well-defined part of the
    Jacobian (the off-diagonal blocks between distinct singular values) exact. Near-degenerate
    but distinct singular values still produce legitimately huge entries; that is the genuine
    ill-conditioning of the problem and is deliberately NOT smoothed over with a threshold,
    which would silently bias the gradient.

    Returns a pystarm Tensor that wraps a NumPy buffer. Do not call .clear() on it.
    '''
    dims = (k, k, n_slices)
    # S is the compact k x nslices matrix of singular values: S.get(i, l) is the i-th
    # singular value of slice l. (The previous version indexed it as S.get(l, j), which is
    # transposed and reads out of bounds whenever k != n_slices.)
    s = np.empty((k, n_slices), dtype=np.float64)
    for l in range(n_slices):
        for i in range(k):
            s[i, l] = S.get(i, l)

    s2 = s ** 2
    # F[i,j,l] = s2[j,l] - s2[i,l]
    F_np = s2[None, :, :] - s2[:, None, :]
    nz = (F_np != 0.0)
    out = np.zeros_like(F_np)
    np.divide(1.0, F_np, out=out, where=nz)

    F_np = np.asfortranarray(out)
    # keep_alive on the binding ties F_np's lifetime to the Tensor, so it is safe to drop
    # the local reference here.
    return pystarm.Tensor(F_np, len(dims), list(dims))


def get_I(dims):
    '''
    Identity tensor: n_slices frontal slices, each the n x n identity.

    dims is (n, n_slices). Returns a pystarm Tensor wrapping a NumPy buffer; do not clear it.

    Note: the gradient routines below no longer need this. They apply the projectors
    (I - U U^T) and (I - V V^T) as X - U (U^T X) and W - (W V) V^T respectively, which is
    both cheaper (no m x m x nslices temporary) and avoids materialising an identity at all.
    It is kept because it is a reasonable utility and the old, broken version was exported.
    '''
    n, n_slices = dims
    I_np = np.zeros((n, n, n_slices), dtype=np.float64, order='F')
    idx = np.arange(n)
    I_np[idx, idx, :] = 1.0
    return pystarm.Tensor(I_np, 3, [n, n, n_slices])


def calc_U_gradient(U, S, VT, dU):
    '''
    Adjoint of the Jacobian of U with respect to A, applied to the cotangent dU.

    MATLAB CODE (JacU.AT):
        UUT  = facewise(U, tran(U));
        tmp1 = @(y) facewise(U,
                             facewise(F .* (facewise(tran(U),y) - facewise(tran(y),U)), S));
        tmp2 = @(y) facewise(eye(m) - UUT, y ./ reshape(s,szS));
        JacU.AT = @(y) facewise(tmp1(y) + tmp2(y), tran(V));

    Here tran(V) is just VT, and (eye(m) - UUT) * Z is applied as Z - U (U^T Z).
    '''
    k = U.getdims()[1]
    n_slices = _nslices(U)
    F = get_F(S, k, n_slices)            # wraps NumPy; never cleared

    # tmp1 = U * ( F .* (U^T y - y^T U) ) * diag(s)
    G = slicewise_matmul(U, dU, transpose_A=True, transpose_B=False)   # U^T y   (k x k)
    GT = slicewise_matmul(dU, U, transpose_A=True, transpose_B=False)  # y^T U   (k x k)
    D = pystarm.tensor_minus_tensor(G, GT)
    H = hadamard_pointwise(F, D)
    HS = slicewise_matmul(H, S, inv_flag=False, ten_left=True)         # (F .* D) * diag(s)
    tmp1 = slicewise_matmul(U, HS)                                     # m x k

    # tmp2 = (I_m - U U^T) * (y * diag(1/s)) = Z - U (U^T Z)
    Z = slicewise_matmul(dU, S, inv_flag=True, ten_left=True)          # y * diag(1/s)  (m x k)
    UtZ = slicewise_matmul(U, Z, transpose_A=True, transpose_B=False)  # U^T Z          (k x k)
    UUtZ = slicewise_matmul(U, UtZ)                                    # U U^T Z        (m x k)
    tmp2 = pystarm.tensor_minus_tensor(Z, UUtZ)

    Tsum = pystarm.tensor_plus_tensor(tmp1, tmp2)
    grad = slicewise_matmul(Tsum, VT)                                  # m x n

    for t in (G, GT, D, H, HS, tmp1, Z, UtZ, UUtZ, tmp2, Tsum):
        t.clear()
    return grad


def calc_S_gradient(U, S, VT, dS):
    '''
    Adjoint of the Jacobian of S with respect to A, applied to the cotangent dS.

    MATLAB CODE (JacS.AT):
        JacS.AT = @(y) facewise(facewise(U, (eye(k) .* y)), tran(V));

    `eye(k) .* y` keeps only the diagonal of each frontal slice of y. pystarm already stores
    a diagonal tensor compactly as a k x nslices matrix, so this is get_slicewise_diagonals
    followed by the three-argument slicewise_matmul(U, diag, VT).

    S is accepted but unused; it is in the signature so that all three adjoints are called
    identically. dS must have square frontal slices (k x k x ...), which is what the chain
    rule produces for the singular-value factor.
    '''
    dS_dims = dS.getdims()
    assert dS_dims[0] == dS_dims[1], (
        "calc_S_gradient expects dS with square frontal slices (k x k x ...); "
        "get_slicewise_diagonals only indexes correctly for square slices."
    )

    d = pystarm.get_slicewise_diagonals(dS)        # Matrix, k x nslices
    grad = pystarm.slicewise_matmul(U, d, VT)      # U * diag(d) * VT
    d.clear()
    return grad


def calc_VT_gradient(U, S, VT, dVT):
    '''
    Adjoint of the Jacobian of V with respect to A, applied to the cotangent dVT.

    MATLAB CODE (JacV.AT), written in terms of V:
        VVT  = facewise(V, tran(V));
        tmp1 = @(y) facewise(facewise(S, (F .* (facewise(tran(V),y) - facewise(tran(y),V)))),
                             tran(V));
        tmp2 = @(y) facewise(tran(y ./ reshape(s,szS)), eye(n) - VVT);
        JacV.AT = @(y) facewise(U, tmp1(y) + tmp2(y));

    Rewritten for a cotangent Y = y^T of VT's shape (k x n), using V = VT^T:
        G    = VT * Y^T                       (= tran(V)*y,  k x k), and y^T V = G^T
        tmp1 = diag(s) * (F .* (G - G^T)) * VT
        W    = diag(1/s) * Y                  (= tran(y ./ s),  k x n)
        tmp2 = W - (W * VT^T) * VT            (= W * (I_n - V V^T))
        grad = U * (tmp1 + tmp2)
    '''
    k = VT.getdims()[0]
    n_slices = _nslices(VT)
    F = get_F(S, k, n_slices)            # wraps NumPy; never cleared

    # tmp1 = diag(s) * ( F .* (VT Y^T - Y VT^T) ) * VT
    G = slicewise_matmul(VT, dVT, transpose_A=False, transpose_B=True)    # VT Y^T  (k x k)
    GT = slicewise_matmul(dVT, VT, transpose_A=False, transpose_B=True)   # Y VT^T  (k x k)
    D = pystarm.tensor_minus_tensor(G, GT)
    H = hadamard_pointwise(F, D)
    SH = slicewise_matmul(H, S, inv_flag=False, ten_left=False)           # diag(s) * (F .* D)
    tmp1 = slicewise_matmul(SH, VT)                                       # k x n

    # tmp2 = (diag(1/s) Y) * (I_n - V V^T) = W - (W VT^T) VT
    W = slicewise_matmul(dVT, S, inv_flag=True, ten_left=False)           # diag(1/s) Y  (k x n)
    WV = slicewise_matmul(W, VT, transpose_A=False, transpose_B=True)     # W VT^T       (k x k)
    WVVt = slicewise_matmul(WV, VT)                                       # W V V^T      (k x n)
    tmp2 = pystarm.tensor_minus_tensor(W, WVVt)

    Tsum = pystarm.tensor_plus_tensor(tmp1, tmp2)
    grad = slicewise_matmul(U, Tsum)                                      # m x n

    for t in (G, GT, D, H, SH, tmp1, W, WV, WVVt, tmp2, Tsum):
        t.clear()
    return grad
