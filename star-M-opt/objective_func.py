'''
Objective and gradient for the low star-M rank approximation.

Two routes are provided and they are not equivalent.

`low_rank_objective_and_gradient` is the one to use. It computes the gradient
without differentiating through the SVD factors at all, which matters because
the factor adjoints in svd_gradient_calculations.py are not correct when the
rank is truncated, and truncation is the only regime this algorithm runs in.

`low_rank_obj_func_gradient` is the original factor-adjoint chain, kept for
reference and comparison. It inherits that truncation defect.

The reformulation. For orthogonal M the Frobenius norm is invariant under a
mode product, so with A_hat = A x_j M for the transform modes,

    f(M) = ||A - A_k(M)||_F^2
         = ||A_hat - truncate_k(A_hat)||_F^2
         = sum over slices of sum_{i>k} sigma_i^2

that is, the discarded singular energy of the transformed tensor. The gradient
of tail singular energy involves singular values only,

    d/dX sum_{i>k} sigma_i(X)^2 = 2 (X - X_k)

so chaining through A_hat gives the gradient as one contraction of the residual
against the data. No singular-vector derivative appears.

Verified numerically in
/storage/vhebbur/survey/research_notes/Matrix exponential strategy for pystarm/
gradient_without_svd_adjoints.md: the directional derivative along the geodesic
matches central differences to ten digits with second-order convergence.

Two conditions. M must be orthogonal, since the derivation uses M^-1 = M^T;
off the group the identity above is false, and a naive ambient finite-difference
check will appear to contradict it. And sigma_k must exceed sigma_{k+1}, or the
truncation is not unique and the objective is not differentiable there.
'''

import numpy as np
from svd_gradient_calculations import calc_U_gradient, calc_S_gradient, calc_VT_gradient
import pystarm


def _as_numpy(obj):
    '''Read a pystarm Matrix or Tensor as a numpy array without copying.'''
    return np.frombuffer(obj, dtype=np.float64).reshape(obj.getdims(), order='F')


def transform_modes(ndim):
    '''
    The modes a transform matrix is applied to, as pystarm mode indices.
    Input:
    - ndim: Number of modes of the tensor.

    Output:
    - modes: List of 0-based mode indices, one per transform matrix. Mode index 2
      is the third tensor mode, matching pystarm.ttm's convention.
    '''
    return list(range(2, ndim))


def low_rank_objective_and_gradient(A, Ms, k, want_gradient=True):
    '''
    Objective and Euclidean gradients for the low star-M rank approximation.
    Input:
    - A: pystarm Tensor of order >= 3. The raw data. Not freed by this function.
    - Ms: Python list of pystarm Matrices, one per transform mode, each orthogonal.
      There must be ndim - 2 of them. Not freed by this function.
    - k: Target rank for the slicewise truncation.
    - want_gradient: If False, only the objective is computed and the gradient
      list comes back empty. The line search uses this.

    Output:
    - f: The objective ||A - A_k(M)||_F^2, as a float.
    - grads: List of numpy arrays, one per transform matrix, each the Euclidean
      gradient of f with respect to that matrix. Empty when want_gradient is False.
      These are ambient gradients; project them to the tangent space with
      optimizer.tangent_direction before stepping.
    '''
    dims = A.getdims()
    modes = transform_modes(len(dims))
    if len(Ms) != len(modes):
        raise ValueError(f'expected {len(modes)} transform matrices, got {len(Ms)}')

    # Forward: move every transform mode into the transform domain, keeping the
    # partial products. partials[j] is A with every transform applied except Ms[j],
    # which is exactly what the gradient for Ms[j] must contract against.
    partials = []
    A_hat = A
    owned = []
    for j, mode in enumerate(modes):
        nxt = pystarm.ttm(A_hat, Ms[j], mode)
        owned.append(nxt)
        A_hat = nxt

    if want_gradient:
        for j, mode in enumerate(modes):
            B = A
            tmp = []
            for i, mi in enumerate(modes):
                if i == j:
                    continue
                B = pystarm.ttm(B, Ms[i], mi)
                tmp.append(B)
            partials.append((B, tmp))

    # Truncate slicewise in the transform domain and form the residual there.
    Uk, Sk, Vkt = pystarm.slicewise_svdx(A_hat, k)
    Ak_hat = pystarm.slicewise_matmul(Uk, Sk, Vkt)
    R = pystarm.tensor_minus_tensor(A_hat, Ak_hat)

    # The objective is the discarded singular energy, which is the squared norm
    # of the residual in the transform domain. No inverse transform is needed:
    # the orthogonal mode product preserves the norm.
    f = float(R.norm()) ** 2

    grads = []
    if want_gradient:
        for j, mode in enumerate(modes):
            B, tmp = partials[j]
            # dF/dM_j = 2 * R_(j) B_(j)^T, with the unfolding taken in mode j.
            G = pystarm.tensor_contract_all_but_one(R, B, mode, False)
            grads.append(2.0 * np.array(_as_numpy(G), copy=True))
            G.clear()
            for t in tmp:
                t.clear()

    R.clear()
    Ak_hat.clear()
    Uk.clear()
    Sk.clear()
    Vkt.clear()
    for t in owned:
        t.clear()

    return f, grads


def low_rank_objective(A, Ms, k):
    '''
    Objective only, for the line search.
    Input:
    - A: pystarm Tensor of order >= 3.
    - Ms: Python list of pystarm Matrices, one per transform mode.
    - k: Target rank.

    Output:
    - f: The objective ||A - A_k(M)||_F^2, as a float.
    '''
    f, _ = low_rank_objective_and_gradient(A, Ms, k, want_gradient=False)
    return f


def tensor_times_matrix(A, M, inv_flag=False):
    '''
    Performs a tensor-times-matrix (TTM) operation for modes >= 3.
    Input:
    - A: pystarm tensor
    - M: python list of pystarm matrices.

    Returns:
    - A_hat: A moved into the transform domain.
    '''
    # Move A and B into transform domain
    current_mode = 2 # start with mode 3-1=2
    A_hat = pystarm.ttm(A, M[0], current_mode) # Args: tensor, matrix, mode (index 0, so if you put 1 it will be mode 1+1=2)
    # For higher orders (>= 4), move those modes into transform domain. 
    if len(A.getdims()) > 3:
        for matrix in M[1:]:
            current_mode += 1
            # Move A into the transform domain for the current mode.
            A_hat = pystarm.ttm(A_hat, matrix, current_mode) # Args: tensor, matrix, mode (index 0, so if you put 1 it will be mode 1+1=2)
    
    return A_hat

def low_rank_obj_func_gradient(A, Ms, k):
    '''
    Calculates the gradient of the star-M SVD. 
    Input:
    - M: python list of pystarm matrices. There should be mode - 2 M's.
    So, if you have a tensor of mode 5, you will need a python array with 3 pystarm Matrices.

    Output: 
    - M_grad: Gradient of the objective function.
    '''
    # Since we have to redo these calculations a lot, will we need to clear all of the tensors used here?
    # Forward pass
    MOpts= []
    mode = 2 # It's 2 because the modes are index 0. We are really starting with mode 3.
    for M in Ms:
        A_hat = pystarm.ttm(A, M, mode)
        Uk_hat, Sk_hat, Vkt_hat = pystarm.slicewise_svdx(A_hat, k)
        Ak_hat = pystarm.slicewise_matmul(Uk_hat, Sk_hat, Vkt_hat)
        # inv_flag still needs to be implemented. line below needs inv_flag=True for M. Or M just needs to be transposed.
        Ak = pystarm.ttm(Ak_hat, M, mode) 

        # Backwards pass
        # Step 4
        R = pystarm.tensor_minus_tensor(A, Ak)
        grd_Ak_wrt_Minv = pystarm.tensor_contract_all_but_one(R, Ak_hat, k, naive=False)
        # inv_flag needs to be on the line below.
        grd_Ak_wrt_Akhat = pystarm.ttm(R, M, mode)
        # Step 3
        
        # Here, we need to represent Ak_hat as two different products: B = U * S, Ak_hat = B * VT
        grd_Akhat_wrt_B = pystarm.slicewise_matmul(grd_Ak_wrt_Akhat, Vkt_hat, transpose_A=False, transpose_B=True)
        grd_Akhat_wrt_VT = pystarm.slicewise_matmul(pystarm.slicewise_matmul(Uk_hat, Sk_hat), grd_Ak_wrt_Akhat, transpose_A=True, transpose_B=False)
        # I guess this doesn't really need a transpose now, does it. S is diagonal. 
        grd_B_wrt_U = pystarm.slicewise_matmul(grd_Akhat_wrt_B, Sk_hat)
        grd_B_wrt_S = pystarm.slicewise_matmul(Uk_hat, grd_Akhat_wrt_B, transpose_A=True, transpose_B=False)
        print(grd_B_wrt_S.getdims())

        # Step 2
        # All three adjoints take the same (U, S, VT, cotangent) signature; the rank is
        # read off the factors rather than passed in, so it cannot disagree with them.
        #
        # CAVEAT: slicewise_svdx above truncates to k. When k < min(m, n) the U and VT
        # adjoints are not the true adjoints of the truncated SVD -- the derivative of a
        # truncated singular vector picks up contributions from the discarded directions
        # i > k that the formula omits. See the docstring of SVDGradientTestCase in
        # tests/test.py, which measures this. The reference MATLAB
        # (objectiveFunctions/objFctnLowRank.m) has the same limitation. The
        # singular-value adjoint is exact either way.
        grd_U = calc_U_gradient(Uk_hat, Sk_hat, Vkt_hat, grd_B_wrt_U)
        grd_S = calc_S_gradient(Uk_hat, Sk_hat, Vkt_hat, grd_B_wrt_S)
        grd_VT = calc_VT_gradient(Uk_hat, Sk_hat, Vkt_hat, grd_Akhat_wrt_VT)

        grd_AkHat = pystarm.tensor_plus_tensor(grd_U, grd_S, grd_VT)
        # Step 1
        grd_Ahat_wrt_M = pystarm.tensor_contract_all_but_one(grd_AkHat, A, k, naive=False)
    
        MOpts.append(grd_Ahat_wrt_M + grd_Ak_wrt_Minv)
        mode +=1
    return MOpts
    




    # Do A x_3 M and get jacobian of A_hat w.r.t M

    

