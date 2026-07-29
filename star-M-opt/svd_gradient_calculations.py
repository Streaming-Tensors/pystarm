'''
These are helper functions for calculating SVD gradients in objective_func.py.

You can find the derivations for these gradient calculations in the supplementary materials of 
OPTIMAL MATRIX-MIMETIC TENSOR ALGEBRAS VIA VARIABLE PROJECTION (2025) by Elizabeth Newman and 
Katherine Keegan.
'''
import numpy as np
import pystarm
from pystarm import slicewise_matmul, hadamard_pointwise
import itertools as it
import math

def get_F(S, k, n_slices):
    '''
    F = initialize k x k x nslices tensor.
    for slice in F:
        for i, j in slice:
            if i != j:
                slice[i][j] = ((S[slice_num][j] ** 2) - (S[slice_num][i] ** 2))** -1
            else:
                slice[i][j] = 0
        
    '''
    dims = (k, k, n_slices)
    F_np = np.zeros(dims)
    for l in range(n_slices):
        for j in range(k):
            for i in range(k):
                if i != j:
                    F_np[i,j,l] = ((S.get(l, j)**2) - (S.get(l, i)**2)) ** -1

    F = pystarm.Tensor(np.asfortranarray(F_np), len(dims), dims)
    return F

def calc_U_gradient(U, S, VT, dU, k):
    '''
    MATLAB CODE:
    s      = facewiseDiag(S);
    szS    = [1,k,szA(3:end)];
    s2     = s.^2;
    F      = -s2 + tran(s2);
    UUT = facewise(U,tran(U));
    tmp1 = @(y) facewise(
                    U,
                    facewise(
                        F .* (facewise(tran(U),y) - facewise(tran(y),U)),
                        S
                    )
                );
    tmp2 = @(y) facewise(eye(m) - UUT,y ./ reshape(s,szS));
    Here, it's computing the inverse by doing elementwise division. Maybe I could add an inverse flag to the 
    kernel? Perhaps.
    JacU.AT = @(y) facewise(tmp1(y) + tmp2(y),tran(V));

    Note the .* is elementwise multiplication
    compute_F:
    F is size k x k x nslices
    F[i,j,l] = (S[j, l]**2 - S[i, l]**2)**-1 if i != j else 0
    '''
    n_slices = math.prod(U.getdims()[2:])
    F = get_F(S, k, n_slices)
    Utrans_dU = slicewise_matmul(U, dU, transpose_A=True, transpose_B=False)
    dUtrans_U = slicewise_matmul(dU, U, transpose_A=True, transpose_B=False)
    U_Utrans = slicewise_matmul(U, U, transpose_A=False, transpose_B=True)
    tmp1 = slicewise_matmul(U, 
                            slicewise_matmul(
                                hadamard_pointwise(
                                    F, 
                                    pystarm.tensor_minus_tensor(
                                        Utrans_dU, dUtrans_U
                                    )
                                ),
                                S
                            )
            )
    tmp2 = slicewise_matmul(
        pystarm.tensor_minus_tensor(np.eye(U.getdims[0]), U_Utrans),
        # TODO: Add the appropriate flags into the kernel so this can be computed. 
        slicewise_matmul(dU, S, inv_M=True)
    )

    dU = slicewise_matmul(
            pystarm.tensor_plus_tensor(tmp1, tmp2),
            VT
        )
    return dU

def calc_S_gradient(U, VT, dS):
    '''
    MATLAB CODE:
    facewise(facewise(U,(eye(k) .* y)),tran(V));
    '''
    # This is the same as multiplying elementwise by the identity. We represent as a matrix to save space,
    # and because we already have an operation that does Tensor * diagTensor * Tensor
    S = pystarm.get_slicewise_diagonals(dS)

    grad = pystarm.slicewise_matmul(U, S, VT)
    S.clear()
    return(grad)

def calc_VT_gradient(dVT):
    print("Nothing here yet.")
    return(dVT)