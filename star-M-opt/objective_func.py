import numpy as np
from svd_gradient_calculations import calc_U_gradient, calc_S_gradient, calc_VT_gradient
import pystarm

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

def low_rank_obj_func_gradient(A, M, k):
    '''
    Calculates the gradient of the star-M SVD. 
    Input:
    - M: python list of pystarm matrices. There should be mode - 2 M's.
    So, if you have a tensor of mode 5, you will need a python array with 3 pystarm Matrices.

    Output: 
    - M_grad: Gradient of the objective function.
    '''
    # Forward pass
    A_hat = tensor_times_matrix(A, M)
    Uk_hat, Sk_hat, Vkt_hat = pystarm.slicewise_svdx(A_hat, k)
    Ak_hat = pystarm.slicewise_matmul(Uk_hat, Sk_hat, Vkt_hat)
    Ak = tensor_times_matrix(Ak_hat, M, inv_flag=True) # inv_flag still needs to be implemented.

    # Backwards pass
    # Step 4
    R = -1 * pystarm.tensor_minus_tensor(Ak, A) # This needs an actual implementation of a subtraction operator. Or just a parallel subtract function.
    grd_Ak_wrt_Minv = pystarm.tensor_contract_all_but_one(R, Ak_hat, k, naive=False)
    grd_Ak_wrt_Akhat = tensor_times_matrix(R, M, inv_flag=True)
    # Step 3
    # Here, we need to represent Ak_hat as two different products: B = U * S, Ak_hat = B * VT
    grd_Akhat_wrt_B = pystarm.slicewise_matmul(grd_Ak_wrt_Akhat, Vkt_hat, A_transpose=False, B_transpose=True)
    grd_Akhat_wrt_VT = pystarm.slicewise_matmul(pystarm.slicewise_matmul(Uk_hat, Sk_hat), grd_Ak_wrt_Akhat, A_transpose=True, B_transpose=False)
    grd_B_wrt_U = pystarm.slicewise_matmul(grd_Akhat_wrt_B, Sk_hat, A_transpose=False, B_transpose=True)
    grd_B_wrt_S = pystarm.slicewise_matmul(Uk_hat, grd_Akhat_wrt_B, A_transpose=True, B_transpose=False)

    # Step 2
    grd_U = calc_U_gradient(grd_B_wrt_U)
    grd_S = calc_S_gradient(grd_B_wrt_S)
    grd_VT = calc_VT_gradient(grd_Akhat_wrt_VT)

    grd_AkHat = grd_U + grd_S + grd_VT
    # Step 1
    grd_Ahat_wrt_M = pystarm.tensor_contract_all_but_one(grd_AkHat, A, k, naive=False)
    
    return grd_Ahat_wrt_M + grd_Ak_wrt_Minv




    # Do A x_3 M and get jacobian of A_hat w.r.t M

    

