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
        # inv_flag still needs to be implemented. line below needs inv_flag=True
        Ak = pystarm.ttm(Ak_hat, M, mode) 

        # Backwards pass
        # Step 4
        R = pystarm.tensor_minus_tensor(A, Ak) # This needs an actual implementation of a subtraction operator. Or just a parallel subtract function.
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
        grd_U = calc_U_gradient(Uk_hat, Sk_hat, Vkt_hat, grd_B_wrt_U, k)
        grd_S = calc_S_gradient(Uk_hat, Vkt_hat, grd_B_wrt_S)
        grd_VT = calc_VT_gradient(grd_Akhat_wrt_VT)

        grd_AkHat = pystarm.tensor_plus_tensor(grd_U, grd_S, grd_VT)
        # Step 1
        grd_Ahat_wrt_M = pystarm.tensor_contract_all_but_one(grd_AkHat, A, k, naive=False)
    
        MOpts.append(grd_Ahat_wrt_M + grd_Ak_wrt_Minv)
        mode +=1
    return MOpts
    




    # Do A x_3 M and get jacobian of A_hat w.r.t M

    

