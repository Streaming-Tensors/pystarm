import numpy as np
import pystarm

def tensor_times_matrix(A, M, inv_flag=False):
    '''
    Performs a tensor-times-matrix (TTM) operation for modes >= 3.
    Input:
    - A: pystarm tensor
    - M: python list of pystarm matrices.

    Returns:
    - A_hat: A moved into the transform domain.
    - _Ahat_A: Jacobian of A_hat w.r.t A
    - Jac_Ahat_M: Jacobian of A_hat w.r.t M
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
    A_k_hat = pystarm.slicewise_matmul(Uk_hat, Sk_hat, Vkt_hat)
    A_k = tensor_times_matrix(A, M, inv_flag=True)




    # Do A x_3 M and get jacobian of A_hat w.r.t M

    

