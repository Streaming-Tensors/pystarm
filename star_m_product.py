import pystarm
import numpy as np
import pyttb as ttb

def starM_product(A_np, B_np, M_np, A_transpose=False, B_transpose=False):
    '''
    This function computes the starM Product of any given order >= 3 input tensor and an
    orthogonal transformation matrix M.

    A and B must be numpy tensors, M must be a python array of numpy matrices.

    Returns C, starM product of A and B
    '''
    # Go ahead and calculate M^(-1), will be used in the last step.
    # Since all matrices are orthogonal, no need to make a list of inverse matrices. just pass in the transpose.
    A_ndim = A_np.ndim if not A_transpose else A_np.T.ndim
    B_ndim = B_np.ndim if not B_transpose else B_np.T.ndim

    # Make it such that for order 3, user does not have to pass in a list of numpy arrays for M.
    if(type(M_np) != list and A_ndim == 3):
        M_np = [M_np]

    assert A_ndim >= 3, "A must be a tensor with ndim >= 3"
    assert B_ndim >= 3, "B must be a tensor with ndim >= 3"
    
    A_shape = A_np.shape if not A_transpose else A_np.T.shape
    B_shape = B_np.shape if not B_transpose else B_np.T.shape

    # Ensure dimensions are correct.
    assert A_shape[A_ndim - 1] == B_shape[B_ndim - 1], "A and B must have the same number of slices."
    assert A_ndim == len(M_np) + 2, "You must have (order - 2) transformation matrices inside of a python list. I.e. order 4 must have a list with 2 transformation matrices."
    assert np.allclose(np.dot(M_np[0], M_np[0].T), np.identity(len(M_np[0]))), "M must be an orthogonal matrix."

    # Convert from numpy to starM Tensor/Matrix for pystarm API. args are array (numpy), ndim/order, and dimensions
    A = pystarm.Tensor(A_np, A_ndim, A_shape) # 
    B = pystarm.Tensor(B_np, B_ndim, B_shape)


    # Do initial mode 3 calculation, prevents A_hat and B_hat from going out of range
    M_shape = M_np[0].shape
    M = pystarm.Matrix(M_np[0], M_shape[0], M_shape[1])

    # Move A and B into transform domain
    current_mode = 2 # start with mode 3-1=2
    A_hat = pystarm.ttm(A, M, current_mode) # Args: tensor, matrix, mode (index 0, so if you put 1 it will be mode 1+1=2)
    B_hat = pystarm.ttm(B, M, current_mode)
    # For higher orders (>= 4), move those modes into transform domain. 
    if A_ndim > 3:
        for matrix in M_np[1:]:
            current_mode += 1
            # Check for M's orthogonality.
            assert np.allclose(np.dot(matrix, matrix.T), np.identity(len(matrix))), "All M's must be orthogonal matrices."
            M_shape = matrix.shape
            M = pystarm.Matrix(matrix, M_shape[0], M_shape[1])
            # Move A and B into transform domain
            A_hat = pystarm.ttm(A_hat, M, current_mode) # Args: tensor, matrix, mode (index 0, so if you put 1 it will be mode 1+1=2)
            B_hat = pystarm.ttm(B_hat, M, current_mode)

    # Slicewise multiply A and B
    C_hat = pystarm.slicewise_matmul(A_hat, B_hat, A_transpose, B_transpose)

    current_mode = 2 # reset mode counter
    # Return product to original domain. NOTE: Since all M's are orthogonal, the inverse = transpose. That's what I'm doing here.
    M_inv = pystarm.Matrix(M_np[0].T, M_np[0].shape[1], M_np[0].shape[0])
    C = pystarm.ttm(C_hat, M_inv, current_mode)

    if A_ndim > 3:
        for matrix in M_np[1:]:
            current_mode += 1
            M_inv = pystarm.Matrix(matrix.T, matrix.shape[1], matrix.shape[0])
            C = pystarm.ttm(C, M_inv, current_mode)
    
    # Return to numpy
    C_np = np.frombuffer(C, dtype=np.float64).reshape(C.getdims(), order='F', copy = False)
    
    return C_np

import pyttb
import numpy as np

def facewise_product_pyttb(A, B):
    """
    Compute the facewise (frontal-slice-wise) matrix product of two pyttb tensors.
    
    A: pyttb.tensor
    B: pyttb.tensor
    
    Returns: pyttb.tensor
    """
    # Extract underlying numpy arrays
    A_data = A.data 
    B_data = B.data  

    ndim = A_data.ndim

    # Move frontal slice indicies to the front for batched matmul
    # Build the permutation: [2, 3, ..., ndim-1, 0, 1]
    batch_axes = list(range(2, ndim))
    perm_forward = batch_axes + [0, 1]
    
    A_t = np.transpose(A_data, perm_forward)
    B_t = np.transpose(B_data, perm_forward)

    # Batched matrix multiply over all leading (batch) dimensions
    C_t = np.matmul(A_t, B_t)

    # Invert the permutation: move last two axes back to positions 0, 1
    # From (K1, K2, ..., Kn, I, L) -> (I, L, K1, K2, ..., Kn)
    perm_back = [ndim - 2, ndim - 1] + list(range(ndim - 2))
    
    C_data = np.transpose(C_t, perm_back)

    return pyttb.tensor(C_data)

def starM_product_ttb(A_np, B_np, M_np):
    '''
    Compute the star-M product using the pyttb library. Used for the test.py script to verify the results of `starM_product`.
    Inputs: 
    - A_np, B_np - numpy tensors (3D ndarray) with ndim >= 3. 
    - M_np, if ndim == 3, it can be a single numpy matrix. If ndim > 3, must be a python list with ndim - 2 numpy matrices.

    Returns:
    - C -> starM product of A and B, numpy tensor (3D ndarray).
    '''
    # Move to transform domain + calculate facewise product
    ndim = A_np.ndim
    if(type(M_np) != list and ndim == 3):
        M_np = [M_np]
    
    # Move mode 3 into transform domain. 
    current_mode = 2 # Since python does index 0, so mode 3 = 3-1=2
    A_hat = ttb.tensor(A_np, copy=True).ttm(M_np[0], current_mode)
    B_hat = ttb.tensor(B_np, copy=True).ttm(M_np[0], current_mode)

    # Move modes (4, ndim) into transform domain.
    if ndim > 3:
        for matrix in M_np[1:]:
            current_mode += 1
            # Check for M's orthogonality.
            assert np.allclose(np.dot(matrix, matrix.T), np.identity(len(matrix))), "All M's must be orthogonal matrices."

            # Move A and B into transform domain for current_mode
            A_hat = A_hat.ttm(matrix, current_mode)
            B_hat = B_hat.ttm(matrix, current_mode)

    C_hat = facewise_product_pyttb(A_hat, B_hat)

    # Move back to original domain and convert to numpy array
    current_mode = 2
    C = C_hat.ttm(M_np[0].T, current_mode)
    if ndim > 3:
        for matrix in M_np[1:]:
            current_mode += 1
            C = C.ttm(matrix, current_mode)

    return C.data

# Main

A_nelm = 120
A_dims = (5,4,3,2)
A_ndim = len(A_dims)

B_nelm = 120
B_dims = (4,5,3,2)
B_ndim = len(B_dims)

# Create numpy tensors
A_np = np.random.rand(A_nelm).reshape(A_dims, order='F')
B_np = np.arange(B_nelm, dtype=np.float64).reshape(B_dims, order='F')

# Create identity matrix as M
M_np = [np.identity(n) for n in A_dims[2:]]

# print(f"{A_ndim} and {len(M)}")

print(f"A = {A_np}\n\n-----------\nB = {B_np}\n\n-----------\n")
C_ttb = starM_product_ttb(A_np, B_np, M_np)
C_np = starM_product(A_np,B_np,M_np)

print(f"C_ttb: \n\n{C_ttb}\n\nC_np:\n\n{C_np}")
print(np.allclose(C_ttb, C_np))
    
