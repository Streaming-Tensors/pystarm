'''
Author: Stefan Smith
Institution: Argonne National Laboratory
July 2026

This is the main code for parallel star-M optimization. This code takes an input tensor as input and finds both the
optimal transformation matrix and rank-k approximation. Utilizes the pystarm library outlined in Hussain et. al 2026.

This is a parallel implementation of the algorithm outlined in Newman and Keegan's "Optimal Matrix-Mimetic Tensor Algebras via Variable Projection"
which also expands its capabilities to higher-order tensors. 
'''

import numpy as np
import math
import pystarm
import argparse 
from objective_func import low_rank_obj_func_gradient

def riemannian_gradient_descent(A_np, M_0, max_iter=100, k=1):
    '''
    Performs riemannian gradient descent to optimize transformation matrices.
    Input:
    - A_np: Pystarm tensor of order >= 3. This is your raw data.
    - M_0: Initial transformation matrices. Must be orthogonal, must be python list of pystarm matries.
    - Max_iter: Number of gradient descent iterations. Defaults to 100.
    - k: Rank approximation for A. Defaults to 1.

    Output:
    - A_k: Optimal rank-k approximation of A. Pystarm tensor
    - M_opt: Optimal transformation matrices for A. Python list of pystarm matrices.
    '''

    return low_rank_obj_func_gradient(A_np, M_0, k), [1]
    

def optimize_transformation_matrix(A: np.array, M_0, max_iter=100, k=1):
    '''
    Runs a full optimization of M. Also does sanity checks on all of the inputs to make sure nothing can go wrong. 
    '''

    # ALL ASSERTIONS
    mode = A.ndim
    assert mode >= 3, "A must be a tensor! i.e A.ndim >= 3."
    assert len(M_0) == mode - 2, f"M_0 must be a python array of mode-2 transformation matrices. len(M_0) = {len(M_0)}, mode-2 = {mode - 2}"

    dims = A.shape
    # Go through each mode >= 3 and make sure transformation matrix arrays are of valid dimensions. 
    for m in range(2, mode):
        assert dims[m] == M_0[m-2].shape[0]
        assert np.allclose(np.dot(M_0[m-2], M_0[m-2].T), np.identity(len(M_0[m-2]))), f"All M's must be orthogonal matrices. M[{m-2}] violates this."
    
    
    # Convert to pystarm for parallelization.
    A_starm = pystarm.Tensor(A, mode, dims)

    M_starm_0 = []
    for matrix in M_0:
        M_starm_0.append(pystarm.Matrix(matrix, matrix.shape[0], matrix.shape[1]))

    # Optimize!
    A_k, M_opt = riemannian_gradient_descent(A_starm, M_starm_0, max_iter, k)
    return A_k, M_opt

def initialize_vars(dims:tuple):
    '''
    Sets up the initial tensor and transformation matrix.
    Input:
    - dims: Dimensions of the starting tensor.

    Output:
    - A: Starting tensor, numpy array, with shape of `dims` 
    - M: Python array of length mode-2 with transformation matrices. Will be identity matrices.
    '''
    nelem = np.prod(dims)
    A = np.random.rand(nelem).reshape(dims)

    M = []
    # Make transformation matrices for modes >= 3
    for dim in dims[2:]:
        M.append(np.identity(dim))
    
    return A, M

def parse_arguments():
    '''
    Parse user supplied command line arugments and return their values.
    '''
    parser = argparse.ArgumentParser(description="Parallel-star-M-opt: A parallel implementation of star-M-opt for rank-k approximation of higher-order tensors.")

    parser.add_argument("-d", "--dims", default =(5,4,3,2), type=tuple, help="Dimensions of starting tensor. Must be a tuple. Default = (5,4,3,2)")
    parser.add_argument("-M", "--M", default="I", help="Starting transformation matrix. Default = 'I' ")
    parser.add_argument("-n", "--num_iter", type=int, default=100, help="Number of gradient descent iterations.")
    parser.add_argument("-k", "--k", type=int, default=1, help="Rank to approximate original tensor at.")

    parser.add_argument("--output_path", type=str, default="M-opt", help="Path to output file. Will store an optimal M as a .csv. Please provide file name without the .csv extension.")

    args = parser.parse_args()

    # Access parsed arguments
    print(f"Input file: {args.dims}")
    print(f"Starting matrix: {args.M}")
    print(f"Num iter = {args.num_iter}")
    print(f"k = {args.k}")
    print(f"Output path = {args.output_path}")

    return args.dims, args.M, args.num_iter, args.k, args.output_path


if __name__ == "__main__":
    dims, M_0, num_iter, k, output_path = parse_arguments()
    A, M = initialize_vars(dims)
    print(M)
    optimize_transformation_matrix(A, M)

