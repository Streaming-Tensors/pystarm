'''
Author: Stefan Smith
Institution: Argonne National Laboratory
July 2026

This is the main code for parallel star-M optimization. This code takes an input tensor as input and finds both the
optimal transformation matrix and rank-k approximation. Utilizes the pystarm library outlined in Hussain et. al 2026.

This is a parallel implementation of the algorithm outlined in Newman and Keegan's "Optimal Matrix-Mimetic Tensor Algebras via Variable Projection"
which also expands its capabilities to higher-order tensors. 
'''

# Since pystarm is in the parent directory, we need to do this to import it.
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pystarm

import numpy as np
import math

import argparse
from objective_func import low_rank_objective_and_gradient
from optimizer import riemannian_gradient_descent as _descend


def riemannian_gradient_descent(A_starm, M_0, max_iter=100, k=1, tol=1e-8,
                                verbose=True):
    '''
    Optimizes the transformation matrices by Riemannian gradient descent.
    Input:
    - A_starm: Pystarm Tensor of order >= 3. This is your raw data. Not freed here.
    - M_0: Initial transformation matrices. Must be orthogonal. Python list of
      numpy arrays, one per transform mode (modes 3 and above).
    - max_iter: Number of gradient descent iterations per mode. Defaults to 100.
    - k: Rank approximation for A. Defaults to 1.
    - tol: Tolerance on the tangent gradient norm. Defaults to 1e-8.
    - verbose: Print per-iteration progress. Defaults to True.

    Output:
    - M_opt: Optimal transformation matrices. Python list of numpy arrays.
    - info: Dict with the final objective and the per-mode stopping information.

    For a third-order tensor there is one transform matrix and this is plain
    gradient descent. For higher orders there are several, and they are
    optimized by sweeping over the modes, holding the others fixed while one is
    updated. That mirrors optimizers/alternatingDescent.m in Newman and
    Keegan's reference implementation. Each sweep is monotone in the objective,
    since every inner descent is, so the sweep cannot make things worse.
    '''
    Ms = [np.asfortranarray(np.array(M, dtype=np.float64, copy=True))
          for M in M_0]
    nmodes = len(Ms)

    def wrap(Mnp):
        return pystarm.Matrix(np.asfortranarray(Mnp), Mnp.shape[0], Mnp.shape[1])

    def objective_for(j):
        def f(Mnp):
            trial = list(Ms)
            trial[j] = Mnp
            val, _ = low_rank_objective_and_gradient(
                A_starm, [wrap(M) for M in trial], k, want_gradient=False)
            return val
        return f

    def gradient_for(j):
        def g(Mnp):
            trial = list(Ms)
            trial[j] = Mnp
            _, grads = low_rank_objective_and_gradient(
                A_starm, [wrap(M) for M in trial], k)
            return grads[j]
        return g

    # One sweep for a single transform; several for a higher-order tensor,
    # where the modes are coupled and one pass over them is not enough.
    nsweeps = 1 if nmodes == 1 else max(2, math.ceil(max_iter / 25))
    per_mode = max_iter if nmodes == 1 else max(10, max_iter // nsweeps)

    info = {'modes': []}
    for sweep in range(nsweeps):
        for j in range(nmodes):
            if verbose and nmodes > 1:
                print(f'--- sweep {sweep + 1}/{nsweeps}, mode {j + 3}')
            Ms[j], sub = _descend(objective_for(j), gradient_for(j), Ms[j],
                                  max_iter=per_mode, tol=tol, verbose=verbose)
            info['modes'].append({'sweep': sweep, 'mode': j, **sub})

    info['f'] = info['modes'][-1]['f'] if info['modes'] else None
    return Ms, info



def optimize_transformation_matrix(A: np.array, M_0, max_iter=100, k=1):
    '''
    Runs a full optimization of M. Also does sanity checks on all of the inputs to make sure nothing can go wrong. 
    Inputs:
    - A: numpy array/tensor. This is the data you want the transformation matrices to be optimized to.
    - M_0: A python list of numpy matrices. These are your initial transformation matrices.
    - max_iter: Default is 100. This is the number of gradient descent iterations you want to run.
    - k: Default is 1. The rank you want to approximate at. 
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
    
    # Convert to pystarm for parallelization. Column-major is required: the
    # wrapping constructors read every buffer as Fortran-ordered and now raise
    # rather than silently interpreting a C-ordered array as its transpose.
    A_f = np.asfortranarray(A, dtype=np.float64)
    A_starm = pystarm.Tensor(A_f, mode, list(dims))

    # Optimize. The transform matrices stay as numpy arrays across iterations;
    # they are wrapped per evaluation, which costs O(n^2) against an objective
    # that is O(size of the tensor), and avoids the ownership trap where
    # clearing a Matrix that wraps a numpy buffer double-frees.
    M_opt, info = riemannian_gradient_descent(A_starm, M_0, max_iter, k)

    # Build the rank-k approximation at the optimum, in the original domain.
    A_k = reconstruct(A_starm, M_opt, k)

    # A_starm wraps A_f, which numpy owns: do not clear it.
    return A_k, M_opt, info


def reconstruct(A_starm, Ms, k):
    '''
    Builds the rank-k star-M approximation of A at the given transforms.
    Input:
    - A_starm: Pystarm Tensor of order >= 3. Not freed here.
    - Ms: Python list of numpy transform matrices, one per transform mode.
    - k: Target rank.

    Output:
    - A_k: The approximation as a numpy array with A's shape.
    '''
    modes = list(range(2, len(A_starm.getdims())))
    wrapped = [pystarm.Matrix(np.asfortranarray(M), M.shape[0], M.shape[1])
               for M in Ms]

    owned = []
    cur = A_starm
    for j, mode_idx in enumerate(modes):            # into the transform domain
        cur = pystarm.ttm(cur, wrapped[j], mode_idx)
        owned.append(cur)

    Uk, Sk, Vkt = pystarm.slicewise_svdx(cur, k)
    Ak_hat = pystarm.slicewise_matmul(Uk, Sk, Vkt)
    owned += [Uk, Sk, Vkt, Ak_hat]

    cur = Ak_hat
    for j, mode_idx in reversed(list(enumerate(modes))):   # and back out again
        cur = pystarm.ttm(cur, wrapped[j], mode_idx, True)  # M_transpose=True
        owned.append(cur)

    out = np.array(
        np.frombuffer(cur, dtype=np.float64).reshape(cur.getdims(), order='F'),
        copy=True, order='F')

    for t in owned:
        t.clear()
    return out

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

