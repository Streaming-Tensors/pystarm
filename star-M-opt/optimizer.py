'''
Riemannian gradient descent on the orthogonal group.

Mirrors the structure of `optimizers/gradientDescent.m` in Newman and Keegan's
reference implementation (github.com/elizabethnewman/star-M-opt), driving the
Armijo line search in `linesearch.py`.

The optimizer is deliberately decoupled from the t-SVDM objective: it takes
callables for the objective and its Euclidean gradient. That keeps it testable
against problems with known minimisers, and means it does not block on
`objective_func.py` being finished.

One deliberate departure from the reference, and it is not cosmetic.

The reference stops when the EUCLIDEAN gradient norm falls below a relative
tolerance of 1e-10. On the orthogonal group that quantity does not vanish at a
stationary point: the Euclidean gradient generally has a nonzero normal
component there. Measured on the reference implementation, it sat flat at
1.478 for twenty-two iterations while the tangent gradient fell from 1.3 to
4.3e-8, so the stated tolerance never fired and the line search breaking was
the de facto convergence test. See tsvdm_accuracy_requirements.md in the
survey notes.

This implementation stops on the norm of the tangent direction instead, which
is the quantity that actually vanishes at a stationary point. The line-search
failure remains a secondary termination path, but it is no longer the only one.
'''

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pystarm

from linesearch import ArmijoLinesearch, FLAG_SUCCESS


# Why the optimizer stopped.
STOP_TANGENT_TOL = 'tangent gradient below tolerance'
STOP_RELATIVE_TOL = 'tangent gradient fell below relative tolerance'
STOP_LINESEARCH = 'line search could not find a decreasing step'
STOP_MAX_ITER = 'maximum iterations reached'


def tangent_direction(M, euclidean_grad):
    '''Riemannian steepest-descent direction at M.

    Built exactly as the reference does: A = -M^T g, then the skew part
    0.5*(A - A^T). The result is skew-symmetric bit-exactly in IEEE double,
    which is what lets the retraction take the structure-exploiting path.
    '''
    A = -(M.T @ euclidean_grad)
    return 0.5 * (A - A.T)


def orthogonality_error(M):
    '''||M^T M - I||_2, the quantity that must stay small for M^T to serve as
    the inverse transform.'''
    n = M.shape[0]
    return float(np.linalg.norm(M.T @ M - np.eye(n), 2))


def reorthogonalize(M):
    '''One Newton-Schulz step, which the survey measured as both faster and
    more accurate than an SVD-based polar correction.'''
    return M @ (1.5 * np.eye(M.shape[0]) - 0.5 * (M.T @ M))


def riemannian_gradient_descent(objective, euclidean_gradient, M0,
                                max_iter=100, tol=1e-8, rel_tol=1e-10,
                                linesearch=None, reortho_tol=1e-13,
                                verbose=True, history=False):
    '''
    Minimises an objective over the orthogonal group by Riemannian gradient descent.
    Input:
    - objective: Callable taking an (n,n) numpy array and returning a float.
    - euclidean_gradient: Callable taking an (n,n) numpy array and returning the
      ordinary (unprojected) gradient as an (n,n) numpy array.
    - M0: Starting iterate. Must be orthogonal, shape (n,n).
    - max_iter: Outer iteration cap. Defaults to 100.
    - tol: Absolute tolerance on the tangent gradient norm. Defaults to 1e-8.
    - rel_tol: Tolerance on the tangent gradient norm relative to its initial value.
      This is the reference's criterion applied to the correct quantity; see the
      module docstring. Defaults to 1e-10.
    - linesearch: An ArmijoLinesearch instance, or None for the default.
    - reortho_tol: Re-orthogonalize whenever ||M^T M - I|| exceeds this. Pass None
      to disable. Defaults to 1e-13.
    - history: If True, also return a list of per-iteration records.

    Output:
    - M: The optimized orthogonal matrix, an (n,n) numpy array.
    - info: Dict with 'f', 'gnorm', 'gnorm_rel', 'iterations', 'stop_reason',
      'orthogonality_error' and 'reorthogonalizations'.
    - records: Only when history is True. List of per-iteration dicts.
    '''
    M = np.asfortranarray(np.array(M0, dtype=np.float64, copy=True))
    n = M.shape[0]
    if M.shape[0] != M.shape[1]:
        raise ValueError('M0 must be square')

    err0 = orthogonality_error(M)
    if err0 > 1e-8:
        raise ValueError(f'M0 is not orthogonal: ||M^T M - I|| = {err0:.3e}')

    ls = ArmijoLinesearch() if linesearch is None else linesearch

    f = float(objective(M))
    S = tangent_direction(M, euclidean_gradient(M))
    gnorm = float(np.linalg.norm(S, 'fro'))
    gnorm0 = gnorm if gnorm > 0 else 1.0

    records = []
    stop = STOP_MAX_ITER
    n_reortho = 0

    if verbose:
        print(f"{'iter':>5} {'f':>16} {'||grad_tan||':>14} {'step':>10} "
              f"{'ls':>3} {'ortho':>10}")
        print(f"{0:5d} {f:16.8e} {gnorm:14.6e} {'':>10} {'':>3} "
              f"{orthogonality_error(M):10.2e}")

    for it in range(1, max_iter + 1):
        if gnorm <= tol:
            stop = STOP_TANGENT_TOL
            break
        if gnorm / gnorm0 <= rel_tol:
            stop = STOP_RELATIVE_TOL
            break

        # Hand the current iterate and direction to the line search. Both wrap
        # NumPy buffers, so neither may be cleared; only the returned matrix is
        # ours to free.
        M_buf = np.asfortranarray(M)
        S_buf = np.asfortranarray(S)
        M_mat = pystarm.Matrix(M_buf, n, n)
        S_mat = pystarm.Matrix(S_buf, n, n)

        def obj_on_matrix(X_mat):
            X = np.frombuffer(X_mat, dtype=np.float64).reshape((n, n), order='F')
            return float(objective(X))

        M_new, ls_info = ls.step(obj_on_matrix, M_mat, S_mat, f)

        if M_new is None:
            stop = STOP_LINESEARCH
            if verbose:
                print(f'  line search failed at iteration {it} '
                      f'(flag {ls_info["flag"]})')
            break

        M = np.array(
            np.frombuffer(M_new, dtype=np.float64).reshape((n, n), order='F'),
            copy=True, order='F')
        M_new.clear()

        f = ls_info['f']

        # Guard the property the whole pipeline depends on: M^T is used as the
        # inverse transform, so drift off the group is an error floor on every
        # reconstruction. Correcting it is O(n^3) and rare, so it is cheap
        # insurance.
        if reortho_tol is not None:
            oe = orthogonality_error(M)
            if oe > reortho_tol:
                M = np.asfortranarray(reorthogonalize(M))
                n_reortho += 1

        S = tangent_direction(M, euclidean_gradient(M))
        gnorm = float(np.linalg.norm(S, 'fro'))

        if history:
            records.append({'iter': it, 'f': f, 'gnorm': gnorm,
                            'alpha': ls_info['alpha'],
                            'ls_iter': ls_info['ls_iter'],
                            'ortho': orthogonality_error(M)})

        if verbose and (it <= 3 or it % 10 == 0):
            print(f"{it:5d} {f:16.8e} {gnorm:14.6e} {ls_info['alpha']:10.3e} "
                  f"{ls_info['ls_iter']:3d} {orthogonality_error(M):10.2e}")

    info = {'f': f, 'gnorm': gnorm, 'gnorm_rel': gnorm / gnorm0,
            'iterations': it, 'stop_reason': stop,
            'orthogonality_error': orthogonality_error(M),
            'reorthogonalizations': n_reortho}

    if verbose:
        print(f'stopped after {it} iterations: {stop}')
        print(f'  f = {f:.8e}, ||grad_tan|| = {gnorm:.6e}, '
              f'ortho error = {info["orthogonality_error"]:.3e}, '
              f'reorthogonalizations = {n_reortho}')

    return (M, info, records) if history else (M, info)
