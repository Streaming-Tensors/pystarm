'''
Armijo backtracking line search on the orthogonal group.

Follows Algorithm 4.2 of Boumal, "An Introduction to Optimization on Smooth
Manifolds" (2023), and mirrors the structure of `optimizers/linesearch/
armijoLinesearch.m` in Newman and Keegan's reference implementation
(github.com/elizabethnewman/star-M-opt), so the two can be compared directly.

Each trial iterate is formed by retracting along a skew-symmetric direction:

    M(mu) = M0 @ R(mu * S)

where R is the matrix exponential by default, which is the exact geodesic, or
the Cayley transform if selected. Both land on the orthogonal group to working
precision.

Two deliberate departures from the MATLAB reference, both argued from
measurement in the survey notes under
/storage/vhebbur/survey/research_notes/:

1.  The reference has an escape hatch that replaces the search direction with
    the raw Euclidean gradient when the directional derivative is not negative:

        if tau >= 0, s = -g0

    That branch is dead on the Stiefel/orthogonal path, because the direction
    is built as s = 0.5*(A - A') and tau = -||s||_F^2 <= 0 identically. If it
    ever did fire it would feed a non-skew matrix to the retraction and throw
    the iterate clean off the manifold; it was measured at ||M'M - I|| = 4.1e2.
    It is a vestige of the Euclidean case and is not ported. Instead a
    non-negative tau is reported as a termination condition, which is what it
    actually means here: the tangent gradient has underflowed to zero.

    Note the subtlety that motivated making this explicit: `-0.0 >= 0` is True
    in IEEE arithmetic, so an underflowed ||s||^2 would have fired the branch.

2.  Reuse of a single decomposition across trial steps is available but OFF by
    default. Backtracking is sequential, so the number of trials is not known
    in advance, and the measured crossover where decomposing once pays for
    itself is far beyond the number of backtracks actually observed, which is
    one to four and averages under two. Pass reuse=True only if your problem
    genuinely backtracks deeply.

Memory: pystarm objects own C++ memory and must be freed explicitly with
.clear(). This module frees every intermediate it creates. The accepted
iterate is returned to the caller, who then owns it.
'''

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pystarm


# Termination flags, following the reference's convention where 1 is success.
FLAG_SUCCESS = 1
FLAG_MAX_ITER = -1
FLAG_STEP_UNDERFLOW = -2
FLAG_NOT_DESCENT = -3   # not in the reference; see note 1 in the module docstring


def _frobenius_norm_squared(M):
    '''Squared Frobenius norm of a pystarm Matrix, without copying.

    Matrix exposes the buffer protocol but, unlike Tensor, does not expose a
    norm method, so this reads the buffer directly. The buffer is flat and
    column-major, and a Frobenius norm does not care about the ordering.
    '''
    flat = np.frombuffer(M, dtype=np.float64)
    return float(np.dot(flat, flat))


class ArmijoLinesearch:
    '''Backtracking line search with an adaptive initial step.

    Parameters mirror the reference implementation:

      alpha      initial trial step, adapted between calls
      beta       backtracking factor, the step is multiplied by this on reject
      gamma      sufficient-decrease constant in the Armijo test
      max_iter   maximum backtracking steps per call
      alpha_tol  step size below which the search is considered to have failed

    The adaptive rule is the reference's: on a first-try success the next call
    starts at twice the accepted step, otherwise it starts at the accepted
    step. This is what makes the step size track the local geometry instead of
    restarting from 1 every iteration.
    '''

    def __init__(self, alpha=1.0, beta=0.5, gamma=1e-4, max_iter=20,
                 alpha_tol=1e-14, retraction=None, reuse=False, verbose=False):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.max_iter = max_iter
        self.alpha_tol = alpha_tol
        self.retraction = (pystarm.RETRACT_EXPM if retraction is None
                           else retraction)
        self.reuse = reuse
        self.verbose = verbose

    # -- retraction helpers ------------------------------------------------

    def _retract_direct(self, S, mu):
        '''R(mu*S) computed from scratch. Returns a Matrix the caller frees.'''
        if self.retraction == pystarm.RETRACT_CAYLEY:
            return pystarm.cayley(S, mu)
        return pystarm.expm(S, mu)

    def _retract_from_decomposition(self, Q, theta, mu):
        '''R(mu*S) from a precomputed skew Schur factorization.'''
        if self.retraction == pystarm.RETRACT_CAYLEY:
            return pystarm.cayley_skew_apply(Q, theta, mu)
        return pystarm.expm_skew_apply(Q, theta, mu)

    # -- the search --------------------------------------------------------

    def step(self, objective, M0, S, f0):
        '''
        Takes one backtracking line search step along a skew-symmetric direction.
        Input:
        - objective: Callable taking a pystarm Matrix and returning a float. Must not
          free or retain the matrix it is given.
        - M0: Current iterate. Orthogonal pystarm Matrix. Not freed by this function.
        - S: Search direction. Skew-symmetric pystarm Matrix. Not freed by this function.
        - f0: Objective value at M0, passed in so it is not recomputed here.

        Output:
        - M: The new iterate, a pystarm Matrix the caller owns and must .clear().
          None if no acceptable step was found, in which case the caller should keep
          its existing iterate.
        - info: Dict with 'alpha' (accepted step), 'ls_iter' (backtracks used),
          'flag' (see the FLAG_ constants) and 'f' (objective at the new iterate).
        '''
        # tau is the exact directional derivative of the objective along the
        # retraction at mu = 0, for any retraction with R'(0) = I. Because the
        # direction is skew and the metric is the Frobenius inner product, it
        # reduces to minus the squared norm, with no gradient needed.
        tau = -_frobenius_norm_squared(S)

        if not tau < 0.0:
            # The tangent gradient has underflowed. See note 1 above: the
            # reference would switch directions here; we report convergence.
            if self.verbose:
                print('armijo: direction has zero norm, treating as converged')
            return None, {'alpha': 0.0, 'ls_iter': 0,
                          'flag': FLAG_NOT_DESCENT, 'f': f0}

        Q = None
        theta = None
        if self.reuse:
            # One decomposition serves every trial step. Off by default; see
            # note 2 in the module docstring.
            Q, theta = pystarm.skew_schur(S)

        mu = self.alpha
        accepted = None
        f_accepted = f0
        it = 1

        try:
            while it <= self.max_iter:
                if self.reuse:
                    R = self._retract_from_decomposition(Q, theta, mu)
                else:
                    R = self._retract_direct(S, mu)

                M = pystarm.matmul(M0, R)
                R.clear()

                f = objective(M)

                if self.verbose:
                    print(f'armijo: iter {it:2d}  mu {mu:.6e}  f {f:.12e}  '
                          f'target {f0 + mu * self.gamma * tau:.12e}')

                # Armijo sufficient decrease. Strict, matching the reference.
                if f < f0 + mu * self.gamma * tau:
                    accepted = M
                    f_accepted = f
                    break

                M.clear()
                mu *= self.beta
                it += 1
        finally:
            if Q is not None:
                Q.clear()

        if accepted is None:
            flag = FLAG_MAX_ITER
        elif mu < self.alpha_tol:
            # Accepted, but on a step so small the search has effectively
            # stalled. The caller should treat this as a termination signal.
            flag = FLAG_STEP_UNDERFLOW
        else:
            flag = FLAG_SUCCESS

        # Adapt the starting step for the next call, as the reference does.
        self.alpha = 2.0 * mu if it == 1 else mu

        return accepted, {'alpha': mu, 'ls_iter': it, 'flag': flag,
                          'f': f_accepted}
