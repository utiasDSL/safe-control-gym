"""General MPC utility functions.

"""
import numpy as np
import scipy
from scipy.spatial import ConvexHull
import cdd
from pytope import Polytope
import pytope


def get_cost_weight_matrix(weights,
                           dim
                           ):
    """Gets weight matrix from input args.

    """
    if len(weights) == dim:
        W = np.diag(weights)
    elif len(weights) == 1:
        W = np.diag(weights * dim)
    else:
        raise Exception("Wrong dimension for cost weights.")
    return W


def discretize_linear_system(A,
                             B,
                             dt,
                             exact=False
                             ):
    """Discretize a linear system.
    
    dx/dt = A x + B u
    --> xd[k+1] = Ad xd[k] + Bd ud[k] where xd[k] = x(k*dt)

    Args:
        A: np.array, system transition matrix.
        B: np.array, input matrix.
        dt: scalar, step time interval.
        exact: bool, if to use exact discretization.

    Returns:
        Discretized matrices Ad, Bd.

    """
    state_dim, input_dim = A.shape[1], B.shape[1]
    if exact:
        M = np.zeros((state_dim + input_dim, state_dim + input_dim))
        M[:state_dim, :state_dim] = A
        M[:state_dim, state_dim:] = B
        Md = scipy.linalg.expm(M * dt)
        Ad = Md[:state_dim, :state_dim]
        Bd = Md[:state_dim, state_dim:]
    else:
        I = np.eye(state_dim)
        Ad = I + A * dt
        Bd = B * dt
    return Ad, Bd

def compute_min_RPI(A, wmax, eps=1e-5, s_max=50):
    """Compute the minimal robust positively invariant set (mRPI)
    
    From: Rakovic et al. (2005, IEEE Tran on Automatic Control)

    Args:
        A: np.array, system transition matrix
        w: np.array, defines the disturbance bounds |w| < wmax
        eps: float, precision criteria for stopping (eq 14 : Rakovic, 2005)
        s_max: int, max number of iterations
    """
    if abs(np.linalg.det(np.eye(A.shape[0]) + A) - 1) < 1.0e-8:
        raise NotImplementedError("mRPI for nilpotent A not yet implemented")

    W = Polytope(lb=-wmax, ub=wmax)
    return eps_MRPI(A, W, eps, s_max)

# def compute_min_RPI(A, wmax, eps=1e-5, s_max=50):

# copied and modified from https://github.com/heirung/pytope
def eps_MRPI(A, W, epsilon, s_max=20, timeout=10.0):
    """ Determines an outer epsilon-approximation of a minimal RPI set.
    
    Implements Algorithm 1 Raković et al. [1] for determining an outer 
    epsilon-approximation of the minimal RPI set for the autonomous system
        x+ = A*x + w
    using the following algorithm (copied directly from the paper and referenced 
    throughout the code):
    
        ALGORITHM 1: Computation of an RPI, outer epsilon-approximation of the MRPI 
        set F_infinity
        REQUIRE: A, W, and epsilon > 0
        ENSURE: F(alpha, s) such that 
                F_infinity <= F(alpha, s) <= F_infinity + B_infinity^n(epsilon)
            1: Choose any s in N (ideally, set s <- 0).
            2: REPEAT
            3:   Increment s by one.
            4:   Compute alpha^o(s) as in (11) and set alpha <- alpha^o(s).
            5:   Compute M(s) as in (13).
            6: UNTIL alpha <= epsilon / (epsilon + M(s))
            7: Compute F_s as the Minkowski sum (2) and scale it to give 
                 F(alpha, s) := (1 - alpha)^(-1) * F_s.
    The s-term Minkowski sum (2) is computed in V-rep; computing the sum in
    H-rep can be both slow and numerically more challenging.
    Args:
        A: A numpy array (the state transition matrix --  must be strictly stable).
        W: A Polytope instance that bounds the disturbance w (must be compact and
            contain the origin).
        epsilon: A (positive) error bound (the radius of the infinity-norm ball).
        s_max: An optional maximum value of s, at which the algorithm terminates.
    Returns:
        F_alpha_s: A Polytope instance that is the outer-epsilon approximation of
            the MRPI set for (A, W).
        result: A dict with keys
            alpha: A scalar in [0, 1]:  A^s W subset alpha W  (Eq. (4)).
            s: A positive integer:  F_alpha_s := (1 - alpha)^(-1) F_s  (Eq. (5)).
            M: A numpy array (shape (s + 1,)) of the numbers M(k), k = 0, ..., s.
                The last element, M[-1], is M(s), which satisfies
                alpha <= epsilon / (epsilon + M(s))  (Eq. (14)).
            status: 0 if the algorithm terminated successfully, otherwise -1.
            alpha_o_s: A numpy array (shape (s + 1,)) of the number alpha at
                every iteration k = 0, ..., s.
            F_s: A numpy array of Polytope instances: F_s[s] is the s-term Minkowski
                sum from i = 0 to s over A^i W  (Eq. (2)).
            eps_min: The minimal epsilon that does not require increasing s.
    Raises:
        ValueError: An argument did not satisfy a necessary condition or the support
            function could not be evaluated successfully.
    Paper reference:
    [1] Raković, S.V., Kerrigan, E.C., Kouramas, K.I., & Mayne, D.Q. (2005).
    Invariant approximations of the minimal robust positively invariant set. IEEE
    Transactions on Automatic Control, 50(3), 406-410.
    """

    status = -1  # set to 0 at successful termination (as in SciPy's linprog)

    m, n = A.shape
    if m != n:
        raise ValueError('A must be a square matrix')

    # The disturbance set W is in the form
    # W := {w in R^n | f_i' * w <= g_i, i in I}
    W.minimize_V_rep()
    F = W.A  # the columns f_i of A in the H-rep [A b] of W
    g = W.b  # the right-hand side b in the H-rep [A b] of W
    I = g.size  # the number of inequalities in the H-rep of W

    if not all(g > 0):
        raise ValueError('W does not contain the origin: g > 0 is not satisfied')

    # array of upper bounds on alpha values -- the scaling factor in the subset
    # condition A^s W subset alpha * W  (Eq. (10))
    alpha_o_s = np.full(s_max, np.nan)

    # To determine M(s) (used to bound the approximation error on F(alpha, s)):
    # Store support functions for each power of A, A^(s-1),
    # and each direction j = 1, ..., n. One row per s, each row has n support
    # functions for A^(s-1) positive and n for A^(s-1) negative; see (13).
    # M(s) is the maximum of all elements of each row s.
    # Store all values used to determine M(s) -- this is not necessary but useful
    # when debugging numerically challenging cases. Note that the first row
    # (s = 0) remains all zero (which is OK).
    M_s_row = np.zeros((s_max, 2 * n))  # each M(s) is the max over 2n values
    M = np.full(s_max, np.nan)  # M[s] is that maximum for each s

    # Pre-compute all powers of A, A^s, s = 0, ..., s_max
    A_pwr = np.stack([np.linalg.matrix_power(A, i) for i in range(s_max)])

    alpha_o = np.full(I, np.nan)

    # Step 1: Choose any s in N [natural numbers] (ideally, set s <- 0).
    s = 0

    # Step 2: repeat
    while s < s_max - 1:

        # Step 3: Increment s by one.
        s += 1

        # Step 4: Compute alpha^o(s) as in (11) and set alpha <- alpha^o(s).
        # alpha^o(s) = max_{i in I) h_W((A^s)' f_i) / g_i
        for i in range(I):
            fi = F[i, :].T
            h_W_i, status = W.support(A_pwr[s].T @ fi)
            if not status.success:
                print(f'Unsuccessful evaluation of the support function '
                            f'h_W((A^{s})'' * f_{s}): {status.message}')
            alpha_o[i] = h_W_i / g[i]
        alpha_o_s[s] = np.max(alpha_o)
        alpha = alpha_o_s[s]

        # Step 5: Compute M(s) as in (13).
        # M(s) = max_j {sum_i(h_W_sm1_pos_j), sum_i(h_W_sm1_neg_j)}  (Eq. (13))
        # At iteration s, evaluate the support for the rows of A^(s-1) and use the
        # supports evaluated at previous iterations s to evaluate the sum over i,
        # i = 0, ..., s - 1.
        h_W_sm1_pos_j = np.full(n, np.nan)  # h_W((A^(s-1))' * e_j, j = 0, ..., n-1
        h_W_sm1_neg_j = np.full(n, np.nan)  # h_W((-A^(s-1))' * e_j, j = 0, ..., n-1
        # Evaluate support in direction +- (A^i)' * e_j, with e_j the jth standard
        # basis vector in R^n. That is, (A^i)' * e_j is the jth column of (A^i)', or
        # the jth row of A^i (A_pwr_i[j])
        for j in range(n):
            A_pwr_i = A_pwr[s - 1]  # i = 0, ..., s - 1
            h_W_sm1_pos_j[j], status_lhs = W.support(A_pwr_i[j])
            h_W_sm1_neg_j[j], status_rhs = W.support(-A_pwr_i[j])
            if not all(status.success for status in (status_lhs, status_rhs)):
                raise ValueError(f'Unsuccessful evaluation of the support function in '
                                                 f'the direction of row {j} of A^{s - 1} (s = {s})')
        # Store all 2n support-function evaluations for this iteration s. That is,
        # {h_W((A^(s-1))' * e_j,  h_W((-A^(s-1))' * e_j}, j = 0, ..., n-1:
        M_s_row[s] = M_s_row[s - 1] + np.concatenate((h_W_sm1_pos_j, h_W_sm1_neg_j))
        # Take the sum over i from 0 to s - 1 (so include row s, hence ": s + 1"
        # M_s_argument = np.sum(M_s_row[s], axis=0)
        M[s] = np.max(M_s_row[s])  # Eq. (13), see above

        # Step 6: until alpha <= epsilon / (epsilon + M(s))
        print('alpha: {} break {}'.format(alpha, epsilon / (epsilon + M[s])))
        if alpha <= epsilon / (epsilon + M[s]):
            status = 0  # success
            break

    s_final = s
    print('compute F_s')
    # Step 7: Compute F_s as the Minkowski sum (2) and scale it to give
    # F(alpha, s) = (1 - alpha)^(-1) F_s.
    # F_s = sum_{i = 0}^{s - 1} A^i W,  F_0 = {0}  (Eq. (2))
    F_s = np.full(s_final + 1, Polytope(n=n))  # F_s, s = 0, ..., s_final
    for s in range(1, s_final + 1):  # determine F_s for s = 1, ..., s_final
        print(s)
        F_s[s] = F_s[s - 1] + A_pwr[s - 1] * W  # F_s[0] is empty
        print(F_s[s].b.shape)
        minimize_V_rep(F_s[s])  # critical when s_final is large
    # Scale to obtain the epsilon-approximation of the minimal RPI:
    F_alpha_s = F_s[s_final] * (1 / (1 - alpha))
    # TODO: Improve performance for large s_final by not constructing polytopes
    # for every s -- instead compute the vertices directly for every power of A
    # and add them together at the end (and finally remove redundant vertices)

    # The smallest epsilon for s_final terms in the Minkowski sum:
    eps_min = M[s_final] * alpha / (1 - alpha)

    result = {'alpha': alpha, 's': s_final, 'M': M[: s_final + 1],
                        'status': status, 'alpha_o_s': alpha_o_s[: s_final + 1], 'F_s': F_s,
                        'eps_min': eps_min}

    return F_alpha_s, result

#copied and modified from https://github.com/heirung/pytope
def minimize_V_rep(P):
    # Minimize the number of vertices used to represent the polytope by removing
    # redundant points from the vertex list.
    # TODO: find and account for cases where this does not work
    # (1) Qhull does not support degenerate polytopes (not full dimensional),
    # such as lines in n = 2 amd planes in n = 3. ("QhullError: Raised when
    # Qhull encounters an error condition, such as geometrical degeneracy when
    # options to resolve are not enabled.")
    # (2) Qhull does not support 1-D polytopes ("ValueError: Need at least 2-D
    # data")
    if P.is_full_dimensional and P.n >= 2 and False:
        print('convex')
        # Indices of the unique vertices forming the convex hull:
        i_V_minimal = ConvexHull(P.V).vertices
        P.V = P.V[i_V_minimal, :]
    else:  # TODO: Use cdd for both cases?
        # cdd handles this case
        V_cdd = cdd.Matrix(np.hstack((np.ones((P.nV, 1)), P.V)),
                                             number_type='float')
        V_cdd.rep_type = cdd.RepType.GENERATOR  # specify that this is V-rep
        V_cdd.canonicalize()  # removes redundant vertices
        # TODO: the lines below are copied from determine_V_rep() -- refactor
        tV = np.array(V_cdd)
        if tV.any():  # tV == [] if the Polytope is empty
            V_rows = tV[:, 0] == 1  # bool array of which rows contain vertices
            R_rows = tV[:, 0] == 0  # and which contain rays (~ V_rows)
            V = tV[V_rows, 1:]  # array of vertices (one per row)
            R = tV[R_rows, 1:]  # and of rays
            if R_rows.any():
                raise NotImplementedError('Support for rays not implemented')
        else:
            V = np.empty((0, P.n))
        P._set_V(V)

