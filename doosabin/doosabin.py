# doosabin.py

# Imports
import numpy as np

# doosabin_weights
def doosabin_weights(N):
    assert N >= 3
    # Calculate the weights
    i = np.arange(N)
    weights = (3 + 2 * np.cos(2 * np.pi * i / N)) / (4 * N)
    weights[0] = (N + 5.0)/(4.0 *N)
    return weights

# extended_subdivision_matrix
def extended_subdivision_matrix(N):
    assert N >= 3
    a = doosabin_weights(N)
    c, d, e, _ = doosabin_weights(4)
    A = []
    for i in xrange(N):
        r = np.roll(a, i).tolist()
        r.extend([0.0] * 5)
        A.append(r)
    A.append([d] + [0.0] * (N - 2) + [c, d, e] + [0.0] * 3)
    A.append([c] + [0.0] * (N - 2) + [d, e, d] + [0.0] * 3)
    A.append([c] + [0.0] * N + [d, e, d] + [0.0])
    A.append([c, d] + [0.0] * (N + 1) + [d, e])
    A.append([d, c] + [0.0] * (N + 1) + [e, d])
    return np.array(A, dtype=np.float64)

# bigger_subdivision_matrix
def bigger_subdivision_matrix(N):
    assert N >= 3
    c, d, e, _ = doosabin_weights(4)
    return np.r_['0,2',
        extended_subdivision_matrix(N),
        [e] + [0.0] * (N - 2) + [d, c, d] + [0.0] * 3,
        [d] + [0.0] * (N - 2) + [e, d, c] + [0.0] * 3,
        [d] + [0.0] * N + [c, d, e] + [0.0],
        [e] + [0.0] * N + [d, c, d] + [0.0],
        [d] + [0.0] * N + [e, d, c] + [0.0],
        [d, e] + [0.0] * (N + 1) + [c, d],
        [e, d] + [0.0] * (N + 1) + [d, c]]

# picker_matrix
PICKING_INDICES = [
    lambda N: [N + 3, N + 4, 1, 0, N + 1, N + 2, N + 9, N + 10, N + 11],
    lambda N: [N + 2, N + 3, 0, N + 1, N + 6, N + 7, N + 8, N + 9, N + 10],
    lambda N: [N + 1, 0, N - 1, N, N + 5, N + 6, N + 7, N + 2, N + 3],
    lambda N: range(N + 5),
]
def picker_matrix(N, k):
    M = N + 12
    j = PICKING_INDICES[k](N)
    n = len(j)
    P = np.zeros((n, M), dtype=np.float64)
    for i in xrange(n):
        P[i, j[i]] = 1
    return P

# uniform_quadratic_bspline_position_basis
def uniform_quadratic_bspline_position_basis(u, k):
    u = np.atleast_1d(u)
    if k == 0:
        return 0.5 * (1 - u)**2
    elif k == 1:
        return -(u**2) + u + 0.5
    elif k == 2:
        return 0.5 * (u**2)
    else:
        raise ValueError

# uniform_quadratic_bspline_first_derivative_basis
def uniform_quadratic_bspline_first_derivative_basis(u, k):
    u = np.atleast_1d(u)
    if k == 0:
        return - (1 - u)
    elif k == 1:
        return -2.0 * u + 1
    elif k == 2:
        return u
    else:
        raise ValueError

# uniform_quadratic_bspline_second_derivative_basis
def uniform_quadratic_bspline_second_derivative_basis(u, k):
    u = np.atleast_1d(u)
    if k == 0:
        return 1.0
    elif k == 1:
        return -2.0
    elif k == 2:
        return 1.0
    else:
        raise ValueError

# biquadratic_bspline_basis_i
BIQUADRATIC_BSPLINE_BASIS_M = [1, 1, 0, 0, 0, 1, 2, 2, 2]
BIQUADRATIC_BSPLINE_BASIS_N = [1, 0, 0, 1, 2, 2, 2, 1, 0]
def biquadratic_bspline_basis_i(f, g, U, i):
    U = np.atleast_2d(U)
    return (f(U[:, 0], BIQUADRATIC_BSPLINE_BASIS_M[i]) *
            g(U[:, 1], BIQUADRATIC_BSPLINE_BASIS_N[i]))

# biquadratic_bspline_basis
def biquadratic_bspline_basis(f, g=None, func_name=None):
    if g is None:
        g = f
    def basis_function(U):
        U = np.atleast_2d(U)
        B = np.empty((U.shape[0], 9), dtype=np.float64)
        for i in xrange(9):
            B[:, i] = biquadratic_bspline_basis_i(f, g, U, i)
        return B

    if func_name is not None:
        basis_function.func_name = func_name
    return basis_function

# biquadratic_bspline_position_basis
biquadratic_bspline_position_basis = biquadratic_bspline_basis(
    uniform_quadratic_bspline_position_basis,
    uniform_quadratic_bspline_position_basis,
    'biquadratic_bspline_position_basis')

# biquadratic_bspline_du_basis
biquadratic_bspline_du_basis = biquadratic_bspline_basis(
    uniform_quadratic_bspline_first_derivative_basis,
    uniform_quadratic_bspline_position_basis,
    'biquadratic_bspline_du_basis')

# biquadratic_bspline_du_du_basis
biquadratic_bspline_du_du_basis = biquadratic_bspline_basis(
    uniform_quadratic_bspline_second_derivative_basis,
    uniform_quadratic_bspline_position_basis,
    'biquadratic_bspline_du_du_basis')

# transform_u_to_required_depth
def transform_u_to_required_depth(u):
    u = np.copy(u)
    n = int(np.floor(1.0 - np.log2(np.max(u))))
    u *= 2**(n - 1)
    if u[0] > 0.5:
        if u[1] > 0.5:
            k = 1
            u[0] = 2 * u[0] - 1
            u[1] = 2 * u[1] - 1
        else:
            k = 0
            u[0] = 2 * u[0] - 1
            u[1] = 2 * u[1]
    else:
        assert u[1] > 0.5
        k = 2
        u[0] = 2 * u[0]
        u[1] = 2 * u[1] - 1

    return n, k, u

# recursive_evaluate
def recursive_evaluate(b, N, u, X, p, verbose=False):
    u0 = np.array(u)
    n, k, u = transform_u_to_required_depth(u)
    if N != 4:
        assert n >= 1

    if verbose:
        print 'u = (%.4g, %.4g) -> n = %d, k = %d, u = (%.4g, %.4g)' % (
            u0[0], u0[1], n, k, u[0], u[1])

    A_ = bigger_subdivision_matrix(N)
    P3 = picker_matrix(N, 3)
    A_Anm1 = A_.copy()
    for i in xrange(n - 1):
        A_Anm1 = np.dot(A_, np.dot(P3, A_Anm1))

    m = (2.0 ** p) ** n * np.dot(b(u)[0], np.dot(picker_matrix(N, k),
                                                 A_Anm1))
    if verbose:
        print ' m = %s' % m
    return np.dot(m, X)
