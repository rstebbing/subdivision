##########################################
# File: doosabin.py                      #
# Copyright Richard Stebbing 2014.       #
# Distributed under the MIT License.     #
# (See accompany file LICENSE or copy at #
#  http://opensource.org/licenses/MIT)   #
##########################################

# Imports
import numpy as np
from collections import defaultdict

# Requires `rscommon`.
from rscommon.itertools_ import count, pairwise

# `doosabin_` doesn't need to be available.
try:
    import doosabin_
except ImportError:
    pass

# __all__.
__all__ = ['g',
           'doosabin_weights',
           'extended_subdivision_matrix',
           'bigger_subdivision_matrix',
           'picker_matrix',
           'transform_u_to_subdivided_patch',
           'recursive_evaluate',
           'biquadratic_bspline_position_basis',
           'biquadratic_bspline_du_basis',
           'biquadratic_bspline_dv_basis',
           'biquadratic_bspline_du_du_basis',
           'biquadratic_bspline_du_dv_basis',
           'biquadratic_bspline_dv_dv_basis',
           'raise_if_mesh_is_invalid',
           'subdivide',
           'is_initial_subdivision_required',
           'surface']

# Doo-Sabin Subdivision Matrices

# `g` is a namespace which provides `cos`, `pi` and `Rational`.
# Setting `g` at the module level allows manipulation of the underlying numeric
# type used for the Doo-Sabin weights, matrices, and basis functions.
# e.g. `g` can be replaced with the sympy module.
class g(object):
    cos = np.cos
    pi = np.pi
    @staticmethod
    def Rational(a, b):
        return float(a) / b

# doosabin_weights
def doosabin_weights(N):
    if N < 3:
        raise ValueError('N < 3 (= %d)' % N)
    weights = [(3 + 2 * g.cos(2 * g.pi * i / N)) / (4 * N)
               for i in range(N)]
    weights[0] = g.Rational(N + 5, 4 * N)
    return weights

# extended_subdivision_matrix
def extended_subdivision_matrix(N):
    if N < 3:
        raise ValueError('N < 3 (= %d)' % N)
    a = doosabin_weights(N)
    c, d, e, _ = doosabin_weights(4)
    A = []
    for i in range(N):
        r = a[-i:] + a[:-i]
        r.extend([0] * 5)
        A.append(r)
    A.append([d] + [0] * (N - 2) + [c, d, e] + [0] * 3)
    A.append([c] + [0] * (N - 2) + [d, e, d] + [0] * 3)
    A.append([c] + [0] * N + [d, e, d] + [0])
    A.append([c, d] + [0] * (N + 1) + [d, e])
    A.append([d, c] + [0] * (N + 1) + [e, d])
    return A

# bigger_subdivision_matrix
def bigger_subdivision_matrix(N):
    if N < 3:
        raise ValueError('N < 3 (= %d)' % N)
    c, d, e, _ = doosabin_weights(4)
    A_ = extended_subdivision_matrix(N)
    A_.append([e] + [0] * (N - 2) + [d, c, d] + [0] * 3)
    A_.append([d] + [0] * (N - 2) + [e, d, c] + [0] * 3)
    A_.append([d] + [0] * N + [c, d, e] + [0])
    A_.append([e] + [0] * N + [d, c, d] + [0])
    A_.append([d] + [0] * N + [e, d, c] + [0])
    A_.append([d, e] + [0] * (N + 1) + [c, d])
    A_.append([e, d] + [0] * (N + 1) + [d, c])
    return A_

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
    P = [[0] * M for _ in range(n)]
    for i in range(n):
        P[i][j[i]] = 1
    return P

# Basis Functions

# uniform_quadratic_bspline_position_basis
def uniform_quadratic_bspline_position_basis(u, k):
    if k == 0:
        return g.Rational(1, 2) * (1 - u)**2
    elif k == 1:
        return -(u**2) + u + g.Rational(1, 2)
    elif k == 2:
        return g.Rational(1, 2) * (u**2)
    else:
        raise ValueError('k not in {0, 1, 2} (= %d)' % k)

# uniform_quadratic_bspline_first_derivative_basis
def uniform_quadratic_bspline_first_derivative_basis(u, k):
    if k == 0:
        return - (1 - u)
    elif k == 1:
        return -2 * u + 1
    elif k == 2:
        return u
    else:
        raise ValueError('k not in {0, 1, 2} (= %d)' % k)

# uniform_quadratic_bspline_second_derivative_basis
def uniform_quadratic_bspline_second_derivative_basis(u, k):
    if k == 0:
        return 1
    elif k == 1:
        return -2
    elif k == 2:
        return 1
    else:
        raise ValueError('k not in {0, 1, 2} (= %d)' % k)

# biquadratic_bspline_basis_i
BIQUADRATIC_BSPLINE_BASIS_M = [1, 1, 0, 0, 0, 1, 2, 2, 2]
BIQUADRATIC_BSPLINE_BASIS_N = [1, 0, 0, 1, 2, 2, 2, 1, 0]
def biquadratic_bspline_basis_i(f, g, u, v, i):
    return (f(u, BIQUADRATIC_BSPLINE_BASIS_M[i]) *
            g(v, BIQUADRATIC_BSPLINE_BASIS_N[i]))

# Evaluation Functions

# transform_u_to_subdivided_patch
def transform_u_to_subdivided_patch(u):
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
def recursive_evaluate(p, b, N, u, X=None):
    n, k, u = transform_u_to_subdivided_patch(u)
    if N != 4:
        assert n >= 1, 'n < 1 (= %d)' % n

    A_ = bigger_subdivision_matrix(N)
    P3 = picker_matrix(N, 3)
    x = 2.0**(p * n) * np.dot(b(u).ravel(), picker_matrix(N, k))
    for i in range(n - 1):
        x = np.dot(x, np.dot(A_, P3))
    x = np.dot(x, A_)
    return x if X is None else np.dot(x, X)

# biquadratic_bspline_basis
NUM_BIQUADRATIC_BSPLINE_BASIS = 9
def biquadratic_bspline_basis(f, g=None, func_name=None):
    if g is None:
        g = f
    def basis_function(U):
        u, v = np.atleast_2d(U).T
        B = np.empty((len(u), NUM_BIQUADRATIC_BSPLINE_BASIS), dtype=np.float64)
        for i in range(NUM_BIQUADRATIC_BSPLINE_BASIS):
            B[:, i] = biquadratic_bspline_basis_i(f, g, u, v, i)
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

# biquadratic_bspline_dv_basis
biquadratic_bspline_dv_basis = biquadratic_bspline_basis(
    uniform_quadratic_bspline_position_basis,
    uniform_quadratic_bspline_first_derivative_basis,
    'biquadratic_bspline_dv_basis')

# biquadratic_bspline_du_du_basis
biquadratic_bspline_du_du_basis = biquadratic_bspline_basis(
    uniform_quadratic_bspline_second_derivative_basis,
    uniform_quadratic_bspline_position_basis,
    'biquadratic_bspline_du_du_basis')

# biquadratic_bspline_du_dv_basis
biquadratic_bspline_du_dv_basis = biquadratic_bspline_basis(
    uniform_quadratic_bspline_first_derivative_basis,
    uniform_quadratic_bspline_first_derivative_basis,
    'biquadratic_bspline_du_dv_basis')

# biquadratic_bspline_dv_dv_basis
biquadratic_bspline_dv_dv_basis = biquadratic_bspline_basis(
    uniform_quadratic_bspline_position_basis,
    uniform_quadratic_bspline_second_derivative_basis,
    'biquadratic_bspline_dv_dv_basis')

# General Mesh Operations

# raise_if_mesh_is_invalid
def raise_if_mesh_is_invalid(T):
    # Ensure all faces have at least 3 unique integer entries.
    unique_i = set()
    for i, t in enumerate(T):
        if len(t) != len(set(t)):
            raise ValueError('T[%d] contains duplicate entries' % i)
        if len(t) < 3:
            raise ValueError('len(T[%d]) < 3 (= %d)' % (i, len(t)))
        for j in t:
            if not (isinstance(j, int) or issubclass(type(j), np.integer)):
                raise ValueError('T contains non-integer entry "%s"' % j)
            unique_i.add(j)

    # Ensure vertex indexing is 0-based and contiguous.
    unique_i = sorted(unique_i)
    if unique_i[0] != 0:
        raise ValueError(
            'labels in T are not zero-based: min(T) == %d (!= 0)' %
            unique_i[0])
    if unique_i[-1] != len(unique_i) - 1:
        raise ValueError(
            'labels in T are not contiguous: max(T) == %d (!= %d)' %
            (unique_i[-1], len(unique_i) - 1))

    # Ensure all faces are labelled consistently.
    full_edge_to_half_edges = {}
    for t in T:
        for half_edge in pairwise(t, repeat=True):
            i, j = half_edge
            full_edge = (i, j) if i < j else (j, i)
            try:
                half_edges = full_edge_to_half_edges[full_edge]
            except KeyError:
                half_edges = set()
                full_edge_to_half_edges[full_edge] = half_edges
            else:
                if half_edge in half_edges:
                    raise ValueError('half edge (%d, %d) encountered '
                                     '(at least) twice in T' % (i, j))
            half_edges.add(half_edge)

# subdivide
def subdivide(T, X=None):
    raise_if_mesh_is_invalid(T)

    # Get necessary topology information about `T`.
    vertex_to_half_edges = defaultdict(list)
    vertex_to_faces = defaultdict(list)
    half_edge_to_opposite_edge = {}
    half_edge_to_face = {}

    for face_index, face in enumerate(T):
        for half_edge in pairwise(face, repeat=True):
            i, j = half_edge
            vertex_to_half_edges[i].append(half_edge)
            vertex_to_faces[i].append(face_index)
            half_edge_to_opposite_edge[half_edge] = None
            half_edge_to_face[half_edge] = face_index

    for face in T:
        for half_edge in pairwise(face, repeat=True):
            opposite_half_edge = half_edge[::-1]
            if opposite_half_edge in half_edge_to_opposite_edge:
                half_edge_to_opposite_edge[half_edge] = opposite_half_edge

    # Subdivide all vertices to fill `child_X` if `X` is available.
    if X is None:
        child_X = None
    else:
        N_child_X = sum(map(len, T))
        child_X = np.empty((N_child_X, X.shape[1]), dtype=np.float64)
        child_V_index = count()

        for face_index, face in enumerate(T):
            X_face = X[face]
            w = doosabin_weights(X_face.shape[0])

            for i in face:
                child_i = next(child_V_index)
                child_X[child_i] = np.dot(w, X_face)
                w = np.roll(w, 1)

    # Fill `V_to_child_X`.
    V_to_child_X = {}
    child_V_index = count()

    for face_index, face in enumerate(T):
        for i in face:
            child_i = next(child_V_index)
            V_to_child_X[face_index, i] = child_i

    # Initialise `child_T` to original faces but with new child vertex indices.
    child_T = []
    child_V_index = count()

    for face_index, face in enumerate(T):
        child_face = map(lambda i: next(child_V_index), face)
        child_T.append(child_face)

    # Add child faces which occur at patch vertices.
    for i, half_edges in vertex_to_half_edges.items():
        # Check if vertex `i` is a patch vertex.
        # I.e. all half edges from `i` have opposities which terminate at `i`.
        opposite_edges = map(lambda h: half_edge_to_opposite_edge[h],
                             half_edges)

        if None in opposite_edges:
            continue

        # Get `ordered_face_indices` which respects the same vertex ordering.
        # (The "within-face" vertex ordering.)
        unordered_face_indices = list(vertex_to_faces[i])
        ordered_face_indices = [unordered_face_indices.pop(0)]

        while len(unordered_face_indices) > 1:
            last_face_index = ordered_face_indices[-1]
            for half_edge in pairwise(T[last_face_index], repeat=True):
                if half_edge[1] == i:
                    break

            next_half_edge = half_edge[::-1]

            face_index = half_edge_to_face[next_half_edge]
            index = unordered_face_indices.index(face_index)
            ordered_face_indices.append(unordered_face_indices.pop(index))

        ordered_face_indices.append(unordered_face_indices.pop(0))

        # Construct `child_face` from `ordered_face_indices`.
        child_face = map(lambda f: V_to_child_X[f, i], ordered_face_indices)
        child_T.append(child_face)

    # Add child faces which occur across edges.
    for face_index, face in enumerate(T):
        for half_edge in pairwise(face, repeat=True):
            # Only process the edge once.
            i, j = half_edge
            if j < i:
                continue

            # Can't construct a face across a single half edge.
            opposite_half_edge = half_edge_to_opposite_edge[half_edge]
            if opposite_half_edge is None:
                continue

            # Construct `child_face`.
            opposite_face_index = half_edge_to_face[opposite_half_edge]
            keys = [(opposite_face_index, i),
                    (opposite_face_index, j),
                    (face_index, j),
                    (face_index, i)]
            child_face = map(lambda k: V_to_child_X[k], keys)
            child_T.append(child_face)

    return (child_T, child_X) if X is not None else child_T

# is_initial_subdivision_required
def is_initial_subdivision_required(T):
    raise_if_mesh_is_invalid(T)

    vertex_to_half_edges = defaultdict(list)
    half_edges = set()
    for face_index, face in enumerate(T):
        for half_edge in pairwise(face, repeat=True):
            i, j = half_edge
            vertex_to_half_edges[i].append(half_edge)
            half_edges.add(half_edge)
    half_edges = frozenset(half_edges)

    # Subdivision required if valence of a patch centre is not four.
    for i, from_i in vertex_to_half_edges.items():
        if len(from_i) == 4:
            continue

        for half_edge in from_i:
            if half_edge[::-1] not in half_edges:
                break
        else:
            return True

    return False

# surface
def surface(T):
    if is_initial_subdivision_required(T):
        raise ValueError('T has a patch centre with valency != 4')
    return doosabin_.Surface(T)
