# loop.py

# Imports
import numpy as np
import re
import sympy as sp
from functools import partial
from operator import add, mul

# Requires common/python on `PYTHONPATH`.
from sympy_ import sympy_polynomial_to_function

# Loop Subdivision Matrices

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

# subdivision_matrix
def subdivision_matrix(N):
    alpha = g.Rational(5, 8) - (3 + 2 * g.cos(2 * g.pi / N))**2 / 64
    a = 1 - alpha
    b = alpha / N
    c = g.Rational(3, 8)
    d = g.Rational(1, 8)

    S = []
    S.append([a] + [b] * N)
    for i in range(N):
        S.append([c] + [0] * N)
        S[i + 1][i + 1] = c
        S[i + 1][(i + 1) % N + 1] = d
        S[i + 1][(i - 1) % N + 1] = d
    return S

# extended_subdivision_matrix
def extended_subdivision_matrix(N):
    S = subdivision_matrix(N)

    A = [s + [0] * 5 for s in S]
    a, b, c, d = [g.Rational(i, 16) for i in [1, 2, 6, 10]]
    A.append([b, c] + [0] * (N - 2) + [c, b] + [0] * 4)
    A.append([a, d, a] + [0] * (N - 3) + [a, a, a, a] + [0] * 2)
    A.append([b, c, c] + [0] * N + [b] + [0] * 2)
    A.append([a, a] + [0] * (N - 3) + [a, d, a] + [0] * 2 + [a, a])
    A.append([b] + [0] * (N - 2) + [c, c] + [0] * 4 + [b])
    return A

# bigger_subdivision_matrix
def bigger_subdivision_matrix(N):
    A_ = extended_subdivision_matrix(N)

    a, b = [g.Rational(i, 8) for i in [1, 3]]
    A_.append([0, b] + [0] * (N - 2) + [a, b, a, 0, 0, 0])
    A_.append([0, b] + [0] * (N - 1) + [a, b, a, 0, 0])
    A_.append([0, b, a] + [0] * (N - 1) + [a, b, 0, 0])
    A_.append([0, a] + [0] * (N - 2) + [b, b, 0, 0, a, 0])
    A_.append([0] * N + [b, a, 0, 0, b, a])
    A_.append([0] * (N - 1) + [a, b, 0, 0, 0, a, b])
    return A_

# picker_matrix
PICKING_INDICES = [
    lambda N: [1, N + 2, N + 3, 2, 0, N, N + 1, N + 6, N + 7, N + 8, N + 9,
               N + 4],
    lambda N: [N + 1, N, N + 4, N + 9, N + 6, N + 2, 1, 0, N-1, N + 5, 2,
               N + 3],
    lambda N: [N, N + 1, 1, 0, N-1, N + 5, N + 4, N + 9, N + 6, N + 2, N + 10,
               N + 11],
    lambda N: range(N + 6),
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

# From Stam, 'Evaluation of Loop Subdivision Surfaces'
# http://www.dgp.toronto.edu/~stam/reality/Research/pub.html

# SOURCE_POLYNOMIALS
SOURCE_POLYNOMIALS = '''
u4 + 2u3v,
u4 + 2u3w,
u4 + 2u3w + 6u3v + 6u2vw + 12u2v2 + 6uv2w + 6uv3 + 2v3w + v4,
6u4 + 24u3w + 24u2w2 + 8uw3 + w4 + 24u3v + 60u2vw + 36uvw2 + 6vw3 + 24u2v2 + 36uv2w + 12v2w2 + 8uv3 + 6v3w + v4,
u4 + 6u3w + 12u2w2 + 6uw3 + w4 + 2u3v + 6u2vw + 6uvw2 + 2vw3,
2uv3 + v4,
u4 + 6u3w + 12u2w2 + 6uw3 + w4 + 8u3v + 36u2vw + 36uvw2 + 8vw3 + 24u2v2 + 60uv2w + 24v2w2 + 24uv3 + 24v3w + 6v4,
u4 + 8u3w + 24u2w2 + 24uw3 + 6w4 + 6u3v + 36u2vw + 60uvw2 + 24vw3 + 12u2v2 + 36uv2w + 24v2w2 + 6uv3 + 8v3w + v4,
2uw3 + w4,
2v3w + v4,
2uw3 + w4 + 6uvw2 + 6vw3 + 6uv2w + 12v2w2 + 2uv3 + 6v3w + v4,
w4 + 2vw3'''

# SOURCE_DIVISOR
SOURCE_DIVISOR = 12

# REQUIRED_PERMUTATION
# NOTE `SOURCE_POLYNOMIALS` is intended for labeling from "Figure 1",
# but the labeling in "Figure 2" onwards is what is actually used for
# subdivision
REQUIRED_PERMUTATION = [3, 6, 2, 0, 1, 4, 7, 10, 9, 5, 11, 8]

# parse_source_polynomials_to_sympy
PARSED_SOURCE_POLYNOMIALS = None
def parse_source_polynomials_to_sympy():
    global PARSED_SOURCE_POLYNOMIALS
    if PARSED_SOURCE_POLYNOMIALS is not None:
        return PARSED_SOURCE_POLYNOMIALS

    b_strs = SOURCE_POLYNOMIALS.replace('\n', ' ').split(',')
    assert len(b_strs) == 12

    b_strs = [b_strs[i] for i in REQUIRED_PERMUTATION]

    # NOTE No substitution for `u`, `v` and `w`.
    req_symbs = 'uvw'
    symb_list = map(sp.Symbol, req_symbs)
    symbs = dict(zip(req_symbs, symb_list))
    b = map(partial(parse_source_polynomial_to_sympy, symbs),
            b_strs)

    return symb_list, sp.Matrix(b)

# parse_source_polynomial_to_sympy
def parse_source_polynomial_to_sympy(symbs, b_str):
    term_strs = map(str.strip, b_str.split('+'))
    terms = map(partial(parse_term, symbs), term_strs)
    return reduce(add, terms) / SOURCE_DIVISOR

# parse_term
PARSE_TERM_RE = re.compile(r'([0-9]*)([u]?)([0-9]*)'
                           r'([v]?)([0-9]*)'
                           r'([w]?)([0-9]*)')
def parse_term(symbs, term_str):
    g = PARSE_TERM_RE.search(term_str).groups()
    l = [int(g[0]) if g[0] else 1]

    for i in 1, 3, 5:
        s = symbs.get(g[i], 1)
        p = int(g[i + 1]) if g[i + 1] else 1
        l.append(s ** p)

    return reduce(mul, l)

# `triangle_bspline_basis_uvw`
((u, v, w),
 triangle_bspline_basis_uvw) = parse_source_polynomials_to_sympy()

# Use `u` for the first coordinate and `v` for the second.
# (Stam uses `v` and `w` respectively.)
triangle_bspline_basis_uvw = triangle_bspline_basis_uvw.subs(
    {v : u, w : v, u : w}, simultaneous=True)

# Eliminate `w = 1 - u - v`.
# `triangle_bspline_basis_uv`.
triangle_bspline_basis_uv = triangle_bspline_basis_uvw.subs(
    {w : 1 - u - v}).expand()

# Evaluation Functions

# transform_u_to_subdivided_patch
def transform_u_to_subdivided_patch(u):
    u = np.copy(u)
    n = int(np.floor(1.0 - np.log2(np.sum(u))))

    u *= 2**(n - 1)
    if u[0] > 0.5:
        k = 0
        u[0] = 2 * u[0] - 1
        u[1] = 2 * u[1]
    elif u[1] > 0.5:
        k = 2
        u[0] = 2 * u[0]
        u[1] = 2 * u[1] - 1
    else:
        k = 1
        u[0] = 1 - 2 * u[0]
        u[1] = 1 - 2 * u[1]

    return n, k, u

# recursive_evaluate
def recursive_evaluate(p, b, N, u, X):
    n, k, u = transform_u_to_subdivided_patch(u)
    if N != 6:
        assert n >= 1, 'n < 1 (= %d)' % n

    A_ = bigger_subdivision_matrix(N)
    P3 = picker_matrix(N, 3)
    m = 2.0 ** (p * n) * (-1 if k == 1 else 1) ** p
    x = m * np.dot(b(u).ravel(), picker_matrix(N, k))
    for i in range(n - 1):
        x = np.dot(x, np.dot(A_, P3))
    return np.dot(x, np.dot(A_, X))

# exprs_to_basis
def exprs_to_basis(exprs, func_name=None):
    bs = [sympy_polynomial_to_function(e, (u, v)) for e in exprs]
    def basis_function(U):
        u, v = np.atleast_2d(U).T
        B = np.empty((len(u), len(bs)), dtype=np.float64)
        for i, b in enumerate(bs):
            B[:, i] = b(u, v)
        return B

    if func_name is not None:
        basis_function.func_name = func_name
    return basis_function

# triangle_bspline_position_basis
triangle_bspline_position_basis = exprs_to_basis(
    triangle_bspline_basis_uv,
    'triangle_bspline_position_basis')

# triangle_bspline_du_basis
def Du(b): return [sp.diff(f, u) for f in b]
triangle_bspline_du_basis = exprs_to_basis(
    Du(triangle_bspline_basis_uv),
    'triangle_bspline_du_basis')

# triangle_bspline_dv_basis
def Dv(b): return [sp.diff(f, v) for f in b]
triangle_bspline_dv_basis = exprs_to_basis(
    Dv(triangle_bspline_basis_uv),
    'triangle_bspline_dv_basis')

# triangle_bspline_du_du_basis
triangle_bspline_du_du_basis = exprs_to_basis(
    Du(Du(triangle_bspline_basis_uv)),
    'triangle_bspline_du_du_basis')

# triangle_bspline_du_dv_basis
triangle_bspline_du_dv_basis = exprs_to_basis(
    Dv(Du(triangle_bspline_basis_uv)),
    'triangle_bspline_du_dv_basis')

# triangle_bspline_dv_dv_basis
triangle_bspline_dv_dv_basis = exprs_to_basis(
    Dv(Dv(triangle_bspline_basis_uv)),
    'triangle_bspline_dv_dv_basis')
