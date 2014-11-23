# derivatives_symbolic.py

# Imports
import argparse
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from itertools import count

import doosabin
from common import example_extraordinary_patch

# Use a symbolic backend for `doosabin`.
doosabin.g = sp

# `u`.
u, v = sp.symbols('u v')

# recursive_evaluate_basis
def recursive_evaluate_basis(p, b, N, k):
    A_ = sp.Matrix(doosabin.bigger_subdivision_matrix(N))
    P3 = sp.Matrix(doosabin.picker_matrix(N, 3))
    A_Anm1 = A_
    for i in count(1):
        yield 2**(p * i) * b.T * (
            sp.Matrix(doosabin.picker_matrix(N, k)) * A_Anm1)
        A_Anm1 = A_ * P3 * A_Anm1

# du_du_k_0
def du_du_k_0(N):
    # `b` and `p` corresponding to `biquadratic_bspline_du_du_basis`.
    p = 2
    h = sp.S.Half
    b = sp.Matrix([-1, -1, h, h, 0, 0, 0, h, h])
    return recursive_evaluate_basis(p, b, N, 0)

# du_k_0
def du_k_0(N):
    # `b` and `p` corresponding to `biquadratic_bspline_du_basis` for
    # v = 0.
    p = 1
    b = sp.Matrix([
        doosabin.biquadratic_bspline_basis_i(
            doosabin.uniform_quadratic_bspline_first_derivative_basis,
            doosabin.uniform_quadratic_bspline_position_basis,
            u, v, i)
         for i in range(doosabin.NUM_BIQUADRATIC_BSPLINE_BASIS)
        ]).subs({v : 0})
    return recursive_evaluate_basis(p, b, N, 0)

# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('N', nargs='?', type=int, default=6)
    parser.add_argument('n', nargs='?', type=int, default=16)
    args = parser.parse_args()

    # Generate example extraordinary patch with an extraordinary face of `N`
    # sides.
    print 'N:', args.N
    X = example_extraordinary_patch(args.N)
    np.random.seed(1337)
    X += 0.1 * np.random.randn(X.size).reshape(X.shape)

    # Evaluate the first derivative at various levels, computing the basis
    # vector symbolically and substituting `u` with 0.
    print 'biquadratic_bspline_du_basis:'
    g = du_k_0(args.N)
    for i in range(args.n):
        # `b` is exact.
        b = next(g).subs({u : 0})
        q = np.dot(map(np.float64, b), X)
        print '  (2^%d, 0) -> (%+.3e, %+.3e) <%.3e>' % (
            -(i + 1), q[0], q[1], np.linalg.norm(q))

    # Evaluate second derivative at various levels, computing the basis vector
    # symbolically (i.e. exactly).
    # NOTE For N = 6 and similar, the second derivatives are constant per
    # subdivision level and discontinuous between. The result of this is that
    # the second derivatives reported here are "off by one" in comparison to
    # derivatives_numeric.py. This is because `u = 2^-n` is handled after
    # `n + 1` subdivisions in derivatives_numeric.py, but only `n` subdivisions
    # here.
    print 'biquadratic_bspline_du_du_basis: (visualisation)'
    g = du_du_k_0(args.N)
    du_du = np.empty((args.n, 2), dtype=np.float64)
    for i in range(args.n):
        # `b` is exact.
        b = next(g)
        du_du[i] = np.dot(map(np.float64, b), X)

    norm_du_du = np.array(map(np.linalg.norm, du_du))

    f, ax = plt.subplots()
    ax.plot(1 + np.arange(args.n), norm_du_du, 'o-')
    for i, n in enumerate(norm_du_du):
        ax.text(i + 1, n, '%.3e' % n, horizontalalignment='center')
    ax.set_title('|du_du|, N = %d' % args.N)
    ax.set_xlabel('n')
    plt.show()

if __name__ == '__main__':
    main()
