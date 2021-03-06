##########################################
# File: derivatives_symbolic.py          #
# Copyright Richard Stebbing 2014.       #
# Distributed under the MIT License.     #
# (See accompany file LICENSE or copy at #
#  http://opensource.org/licenses/MIT)   #
##########################################

# Imports
import argparse
import numpy as np
import sympy as sp
from itertools import count

import matplotlib.pyplot as plt

# Note: Module is imported directly for access to
# `NUM_BIQUADRATIC_BSPLINE_BASIS` and other non-exported variables which aren't
# (typically) necessary.
from subdivision.doosabin import doosabin
from common import example_extraordinary_patch

# Requires `rscommon`.
from rscommon.matplotlib_ import set_xaxis_ticks

# Use a Sympy as the backend for `doosabin`.
doosabin.g = sp

# Global symbols.
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

# du_k_0
def du_k_0(N):
    # `b` and `p` corresponding to `biquadratic_bspline_du_basis` for
    # `v = 0`.
    p = 1
    b = sp.Matrix([
        doosabin.biquadratic_bspline_basis_i(
            doosabin.uniform_quadratic_bspline_first_derivative_basis,
            doosabin.uniform_quadratic_bspline_position_basis,
            u, v, i)
         for i in range(doosabin.NUM_BIQUADRATIC_BSPLINE_BASIS)
        ]).subs({v : 0})
    return recursive_evaluate_basis(p, b, N, 0)

# du_du_k_0
def du_du_k_0(N):
    # `b` and `p` corresponding to `biquadratic_bspline_du_du_basis`.
    p = 2
    h = sp.S.Half
    b = sp.Matrix([-1, -1, h, h, 0, 0, 0, h, h])
    return recursive_evaluate_basis(p, b, N, 0)

# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('N', nargs='?', type=int, default=6)
    parser.add_argument('n', nargs='?', type=int, default=16)
    parser.add_argument('--seed', type=int, default=-1)
    args = parser.parse_args()

    # Generate example extraordinary patch with an extraordinary face of `N`
    # sides.
    # Use `seed < 0` to signal evaluating the linear weights only
    # (i.e. `X = None`).
    print 'N:', args.N
    print 'seed:', args.seed
    if args.seed >= 0:
        X = example_extraordinary_patch(args.N)
        np.random.seed(args.seed)
        X += 0.1 * np.random.randn(X.size).reshape(X.shape)
        print 'X:', X.shape
    else:
        X = None
        print 'X: None'

    generators_and_subs = [('biquadratic_bspline_du_basis', du_k_0, {u : 1}),
                           ('biquadratic_bspline_du_du_basis', du_du_k_0, {})]

    f, axs = plt.subplots(2, 1)

    for ax, (name, g, subs) in zip(axs, generators_and_subs):
        print '%s:' % name

        # Ignore first subdivision so that the reported results correspond with
        # those generated by derivatives_numeric.py (which implicitly skips the
        # first subdivision due to the calculation for `n` in
        # `transform_u_to_subdivided_patch`).
        g = g(args.N)
        next(g)

        norms = []
        for i in range(args.n):
            b = next(g).subs(subs)
            q = map(np.float64, b)
            if X is not None:
                q = np.dot(q, X)
            n = np.linalg.norm(q)
            norms.append(n)

            print ('  (2^%d, 0) ->' % (-(i + 1))),
            if X is not None:
                print ('(%+.3e, %+.3e)' % (q[0], q[1])),
            print '<%.3e>' % n

        ax.plot(norms, 'o-')
        for i, n in enumerate(norms):
            ax.text(i, n, '$%.3e$' % n, horizontalalignment='center')
        set_xaxis_ticks(ax, ['$2^{%d}$' % (-(i + 1)) for i in range(args.n)])
        ax.set_yticks([])

    axs[0].set_title(r'$|N = %d, \partial u|$' % args.N)
    axs[1].set_title(r'$|N = %d, \partial u^2|$' % args.N)

    plt.show()

if __name__ == '__main__':
    main()
