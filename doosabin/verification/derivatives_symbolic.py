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

# du_du_k_0
def du_du_k_0(N):
    # `b` and `p` corresponding to `biquadratic_bspline_du_du_basis`.
    h = sp.S.Half
    b = sp.Matrix([-1, -1, h, h, 0, 0, 0, h, h])
    p = 2

    k = 0

    A_ = sp.Matrix(doosabin.bigger_subdivision_matrix(N))
    P3 = sp.Matrix(doosabin.picker_matrix(N, 3))
    A_Anm1 = A_
    for i in count(1):
        yield 2**(p * i) * b.T * (
            sp.Matrix(doosabin.picker_matrix(N, k)) * A_Anm1)
        A_Anm1 = A_ * P3 * A_Anm1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('N', nargs='?', type=int, default=6)
    parser.add_argument('n', nargs='?', type=int, default=8)
    args = parser.parse_args()

    # Generate example extraordinary patch with an extraordinary face of `N`
    # sides.
    print 'N:', args.N
    X = example_extraordinary_patch(args.N)
    np.random.seed(1337)
    X += 0.1 * np.random.randn(X.size).reshape(X.shape)

    # Evaluate second derivative at various levels, computing the basis vector
    # symbolically (i.e. exactly).
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
