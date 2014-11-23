# derivatives_numeric.py

# Imports
import argparse
import numpy as np

import matplotlib.pyplot as plt

import loop
from common import example_extraordinary_patch

# Requires common/python on `PYTHONPATH`.
from matplotlib_ import label_xaxis

# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('N', nargs='?', type=int, default=5)
    parser.add_argument('n', nargs='?', type=int, default=16)
    parser.add_argument('--seed', type=int, default=1337)
    args = parser.parse_args()

    # Generate example extraordinary patch with an extraordinary face of `N`
    # sides.
    print 'N:', args.N
    X = example_extraordinary_patch(args.N)
    np.random.seed(args.seed)
    X += 0.1 * np.random.randn(X.size).reshape(X.shape)

    # Evaluate with basis functions for small `u` with `v = 0`.
    powers_and_basis_functions = [
        (1, loop.triangle_bspline_du_basis),
        (2, loop.triangle_bspline_du_du_basis)]
    u = 2.0 ** (-np.arange(1, args.n + 1))
    U = np.c_[u, [0.0] * u.size]

    f, axs = plt.subplots(2, 1)

    for ax, (p, b) in zip(axs, powers_and_basis_functions):
        print '%s:' % b.func_name

        norms = []
        for u in U:
            q = loop.recursive_evaluate(p, b, args.N, u, X)
            n = np.linalg.norm(q)
            norms.append(n)

            print '  (2^%d, %g) -> (%+.3e, %+.3e) <%.3e>' % (
                np.around(np.log2(u[0])), u[1], q[0], q[1], n)

        ax.plot(norms, 'o-')
        for i, n in enumerate(norms):
            ax.text(i, n, '$%.3e$' % n, horizontalalignment='center')
        label_xaxis(ax, ['$2^{%d}$' % (-(i + 1)) for i in range(args.n)])
        ax.set_yticks([])

    axs[0].set_title(r'$N = %d\;|\partial u|$' % args.N)
    axs[1].set_title(r'$N = %d\;|\partial u^2|$' % args.N)

    plt.show()

if __name__ == '__main__':
    main()
