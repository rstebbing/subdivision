# derivatives_numeric.py

# Imports
import argparse
import numpy as np

import doosabin
from common import example_extraordinary_patch

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

    # Evaluate with basis functions for small `u`.
    powers_and_basis_functions = [
        (1, doosabin.biquadratic_bspline_du_basis),
        (2, doosabin.biquadratic_bspline_du_du_basis)]
    u = 2.0 ** (-np.arange(1, args.n + 1))
    U = np.c_[u, [0.0] * u.size]

    for p, b in powers_and_basis_functions:
        print '%s:' % b.func_name
        for u in U:
            q = doosabin.recursive_evaluate(p, b, args.N, u, X)
            print '  (2^%d, %g) -> (%+.3e, %+.3e) <%.3e>' % (
                np.around(np.log2(u[0])), u[1], q[0], q[1], np.linalg.norm(q))

if __name__ == '__main__':
    main()
