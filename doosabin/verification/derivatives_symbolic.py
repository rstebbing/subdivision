# derivatives_symbolic.py

# Imports
import numpy as np
import sympy as sp
from itertools import count

# doosabin_weights
def doosabin_weights(N):
    assert N >= 3
    weights = sp.Matrix([(3 + 2 * sp.cos(2 * sp.pi * i / N)) / (4 * N)
                         for i in range(N)])
    weights[0] = sp.Rational(N + 5, 4 * N)
    return weights

# extended_subdivision_matrix
def extended_subdivision_matrix(N):
    assert N >= 3
    a = list(doosabin_weights(N))
    c, d, e, _ = doosabin_weights(4)
    A = []
    for i in xrange(N):
        r = a[-i:] + a[:-i]
        r.extend([0] * 5)
        A.append(r)
    A.append([d] + [0] * (N - 2) + [c, d, e] + [0] * 3)
    A.append([c] + [0] * (N - 2) + [d, e, d] + [0] * 3)
    A.append([c] + [0] * N + [d, e, d] + [0])
    A.append([c, d] + [0] * (N + 1) + [d, e])
    A.append([d, c] + [0] * (N + 1) + [e, d])
    return sp.Matrix(A)

# bigger_subdivision_matrix
def bigger_subdivision_matrix(N):
    assert N >= 3
    c, d, e, _ = doosabin_weights(4)
    A_ = extended_subdivision_matrix(N).tolist()
    A_.append([e] + [0] * (N - 2) + [d, c, d] + [0] * 3)
    A_.append([d] + [0] * (N - 2) + [e, d, c] + [0] * 3)
    A_.append([d] + [0] * N + [c, d, e] + [0])
    A_.append([e] + [0] * N + [d, c, d] + [0])
    A_.append([d] + [0] * N + [e, d, c] + [0])
    A_.append([d, e] + [0] * (N + 1) + [c, d])
    A_.append([e, d] + [0] * (N + 1) + [d, c])
    return sp.Matrix(A_)

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
    P = sp.zeros(n, M)
    for i in xrange(n):
        P[i, j[i]] = 1
    return P

if __name__ == '__main__':
    def N_du_du_k_0(N):
        # `b` and `p` corresponding to `biquadratic_bspline_du_du_basis`.
        b = sp.Matrix([-1, -1, sp.S.Half, sp.S.Half, 0, 0, 0, sp.S.Half, sp.S.Half])
        p = 2
        k = 0

        A_ = bigger_subdivision_matrix(N)
        P3 = picker_matrix(N, 3)
        A_Anm1 = A_
        for i in count(1):
            yield 2**(p * i) * b.T * (picker_matrix(N, k) * A_Anm1)
            A_Anm1 = A_ * (P3 * A_Anm1)

    N = 6
    g = N_du_du_k_0(N)
    b = []
    for i in xrange(16):
        b.append(np.linalg.norm(map(float, next(g))))

    import matplotlib.pyplot as plt
    f, ax = plt.subplots()
    ax.plot(np.arange(1, len(b) + 1), b, 'o-')
    ax.set_title('|du_du|, N = %d' % N)
    ax.set_xlabel('n')
    plt.show()
