# evaluation.py

# Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from itertools import cycle

import doosabin
from common import example_extraordinary_patch

# visualise_example_subdivision
def visualise_example_subdivision(N, n):
    X0 = example_extraordinary_patch(N)
    A = doosabin.extended_subdivision_matrix(N)
    A_ = doosabin.bigger_subdivision_matrix(N)
    Xs = [X0]
    Xs_ = []
    for i in xrange(n):
        Xi = Xs[-1]
        Xs.append(np.dot(A, Xi))
        Xs_.append(np.dot(A_, Xi))

    colours = cm.Set1(np.linspace(0.0, 1.0, 9, endpoint=True))[:, :3]
    colour_iterable = cycle(colours)
    f, ax = plt.subplots()
    ax.set_aspect('equal')
    for i, Xi in enumerate(Xs):
        x, y = Xi.T
        c = next(colour_iterable)
        ax.plot(x, y, 'o-', c=c)
        if i >= 1:
            x1, y1 = Xs_[i - 1][-7:].T
            ax.plot(x1, y1, 'o-', c=c)
            ax.plot([x[-1], x1[0]], [y[-1], y1[0]], '--', c=c)

    min_, max_ = np.amin(X0, axis=0), np.amax(X0, axis=0)
    ax.set_xlim(min_[0] - 0.2, max_[0] + 0.2)
    ax.set_ylim(min_[1] - 0.2, max_[1] + 0.2)
    ax.set_xticks([])
    ax.set_yticks([])
    return f, ax

if __name__ == '__main__':
    visualise_example_subdivision(7, 2)
    plt.show()

    # Check evaluation for an extraordinary patch.
    N = 7
    X = example_extraordinary_patch(N)
    # x = 0.5 * (np.linspace(0.0, 5.0, X.size) % 1.0)
    # X += x.reshape(X.shape)
    t, step = np.linspace(0.0, 1.0, 11, endpoint=False, retstep=True)
    t += 0.5 * step
    U = np.dstack(np.broadcast_arrays(t[:, np.newaxis], t)).reshape(-1, 2)

    P = []
    for u in U:
        P.append(doosabin.recursive_evaluate(
            doosabin.biquadratic_bspline_position_basis,
            N, u, X, 0))

    f, ax = plt.subplots()
    ax.set_aspect('equal')
    x, y = X.T
    ax.plot(x, y, 'ro-')
    x, y = np.transpose(P)
    ax.plot(x, y, 'b.')
    plt.show()
