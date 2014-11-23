# common.py

# Imports
import numpy as np

# example_extraordinary_patch
def example_extraordinary_patch(N):
    t = np.linspace(0, -2 * np.pi, N, endpoint=False)
    X = np.c_[np.cos(t), np.sin(t)]

    s = np.linalg.norm(X[1] - X[0])
    x0 = np.mean(X, axis=0)
    X = (X - x0) / s + x0

    X -= X[0]

    t = np.deg2rad(-45)
    a, b = np.cos(t), np.sin(t)
    R = np.r_[a, -b, b, a].reshape(2, 2)
    X = np.dot(X, R.T)

    return np.r_['0,2', X,
                 (1, X[-1, 1]),
                 (1, 0),
                 (1, -1),
                 (0, -1),
                 (X[1, 0], -1)]

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    f, ax = plt.subplots()
    ax.set_aspect('equal')
    X = example_extraordinary_patch(7)
    x, y = X.T
    ax.plot(x, y, 'o-')
    for i, (x, y) in enumerate(X):
        ax.text(x, y, '%d' % (i + 1))
    plt.show()
