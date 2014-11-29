# visualise_doosabin_subdivision.py

# Imports
import argparse
import numpy as np

import doosabin

# Requires common/python on `PYTHONPATH`.
import vtk_

# EXAMPLES
EXAMPLES = {
   # From cpp/doosabin_test.cpp.
   'doosabin_test' : dict(
        T = [[0, 1, 5, 4],
             [3, 8, 7],
             [6, 7, 12, 11],
             [1, 2, 6, 5],
             [2, 3, 7, 6],
             [4, 5, 10, 9],
             [5, 6, 11, 10],
             [7, 8, 13, 12]],
        X = np.array([[0, 2, 0],
                      [1, 2, 0],
                      [2, 2, 0],
                      [3, 2, 0],
                      [0, 1, 0],
                      [1, 1, 0],
                      [2, 1, 2], # X[6, 2] += 1.0
                      [3, 1, 0],
                      [4, 1, 0],
                      [0, 0, 0],
                      [1, 0, 0],
                      [2, 0, 0],
                      [3, 0, 0],
                      [4, 0, 0]], dtype=np.float64)),
    'cube' : dict(
        T = [[0, 1, 3, 2],
             [4, 6, 7, 5],
             [1, 5, 7, 3],
             [6, 4, 0, 2],
             [0, 4, 5, 1],
             [3, 7, 6, 2]],
        X = 3.0 * np.array([[0, 0, 0],
                            [1, 0, 0],
                            [0, 1, 0],
                            [1, 1, 0],
                            [0, 0, 1],
                            [1, 0, 1],
                            [0, 1, 1],
                            [1, 1, 1]], dtype=np.float64)),
}
EXAMPLES_KEYS = sorted(EXAMPLES.keys())

# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('example', choices=EXAMPLES_KEYS)
    parser.add_argument('sample_density', nargs='?', type=int, default=16)
    parser.add_argument('-d', '--disable', action='append', default=[],
                        choices={'T', 'M', 'N'})
    parser.add_argument('--num-subdivisions', type=int, default=0)
    args = parser.parse_args()

    z = EXAMPLES[args.example]
    T, X = z['T'], z['X']

    num_subdivisions = max(args.num_subdivisions,
                           doosabin.is_initial_subdivision_required(T))
    for i in xrange(num_subdivisions):
        T, X = doosabin.subdivide(T, X)

    surface = doosabin.Surface(T)

    pd, Ud, Td = surface.uniform_parameterisation(args.sample_density)
    M = surface.M(pd, Ud, X)
    Mu = surface.Mu(pd, Ud, X)
    Mv = surface.Mv(pd, Ud, X)
    N = np.cross(Mu, Mv)
    N /= np.linalg.norm(N, axis=1)[:, np.newaxis]

    a = {}
    if 'T' not in args.disable:
        a['T_points'] = vtk_.points(X, ('SetRadius', 0.05),
                                       ('SetThetaResolution', 16),
                                       ('SetPhiResolution', 16))
        a['T_mesh'] = vtk_.mesh(T, X, ('SetRadius', 0.025),
                                      ('SetNumberOfSides', 16))

    if 'M' not in args.disable:
        a['M'] = vtk_.surface(Td, M)
        a['M'].GetProperty().SetColor(0.216, 0.494, 0.722)
        a['M'].GetProperty().SetSpecular(1.0)

    if 'N' not in args.disable:
        i = np.arange(pd.size)
        a['N'] = vtk_.tubes(np.c_[i, i + pd.size], np.r_['0,2', M, M + 0.1 * N],
                            ('SetRadius', 0.005),
                            ('SetNumberOfSides', 16))

    ren, iren = vtk_.renderer(*a.values())
    iren.Initialize()
    iren.Start()

if __name__ == '__main__':
    main()
