# generate_doosabin_test_data.py

# Imports
from cStringIO import StringIO
from functools import partial
import numpy as np
np.random.seed(1337)

from verification import doosabin
from verification.common import example_extraordinary_patch

s = StringIO()
def savetxt(type_, fmt, name, X):
    X = np.atleast_2d(X)

    s.write('static const %s %s[] = {\n' % (type_, name))
    for i, xi in enumerate(X):
        s.write('  ')
        for j, xij in enumerate(xi):
            s.write(fmt % xij)
            if i >= len(X) - 1 and j >= len(xi) - 1:
                s.write('\n')
            else:
                s.write(',')
                s.write(' ' if j < len(xi) - 1 else '\n')
    s.write('};\n')

def savetxt_MapConstMatrixXd(name, X):
    savetxt('double', '%+.6e', '{0}Data'.format(name), X)
    s.write('static const MapConstMatrixXd {0}({0}Data, {1}, {2});\n'.format(
        name, X.shape[1], X.shape[0]))

t, step = np.linspace(0.0, 1.0, 3, endpoint=False, retstep=True)
t += 0.5 * step
U = np.dstack(np.broadcast_arrays(t[:, np.newaxis], t)).reshape(-1, 2)
savetxt_MapConstMatrixXd('kU', U)
s.write('\n')

for N in [3, 4, 5, 6]:
    T, X = example_extraordinary_patch(N, return_mesh=True)
    X += 0.1 * np.random.randn(X.size).reshape(X.shape)
    x0 = np.mean(X, axis=0)
    X = np.c_[X, -np.linalg.norm(X, axis=1)]

    T_ = [len(T)]
    for t in T:
        T_.append(len(t))
        T_.extend(t)
    T_ = np.array(T_, dtype=np.int32)
    savetxt('int', '%d', 'kT%dData' % N, T_)
    s.write(('static const std::vector<int> kT{0} = '
             'InitialiseVector(kT{0}Data, {1});\n\n').format(N, len(T_)))

    savetxt_MapConstMatrixXd('kX%d' % N, X)
    s.write('\n')

    names_powers_and_basis_functions = [
        ('M', 0, doosabin.biquadratic_bspline_position_basis),
        ('Mu', 1, doosabin.biquadratic_bspline_du_basis),
        ('Muu', 2, doosabin.biquadratic_bspline_du_du_basis)]

    for n, p, b in names_powers_and_basis_functions:
        P = np.array([doosabin.recursive_evaluate(p, b, N, u, X)
                      for u in U])
        savetxt_MapConstMatrixXd('k%s%d' % (n, N), P)
        s.write('\n')

with open('doosabin_test_data.txt', 'w') as fp:
    fp.write(s.getvalue())
