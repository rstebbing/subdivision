##########################################
# File: common.py                        #
# Copyright Richard Stebbing 2014.       #
# Distributed under the MIT License.     #
# (See accompany file LICENSE or copy at #
#  http://opensource.org/licenses/MIT)   #
##########################################

# Imports
import numpy as np

# example_extraordinary_patch
def example_extraordinary_patch(N):
    if N < 3:
        raise ValueError('N < 3 (= %d)' % N)
    t = np.linspace(0, -2 * np.pi, N, endpoint=False)
    t -= np.pi/2 + (np.pi / N)
    h = 0.5 * np.sqrt(3)
    X = np.r_['0,2',
              [0.0, 0.0],
              np.c_[np.cos(t), np.sin(t)],
              [0.0, -2 * h],
              [-1.0, -2 * h],
              [-1.5, -h],
              [1.0, -2 * h],
              [1.5, -h]]
    return X
