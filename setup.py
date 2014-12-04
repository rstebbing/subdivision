##########################################
# File: setup.py                         #
# Copyright Richard Stebbing 2014.       #
# Distributed under the MIT License.     #
# (See accompany file LICENSE or copy at #
#  http://opensource.org/licenses/MIT)   #
##########################################

# Imports
import os
import sys

from distutils.core import setup
from distutils.extension import Extension

from Cython.Distutils import build_ext

# Numpy from http://www.scipy.org/Cookbook/SWIG_NumPy_examples
import numpy as np

try:
    NUMPY_INCLUDE = np.get_include()
except AttributeError:
    NUMPY_INCLUDE = np.get_numpy_include()

# `EIGEN_INCLUDE` and `COMMON_CPP_INCLUDE` from site.cfg.
import ConfigParser
c = ConfigParser.ConfigParser()
# Preserve case. See:
# http://stackoverflow.com/questions/1611799/preserve-case-in-configparser
c.optionxform = str
c.read('site.cfg')
EIGEN_INCLUDE = c.get('Include', 'EIGEN_INCLUDE')
COMMON_CPP_INCLUDE = c.get('Include', 'COMMON_CPP_INCLUDE')

# Setup.
include_dirs = [NUMPY_INCLUDE, EIGEN_INCLUDE, COMMON_CPP_INCLUDE,
                'include/',
                '.']

modules = [
    'subdivision.doosabin',
    'subdivision.loop',
]

setup(name='subdivision',
      version='0.1',
      author='Richard Stebbing',
      author_email='richie.stebbing@gmail.com',
      license='MIT',
      url='https://github.com/rstebbing/subdivision',
      packages=['subdivision'] + modules,
      ext_modules=[
        Extension('subdivision.doosabin.doosabin_',
                  ['subdivision/doosabin/doosabin_.pyx'],
                  language='c++',
                  include_dirs=include_dirs +
                      ['cpp/doosabin/include/']),
      ],
      cmdclass={'build_ext' : build_ext})
