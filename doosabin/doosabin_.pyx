# doosabin_.pyx
# cython: boundscheck=False

# Imports
import numpy as np
cimport numpy as np
np.import_array()

# Requires common/python on `PYTHONPATH`.
from argcheck import check_ndarray_or_raise, check_type_or_raise
from itertools_ import pairwise
from face_array import (raw_face_array_to_sequence,
                        sequence_to_raw_face_array)

# Type
ctypedef np.float64_t DTYPE_t
DTYPE = np.float64

cdef extern from '<vector>' namespace 'std':
    cdef cppclass vector[T]:
        T operator[](int)
        int size()

cdef extern from 'doosabin_pyx.h':
    cdef cppclass Surface_cpp 'Surface':
        Surface_cpp(np.ndarray)

        np.ndarray M(np.ndarray, np.ndarray, np.ndarray)
        np.ndarray Mu(np.ndarray, np.ndarray, np.ndarray)
        np.ndarray Mv(np.ndarray, np.ndarray, np.ndarray)
        np.ndarray Muu(np.ndarray, np.ndarray, np.ndarray)
        np.ndarray Muv(np.ndarray, np.ndarray, np.ndarray)
        np.ndarray Mvv(np.ndarray, np.ndarray, np.ndarray)

        int number_of_vertices()
        int number_of_faces()
        int number_of_patches()
        vector[int]& patch_vertex_indices(int)

        object UniformParameterisation(int)

cdef class Surface:
    cdef Surface_cpp* _surface

    def __cinit__(self, T):
        # Ensure all faces have at least 3 unique integer entries.
        unique_i = set()
        for i, t in enumerate(T):
            if len(t) != len(set(t)):
                raise ValueError('T[%d] contains duplicate entries' % i)
            if len(t) < 3:
                raise ValueError('len(T[%d]) < 3 (= %d)' % (i, len(t)))
            for j in t:
                if not (isinstance(j, int) or issubclass(type(j), np.integer)):
                    raise ValueError('T contains non-integer entry "%s"' % j)
                unique_i.add(j)

        # Ensure vertex indexing is 0-based and contiguous.
        unique_i = sorted(unique_i)
        if unique_i[0] != 0:
            raise ValueError(
                'labels in T are not zero-based: min(T) == %d (!= 0)' %
                unique_i[0])
        if unique_i[-1] != len(unique_i) - 1:
            raise ValueError(
                'labels in T are not contiguous: max(T) == %d (!= %d)' %
                (unique_i[-1], len(unique_i) - 1))

        # Ensure all faces are labelled consistently.
        full_edge_to_half_edges = {}
        for t in T:
            for half_edge in pairwise(t, repeat=True):
                i, j = half_edge
                full_edge = (i, j) if i < j else (j, i)
                try:
                    half_edges = full_edge_to_half_edges[full_edge]
                except KeyError:
                    half_edges = set()
                    full_edge_to_half_edges[full_edge] = half_edges
                else:
                    if half_edge in half_edges:
                        raise ValueError('half edge (%d, %d) encountered '
                                         '(at least) twice in T' % (i, j))
                half_edges.add(half_edge)

        self._surface = new Surface_cpp(sequence_to_raw_face_array(T))

    def __dealloc__(self):
        del self._surface

    @staticmethod
    def __evaluate(f):
        def wrapped_f(self, p, U, X):
            p = np.require(np.atleast_1d(p), np.int32)
            check_ndarray_or_raise('p', p, np.int32, 1, None)
            if np.any((p < 0) | (p >= self.number_of_patches)):
                raise ValueError('p < 0 or p >= %d' % self.number_of_patches)

            U = np.require(np.atleast_2d(U), DTYPE)
            check_ndarray_or_raise('U', U, DTYPE, 2, (p.shape[0], 2))
            if np.any((U < 0.0) | (U > 1.0)):
                raise ValueError('U < 0.0 or U > 1.0')

            X = np.require(np.atleast_2d(X), DTYPE)
            check_ndarray_or_raise('X', X, DTYPE, 2,
                                   (self.number_of_vertices, 3))

            return f(self, p, U, X)
        return wrapped_f

    @__evaluate
    def M(self, np.ndarray p, np.ndarray U, np.ndarray X):
        return self._surface.M(p, U, X)

    @__evaluate
    def Mu(self, np.ndarray p, np.ndarray U, np.ndarray X):
        return self._surface.Mu(p, U, X)

    @__evaluate
    def Mv(self, np.ndarray p, np.ndarray U, np.ndarray X):
        return self._surface.Mv(p, U, X)

    @__evaluate
    def Muu(self, np.ndarray p, np.ndarray U, np.ndarray X):
        return self._surface.Muu(p, U, X)

    @__evaluate
    def Muv(self, np.ndarray p, np.ndarray U, np.ndarray X):
        return self._surface.Muv(p, U, X)

    @__evaluate
    def Mvv(self, np.ndarray p, np.ndarray U, np.ndarray X):
        return self._surface.Mvv(p, U, X)

    # TODO `Mx`.

    def uniform_parameterisation(self, np.int32_t N):
        p, U, T = self._surface.UniformParameterisation(N)
        # Return `T` as a list of list objects.
        return p, U, raw_face_array_to_sequence(map(int, T))

    @property
    def number_of_vertices(self):
        return self._surface.number_of_vertices()

    @property
    def number_of_faces(self):
        return self._surface.number_of_faces()

    @property
    def number_of_patches(self):
        return self._surface.number_of_patches()
