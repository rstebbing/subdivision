// doosabin_pyx.h
#ifndef DOOSABIN_PYX_H
#define DOOSABIN_PYX_H

// Includes
#include "doosabin.h"
#include "Mesh/mesh_python.h"

// Surface
class Surface {
 public:
  typedef doosabin::Surface<double> DooSabinSurface;

  explicit Surface(PyArrayObject* npy_raw_face_array)
    : surface_(mesh::GeneralMesh(
          mesh_python::PyArrayObject_to_CellArray(npy_raw_face_array)))
  {}

  #define EVALUATE(M) \
  void M(int p, const double* u, const double* X, double* r) const { \
    const Eigen::Map<const Eigen::Vector2d> _u(u); \
    const Eigen::Map<const Eigen::MatrixXd> _X( \
      X, 3, surface_.patch_vertex_indices(p).size()); \
    Eigen::Map<Eigen::Vector3d> _r(r); \
    surface_.M(p, _u, _X, &_r); \
  }
  EVALUATE(M);
  EVALUATE(Mu);
  EVALUATE(Mv);
  EVALUATE(Muu);
  EVALUATE(Muv);
  EVALUATE(Mvv);
  #undef EVALUATE

  inline int number_of_vertices() const {
    return surface_.number_of_vertices();
  }

  inline int number_of_faces() const {
    return surface_.number_of_faces();
  }

  inline int number_of_patches() const {
    return surface_.number_of_patches();
  }

  inline const std::vector<int>& patch_vertex_indices(const int p) const {
    return surface_.patch_vertex_indices(p);
  }

  PyObject* UniformParameterisation(int N) {
    Eigen::VectorXi p;
    Eigen::MatrixXd U;
    std::vector<int> T;
    surface_.UniformParameterisation(N, &p, &U, &T);

    npy_intp dims[2];

    dims[0] = p.size();
    PyArrayObject* npy_p = (PyArrayObject*)PyArray_EMPTY(
      1, dims, NPY_INT32, 0);
    std::copy(p.data(), p.data() + p.size(),
              (int*)PyArray_DATA(npy_p));

    dims[0] = U.cols();
    dims[1] = U.rows();
    PyArrayObject* npy_U = (PyArrayObject*)PyArray_EMPTY(
      2, dims, NPY_FLOAT64, 0);
    std::copy(U.data(), U.data() + U.rows() * U.cols(),
              (double*)PyArray_DATA(npy_U));

    dims[0] = T.size();
    PyArrayObject* npy_T = (PyArrayObject*)PyArray_EMPTY(
      1, dims, NPY_INT32, 0);
    std::copy(T.data(), T.data() + T.size(),
              (int*)PyArray_DATA(npy_T));

    PyObject* r = PyTuple_New(3);
    PyTuple_SET_ITEM(r, 0, (PyObject*)npy_p);
    PyTuple_SET_ITEM(r, 1, (PyObject*)npy_U);
    PyTuple_SET_ITEM(r, 2, (PyObject*)npy_T);
    return r;
  }

 private:
  DooSabinSurface surface_;
};

#endif // DOOSABIN_PYX_H
