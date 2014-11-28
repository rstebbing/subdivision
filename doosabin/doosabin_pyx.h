// doosabin_pyx.h
#ifndef DOOSABIN_PYX_H
#define DOOSABIN_PYX_H

// Includes
#include "doosabin.h"
#include "Math/linalg_python.h"
#include "Mesh/mesh_python.h"

// Surface
class Surface {
 public:
  typedef doosabin::Surface<double> DooSabinSurface;
  typedef linalg::MatrixOfColumnPointers<double> MatrixOfColumnPointers;

  explicit Surface(PyArrayObject* npy_raw_face_array)
    : surface_(mesh::GeneralMesh(
          mesh_python::PyArrayObject_to_CellArray(npy_raw_face_array)))
  {}

#define EVALUATE(M) \
 private: \
  class _##M { \
   public: \
    _##M(const DooSabinSurface* surface) : surface_(surface) {} \
    inline void operator()(int p, const double* u, \
                           const MatrixOfColumnPointers& X, \
                           Eigen::Map<Eigen::Vector3d>* r) const { \
      surface_->M(p, u, X, r); \
    } \
   private: \
    const DooSabinSurface* surface_; \
  }; \
 public: \
  inline PyArrayObject* M(PyArrayObject* npy_p, \
                          PyArrayObject* npy_U, \
                          PyArrayObject* npy_X) const { \
    return EvaluateImpl<_##M>(npy_p, npy_U, npy_X); \
  }
  EVALUATE(M);
  EVALUATE(Mu);
  EVALUATE(Mv);
  EVALUATE(Muu);
  EVALUATE(Muv);
  EVALUATE(Mvv);
#undef EVALUATE

  PyObject* UniformParameterisation(int N) const {
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

 private:
  template <typename Evaluator>
  PyArrayObject* EvaluateImpl(PyArrayObject* npy_p,
                              PyArrayObject* npy_U,
                              PyArrayObject* npy_X) const {
    auto p = linalg_python::PyArrayObject_to_VectorMap<int>(npy_p);
    auto U = linalg_python::PyArrayObject_to_MatrixMap<double>(npy_U);
    auto X = linalg_python::PyArrayObject_to_MatrixMap<double>(npy_X);

    // Sort `p` so that all evaluations over each patch are done together.
    std::vector<int> argsort_p(p->size());
    for (size_t i = 0; i < argsort_p.size(); ++i) {
      argsort_p[i] = static_cast<int>(i);
    }
    auto& _p = *p;
    std::sort(argsort_p.begin(), argsort_p.end(), [&_p](int i, int j) {
      return _p[i] < _p[j];
    });

    // Allocate output array `R`.
    npy_intp dims[2] = {argsort_p.size(), 3};
    PyArrayObject* npy_R = (PyArrayObject*)PyArray_EMPTY(
      2, dims, NPY_FLOAT64, 0);

    // `Xp_data` holds the pointers for the instance of
    // `MatrixOfColumnPointers`.
    std::vector<const double*> Xp_data;
    Xp_data.reserve(doosabin::kMaxNNoAlloc);

    const Evaluator e(&surface_);

    int p0 = -1;
    for (size_t i = 0; i < argsort_p.size(); ++i) {
      int j = argsort_p[i];
      int pj = _p[j];
      if (pj != p0) {
        Xp_data.clear();
        for (int l : surface_.patch_vertex_indices(pj)) {
          Xp_data.push_back(X->data() + 3 * l);
        }
        p0 = pj;
      }

      Eigen::Map<Eigen::Vector3d> _r((double*)PyArray_DATA(npy_R) + 3 * j);
      MatrixOfColumnPointers Xp(Xp_data.data(), 3, Xp_data.size());
      e(pj, U->data() + 2 * j, Xp, &_r);
    }

    return npy_R;
  }

 private:
  DooSabinSurface surface_;
};

#endif // DOOSABIN_PYX_H
