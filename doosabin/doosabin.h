// doosabin.h
#ifndef DOOSABIN_H
#define DOOSABIN_H

// Includes
#include <algorithm>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <type_traits>
#include <vector>

#include "Eigen/Dense"

#include "Mesh/general_mesh.h"
#include "Math/modulo.h"

#include "uniform_quadratic_bspline.h"

// doosabin
namespace doosabin {

// Constants.

// `kMaxSubdivisionDepth` and `kUEps` set the subdivision limit and
// adjustment to coordinates on the penultimate subdivision level.
static const size_t kMaxSubdivisionDepth = 10;
static const double kUEps = 1e-6;

// For `N <= kMaxNNoAlloc`, the intermediate subdivided basis vector in
// `EvaluateInternal` is stored on the stack. For `N > kMaxNNoAlloc`,
// a call to `malloc` is necessary.
static const int kMaxNNoAlloc = 16;

// Types.
using mesh::FaceArray;
using mesh::GeneralMesh;
using math::modulo;

// DooSabinWeights
template <typename Weights>
void DooSabinWeights(int N, Weights* w) {
  typedef typename Weights::Scalar T;

  assert(N >= 3);

  if (w->size() != N) {
    w->resize(N);
  }

  for (int i = 1; i < N; ++i) {
    (*w)[i] = (3 + 2 * cos(2 * T(M_PI) * i / N)) / (4 * N);
  }
  (*w)[0] = T(N + 5) / (4 * N);
}

// BiquadraticBsplineBasis
const size_t kNumBiquadraticBsplineBasis = 9;
const int kBiquadraticBsplineBasis[kNumBiquadraticBsplineBasis][2] = {{1, 1},
                                                                      {1, 0},
                                                                      {0, 0},
                                                                      {0, 1},
                                                                      {0, 2},
                                                                      {1, 2},
                                                                      {2, 2},
                                                                      {2, 1},
                                                                      {2, 0}};

template <typename F, typename G, typename U, typename B>
inline void BiquadraticBsplineBasis(const U& u, B* b) {
  if (b->size() != kNumBiquadraticBsplineBasis) {
    b->resize(kNumBiquadraticBsplineBasis);
  }

  static const F f;
  static const G g;

  for (typename B::Index i = 0; i < kNumBiquadraticBsplineBasis; ++i) {
    (*b)[i] = f(u[0], kBiquadraticBsplineBasis[i][0]) *
              g(u[1], kBiquadraticBsplineBasis[i][1]);
  }
}

// kValidUOffsets
static const int kValidUOffsets[][2] = {{-1, -1},
                                        {-1,  1},
                                        { 1,  1},
                                        { 1, -1}};

// Surface
template <typename Scalar>
class Surface;

// Patch
template <typename Scalar>
class Patch {
 public:
  friend Surface<Scalar>;

  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
  typedef Eigen::Matrix<Scalar, 2, 1> Vector2;

  explicit Patch(FaceArray* face_array,
        const Patch* parent = nullptr,
        size_t depth = 0)
      : _parent(parent),
        _depth(depth),
        _face_array(face_array) {
    Initialise();
    if (_depth == 0) {
      _S.setIdentity(_I.size(), _I.size());
      if (!_is_valid) {
        Subdivide();
      }
    }
  }

  #define EVALUATE(M, F, G, S) \
  template <typename U, typename TX, typename R> \
  void M(const U& u, const TX& X, R* r) const { \
    Evaluate<uniform_quadratic_bspline:: F, \
             uniform_quadratic_bspline:: G, S>(u, X, r); \
  }
  EVALUATE(M, Position, Position, MultiplyAndScale<1>);
  EVALUATE(Mu, FirstDerivative, Position, MultiplyAndScale<2>);
  EVALUATE(Mv, Position, FirstDerivative, MultiplyAndScale<2>);
  EVALUATE(Muu, SecondDerivative, Position, MultiplyAndScale<4>);
  EVALUATE(Muv, FirstDerivative, FirstDerivative, MultiplyAndScale<4>);
  EVALUATE(Mvv, Position, SecondDerivative, MultiplyAndScale<4>);
  EVALUATE(Mx, Position, Position, MultiplyAndRepeat);
  #undef EVALUATE

  const std::vector<int>& vertex_indices() const {
    return _I;
  }

  const std::vector<int>& ordered_face_indices() const {
    return _ordered_face_indices;
  }

 private:
  size_t GetFaceIndexWithAdjacentVertex(int j) const {
    for (size_t i = 0; i < _face_array->number_of_faces(); ++i) {
      if (_face_array->face(i)[1] == j) {
        return i;
      }
    }
    return std::numeric_limits<size_t>::max();
  }

  size_t GetFaceIndexOfFace(const std::vector<int>& face) {
    auto it = std::find(face.begin(), face.end(), _I[0]);
    if (it != face.end()) {
      size_t j = std::distance(face.begin(), it);
      int half_edge = _face_array->FindHalfEdge(
        _I[0], face[(j + 1) % face.size()]);
      if (half_edge >= 0) {
        return half_edge;
      }
    }
    return std::numeric_limits<size_t>::max();
  }

 private:
  // Initialise
  void Initialise() {
    // Set `_is_valid`.
    assert(_face_array->number_of_faces() == 4);
    _is_valid = true;
    for (size_t j = 0; j < _face_array->number_of_faces(); ++j) {
      if (_face_array->number_of_sides(j) != 4) {
        _is_valid = false;
        break;
      }
    }

    // Reorder the vertex indices and face ordering in `_face_array`.

    // Determine the common vertex `i` for all faces.
    const int i = _face_array->FindCommonVertex();
    assert(i >= 0);

    // Order the vertices in each face so that `i` is the first vertex.
    // Order the faces to also respect the vertex ordering.
    const size_t n_faces = _face_array->number_of_faces();
    size_t faces_remaining = n_faces;

    _ordered_face_indices.reserve(n_faces);

    // Order first face.
    _face_array->RotateFaceToVertex(0, i);
    _ordered_face_indices.push_back(0);

    // Order remaining faces.
    while (--faces_remaining) {
      size_t last_face_index = _ordered_face_indices.back();
      int n = _face_array->number_of_sides(last_face_index);
      auto* f = _face_array->face(last_face_index);
      const int last_vertex = f[n - 1];

      // Search for half edge (`i`, `last_vertex`) to find `next_face_offset`.
      size_t next_face_index = std::numeric_limits<size_t>::max();

      for (size_t j = 0; j < n_faces; ++j) {
        n = _face_array->number_of_sides(j);
        f = _face_array->face(j);

        for (int k = 0; k < n; ++k) {
          if (i == f[k] && last_vertex == f[(k + 1) % n]) {
            next_face_index = j;
            break;
          }
        }

        if (next_face_index != std::numeric_limits<size_t>::max()) {
          break;
        }
      }

      // Ensure the half edge was found.
      assert(next_face_index != std::numeric_limits<size_t>::max());

      _face_array->RotateFaceToVertex(next_face_index, i);
      _ordered_face_indices.push_back(next_face_index);
    }

    _face_array->PermuteFaces(_ordered_face_indices);

    // Construct `_I` from the reordered `_face_array`.
    _I.push_back(i);

    for (size_t j = 0; j < _face_array->number_of_faces(); ++j) {
      int n = _face_array->number_of_sides(j);
      auto* f = _face_array->face(j);
      std::copy(f + 1, f + n - 1, std::back_inserter(_I));
    }
  }

  // Subdivision
  void Subdivide() {
    // Create `S` to include all of the child vertices.
    size_t n_child_vertices = 0;
    for (size_t i = 0; i < 4; ++i) {
      n_child_vertices += _face_array->number_of_sides(i);
    }

    Matrix S(n_child_vertices, _I.size());
    S.fill(0);

    // Fill `S` using the Doo-Sabin subdivision weights.
    int child_index = 0;
    for (size_t i = 0; i < 4; ++i) {
      // Get subdivision weights of face `i` with `n` vertices.
      const int n = _face_array->number_of_sides(i);
      Vector w;
      DooSabinWeights(n, &w);

      // Get `face_indices_in_I`.
      std::vector<size_t> face_indices_in_I(n);
      auto f = _face_array->face(i);

      for (int j = 0; j < n; ++j) {
        // Find index of vertex `f[j]` in `_I`.
        auto it = std::find(_I.begin(), _I.end(), f[j]);
        assert(it != _I.end());
        face_indices_in_I[j] = std::distance(_I.begin(), it);
      }

      // Copy `w` into `S` for each child vertex.
      for (int j = 0; j < n; ++j) {
        for (int k = 0; k < n; ++k) {
          S(child_index, face_indices_in_I[k]) = w[modulo(k - j, n)];
        }

        ++child_index;
      }
    }

    // Get `S` to reference the top level vertices.
    assert(_S.rows() > 0 && _S.cols() > 0);
    S *= _S;

    // Build `child_face_array`.
    child_index = 0;
    std::vector<int> raw_child_face_array;
    raw_child_face_array.push_back(4);
    for (size_t i = 0; i < 4; ++i) {
      int n = _face_array->number_of_sides(i);
      raw_child_face_array.push_back(n);

      for (int j = 0; j < n; ++j) {
        raw_child_face_array.push_back(child_index++);
      }
    }
    FaceArray child_face_array(std::move(raw_child_face_array));

    // Build child patches.
    for (int i = 0; i < 4; ++i) {
      // Four child faces are created because patch has valency four.
      raw_child_face_array.push_back(4);

      auto face = child_face_array.face(i);
      auto next_face = child_face_array.face((i + 1) % 4);
      auto opp_face = child_face_array.face((i + 2) % 4);
      auto prev_face = child_face_array.face((i + 3) % 4);

      // First child face.
      const int n = child_face_array.number_of_sides(i);
      raw_child_face_array.push_back(n);
      std::copy(face, face + n, std::back_inserter(raw_child_face_array));

      // Next three generated faces.
      const int n_prev = child_face_array.number_of_sides((i + 3) % 4);

      const int child_faces[][4] = {
        {face[0], face[modulo(-1, n)], next_face[1], next_face[0]},
        {face[0], next_face[0], opp_face[0], prev_face[0]},
        {face[0], prev_face[0], prev_face[modulo(-1, n_prev)], face[1]}
      };

      for (int j = 0; j < 3; ++j) {
        raw_child_face_array.push_back(4);
        std::copy(child_faces[j],
                  child_faces[j] + 4,
                  std::back_inserter(raw_child_face_array));
      }

      // Build `child_patch_array`.
      auto child_patch_array = new FaceArray(std::move(raw_child_face_array));

      // Permute the patch faces to preserve `u` directionality.
      std::vector<int> child_face_permutation(4);
      child_face_permutation[0] = modulo(0 - i, 4);
      child_face_permutation[1] = modulo(1 - i, 4);
      child_face_permutation[2] = modulo(2 - i, 4);
      child_face_permutation[3] = modulo(3 - i, 4);
      child_patch_array->PermuteFaces(child_face_permutation);

      // Build `child`.
      auto child = new Patch<Scalar>(child_patch_array, this, _depth + 1);

      // Set the child subdivision matrix `_S` and subdivide if
      // necessary.
      // NOTE `Subdivide` is only called in `Initialise` for `_depth = 0` so
      // this is all OK.
      child->_S.resize(child->_I.size(), S.cols());
      for (size_t i = 0; i < child->_I.size(); ++i) {
        child->_S.row(i) = S.row(child->_I[i]);
      }
      if (!child->_is_valid && child->_depth < kMaxSubdivisionDepth) {
        child->Subdivide();
      }

      _children.emplace_back(child);
    }
  }

  // Evaluation
  template <typename F, typename G, typename E, typename U, typename TX,
            typename R>
  void Evaluate(const U& u, const TX& X, R* r) const {
    Vector2 _u(u);
    assert(r != nullptr);
    EvaluateInternal<F, G, E>(&_u, X, r);
  }

  template <typename F, typename G, typename E, typename TX, typename R>
  void EvaluateInternal(Vector2* u, const TX& X, R* r) const {
    if (_is_valid) {
      // Get the basis vector for the quantity and evaluate.
      Eigen::Matrix<Scalar, kNumBiquadraticBsplineBasis, 1> b;
      BiquadraticBsplineBasis<F, G>(*u, &b);
      static const E e;
      e(_depth, _S, b, X, r);
    } else {
      assert(_depth <= (kMaxSubdivisionDepth - 1));
      if (_depth == (kMaxSubdivisionDepth - 1)) {
        // On second to last level of subdivision, adjust `U` so that it falls
        // within a valid patch.
        AdjustUForValidChild(u);
      }

      // Get child and translate `u` for child patch.
      _children[PassToChild(u)]->EvaluateInternal<F, G, E>(u, X, r);
    }
  }

  void AdjustUForValidChild(Vector2* u) const {
    assert(_children.size() > 0);

    for (size_t i = 0; i < _children.size(); ++i) {
      if (_children[i]->_is_valid) {
        (*u)[0] = Scalar(0.5) + kValidUOffsets[i][0] * Scalar(kUEps);
        (*u)[1] = Scalar(0.5) + kValidUOffsets[i][1] * Scalar(kUEps);
        return;
      }
    }
  }

  int PassToChild(Vector2* u) const {
    int child_index;
    auto& _u = *u;
    if (_u[0] > Scalar(0.5)) {
      _u[0] -= Scalar(0.5);

      if (_u[1] > Scalar(0.5)) {
        _u[1] -= Scalar(0.5);
        child_index = 2;
      } else {
        child_index = 3;
      }
    } else {
      if (_u[1] > Scalar(0.5)) {
        _u[1] -= Scalar(0.5);
        child_index = 1;
      } else {
        child_index = 0;
      }
    }

    _u[0] *= Scalar(2);
    _u[1] *= Scalar(2);

    return child_index;
  }

  template <int Exponent>
  struct MultiplyAndScale {
    template <typename S, typename B, typename TX, typename R>
    inline void operator()(size_t depth, const S& S, const B& b, const TX& X,
                           R* r) const {
      if (S.cols() <= kMaxNNoAlloc) {
        Scalar St_b_data[kMaxNNoAlloc];
        Eigen::Map<Vector> St_b(St_b_data, S.cols());
        St_b.noalias() = S.transpose() * b;
        r->noalias() = X * St_b;
      } else {
        r->noalias() = X * (S.transpose() * b);
      }
      if (Exponent > 1) {
        *r *= pow(Scalar(Exponent), static_cast<int>(depth));
      }
    }
  };

  class MultiplyAndRepeat {
   public:
    template <typename S, typename B, typename TX, typename R>
    inline void operator()(size_t depth, const S& S, const B& b, const TX& X,
                           R* r) const {
      if (S.cols() <= kMaxNNoAlloc) {
        Scalar St_b_data[kMaxNNoAlloc];
        Eigen::Map<Vector> St_b(St_b_data, S.cols());
        St_b.noalias() = S.transpose() * b;
        Repeat(St_b, X.rows(), r);
      } else {
        Vector St_b;
        St_b.noalias() = S.transpose() * b;
        Repeat(St_b, X.rows(), r);
      }
    }

   private:
    template <typename STB, typename R>
    inline void Repeat(const STB& St_b, typename STB::Index d, R* r) const {
      typedef typename STB::Index Index;
      const Index n = St_b.size();
      r->resize((d * d) * n);
      r->setZero();

      for (Index i = 0; i < n; ++i) {
        for (Index j = 0; j < d; ++j) {
          (*r)[d * d * i + d * j + j] = St_b[i];
        }
      }
    }
  };

 private:
  std::unique_ptr<FaceArray> _face_array;
  const Patch<Scalar>* _parent;
  size_t _depth;

  bool _is_valid;
  std::vector<int> _ordered_face_indices;
  std::vector<int> _I;

  Matrix _S;

  std::vector<std::unique_ptr<Patch<Scalar>>> _children;
};

// Surface
template <typename Scalar>
class Surface {
 public:
  typedef Patch<Scalar> Patch;
  typedef typename Patch::Matrix Matrix;
  typedef typename Patch::Vector Vector;
  typedef typename Patch::Vector2 Vector2;

  // Copy of `control_mesh` is required since mutable routines (e.g.
  // `EnsureVertices`) are used.
  explicit Surface(const GeneralMesh& control_mesh)
      : _control_mesh(control_mesh) {
    Initialise();
  }

  #define EVALUATE(M) \
  template <typename U, typename TX, typename R> \
  void M(int p, const U& u, const TX& X, R* r) const { \
    _patches[p]->M(u, X, r); \
  }
  EVALUATE(M);
  EVALUATE(Mu);
  EVALUATE(Mv);
  EVALUATE(Muu);
  EVALUATE(Muv);
  EVALUATE(Mvv);
  EVALUATE(Mx);
  #undef EVALUATE

  template <typename P, typename TU>
  void UniformParameterisation(int N, P* p, TU* U,
                               std::vector<int>* T = nullptr) {
    // Ensure `N >= 1`.
    N = std::max(N, 1);

    size_t samples_per_patch = N * N;
    size_t num_samples = _patch_vertex_indices.size() * samples_per_patch;

    p->resize(num_samples);
    U->resize(2, num_samples);

    const Scalar delta = Scalar(1) / N;

    for (size_t i = 0; i < _patch_vertex_indices.size(); ++i) {
      for (size_t j = 0; j < N; ++j) {
        for (size_t k = 0; k < N; ++k) {
          size_t l = i * (N * N) + j * N + k;
          (*p)[l] = static_cast<std::decay<decltype((*p)[0])>::type>(i);
          (*U)(0, l) = (Scalar(0.5) + k) * delta;
          (*U)(1, l) = (Scalar(0.5) + j) * delta;
        }
      }
    }

    if (T == nullptr) {
      return;
    }

    auto& T_ = *T;

    // Reserve number of faces.
    T_.clear();
    T_.push_back(0);

    // Add quadrilaterals within each patch.
    for (size_t i = 0; i < _patch_vertex_indices.size(); ++i) {
      for (int j = 0; j < (N - 1); ++j) {
        for (int k = 0; k < (N - 1); ++k) {
          int l = static_cast<int>(i) * (N * N) + j * N + k;
          T_.push_back(4);
          T_.push_back(l);
          T_.push_back(l + 1);
          T_.push_back(l + N + 1);
          T_.push_back(l + N);
          ++T_[0];
        }
      }
    }

    // Prepare `border_offsets`.
    std::vector<std::vector<int>> border_offsets(4);
    for (int i = 0; i < N; ++i) {
      border_offsets[0].push_back(i);
      border_offsets[1].push_back((N - 1) + i * N);
      border_offsets[2].push_back(N * N - 1 - i);
      border_offsets[3].push_back(N * (N - 1 - i));
    }

    // Add quadrilaterals between patches.
    for (size_t i_index = 0; i_index < _patch_vertex_indices.size();
         ++i_index) {
      int i = _patch_vertex_indices[i_index];
      int i_offset = static_cast<int>(i_index) * (N * N);

      for (int half_edge_index : _control_mesh.half_edges_from_vertex(i)) {
        // Find adjacent patch at vertex `j` (with patch offset `j_offset`).
        // Skip if `i < j` so that the boundary is only processed once.
        int j = _control_mesh.half_edge(half_edge_index).second;
        if (i < j) {
          continue;
        }

        int j_index = _vertex_to_patch_index[j];
        if (j_index < 0) {
          continue;
        }

        int j_offset = j_index * (N * N);

        // Get the border offsets for each patch.
        auto& i_vertex_offsets = border_offsets[
          _patches[i_index]->GetFaceIndexWithAdjacentVertex(j)];
        auto& j_vertex_offsets = border_offsets[
          _patches[j_index]->GetFaceIndexWithAdjacentVertex(i)];

        // Add quadrilaterals.
        for (int k = 0; k < (N - 1); ++k) {
          T_.push_back(4);
          T_.push_back(i_offset + i_vertex_offsets[k]);
          T_.push_back(j_offset + j_vertex_offsets[N - 1 - k]);
          T_.push_back(j_offset + j_vertex_offsets[N - 2 - k]);
          T_.push_back(i_offset + i_vertex_offsets[k + 1]);
          ++T_[0];
        }
      }
    }

    // Add faces at corners of patches.
    std::vector<int> current_face, next_face;

    for (size_t face_index = 0; face_index < _control_mesh.number_of_faces();
         ++face_index) {
      auto f = _control_mesh.face(face_index);
      current_face.clear();
      std::copy(f, f + _control_mesh.number_of_sides(face_index),
                std::back_inserter(current_face));

      next_face.clear();
      next_face.push_back(0);
      bool is_next_face_valid = true;

      for (int i : current_face) {
        // Get patch index `i_index` for vertex `i` from `face`.
        int i_index = _vertex_to_patch_index[i];
        if (i_index < 0) {
          is_next_face_valid = false;
          break;
        }

        // Get offset of `current_face` in the patch and set
        // `i_vertex_offsets`.
        auto& i_vertex_offsets = border_offsets[
          _patches[i_index]->GetFaceIndexOfFace(current_face)];
        next_face.push_back(i_vertex_offsets[N - 1] + i_index * (N * N));
        ++next_face[0];
      }

      if (is_next_face_valid) {
        std::copy(next_face.begin(), next_face.end(), std::back_inserter(T_));
        ++T_[0];
      }
    }
  }

  // FIXME This should be `const` but `_control_mesh` needs mutation.
  int number_of_vertices() {
    return _control_mesh.number_of_vertices();
  }

  int number_of_faces() const {
    return _control_mesh.number_of_faces();
  }

  // const std::vector<int>& face_vertex_indices(const int f) const;

  int number_of_patches() const {
    return static_cast<int>(_patch_vertex_indices.size());
  }

  const std::vector<int>& patch_vertex_indices(int p) const {
    return _patches[p]->vertex_indices();
  }

  const std::vector<int>& adjacent_patch_indices(int p) const {
    return _adjacent_patch_indices[p];
  }

 private:
  void Initialise() {
    InitialisePatchIndices();
    InitialisePatches();
  }

  void InitialisePatchIndices() {
    // FIXME This is assuming contiguous vertex labelling starting at 0.
    _control_mesh.EnsureVertices();
    auto& vertices = _control_mesh.vertices();
    _vertex_to_patch_index.resize(vertices.size());

    int patch_index = 0;
    for (int i : vertices) {
      if (_control_mesh.is_vertex_closed(i)) {
        assert(_control_mesh.AdjacentVertices(i).size() == 4);
        _patch_vertex_indices.push_back(i);
        _vertex_to_patch_index[i] = patch_index++;
      } else {
        _vertex_to_patch_index[i] = -1;
      }
    }
  }

  void InitialisePatches() {
    std::map<int, std::vector<std::pair<int, int>>> vertex_to_half_edges;
    std::map<std::pair<int, int>, int> half_edge_to_vertex;

    _patches.reserve(_patch_vertex_indices.size());
    for (int i : _patch_vertex_indices) {
      // Initialise `patch`.
      std::vector<int> face_indices = _control_mesh.FacesAtVertex(i);
      auto patch = new Patch(new FaceArray(_control_mesh.Faces(face_indices)));

      // Get the permuted face indices.
      std::vector<int> permuted_face_indices;
      permuted_face_indices.reserve(face_indices.size());
      for (size_t j : patch->ordered_face_indices()) {
        permuted_face_indices.push_back(face_indices[j]);
      }
      assert(permuted_face_indices.size() == 4);

      // A "half edge" is an ordered pair, where each entry is a face index.
      for (size_t j = 0; j < 4; ++j) {
        auto half_edge = std::make_pair(permuted_face_indices[j],
                                        permuted_face_indices[(j + 1) % 4]);
        vertex_to_half_edges[i].push_back(half_edge);
        half_edge_to_vertex[half_edge] = i;
      }

      // Save the generated patch.
      _patches.emplace_back(patch);
    }

    // Set `_adjacent_patch_indices`.
    _control_mesh.EnsureVertices();
    _adjacent_patch_indices.resize(_control_mesh.number_of_vertices());
    for (int i : _patch_vertex_indices) {
      for (auto& half_edge : vertex_to_half_edges[i]) {
        auto opposite_half_edge = std::make_pair(half_edge.second,
                                                 half_edge.first);
        int adj_patch_index = -1;
        auto it = half_edge_to_vertex.find(opposite_half_edge);
        if (it != half_edge_to_vertex.end()) {
          adj_patch_index = _vertex_to_patch_index[it->second];
        }
        _adjacent_patch_indices[i].push_back(adj_patch_index);
      }
    }
  }

 private:
  GeneralMesh _control_mesh;

  std::vector<int> _patch_vertex_indices;
  std::vector<int> _vertex_to_patch_index;
  std::vector<std::vector<int>> _adjacent_patch_indices;
  std::vector<std::unique_ptr<Patch>> _patches;
};

} // namespace doosabin

#endif // DOOSABIN_H
