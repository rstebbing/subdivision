////////////////////////////////////////////
// File: doosabin.h                       //
// Copyright Richard Stebbing 2014.       //
// Distributed under the MIT License.     //
// (See accompany file LICENSE or copy at //
//  http://opensource.org/licenses/MIT)   //
////////////////////////////////////////////
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
#include "Math/linalg.h"
#include "Math/modulo.h"

#include "uniform_quadratic_bspline.h"

// doosabin
namespace doosabin {

// Constants.

// `kMaxSubdivisionDepth` and `kUEps` set the subdivision limit and
// adjustment to coordinates on the penultimate subdivision level.
static const int kMaxSubdivisionDepth = 10;
static const double kUEps = 1e-6;

// For `N <= kMaxNNoAlloc`, the intermediate subdivided basis vector in
// `EvaluateInternal` is stored on the stack. For `N > kMaxNNoAlloc`,
// a call to `malloc` is necessary.
static const int kMaxNNoAlloc = 16;

// Types.
using mesh::FaceArray;
using mesh::GeneralMesh;
using linalg::MatrixOfColumnPointers;
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
const int kNumBiquadraticBsplineBasis = 9;
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
        int depth = 0)
      : parent_(parent),
        depth_(depth),
        face_array_(face_array) {
    Initialise();
    if (depth_ == 0) {
      // Required by `GetFaceIndexOfFace`.
      face_array_->EnsureHalfEdgeToFaceIndex();

      S_.setIdentity(I_.size(), I_.size());
      if (!is_valid_) {
        Subdivide();
      }
    }
  }

  #define EVALUATE(M, F, G, S) \
  template <typename U, typename TX, typename R> \
  inline void M(const U& u, const TX& X, R* r) const { \
    Evaluate<uniform_quadratic_bspline:: F, \
             uniform_quadratic_bspline:: G, S>(u, X, r); \
  }
  EVALUATE(M, Position, Position, MultiplyAndScale<1>);
  EVALUATE(Mu, FirstDerivative, Position, MultiplyAndScale<2>);
  EVALUATE(Mv, Position, FirstDerivative, MultiplyAndScale<2>);
  EVALUATE(Muu, SecondDerivative, Position, MultiplyAndScale<4>);
  EVALUATE(Muv, FirstDerivative, FirstDerivative, MultiplyAndScale<4>);
  EVALUATE(Mvv, Position, SecondDerivative, MultiplyAndScale<4>);
  EVALUATE(Mx, Position, Position, MultiplyAndRepeat<1>);
  EVALUATE(Mux, FirstDerivative, Position, MultiplyAndRepeat<2>);
  EVALUATE(Mvx, Position, FirstDerivative, MultiplyAndRepeat<2>);
  #undef EVALUATE

  const std::vector<int>& vertex_indices() const {
    return I_;
  }

  const std::vector<int>& ordered_face_indices() const {
    return ordered_face_indices_;
  }

 private:
  int GetFaceIndexWithAdjacentVertex(int j) const {
    for (int i = 0; i < face_array_->number_of_faces(); ++i) {
      if (face_array_->face(i)[1] == j) {
        return i;
      }
    }
    return -1;
  }

  int GetFaceIndexOfFace(const std::vector<int>& face) const {
    auto it = std::find(face.begin(), face.end(), I_[0]);
    if (it != face.end()) {
      int face_index = face_array_->HalfEdgeToFaceIndex(
        I_[0], face[(std::distance(face.begin(), it) + 1) % face.size()]);
      if (face_index >= 0) {
        return face_index;
      }
    }
    return -1;
  }

 private:
  // Initialise
  void Initialise() {
    // Set `is_valid_`.
    assert(face_array_->number_of_faces() == 4);
    is_valid_ = true;
    for (int j = 0; j < face_array_->number_of_faces(); ++j) {
      if (face_array_->number_of_sides(j) != 4) {
        is_valid_ = false;
        break;
      }
    }

    // Reorder the vertex indices and face ordering in `face_array_`.

    // Determine the common vertex `i` for all faces.
    const int i = face_array_->FindCommonVertex();
    assert(i >= 0);

    // Order the vertices in each face so that `i` is the first vertex.
    // Order the faces to also respect the vertex ordering.
    const int n_faces = face_array_->number_of_faces();
    int faces_remaining = n_faces;

    ordered_face_indices_.reserve(n_faces);

    // Order first face.
    face_array_->RotateFaceToVertex(0, i);
    ordered_face_indices_.push_back(0);

    // Order remaining faces.
    while (--faces_remaining) {
      int last_face_index = ordered_face_indices_.back();
      int n = face_array_->number_of_sides(last_face_index);
      auto* f = face_array_->face(last_face_index);
      const int last_vertex = f[n - 1];

      // Search for half edge (`i`, `last_vertex`) to find `next_face_offset`.
      int next_face_index = -1;

      for (int j = 0; j < n_faces; ++j) {
        n = face_array_->number_of_sides(j);
        f = face_array_->face(j);

        for (int k = 0; k < n; ++k) {
          if (i == f[k] && last_vertex == f[(k + 1) % n]) {
            next_face_index = j;
            break;
          }
        }

        if (next_face_index >= 0) {
          break;
        }
      }

      // Ensure the half edge was found.
      assert(next_face_index >= 0);

      face_array_->RotateFaceToVertex(next_face_index, i);
      ordered_face_indices_.push_back(next_face_index);
    }

    face_array_->PermuteFaces(ordered_face_indices_);

    // Construct `I_` from the reordered `face_array_`.
    I_.push_back(i);

    for (int j = 0; j < face_array_->number_of_faces(); ++j) {
      int n = face_array_->number_of_sides(j);
      auto* f = face_array_->face(j);
      std::copy(f + 1, f + n - 1, std::back_inserter(I_));
    }
  }

  // Subdivision
  void Subdivide() {
    // Create `S` to include all of the child vertices.
    int n_child_vertices = 0;
    for (int i = 0; i < 4; ++i) {
      n_child_vertices += face_array_->number_of_sides(i);
    }

    Matrix S(n_child_vertices, I_.size());
    S.fill(0);

    // Fill `S` using the Doo-Sabin subdivision weights.
    int child_index = 0;
    for (int i = 0; i < 4; ++i) {
      // Get subdivision weights of face `i` with `n` vertices.
      const int n = face_array_->number_of_sides(i);
      Vector w;
      DooSabinWeights(n, &w);

      // Get `face_indices_in_I`.
      std::vector<int> face_indices_in_I(n);
      auto f = face_array_->face(i);

      for (int j = 0; j < n; ++j) {
        // Find index of vertex `f[j]` in `I_`.
        auto it = std::find(I_.begin(), I_.end(), f[j]);
        assert(it != I_.end());
        face_indices_in_I[j] = static_cast<int>(std::distance(I_.begin(), it));
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
    assert(S_.rows() > 0 && S_.cols() > 0);
    S *= S_;

    // Build `child_face_array`.
    child_index = 0;
    std::vector<int> raw_child_face_array;
    raw_child_face_array.push_back(4);
    for (int i = 0; i < 4; ++i) {
      int n = face_array_->number_of_sides(i);
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
      auto child = new Patch<Scalar>(child_patch_array, this, depth_ + 1);

      // Set the child subdivision matrix `S_` and subdivide if
      // necessary.
      // NOTE `Subdivide` is only called in `Initialise` for `depth_ = 0` so
      // this is all OK.
      child->S_.resize(child->I_.size(), S.cols());
      for (int i = 0; i < child->I_.size(); ++i) {
        child->S_.row(i) = S.row(child->I_[i]);
      }
      if (!child->is_valid_ && child->depth_ < kMaxSubdivisionDepth) {
        child->Subdivide();
      }

      children_.emplace_back(child);
    }
  }

  // Evaluation
  template <typename F, typename G, typename E, typename U, typename TX,
            typename R>
  inline void Evaluate(const U& u, const TX& X, R* r) const {
    Vector2 _u(u);
    assert(r != nullptr);
    EvaluateInternal<F, G, E>(&_u, X, r);
  }

  template <typename F, typename G, typename E, typename TX, typename R>
  void EvaluateInternal(Vector2* u, const TX& X, R* r) const {
    if (is_valid_) {
      // Get the basis vector for the quantity and evaluate.
      Eigen::Matrix<Scalar, kNumBiquadraticBsplineBasis, 1> b;
      BiquadraticBsplineBasis<F, G>(*u, &b);
      static const E e;
      e(depth_, S_, b, X, r);
    } else {
      assert(depth_ <= (kMaxSubdivisionDepth - 1));
      if (depth_ == (kMaxSubdivisionDepth - 1)) {
        // On second to last level of subdivision, adjust `U` so that it falls
        // within a valid patch.
        AdjustUForValidChild(u);
      }

      // Get child and translate `u` for child patch.
      children_[PassToChild(u)]->EvaluateInternal<F, G, E>(u, X, r);
    }
  }

  void AdjustUForValidChild(Vector2* u) const {
    assert(children_.size() > 0);

    for (int i = 0; i < children_.size(); ++i) {
      if (children_[i]->is_valid_) {
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
  class MultiplyAndScale {
   public:
    // Define `MatrixVectorMultiply` to facilitate non-Eigen type
    // `MatrixOfColumnPointers` for `TX`.
    // TODO Add `MatrixOfColumnPointers` as an Eigen extension.
    template <typename S, typename B, typename TX, typename R>
    inline void operator()(int depth, const S& S, const B& b, const TX& X,
                           R* r) const {
      if (S.cols() <= kMaxNNoAlloc) {
        Scalar St_b_data[kMaxNNoAlloc];
        Eigen::Map<Vector> St_b(St_b_data, S.cols());
        St_b.noalias() = S.transpose() * b;
        MatrixVectorMultiply(X, St_b, r);
      } else {
        // Make evaluation of the matrix product explicit for
        // `MatrixOfColumnPointers`. None of this is particularly great ...
        Vector St_b = S.transpose() * b;
        MatrixVectorMultiply(X, St_b, r);
      }
      if (Exponent > 1) {
        *r *= pow(Scalar(Exponent), depth);
      }
    }

   private:
    template <typename TX, typename Y, typename R>
    inline void MatrixVectorMultiply(const TX& X, const Y& y, R* r) const {
      r->noalias() = X * y;
    }

    template <typename Y, typename R>
    inline void MatrixVectorMultiply(const MatrixOfColumnPointers<Scalar>& X,
                                     const Y& y,
                                     R* r) const {
      X.MultiplyVector(y, r);
    }
  };

  template <int Exponent>
  class MultiplyAndRepeat {
   public:
    template <typename S, typename B, typename TX, typename R>
    inline void operator()(int depth, const S& S, const B& b, const TX& X,
                           R* r) const {
      if (S.cols() <= kMaxNNoAlloc) {
        Scalar St_b_data[kMaxNNoAlloc];
        Eigen::Map<Vector> St_b(St_b_data, S.cols());
        DoMultiplyAndRepeat(depth, S, b, &St_b, X.rows(), r);
      } else {
        Vector St_b;
        DoMultiplyAndRepeat(depth, S, b, &St_b, X.rows(), r);
      }
    }

   private:
    template <typename S, typename B, typename STB, typename R>
    inline void DoMultiplyAndRepeat(int depth, const S& S, const B& b,
                                    STB* St_b, typename STB::Index d,
                                    R* r) const {
      St_b->noalias() = S.transpose() * b;
      if (Exponent > 1) {
        *St_b *= pow(Scalar(Exponent), depth);
      }

      typedef typename STB::Index Index;
      const Index n = St_b->size();
      r->resize((d * d) * n);
      r->setZero();

      for (Index i = 0; i < n; ++i) {
        for (Index j = 0; j < d; ++j) {
          (*r)[d * d * i + d * j + j] = (*St_b)[i];
        }
      }
    }
  };

 private:
  std::unique_ptr<FaceArray> face_array_;
  const Patch<Scalar>* parent_;
  int depth_;

  bool is_valid_;
  std::vector<int> ordered_face_indices_;
  std::vector<int> I_;

  Matrix S_;

  std::vector<std::unique_ptr<Patch<Scalar>>> children_;
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
      : control_mesh_(control_mesh) {
    Initialise();
  }

  #define EVALUATE(M) \
  template <typename U, typename TX, typename R> \
  inline void M(int p, const U& u, const TX& X, R* r) const { \
    patches_[p]->M(u, X, r); \
  }
  EVALUATE(M);
  EVALUATE(Mu);
  EVALUATE(Mv);
  EVALUATE(Muu);
  EVALUATE(Muv);
  EVALUATE(Mvv);
  EVALUATE(Mx);
  EVALUATE(Mux);
  EVALUATE(Mvx);
  #undef EVALUATE

  template <typename P, typename TU>
  void UniformParameterisation(int N, P* p, TU* U,
                               std::vector<int>* T = nullptr) const {
    typedef std::decay<decltype((*p)[0])>::type PatchIndex;

    // Ensure `N >= 1`.
    N = std::max(N, 1);

    const int samples_per_patch = N * N;
    const int num_samples = number_of_patches() * samples_per_patch;

    p->resize(num_samples);
    U->resize(2, num_samples);

    const Scalar delta = Scalar(1) / N;

    for (int i = 0; i < number_of_patches(); ++i) {
      for (int j = 0; j < N; ++j) {
        for (int k = 0; k < N; ++k) {
          int l = i * (N * N) + j * N + k;
          (*p)[l] = static_cast<PatchIndex>(i);
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
    for (int i = 0; i < number_of_patches(); ++i) {
      for (int j = 0; j < (N - 1); ++j) {
        for (int k = 0; k < (N - 1); ++k) {
          int l = i * (N * N) + j * N + k;
          T_.push_back(4);
          T_.push_back(l + 1);
          T_.push_back(l);
          T_.push_back(l + N);
          T_.push_back(l + N + 1);
          ++T_[0];
        }
      }
    }

    // Prepare `border_offsets`.
    std::vector<std::vector<int>> border_offsets(4);
    for (int i = 0; i < N; ++i) {
      border_offsets[0].push_back(i);
      border_offsets[1].push_back(N * (N - 1 - i));
      border_offsets[2].push_back(N * N - 1 - i);
      border_offsets[3].push_back((N - 1) + i * N);
    }

    // Add quadrilaterals between patches.
    for (int i_index = 0; i_index < number_of_patches(); ++i_index) {
      int i = patch_vertex_indices_[i_index];
      int i_offset = i_index * (N * N);

      for (int half_edge_index : control_mesh_.half_edges_from_vertex(i)) {
        // Find adjacent patch at vertex `j` (with patch offset `j_offset`).
        // Skip if `i < j` so that the boundary is only processed once.
        int j = control_mesh_.half_edge(half_edge_index).second;
        if (i < j) {
          continue;
        }

        int j_index = vertex_to_patch_index_[j];
        if (j_index < 0) {
          continue;
        }

        int j_offset = j_index * (N * N);

        // Get the border offsets for each patch.
        auto& i_vertex_offsets = border_offsets[
          patches_[i_index]->GetFaceIndexWithAdjacentVertex(j)];
        auto& j_vertex_offsets = border_offsets[
          patches_[j_index]->GetFaceIndexWithAdjacentVertex(i)];

        // Add quadrilaterals.
        for (int k = 0; k < (N - 1); ++k) {
          T_.push_back(4);
          T_.push_back(i_offset + i_vertex_offsets[k + 1]);
          T_.push_back(j_offset + j_vertex_offsets[N - 2 - k]);
          T_.push_back(j_offset + j_vertex_offsets[N - 1 - k]);
          T_.push_back(i_offset + i_vertex_offsets[k]);
          ++T_[0];
        }
      }
    }

    // Add faces at corners of patches.
    std::vector<int> current_face, next_face;

    for (int face_index = 0; face_index < control_mesh_.number_of_faces();
         ++face_index) {
      auto f = control_mesh_.face(face_index);
      current_face.clear();
      std::copy(f, f + control_mesh_.number_of_sides(face_index),
                std::back_inserter(current_face));

      next_face.clear();
      next_face.push_back(0);
      bool is_next_face_valid = true;

      for (int i : current_face) {
        // Get patch index `i_index` for vertex `i` from `face`.
        int i_index = vertex_to_patch_index_[i];
        if (i_index < 0) {
          is_next_face_valid = false;
          break;
        }

        // Get offset of `current_face` in the patch and set
        // `i_vertex_offsets`.
        auto& i_vertex_offsets = border_offsets[
          patches_[i_index]->GetFaceIndexOfFace(current_face)];
        next_face.push_back(i_index * (N * N) + i_vertex_offsets[0]);
        ++next_face[0];
      }

      if (is_next_face_valid) {
        std::copy(next_face.begin(), next_face.end(), std::back_inserter(T_));
        ++T_[0];
      }
    }
  }

  inline int number_of_vertices() const {
    return control_mesh_.number_of_vertices();
  }

  inline int number_of_faces() const {
    return control_mesh_.number_of_faces();
  }

  inline int number_of_patches() const {
    return static_cast<int>(patch_vertex_indices_.size());
  }

  inline const std::vector<int>& patch_vertex_indices(int p) const {
    return patches_[p]->vertex_indices();
  }

  inline const std::vector<int>& adjacent_patch_indices(int p) const {
    return adjacent_patch_indices_[p];
  }

 private:
  void Initialise() {
    InitialisePatchIndices();
    InitialisePatches();
  }

  void InitialisePatchIndices() {
    // FIXME This is assuming contiguous vertex labelling starting at 0.
    control_mesh_.EnsureVertices();
    auto& vertices = control_mesh_.vertices();
    vertex_to_patch_index_.resize(vertices.size());

    int patch_index = 0;
    for (int i : vertices) {
      if (control_mesh_.is_vertex_closed(i)) {
        assert(control_mesh_.AdjacentVertices(i).size() == 4);
        patch_vertex_indices_.push_back(i);
        vertex_to_patch_index_[i] = patch_index++;
      } else {
        vertex_to_patch_index_[i] = -1;
      }
    }
  }

  void InitialisePatches() {
    std::map<int, std::vector<std::pair<int, int>>> vertex_to_half_edges;
    std::map<std::pair<int, int>, int> half_edge_to_vertex;

    patches_.reserve(patch_vertex_indices_.size());
    for (int i : patch_vertex_indices_) {
      // Initialise `patch`.
      std::vector<int> face_indices = control_mesh_.FacesAtVertex(i);
      auto patch = new Patch(new FaceArray(control_mesh_.Faces(face_indices)));

      // Get the permuted face indices.
      std::vector<int> permuted_face_indices;
      permuted_face_indices.reserve(face_indices.size());
      for (int j : patch->ordered_face_indices()) {
        permuted_face_indices.push_back(face_indices[j]);
      }
      assert(permuted_face_indices.size() == 4);

      // A "half edge" is an ordered pair, where each entry is a face index.
      for (int j = 0; j < 4; ++j) {
        auto half_edge = std::make_pair(permuted_face_indices[j],
                                        permuted_face_indices[(j + 1) % 4]);
        vertex_to_half_edges[i].push_back(half_edge);
        half_edge_to_vertex[half_edge] = i;
      }

      // Save the generated patch.
      patches_.emplace_back(patch);
    }

    // Set `adjacent_patch_indices_`.
    control_mesh_.EnsureVertices();

    int p = 0;
    adjacent_patch_indices_.resize(patches_.size());
    for (int i : patch_vertex_indices_) {
      for (auto& half_edge : vertex_to_half_edges[i]) {
        auto opposite_half_edge = std::make_pair(half_edge.second,
                                                 half_edge.first);
        int adj_patch_index = -1;
        auto it = half_edge_to_vertex.find(opposite_half_edge);
        if (it != half_edge_to_vertex.end()) {
          adj_patch_index = vertex_to_patch_index_[it->second];
        }
        adjacent_patch_indices_[p].push_back(adj_patch_index);
      }
      ++p;
    }
  }

 private:
  GeneralMesh control_mesh_;

  std::vector<int> patch_vertex_indices_;
  std::vector<int> vertex_to_patch_index_;
  std::vector<std::vector<int>> adjacent_patch_indices_;
  std::vector<std::unique_ptr<Patch>> patches_;
};

// SurfaceWalker
template <typename Scalar>
class SurfaceWalker {
  typedef Surface<Scalar> Surface;
  typedef typename Surface::Matrix Matrix;
  typedef typename Surface::Vector Vector;
  typedef typename Surface::Vector2 Vector2;

 public:
  SurfaceWalker(const Surface* surface)
    : surface_(surface) {}

  template <typename TX, typename U, typename Delta, typename U1>
  bool ApplyUpdate(const TX& X, const int p, const U& u, const Delta& delta,
                   int* p1, U1* u1) const {
    int p1_depth = -1;
    // TODO Replace with better strategy which doesn't allocate dynamically.
    std::vector<unsigned char> patch_index_explored(
      surface_->number_of_patches());
    return ApplyUpdateInternal(X, p, u, delta, p1, u1,
                               0, &p1_depth, &patch_index_explored);
  }

 private:
  template <typename TX, typename U, typename Delta, typename U1>
  bool ApplyUpdateInternal(
      const TX& X, const int p, const U& u, const Delta& delta,
      int* p1, U1* u1,
      int depth,
      int* p1_depth,
      std::vector<unsigned char>* patch_index_explored) const {
    assert(X.rows() == 3);

    Scalar t[4];
    int num_intersections = WhichEdgesBroke(u, delta, t);
    if (num_intersections == 0) {
      *p1 = p;
      (*u1)[0] = u[0] + delta[0];
      (*u1)[1] = u[1] + delta[1];
      return true;
    }
    Vector2 _u, _delta;
    _u << u[0], u[1];
    _delta << delta[0], delta[1];

    (*patch_index_explored)[p] = 1;

    assert(num_intersections <= 2);

    int p_edge_index = 0;
    for (int n = 0; n < num_intersections; ++n, ++p_edge_index) {
      while (t[p_edge_index] < 0) {
        ++p_edge_index;
      }

      Vector2 u_1;
      u_1 = _u + t[p_edge_index] * _delta;

      int p1_edge_index, p_1;
      bool has_adjacent_patch = GotoAdjacentPatch(p, p_edge_index,
                                                  &p_1, &p1_edge_index);
      if (has_adjacent_patch) {
        has_adjacent_patch &= 0 == (*patch_index_explored)[p_1];
      }

      if (!has_adjacent_patch) {
        if (depth > *p1_depth) {
          *p1_depth = depth;
          *p1 = p;
          (*u1)[0] = u_1[0];
          (*u1)[1] = u_1[1];
        }
        continue;
      }

      Eigen::Matrix<Scalar, 4, 1> y0, y1;
      y0 << 1 - u_1[0], u_1[1], u_1[0], 1 - u_1[1];
      y1[0] = y0[(p_edge_index + 3) % 4];
      y1[1] = y0[(p_edge_index + 0) % 4];
      y1[2] = y0[(p_edge_index + 1) % 4];
      y1[3] = y0[(p_edge_index + 2) % 4];

      Vector2 u_p1;
      u_p1 << y1[(p1_edge_index + 3) % 4], y1[p1_edge_index];

      Eigen::Matrix<Scalar, 3, 2> M0, M1;
      FillPatchTransformationMatrix(X, p, u_1, &M0);
      FillPatchTransformationMatrix(X, p_1, u_p1, &M1);
      Eigen::Matrix<Scalar, 2, 2> A0 = M1.transpose() * M1,
                                  A0_I = A0.inverse();
      Eigen::Matrix<Scalar, 2, 3> M1_I = A0_I * M1.transpose();
      Eigen::Matrix<Scalar, 2, 2> M = M1_I * M0;
      Vector2 delta_1 = M * _delta;
      delta_1 *= std::max(Scalar(0), 1 - t[p_edge_index]);

      if (ApplyUpdateInternal(X, p_1, u_p1, delta_1, p1, u1,
                              depth + 1, p1_depth, patch_index_explored)) {
        return true;
      }
    }

    return false;
  }

  template <typename U, typename Delta>
  int WhichEdgesBroke(const U& u,
                      const Delta& delta,
                      Scalar* t) const {
    // Note that the transformation to `x` and `delta_x` isn't strictly
    // required; it just made for easier reasoning in initial testing.
    static const Scalar X0_data[8] = {0, 0,
                                      1, 0,
                                      1, 1,
                                      0, 1};
    static const Eigen::Map<const Eigen::Matrix<Scalar, 2, 4>> X0(X0_data);

    Eigen::Vector2d x = u[0] * X0.col(1) + u[1] * X0.col(3);
    Eigen::Vector2d delta_x = delta[0] * X0.col(1) + delta[1] * X0.col(3);

    Eigen::Matrix2d A;
    A.col(0) = delta_x;

    int num_intersections = 0;

    // Fill `t` in reverse order so that `t` corresponds to patch edge
    // ordering and not patch domain edge ordering.
    for (int i = 0; i < 4; ++i) {
      Eigen::Vector2d m = X0.col((i + 1) % 4) - X0.col(i);
      A.col(1) = -m;

      Eigen::Matrix2d A_I = A.inverse();
      Eigen::Vector2d v = A_I * (X0.col(i) - x);
      if (-kUEps <= v[0] && v[0] <= 1.0 + kUEps &&
          -kUEps <= v[1] && v[1] <= 1.0 + kUEps &&
          ((delta_x[0] * m[1] - m[0] * delta_x[1]) >= 0.0)) {
        // `t` is clamped to [0, 1] so that the delta can never be increased
        // in the calling function.
        t[3 - i] = std::max(0.0, std::min(1.0, v[0]));
        ++num_intersections;
      }
      else {
        t[3 - i] = Scalar(-1);
      }
    }

    return num_intersections;
  }

  bool GotoAdjacentPatch(int p, int p_edge_index,
                         int* p1, int* p1_edge_index) const {
    *p1 = surface_->adjacent_patch_indices(p)[p_edge_index];
    if (*p1 < 0) {
      return false;
    }

    auto & adjacent_patches_in_p1 = surface_->adjacent_patch_indices(*p1);
    auto i = std::find(adjacent_patches_in_p1.begin(),
                       adjacent_patches_in_p1.end(),
                       p);
    *p1_edge_index = static_cast<int>(std::distance(
      adjacent_patches_in_p1.begin(), i));
    return true;
  }

  template <typename TX, typename U>
  void FillPatchTransformationMatrix(const TX& X, int p, const U& u,
                                     Eigen::Matrix<Scalar, 3, 2>* M) const {
    auto& patch_vertex_indices = surface_->patch_vertex_indices(p);

    // TODO Replace with better strategy which doesn't allocate dynamically.
    const std::vector<int>::size_type n = patch_vertex_indices.size();
    Matrix Xp(X.rows(), n);
    for (std::vector<int>::size_type i = 0; i < n; ++i) {
      Xp.col(i) = X.col(patch_vertex_indices[i]);
    }

    Eigen::Map<Eigen::Matrix<Scalar, 3, 1>> mu(M->data() + 0),
                                            mv(M->data() + 3);
    surface_->Mu(p, u, Xp, &mu);
    surface_->Mv(p, u, Xp, &mv);
  }

 private:
  const Surface* surface_;
};

} // namespace doosabin

#endif // DOOSABIN_H
