////////////////////////////////////////////
// File: doosabin_test.cpp                //
// Copyright Richard Stebbing 2014.       //
// Distributed under the MIT License.     //
// (See accompany file LICENSE or copy at //
//  http://opensource.org/licenses/MIT)   //
////////////////////////////////////////////

// Includes
#include <algorithm>
#include <iostream>
#include <iterator>
#include <memory>
#include <vector>

#include "Eigen/Dense"

#include "doosabin.h"

// InitialiseVector
std::vector<int> InitialiseVector(const int* T, int size) {
  std::vector<int> v;
  std::copy(T, T + size, std::back_inserter(v));
  return v;
}

// Data.
typedef Eigen::Map<const Eigen::MatrixXd> MapConstMatrixXd;

// Generated by generate_test_doosabin_data.py.
static const double kUData[] = {
  +1.666667e-01, +1.666667e-01,
  +1.666667e-01, +5.000000e-01,
  +1.666667e-01, +8.333333e-01,
  +5.000000e-01, +1.666667e-01,
  +5.000000e-01, +5.000000e-01,
  +5.000000e-01, +8.333333e-01,
  +8.333333e-01, +1.666667e-01,
  +8.333333e-01, +5.000000e-01,
  +8.333333e-01, +8.333333e-01
};
static const MapConstMatrixXd kU(kUData, 2, 9);

static const int kT3Data[] = {
  4, 3, 0, 1, 2, 4, 0, 2, 3, 4, 4, 0, 4, 5, 6, 4, 0, 6, 7, 1
};
static const std::vector<int> kT3 = InitialiseVector(kT3Data, 20);

static const double kX3Data[] = {
  -7.031873e-02, -4.902824e-02, -8.572335e-02,
  -9.981073e-01, +8.331117e-02, -1.001578e+00,
  -2.381526e-01, +7.647994e-01, -8.010211e-01,
  +9.442749e-01, +9.996475e-01, -1.375118e+00,
  +1.154884e+00, -1.370737e-01, -1.162990e+00,
  +1.142529e+00, -1.027946e+00, -1.536895e+00,
  -5.596279e-02, -8.813617e-01, -8.831366e-01,
  -7.960739e-01, -1.169122e+00, -1.414419e+00
};
static const MapConstMatrixXd kX3(kX3Data, 3, 8);

static const double kM3Data[] = {
  -3.226513e-01, +1.710868e-01, -5.097319e-01,
  -4.968058e-02, +2.034774e-01, -5.068752e-01,
  +2.836774e-01, +2.276860e-01, -6.620245e-01,
  -3.434957e-01, -6.162111e-02, -5.252902e-01,
  -2.765453e-02, -6.795005e-02, -4.694818e-01,
  +3.228742e-01, -7.152057e-02, -6.169194e-01,
  -3.455167e-01, -3.370071e-01, -6.498679e-01,
  -2.104001e-02, -3.435969e-01, -5.819555e-01,
  +3.426268e-01, -3.632328e-01, -7.035485e-01
};
static const MapConstMatrixXd kM3(kM3Data, 3, 9);

static const double kMu3Data[] = {
  -2.567376e-02, -6.903217e-01, +2.137669e-01,
  +1.196294e-01, -8.342359e-01, +3.822910e-01,
  +1.501383e-01, -9.117816e-01, +3.379506e-01,
  -4.866974e-02, -7.497296e-01, -2.316003e-01,
  +1.252690e-02, -7.943291e-01, -1.579302e-01,
  +8.504253e-02, -8.834581e-01, -6.732047e-02,
  +3.654390e-02, -9.025862e-01, -5.158661e-01,
  +2.716018e-02, -8.595519e-01, -5.169122e-01,
  +3.347293e-02, -8.668155e-01, -4.524539e-01
};
static const MapConstMatrixXd kMu3(kMu3Data, 3, 9);

static const double kMuu3Data[] = {
  -6.776670e-01, +3.474245e-01, -2.242298e+00,
  -3.213076e-01, +1.197205e-01, -1.620663e+00,
  -1.952874e-01, +8.497075e-02, -1.215813e+00,
  +1.339051e-01, -3.534399e-01, -1.034036e+00,
  -3.213076e-01, +1.197205e-01, -1.620663e+00,
  -1.952874e-01, +8.497075e-02, -1.215813e+00,
  +2.556409e-01, -4.585696e-01, -8.527972e-01,
  +4.389983e-02, -1.956685e-01, -1.076946e+00,
  -1.547088e-01, +4.992752e-02, -1.155400e+00
};
static const MapConstMatrixXd kMuu3(kMuu3Data, 3, 9);

static const int kT4Data[] = {
  4, 4, 0, 1, 2, 3, 4, 0, 3, 4, 5, 4, 0, 5, 6, 7, 4, 0, 7, 8, 1
};
static const std::vector<int> kT4 = InitialiseVector(kT4Data, 21);

static const double kX4Data[] = {
  -6.995228e-02, +5.829628e-02, -9.105921e-02,
  -9.021777e-01, -1.217372e-01, -9.103541e-01,
  -1.132940e+00, +9.998545e-01, -1.511046e+00,
  -1.314653e-01, +9.620388e-01, -9.709798e-01,
  +1.126521e+00, +1.012067e+00, -1.514374e+00,
  +1.014794e+00, -2.753726e-01, -1.051493e+00,
  +9.643104e-01, -9.992282e-01, -1.388651e+00,
  +1.478277e-01, -1.095761e+00, -1.105688e+00,
  -8.670992e-01, -1.098585e+00, -1.399553e+00
};
static const MapConstMatrixXd kX4(kX4Data, 3, 9);

static const double kM4Data[] = {
  -3.826901e-01, +3.179739e-01, -6.701556e-01,
  -5.733081e-02, +3.190177e-01, -6.008370e-01,
  +2.955444e-01, +2.871287e-01, -7.005476e-01,
  -3.397607e-01, -2.157935e-02, -5.888701e-01,
  -2.583775e-02, -1.831594e-02, -5.206695e-01,
  +3.099352e-01, -5.533949e-02, -6.238391e-01,
  -2.931934e-01, -3.724479e-01, -6.853571e-01,
  +1.681033e-02, -3.666713e-01, -6.246301e-01,
  +3.374798e-01, -3.936104e-01, -7.141680e-01
};
static const MapConstMatrixXd kM4(kM4Data, 3, 9);

static const double kMu4Data[] = {
  +1.233309e-01, -1.001687e+00, +5.105153e-01,
  +7.774664e-02, -9.954684e-01, +5.166949e-01,
  +2.344166e-02, -1.033700e+00, +4.806812e-01,
  +1.342450e-01, -1.035633e+00, -2.280234e-02,
  +1.112117e-01, -1.028534e+00, -3.568961e-02,
  +6.290313e-02, -1.021109e+00, -2.043063e-02,
  +1.451591e-01, -1.069579e+00, -5.561200e-01,
  +1.446768e-01, -1.061599e+00, -5.880741e-01,
  +1.023646e-01, -1.008517e+00, -5.215424e-01
};
static const MapConstMatrixXd kMu4(kMu4Data, 3, 9);

static const double kMuu4Data[] = {
  +3.274228e-02, -1.018377e-01, -1.599953e+00,
  +1.003952e-01, -9.919544e-02, -1.657154e+00,
  +1.183844e-01, +3.777552e-02, -1.503335e+00,
  +3.274228e-02, -1.018377e-01, -1.599953e+00,
  +1.003952e-01, -9.919544e-02, -1.657154e+00,
  +1.183844e-01, +3.777552e-02, -1.503335e+00,
  +3.274228e-02, -1.018377e-01, -1.599953e+00,
  +1.003952e-01, -9.919544e-02, -1.657154e+00,
  +1.183844e-01, +3.777552e-02, -1.503335e+00
};
static const MapConstMatrixXd kMuu4(kMuu4Data, 3, 9);

static const int kT5Data[] = {
  4, 5, 0, 1, 2, 3, 4, 4, 0, 4, 5, 6, 4, 0, 6, 7, 8, 4, 0, 8, 9, 1
};
static const std::vector<int> kT5 = InitialiseVector(kT5Data, 22);

static const double kX5Data[] = {
  +4.715572e-02, -8.746529e-04, -4.716383e-02,
  -9.509865e-01, -4.457899e-02, -9.520308e-01,
  -1.442518e+00, +7.812036e-01, -1.640468e+00,
  -6.082452e-01, +1.351513e+00, -1.482076e+00,
  +5.355020e-02, +1.044656e+00, -1.046028e+00,
  +1.064166e+00, +1.013670e+00, -1.469686e+00,
  +1.119318e+00, -1.046300e-01, -1.124197e+00,
  +1.013989e+00, -1.173066e+00, -1.550566e+00,
  -1.306231e-02, -1.131026e+00, -1.131101e+00,
  -1.204820e+00, -1.106618e+00, -1.635908e+00
};
static const MapConstMatrixXd kX5(kX5Data, 3, 10);

static const double kM5Data[] = {
  -3.708055e-01, +4.144311e-01, -7.844953e-01,
  +2.120288e-02, +3.646971e-01, -6.497883e-01,
  +3.955574e-01, +3.226096e-01, -7.161401e-01,
  -3.360018e-01, +9.684813e-03, -6.429113e-01,
  +2.506529e-02, -7.153237e-03, -5.479422e-01,
  +3.916642e-01, -3.995616e-02, -6.386168e-01,
  -3.329400e-01, -3.830193e-01, -7.178995e-01,
  +2.290607e-02, -3.892643e-01, -6.402617e-01,
  +3.778357e-01, -4.070386e-01, -7.398227e-01
};
static const MapConstMatrixXd kM5(kM5Data, 3, 9);

static const double kMu5Data[] = {
  +7.796229e-02, -1.162751e+00, +6.466255e-01,
  -1.213844e-02, -1.068564e+00, +5.506475e-01,
  -4.166563e-04, -1.077411e+00, +4.955373e-01,
  +7.781405e-02, -1.214347e+00, +1.305001e-01,
  +3.531290e-02, -1.162538e+00, +6.042913e-02,
  -2.294275e-02, -1.097983e+00, -3.039733e-02,
  -5.944334e-02, -1.141878e+00, -5.804298e-01,
  -4.826822e-02, -1.130128e+00, -6.143463e-01,
  -6.002801e-02, -1.104512e+00, -5.768383e-01
};
static const MapConstMatrixXd kMu5(kMu5Data, 3, 9);

static const double kMuu5Data[] = {
  +6.361083e-01, -7.713502e-01, -6.798297e-01,
  +1.423540e-01, -2.819238e-01, -1.470655e+00,
  -6.757828e-02, -6.171492e-02, -1.577804e+00,
  -2.126291e-01, +5.073298e-02, -1.837892e+00,
  +1.423540e-01, -2.819238e-01, -1.470655e+00,
  -6.757828e-02, -6.171492e-02, -1.577804e+00,
  -4.117722e-01, +2.174050e-01, -2.132790e+00,
  -2.507434e-01, +9.723105e-02, -2.024326e+00,
  -1.112558e-01, -1.958661e-02, -1.639323e+00
};
static const MapConstMatrixXd kMuu5(kMuu5Data, 3, 9);

static const int kT6Data[] = {
  4, 6, 0, 1, 2, 3, 4, 5, 4, 0, 5, 6, 7, 4, 0, 7, 8, 9, 4, 0, 9, 10, 1
};
static const std::vector<int> kT6 = InitialiseVector(kT6Data, 23);

static const double kX6Data[] = {
  -3.316184e-03, +1.466396e-01, -1.466771e-01,
  -8.782615e-01, -1.918201e-01, -8.989651e-01,
  -1.603288e+00, +4.230092e-01, -1.658152e+00,
  -1.357415e+00, +1.444652e+00, -1.982321e+00,
  -5.482907e-01, +1.427391e+00, -1.529074e+00,
  +2.840497e-01, +1.042238e+00, -1.080252e+00,
  +8.416545e-01, +9.857301e-01, -1.296166e+00,
  +1.008525e+00, +6.405078e-02, -1.010557e+00,
  +9.209342e-01, -9.228818e-01, -1.303775e+00,
  -1.950678e-01, -1.129401e+00, -1.146123e+00,
  -1.073278e+00, -9.969309e-01, -1.464854e+00
};
static const MapConstMatrixXd kX6(kX6Data, 3, 11);

static const double kM6Data[] = {
  -4.282846e-01, +4.938261e-01, -9.224131e-01,
  +9.856726e-04, +4.535907e-01, -7.260148e-01,
  +3.686767e-01, +4.182384e-01, -7.212215e-01,
  -3.707573e-01, +6.847515e-02, -7.140361e-01,
  -2.632367e-02, +9.309027e-02, -6.066802e-01,
  +3.274452e-01, +8.402156e-02, -6.441039e-01,
  -3.655088e-01, -3.370305e-01, -7.322687e-01,
  -5.037512e-02, -2.997721e-01, -6.630204e-01,
  +2.830533e-01, -2.796919e-01, -7.272660e-01
};
static const MapConstMatrixXd kM6(kM6Data, 3, 9);

static const double kMu6Data[] = {
  +1.283233e-01, -1.200172e+00, +7.860401e-01,
  -1.451944e-01, -9.856397e-01, +5.410521e-01,
  -1.254405e-01, -9.531479e-01, +4.628320e-01,
  +1.193375e-01, -1.270521e+00, +3.375017e-01,
  -1.866168e-02, -1.177363e+00, +1.749556e-01,
  -1.219484e-01, -1.052153e+00, -1.263124e-04,
  -8.784660e-02, -1.162512e+00, -4.468972e-01,
  -1.256470e-01, -1.179812e+00, -5.129967e-01,
  -1.444029e-01, -1.130128e+00, -4.988464e-01
};
static const MapConstMatrixXd kMu6(kMu6Data, 3, 9);

static const double kMuu6Data[] = {
  +1.143077e+00, -1.187998e+00, +1.750282e-01,
  +3.795981e-01, -5.751694e-01, -1.098290e+00,
  +1.047607e-02, -2.970156e-01, -1.388875e+00,
  -4.169690e-01, +1.146019e-01, -1.852496e+00,
  +3.795981e-01, -5.751694e-01, -1.098290e+00,
  +1.047607e-02, -2.970156e-01, -1.388875e+00,
  -6.215522e-01, +3.240269e-01, -2.353197e+00,
  -3.209560e-01, -7.346010e-03, -2.063857e+00,
  -6.736327e-02, -2.339241e-01, -1.496160e+00
};
static const MapConstMatrixXd kMuu6(kMuu6Data, 3, 9);

// end generated by generate_test_doosabin_data.py.

// For `Patch` and `Surface`.
static const int kNumPatches = 4;
static const MapConstMatrixXd* kXs[kNumPatches] = {&kX3, &kX4, &kX5, &kX6};
static const MapConstMatrixXd* kMs[kNumPatches] = {&kM3, &kM4, &kM5, &kM6};
static const MapConstMatrixXd* kMus[kNumPatches] = {&kMu3, &kMu4, &kMu5, &kMu6};
static const MapConstMatrixXd* kMuus[kNumPatches] = {&kMuu3, &kMuu4, &kMuu5, &kMuu6};

// For `SurfaceWalker`.
static const int kTData[] = {8,
                             4, 0, 1, 5, 4,
                             3, 3, 8, 7,
                             4, 6, 7, 12, 11,
                             4, 1, 2, 6, 5,
                             4, 2, 3, 7, 6,
                             4, 4, 5, 10, 9,
                             4, 5, 6, 11, 10,
                             4, 7, 8, 13, 12};
static const std::vector<int> kT = InitialiseVector(
  kTData, sizeof(kTData) / sizeof(kTData[0]));
static const double kXData[] = {0, 2, 0,
                                1, 2, 0,
                                2, 2, 0,
                                3, 2, 0,
                                0, 1, 0,
                                1, 1, 0,
                                2, 1, 0,
                                3, 1, 0,
                                4, 1, 0,
                                0, 0, 0,
                                1, 0, 0,
                                2, 0, 0,
                                3, 0, 0,
                                4, 0, 0};
static const Eigen::Map<const Eigen::MatrixXd> kX(
  kXData, 3, sizeof(kXData) / (3 * sizeof(kXData[0])));

static const double kU0[] = {0.1, 0.2};
static const int kP0 = 0;
static const double kDelta[] = {-0.2,  0.0,
                                 0.0, -0.3,
                                 0.5,  0.5,
                                 1.8,  0.8,
                                 0.0,  1.0,
                                 0.1,  1.0,
                                -0.2,  1.0,
                                -0.2,  2.0,
                                 0.5,  2.0};
static const int kP1[] = {0, 0, 0, 0, 1, 1, 0, 1, 2};
static const double kU1[] = {0.0, 0.2,
                             0.1, 0.0,
                             0.6, 0.7,
                             1.0, 0.6,
                             0.9, 0.8,
                             0.8, 0.8,
                             0.0, 0.7,
                             1.0, 0.8,
                             0.8, 0.6};
static const int kNumDeltas = sizeof(kDelta) / (2 * sizeof(kDelta[0]));

// main
int main() {
  // `Patch`.
  typedef doosabin::Patch<double> Patch;

  Patch patches[kNumPatches] = {Patch(new doosabin::FaceArray(kT3)),
                                Patch(new doosabin::FaceArray(kT4)),
                                Patch(new doosabin::FaceArray(kT5)),
                                Patch(new doosabin::FaceArray(kT6))};

  #define TEST_EVALUATION(M, MS) \
  for (int i = 0; i < kNumPatches; ++i) { \
    for (MapConstMatrixXd::Index j = 0; j < kU.cols(); ++j) { \
      Eigen::Vector3d r; \
      patches[i].M(kU.col(j), *kXs[i], &r); \
      if (!r.isApprox(MS[i]->col(j), 1e-5)) { \
        std::cerr << #M << " [" << i << ", " << j << "]: " \
                  << r.transpose() << " != " << MS[i]->col(j).transpose() \
                  << std::endl; \
        return 1; \
      } \
    } \
  }
  TEST_EVALUATION(M, kMs);
  TEST_EVALUATION(Mu, kMus);
  TEST_EVALUATION(Muu, kMuus);
  #undef TEST_EVALUATION

  // `Surface`.
  typedef doosabin::Surface<double> Surface;

  Surface surfaces[kNumPatches] = {Surface(doosabin::GeneralMesh(kT3)),
                                   Surface(doosabin::GeneralMesh(kT4)),
                                   Surface(doosabin::GeneralMesh(kT5)),
                                   Surface(doosabin::GeneralMesh(kT6))};

  #define TEST_EVALUATION(M, MS) \
  for (int i = 0; i < kNumPatches; ++i) { \
    for (MapConstMatrixXd::Index j = 0; j < kU.cols(); ++j) { \
      Eigen::Vector3d r; \
      surfaces[i].M(0, kU.col(j), *kXs[i], &r); \
      if (!r.isApprox(MS[i]->col(j), 1e-5)) { \
        std::cerr << #M << " [" << i << ", " << j << "]: " \
                  << r.transpose() << " != " << MS[i]->col(j).transpose() \
                  << std::endl; \
        return 1; \
      } \
    } \
  }
  TEST_EVALUATION(M, kMs);
  TEST_EVALUATION(Mu, kMus);
  TEST_EVALUATION(Muu, kMuus);
  #undef TEST_EVALUATION

  // `SurfaceWalker`.
  doosabin::GeneralMesh T(kT);
  Surface surface(T);

  typedef doosabin::SurfaceWalker<double> SurfaceWalker;
  SurfaceWalker walker(&surface);

  for (int i = 0; i < kNumDeltas; ++i) {
    int p1;
    Eigen::Vector2d u1;
    walker.ApplyUpdate(kX, kP0, kU0, kDelta + 2 * i, &p1, &u1);

    const Eigen::Map<const Eigen::Vector2d> ku1(kU1 + 2 * i);
    if (p1 != kP1[i] || !u1.isApprox(ku1, 1e-5)) {
      std::cerr << i << ": p1, u1 != " << kP1[i] << ", " << ku1.transpose()
                << " (= " << p1 << ", " << u1.transpose() << ")" << std::endl;
      return 1;
    }
  }

  // Example usage of `MatrixOfColumnPointers`.
  const double* kX3PointersData[] = {&kX3Data[0],
                                     &kX3Data[3],
                                     &kX3Data[6],
                                     &kX3Data[9],
                                     &kX3Data[12],
                                     &kX3Data[15],
                                     &kX3Data[18],
                                     &kX3Data[21]};
  doosabin::MatrixOfColumnPointers<double> kX3Pointers(kX3PointersData, 3, 8);
  Eigen::Vector3d r;
  patches[0].M(kU.col(0), kX3Pointers, &r);
  if (!r.isApprox(kMs[0]->col(0), 1e-5)) {
    std::cerr << "M [0, 0]: " << r.transpose() << " != "
              << kMs[0]->col(0).transpose() << std::endl;
    return 1;
  }

  return 0;
}
