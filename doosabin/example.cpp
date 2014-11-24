// example.cpp

// Includes
#include <iostream>

#include "doosabin.h"

// main
int main() {
  const int kCellArray[] = {4, 4, 2, 3, 4, 1,
                               4, 1, 4, 5, 6,
                               4, 6, 7, 8, 1,
                               4, 9, 2, 1, 8};
  std::vector<int> cell_array;
  std::copy(std::begin(kCellArray), std::end(kCellArray),
            std::back_inserter(cell_array));

  doosabin::FaceArray face_array(std::move(cell_array));
  std::cout << face_array << std::endl;

  // doosabin::InternalPatch<double> patch(nullptr, 0, std::move(face_array));
  doosabin::Patch<double> patch(std::move(face_array));
  patch.SubdivideChildren();

  Eigen::MatrixXd X(9, 2);
  X << 0, 0,
       1, 0,
       1, -1,
       0, -1,
       -1, -1,
       -1, 0,
       -1, 1,
       0, 1,
       1, 1;
  std::cout << X << std::endl;

  patch.SetVertices(X.transpose());

  Eigen::MatrixXd U(3, 2);
  U << 0, 0,
       1, 0,
       0, 1;
  Eigen::Vector2d r;

  for (int i = 0; i < 3; ++i) {
    Eigen::Vector2d u = U.row(i);
    patch.EvaluatePosition(u, r);
    std::cout << "u = " << u.transpose();
    std::cout << ", r = " << r.transpose() << std::endl;
  }

  return 0;
}
