#ifndef UNIFORM_QUADRATIC_BSPLINE_H
#define UNIFORM_QUADRATIC_BSPLINE_H

// uniform_quadratic_bspline
namespace uniform_quadratic_bspline {

// Position
struct Position {
  template <typename T>
  inline T operator()(const T& u, int k) const {
    switch (k) {
      case 0:
        return T(0.5) * (T(1) - u) * (T(1) - u);

      case 1:
        return -u * u + u + T(0.5);

      case 2:
        return T(0.5) * (u * u);

      default:
        break;
      };

    return T(0);
  }
};

// FirstDerivative
struct FirstDerivative {
  template <typename T>
  inline T operator()(const T& u, int k) const {
    switch (k) {
      case 0:
        return T(-1) + u;

      case 1:
        return T(-2) * u + 1;

      case 2:
        return u;

      default:
        break;
      };

    return T(0);
  }
};

// SecondDerivative
struct SecondDerivative {
  template <typename T>
  inline T operator()(const T&, int k) const {
    switch (k) {
      case 0:
        return T(1);

      case 1:
        return T(-2);

      case 2:
        return T(1);

      default:
        break;
      };

    return T(0);
  }
};

} // namespace uniform_quadratic_bspline

#endif // UNIFORM_QUADRATIC_BSPLINE_H
