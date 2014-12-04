subdivision
===========

![Doo-Sabin subdivision](https://github.com/rstebbing/subdivision/raw/master/README.png)

[Uniform quadratic B-splines][2] define smooth curves which are parameterised by a finite number of control vertices.
[Biquadratic B-splines][3] and [quadratic triangle B-splines][1] are generalisations which define surfaces, where the influence of each control vertex is specified by its index in a _regular_ quadrilateral or triangle mesh.
For general meshes containing _irregularities_ (e.g. non-quadrilateral faces, vertices with valency not equal to four/six) biquadratic and triangle B-splines cannot be applied directly.
Instead, _subdivision_ is necessary to refine the control mesh to "uncover" regular _patches_.
To this end, the [Doo-Sabin][5] algorithm generalises biquadratic B-splines to general meshes, and the [Loop][4] subdivision algorithm generalises quadratic triangle B-splines to irregular triangle meshes.

Unlike their regular counterparts, evaluating positions and derivatives on Loop and Doo-Sabin subdivisions surfaces is more involved.
The purpose of this repository is to provide the code necessary to (a) understand subdivision surface evaluation, (b) enable straightforward adoption and evaluation of subdivision surfaces, and (c) verify properties of the first and second derivatives for each surface type.

The code for fast evaluation of points and derivatives for Loop subdivision surfaces is not currently available but will be added in due course.

Author: Richard Stebbing

License: MIT (refer to LICENSE).

Dependencies
------------
Core:
* [Eigen 3][6]
* [rstebbing/common][7]
* Numpy

Visualisation:
* matplotlib
* VTK

Doo-Sabin Subdivision Surfaces
------------------------------
### Usage
#### C++
The `Surface` class in [doosabin.h](cpp/doosabin/include/doosabin.h) implements the methods for evaluating positions and derivatives of points on a Doo-Sabin subdivision surface.

To build the test under [cpp/doosabin](cpp/doosabin):

1. Run CMake with an out of source build.
2. Set `EIGEN_INCLUDE_DIR` to the full path up to and including eigen3/.
3. Set `COMMON_CPP_INCLUDE_DIR` to the full path to [rstebbing/common/cpp](https://github.com/rstebbing/common/tree/master/cpp).
(Add `-std=c++11` to `CMAKE_CXX_FLAGS` if compiling with gcc.)
4. Configure.
5. Build.

#### Python
To build the Cython extension module and install `subdivision` as a package:

1. Set `EIGEN_INCLUDE` and `COMMON_CPP_INCLUDE` in site.cfg.
2. Build the Python package: `python setup.py build`.
(Use `export CFLAGS=-std=c++11` beforehand if compiling with gcc.)
3. Install: `python setup.py install`.

To visualise the Doo-Sabin surface defined by a subdivided cube, from the [examples](examples) directory and with [rstebbing/common](https://github.com/rstebbing/common/tree/master) installed:
```
python doosabin/visualise_subdivision.py cube
```

More generally, to evaluate points uniformly distributed across a surface defined by the mesh `T` and matrix of control vertices `X`:
```
>>> import doosabin
>>> surface = doosabin.surface(T)
>>> pd, Ud, Td = surface.uniform_parameterisation(sample_density)
>>> M = surface.M(pd, Ud, X)
```
where `sample_density` is a positive integer controlling the number of evaluated points and `M` is the matrix of evaluated points.

### Verification
From this directory:

- Evaluate points on an extraordinary patch containing a face with `6` sides and show `2` applications of Doo-Sabin subdivision:
  ```
  python verification/doosabin/evaluation.py 6 2
  ```

- Show the magnitudes of the first and second derivative weight vectors (evaluated numerically) as the position being evaluated tends to the origin:
  ```
  python verification/doosabin/derivatives_numeric.py 6 16
  ```

- Evaluate the same vectors symbolically:
  ```
  python verification/doosabin/derivatives_symbolic.py 6 16
  ```

Loop Subdivision Surfaces
-------------------------
### Verification
From this directory:
- Evaluate points on an extraordinary patch containing a vertex with valency `5` and show `2` applications of Loop subdivision:
  ```
  python verification/loop/evaluation.py 5 2
  ```

- Show the magnitudes of the first and second derivative weight vectors (evaluated numerically) as the position being evaluated tends to the origin:
  ```
  python verification/loop/derivatives_numeric.py 5 16
  ```

[1]: http://www.cs.cmu.edu/afs/cs/user/glmiller/public/computational-geometry/15-852-F12/RelatedWork/Loop-by-Stam.pdf
[2]: http://graphics.cs.ucdavis.edu/education/CAGDNotes/CAGDNotes/Quadratic-B-Spline-Curve-Refinement/Quadratic-B-Spline-Curve-Refinement.html
[3]: http://graphics.cs.ucdavis.edu/education/CAGDNotes/CAGDNotes/Quadratic-B-Spline-Surface-Refinement/Quadratic-B-Spline-Surface-Refinement.html
[4]: http://research.microsoft.com/en-us/um/people/cloop/thesis.pdf
[5]: http://www.cs.caltech.edu/~cs175/cs175-02/resources/DS.pdf
[6]: http://eigen.tuxfamily.org
[7]: http://github.com/rstebbing/common
