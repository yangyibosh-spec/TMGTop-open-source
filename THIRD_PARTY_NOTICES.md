# Third-Party Notices

TMGTop is maintained by:

Yibo Yang  
School of Chemical Engineering, Dalian University of Technology  
Dalian 116024

## Method of Moving Asymptotes

The implementation in `mma.cpp` and `mma.h` is a C++ adaptation of the
MATLAB GCMMA-MMA-code developed by Krister Svanberg.

Original work:

Copyright (C) 2006-2009 Krister Svanberg

Original distribution:

<https://www.smoptit.se/>

License:

GNU General Public License, version 3 or (at your option) any later version
(`GPL-3.0-or-later`).

Changes in this repository:

- translated and adapted the MMA implementation from MATLAB to C++;
- integrated the implementation with the TMGTop transient thermal topology
  optimization solver;
- added scalar-dual and numerical-stability handling used by the Case-2
  implementation; and
- modified data structures and interfaces for Eigen and the surrounding
  MPI/CUDA solver.

These adaptations and subsequent modifications were made in 2026 by Yibo Yang.

Relevant publication:

K. Svanberg, "The Method of Moving Asymptotes--A New Method for Structural
Optimization," *International Journal for Numerical Methods in Engineering*,
vol. 24, no. 2, pp. 359--373, 1987.
<https://doi.org/10.1002/nme.1620240207>

The copyright and license notices above must be retained when the corresponding
source files or derived works are redistributed.
