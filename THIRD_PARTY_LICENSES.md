# Third-party licenses and notices

This repository is distributed under the MIT License (see `LICENSE.txt`), **except** for the
third-party component listed below, which keeps its own license. Redistribution of the
repository (source form) and of the compiled `voronoi3d_engine.dll` (binary form) must
include the notice that follows.

## Voro++ (bundled, `engine/voro/`)

* Project: Voro++, a three-dimensional Voronoi cell library in C++
* Author: Chris H. Rycroft (Harvard University / Lawrence Berkeley National Laboratory)
* Web site: <https://math.lbl.gov/voro++/>
* Reference: C. H. Rycroft, "Voro++: A three-dimensional Voronoi cell library in C++",
  *Chaos* **19**, 041111 (2009), doi:10.1063/1.3215722
* License: modified BSD (three clauses plus the LBNL "Enhancements" grant). Full text:

```text
Voro++ Copyright (c) 2008, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from the U.S. Dept. of Energy). All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met: 

(1) Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer. 

(2) Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution. 

(3) Neither the name of the University of California, Lawrence Berkeley
National Laboratory, U.S. Dept. of Energy nor the names of its contributors may
be used to endorse or promote products derived from this software without
specific prior written permission. 

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 

You are under no obligation whatsoever to provide any bug fixes, patches, or
upgrades to the features, functionality or performance of the source code
("Enhancements") to anyone; however, if you choose to make your Enhancements
available either publicly, or directly to Lawrence Berkeley National
Laboratory, without imposing a separate written license agreement for such
Enhancements, then you hereby grant the following license: a  non-exclusive,
royalty-free perpetual license to install, use, modify, prepare derivative
works, incorporate into other computer software, distribute, and sublicense
such enhancements or derivative works thereof, in binary and source code form.
```

### Modifications made to Voro++

The Voro++ sources were modified in two files only, `engine/voro/cell.cc` and
`engine/voro/common.cc`: the calls to `exit()` in the fatal-error paths were replaced by
`throw std::runtime_error(...)`, so that a host application (ParaView) is not terminated
by a numerical failure; the exception is caught by the engine wrapper and reported to the
user. Both files carry a "MODIFIED" notice at the top. These changes are made available
under the "Enhancements" clause of the Voro++ license. The name of the University of
California, Lawrence Berkeley National Laboratory or the U.S. Dept. of Energy is not used
to endorse or promote this plugin.

## Components that are used but NOT distributed in this repository

These are loaded at run time from the user's own installation and are covered by their
own licenses; check the license of the version you use.

| Component | Role | License (as published by the projects) |
|---|---|---|
| ParaView / VTK | host application, `vtkmodules`, `paraview.util.vtkAlgorithm` | BSD 3-Clause |
| NumPy | array computations (shipped with ParaView) | BSD 3-Clause |
| SciPy (optional) | alternative Voronoi engine (Qhull) | BSD 3-Clause (Qhull: its own permissive license) |
| `tess` (optional) | alternative Voro++ binding | see the package |
| Microsoft Visual C++ | compiler used to build the DLL (`/MT` static runtime) | Microsoft license terms for the redistributable runtime |
