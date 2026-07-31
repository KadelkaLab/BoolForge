# BoolForge

**BoolForge** is a Python toolbox for generating, sampling, and analyzing
Boolean functions and Boolean networks, with a particular emphasis on
**canalization** and the uniform random generation of functions with prescribed structure.

While many existing tools focus on simulation and dynamical analysis,
BoolForge emphasizes **controlled generation and analysis**
of Boolean functions and networks, enabling systematic studies of
canalization, robustness, and ensemble properties.

The package provides tools for:

- random sampling of Boolean functions with prescribed canalizing structure,
- generation of Boolean networks with controlled update rules and wiring diagrams,
- analysis of canalization, activity, sensitivity, and related structural measures,
- interoperability with other Boolean network software and model formats.

BoolForge is designed for researchers working with regulatory networks,
discrete dynamical systems, and random Boolean network ensembles in
systems biology and network science.

---

## Installation

### Stable release (recommended)

Install the latest stable version from PyPI:

```bash
pip install boolforge
```

BoolForge requires **Python 3.10 or later**.

---

### Development version

To install the latest development version directly from GitHub:

```bash
pip install git+https://github.com/ckadelka/BoolForge
```

---

## Optional dependencies (extended functionality)

BoolForge is fully usable with its core dependencies, but some features rely on
optional packages that can be installed via *extras*.

### Performance acceleration

Some internal routines are automatically accelerated if
[numba](https://numba.pydata.org/) is available.
Exact asynchronous attractor computation requires numba.

To enable numba acceleration:

```bash
pip install boolforge[speed]
```

When numba is not installed, BoolForge transparently falls back to
pure-Python implementations.

---

### Plotting and visualization

Plotting of wiring diagrams and network structure requires
[matplotlib](https://matplotlib.org/).

To enable plotting:

```bash
pip install boolforge[plot]
```

---

### CANA integration

Some methods interface with the
[CANA](https://github.com/CASCI-lab/CANA) package for advanced canalization
measures.

To enable CANA-based functionality:

```bash
pip install boolforge[cana]
```

---

### Symbolic logic and expression minimization

Symbolic representations and logical expression minimization rely on
[PyEDA](https://pyeda.readthedocs.io/).

To enable symbolic functionality:

```bash
pip install boolforge[symbolic]
```

---

### Biological model retrieval

The retrival and loading of hundreds of published biological Boolean
network models relies on the
[requests](https://requests.readthedocs.io/en/latest/) package for web access.

To enable biological model retrieval:

```bash
pip install boolforge[bio]
```

---

### All optional features

To install BoolForge with **all optional dependencies**:

```bash
pip install boolforge[all]
```

---

## Compatibility and interoperability

BoolForge supports import and export of Boolean network representations used by
other software packages.

In particular, BoolForge supports the **.bnet format** commonly used by
[pyboolnet](https://github.com/hklarner/pyboolnet), without requiring pyboolnet
itself to be installed.

BoolForge also supports conversion to and from the format used by
 [CANA](https://github.com/CASCI-lab/CANA).
 
---

## Documentation

Full documentation, including tutorials and API reference, is available at:

https://kadelkalab.github.io/BoolForge/

---

## Citation

If you use BoolForge in your research, please cite the accompanying
application note:

Kadelka, C., & Coberly, B. (2025).  
*BoolForge: Controlled Generation and Analysis of Boolean Functions and Networks*.  
arXiv:2509.02496.  
https://arxiv.org/abs/2509.02496

A machine-readable citation file (`CITATION.cff`) is included in the repository
and can be used directly by GitHub, Zenodo, and reference managers.

---

## License

BoolForge is released under the MIT License.
