# SECSpy

SECSpy is a Python toolkit for modelling ionospheric equivalent currents with
Spherical Elementary Current Systems (SECS). It supports divergence-free and
curl-free SECS representations, magnetic-field forward modelling, amplitude
inversion from ground magnetometer data, and helper routines for grid checking
and plotting.

The core equations follow:

- Vanhamaki, H., & Juusola, L. (2020), *Introduction to Spherical Elementary
  Current Systems*, https://doi.org/10.1007/978-3-030-26732-2_2
- Marsal, S., Torta, J. M., Segarra, A., & Araki, T. (2016), Use of spherical 
  elementary currents to map the polar current systems associated with the 
  geomagnetic sudden commencements on 2013 and 2015 St. Patrick's Day storms. 
  Journal of Geophysical Research: Space Physics, 122(1), 194–211. 
  https://doi.org/10.1002/2016JA023166
- Juusola, L., Kauristie, K., Vanhamäki, H., Aikio, A., & van de Kamp, M. (2016), 
  Comparison of auroral ionospheric and field-aligned currents derived from Swarm 
  and ground magnetic field measurements. Journal of Geophysical Research: Space 
  Physics, 122(1), 9729–9748. https://doi.org/10.1002/2016JA022961

## Repository Layout

```text
src/SECSpy/
  SECS.py             Core SECS model, G matrices, inversions, fields, currents
  Grid_tools.py       Grid quality checks and grid adjustment helpers
  Plotting_Tools.py   Matplotlib plotting utilities
  Example.py          Small built-in example grid and magnetometer data
  utils.py            Animation, date, and small workflow helpers

notebooks/
  Vanhamaki(2020)_example.ipynb

development_code/EZIE/
  Experimental EZIE workflow scripts
```

## Installation

SECSpy can be installed directly from a local checkout:

```bash
git clone https://github.com/08walkersj/SECSpy.git
cd SECSpy
python -m pip install .
```

For development, install it in editable mode:

```bash
python -m pip install -e .
```

You can also install it directly from Git:

```bash
python -m pip install "git+https://github.com/08walkersj/SECSpy.git"
```

The core installation includes `numpy` and `scipy`. Optional dependency groups
are available for common workflows:

```bash
python -m pip install ".[data]"
python -m pip install ".[plotting]"
python -m pip install ".[video]"
python -m pip install ".[ezie]"
python -m pip install ".[all]"
```

The `ezie` extra now includes `secsy` and `polplot` for experimental EZIE
workflow support. `chaosmagpy` has been removed from optional dependencies as it
is not used by the package source.

Some development and EZIE scripts also use packages such as `xarray`,
`polplot`, and other workflow-specific tools. Those scripts should be treated as
workflow examples rather than the minimal package interface.

## Quick Start

```python
from SECSpy import SECS, ExampleData, ExampleGrid

# Example SEC pole grid and evaluation grid, all in geographic degrees.
node_lons, node_lats, eval_lons, eval_lats = ExampleGrid()

# Example magnetometer measurements in Tesla.
south, east, up, mag_lons, mag_lats = ExampleData()

model = SECS(
    node_lons,
    node_lats,
    eval_lons,
    eval_lats,
)

# Build the fitting matrix for the magnetometer locations.
G, invG = model.Fitting_Matrix(
    mag_lons,
    mag_lats,
    system="divergence-free",
)

# Evaluate the reconstructed magnetic field on the evaluation grid.
br, be, bn = model.Magnetic_Field(
    east,
    south,
    up,
    system="divergence-free",
)

print(br.shape, be.shape, bn.shape)
```

Run the example from the repository root with:

```bash
python your_script.py
```

## Main API

### `SECSpy.SECS`

The `SECS` class is the primary interface. It is initialised with SEC pole
longitudes and latitudes, plus evaluation longitudes and latitudes:

```python
from SECSpy import SECS

model = SECS(
    longitude=node_lons,
    latitude=node_lats,
    eval_longitude=eval_lons,
    eval_latitude=eval_lats,
    current_radius=6.371e6 + 110e3,
    mode="default",
)
```

Important methods:

- `Fitting_Matrix(...)`: builds a magnetic-field fitting matrix and stores the
  inverse operator used by later evaluations.
- `G_Matrix(...)`: returns the magnetic-field design matrix without computing
  the full fitting matrix.
- `eval_G_Matrix_B(...)`: returns magnetic-field evaluation matrices.
- `eval_G_Matrix_J(...)`: returns current-density evaluation matrices.
- `Magnetic_Field(...)`: reconstructs magnetic-field components from input
  magnetometer data.
- `Currents(...)`: reconstructs equivalent current components.
- `Amplitude(...)`: returns fitted SECS amplitudes.
- `Performance_Parameter(...)`: evaluates model performance against
  observations.

Supported SECS systems:

- `"divergence-free"` for equivalent ionospheric currents
- `"curl-free"` for curl-free systems

Supported model modes:

- `"default"` for one pole set, or independent ionospheric and telluric pole
  sets when `image_current_radius` and optional secondary pole locations are
  provided
- `"image"` for the image-current relationship described by Juusola et al.
  (2016), requiring `image_current_radius`

## Units and Component Conventions

- Longitudes and latitudes are in degrees.
- Radii and altitudes used by the SECS model are in metres.
- Magnetic-field inputs and outputs are in Tesla.
- The magnetometer input order used by high-level methods is `East`, `South`,
  `Up`, while the internal data vector is assembled as `South`, `East`, `Up`.
- Returned magnetic-field components are radial/up (`Br`), east (`Be`), and
  north (`Bn`) unless otherwise documented by the method.

## Grid and Plotting Helpers

`SECSpy.Grid_tools` provides:

- `GridCheck(...)` to identify SEC poles too close to magnetometer locations
- `ImproveGrid(...)` to search nearby grid-centre adjustments that reduce
  problematic pole/magnetometer proximity

`SECSpy.Plotting_Tools` provides Matplotlib helpers for image annotations,
coastline fills, contour labels, and subplot alignment.

## Notes

- The built-in example data are intended for smoke testing and demonstrations,
  not as a reference data product.
- `development_code/EZIE/` contains active workflow scripts and may depend on
  local paths or data directories that are not part of the minimal package API.
- There is currently no automated test suite or package build configuration in
  this repository.
