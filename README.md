# SECSpy

SECSpy is a Python toolkit for modelling ionospheric equivalent currents with
Spherical Elementary Current Systems (SECS). It includes tools for building SECS
forward matrices, fitting amplitudes from ground magnetometer data, evaluating
magnetic fields and equivalent currents, checking grids, and plotting results.

The package also includes an EZIE/SuperMAG workflow for creating event-centred
cubed-sphere grids, downloading SuperMAG data, running SECS inversions, and
making pass/event plots.



## Usage Policy
For inverions complimenting the EZIE mission and satellites please acknowledge us appropriately:
"For the SECS inversion of ground magnetometer data used for comparison with EZIE, we gratefully acknowledge the Magnetic Network of Ireland (MagIE), supported by the Dublin Institute for Advanced Studies (DIAS), Met Éireann, and Geological Survey Ireland (GSI)."

## Installation

Install from a local checkout:

```bash
git clone https://github.com/08walkersj/SECSpy.git
cd SECSpy
python -m pip install .
```

If the repository is shared as a copied folder or zip file, unzip it first and
install from the folder path:

```bash
python -m pip install /path/to/SECSpy
```

For development, install in editable mode:

```bash
python -m pip install -e .
```

Install directly from Git:

```bash
python -m pip install "git+https://github.com/08walkersj/SECSpy.git"
```

The base install only requires `numpy` and `scipy`. Optional extras are
available for common workflows:

```bash
python -m pip install ".[data]"
python -m pip install ".[plotting]"
python -m pip install ".[video]"
python -m pip install ".[ezie]"
python -m pip install ".[all]"
```

Use `.[ezie]` or `.[all]` for the EZIE inversion notebook, because it depends on
packages such as `apexpy`, `pandas`, `xarray`, `secsy`, and `polplot`.

## Usage

The notebooks are the main usage examples:

- `notebooks/Vanhamäki(2020)_example.ipynb` demonstrates the core SECS model
  using the bundled example grid and magnetometer data.
- `notebooks/EZIE_inversion_example.ipynb` demonstrates the EZIE/SuperMAG
  inversion workflow, including grid creation, SuperMAG download, inversion,
  saving/loading state, and plotting.

Run notebooks from the repository root, or install the package first so
`SECSpy` is importable from any working directory.

## Main Modules

- `SECSpy.SECS`: core SECS model, G matrices, inversions, fields, currents, and
  amplitudes.
- `SECSpy.Grid_tools`: grid quality checks and grid adjustment helpers.
- `SECSpy.Plotting_Tools`: Matplotlib helpers for contours, coastlines, labels,
  and subplot alignment.
- `SECSpy.EZIE_inversion`: EZIE/SuperMAG inversion workflow.

## References

The core SECS methods follow:

- Vanhamäki, H., & Juusola, L. (2020), *Introduction to Spherical Elementary
  Current Systems*, https://doi.org/10.1007/978-3-030-26732-2_2
- Marsal, S., Torta, J. M., Segarra, A., & Araki, T. (2016), *Journal of
  Geophysical Research: Space Physics*, https://doi.org/10.1002/2016JA023166
- Juusola, L., Kauristie, K., Vanhamäki, H., Aikio, A., & van de Kamp, M.
  (2016), *Journal of Geophysical Research: Space Physics*,
  https://doi.org/10.1002/2016JA022961
