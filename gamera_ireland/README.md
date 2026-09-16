# GAMERA Ireland

Reproducible Python workflow for analysing ground magnetic perturbations over
Ireland from MAGE/GAMERA simulations.

## Scientific Aim

The project will use strong-event MAGE/GAMERA simulations to study ground-level
geomagnetic perturbations over Ireland and to design dense virtual magnetometer
networks. Initial real-station references include Dunsink, Valentia, Armagh,
Birr, and Florence Court.

Target ground magnetic quantities:

- Delta Bx
- Delta By
- Delta Bz
- Delta H
- dB/dt

## Intended Pipeline

```text
solar wind event
  -> MAGE
  -> GAMERA + REMIX + RAIJU
  -> ionospheric and magnetospheric current systems
  -> calcdb.x
  -> ground Delta B
  -> extract Ireland domain
  -> virtual magnetometer network
  -> evaluate station configurations
  -> optimise station placement
```

## Project Layout

```text
gamera_ireland/
  README.md
  environment/
  config/
  scripts/
    check_environment.py
    inspect_hdf5.py
    make_ireland_grid.py
    prepare_omni_bcwind.py
  src/
    gamera_ireland/
      __init__.py
      io.py
      coordinates.py
      virtual_stations.py
      metrics.py
      plotting.py
  notebooks/
  data/
    raw/
    processed/
  simulations/
  output/
    figures/
    station_timeseries/
    optimisation/
```

## Current Scope

This repository currently contains scaffolding and diagnostics only. It does not
assume any MAGE, GAMERA, REMIX, RAIJU, or `deltab.h5` dataset names. Real output
files should first be inspected with `scripts/inspect_hdf5.py`.

KaiPy is treated as an analysis package. The simulation executables, including
`calcdb.x`, are expected to come from the JHUAPL Kaiju/MAGE source tree and must
be installed or built separately.

## Diagnostics

Check the active Python environment:

```bash
conda activate SECSpy
python gamera_ireland/scripts/check_environment.py
```

Inspect an HDF5 file without assuming its schema:

```bash
python gamera_ireland/scripts/inspect_hdf5.py path/to/file.h5
```

## May 2024 Run Preparation

The May 2024 event configuration is stored in:

```text
gamera_ireland/config/may2024_event.json
```

It uses the same central event time as
`development_code/Gap_Analysis/create_first_inversion.py`:

```text
event: 2024-05-10T00:00:00
start: 2024-05-09T00:00:00
end:   2024-05-15T00:00:00
```

Create the Ireland virtual magnetometer grid:

```bash
conda activate SECSpy
python gamera_ireland/scripts/make_ireland_grid.py
```

This writes:

- `gamera_ireland/config/ireland_virtual_grid.csv`
- `gamera_ireland/config/ireland_reference_stations.csv`
- `gamera_ireland/config/ireland_grid_metadata.json`

Prepare GAMERA solar-wind boundary conditions from a Solar_Wind_Tools OMNI HDF
file:

```bash
conda activate SECSpy
python gamera_ireland/scripts/prepare_omni_bcwind.py --f107 180 --kp 5
```

If the OMNI HDF file does not exist yet, request the Solar_Wind_Tools download:

```bash
conda activate SECSpy
python gamera_ireland/scripts/prepare_omni_bcwind.py --download --f107 180 --kp 5
```

The downloaded/raw OMNI HDF path is configured as:

```text
gamera_ireland/data/raw/omni_1min_202405.h5
```

The prepared GAMERA boundary-condition file is:

```text
gamera_ireland/simulations/may2024_ireland/bcwind.h5
```

A preliminary MAGE/Voltron run-preparation options file is stored at:

```text
gamera_ireland/simulations/may2024_ireland/geospace_local_dgrid.json
```

It targets a local D-grid GAMERA-RAIJU run using the prepared `bcwind.h5`.
The generated run files are:

- `gamera_ireland/simulations/may2024_ireland/lfmD.h5`
- `gamera_ireland/simulations/may2024_ireland/raijuconfig.h5`
- `gamera_ireland/simulations/may2024_ireland/may2024_ireland.xml`
- `gamera_ireland/simulations/may2024_ireland/may2024_smoke.xml`

MAGE preprocessing uses the `MAGEPy312` environment with public
`kaipy==1.1.5`; compiled Kaiju executables run from `SECSpy`.

`--f107` and `--kp` are explicit fallbacks used only if those values are not
available in the OMNI HDF table.

Create a divergence-free SECS current test field over the Ireland grid:

```bash
conda activate SECSpy
python gamera_ireland/scripts/make_divergence_free_currents.py
```

This writes:

```text
gamera_ireland/output/optimisation/divergence_free_unit_current.h5
```

That file is a SECS basis/test current product, not a GAMERA-derived physical
current system. Physical ionospheric/magnetospheric currents require a real
MAGE/GAMERA/REMIX/RAIJU run.

## Next Development Steps

1. Run the full local D-grid MAGE/GAMERA/REMIX/RAIJU model with
   `may2024_ireland.xml`.
2. Run `calcdb.x` on the real simulation output to produce `*.deltab.h5`.
3. Inspect the HDF5 structure with `scripts/inspect_hdf5.py`.
4. Add schema-aware readers only after real dataset names, dimensions, units,
   coordinates, and attributes have been confirmed.
5. Extract Ireland grid/station time series and implement network evaluation.

See `environment/kaiju_build_notes.md` for the current Kaiju/MAGE build state
and run notes.
