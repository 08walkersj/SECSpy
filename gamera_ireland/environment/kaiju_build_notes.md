# Kaiju/MAGE Build Notes

Date: 2026-08-14

Runtime environment for compiled Kaiju executables: `SECSpy`

Run-preparation environment for public KaiPy: `MAGEPy312`

## Built Executables

The current public Kaiju/MAGE source was cloned to:

```text
gamera_ireland/external/kaiju-master
```

Source state:

```text
branch: master
commit: 9e19bfc
tag context: MAGE_1.25.1
```

Configured with:

```bash
conda run -n SECSpy cmake \
  -S gamera_ireland/external/kaiju-master \
  -B gamera_ireland/external/kaiju-master/build_serial \
  -DALLOW_INVALID_COMPILERS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_Fortran_COMPILER=/home/simon/anaconda3/envs/SECSpy/bin/gfortran \
  -DCMAKE_C_COMPILER=/home/simon/anaconda3/envs/SECSpy/bin/gcc \
  -DHDF5_ROOT=/home/simon/anaconda3/envs/SECSpy
```

Built with:

```bash
conda run -n SECSpy cmake --build \
  gamera_ireland/external/kaiju-master/build_serial \
  --target calcdb.x gamera.x remix.x raijuSA.x raijuOWD.x voltron.x -j 4
```

Resulting serial executables:

```text
gamera_ireland/external/kaiju-master/build_serial/bin/calcdb.x
gamera_ireland/external/kaiju-master/build_serial/bin/gamera.x
gamera_ireland/external/kaiju-master/build_serial/bin/remix.x
gamera_ireland/external/kaiju-master/build_serial/bin/raijuSA.x
gamera_ireland/external/kaiju-master/build_serial/bin/raijuOWD.x
gamera_ireland/external/kaiju-master/build_serial/bin/voltron.x
```

Smoke test:

```bash
conda run -n SECSpy gamera_ireland/external/kaiju-master/build_serial/bin/voltron.x
```

Expected no-input result:

```text
No input deck specified, defaulting to Input.xml
Error opening input deck, exiting ...
```

## Older MAGE 1.0.5 Attempt

The MAGE 1.0.5 tag was cloned to:

```text
gamera_ireland/external/kaiju
```

Source state:

```text
tag: MAGE_1.0.5
commit: d86b87d
```

`calcdb.x`, `gamera.x`, and `remix.x` built successfully after a local GNU
Fortran compatibility patch in `src/base/transforms/apex.F90`.

`voltron.x` did not link because the old RCM path requires an unavailable
third-party routine:

```text
undefined reference to `claw2ez_`
```

That matches the MAGE 1.0.5 README warning that external RCM code is required.

## Public KaiPy Run Preparation

The public Kaiju/MAGE executable build is present. MAGE run preparation now uses
a separate Python 3.12 conda environment because `SECSpy` is Python 3.14 and
public `kaipy >= 1.1.5` declares `Python >=3.9,<3.13`.

Created environment:

```bash
conda create -n MAGEPy312 -c conda-forge python=3.12 pip
conda run -n MAGEPy312 python -m pip install "kaipy==1.1.5"
```

Verified in `MAGEPy312`:

- `kaipy/scripts/preproc/genRAIJU.py`
- `kaipy/gamera/lfmG.X.txt`
- `kaipy/gamera/lfmG.Y.txt`
- `kaipy/raiju/waveModel/chorus_polynomial.txt`

Generated in `gamera_ireland/simulations/may2024_ireland`:

```text
lfmD.h5
raijuconfig.h5
may2024_ireland.xml
may2024_ireland.pbs
may2024_ireland_pbs.sh
```

Run-preparation command:

```bash
cd gamera_ireland/simulations/may2024_ireland
KAIJUHOME=/home/simon/gits/SECpy_cleaned/gamera_ireland/external/kaiju-master \
conda run -n MAGEPy312 python /home/simon/gits/SECpy_cleaned/gamera_ireland/external/kaiju-master/scripts/makeitso/makeitso.py \
  --options_path geospace_local_dgrid.json --clobber --verbose
```

The May 2024 solar wind reaches about `1.026e6 m/s`, so the GAMERA Boris speed
cap was increased from `Ca=10.0` to `Ca=20.0` in
`geospace_local_dgrid.json`.

## Smoke Test

Added a short test deck:

```text
gamera_ireland/simulations/may2024_ireland/may2024_smoke.xml
```

It reuses `bcwind.h5`, `lfmD.h5`, and `raijuconfig.h5`, but runs a short
`may2024_smoke` case.

Command:

```bash
cd gamera_ireland/simulations/may2024_ireland
conda run -n SECSpy /home/simon/gits/SECpy_cleaned/gamera_ireland/external/kaiju-master/build_serial/bin/voltron.x may2024_smoke.xml
```

The smoke test got through MAGE initialization and wrote real component output:

```text
may2024_smoke.gam.h5
may2024_smoke.gamCpl.h5
may2024_smoke.mix.h5
may2024_smoke.raiju.h5
may2024_smoke.volt.h5
```

The smoke process was manually terminated after it had produced output. Any HDF5
write diagnostics at the end of that captured run are from terminating during
file writes, not from the original preprocessing or startup path.

## Next Required Step

Run the full D-grid model, preferably with an explicit small thread count:

```bash
cd gamera_ireland/simulations/may2024_ireland
OMP_NUM_THREADS=4 \
conda run -n SECSpy /home/simon/gits/SECpy_cleaned/gamera_ireland/external/kaiju-master/build_serial/bin/voltron.x may2024_ireland.xml
```

For production-scale May 2024 runs, build and use `voltron_mpi.x` rather than
the serial executable.
