# LOMPE-like GAMERA candidate runs

This note records public, from-scratch GAMERA/ReMIX/RAIJU attempts intended to
produce fields similar to the LOMPE paper synthetic data. These are not exact
reproductions of the paper's original run deck.

## Target

Reference file:

`/home/simon/gits/lompe/examples/sample_dataset/gamera.nc`

The target is a post-processed single-time GAMERA product on a 44 x 360
mlat/mlt grid:

- time: `1998-04-23T20:20:30.009514`
- step: `462`
- fields: `FAC`, `potential`, `Etheta`, `Ephi`, `SigmaPed`, `SigmaHall`,
  `dBg_r`, `dBg_theta`, `dBg_phi`, `dBspace_r`, `dBspace_theta`,
  `dBspace_phi`

## Candidate generator

Parameterized setup script:

```bash
conda run -n SECSpy python gamera_ireland/scripts/setup_lompe_candidate_run.py --help
```

The generated case uses:

- public Kaiju executable: `voltron.x`
- public KaiPy preprocessing: `genLFM.py -gid D` and `genRAIJU.py`
- grid: `lfmD.h5`, 48 x 48 x 64
- coupled models: GAMERA + ReMIX + RAIJU
- ground perturbations: `calcdb.x`
- ReMIX grid: default 45 vertex latitudes x 360 longitudes, low-latitude
  boundary about 45 degrees

The intent is that case studies can be varied by replacing or regenerating only
`bcwind.h5`, while leaving the model/grid/post-processing path fixed.

## Attempts

Scores were computed by
`gamera_ireland/scripts/compare_lompe_gamera_candidate.py`. Each field is
interpolated onto the LOMPE `mlat/mlt` grid. The comparison allows an MLT roll
because the original pressure-pulse phase/local-time setup is not known.

Lower normalized RMSE is better. `scale` is candidate standard deviation divided
by target standard deviation.

| case | driver | best step | mean NRMSE | notes |
| --- | --- | ---: | ---: | --- |
| baseline | density 5 to 25 cm-3, Vx=-400 km/s, Bz=-5 nT, 2 min | Step#2 | 1.108 | correct workflow but too weak |
| candidate_a | density 22.5 to 40 cm-3, Vx=-450 km/s, Bz=-10 nT, 5 min | Step#5 | 0.977 | current best base case |
| candidate_b | density 15 to 25 cm-3, Vx=-450 km/s, Bz=-15 nT, 4 min | Step#4 | 1.006 | similar but slightly weaker fit |

Best current base case:

```bash
conda run -n SECSpy python gamera_ireland/scripts/setup_lompe_candidate_run.py \
  --run-dir gamera_ireland/simulations/lompe_candidate_a \
  --runid lompe_candidate_a \
  --duration-s 300 \
  --spinup-s 60 \
  --dtout-s 60 \
  --density-base 5 \
  --density-delta 35 \
  --pulse-start-s -60 \
  --pulse-end-s 900 \
  --vx-kms -450 \
  --bz-south-nt -10 \
  --f107 120 \
  --kp 4 \
  --clobber

cd gamera_ireland/simulations/lompe_candidate_a
conda run -n SECSpy /home/simon/gits/SECpy_cleaned/gamera_ireland/external/kaiju-master/build_serial/bin/voltron.x lompe_candidate_a.xml
conda run -n SECSpy /home/simon/gits/SECpy_cleaned/gamera_ireland/external/kaiju-master/build_serial/bin/calcdb.x calcdb_lompe_candidate_a.xml
cd /home/simon/gits/SECpy_cleaned
conda run -n SECSpy python gamera_ireland/scripts/compare_lompe_gamera_candidate.py \
  gamera_ireland/simulations/lompe_candidate_a \
  --runid lompe_candidate_a \
  --output gamera_ireland/output/lompe_candidate_a_comparison.json
```

Best Candidate A field scores at `Step#5`:

| field | NRMSE | corr | scale |
| --- | ---: | ---: | ---: |
| `FAC` | 0.908 | 0.425 | 0.415 |
| `potential` | 0.669 | 0.803 | 0.507 |
| `SigmaPed` | 0.964 | 0.662 | 1.058 |
| `SigmaHall` | 1.760 | 0.473 | 1.989 |
| `dBg_r` | 0.883 | 0.545 | 0.821 |
| `dBg_theta` | 0.823 | 0.640 | 0.759 |
| `dBg_phi` | 0.833 | 0.591 | 0.686 |

## Open Issues

- Voltron exits with `Fin`, and the output files are readable, but the serial
  runs print repeated HDF5 write diagnostics during early output/restart
  handling. Treat the current products as usable for workflow development, but
  resolve this before long production runs.
- The original LOMPE synthetic local-time phase is unknown, so the comparison
  currently allows an MLT roll. A production validation should use one common
  roll/coordinate convention across all fields.
- Candidate A is close enough to use as the current base case, but it still
  under-represents FAC and potential variance while over-representing Hall
  conductance variance. The next tuning direction is moderate dynamic pressure,
  longer runtime, and ReMIX conductance settings.
