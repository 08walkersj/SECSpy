# LOMPE Synthetic GAMERA Reproduction

This is the current verified baseline for comparing our GAMERA/Ireland workflow
against a known published example.

## Available Local Reference

The local LOMPE checkout contains the synthetic GAMERA/ReMIX product used by
the LOMPE example notebooks:

- `/home/simon/gits/lompe/examples/sample_dataset/gamera.nc`
- `/home/simon/gits/lompe/examples/synthetic_global_data_example.ipynb`
- `/home/simon/gits/lompe/examples/lompe_paper_figures/03_synthetic_demo_for_lompe.ipynb`

The dataset contains:

- `FAC`
- `potential`
- `Etheta`, `Ephi`
- `SigmaPed`, `SigmaHall`
- `dBg_r`, `dBg_theta`, `dBg_phi`
- `dBspace_r`, `dBspace_theta`, `dBspace_phi`

The dataset is a precomputed GAMERA/ReMIX product from a synthetic solar-wind
pressure-increase simulation. The original GAMERA input deck used to generate
this NetCDF file is not present in the local LOMPE repository, and the NetCDF
file contains no provenance attributes identifying the original run deck.

## Verified Reproduction Step

The script below reproduces the GAMERA truth panels without doing any LOMPE
inversion:

```bash
conda run -n SECSpy python gamera_ireland/scripts/reproduce_lompe_gamera_truth.py --kind global
conda run -n SECSpy python gamera_ireland/scripts/reproduce_lompe_gamera_truth.py --kind local
```

Outputs:

- `gamera_ireland/output/figures/lompe_gamera_truth_global.png`
- `gamera_ireland/output/figures/lompe_gamera_truth_global.json`
- `gamera_ireland/output/figures/lompe_gamera_truth_local.png`
- `gamera_ireland/output/figures/lompe_gamera_truth_local.json`

## Sign Conventions Copied From The LOMPE Notebook

- `En = -Etheta`
- `Ee = Ephi`
- `dBn = -dBg_theta`
- `dBe = dBg_phi`
- `dBr = dBg_r`
- `FAC = FAC * 1e-6`

## Next Step

Before generating new Ireland products, reproduce the full LOMPE synthetic
comparison figure from the same `gamera.nc` file. That will verify the LOMPE
model/inversion path separately from the GAMERA truth-data reading path.
