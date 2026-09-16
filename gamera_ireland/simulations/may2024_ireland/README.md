# May 2024 Ireland GAMERA Run Directory

This directory is for the May 2024 storm workflow.

Prepared inputs:

- `bcwind.h5`: GAMERA solar-wind boundary conditions generated from
  Solar_Wind_Tools OMNI 1-minute data.
- `run_manifest.json`: provenance for the prepared input files.

Create or refresh the prepared files from the repository root with:

```bash
conda activate SECSpy
python gamera_ireland/scripts/make_ireland_grid.py
python gamera_ireland/scripts/prepare_omni_bcwind.py --download --f107 180 --kp 5
```

This directory does not yet contain a Kaiju/MAGE XML run deck or simulation
output. After Kaiju is built, the next step is to create a MAGE/GAMERA run XML
that references `bcwind.h5`, run the model, then use `calcdb.x` to generate the
ground perturbation HDF5 file.
