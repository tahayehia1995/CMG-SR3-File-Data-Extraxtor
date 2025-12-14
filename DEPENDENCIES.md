## Dependencies

### Install

Use the single consolidated dependency file:

- `dependencies.txt`

Install into your active Python environment (or the launcher venv):

```bash
pip install -r dependencies.txt
```

### What these libraries are for

This project is an interactive SR3 data extractor and visualizer for batch processing CMG (Computer Modelling Group)
reservoir simulation data. It provides tools for extracting and visualizing spatial properties and time series data
from multiple `.sr3` files.

#### Core Python (built-in)

- `os`, `glob`, `pathlib`, `warnings`, `datetime`, `sys`, etc.

#### Data processing

- `numpy`: array operations and numerical computing
- `pandas`: tabular data processing / CSV operations
- `h5py`: reading/writing HDF5 (`.h5`) and accessing SR3-derived data structure

#### Visualization

- `matplotlib`: plotting spatial and time series outputs

#### UI / runtime

- `streamlit`: web UI framework
- `ipython`, `ipywidgets`: imported by some core modules; included to avoid import errors

#### Optional

- `tqdm`: progress bars
- `pyngrok`: optional internet tunneling support




