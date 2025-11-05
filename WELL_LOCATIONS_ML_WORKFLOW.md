# Well Locations File - ML Workflow Compatibility

## Overview

The `well_locations.h5` file is now structured to match the spatial properties files format, making it compatible with ML workflows. It includes:

1. **Binary well location data**: `(N_cases, N_timesteps, Nx, Ny, Nz)` array where `1` = well completion, `0` = no well
2. **Well mapping**: `(N_cases, N_timesteps, Nx, Ny, Nz)` array mapping each cell to well index (`-1` = no well, `0` to `N_wells-1` = well index)
3. **Enhanced visualization**: Well names and types (I/P/U) displayed on 2D layer plots

## File Structure

```
well_locations.h5
├── data                    # (N_cases, N_timesteps, Nx, Ny, Nz) uint8 - binary mask
├── well_mapping            # (N_cases, N_timesteps, Nx, Ny, Nz) int16 - well indices
└── metadata/
    ├── n_cases             # Attribute: number of cases
    ├── n_timesteps         # Attribute: number of timesteps
    ├── nx, ny, nz          # Attributes: grid dimensions
    ├── grid_dimensions      # Attribute: tuple (nx, ny, nz)
    ├── total_cells          # Attribute: total grid cells
    ├── active_cells         # Attribute: number of active cells
    ├── timesteps            # Dataset: timestep indices (0, 1, 2, ...)
    ├── target_days          # Dataset: simulation days
    ├── sr3_files            # Dataset: source SR3 filenames
    ├── well_names           # Dataset: well names (string array)
    ├── well_types           # Dataset: well types ('I', 'P', 'U')
    ├── completion_counts    # Dataset: (N_cases, N_timesteps) completion counts
    └── active_cell_mapping  # Dataset: active cell mapping (optional)
```

## ML Workflow Usage

### Basic Data Access

```python
import h5py
import numpy as np

with h5py.File('well_locations.h5', 'r') as f:
    # Binary well locations (same format as spatial properties)
    well_locations = f['data'][...]  # Shape: (N_cases, N_timesteps, Nx, Ny, Nz)
    
    # Well mapping for enhanced access
    well_mapping = f['well_mapping'][...]  # Shape: (N_cases, N_timesteps, Nx, Ny, Nz)
    
    # Metadata
    meta = f['metadata']
    n_cases = meta.attrs['n_cases']
    n_timesteps = meta.attrs['n_timesteps']
    nx, ny, nz = meta.attrs['nx'], meta.attrs['ny'], meta.attrs['nz']
    
    # Well information
    well_names = [name.decode() for name in meta['well_names'][...]]
    well_types = [wt.decode() for wt in meta['well_types'][...]]
```

### Example 1: Single Layer Access

```python
# Get well locations for specific case, timestep, layer
case_idx = 0
timestep_idx = 0
layer_idx = 10

layer_wells = well_locations[case_idx, timestep_idx, :, :, layer_idx]
# Shape: (Nx, Ny), dtype: uint8, values: 0 or 1
```

### Example 2: Batch Access for Training

```python
# Get all timesteps for all cases (full dataset)
X_wells = well_locations[:, :, :, :, :]
# Shape: (N_cases, N_timesteps, Nx, Ny, Nz)

# Get specific case for training
case_data = well_locations[case_idx, :, :, :, :]
# Shape: (N_timesteps, Nx, Ny, Nz)

# Get specific timestep across all cases
timestep_data = well_locations[:, timestep_idx, :, :, :]
# Shape: (N_cases, Nx, Ny, Nz)
```

### Example 3: Separate Injectors and Producers

```python
# Get well type indices
injector_indices = [i for i, wt in enumerate(well_types) if wt == 'I']
producer_indices = [i for i, wt in enumerate(well_types) if wt == 'P']

# Create separate masks
injector_mask = np.zeros_like(well_locations, dtype=np.uint8)
producer_mask = np.zeros_like(well_locations, dtype=np.uint8)

for case in range(n_cases):
    for ts in range(n_timesteps):
        for i_idx in injector_indices:
            injector_mask[case, ts][well_mapping[case, ts] == i_idx] = 1
        for p_idx in producer_indices:
            producer_mask[case, ts][well_mapping[case, ts] == p_idx] = 1

# Now injector_mask and producer_mask have same shape as well_locations
# Use them separately in ML models
```

### Example 4: Get Well Names for Each Cell

```python
# For a specific layer
layer_mapping = well_mapping[case_idx, timestep_idx, :, :, layer_idx]

# Find cells with wells
well_cells = np.where(layer_mapping >= 0)

# Get well information for each cell
for i, j in zip(well_cells[0], well_cells[1]):
    well_idx = layer_mapping[i, j]
    well_name = well_names[well_idx]
    well_type = well_types[well_idx]
    print(f"Cell ({i}, {j}): Well {well_name} ({well_type})")
```

### Example 5: Combine with Spatial Properties

```python
# Load spatial property (e.g., pressure)
with h5py.File('batch_spatial_properties_PRES.h5', 'r') as f:
    pressure = f['data'][...]  # Shape: (N_cases, N_timesteps, Nx, Ny, Nz)

# Load well locations
with h5py.File('well_locations.h5', 'r') as f:
    wells = f['data'][...]  # Shape: (N_cases, N_timesteps, Nx, Ny, Nz)

# Ensure shapes match
assert pressure.shape == wells.shape

# Use together in ML workflow
# Example: Extract features at well locations
case_idx = 0
timestep_idx = 0
layer_idx = 10

pressure_layer = pressure[case_idx, timestep_idx, :, :, layer_idx]
wells_layer = wells[case_idx, timestep_idx, :, :, layer_idx]

# Get pressure values at well locations
pressure_at_wells = pressure_layer[wells_layer == 1]
```

## Visualization

The enhanced visualization displays wells with:
- **Injectors (I)**: Blue triangles (^) with blue labels
- **Producers (P)**: Red inverted triangles (v) with red labels  
- **Unknown (U)**: Yellow circles (o) with black labels

Well names are displayed as text labels near markers (up to 20 per type to avoid clutter).

## Testing

Run the test script to verify file structure and ML workflow compatibility:

```bash
python test_well_locations.py [path_to_well_locations.h5] [path_to_spatial_file.h5]
```

The test script verifies:
1. File structure matches spatial files format
2. Data can be accessed in ML workflow style
3. Well mapping works correctly
4. All required metadata is present

## Compatibility

- ✅ **Shape compatibility**: `(N_cases, N_timesteps, Nx, Ny, Nz)` matches spatial properties format
- ✅ **Data type**: Binary `uint8` for efficient storage (same as spatial masks)
- ✅ **Metadata structure**: Matches spatial files metadata group
- ✅ **Compression**: Uses gzip compression with chunking for efficient I/O
- ✅ **Backward compatibility**: Old files without `well_mapping` still work (legacy mode)

## Notes

- The `well_mapping` dataset enables enhanced visualization but is optional for basic ML workflows
- The binary `data` array is sufficient for most ML tasks (predicting well locations, etc.)
- Use `well_mapping` when you need to distinguish between different wells or well types
- Well types are inferred from well names (I* = Injector, P* = Producer)

