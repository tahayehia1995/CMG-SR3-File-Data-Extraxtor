# SR3 File Structure and Extraction Mechanisms

## Table of Contents
1. [SR3 File Overview](#sr3-file-overview)
2. [SR3 File Anatomy](#sr3-file-anatomy)
3. [Extraction Mechanisms](#extraction-mechanisms)
4. [Data Processing Workflow](#data-processing-workflow)
5. [Output Format](#output-format)

---

## SR3 File Overview

### What is an SR3 File?

**SR3** (Simulation Results 3) files are binary data files created by **CMG (Computer Modelling Group)** simulators (GEM, IMEX, STARS). These files contain comprehensive reservoir simulation results in a structured HDF5 format.

### File Format

- **Format**: HDF5 (Hierarchical Data Format version 5)
- **Version**: SR3 Version 1.0.8 (as of analyzed file)
- **Simulator**: CMG GEM 2025.20
- **Structure**: Hierarchical organization with groups and datasets
- **Purpose**: Store spatial grid properties, time series data, and simulation metadata

### File Characteristics

- **Size**: Typically ranges from several MB to GB depending on grid size and number of timesteps
- **Compression**: Uses HDF5's built-in compression capabilities
- **Access**: Read-only format (simulators write, analysis tools read)
- **Cross-platform**: HDF5 format ensures compatibility across operating systems

---

## SR3 File Anatomy

### Top-Level Structure

SR3 files are organized into four main top-level groups:

```
SR3 File (HDF5)
├── General/              # Metadata, time tables, component information
├── SpatialProperties/    # 3D grid properties at different timesteps
├── TimeSeries/          # Time-varying data (wells, groups, layers, sectors)
└── Tables/              # Tabular data and relationships
```

### 1. General Group

The `General` group contains metadata and reference tables essential for understanding the simulation.

#### Structure:
```
General/
├── MasterTimeTable/      # Universal time reference
├── ComponentTable/       # Chemical components in simulation
├── UnitsTable/           # Unit conversion information
├── EventTable/          # Simulation events
├── HistoryTable/        # Historical data
├── NameRecordTable/     # Named records
├── TableAssociations/   # Table relationships
└── UnitConversionTable/ # Unit conversion factors
```

#### MasterTimeTable

**Purpose**: Provides universal time reference for all data in the file.

**Structure**:
- **Format**: Structured array (numpy record array)
- **Fields**:
  - `Index`: Timestep index (0-based)
  - `Offset in days`: Simulation time in days from start
  - `Date`: Calendar date (YYYYMMDD format with fractional days)

**Example Record**:
```python
{
    'Index': 0,
    'Offset in days': 0.0,
    'Date': 20250101.0
}
```

**Key Characteristics**:
- Contains all timesteps in the simulation
- Used for synchronizing data across multiple SR3 files
- Critical for batch processing multiple simulation cases
- Time range: Typically spans entire simulation period (e.g., 0 to 11,321 days)

**Usage in Extraction**:
- Maps timestep indices to simulation days
- Enables date-based extraction across files with different timestep frequencies
- Provides common reference point for batch processing

#### ComponentTable

**Purpose**: Lists chemical components present in the simulation.

**Structure**:
- Array of component names
- Examples: `CO2`, `N2`, `C1`, `C2`, `C3`, `WATER`, etc.

**Usage**: Identifies which components are tracked in the simulation results.

---

### 2. SpatialProperties Group

The `SpatialProperties` group contains 3D grid properties organized by timestep.

#### Structure:
```
SpatialProperties/
├── 000000/              # Timestep 0 (6-digit zero-padded)
│   ├── GRID/            # Grid structure and mapping
│   │   ├── IGNTID/      # I-direction indices
│   │   ├── IGNTJD/      # J-direction indices
│   │   ├── IGNTKD/      # K-direction (layer) indices
│   │   └── IPSTCS/      # Active cell mapping
│   ├── POROS/           # Porosity property (1D array)
│   ├── PRES/            # Pressure property (1D array)
│   ├── SG/              # Gas saturation (1D array)
│   └── ...              # Other properties
├── 000035/              # Timestep 35
├── 000062/              # Timestep 62
└── ...
```

#### Timestep Organization

- **Format**: 6-digit zero-padded integers (e.g., `000000`, `000035`, `000062`)
- **Meaning**: These are indices into the `MasterTimeTable`
- **Frequency**: Not all MasterTimeTable timesteps have spatial data (typically fewer spatial snapshots)

#### GRID Subgroup

The `GRID` subgroup contains essential information for mapping 1D active cell arrays to 3D grid positions.

##### Key Datasets:

1. **IGNTID** (I-direction Grid Cell Indices)
   - **Type**: Integer array
   - **Format**: 1-based indices
   - **Purpose**: Maps active cells to I (x-direction) coordinates
   - **Range**: 1 to `ni` (number of cells in I-direction)

2. **IGNTJD** (J-direction Grid Cell Indices)
   - **Type**: Integer array
   - **Format**: 1-based indices
   - **Purpose**: Maps active cells to J (y-direction) coordinates
   - **Range**: 1 to `nj` (number of cells in J-direction)

3. **IGNTKD** (K-direction Grid Cell Indices)
   - **Type**: Integer array
   - **Format**: 1-based indices
   - **Purpose**: Maps active cells to K (z-direction/layer) coordinates
   - **Range**: 1 to `nk` (number of layers)

4. **IPSTCS** (Active Cell Mapping)
   - **Type**: Integer array
   - **Format**: 1-based grid indices
   - **Purpose**: Maps 1D active cell array positions to 3D grid positions
   - **Key**: Value of 0 indicates inactive cell
   - **Length**: Equal to number of active cells

##### Grid Dimensions

Grid dimensions are determined from the maximum values in IGNTID, IGNTJD, and IGNTKD:

```python
ni = max(IGNTID)  # I-direction (x-axis)
nj = max(IGNTJD)  # J-direction (y-axis)
nk = max(IGNTKD)  # K-direction (z-axis, layers)
```

**Example**: A grid with dimensions 34 × 16 × 25 means:
- 34 cells in I-direction (x-axis)
- 16 cells in J-direction (y-axis)
- 25 layers in K-direction (z-axis)
- Total cells: 13,600 (34 × 16 × 25)

##### Active vs Inactive Cells

- **Active Cells**: Cells that participate in simulation calculations
- **Inactive Cells**: Cells excluded from simulation (e.g., outside reservoir boundaries)
- **Storage Efficiency**: Only active cell values are stored, significantly reducing file size
- **Example**: Grid with 13,600 total cells might have 10,065 active cells (74% active)

#### Property Storage Format

**Key Insight**: Properties are stored as **1D arrays containing only active cell values**.

- **Array Length**: Equal to number of active cells (not total grid cells)
- **Data Type**: Typically `float32` or `float64`
- **Order**: Same order as IPSTCS mapping

**Example**:
```python
# Property array (e.g., POROS)
poros_data.shape  # (10065,) - only active cells

# Grid dimensions
grid_dims = (34, 16, 25)  # ni, nj, nk

# Total cells
total_cells = 34 * 16 * 25  # 13,600

# Active cells
active_cells = 10,065  # 74% of total
```

#### Available Properties

Properties vary by simulation type and configuration. Common categories:

**Static Properties** (do not change with time):
- `POROS` - Porosity
- `PERMI`, `PERMJ`, `PERMK` - Permeability in I, J, K directions
- `NET/GROSS` - Net-to-gross ratio

**Dynamic Properties** (change with time):
- `PRES` - Pressure
- `SG`, `SW`, `SO` - Gas, Water, Oil saturations
- `MASDENG`, `MASDENO`, `MASDENW` - Mass densities
- `KRG`, `KRW`, `KRO` - Relative permeabilities

**Example**: A typical file may contain 50-60 different properties.

---

### 3. TimeSeries Group

The `TimeSeries` group contains time-varying data organized by entity type.

#### Structure:
```
TimeSeries/
├── WELLS/               # Well data
│   ├── Timesteps/      # Timestep indices
│   ├── Variables/      # Variable names
│   ├── Origins/        # Well names
│   └── Data/           # Data array (timesteps × variables × wells)
├── GROUPS/             # Group data
├── LAYERS/             # Layer data
└── SECTORS/            # Sector data
```

#### WELLS Subgroup

**Purpose**: Stores time series data for all wells in the simulation.

##### Datasets:

1. **Timesteps**
   - **Type**: Integer array
   - **Shape**: `(N_timesteps,)`
   - **Purpose**: Indices into MasterTimeTable
   - **Note**: May differ from spatial timesteps (wells often have more frequent data)

2. **Variables**
   - **Type**: String array
   - **Shape**: `(N_variables,)`
   - **Purpose**: Names of well variables
   - **Examples**: `BHP`, `GASVOLRC`, `WATVOLRC`, `OILVOLRC`, etc.
   - **Count**: Typically 100-400+ variables depending on simulation complexity

3. **Origins** (Well Names)
   - **Type**: String array
   - **Shape**: `(N_wells,)`
   - **Purpose**: Well identifiers
   - **Examples**: `I1`, `I2`, `P1`, `P2` (Injectors and Producers)

4. **Data**
   - **Type**: Float array
   - **Shape**: `(N_timesteps, N_variables, N_wells)`
   - **Purpose**: Time series data for all wells
   - **Note**: Shape order is timesteps × variables × wells

**Example**:
```python
# Data array shape
data.shape  # (33, 410, 6)
# 33 timesteps, 410 variables, 6 wells

# Accessing data
bhp_data = data[:, variable_index['BHP'], :]  # BHP for all wells, all timesteps
well1_data = data[:, :, 0]  # All variables for well 0 (first well)
```

---

### 4. Tables Group

The `Tables` group contains additional tabular data and relationships. Structure varies by simulation type and may include:
- Completion tables
- Connection tables
- Well trajectory data
- Other simulation-specific tables

---

## Extraction Mechanisms

### Overview

The extraction process converts SR3 files from their native format (1D active cell arrays) into standard 3D/4D/5D arrays suitable for analysis and visualization.

### Key Challenges

1. **Active Cell Mapping**: Converting 1D active cell arrays to 3D grid positions
2. **Time Synchronization**: Aligning data across multiple SR3 files with different timestep frequencies
3. **Batch Processing**: Efficiently processing multiple files in parallel
4. **Layer Selection**: Extracting only specific layers when needed

---

### 1. Active Cell Mapping to 3D Grid

#### Problem

Properties in SR3 files are stored as 1D arrays containing only active cell values. To visualize or analyze data, we need to map these back to their 3D grid positions.

#### Solution: IPSTCS Mapping

The `IPSTCS` array provides the mapping from 1D active cell positions to 3D grid coordinates.

**Algorithm** (`_map_active_to_3d_vectorized`):

```python
def map_active_to_3d(active_data, ipstcs, grid_dims, inactive_value=0.0):
    """
    Map 1D active cell array to 3D grid.
    
    Args:
        active_data: 1D array of active cell values (length = N_active)
        ipstcs: Active cell mapping array (1-based grid indices)
        grid_dims: Tuple (ni, nj, nk) - grid dimensions
        inactive_value: Value to assign to inactive cells
    
    Returns:
        3D array of shape (ni, nj, nk)
    """
    ni, nj, nk = grid_dims
    
    # Initialize 3D grid with inactive value
    grid_3d = np.full((ni, nj, nk), inactive_value, dtype=np.float32)
    
    # Convert 1-based indices to 0-based
    grid_indices = ipstcs - 1
    
    # Calculate I, J, K coordinates from linear index
    k = grid_indices // (ni * nj)  # Layer
    j = (grid_indices % (ni * nj)) // ni  # J-direction
    i = grid_indices % ni  # I-direction
    
    # Filter valid coordinates
    valid_mask = (i < ni) & (j < nj) & (k < nk)
    
    # Assign active cell values to 3D grid
    grid_3d[i[valid_mask], j[valid_mask], k[valid_mask]] = active_data[valid_mask]
    
    return grid_3d
```

#### Coordinate Calculation Formula

Given a 1-based grid index `idx` (from IPSTCS):

```python
# Convert to 0-based
idx_0 = idx - 1

# Calculate coordinates
k = idx_0 // (ni * nj)        # Layer (K-direction)
j = (idx_0 % (ni * nj)) // ni  # J-direction
i = idx_0 % ni                # I-direction
```

**Example**:
```python
# Grid dimensions
ni, nj, nk = 34, 16, 25

# IPSTCS value (1-based)
ipstcs_value = 1000

# Convert to 0-based
idx_0 = 1000 - 1 = 999

# Calculate coordinates
k = 999 // (34 * 16) = 999 // 544 = 1
j = (999 % 544) // 34 = 455 // 34 = 13
i = 999 % 34 = 13

# Result: Cell at position (13, 13, 1) in 0-based indexing
```

#### Vectorized Implementation

The code uses NumPy vectorized operations for efficiency:

```python
# Vectorized coordinate calculation
grid_indices = ipstcs - 1  # Convert all at once
k = grid_indices // (ni * nj)
j = (grid_indices % (ni * nj)) // ni
i = grid_indices % ni

# Advanced indexing for bulk assignment
grid_3d[i[valid_mask], j[valid_mask], k[valid_mask]] = active_data[valid_mask]
```

**Performance**: Processes thousands of cells in milliseconds.

---

### 2. Time Synchronization Across Files

#### Problem

When batch processing multiple SR3 files:
- Each file may have different timestep frequencies
- Timestep indices differ between files
- Need to extract data at common time points

#### Solution: Date-Based Extraction

Use simulation days from `MasterTimeTable` as universal reference.

**Process**:

1. **Extract Time Mappings**:
   ```python
   # For each file, extract MasterTimeTable
   time_table = f['General/MasterTimeTable'][...]
   days = time_table['Offset in days']  # Simulation days
   timesteps = time_table['Index']      # Timestep indices
   ```

2. **Find Common Days**:
   ```python
   # Get spatial timesteps for each file
   spatial_timesteps = [0, 35, 62, 97, ...]  # From SpatialProperties keys
   
   # Map to simulation days
   spatial_days = [days[ts] for ts in spatial_timesteps]
   
   # Find intersection across all files
   common_days = set(spatial_days[0])
   for file_days in spatial_days[1:]:
       common_days &= set(file_days)
   ```

3. **Create Extraction Plan**:
   ```python
   extraction_plan = {
       'target_day': 365.0,
       'files': {
           'file1.sr3': {'spatial_timestep': 35, 'master_index': 35},
           'file2.sr3': {'spatial_timestep': 42, 'master_index': 42},
           # ...
       }
   }
   ```

4. **Extract Using Plan**:
   ```python
   for file_name, plan_entry in extraction_plan['files'].items():
       spatial_timestep = plan_entry['spatial_timestep']
       # Extract from SpatialProperties/{spatial_timestep:06d}
   ```

**Benefits**:
- Ensures data extracted at same simulation times across all files
- Handles files with different timestep frequencies
- Enables meaningful comparison between simulation cases

---

### 3. Layer Selection Mechanism

#### Problem

Users may only need specific layers (K-direction) rather than all layers, reducing file size and processing time.

#### Solution: Layer Filtering

**Process**:

1. **User Selection**:
   - User selects desired layers (e.g., layers 5-10)
   - Empty selection = all layers (default)

2. **Extraction**:
   ```python
   # Extract full 3D grid
   grid_3d = map_active_to_3d(active_data, ipstcs, grid_dims)
   # Shape: (ni, nj, nk)
   
   # Slice to selected layers
   if selected_layers:
       grid_3d_selected = grid_3d[:, :, selected_layers]
       # Shape: (ni, nj, len(selected_layers))
   else:
       grid_3d_selected = grid_3d  # All layers
   ```

3. **Storage**:
   - Save only selected layers
   - Store layer indices in metadata for reference

**Example**:
```python
# Original grid
grid_dims = (34, 16, 25)  # 25 layers

# User selects layers 5-10
selected_layers = [5, 6, 7, 8, 9, 10]

# Extracted data shape
extracted_shape = (34, 16, 6)  # Only 6 layers

# File size reduction: ~76% smaller (6/25)
```

---

### 4. Batch Processing Workflow

#### Overview

Process multiple SR3 files in parallel to extract spatial properties and time series data.

#### Workflow

1. **File Loading**:
   ```python
   # Load directory of SR3 files
   sr3_files = [Path('file1.sr3'), Path('file2.sr3'), ...]
   
   # Analyze each file
   for file in sr3_files:
       # Extract grid dimensions
       # Extract available properties
       # Extract time mappings
   ```

2. **Date Synchronization**:
   ```python
   # Find common dates across all files
   common_days = find_universal_time_points(sr3_files, mode='yearly')
   
   # Create extraction plan
   extraction_plan = create_extraction_plan(common_days, sr3_files)
   ```

3. **Parallel Extraction**:
   ```python
   # Prepare worker arguments
   worker_args = []
   for case_idx, sr3_file in enumerate(sr3_files):
       worker_args.append((
           case_idx,
           sr3_file,
           extraction_plan[filename],
           selected_properties,
           selected_layers,
           ...
       ))
   
   # Process in parallel
   with mp.Pool(processes=cpu_count()) as pool:
       results = pool.map(extract_single_file_spatial, worker_args)
   ```

4. **Data Aggregation**:
   ```python
   # Combine results into batch arrays
   batch_data = np.zeros((n_cases, n_timesteps, nx, ny, nz))
   
   for result in results:
       case_idx = result['case_idx']
       batch_data[case_idx] = result['data']
   ```

5. **Save to HDF5**:
   ```python
   # Save each property to separate file
   for prop in selected_properties:
       output_file = f"batch_spatial_properties_{prop}.h5"
       save_batch_spatial_to_hdf5(
           batch_data[prop],
           prop,
           output_file,
           selected_layers=selected_layers
       )
   ```

---

## Data Processing Workflow

### Complete Extraction Pipeline

```
┌─────────────────┐
│  Load SR3 Files │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Analyze Files    │
│ - Grid dims      │
│ - Properties     │
│ - Time mappings  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Sync Dates      │
│ - Find common   │
│ - Create plan   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Extract Data    │
│ - Parallel proc  │
│ - Map to 3D     │
│ - Filter layers │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Save HDF5        │
│ - Batch format  │
│ - Metadata      │
└─────────────────┘
```

### Step-by-Step Process

#### Step 1: File Analysis

```python
# For each SR3 file:
with h5py.File(sr3_file, 'r') as f:
    # Get grid dimensions
    grid_group = f['SpatialProperties/000000/GRID']
    ni = max(grid_group['IGNTID'][...])
    nj = max(grid_group['IGNTJD'][...])
    nk = max(grid_group['IGNTKD'][...])
    
    # Get active cell mapping
    ipstcs = grid_group['IPSTCS'][...]
    
    # Get available properties
    timestep_group = f['SpatialProperties/000000']
    properties = [k for k in timestep_group.keys() if k != 'GRID']
    
    # Get time mapping
    time_table = f['General/MasterTimeTable'][...]
    days = time_table['Offset in days']
```

#### Step 2: Date Synchronization

```python
# Collect spatial days from all files
all_spatial_days = []
for file in sr3_files:
    spatial_days = get_spatial_days(file)
    all_spatial_days.append(set(spatial_days))

# Find common days
common_days = set.intersection(*all_spatial_days)

# Filter by mode (yearly, monthly, etc.)
if mode == 'yearly':
    filtered_days = filter_yearly(common_days)
elif mode == 'monthly':
    filtered_days = filter_monthly(common_days)
```

#### Step 3: Extraction

```python
def extract_single_file_spatial(args):
    case_idx, sr3_file, file_lookup, target_days, properties, grid_dims, ipstcs, selected_layers = args
    
    # Initialize result arrays
    result = {}
    for prop in properties:
        n_selected_layers = len(selected_layers) if selected_layers else grid_dims[2]
        result[prop] = np.zeros((len(target_days), grid_dims[0], grid_dims[1], n_selected_layers))
    
    # Extract for each target day
    with h5py.File(sr3_file, 'r') as f:
        for ts_idx, target_day in enumerate(target_days):
            # Find timestep for this day
            spatial_timestep = file_lookup[target_day]['spatial_timestep']
            
            # Extract properties
            timestep_group = f[f'SpatialProperties/{spatial_timestep:06d}']
            
            for prop in properties:
                # Get 1D active cell data
                active_data = timestep_group[prop][...]
                
                # Map to 3D grid
                grid_3d = map_active_to_3d(active_data, ipstcs, grid_dims)
                
                # Slice to selected layers
                if selected_layers:
                    grid_3d = grid_3d[:, :, selected_layers]
                
                result[prop][ts_idx] = grid_3d
    
    return {'case_idx': case_idx, 'data': result}
```

#### Step 4: Saving

```python
def save_batch_spatial_to_hdf5(data_array, property_name, output_file, selected_layers):
    """
    Save batch spatial data.
    
    Args:
        data_array: Shape (N_cases, N_timesteps, Nx, Ny, Nz_selected)
        property_name: Name of property
        output_file: Output HDF5 file path
        selected_layers: List of layer indices extracted
    """
    with h5py.File(output_file, 'w') as f:
        # Save data
        f.create_dataset('data', data=data_array, compression='gzip')
        
        # Save metadata
        meta = f.create_group('metadata')
        meta.attrs['property_name'] = property_name
        meta.attrs['n_cases'] = data_array.shape[0]
        meta.attrs['n_timesteps'] = data_array.shape[1]
        meta.attrs['nx'] = data_array.shape[2]
        meta.attrs['ny'] = data_array.shape[3]
        meta.attrs['nz'] = data_array.shape[4]  # Selected layers
        meta.create_dataset('selected_layers', data=selected_layers)
```

---

## Output Format

### HDF5 File Structure

Extracted data is saved in HDF5 format with the following structure:

```
batch_spatial_properties_POROS.h5
├── data/                    # Main data array
│   └── Shape: (N_cases, N_timesteps, Nx, Ny, Nz_selected)
└── metadata/                # Metadata group
    ├── property_name        # Attribute: "POROS"
    ├── n_cases              # Attribute: Number of cases
    ├── n_timesteps          # Attribute: Number of timesteps
    ├── nx, ny, nz           # Attributes: Grid dimensions
    ├── original_nz           # Attribute: Original number of layers
    ├── selected_layers      # Dataset: Array of layer indices
    ├── timesteps            # Dataset: Timestep indices
    ├── target_days          # Dataset: Simulation days
    ├── sr3_files            # Dataset: Source file names
    └── active_cell_mapping  # Dataset: IPSTCS array
```

### Data Array Shapes

#### Spatial Properties

**Full Extraction** (all layers):
- Shape: `(N_cases, N_timesteps, Nx, Ny, Nz)`
- Example: `(14, 34, 34, 16, 25)` = 14 cases, 34 timesteps, 34×16×25 grid

**Layer Selection** (selected layers):
- Shape: `(N_cases, N_timesteps, Nx, Ny, N_selected_layers)`
- Example: `(14, 34, 34, 16, 6)` = 14 cases, 34 timesteps, 34×16×6 grid (6 selected layers)

#### Time Series

- Shape: `(N_cases, N_timesteps, N_wells)`
- Example: `(14, 33, 6)` = 14 cases, 33 timesteps, 6 wells
- Note: One file per variable

### Metadata Information

**Essential Metadata**:
- `property_name`: Name of extracted property
- `n_cases`: Number of simulation cases processed
- `n_timesteps`: Number of time points extracted
- `nx`, `ny`, `nz`: Grid dimensions (nz = selected layers if layer selection used)
- `original_nz`: Original number of layers in grid
- `selected_layers`: Array of layer indices that were extracted
- `timesteps`: Timestep indices from MasterTimeTable
- `target_days`: Simulation days corresponding to timesteps
- `sr3_files`: List of source SR3 file names
- `active_cell_mapping`: IPSTCS array for reference

**Compression**:
- Uses HDF5's gzip compression
- Chunking optimized for access patterns
- Shuffle filter enabled for better compression

---

## Key Implementation Details

### Performance Optimizations

1. **Vectorized Operations**: Uses NumPy vectorized operations for coordinate calculations
2. **Parallel Processing**: Multiprocessing for batch extraction across files
3. **Memory Efficiency**: Processes files one at a time, avoids loading all data into memory
4. **Chunking**: HDF5 chunking optimized for typical access patterns

### Error Handling

- **Missing Timesteps**: Fills with inactive cell value if timestep not found
- **Missing Properties**: Fills with inactive cell value if property not found
- **File Errors**: Continues processing other files if one fails
- **Validation**: Checks file structure before processing

### Backward Compatibility

- **Layer Selection**: If `selected_layers` is None or empty, extracts all layers (default behavior)
- **Metadata**: Old files without `selected_layers` metadata assumed to have all layers
- **Format**: Output format compatible with existing visualization tools

---

## Summary

SR3 files use a sophisticated HDF5 structure optimized for storing large-scale reservoir simulation results:

1. **Efficient Storage**: Only active cells stored, reducing file size
2. **Time Synchronization**: MasterTimeTable provides universal time reference
3. **Flexible Extraction**: Supports property selection, date filtering, and layer selection
4. **Batch Processing**: Parallel extraction across multiple simulation cases
5. **Standard Output**: HDF5 format ensures compatibility with analysis tools

The extraction mechanisms convert SR3's native 1D active cell format into standard multi-dimensional arrays suitable for visualization, analysis, and machine learning applications.

