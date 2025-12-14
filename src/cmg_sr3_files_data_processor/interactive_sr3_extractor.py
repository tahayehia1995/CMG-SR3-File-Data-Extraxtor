
#%%
#!/usr/bin/env python3
"""
Enhanced Interactive SR3 Data Extractor with Comprehensive Analysis
================================================================

🚀 ADVANCED BATCH PROCESSING DASHBOARD for CMG SR3 files with deep analysis capabilities.

🌟 NEW ENHANCED FEATURES (Version 2.0):
======================================

📊 **Comprehensive File Structure Analysis:**
   - Real-time file structure overview with detailed statistics
   - Active/inactive cell pattern comparison across all files
   - File size, timestep availability, and data consistency analysis
   - Color-coded inconsistency detection and warnings

🕐 **Advanced Multi-Mode Date Synchronization:**
   - 📅 YEARLY: January 1st dates across simulation period
   - 📆 MONTHLY: End-of-month dates synchronized across all files  
   - 📊 DAILY: Regular intervals (configurable, e.g. every 30 days)
   - 🎯 CUSTOM: Full manual selection with preview

🔄 **Enhanced Inactive Cell Management:**
   - Per-file inactive cell pattern analysis and comparison
   - 4D inactive cell arrays: (N_cases, Nx, Ny, Nz)
   - Comprehensive consistency checking and reporting
   - Detailed statistics and metadata for each case

📈 **Interactive Data Preview:**
   - Live date preview based on selected filter mode
   - Cross-file availability visualization
   - Smart default selection with temporal spread
   - Real-time synchronization status monitoring

⚡ **Improved User Experience:**
   - Enhanced dashboard with tabbed sections
   - File structure overview panel
   - Progress tracking with detailed feedback
   - Color-coded status indicators

Core Features:
=============
1. **Spatial Properties**: Extract grid properties with synchronized dates
2. **Time Series Data**: Extract well data with cross-file consistency  
3. **Inactive Cells**: Comprehensive per-case inactive cell analysis
4. **Batch Processing**: Unified output format (N_cases, N_dates, N_grid/N_wells)

Enhanced Output Structure:
========================
{output_folder}/
├── inactive_cell_locations.h5          # 4D: (N_cases, Nx, Ny, Nz) per-case analysis
├── batch_spatial_properties_POROS.h5   # 5D: (N_cases, N_dates, Nx, Ny, Nz)
├── batch_spatial_properties_PRES.h5    # 5D: (N_cases, N_dates, Nx, Ny, Nz)  
├── batch_timeseries_data_BHP.h5        # 3D: (N_cases, N_dates, N_wells)
└── ...                                  # Additional synchronized data

🎯 Perfect for machine learning datasets requiring consistent temporal sampling!

Data Access:
===========
Use load_batch_data_example() for easy data loading and access patterns.
"""

import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import warnings
import glob
import os
from collections import defaultdict
import multiprocessing as mp
from functools import partial
warnings.filterwarnings('ignore')

# === DATE-BASED EXTRACTION FUNCTIONS ===
def analyze_common_dates_across_files(sr3_files):
    """
    Find common dates separately for spatial and timeseries data across all files
    Returns extraction plan with separate spatial and timeseries date mappings
    """
    
    file_date_data = {}
    
    for sr3_file in sr3_files:
        filename = Path(sr3_file).name
        
        try:
            with h5py.File(sr3_file, 'r') as f:
                # Load master time table
                time_table = f['General/MasterTimeTable'][...]
                master_timesteps = time_table['Index']
                master_dates = time_table['Date'] 
                master_days = time_table['Offset in days']
                
                # Get spatial timesteps
                spatial_timesteps = []
                if 'SpatialProperties' in f:
                    for key in f['SpatialProperties'].keys():
                        if key.isdigit():
                            spatial_timesteps.append(int(key))
                
                # Get time series timesteps  
                timeseries_timesteps = []
                if 'TimeSeries/WELLS' in f:
                    timeseries_timesteps = list(f['TimeSeries/WELLS/Timesteps'][...])
                
                # Map timesteps to simulation days
                spatial_days = []
                for ts in spatial_timesteps:
                    if ts < len(master_days):
                        spatial_days.append(float(master_days[ts]))
                
                timeseries_days = []
                for ts in timeseries_timesteps:
                    if ts < len(master_days):
                        timeseries_days.append(float(master_days[ts]))
                
                # Store data for this file
                file_date_data[filename] = {
                    'master_timesteps': master_timesteps,
                    'master_dates': master_dates,
                    'master_days': master_days,
                    'spatial_timesteps': spatial_timesteps,
                    'spatial_days': spatial_days,
                    'timeseries_timesteps': timeseries_timesteps,
                    'timeseries_days': timeseries_days
                }
                
        except Exception as e:
            print(f"⚠️ Error analyzing {filename}: {e}")
    
    # Find common SPATIAL dates across all files
    common_spatial_days = None
    for filename, data in file_date_data.items():
        spatial_days_set = set(data['spatial_days'])
        
        if common_spatial_days is None:
            common_spatial_days = spatial_days_set
        else:
            common_spatial_days = common_spatial_days.intersection(spatial_days_set)
    
    # Find common TIMESERIES dates across all files
    common_timeseries_days = None
    for filename, data in file_date_data.items():
        timeseries_days_set = set(data['timeseries_days'])
        
        if common_timeseries_days is None:
            common_timeseries_days = timeseries_days_set
        else:
            common_timeseries_days = common_timeseries_days.intersection(timeseries_days_set)
    
    if not common_spatial_days or len(common_spatial_days) == 0:
        print("❌ No common spatial dates found across all files!")
        return None
    
    if not common_timeseries_days or len(common_timeseries_days) == 0:
        print("❌ No common timeseries dates found across all files!")
        return None
    
    # Create separate extraction plans
    common_spatial_sorted = sorted(common_spatial_days)
    common_timeseries_sorted = sorted(common_timeseries_days)
    
    spatial_extraction_plan = {}
    timeseries_extraction_plan = {}
    
    # Create spatial extraction plan
    for filename, data in file_date_data.items():
        spatial_plan = []
        for target_day in common_spatial_sorted:
            if target_day in data['spatial_days']:
                spatial_idx = data['spatial_days'].index(target_day)
                spatial_timestep = data['spatial_timesteps'][spatial_idx]
                
                spatial_plan.append({
                    'target_day': target_day,
                    'spatial_timestep': spatial_timestep
                })
        
        spatial_extraction_plan[filename] = spatial_plan
    
    # Create timeseries extraction plan
    for filename, data in file_date_data.items():
        timeseries_plan = []
        for target_day in common_timeseries_sorted:
            if target_day in data['timeseries_days']:
                ts_idx = data['timeseries_days'].index(target_day)
                ts_timestep = data['timeseries_timesteps'][ts_idx]
                
                timeseries_plan.append({
                    'target_day': target_day,
                    'timeseries_timestep': ts_timestep
                })
        
        timeseries_extraction_plan[filename] = timeseries_plan
    
    print(f"✅ Found {len(common_spatial_sorted)} common spatial dates")
    print(f"✅ Found {len(common_timeseries_sorted)} common timeseries dates")
    
    return {
        'common_spatial_days': common_spatial_sorted,
        'common_timeseries_days': common_timeseries_sorted,
        'spatial_extraction_plan': spatial_extraction_plan,
        'timeseries_extraction_plan': timeseries_extraction_plan,
        'file_data': file_date_data
    }

def convert_days_to_dates(days_list, master_dates, master_days):
    """Convert simulation days to human-readable dates"""
    dates = []
    for day_val in days_list:
        # Find closest day in master table
        day_idx = np.argmin(np.abs(master_days - day_val))
        date_val = master_dates[day_idx]
        
        if date_val > 20000000:
            date_str = str(int(date_val))
            if len(date_str) >= 8:
                year = int(date_str[:4])
                month = int(date_str[4:6])
                day = int(date_str[6:8])
                try:
                    date_obj = datetime(year, month, day)
                    dates.append(date_obj.strftime('%Y-%m-%d'))
                except:
                    dates.append(f"Day {day_val:.0f}")
            else:
                dates.append(f"Day {day_val:.0f}")
        else:
            dates.append(f"Day {day_val:.0f}")
    
    return dates

def filter_monthly_dates_flexible(common_days, master_dates, master_days):
    """
    Filter common dates to show closest date to the 1st of each month
    
    This function provides optimal monthly coverage by:
    1. Converting all common simulation days to calendar dates
    2. Grouping dates by year-month
    3. Selecting the date closest to the 1st of each month
    """
    monthly_days = []
    monthly_dates = []
    
    # Convert all common days to datetime objects
    day_date_pairs = []
    for day_val in sorted(common_days):
        # Find corresponding date in master table
        day_idx = np.argmin(np.abs(master_days - day_val))
        date_val = master_dates[day_idx]
        
        # Convert date format 20250101.0 → datetime
        if date_val > 20000000:
            date_str = str(int(date_val))
            if len(date_str) >= 8:
                year = int(date_str[:4])
                month = int(date_str[4:6])  
                day = int(date_str[6:8])
                try:
                    date_obj = datetime(year, month, day)
                    day_date_pairs.append((day_val, date_obj))
                except ValueError:
                    continue
    
    # Group by year-month
    monthly_groups = defaultdict(list)
    
    for day_val, date_obj in day_date_pairs:
        year_month = (date_obj.year, date_obj.month)
        monthly_groups[year_month].append((day_val, date_obj))
    
    # For each month, find the date closest to the 1st
    for (year, month), dates_in_month in sorted(monthly_groups.items()):
        target_date = datetime(year, month, 1)
        
        # Find closest date to the 1st of the month
        closest_date = min(dates_in_month, 
                          key=lambda x: abs((x[1] - target_date).days))
        
        monthly_days.append(closest_date[0])
        monthly_dates.append(closest_date[1].strftime('%Y-%m-%d'))
    
    return monthly_days, monthly_dates

def filter_yearly_dates(common_days, master_dates, master_days):
    """
    Filter common dates to show only January 1st of each year (current behavior)
    """
    yearly_days = []
    yearly_dates = []
    
    for day_val in sorted(common_days):
        day_idx = np.argmin(np.abs(master_days - day_val))
        date_val = master_dates[day_idx]
        
        if date_val > 20000000:
            date_str = str(int(date_val))
            if len(date_str) >= 8:
                year = int(date_str[:4])
                month = int(date_str[4:6])  
                day = int(date_str[6:8])
                try:
                    date_obj = datetime(year, month, day)
                    
                    # Only include January 1st
                    if date_obj.month == 1 and date_obj.day == 1:
                        yearly_days.append(day_val)
                        yearly_dates.append(date_obj.strftime('%Y-%m-%d'))
                except ValueError:
                    continue
    
    return yearly_days, yearly_dates

def filter_common_calendar_months(sr3_files, time_mapping, common_timeseries_days):
    """
    Filter to show calendar months consistent with yearly filtering
    
    This function ensures:
    1. Uses the same year coverage as yearly filtering (based on common dates)
    2. For each year with common dates, shows available months
    3. Guarantees at least January for each year (like yearly filtering)
    """
    from collections import defaultdict
    import h5py
    
    # Step 1: First determine which years have common dates (like yearly filtering does)
    yearly_days, yearly_dates = filter_yearly_dates(
        common_timeseries_days, time_mapping['dates'], time_mapping['days']
    )
    
    # Extract years that have common dates
    common_years = set()
    for date in yearly_dates:
        year = date[:4]
        common_years.add(year)
    
    common_years = sorted(list(common_years))
    print(f"   ✅ Using same year coverage as yearly filtering: {len(common_years)} years ({common_years[0]}-{common_years[-1]})")
    
    # Step 2: For each file, find available months within these common years
    file_months = []  # Months available in each file (within common years only)
    
    for sr3_file in sr3_files:
        try:
            with h5py.File(sr3_file, 'r') as f:
                file_month_set = set()
                
                # Get timeseries timesteps
                if 'TimeSeries/WELLS' in f:
                    timeseries_data = f['TimeSeries/WELLS/Data'][...]
                    n_timesteps = timeseries_data.shape[0]
                    
                    # Convert timesteps to months (only for common years)
                    for ts in range(1, min(n_timesteps + 1, len(time_mapping['days']))):
                        day_val = time_mapping['days'][ts]
                        
                        # Find corresponding date
                        day_idx = np.argmin(np.abs(time_mapping['days'] - day_val))
                        date_val = time_mapping['dates'][day_idx]
                        
                        # Convert to YYYYMM format
                        if date_val > 20000000:
                            date_str = str(int(date_val))
                            if len(date_str) >= 6:
                                year = date_str[:4]
                                # Only include months from common years
                                if year in common_years:
                                    year_month = date_str[:6]  # YYYYMM format
                                    file_month_set.add(year_month)
                
                file_months.append(file_month_set)
                print(f"   📄 {sr3_file.name}: {len(file_month_set)} months in common years")
                
        except Exception as e:
            print(f"   ❌ Error reading {sr3_file.name}: {e}")
            continue
    
    if not file_months:
        return [], []
    
    # Step 3: Find months that exist in ALL files (within common years)
    common_months_in_years = file_months[0]
    for month_set in file_months[1:]:
        common_months_in_years = common_months_in_years.intersection(month_set)
    
    # Step 4: Ensure at least January for each common year (like yearly filtering)
    guaranteed_months = set()
    for year in common_years:
        jan_month = f"{year}01"  # January of each year
        guaranteed_months.add(jan_month)
    
    # Step 5: Combine guaranteed months with other common months
    final_months = guaranteed_months.union(common_months_in_years)
    
    # Filter to only include months from common years (redundant but safe)
    final_months = [month for month in final_months if month[:4] in common_years]
    final_months = sorted(final_months)
    
    print(f"   ✅ Selected {len(final_months)} months (guaranteed coverage for all {len(common_years)} years)")
    
    # Step 6: For each selected month, find representative date from reference file
    reference_file = sr3_files[0]
    monthly_days = []
    monthly_dates = []
    
    try:
        with h5py.File(reference_file, 'r') as f:
            # Group reference file dates by month
            month_to_dates = defaultdict(list)
            
            if 'TimeSeries/WELLS' in f:
                timeseries_data = f['TimeSeries/WELLS/Data'][...]
                n_timesteps = timeseries_data.shape[0]
                
                for ts in range(1, min(n_timesteps + 1, len(time_mapping['days']))):
                    day_val = time_mapping['days'][ts]
                    
                    # Find corresponding date
                    day_idx = np.argmin(np.abs(time_mapping['days'] - day_val))
                    date_val = time_mapping['dates'][day_idx]
                    
                    if date_val > 20000000:
                        date_str = str(int(date_val))
                        if len(date_str) >= 8:
                            year_month = date_str[:6]  # YYYYMM
                            
                            # Include if this month is in our selected list
                            if year_month in final_months:
                                year = int(date_str[:4])
                                month = int(date_str[4:6])
                                day = int(date_str[6:8])
                                
                                try:
                                    date_obj = datetime(year, month, day)
                                    month_to_dates[year_month].append((day_val, date_obj))
                                except ValueError:
                                    continue
            
            # Step 7: Select representative date for each month
            for year_month in final_months:
                if year_month in month_to_dates:
                    dates_in_month = month_to_dates[year_month]
                    
                    # Find date closest to 1st of month
                    year = int(year_month[:4])
                    month = int(year_month[4:6])
                    target_date = datetime(year, month, 1)
                    
                    closest_date = min(dates_in_month, 
                                     key=lambda x: abs((x[1] - target_date).days))
                    
                    monthly_days.append(closest_date[0])
                    monthly_dates.append(closest_date[1].strftime('%Y-%m-%d'))
    
    except Exception as e:
        print(f"   ❌ Error processing reference file: {e}")
        return [], []
    
    return monthly_days, monthly_dates

# === OPTIMIZATION HELPER FUNCTIONS ===
def _get_optimal_compression(data_size_bytes):
    """
    Determine optimal compression level based on data size.
    
    Args:
        data_size_bytes: Size of data in bytes
        
    Returns:
        tuple: (compression_type, compression_level)
    """
    # Convert to MB for easier comparison
    size_mb = data_size_bytes / (1024 * 1024)
    
    if size_mb > 1024:  # > 1GB: Use lower compression for speed
        return ('gzip', 1)
    elif size_mb > 100:  # 100MB - 1GB: Balanced compression
        return ('gzip', 4)
    else:  # < 100MB: Maximum compression
        return ('gzip', 9)

def _calculate_chunk_shape(array_shape, dtype=np.float32):
    """
    Calculate optimal chunk shape for HDF5 dataset.
    
    Args:
        array_shape: Shape of the array
        dtype: Data type of the array
        
    Returns:
        tuple: Optimal chunk shape
    """
    # Target chunk size: ~1MB per chunk (good balance for I/O)
    target_chunk_size = 1024 * 1024  # 1MB
    element_size = np.dtype(dtype).itemsize
    
    # For 5D arrays (spatial): (N_cases, N_timesteps, Nx, Ny, Nz)
    # For 3D arrays (timeseries): (N_cases, N_timesteps, N_wells)
    if len(array_shape) == 5:
        # Spatial data: chunk along case and timestep dimensions
        # Prefer chunking smaller dimensions together
        n_cases, n_timesteps, nx, ny, nz = array_shape
        
        # Calculate chunks for spatial dimensions (keep together)
        spatial_size = nx * ny * nz * element_size
        if spatial_size <= target_chunk_size:
            # Can fit entire spatial slice
            chunk_shape = (1, 1, nx, ny, nz)
        else:
            # Need to chunk spatial dimensions
            # Chunk along Z dimension first (typically smallest)
            chunk_z = min(nz, max(1, target_chunk_size // (nx * ny * element_size)))
            chunk_shape = (1, 1, nx, ny, chunk_z)
    
    elif len(array_shape) == 3:
        # Timeseries data: chunk along case and timestep
        n_cases, n_timesteps, n_wells = array_shape
        
        # Calculate optimal chunking
        timestep_size = n_wells * element_size
        if timestep_size <= target_chunk_size:
            # Can fit multiple timesteps
            chunk_timesteps = min(n_timesteps, max(1, target_chunk_size // timestep_size))
            chunk_shape = (1, chunk_timesteps, n_wells)
        else:
            # Must chunk wells dimension
            chunk_wells = min(n_wells, max(1, target_chunk_size // element_size))
            chunk_shape = (1, 1, chunk_wells)
    
    else:
        # Default: use auto-chunking
        return True
    
    return chunk_shape

def _prepare_extraction_lookup(extraction_plan):
    """
    Convert extraction plan list to dictionary for O(1) lookup.
    
    Args:
        extraction_plan: Dictionary mapping filename to list of plan entries
        
    Returns:
        Dictionary mapping filename to dictionary mapping target_day to plan_entry
    """
    lookup = {}
    for filename, plan_list in extraction_plan.items():
        lookup[filename] = {entry['target_day']: entry for entry in plan_list}
    return lookup

def _map_active_to_3d_vectorized(active_data, active_cell_mapping, grid_dims, inactive_value=0.0):
    """Vectorized mapping of 1D active cell data to 3D grid (top-level function for multiprocessing)"""
    ni, nj, nk = grid_dims
    grid_3d = np.full((ni, nj, nk), inactive_value, dtype=np.float32)
    
    # Vectorized approach: filter valid mappings and use advanced indexing
    valid_mask = (active_cell_mapping > 0) & (np.arange(len(active_cell_mapping)) < len(active_data))
    valid_mappings = active_cell_mapping[valid_mask]
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_mappings) == 0:
        return grid_3d
    
    # Convert 1D grid indices to 0-based
    grid_indices = valid_mappings - 1
    
    # Vectorized I-J-K coordinate calculation
    k = grid_indices // (ni * nj)
    j = (grid_indices % (ni * nj)) // ni
    i = grid_indices % ni
    
    # Filter valid coordinates
    valid_coords = (k < nk) & (j < nj) & (i < ni)
    i_valid = i[valid_coords]
    j_valid = j[valid_coords]
    k_valid = k[valid_coords]
    data_valid = active_data[valid_indices[valid_coords]]
    
    # Use advanced indexing for bulk assignment
    grid_3d[i_valid, j_valid, k_valid] = data_valid
    
    return grid_3d

def _extract_single_file_spatial(args):
    """
    Worker function to extract spatial data from a single SR3 file.
    This is a top-level function for multiprocessing compatibility.
    
    Args:
        args: Tuple containing:
            - case_idx: Index of the case
            - sr3_file: Path to SR3 file
            - filename: Filename
            - file_lookup: Dictionary mapping target_day to plan_entry
            - target_days: List of target days to extract
            - selected_properties: List of properties to extract
            - grid_dims: Tuple of (nx, ny, nz)
            - active_cell_mapping: Array of active cell mappings
            - inactive_value: Value to use for inactive cells
            - selected_layers: List of layer indices to extract (None or empty = all layers)
            
    Returns:
        Dictionary with 'case_idx', 'success', and 'data' (dict of property arrays)
    """
    case_idx, sr3_file, filename, file_lookup, target_days, selected_properties, grid_dims, active_cell_mapping, inactive_value, selected_layers = args
    
    nx, ny, nz = grid_dims
    n_timesteps = len(target_days)
    
    # Handle layer selection: if None or empty, extract all layers
    if selected_layers is None or len(selected_layers) == 0:
        selected_layers = list(range(nz))
    
    n_selected_layers = len(selected_layers)
    
    # Initialize result data arrays with selected layers only
    result_data = {}
    for prop in selected_properties:
        result_data[prop] = np.full((n_timesteps, nx, ny, n_selected_layers), inactive_value, dtype=np.float32)
    
    case_successful = True
    
    try:
        with h5py.File(sr3_file, 'r') as f:
            for ts_idx, target_day in enumerate(target_days):
                # Use O(1) dictionary lookup - try direct match first
                plan_entry = file_lookup.get(target_day)
                
                # If no direct match, try with small tolerance (should be rare)
                if plan_entry is None:
                    for day_key in file_lookup.keys():
                        if abs(day_key - target_day) < 0.1:  # Small tolerance for float comparison
                            plan_entry = file_lookup[day_key]
                            break
                
                if not plan_entry:
                    # Fill with inactive cell value for this timestep (already correct shape)
                    for prop in selected_properties:
                        result_data[prop][ts_idx, :, :, :] = inactive_value
                    case_successful = False
                    continue
                
                # Use the mapped timestep for this file
                spatial_timestep = plan_entry['spatial_timestep']
                
                if f'SpatialProperties/{spatial_timestep:06d}' not in f:
                    # Fill with inactive cell value (already correct shape)
                    for prop in selected_properties:
                        result_data[prop][ts_idx, :, :, :] = inactive_value
                    case_successful = False
                    continue
                
                spatial_group = f[f'SpatialProperties/{spatial_timestep:06d}']
                
                for prop in selected_properties:
                    if prop in spatial_group:
                        # Get 1D active cell data
                        active_data = spatial_group[prop][...]
                        
                        # Map to 3D grid with user-specified inactive cell value
                        grid_3d = _map_active_to_3d_vectorized(active_data, active_cell_mapping, grid_dims, inactive_value)
                        
                        # Slice to selected layers only
                        grid_3d_selected = grid_3d[:, :, selected_layers]
                        result_data[prop][ts_idx, :, :, :] = grid_3d_selected
                    else:
                        # Property not found, fill with inactive cell value
                        result_data[prop][ts_idx, :, :, :] = inactive_value
                        case_successful = False
    
    except Exception as e:
        # Fill with inactive cell value for this case (already correct shape)
        for prop in selected_properties:
            result_data[prop][:, :, :, :] = inactive_value
        case_successful = False
    
    return {
        'case_idx': case_idx,
        'success': case_successful,
        'data': result_data
    }

def _extract_single_file_timeseries(args):
    """
    Worker function to extract timeseries data from a single SR3 file.
    This is a top-level function for multiprocessing compatibility.
    
    Args:
        args: Tuple containing:
            - case_idx: Index of the case
            - sr3_file: Path to SR3 file
            - filename: Filename
            - file_lookup: Dictionary mapping target_day to plan_entry
            - target_days: List of target days to extract
            - selected_variables: List of variables to extract
            - n_wells: Number of wells
            
    Returns:
        Dictionary with 'case_idx', 'success', and 'data' (dict of variable arrays)
    """
    case_idx, sr3_file, filename, file_lookup, target_days, selected_variables, n_wells = args
    
    n_timesteps = len(target_days)
    
    # Initialize result data arrays
    result_data = {}
    for var in selected_variables:
        result_data[var] = np.zeros((n_timesteps, n_wells), dtype=np.float32)
    
    case_successful = True
    
    try:
        with h5py.File(sr3_file, 'r') as f:
            # Load well data for this file
            well_timesteps = f['TimeSeries/WELLS/Timesteps'][...]
            well_data_array = f['TimeSeries/WELLS/Data'][...]
            well_variables = [var.decode() if isinstance(var, bytes) else str(var)
                            for var in f['TimeSeries/WELLS/Variables'][...]]
            
            # Find variable indices
            var_indices = {}
            for var in selected_variables:
                for i, available_var in enumerate(well_variables):
                    if var == available_var:
                        var_indices[var] = i
                        break
            
            # Process each target day
            for ts_idx, target_day in enumerate(target_days):
                # Use O(1) dictionary lookup - try direct match first
                plan_entry = file_lookup.get(target_day)
                
                # If no direct match, try with small tolerance (should be rare)
                if plan_entry is None:
                    for day_key in file_lookup.keys():
                        if abs(day_key - target_day) < 0.1:  # Small tolerance for float comparison
                            plan_entry = file_lookup[day_key]
                            break
                
                if not plan_entry:
                    # Fill with zeros for this timestep
                    for var in selected_variables:
                        result_data[var][ts_idx, :] = 0.0
                    case_successful = False
                    continue
                
                # Use the mapped timestep for this file
                ts_timestep = plan_entry['timeseries_timestep']
                
                # Find timestep index in well_timesteps array
                well_ts_idx = -1
                for i, ts in enumerate(well_timesteps):
                    if ts == ts_timestep:
                        well_ts_idx = i
                        break
                
                if well_ts_idx < 0:
                    # Fill with zeros
                    for var in selected_variables:
                        result_data[var][ts_idx, :] = 0.0
                    case_successful = False
                    continue
                
                # Extract data for each variable
                for var in selected_variables:
                    if var in var_indices:
                        var_idx = var_indices[var]
                        # Extract data for all wells at this timestep
                        data = well_data_array[well_ts_idx, var_idx, :]
                        result_data[var][ts_idx, :] = data
                    else:
                        # Variable not found, fill with zeros
                        result_data[var][ts_idx, :] = 0.0
                        case_successful = False
    
    except Exception as e:
        # Fill with zeros for this case
        for var in selected_variables:
            result_data[var][:, :] = 0.0
        case_successful = False
    
    return {
        'case_idx': case_idx,
        'success': case_successful,
        'data': result_data
    }

class BatchSR3Extractor:
    """
    Batch-processing interactive extractor for multiple SR3 files
    """
    
    def __init__(self):
        """Initialize the batch extractor"""
        self.directory_path = None
        self.sr3_files = []
        self.reference_file = None  # First file used for UI initialization
        self.spatial_properties = []
        self.spatial_timesteps = []
        self.time_mapping = {}
        self.well_variables = []
        self.well_names = []
        self.grid_dims = None
        self.active_cell_mapping = None
        
        # Enhanced file analysis data
        self.file_analyses = {}  # Store detailed analysis per file
        self.common_dates = {}   # Store common dates for different frequencies
        self.inactive_cells_data = {}  # Store inactive cell info per file
        self.file_structure_overview = {}  # Store file structure details
        self.date_sync_status = False  # Track if files are synchronized
        
        # Create UI components
        self.create_ui_components()
        self.create_layout()
    
    def create_ui_components(self):
        """Create all UI widgets"""
        
        # Directory selection
        self.directory_path_widget = widgets.Text(
            value=".",
            description="Directory:",
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px')
        )
        
        self.load_button = widgets.Button(
            description="🔍 Load Directory",
            button_style='primary',
            layout=widgets.Layout(width='150px')
        )
        
        self.status_output = widgets.Output()
        
        # Tab selection
        self.tab_selection = widgets.Tab()
        
        # Spatial properties tab widgets
        self.spatial_props_widget = widgets.SelectMultiple(
            options=[],
            description="Properties:",
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px', height='200px')
        )
        
        self.spatial_timesteps_widget = widgets.SelectMultiple(
            options=[],
            description="Dates:",
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px', height='200px')
        )
        
        self.spatial_output_widget = widgets.Text(
            value="batch_spatial_properties",
            description="Output Prefix:",
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='350px')
        )
        
        self.extract_spatial_button = widgets.Button(
            description="🚀 Batch Extract Spatial",
            button_style='success',
            layout=widgets.Layout(width='200px')
        )
        
        # Layer selection widget (will be populated after grid dimensions are known)
        self.spatial_layers_widget = widgets.SelectMultiple(
            options=[],
            description="Layers (Nz):",
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px', height='200px')
        )
        
        # Time series tab widgets
        self.timeseries_vars_widget = widgets.SelectMultiple(
            options=[],
            description="Variables:",
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px', height='400px')
        )
        
        self.timeseries_dates_widget = widgets.SelectMultiple(
            options=[],
            description="Dates:",
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px', height='200px')
        )
        
        self.timeseries_output_widget = widgets.Text(
            value="batch_timeseries_data",
            description="Output Prefix:",
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='350px')
        )
        
        self.extract_timeseries_button = widgets.Button(
            description="🚀 Batch Extract Time Series",
            button_style='success',
            layout=widgets.Layout(width='200px')
        )
        
        # Output folder widget
        self.output_folder_widget = widgets.Text(
            value="sr3_batch_output",
            description="Output Folder:",
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px')
        )
        
        self.create_inactive_cells_widget = widgets.Checkbox(
            value=True,
            description="Extract Inactive Cell Locations",
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px')
        )
        
        self.create_well_locations_widget = widgets.Checkbox(
            value=False,
            description="Extract Well Locations",
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px')
        )
        
        # Enhanced date filtering options
        self.date_filter_mode = widgets.ToggleButtons(
            options=[
                ('📅 Yearly (Jan 1st)', 'yearly'),
                ('📆 Monthly (End of Month)', 'monthly'), 
                ('📊 Daily (Regular Intervals)', 'daily'),
                ('🎯 Custom Selection', 'custom')
            ],
            value='yearly',
            description='Date Filter:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='600px')
        )
        
        self.daily_interval_widget = widgets.IntText(
            value=30,
            description='Daily Interval (days):',
            min=1,
            max=365,
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )
        
        # Inactive cells value input
        self.inactive_cells_value_widget = widgets.FloatText(
            value=0.0,
            description='Inactive cells value:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px'),
            step=0.1,
            tooltip='Value to assign to inactive cells in extracted data (default: 0.0, can use NaN, -999, etc.)'
        )
        
        self.date_preview_widget = widgets.HTML(
            value="<i>Select files first to preview available dates</i>",
            layout=widgets.Layout(height='200px', overflow='auto')
        )
        
        self.update_dates_button = widgets.Button(
            description="🔄 Analyze & Sync Dates",
            button_style='info',
            layout=widgets.Layout(width='200px')
        )
        
        # File structure overview widget
        self.file_structure_widget = widgets.HTML(
            value="<i>Load files to see detailed structure analysis</i>",
            layout=widgets.Layout(height='300px', overflow='auto')
        )
        
        # Progress and output widgets
        self.progress_widget = widgets.IntProgress(
            value=0,
            min=0,
            max=100,
            description='Progress:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px')
        )
        
        self.batch_progress_widget = widgets.IntProgress(
            value=0,
            min=0,
            max=100,
            description='Batch Progress:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px')
        )
        
        self.extraction_output = widgets.Output()
        
        # File info widget
        self.file_info_widget = widgets.HTML()
        
        # Unit conversion widgets
        self.conversion_directory_widget = widgets.Text(
            value='sr3_batch_output',
            description='Directory:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px'),
            placeholder='Path to directory containing .h5 files'
        )
        
        self.load_conversion_files_button = widgets.Button(
            description="📁 Load H5 Files",
            button_style='info',
            layout=widgets.Layout(width='150px')
        )
        
        self.conversion_status_output = widgets.Output(layout=widgets.Layout(height='100px'))
        
        self.conversion_files_widget = widgets.VBox(layout=widgets.Layout(height='400px', overflow='auto'))
        
        self.convert_units_button = widgets.Button(
            description="🔄 Convert Units",
            button_style='success',
            layout=widgets.Layout(width='150px')
        )
        
        self.reset_factors_button = widgets.Button(
            description="🔄 Reset to Defaults",
            button_style='warning',
            layout=widgets.Layout(width='150px')
        )
        
        self.conversion_progress_widget = widgets.IntProgress(
            value=0,
            min=0,
            max=100,
            description='Conversion:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px')
        )
        
        self.conversion_output = widgets.Output()
        
        # Storage for conversion factor widgets
        self.conversion_factor_widgets = {}
        
        # Date range reference widgets for pre-check analysis
        self.reference_start_date_widget = widgets.DatePicker(
            value=datetime(2025, 1, 1).date(),
            description='Reference Start:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px'),
            disabled=False
        )
        
        self.reference_end_date_widget = widgets.DatePicker(
            value=datetime(2055, 12, 31).date(),
            description='Reference End:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px'),
            disabled=False
        )
        
        self.rerun_date_analysis_button = widgets.Button(
            description="🔄 Re-run Date Analysis",
            button_style='info',
            layout=widgets.Layout(width='180px'),
            tooltip='Re-analyze date ranges with new reference dates'
        )
        
        # Event handlers
        self.load_button.on_click(self.on_load_directory)
        self.extract_spatial_button.on_click(self.on_batch_extract_spatial)
        self.extract_timeseries_button.on_click(self.on_batch_extract_timeseries)
        self.update_dates_button.on_click(self.on_update_date_filtering)
        self.date_filter_mode.observe(self.on_date_filter_change, names='value')
        self.load_conversion_files_button.on_click(self.on_load_conversion_files)
        self.convert_units_button.on_click(self.on_convert_units)
        self.reset_factors_button.on_click(self.on_reset_conversion_factors)
        self.output_folder_widget.observe(self.on_output_folder_change, names='value')
        self.rerun_date_analysis_button.on_click(self.on_rerun_date_analysis)
    
    def create_layout(self):
        """Create the main layout"""
        
        # Enhanced date filtering controls section
        date_filtering_section = widgets.VBox([
            widgets.HTML("<h3>🕐 Advanced Date Filtering & Synchronization</h3>"),
            widgets.HTML("<p><i>Choose extraction frequency that will be synchronized across ALL .sr3 files</i></p>"),
            self.date_filter_mode,
            widgets.HBox([
                self.daily_interval_widget,
                self.inactive_cells_value_widget,
                self.update_dates_button
            ]),
            widgets.HTML("<div style='margin-top:10px'><b>📅 Yearly:</b> January 1st of each year<br/>"
                        "<b>📆 Monthly:</b> End of each month (common across files)<br/>"
                        "<b>📊 Daily:</b> Regular intervals (e.g., every 30 days)<br/>"
                        "<b>🎯 Custom:</b> Manually select specific dates<br/>"
                        "<b>🔢 Inactive Cell Value:</b> Value assigned to inactive cells (0.0=zero, NaN=missing, -999=custom)</div>"),
            widgets.HTML("<h4>📋 Date Preview & Availability:</h4>"),
            self.date_preview_widget
        ], layout=widgets.Layout(padding='15px', border='2px solid #ddd', margin='10px 0'))
        
        # File structure overview section
        file_structure_section = widgets.VBox([
            widgets.HTML("<h3>🏗️ File Structure Analysis</h3>"),
            widgets.HTML("<p><i>Detailed analysis of loaded .sr3 files including active/inactive cells and temporal patterns</i></p>"),
            self.file_structure_widget
        ], layout=widgets.Layout(padding='15px', border='2px solid #e6f3ff', margin='10px 0'))
        
        # Date range reference section
        date_reference_section = widgets.VBox([
            widgets.HTML("<h3>📅 Date Range Reference (Pre-check)</h3>"),
            widgets.HTML("<p><i>Set reference date range for analyzing SR3 file consistency</i></p>"),
            widgets.HBox([
                self.reference_start_date_widget,
                self.reference_end_date_widget,
                self.rerun_date_analysis_button
            ]),
            widgets.HTML("<small><i>💡 Files matching this range are considered consistent for batch processing</i></small>")
        ], layout=widgets.Layout(padding='15px', border='2px solid #f0e68c', margin='10px 0'))

        # File loading section
        file_section = widgets.VBox([
            widgets.HTML("<h2>🔍 Enhanced Batch SR3 Data Extractor</h2>"),
            widgets.HTML("<h3>📁 Directory Selection</h3>"),
            widgets.HTML("<p><i>Select a directory containing multiple .sr3 files for comprehensive batch processing</i></p>"),
            widgets.HBox([self.directory_path_widget, self.load_button]),
            self.status_output,
            date_reference_section,
            file_structure_section,
            date_filtering_section,
            self.file_info_widget
        ])
        
        # Spatial properties tab
        spatial_tab = widgets.VBox([
            widgets.HTML("<h3>🗺️ Batch Spatial Properties Extraction</h3>"),
            widgets.HTML("<p>Extract grid properties from ALL .sr3 files and combine into batch format.</p>"),
            widgets.HTML("<p><b>Output format:</b> (N_cases, N_timesteps, Nx, Ny, Nz) per property</p>"),
            widgets.HTML("<p><i>💡 Leave layers empty to extract all layers, or select specific layers to reduce file size</i></p>"),
            widgets.HBox([
                widgets.VBox([
                    widgets.HTML("<b>📋 Select Properties:</b>"),
                    self.spatial_props_widget
                ]),
                widgets.VBox([
                    widgets.HTML("<b>📅 Select Dates:</b>"),
                    self.spatial_timesteps_widget
                ]),
                widgets.VBox([
                    widgets.HTML("<b>📊 Select Layers (Nz):</b>"),
                    self.spatial_layers_widget
                ])
            ]),
            widgets.HBox([self.spatial_output_widget, self.extract_spatial_button]),
            widgets.HBox([self.output_folder_widget, self.create_inactive_cells_widget, self.create_well_locations_widget]),
            self.batch_progress_widget,
            self.progress_widget,
            self.extraction_output
        ])
        
        # Time series tab
        timeseries_tab = widgets.VBox([
            widgets.HTML("<h3>🛢️ Batch Time Series Data Extraction</h3>"),
            widgets.HTML("<p>Extract well variables from ALL .sr3 files and combine into batch format.</p>"),
            widgets.HTML("<p><b>Output format:</b> (N_cases, N_timesteps, N_wells) per variable</p>"),
            widgets.HBox([
                widgets.VBox([
                    widgets.HTML("<b>📊 Select Variables:</b>"),
                    self.timeseries_vars_widget
                ]),
                widgets.VBox([
                    widgets.HTML("<b>📅 Select Dates:</b>"),
                    self.timeseries_dates_widget
                ])
            ]),
            widgets.HBox([self.timeseries_output_widget, self.extract_timeseries_button]),
            widgets.HBox([self.output_folder_widget]),
            self.batch_progress_widget,
            self.progress_widget,
            self.extraction_output
        ])
        
        # Unit conversion tab
        unit_conversion_tab = widgets.VBox([
            widgets.HTML("<h3>🔧 Unit Conversion</h3>"),
            widgets.HTML("<p>Convert units in extracted .h5 files with custom conversion factors.</p>"),
            widgets.HTML("<p><b>Note:</b> Files will be converted in-place with the same names.</p>"),
            widgets.HBox([self.conversion_directory_widget, self.load_conversion_files_button]),
            self.conversion_status_output,
            widgets.HTML("<h4>📋 Available Files & Conversion Factors:</h4>"),
            self.conversion_files_widget,
            widgets.HBox([self.convert_units_button, self.reset_factors_button]),
            self.conversion_progress_widget,
            self.conversion_output
        ])

        # Create tabs
        self.tab_selection.children = [spatial_tab, timeseries_tab, unit_conversion_tab]
        self.tab_selection.titles = ['🗺️ Batch Spatial Properties', '🛢️ Batch Time Series Data', '🔧 Unit Conversion']
        
        # Add tab change observer to auto-load conversion files
        self.tab_selection.observe(self.on_tab_change, names='selected_index')
        
        # Main layout
        self.main_layout = widgets.VBox([
            file_section,
            widgets.HTML("<hr>"),
            self.tab_selection
        ])
    
    def display(self):
        """Display the interactive dashboard"""
        display(self.main_layout)
    
    def on_date_filter_change(self, change):
        """Handle date filter mode change"""
        if hasattr(self, 'sr3_files') and self.sr3_files:
            self.update_date_preview()
    
    def on_tab_change(self, change):
        """Handle tab change - auto-load conversion files when switching to unit conversion tab"""
        if change['new'] == 2:  # Unit conversion tab index
            # Auto-load conversion files if directory exists and no files loaded yet
            if (not self.conversion_factor_widgets and 
                self.conversion_directory_widget.value and 
                Path(self.conversion_directory_widget.value).exists()):
                self.on_load_conversion_files(None)
    
    def on_update_date_filtering(self, button):
        """Comprehensive date analysis and synchronization across all files"""
        with self.status_output:
            clear_output()
            
            if not self.sr3_files:
                print("❌ Please load SR3 files first")
                return
            
            print("🔄 Performing comprehensive date analysis across all files...")
            
            # Perform deep analysis of all files
            self.analyze_all_files()
            
            # Find common dates based on selected mode
            self.synchronize_dates()
            
            # Update UI with synchronized dates
            self.update_ui_options()
            
            # Update file structure overview
            self.update_file_structure_overview()
            
            # Update date preview
            self.update_date_preview()
            
            print("✅ Date analysis and synchronization completed")
    
    def on_load_directory(self, button):
        """Handle directory loading"""
        with self.status_output:
            clear_output()
            print("🔍 Loading directory...")
            
            success = self.load_sr3_directory(self.directory_path_widget.value)
            if success:
                print("✅ Directory loaded successfully!")
                print(f"📁 Found {len(self.sr3_files)} .sr3 files")
                print(f"🎯 Using {self.reference_file.name} as reference for UI")
                
                # Clear existing options first
                self.timeseries_dates_widget.options = []
                self.spatial_timesteps_widget.options = []
                self.spatial_props_widget.options = []
                self.timeseries_vars_widget.options = []
                
                # Update UI with reference file data
                self.update_ui_options()
                self.display_file_info()
                print("✅ Dashboard updated with reference file data!")
                    
            else:
                print("❌ Failed to load directory")
    
    def load_sr3_directory(self, directory_path):
        """Load directory and find all SR3 files with comprehensive pre-checks"""
        self.directory_path = Path(directory_path)
        
        if not self.directory_path.exists():
            print(f"❌ Directory not found: {directory_path}")
            return False
        
        if not self.directory_path.is_dir():
            print(f"❌ Path is not a directory: {directory_path}")
            return False
        
        # Find all .sr3 files in directory
        sr3_pattern = str(self.directory_path / "*.sr3")
        sr3_files = glob.glob(sr3_pattern)
        
        if not sr3_files:
            print(f"❌ No .sr3 files found in directory: {directory_path}")
            return False
        
        # Convert to Path objects and sort
        self.sr3_files = [Path(f) for f in sorted(sr3_files)]
        self.reference_file = self.sr3_files[0]
        
        print(f"✅ Found {len(self.sr3_files)} SR3 files")
        
        # Pre-check: Analyze date ranges across all files
        print(f"\n🔍 Pre-check: Analyzing date ranges across all {len(self.sr3_files)} files...")
        date_analysis_results = self._analyze_sr3_date_ranges()
        
        if date_analysis_results:
            self._display_date_analysis_results(date_analysis_results)
            # Store results for potential re-analysis
            self.last_date_analysis = date_analysis_results
        
        # Load reference file to initialize UI
        return self.load_reference_file()
    
    def load_reference_file(self):
        """Load all files to find common dates and initialize UI with date-based options"""
        try:
            # Analyze common dates across all files
            print("🗓️ Analyzing common dates across all SR3 files...")
            self.date_analysis = analyze_common_dates_across_files(self.sr3_files)
            
            if not self.date_analysis:
                print("❌ No common dates found - extraction will not be reliable")
                return False
            
            common_spatial_days = self.date_analysis['common_spatial_days']
            common_timeseries_days = self.date_analysis['common_timeseries_days']
            print(f"✅ Found {len(common_spatial_days)} common spatial dates")
            print(f"✅ Found {len(common_timeseries_days)} common timeseries dates")
            
            # Use first file for UI structure
            with h5py.File(self.reference_file, 'r') as f:
                # Load spatial properties info
                self._load_spatial_info(f)
                
                # Load time series info
                self._load_timeseries_info(f)
                
                # Load time mapping
                self._load_time_mapping(f)
                
                # Load grid structure
                self._load_grid_structure(f)
                
                # Store separate common dates for UI
                self.common_spatial_days = common_spatial_days
                self.common_timeseries_days = common_timeseries_days
                
                # Convert to dates for display
                self.common_spatial_dates = convert_days_to_dates(
                    common_spatial_days, 
                    self.time_mapping['dates'], 
                    self.time_mapping['days']
                )
                
                self.common_timeseries_dates = convert_days_to_dates(
                    common_timeseries_days, 
                    self.time_mapping['dates'], 
                    self.time_mapping['days']
                )
                
                # Perform enhanced analysis automatically
                print("🔄 Performing enhanced file analysis...")
                self.analyze_all_files()
                self.synchronize_dates()
                self.update_file_structure_overview()
                print("✅ Enhanced analysis completed")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in date-based analysis: {str(e)}")
            return False
    
    def analyze_all_files(self):
        """Perform comprehensive analysis of all .sr3 files"""
        print(f"🔍 Analyzing {len(self.sr3_files)} files in detail...")
        
        self.file_analyses = {}
        self.inactive_cells_data = {}
        
        for i, file_path in enumerate(self.sr3_files):
            # Silent processing - no individual file messages
            
            try:
                with h5py.File(file_path, 'r') as f:
                    # Basic file info
                    file_size = file_path.stat().st_size / (1024*1024)  # MB
                    
                    # Time information
                    time_table = f['General/MasterTimeTable'][...]
                    total_timesteps = len(time_table)
                    
                    # Spatial properties analysis
                    spatial_group = f['SpatialProperties']
                    spatial_timesteps = [int(key) for key in spatial_group.keys() if key.isdigit()]
                    
                    # Get active cells from first timestep
                    first_ts = f'SpatialProperties/{spatial_timesteps[0]:06d}'
                    sample_prop = None
                    for prop_name in f[first_ts].keys():
                        if not isinstance(f[first_ts][prop_name], h5py.Group):
                            sample_prop = f[first_ts][prop_name]
                            break
                    
                    active_cells = len(sample_prop) if sample_prop is not None else 0
                    
                    # Get grid info if available
                    grid_dims = None
                    if f'{first_ts}/GRID' in f:
                        grid_group = f[f'{first_ts}/GRID']
                        if all(key in grid_group for key in ['IGNTID', 'IGNTJD', 'IGNTKD']):
                            ni = grid_group['IGNTID'][...].max()
                            nj = grid_group['IGNTJD'][...].max() 
                            nk = grid_group['IGNTKD'][...].max()
                            grid_dims = (ni, nj, nk)
                    
                    # Timeseries analysis
                    ts_timesteps = 0
                    if 'TimeSeries/WELLS/Timesteps' in f:
                        ts_timesteps = len(f['TimeSeries/WELLS/Timesteps'])
                    
                    # Store analysis
                    self.file_analyses[file_path.name] = {
                        'file_size_mb': file_size,
                        'total_timesteps': total_timesteps,
                        'spatial_timesteps': len(spatial_timesteps),
                        'timeseries_timesteps': ts_timesteps,
                        'active_cells': active_cells,
                        'grid_dims': grid_dims,
                        'date_range': (float(time_table['Date'][0]), float(time_table['Date'][-1])),
                        'day_range': (float(time_table['Offset in days'][0]), float(time_table['Offset in days'][-1]))
                    }
                    
                    # Store inactive cells info
                    if grid_dims:
                        total_cells = grid_dims[0] * grid_dims[1] * grid_dims[2]
                        inactive_cells = total_cells - active_cells
                        inactive_percentage = (inactive_cells / total_cells) * 100
                        
                        self.inactive_cells_data[file_path.name] = {
                            'total_cells': total_cells,
                            'active_cells': active_cells,
                            'inactive_cells': inactive_cells,
                            'inactive_percentage': inactive_percentage,
                            'grid_dims': grid_dims
                        }
                    
            except Exception as e:
                print(f"    ❌ Error analyzing {file_path.name}: {e}")
                continue
        
        print(f"✅ Analysis completed for {len(self.file_analyses)} files")
    
    def find_universal_time_points(self, sr3_files, mode='yearly', daily_interval=365):
        """
        Find time points that exist across ALL .sr3 files using simulation days
        as the universal reference - integrated robust method.
        """
        
        print(f"🔄 Finding universal time points using {mode} mode...")
        
        # Step 1: Collect simulation day ranges from all files
        file_time_data = {}
        
        for file_path in sr3_files:
            try:
                with h5py.File(file_path, 'r') as f:
                    time_table = f['General/MasterTimeTable'][...]
                    
                    # Get available spatial and timeseries timesteps
                    spatial_timesteps = set()
                    if 'SpatialProperties' in f:
                        for key in f['SpatialProperties'].keys():
                            if key.isdigit():
                                spatial_timesteps.add(int(key))
                    
                    timeseries_timesteps = set()
                    if 'TimeSeries/WELLS/Timesteps' in f:
                        timeseries_timesteps = set(f['TimeSeries/WELLS/Timesteps'][...])
                    
                    # Map timesteps to simulation days
                    spatial_days = []
                    for ts in spatial_timesteps:
                        if ts < len(time_table):
                            spatial_days.append(float(time_table['Offset in days'][ts]))
                    
                    timeseries_days = []
                    for ts in timeseries_timesteps:
                        if ts < len(time_table):
                            timeseries_days.append(float(time_table['Offset in days'][ts]))
                    
                    file_time_data[str(file_path)] = {
                        'time_table': time_table,
                        'spatial_days': set(spatial_days),
                        'timeseries_days': set(timeseries_days),
                        'spatial_timesteps': spatial_timesteps,
                        'timeseries_timesteps': timeseries_timesteps
                    }
                    
            except Exception as e:
                print(f"❌ Error reading {file_path}: {e}")
                continue
        
        if not file_time_data:
            return None
        
        # Step 2: Find common simulation days across ALL files
        common_spatial_days = None
        common_timeseries_days = None
        
        for file_data in file_time_data.values():
            if common_spatial_days is None:
                common_spatial_days = file_data['spatial_days'].copy()
            else:
                common_spatial_days.intersection_update(file_data['spatial_days'])
                
            if common_timeseries_days is None:
                common_timeseries_days = file_data['timeseries_days'].copy()
            else:
                common_timeseries_days.intersection_update(file_data['timeseries_days'])
        
        print(f"📊 Found {len(common_spatial_days)} common spatial days")
        print(f"📊 Found {len(common_timeseries_days)} common timeseries days")
        
        # Step 3: Apply filtering based on mode
        if mode == 'yearly':
            filtered_spatial = self._filter_yearly_days(common_spatial_days, file_time_data)
            filtered_timeseries = self._filter_yearly_days(common_timeseries_days, file_time_data)
        elif mode == 'monthly':
            filtered_spatial = self._filter_monthly_days(common_spatial_days, file_time_data)
            filtered_timeseries = self._filter_monthly_days(common_timeseries_days, file_time_data)
        elif mode == 'daily':
            filtered_spatial = self._filter_daily_days(common_spatial_days, daily_interval)
            filtered_timeseries = self._filter_daily_days(common_timeseries_days, daily_interval)
        else:  # custom
            filtered_spatial = sorted(common_spatial_days)
            filtered_timeseries = sorted(common_timeseries_days)
        
        # Step 4: Convert days back to dates and timesteps for each file
        result = {
            'spatial': [],
            'timeseries': [],
            'file_mappings': {}
        }
        
        # Use first file as reference for date conversion
        reference_file = list(file_time_data.keys())[0]
        reference_time_table = file_time_data[reference_file]['time_table']
        
        # Convert spatial days
        for day in filtered_spatial:
            date_info = self._convert_day_to_date(day, reference_time_table)
            if date_info:
                result['spatial'].append({
                    'simulation_day': day,
                    'date': date_info['date'],
                    'raw_date': date_info['raw_date']
                })
        
        # Convert timeseries days  
        for day in filtered_timeseries:
            date_info = self._convert_day_to_date(day, reference_time_table)
            if date_info:
                result['timeseries'].append({
                    'simulation_day': day,
                    'date': date_info['date'], 
                    'raw_date': date_info['raw_date']
                })
        
        # Create per-file mappings
        for file_path, file_data in file_time_data.items():
            result['file_mappings'][file_path] = self._create_file_mapping(
                filtered_spatial, filtered_timeseries, file_data
            )
        
        print(f"✅ Universal synchronization complete:")
        print(f"   📅 Spatial: {len(result['spatial'])} synchronized time points")
        print(f"   📈 Timeseries: {len(result['timeseries'])} synchronized time points")
        
        return result
    
    def _filter_yearly_days(self, common_days, file_time_data):
        """Filter to get January 1st of each year"""
        # Use first file to get date mapping
        reference_file = list(file_time_data.keys())[0]
        reference_time_table = file_time_data[reference_file]['time_table']
        
        yearly_days = []
        seen_years = set()
        
        for day in sorted(common_days):
            date_info = self._convert_day_to_date(day, reference_time_table)
            if date_info and '-01-01' in date_info['date']:
                year = date_info['date'][:4]
                if year not in seen_years:
                    yearly_days.append(day)
                    seen_years.add(year)
        
        return yearly_days
    
    def _filter_monthly_days(self, common_days, file_time_data):
        """Hybrid monthly filter: ALL months for first year, common months for others"""
        # Use first file to get date mapping
        reference_file = list(file_time_data.keys())[0]
        reference_time_table = file_time_data[reference_file]['time_table']
        
        # Step 1: Group common days by year-month
        month_groups = {}  # {(year, month): [list of days]}
        
        for day in sorted(common_days):
            date_info = self._convert_day_to_date(day, reference_time_table)
            if date_info:
                try:
                    year, month, day_of_month = date_info['date'].split('-')
                    year, month = int(year), int(month)
                    
                    month_key = (year, month)
                    if month_key not in month_groups:
                        month_groups[month_key] = []
                    month_groups[month_key].append(day)
                except:
                    continue
        
        # Step 2: Find years and determine first year
        available_years = sorted(set(year for (year, month) in month_groups.keys()))
        first_year = available_years[0] if available_years else None
        
        print(f"   📅 Found data for years: {available_years}")
        print(f"   🎯 First year (detailed): {first_year}")
        
        # Step 3: Apply hybrid logic
        monthly_days = []
        total_months_found = 0
        
        for year in available_years:
            year_months_found = 0
            
            if year == first_year:
                # FIRST YEAR: Try to get ALL 12 months available in this year
                print(f"   📊 First year {year}: Including ALL available months")
                available_first_year_months = [month for (y, month) in month_groups.keys() if y == year]
                
                for month in range(1, 13):  # Jan=1 to Dec=12
                    month_key = (year, month)
                    
                    if month_key in month_groups and month_groups[month_key]:
                        # Pick the middle day from available days in this month
                        available_days = sorted(month_groups[month_key])
                        middle_idx = len(available_days) // 2
                        selected_day = available_days[middle_idx]
                        
                        monthly_days.append(selected_day)
                        year_months_found += 1
                        total_months_found += 1
                
                print(f"   📊 First year {year}: {year_months_found}/12 months included")
            
            else:
                # SUBSEQUENT YEARS: Only include months that are common across all files
                # For now, include available months (typically January, sometimes December)
                for month in range(1, 13):  # Jan=1 to Dec=12
                    month_key = (year, month)
                    
                    if month_key in month_groups and month_groups[month_key]:
                        # Pick the middle day from available days in this month
                        available_days = sorted(month_groups[month_key])
                        middle_idx = len(available_days) // 2
                        selected_day = available_days[middle_idx]
                        
                        monthly_days.append(selected_day)
                        year_months_found += 1
                        total_months_found += 1
                
                if year_months_found > 0:
                    print(f"   📊 Year {year}: {year_months_found} common months included")
        
        # Sort by simulation day to maintain chronological order
        monthly_days.sort()
        
        print(f"   ✅ Hybrid monthly filter found {len(monthly_days)} dates ({total_months_found} months)")
        print(f"   🎯 Strategy: ALL months for {first_year}, common months for other years")
        
        # Show breakdown
        if monthly_days:
            first_date = self._convert_day_to_date(monthly_days[0], reference_time_table)
            last_date = self._convert_day_to_date(monthly_days[-1], reference_time_table)
            if first_date and last_date:
                print(f"   📊 Date range: {first_date['date']} → {last_date['date']}")
        
        return monthly_days
    
    def _filter_daily_days(self, common_days, interval):
        """Filter to get regular day intervals"""
        if not common_days:
            return []
        
        sorted_days = sorted(common_days)
        min_day = sorted_days[0]
        max_day = sorted_days[-1]
        
        # Create target days at regular intervals
        target_days = np.arange(min_day, max_day + interval, interval)
        
        # Find closest actual days to targets
        daily_days = []
        for target in target_days:
            # Find closest available day
            closest_day = min(sorted_days, key=lambda x: abs(x - target))
            if closest_day not in daily_days:  # Avoid duplicates
                daily_days.append(closest_day)
        
        return daily_days
    
    def _convert_day_to_date(self, simulation_day, time_table):
        """Convert simulation day to readable date"""
        days = time_table['Offset in days']
        dates = time_table['Date']
        
        # Find closest day in time table
        idx = np.argmin(np.abs(days - simulation_day))
        date_val = dates[idx]
        
        try:
            if date_val > 20000000:  # Format like 20250101.0
                date_str = str(int(date_val))
                if len(date_str) >= 8:
                    year = int(date_str[:4])
                    month = int(date_str[4:6])
                    day = int(date_str[6:8])
                    return {
                        'date': f"{year}-{month:02d}-{day:02d}",
                        'raw_date': date_val
                    }
            
            return {
                'date': f"Day {simulation_day:.0f}",
                'raw_date': date_val
            }
        except:
            return None
    
    def _create_file_mapping(self, spatial_days, timeseries_days, file_data):
        """Create timestep mapping for a specific file"""
        time_table = file_data['time_table']
        days = time_table['Offset in days']
        timesteps = time_table['Index']
        
        mapping = {
            'spatial_mapping': [],
            'timeseries_mapping': []
        }
        
        # Map spatial days to timesteps
        for target_day in spatial_days:
            # Find closest timestep
            idx = np.argmin(np.abs(days - target_day))
            timestep = timesteps[idx]
            
            # Verify this timestep exists in spatial data
            if timestep in file_data['spatial_timesteps']:
                mapping['spatial_mapping'].append({
                    'simulation_day': target_day,
                    'timestep': timestep,
                    'actual_day': days[idx]
                })
        
        # Map timeseries days to timesteps
        for target_day in timeseries_days:
            # Find closest timestep
            idx = np.argmin(np.abs(days - target_day))
            timestep = timesteps[idx]
            
            # Verify this timestep exists in timeseries data
            if timestep in file_data['timeseries_timesteps']:
                mapping['timeseries_mapping'].append({
                    'simulation_day': target_day,
                    'timestep': timestep,
                    'actual_day': days[idx]
                })
        
        return mapping
    
    def synchronize_dates(self):
        """Find common dates using robust simulation-day based synchronization"""
        if not self.sr3_files:
            return
        
        mode = self.date_filter_mode.value
        daily_interval = self.daily_interval_widget.value
        
        print(f"🔄 ROBUST synchronization using {mode} filter...")
        
        # Use the robust synchronization system
        sync_result = self.find_universal_time_points(
            self.sr3_files, mode=mode, daily_interval=daily_interval
        )
        
        if sync_result:
            # Convert to the format expected by the UI
            self.common_dates = {
                'spatial': sync_result['spatial'],
                'timeseries': sync_result['timeseries']
            }
            
            # Store file mappings for extraction
            self.file_mappings = sync_result['file_mappings']
            self.date_sync_status = True
            
            print(f"✅ ROBUST synchronization complete:")
            print(f"   📅 Spatial: {len(self.common_dates['spatial'])} synchronized dates")
            print(f"   📈 Timeseries: {len(self.common_dates['timeseries'])} synchronized dates")
            
            # Show sample dates
            if self.common_dates['spatial']:
                print(f"   📊 Spatial date range: {self.common_dates['spatial'][0]['date']} → {self.common_dates['spatial'][-1]['date']}")
            if self.common_dates['timeseries']:
                print(f"   📊 Timeseries date range: {self.common_dates['timeseries'][0]['date']} → {self.common_dates['timeseries'][-1]['date']}")
        else:
            print("❌ ROBUST synchronization failed - no common dates found")
            self.common_dates = {'spatial': [], 'timeseries': []}
            self.date_sync_status = False
    
    def _find_common_dates_by_mode(self, all_file_dates, mode):
        """Find common dates based on filter mode"""
        from datetime import datetime
        
        if mode == 'yearly':
            return self._find_yearly_dates(all_file_dates)
        elif mode == 'monthly':
            return self._find_monthly_dates(all_file_dates)
        elif mode == 'daily':
            interval = self.daily_interval_widget.value
            return self._find_daily_dates(all_file_dates, interval)
        else:  # custom
            return self._find_all_common_dates(all_file_dates)
    
    def _find_yearly_dates(self, all_file_dates):
        """Find January 1st dates common across all files"""
        yearly_dates = {'spatial': [], 'timeseries': []}
        
        # Get first file as reference
        first_file = list(all_file_dates.keys())[0]
        ref_data = all_file_dates[first_file]
        
        for i, date_val in enumerate(ref_data['dates']):
            try:
                if date_val > 20000000:  # Format like 20250101.0
                    date_str = str(int(date_val))
                    if len(date_str) >= 8:
                        month = int(date_str[4:6])
                        day = int(date_str[6:8])
                        
                        # Check if it's January 1st
                        if month == 1 and day == 1:
                            timestep = ref_data['timesteps'][i]
                            simulation_day = ref_data['days'][i]
                            
                            # Check if this timestep exists in all files
                            in_all_spatial = True
                            in_all_timeseries = True
                            
                            for file_data in all_file_dates.values():
                                if timestep not in file_data['spatial_timesteps']:
                                    in_all_spatial = False
                                if timestep not in file_data['timeseries_timesteps']:
                                    in_all_timeseries = False
                            
                            if in_all_spatial:
                                yearly_dates['spatial'].append({
                                    'date': date_str[:4] + '-01-01',
                                    'timestep': timestep,
                                    'simulation_day': simulation_day
                                })
                            
                            if in_all_timeseries:
                                yearly_dates['timeseries'].append({
                                    'date': date_str[:4] + '-01-01', 
                                    'timestep': timestep,
                                    'simulation_day': simulation_day
                                })
            except:
                continue
        
        return yearly_dates
    
    def _find_monthly_dates(self, all_file_dates):
        """Find end-of-month dates common across all files"""
        monthly_dates = {'spatial': [], 'timeseries': []}
        
        # Get first file as reference
        first_file = list(all_file_dates.keys())[0]
        ref_data = all_file_dates[first_file]
        
        seen_months = set()
        
        for i, date_val in enumerate(ref_data['dates']):
            try:
                if date_val > 20000000:  # Format like 20250101.0
                    date_str = str(int(date_val))
                    if len(date_str) >= 8:
                        year = int(date_str[:4])
                        month = int(date_str[4:6])
                        day = int(date_str[6:8])
                        
                        month_key = f"{year}-{month:02d}"
                        
                        # Look for end-of-month dates (day >= 28)
                        if day >= 28 and month_key not in seen_months:
                            seen_months.add(month_key)
                            timestep = ref_data['timesteps'][i]
                            simulation_day = ref_data['days'][i]
                            
                            # Check if this timestep exists in all files
                            in_all_spatial = True
                            in_all_timeseries = True
                            
                            for file_data in all_file_dates.values():
                                if timestep not in file_data['spatial_timesteps']:
                                    in_all_spatial = False
                                if timestep not in file_data['timeseries_timesteps']:
                                    in_all_timeseries = False
                            
                            if in_all_spatial:
                                monthly_dates['spatial'].append({
                                    'date': f"{year}-{month:02d}-{day:02d}",
                                    'timestep': timestep,
                                    'simulation_day': simulation_day
                                })
                            
                            if in_all_timeseries:
                                monthly_dates['timeseries'].append({
                                    'date': f"{year}-{month:02d}-{day:02d}",
                                    'timestep': timestep,
                                    'simulation_day': simulation_day
                                })
            except:
                continue
        
        return monthly_dates
    
    def _find_daily_dates(self, all_file_dates, interval):
        """Find dates at regular day intervals common across all files"""
        daily_dates = {'spatial': [], 'timeseries': []}
        
        # Get first file as reference
        first_file = list(all_file_dates.keys())[0]
        ref_data = all_file_dates[first_file]
        
        # Create target days at regular intervals
        min_day = float(ref_data['days'][0])
        max_day = float(ref_data['days'][-1])
        target_days = np.arange(min_day, max_day, interval)
        
        for target_day in target_days:
            # Find closest timestep to target day
            closest_idx = np.argmin(np.abs(ref_data['days'] - target_day))
            timestep = ref_data['timesteps'][closest_idx]
            actual_day = ref_data['days'][closest_idx]
            date_val = ref_data['dates'][closest_idx]
            
            # Convert to readable date
            try:
                if date_val > 20000000:
                    date_str = str(int(date_val))
                    if len(date_str) >= 8:
                        year = int(date_str[:4])
                        month = int(date_str[4:6])
                        day = int(date_str[6:8])
                        readable_date = f"{year}-{month:02d}-{day:02d}"
                    else:
                        readable_date = f"Day {actual_day:.0f}"
                else:
                    readable_date = f"Day {actual_day:.0f}"
            except:
                readable_date = f"Day {actual_day:.0f}"
            
            # Check if this timestep exists in all files
            in_all_spatial = True
            in_all_timeseries = True
            
            for file_data in all_file_dates.values():
                if timestep not in file_data['spatial_timesteps']:
                    in_all_spatial = False
                if timestep not in file_data['timeseries_timesteps']:
                    in_all_timeseries = False
            
            if in_all_spatial:
                daily_dates['spatial'].append({
                    'date': readable_date,
                    'timestep': timestep,
                    'simulation_day': actual_day
                })
            
            if in_all_timeseries:
                daily_dates['timeseries'].append({
                    'date': readable_date,
                    'timestep': timestep,
                    'simulation_day': actual_day
                })
        
        return daily_dates
    
    def _find_all_common_dates(self, all_file_dates):
        """Find all dates common across all files"""
        common_dates = {'spatial': [], 'timeseries': []}
        
        # Get first file as reference
        first_file = list(all_file_dates.keys())[0]
        ref_data = all_file_dates[first_file]
        
        for i, timestep in enumerate(ref_data['timesteps']):
            # Check if this timestep exists in all files
            in_all_spatial = True
            in_all_timeseries = True
            
            for file_data in all_file_dates.values():
                if timestep not in file_data['spatial_timesteps']:
                    in_all_spatial = False
                if timestep not in file_data['timeseries_timesteps']:
                    in_all_timeseries = False
            
            if in_all_spatial or in_all_timeseries:
                date_val = ref_data['dates'][i]
                simulation_day = ref_data['days'][i]
                
                # Convert to readable date
                try:
                    if date_val > 20000000:
                        date_str = str(int(date_val))
                        if len(date_str) >= 8:
                            year = int(date_str[:4])
                            month = int(date_str[4:6])
                            day = int(date_str[6:8])
                            readable_date = f"{year}-{month:02d}-{day:02d}"
                        else:
                            readable_date = f"Day {simulation_day:.0f}"
                    else:
                        readable_date = f"Day {simulation_day:.0f}"
                except:
                    readable_date = f"Day {simulation_day:.0f}"
                
                if in_all_spatial:
                    common_dates['spatial'].append({
                        'date': readable_date,
                        'timestep': timestep,
                        'simulation_day': simulation_day
                    })
                
                if in_all_timeseries:
                    common_dates['timeseries'].append({
                        'date': readable_date,
                        'timestep': timestep,
                        'simulation_day': simulation_day
                    })
        
        return common_dates
    
    def update_file_structure_overview(self):
        """Update the file structure overview widget"""
        if not self.file_analyses:
            self.file_structure_widget.value = "<i>No file analysis available</i>"
            return
        
        # Create comprehensive overview
        overview_html = """
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px;">
            <h4>📊 Comprehensive File Analysis</h4>
        """
        
        # Summary statistics
        total_files = len(self.file_analyses)
        avg_size = np.mean([data['file_size_mb'] for data in self.file_analyses.values()])
        
        # Active cells analysis
        active_cells_counts = [data['active_cells'] for data in self.file_analyses.values()]
        min_active = min(active_cells_counts)
        max_active = max(active_cells_counts)
        unique_active_counts = len(set(active_cells_counts))
        
        overview_html += f"""
            <div style="margin-bottom: 15px; padding: 10px; background-color: #e8f5e8; border-radius: 5px;">
                <b>🗂️ Dataset Summary:</b><br>
                📄 Total Files: {total_files}<br>
                📊 Average File Size: {avg_size:.1f} MB<br>
                🔢 Active Cells Range: {min_active:,} → {max_active:,}<br>
                ⚠️ Different Active Cell Counts: {unique_active_counts} unique patterns
            </div>
        """
        
        # Individual file details
        overview_html += "<div style='max-height: 200px; overflow-y: auto;'>"
        overview_html += "<table style='width: 100%; border-collapse: collapse; font-size: 11px;'>"
        overview_html += """
            <tr style='background-color: #dee2e6; font-weight: bold;'>
                <th style='padding: 5px; border: 1px solid #ccc;'>File</th>
                <th style='padding: 5px; border: 1px solid #ccc;'>Size (MB)</th>
                <th style='padding: 5px; border: 1px solid #ccc;'>Active Cells</th>
                <th style='padding: 5px; border: 1px solid #ccc;'>Inactive %</th>
                <th style='padding: 5px; border: 1px solid #ccc;'>Spatial TS</th>
                <th style='padding: 5px; border: 1px solid #ccc;'>Series TS</th>
            </tr>
        """
        
        for filename, data in self.file_analyses.items():
            inactive_info = self.inactive_cells_data.get(filename, {})
            inactive_pct = inactive_info.get('inactive_percentage', 0)
            
            # Color code based on active cells
            if data['active_cells'] == max_active:
                row_color = "#d4edda"  # Green for max active
            elif data['active_cells'] == min_active:
                row_color = "#f8d7da"  # Red for min active
            else:
                row_color = "#fff3cd"  # Yellow for intermediate
            
            overview_html += f"""
                <tr style='background-color: {row_color};'>
                    <td style='padding: 3px; border: 1px solid #ccc; font-size: 10px;'>{filename[:20]}...</td>
                    <td style='padding: 3px; border: 1px solid #ccc;'>{data['file_size_mb']:.1f}</td>
                    <td style='padding: 3px; border: 1px solid #ccc;'>{data['active_cells']:,}</td>
                    <td style='padding: 3px; border: 1px solid #ccc;'>{inactive_pct:.1f}%</td>
                    <td style='padding: 3px; border: 1px solid #ccc;'>{data['spatial_timesteps']}</td>
                    <td style='padding: 3px; border: 1px solid #ccc;'>{data['timeseries_timesteps']}</td>
                </tr>
            """
        
        overview_html += "</table></div>"
        
        # Inactive cells warning if inconsistent
        if unique_active_counts > 1:
            overview_html += f"""
                <div style="margin-top: 10px; padding: 10px; background-color: #fff3cd; border-radius: 5px; border-left: 4px solid #ffc107;">
                    <b>⚠️ Inactive Cells Inconsistency Detected!</b><br>
                    Different files have different numbers of active cells ({min_active:,} to {max_active:,}).<br>
                    This may indicate different geological scenarios or simulation setups.<br>
                    <small>Extraction will handle this by creating consistent 3D grids with NaN for inactive cells.</small>
                </div>
            """
        
        overview_html += "</div>"
        self.file_structure_widget.value = overview_html
    
    def update_date_preview(self):
        """Update the date preview widget based on current filter mode"""
        if not hasattr(self, 'common_dates') or not self.common_dates:
            self.date_preview_widget.value = "<i>Run date analysis first to see available dates</i>"
            return
        
        mode = self.date_filter_mode.value
        spatial_dates = self.common_dates.get('spatial', [])
        timeseries_dates = self.common_dates.get('timeseries', [])
        
        preview_html = f"""
        <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px;">
            <h5>📅 {mode.upper()} Mode Preview:</h5>
        """
        
        if mode == 'daily':
            interval = self.daily_interval_widget.value
            preview_html += f"<p><b>Interval:</b> Every {interval} days</p>"
        
        preview_html += f"""
            <div style="display: flex; gap: 20px;">
                <div style="flex: 1;">
                    <b>🗺️ Spatial Dates ({len(spatial_dates)}):</b><br>
                    <div style="max-height: 100px; overflow-y: auto; font-size: 11px;">
        """
        
        for i, date_info in enumerate(spatial_dates[:10]):  # Show first 10
            preview_html += f"<div>📅 {date_info['date']} (Day {date_info['simulation_day']:.0f})</div>"
        
        if len(spatial_dates) > 10:
            preview_html += f"<div><i>... and {len(spatial_dates) - 10} more</i></div>"
        
        preview_html += """
                    </div>
                </div>
                <div style="flex: 1;">
                    <b>📈 Timeseries Dates (""" + str(len(timeseries_dates)) + """):</b><br>
                    <div style="max-height: 100px; overflow-y: auto; font-size: 11px;">
        """
        
        for i, date_info in enumerate(timeseries_dates[:10]):  # Show first 10
            preview_html += f"<div>📅 {date_info['date']} (Day {date_info['simulation_day']:.0f})</div>"
        
        if len(timeseries_dates) > 10:
            preview_html += f"<div><i>... and {len(timeseries_dates) - 10} more</i></div>"
        
        preview_html += """
                    </div>
                </div>
            </div>
        </div>
        """
        
        self.date_preview_widget.value = preview_html
    
    def _load_spatial_info(self, f):
        """Load spatial properties and timesteps information"""
        spatial_props = f['SpatialProperties']
        
        # Get available timesteps (exclude 'Statistics')
        self.spatial_timesteps = []
        for key in spatial_props.keys():
            if key.isdigit():
                self.spatial_timesteps.append(int(key))
        self.spatial_timesteps.sort()
        
        # Get available properties from first timestep
        if self.spatial_timesteps:
            first_ts = f'SpatialProperties/{self.spatial_timesteps[0]:06d}'
            self.spatial_properties = []
            
            for prop_name in f[first_ts].keys():
                if not isinstance(f[first_ts][prop_name], h5py.Group):
                    # Only include datasets, not groups (like GRID)
                    self.spatial_properties.append(prop_name)
            
            self.spatial_properties.sort()
    
    def _load_timeseries_info(self, f):
        """Load time series wells and variables information"""
        if 'TimeSeries/WELLS' in f:
            self.well_names = [name.decode() if isinstance(name, bytes) else str(name) 
                              for name in f['TimeSeries/WELLS/Origins'][...]]
            self.well_variables = [var.decode() if isinstance(var, bytes) else str(var)
                                  for var in f['TimeSeries/WELLS/Variables'][...]]
    
    def _load_time_mapping(self, f):
        """Load time mapping information"""
        if 'General/MasterTimeTable' in f:
            time_table = f['General/MasterTimeTable'][...]
            self.time_mapping = {
                'timesteps': time_table['Index'],
                'dates': time_table['Date'],
                'days': time_table['Offset in days']
            }
    
    def _load_grid_structure(self, f):
        """Load grid structure information"""
        if f'SpatialProperties/{self.spatial_timesteps[0]:06d}/GRID' in f:
            grid_group = f[f'SpatialProperties/{self.spatial_timesteps[0]:06d}/GRID']
            
            # Get grid dimensions
            igntid = grid_group['IGNTID'][...]
            igntjd = grid_group['IGNTJD'][...]
            igntkd = grid_group['IGNTKD'][...]
            
            ni, nj, nk = igntid.max(), igntjd.max(), igntkd.max()
            self.grid_dims = (ni, nj, nk)
            
            # Get active cell mapping
            self.active_cell_mapping = grid_group['IPSTCS'][...]
    
    def update_ui_options(self):
        """Update UI options with synchronized dates and improved organization"""
        
        # Update spatial properties options with better categorization
        static_props = [p for p in self.spatial_properties if any(term in p.upper() 
                       for term in ['PORO', 'PERM', 'POR', 'NET', 'GROSS'])]
        dynamic_props = [p for p in self.spatial_properties if any(term in p.upper() 
                        for term in ['PRES', 'SG', 'SW', 'SO', 'KR', 'MASDEN', 'MOLDEN'])]
        
        # Create organized property list with enhanced icons
        organized_props = []
        if static_props:
            organized_props.extend([f"🗿 {prop}" for prop in static_props])
        if dynamic_props:
            organized_props.extend([f"⚡ {prop}" for prop in dynamic_props])
        
        other_props = [p for p in self.spatial_properties if p not in static_props + dynamic_props]
        if other_props:
            organized_props.extend([f"📊 {prop}" for prop in other_props])
        
        self.spatial_props_widget.options = organized_props
        
        # Set default spatial properties
        default_props = []
        for prop in ['POROS', 'PERMI', 'PERMJ', 'PERMK', 'PRES', 'SG', 'SW']:
            for opt in organized_props:
                if prop in opt:
                    default_props.append(opt)
                    break
        self.spatial_props_widget.value = default_props
        
        # === ENHANCED SPATIAL DATES SELECTION ===
        # Use synchronized dates if available, otherwise fall back to old method
        if hasattr(self, 'common_dates') and self.common_dates and 'spatial' in self.common_dates:
            spatial_date_info = self.common_dates['spatial']
            print(f"🗺️ Using synchronized spatial dates: {len(spatial_date_info)} dates")
        else:
            # Fallback to old method if sync hasn't been run
            if hasattr(self, 'date_analysis') and self.date_analysis:
                spatial_days = self.date_analysis['common_spatial_days'] 
                spatial_dates = convert_days_to_dates(spatial_days, self.time_mapping['dates'], self.time_mapping['days'])
                spatial_date_info = [{'date': date, 'simulation_day': day} 
                                   for date, day in zip(spatial_dates, spatial_days)]
                print(f"🗺️ Using fallback spatial dates: {len(spatial_date_info)} dates")
            else:
                spatial_date_info = []
                print("⚠️ No spatial date information available")
        
        # Update spatial timesteps options
        spatial_timestep_options = []
        for date_info in spatial_date_info:
            label = f"{date_info['date']} (Day {date_info['simulation_day']:.0f})"
            spatial_timestep_options.append(label)
        
        self.spatial_timesteps_widget.options = spatial_timestep_options
        
        # Set intelligent default spatial timesteps (spread across time)
        if len(spatial_timestep_options) >= 5:
            indices = [0, len(spatial_timestep_options)//4, len(spatial_timestep_options)//2, 
                      3*len(spatial_timestep_options)//4, len(spatial_timestep_options)-1]
            default_spatial = [spatial_timestep_options[i] for i in indices]
        elif len(spatial_timestep_options) >= 3:
            indices = [0, len(spatial_timestep_options)//2, len(spatial_timestep_options)-1]
            default_spatial = [spatial_timestep_options[i] for i in indices]
        else:
            default_spatial = spatial_timestep_options
        
        self.spatial_timesteps_widget.value = default_spatial
        
        # Update layer selection options
        if self.grid_dims:
            nz = self.grid_dims[2]
            layer_options = [f"Layer {i}" for i in range(nz)]
            self.spatial_layers_widget.options = layer_options
            # Default: empty (all layers selected)
            self.spatial_layers_widget.value = ()
        else:
            self.spatial_layers_widget.options = []
            self.spatial_layers_widget.value = ()
        
        # Update time series variables options (unchanged)
        rate_vars = [v for v in self.well_variables if any(term in v.upper() 
                    for term in ['RATE', 'VOL', 'RC'])]
        pressure_vars = [v for v in self.well_variables if any(term in v.upper() 
                       for term in ['PRES', 'BHP', 'THP'])]
        
        # Create organized variable list
        organized_vars = []
        if rate_vars:
            organized_vars.extend([f"⚡ {var}" for var in rate_vars])
        if pressure_vars:
            organized_vars.extend([f"🔧 {var}" for var in pressure_vars])
        
        other_vars = [v for v in self.well_variables if v not in rate_vars + pressure_vars]
        if other_vars:  # Show all other variables, removed the [:20] limit
            organized_vars.extend([f"📈 {var}" for var in other_vars])
        
        self.timeseries_vars_widget.options = organized_vars
        
        # Set default variables
        default_vars = []
        for var in ['BHP', 'GASVOLRC', 'WATVOLRC']:
            for opt in organized_vars:
                if var in opt:
                    default_vars.append(opt)
                    break
        self.timeseries_vars_widget.value = default_vars
        
        # === ENHANCED TIMESERIES DATES SELECTION ===
        # Use synchronized dates if available, otherwise fall back to old method
        if hasattr(self, 'common_dates') and self.common_dates and 'timeseries' in self.common_dates:
            timeseries_date_info = self.common_dates['timeseries']
            print(f"🛢️ Using synchronized timeseries dates: {len(timeseries_date_info)} dates")
        else:
            # Fallback to old method if sync hasn't been run
            if hasattr(self, 'date_analysis') and self.date_analysis:
                timeseries_days = self.date_analysis['common_timeseries_days']
                filtered_timeseries_days, filtered_timeseries_dates = filter_yearly_dates(
                    timeseries_days, self.time_mapping['dates'], self.time_mapping['days']
                )
                timeseries_date_info = [{'date': date, 'simulation_day': day} 
                                      for date, day in zip(filtered_timeseries_dates, filtered_timeseries_days)]
                print(f"🛢️ Using fallback timeseries dates: {len(timeseries_date_info)} dates")
            else:
                timeseries_date_info = []
                print("⚠️ No timeseries date information available")
        
        # Update time series dates options
        timeseries_timestep_options = []
        for date_info in timeseries_date_info:
            label = f"{date_info['date']} (Day {date_info['simulation_day']:.0f})"
            timeseries_timestep_options.append(label)
        
        self.timeseries_dates_widget.options = timeseries_timestep_options
        
        # Set intelligent default timeseries dates (spread across time)
        if len(timeseries_timestep_options) >= 5:
            indices = [0, len(timeseries_timestep_options)//4, len(timeseries_timestep_options)//2, 
                      3*len(timeseries_timestep_options)//4, len(timeseries_timestep_options)-1]
            default_timeseries = [timeseries_timestep_options[i] for i in indices]
        elif len(timeseries_timestep_options) >= 3:
            indices = [0, len(timeseries_timestep_options)//2, len(timeseries_timestep_options)-1]
            default_timeseries = [timeseries_timestep_options[i] for i in indices]
        else:
            default_timeseries = timeseries_timestep_options
        
        self.timeseries_dates_widget.value = default_timeseries
        
        # Enhanced summary information
        mode = getattr(self.date_filter_mode, 'value', 'yearly') if hasattr(self, 'date_filter_mode') else 'yearly'
        sync_status = "✅ SYNCHRONIZED" if hasattr(self, 'date_sync_status') and self.date_sync_status else "⚠️ NOT SYNCHRONIZED"
        
        print(f"✅ UI updated with {mode.upper()} date filtering ({sync_status}):")
        print(f"🗺️ Spatial: {len(default_spatial)} selected from {len(spatial_date_info)} available dates")
        if len(spatial_date_info) > 0:
            print(f"   Range: {spatial_date_info[0]['date']} → {spatial_date_info[-1]['date']}")
        print(f"🛢️ Timeseries: {len(default_timeseries)} selected from {len(timeseries_date_info)} available dates")
        if len(timeseries_date_info) > 0:
            print(f"   Range: {timeseries_date_info[0]['date']} → {timeseries_date_info[-1]['date']}")
        print(f"📊 Total timeseries variables available: {len(self.well_variables)}")
        
        if hasattr(self, 'date_sync_status') and self.date_sync_status:
            print(f"🔄 Date synchronization ensures consistent extraction across ALL {len(self.sr3_files)} files")
    
    def display_file_info(self):
        """Display file and directory information with date-based extraction details"""
        injectors = [w for w in self.well_names if w.startswith('I')]
        producers = [w for w in self.well_names if w.startswith('P')]
        
        # Date-based extraction info
        date_info = ""
        if hasattr(self, 'date_analysis') and self.date_analysis:
            common_days = self.date_analysis['common_spatial_days'] + self.date_analysis['common_timeseries_days']
            date_range = f"{self.common_spatial_dates[0]} → {self.common_spatial_dates[-1]}"
            date_info = f"""
                    <div style="margin-top: 10px; padding: 10px; background-color: #e8f5e8; border-radius: 5px;">
                        <b>🗓️ Date-Based Extraction:</b> ENABLED ✅<br>
                        <b>📅 Common Dates:</b> {len(common_days)} dates available<br>
                        <b>📊 Date Range:</b> {date_range}<br>
                        <small>Using simulation days as universal reference across all cases</small>
                    </div>"""
        else:
            date_info = f"""
                    <div style="margin-top: 10px; padding: 8px; background-color: #fff3e0; border-radius: 5px;">
                        <b>⚠️ Date-Based Extraction:</b> Not initialized<br>
                        <small>Load files first to analyze common dates</small>
                    </div>"""
        
        info_html = f"""
        <div style="background-color: #f0f8ff; padding: 15px; border-radius: 10px; margin: 10px 0;">
            <h3>📊 Batch Processing Information</h3>
            <div style="display: flex; flex-wrap: wrap; gap: 20px;">
                <div>
                    <b>📁 Input Directory:</b> {self.directory_path}<br>
                    <b>📄 SR3 Files Found:</b> {len(self.sr3_files)}<br>
                    <b>🎯 Reference File:</b> {self.reference_file.name}<br>
                    <b>🏗️ Grid:</b> {self.grid_dims[0]} × {self.grid_dims[1]} × {self.grid_dims[2]}<br>
                    <b>📊 Active Cells:</b> {len(self.active_cell_mapping):,}<br>
                    <b>🔴 Inactive Cells:</b> {(self.grid_dims[0] * self.grid_dims[1] * self.grid_dims[2]) - len(self.active_cell_mapping):,}
                </div>
                <div>
                    <b>📋 Spatial Properties:</b> {len(self.spatial_properties)}<br>
                    <b>🏭 Injectors ({len(injectors)}):</b> {', '.join(injectors)}<br>
                    <b>🛢️ Producers ({len(producers)}):</b> {', '.join(producers)}<br>
                    <b>📊 Well Variables:</b> {len(self.well_variables)}
                </div>
            </div>
            <div style="margin-top: 10px; padding: 10px; background-color: #e6f3ff; border-radius: 5px;">
                <b>📦 Enhanced Batch Output Format:</b><br>
                <b>Spatial:</b> (N_cases={len(self.sr3_files)}, N_dates, Nx={self.grid_dims[0]}, Ny={self.grid_dims[1]}, Nz={self.grid_dims[2]}) per property<br>
                <b>Time Series:</b> (N_cases={len(self.sr3_files)}, N_dates, N_wells={len(self.well_names)}) per variable<br>
                <b>🔴 Inactive Cells:</b> Saved as 3D boolean mask ({self.grid_dims[0]}, {self.grid_dims[1]}, {self.grid_dims[2]})<br>
                <b>🗓️ Date-Based:</b> Uses simulation days as universal time reference
            </div>
            {date_info}
            <div style="margin-top: 10px; padding: 10px; background-color: #fff3e0; border-radius: 5px;">
                <b>📂 Output Folder:</b> {self.output_folder_widget.value}/<br>
                <small>All .h5 files will be saved in this folder including inactive cell locations</small>
            </div>
        </div>
        """
        self.file_info_widget.value = info_html
    
    def get_spatial_dates(self):
        """Get human-readable dates for spatial timesteps"""
        dates = []
        for timestep in self.spatial_timesteps:
            timestep_idx = np.where(self.time_mapping['timesteps'] == timestep)[0]
            if len(timestep_idx) > 0:
                excel_date = self.time_mapping['dates'][timestep_idx[0]]
                days_offset = self.time_mapping['days'][timestep_idx[0]]
                
                # Convert date using the same logic as time series
                try:
                    if excel_date > 20000000:  # Date format like 20030101.0
                        # Convert from format like 20030101.0 to datetime
                        date_str = str(int(excel_date))
                        if len(date_str) >= 8:
                            year = int(date_str[:4])
                            month = int(date_str[4:6])
                            day = int(date_str[6:8])
                            date_obj = datetime(year, month, day)
                            date_str = date_obj.strftime('%Y-%m-%d')
                        else:
                            date_str = f"Day {int(days_offset)}"
                    elif excel_date > 40000:  # Excel date format
                        ref_date = datetime(1900, 1, 1)
                        date_obj = ref_date + timedelta(days=excel_date - 2)
                        date_str = date_obj.strftime('%Y-%m-%d')
                    else:
                        date_str = f"Day {int(days_offset)}"
                except:
                    date_str = f"Timestep {timestep}"
                
                dates.append({
                    'timestep': timestep,
                    'date': date_str,
                    'days': days_offset
                })
        
        return dates
    
    def get_timeseries_dates(self):
        """Get yearly timesteps for time series data"""
        # Get all available timesteps and convert to dates
        dates = []
        for i, (timestep, excel_date, days) in enumerate(zip(
            self.time_mapping['timesteps'],
            self.time_mapping['dates'], 
            self.time_mapping['days']
        )):
            try:
                if excel_date > 40000:  # Excel date format
                    ref_date = datetime(1900, 1, 1)
                    date_obj = ref_date + timedelta(days=excel_date - 2)
                    
                    # Only include yearly timesteps (January 1st)
                    if date_obj.month == 1 and date_obj.day == 1:
                        dates.append({
                            'timestep': timestep,
                            'date': date_obj.strftime('%Y-%m-%d'),
                            'year': date_obj.year,
                            'days': days
                        })
            except:
                continue
        
        return dates
    
    def on_batch_extract_spatial(self, button):
        """Handle batch spatial properties extraction"""
        with self.extraction_output:
            clear_output()
            
            # Get selected properties (remove emoji prefixes)
            selected_props = [prop.split(' ', 1)[1] for prop in self.spatial_props_widget.value]
            
            # Get selected dates (extract timesteps from date labels)
            selected_date_labels = []
            for label in self.spatial_timesteps_widget.value:
                selected_date_labels.append(label)
            
            output_prefix = self.spatial_output_widget.value
            
            if not selected_props or not selected_date_labels:
                print("❌ Please select properties and dates")
                return
            
            print(f"🚀 Batch extracting {len(selected_props)} properties for {len(selected_date_labels)} dates...")
            print(f"📁 Processing {len(self.sr3_files)} files...")
            print(f"📊 Output prefix: {output_prefix}")
            
            success = self.batch_extract_spatial_properties(selected_props, selected_date_labels, output_prefix)
            
            if success:
                print("✅ Batch spatial data extraction completed successfully!")
            else:
                print("❌ Batch extraction failed")
    
    def on_batch_extract_timeseries(self, button):
        """Handle batch time series extraction using date-based approach"""
        with self.extraction_output:
            clear_output()
            
            # Get selected variables (remove emoji prefixes)
            selected_vars = [var.split(' ', 1)[1] for var in self.timeseries_vars_widget.value]
            
            # Get selected date labels directly (no need to parse timesteps)
            selected_date_labels = list(self.timeseries_dates_widget.value)
            
            output_prefix = self.timeseries_output_widget.value
            
            if not selected_vars or not selected_date_labels:
                print("❌ Please select variables and dates")
                return
            
            print(f"🚀 Batch extracting {len(selected_vars)} variables for {len(selected_date_labels)} dates...")
            print(f"📁 Processing {len(self.sr3_files)} files...")
            print(f"📊 Output prefix: {output_prefix}")
            
            success = self.batch_extract_timeseries_data(selected_vars, selected_date_labels, output_prefix)
            
            if success:
                print("✅ Batch time series data extraction completed successfully!")
            else:
                print("❌ Batch extraction failed")
    
    def batch_extract_spatial_properties(self, selected_properties, selected_date_labels, output_prefix):
        """Extract spatial properties using date-based approach for robust cross-case extraction"""
        
        try:
            # Create output folder
            output_folder = Path(self.output_folder_widget.value)
            output_folder.mkdir(parents=True, exist_ok=True)
            print(f"📁 Created output folder: {output_folder}")
            
            # Parse selected dates to get target days
            target_days = []
            for label in selected_date_labels:
                # Extract day value from label format "YYYY-MM-DD (Day XXX.0)"
                try:
                    day_part = label.split('Day ')[1].split(')')[0]
                    target_days.append(float(day_part))
                except:
                    print(f"⚠️ Could not parse date label: {label}")
                    continue
            
            if not target_days:
                print("❌ No valid dates selected")
                return False
            
            # Save inactive cell locations if requested
            if self.create_inactive_cells_widget.value:
                print("🔍 Extracting inactive cell locations...")
                self._save_inactive_cells_to_hdf5(output_folder)
            
            # Save well locations if requested
            if self.create_well_locations_widget.value:
                print("🛢️ Extracting well locations...")
                self._save_well_locations_to_hdf5(output_folder, target_days)
            
            print(f"🗓️ Extracting data for {len(target_days)} dates")
            
            n_cases = len(self.sr3_files)
            n_timesteps = len(target_days)
            nx, ny, nz = self.grid_dims
            
            # Get selected layers from widget (empty = all layers)
            selected_layers = []
            if hasattr(self, 'spatial_layers_widget') and self.spatial_layers_widget.value:
                # Convert widget values (strings like "Layer 0") to integers
                widget_values = self.spatial_layers_widget.value
                selected_layers = []
                for val in widget_values:
                    try:
                        # Extract layer number from "Layer X" format
                        layer_num = int(val.replace("Layer ", ""))
                        selected_layers.append(layer_num)
                    except (ValueError, AttributeError):
                        # If it's already an integer, use it directly
                        try:
                            selected_layers.append(int(val))
                        except (ValueError, TypeError):
                            pass
            
            # If no layers selected, extract all layers
            if not selected_layers:
                selected_layers = list(range(nz))
            
            n_selected_layers = len(selected_layers)
            
            print(f"📊 Batch format: ({n_cases}, {n_timesteps}, {nx}, {ny}, {n_selected_layers})")
            if n_selected_layers < nz:
                print(f"📋 Extracting {n_selected_layers} selected layers: {selected_layers}")
            else:
                print(f"📋 Extracting all {nz} layers")
            
            # Get user-specified inactive cell value
            inactive_value = self.inactive_cells_value_widget.value
            
            # Initialize data arrays for each property using inactive cell value (with selected layers only)
            batch_data = {}
            for prop in selected_properties:
                batch_data[prop] = np.full((n_cases, n_timesteps, nx, ny, n_selected_layers), inactive_value, dtype=np.float32)
            
            # Get extraction plan and convert to lookup dictionary
            extraction_plan = self.date_analysis['spatial_extraction_plan']
            extraction_lookup = _prepare_extraction_lookup(extraction_plan)
            
            # Show initial processing message
            print(f"⏳ Processing {n_cases} cases × {n_timesteps} timesteps × {len(selected_properties)} properties...")
            print(f"   Using parallel processing with {mp.cpu_count()} CPU cores")
            
            # Prepare arguments for worker function
            worker_args = []
            for case_idx, sr3_file in enumerate(self.sr3_files):
                filename = Path(sr3_file).name
                file_lookup = extraction_lookup.get(filename, {})
                worker_args.append((
                    case_idx,
                    str(sr3_file),
                    filename,
                    file_lookup,
                    target_days,
                    selected_properties,
                    self.grid_dims,
                    self.active_cell_mapping,
                    inactive_value,
                    selected_layers
                ))
            
            # Process files in parallel
            num_workers = mp.cpu_count()
            successful_cases = 0
            
            with mp.Pool(processes=num_workers) as pool:
                # Process files in chunks for progress updates
                chunk_size = max(1, n_cases // (num_workers * 4))  # Update progress 4x per worker
                results = []
                
                for i, result in enumerate(pool.imap(_extract_single_file_spatial, worker_args, chunksize=chunk_size)):
                    results.append(result)
                    self.batch_progress_widget.value = int(((i + 1) / n_cases) * 100)
                    
                    if result['success']:
                        successful_cases += 1
                
                # Combine results into batch_data arrays
                for result in results:
                    case_idx = result['case_idx']
                    for prop in selected_properties:
                        batch_data[prop][case_idx] = result['data'][prop]
            
            # Save each property to separate HDF5 file in output folder
            # Convert target_days back to timestep format for metadata compatibility
            metadata_timesteps = [int(day) for day in target_days]  # Use days as pseudo-timesteps
            
            # Summary of saved files
            saved_files = []
            
            for prop in selected_properties:
                output_file = output_folder / f"{output_prefix}_{prop}.h5"
                self._save_batch_spatial_to_hdf5(batch_data[prop], prop, metadata_timesteps, output_file, target_days, selected_layers)
                saved_files.append(f"{output_prefix}_{prop}.h5")
            
            self.batch_progress_widget.value = 100
            self.progress_widget.value = 100
            
            # Print summary
            print("\n" + "=" * 50)
            print(f"✅ EXTRACTION COMPLETE - SUMMARY")
            print("=" * 50)
            print(f"📊 Processed {n_cases} cases with {successful_cases} fully successful")
            print(f"🗓️ Extracted {n_timesteps} timesteps per case")
            print(f"📋 Extracted {len(selected_properties)} properties: {', '.join(selected_properties)}")
            print(f"📊 Extracted {n_selected_layers} layers (out of {nz} total)")
            print(f"💾 Saved {len(saved_files)} files to {output_folder}:")
            for i, file in enumerate(saved_files):
                print(f"   {i+1}. {file}")
            print("=" * 50)
            
            return True
            
        except Exception as e:
            print(f"❌ Error in date-based spatial extraction: {str(e)}")
            return False
    
    def batch_extract_timeseries_data(self, selected_variables, selected_date_labels, output_prefix):
        """Extract time series data using date-based approach for robust cross-case extraction"""
        
        try:
            # Create output folder
            output_folder = Path(self.output_folder_widget.value)
            output_folder.mkdir(parents=True, exist_ok=True)
            print(f"📁 Using output folder: {output_folder}")
            
            # Parse selected dates to get target days
            target_days = []
            for label in selected_date_labels:
                # Extract day value from label format "YYYY-MM-DD (Day XXX.0)"
                try:
                    day_part = label.split('Day ')[1].split(')')[0]
                    target_days.append(float(day_part))
                except:
                    print(f"⚠️ Could not parse date label: {label}")
                    continue
            
            if not target_days:
                print("❌ No valid dates selected")
                return False
            
            print(f"🗓️ Extracting time series for {len(target_days)} dates")
            
            n_cases = len(self.sr3_files)
            n_timesteps = len(target_days)
            n_wells = len(self.well_names)
            
            print(f"📊 Batch format: ({n_cases}, {n_timesteps}, {n_wells})")
            
            # Initialize data arrays for each variable
            batch_data = {}
            for var in selected_variables:
                batch_data[var] = np.zeros((n_cases, n_timesteps, n_wells), dtype=np.float32)
            
            # Get extraction plan and convert to lookup dictionary
            extraction_plan = self.date_analysis['timeseries_extraction_plan']
            extraction_lookup = _prepare_extraction_lookup(extraction_plan)
            
            # Show initial processing message
            print(f"⏳ Processing {n_cases} cases × {n_timesteps} timesteps × {len(selected_variables)} variables...")
            print(f"   Using parallel processing with {mp.cpu_count()} CPU cores")
            
            # Prepare arguments for worker function
            worker_args = []
            for case_idx, sr3_file in enumerate(self.sr3_files):
                filename = Path(sr3_file).name
                file_lookup = extraction_lookup.get(filename, {})
                worker_args.append((
                    case_idx,
                    str(sr3_file),
                    filename,
                    file_lookup,
                    target_days,
                    selected_variables,
                    n_wells
                ))
            
            # Process files in parallel
            num_workers = mp.cpu_count()
            successful_cases = 0
            
            with mp.Pool(processes=num_workers) as pool:
                # Process files in chunks for progress updates
                chunk_size = max(1, n_cases // (num_workers * 4))  # Update progress 4x per worker
                results = []
                
                for i, result in enumerate(pool.imap(_extract_single_file_timeseries, worker_args, chunksize=chunk_size)):
                    results.append(result)
                    self.batch_progress_widget.value = int(((i + 1) / n_cases) * 100)
                    
                    if result['success']:
                        successful_cases += 1
                
                # Combine results into batch_data arrays
                for result in results:
                    case_idx = result['case_idx']
                    for var in selected_variables:
                        batch_data[var][case_idx] = result['data'][var]
            
            # Save each variable to separate HDF5 file in output folder
            # Create date info for metadata
            selected_dates = []
            for day in target_days:
                date_str = convert_days_to_dates([day], self.time_mapping['dates'], self.time_mapping['days'])[0]
                selected_dates.append({
                    'timestep': int(day),  # Use day as pseudo-timestep
                    'date': date_str,
                    'days': day
                })
            
            # Summary of saved files
            saved_files = []
            
            for var in selected_variables:
                output_file = output_folder / f"{output_prefix}_{var}.h5"
                self._save_batch_timeseries_to_hdf5(batch_data[var], var, selected_dates, output_file)
                saved_files.append(f"{output_prefix}_{var}.h5")
            
            self.batch_progress_widget.value = 100
            self.progress_widget.value = 100
            
            # Print summary
            print("\n" + "=" * 50)
            print(f"✅ EXTRACTION COMPLETE - SUMMARY")
            print("=" * 50)
            print(f"📊 Processed {n_cases} cases with {successful_cases} fully successful")
            print(f"🗓️ Extracted {n_timesteps} timesteps per case")
            print(f"🛢️ Extracted {len(selected_variables)} variables across {n_wells} wells")
            print(f"💾 Saved {len(saved_files)} files to {output_folder}:")
            for i, file in enumerate(saved_files):
                print(f"   {i+1}. {file}")
            print("=" * 50)
            
            return True
            
        except Exception as e:
            print(f"❌ Error in date-based timeseries extraction: {str(e)}")
            return False
    
    def _save_batch_spatial_to_hdf5(self, data_array, property_name, timesteps, output_file, target_days, selected_layers=None):
        """Save batch spatial data to HDF5 file with adaptive compression and chunking"""
        # Calculate data size for compression optimization
        data_size_bytes = data_array.nbytes
        compression_type, compression_level = _get_optimal_compression(data_size_bytes)
        
        # Calculate optimal chunk shape
        chunk_shape = _calculate_chunk_shape(data_array.shape, data_array.dtype)
        
        # Handle layer selection metadata
        original_nz = self.grid_dims[2]
        if selected_layers is None or len(selected_layers) == 0:
            selected_layers = list(range(original_nz))
        n_selected_layers = len(selected_layers)
        
        with h5py.File(output_file, 'w') as f:
            # Save main data array with optimized compression and chunking
            f.create_dataset(
                'data', 
                data=data_array, 
                compression=compression_type,
                compression_opts=compression_level,
                chunks=chunk_shape,
                shuffle=True  # Enable shuffle filter for better compression
            )
            
            # Save metadata
            meta_group = f.create_group('metadata')
            meta_group.attrs['property_name'] = property_name
            meta_group.attrs['n_cases'] = len(self.sr3_files)
            meta_group.attrs['n_timesteps'] = len(timesteps)
            meta_group.attrs['nx'] = self.grid_dims[0]
            meta_group.attrs['ny'] = self.grid_dims[1]
            meta_group.attrs['nz'] = n_selected_layers  # Number of selected layers
            meta_group.attrs['original_nz'] = original_nz  # Original number of layers
            meta_group.attrs['grid_dimensions'] = self.grid_dims
            meta_group.attrs['total_cells'] = self.grid_dims[0] * self.grid_dims[1] * self.grid_dims[2]
            meta_group.attrs['active_cells'] = len(self.active_cell_mapping)
            meta_group.create_dataset('timesteps', data=timesteps)
            meta_group.create_dataset('sr3_files', data=[str(f).encode() for f in self.sr3_files])
            meta_group.create_dataset('active_cell_mapping', data=self.active_cell_mapping)
            
            # Save selected layers information
            meta_group.create_dataset('selected_layers', data=selected_layers)
            
            # Add description
            f.attrs['description'] = f'Batch spatial data for property {property_name}'
            if n_selected_layers < original_nz:
                f.attrs['format'] = f'Shape: (N_cases, N_timesteps, Nx, Ny, N_selected_layers={n_selected_layers})'
            else:
                f.attrs['format'] = 'Shape: (N_cases, N_timesteps, Nx, Ny, Nz)'
            f.attrs['extraction_date'] = datetime.now().isoformat()
            f.attrs['compression'] = f'{compression_type} level {compression_level}'
            
            # Save target days
            meta_group.create_dataset('target_days', data=target_days)
    
    def _save_batch_timeseries_to_hdf5(self, data_array, variable_name, selected_dates, output_file):
        """Save batch time series data to HDF5 file with adaptive compression and chunking"""
        # Calculate data size for compression optimization
        data_size_bytes = data_array.nbytes
        compression_type, compression_level = _get_optimal_compression(data_size_bytes)
        
        # Calculate optimal chunk shape
        chunk_shape = _calculate_chunk_shape(data_array.shape, data_array.dtype)
        
        with h5py.File(output_file, 'w') as f:
            # Save main data array with optimized compression and chunking
            f.create_dataset(
                'data', 
                data=data_array, 
                compression=compression_type,
                compression_opts=compression_level,
                chunks=chunk_shape,
                shuffle=True  # Enable shuffle filter for better compression
            )
            
            # Save metadata
            meta_group = f.create_group('metadata')
            meta_group.attrs['variable_name'] = variable_name
            meta_group.attrs['n_cases'] = len(self.sr3_files)
            meta_group.attrs['n_timesteps'] = len(selected_dates)
            meta_group.attrs['n_wells'] = len(self.well_names)
            meta_group.create_dataset('timesteps', data=[d['timestep'] for d in selected_dates])
            meta_group.create_dataset('dates', data=[d['date'].encode() for d in selected_dates])
            meta_group.create_dataset('well_names', data=[name.encode() for name in self.well_names])
            meta_group.create_dataset('sr3_files', data=[str(f).encode() for f in self.sr3_files])
            
            # Add description
            f.attrs['description'] = f'Batch time series data for variable {variable_name}'
            f.attrs['format'] = 'Shape: (N_cases, N_timesteps, N_wells)'
            f.attrs['extraction_date'] = datetime.now().isoformat()
            f.attrs['compression'] = f'{compression_type} level {compression_level}'

    def _map_active_to_3d(self, active_data, inactive_value=0.0):
        """Map 1D active cell data to 3D grid with user-specified inactive cell value (vectorized)"""
        return _map_active_to_3d_vectorized(active_data, self.active_cell_mapping, self.grid_dims, inactive_value)
    
    def _create_inactive_cells_mask(self):
        """Create 3D boolean mask for inactive cells (True = inactive, False = active)"""
        ni, nj, nk = self.grid_dims
        inactive_mask = np.ones((ni, nj, nk), dtype=bool)  # Start with all cells as inactive
        
        # Mark active cells as False (not inactive)
        for grid_cell in self.active_cell_mapping:
            if grid_cell > 0:
                # Convert 1D grid index to I-J-K (CMG uses 1-based indexing)
                grid_idx = grid_cell - 1  # Convert to 0-based
                
                # Calculate I-J-K coordinates
                k = grid_idx // (ni * nj)
                j = (grid_idx % (ni * nj)) // ni
                i = grid_idx % ni
                
                if k < nk and j < nj and i < ni:
                    inactive_mask[i, j, k] = False  # Mark as active (not inactive)
        
        return inactive_mask
    
    def _create_inactive_cells_mask_for_case(self, active_cell_mapping):
        """Create 3D boolean mask for inactive cells for a specific case"""
        ni, nj, nk = self.grid_dims
        inactive_mask = np.ones((ni, nj, nk), dtype=bool)  # Start with all cells as inactive
        
        # Mark active cells as False (not inactive)
        for grid_cell in active_cell_mapping:
            if grid_cell > 0:
                # Convert 1D grid index to I-J-K (CMG uses 1-based indexing)
                grid_idx = grid_cell - 1  # Convert to 0-based
                
                # Calculate I-J-K coordinates
                k = grid_idx // (ni * nj)
                j = (grid_idx % (ni * nj)) // ni
                i = grid_idx % ni
                
                if k < nk and j < nj and i < ni:
                    inactive_mask[i, j, k] = False  # Mark as active (not inactive)
        
        return inactive_mask
    
    def _save_inactive_cells_to_hdf5(self, output_folder):
        """Save inactive cell locations for ALL cases to HDF5 file"""
        try:
            print("🔍 Extracting inactive cell locations for ALL cases...")
            output_file = Path(output_folder) / "inactive_cell_locations.h5"
            
            n_cases = len(self.sr3_files)
            ni, nj, nk = self.grid_dims
            
            # Initialize arrays for all cases
            all_inactive_masks = np.zeros((n_cases, ni, nj, nk), dtype=bool)
            all_active_masks = np.zeros((n_cases, ni, nj, nk), dtype=bool)
            
            # Store active cell mappings for each case
            all_active_mappings = []
            
            # Process each SR3 file to extract its inactive cell pattern
            print(f"⏳ Processing {n_cases} cases for inactive cell patterns...")
            
            for case_idx, sr3_file in enumerate(self.sr3_files):
                filename = Path(sr3_file).name
                
                try:
                    with h5py.File(sr3_file, 'r') as f:
                        # Get the first available spatial timestep for this file
                        spatial_timesteps = []
                        if 'SpatialProperties' in f:
                            for key in f['SpatialProperties'].keys():
                                if key.isdigit():
                                    spatial_timesteps.append(int(key))
                        
                        if not spatial_timesteps:
                            # Use reference mapping as fallback
                            case_inactive_mask = self._create_inactive_cells_mask_for_case(self.active_cell_mapping)
                            all_inactive_masks[case_idx] = case_inactive_mask
                            all_active_masks[case_idx] = ~case_inactive_mask
                            all_active_mappings.append(self.active_cell_mapping)
                            continue
                        
                        first_ts = min(spatial_timesteps)
                        grid_path = f'SpatialProperties/{first_ts:06d}/GRID'
                        
                        if grid_path in f:
                            # Extract active cell mapping for this specific case
                            grid_group = f[grid_path]
                            case_active_mapping = grid_group['IPSTCS'][...]
                            
                            # Verify grid dimensions are consistent
                            igntid = grid_group['IGNTID'][...]
                            igntjd = grid_group['IGNTJD'][...]
                            igntkd = grid_group['IGNTKD'][...]
                            
                            case_ni, case_nj, case_nk = igntid.max(), igntjd.max(), igntkd.max()
                            
                            if (case_ni, case_nj, case_nk) != (ni, nj, nk):
                                case_active_mapping = self.active_cell_mapping
                            
                            # Create inactive mask for this case
                            case_inactive_mask = self._create_inactive_cells_mask_for_case(case_active_mapping)
                            all_inactive_masks[case_idx] = case_inactive_mask
                            all_active_masks[case_idx] = ~case_inactive_mask
                            all_active_mappings.append(case_active_mapping)
                            
                        else:
                            # Use reference mapping as fallback
                            case_inactive_mask = self._create_inactive_cells_mask_for_case(self.active_cell_mapping)
                            all_inactive_masks[case_idx] = case_inactive_mask
                            all_active_masks[case_idx] = ~case_inactive_mask
                            all_active_mappings.append(self.active_cell_mapping)
                
                except Exception as e:
                    # Use reference mapping as fallback
                    case_inactive_mask = self._create_inactive_cells_mask_for_case(self.active_cell_mapping)
                    all_inactive_masks[case_idx] = case_inactive_mask
                    all_active_masks[case_idx] = ~case_inactive_mask
                    all_active_mappings.append(self.active_cell_mapping)
            
            # Save to HDF5 file
            with h5py.File(output_file, 'w') as f:
                # Save 4D arrays: (N_cases, Nx, Ny, Nz)
                f.create_dataset('inactive_mask', data=all_inactive_masks, compression='gzip', compression_opts=9)
                f.create_dataset('active_mask', data=all_active_masks, compression='gzip', compression_opts=9)
                
                # Save metadata
                meta_group = f.create_group('metadata')
                meta_group.attrs['n_cases'] = n_cases
                meta_group.attrs['nx'] = self.grid_dims[0]
                meta_group.attrs['ny'] = self.grid_dims[1]
                meta_group.attrs['nz'] = self.grid_dims[2]
                meta_group.attrs['grid_dimensions'] = self.grid_dims
                meta_group.attrs['total_cells_per_case'] = self.grid_dims[0] * self.grid_dims[1] * self.grid_dims[2]
                
                # Calculate statistics across all cases
                total_active_per_case = np.sum(all_active_masks, axis=(1,2,3))
                total_inactive_per_case = np.sum(all_inactive_masks, axis=(1,2,3))
                
                meta_group.create_dataset('active_cells_per_case', data=total_active_per_case)
                meta_group.create_dataset('inactive_cells_per_case', data=total_inactive_per_case)
                
                # Save case filenames
                meta_group.create_dataset('sr3_files', data=[str(f).encode() for f in self.sr3_files])
                
                # Save active cell mappings for each case (as variable-length dataset)
                max_mapping_length = max(len(mapping) for mapping in all_active_mappings)
                padded_mappings = np.full((n_cases, max_mapping_length), -1, dtype=np.int32)
                for i, mapping in enumerate(all_active_mappings):
                    padded_mappings[i, :len(mapping)] = mapping
                
                meta_group.create_dataset('active_cell_mappings', data=padded_mappings)
                meta_group.create_dataset('active_cell_mapping_lengths', data=[len(m) for m in all_active_mappings])
                
                # Add description
                f.attrs['description'] = 'Inactive and active cell locations for all cases in 3D grid'
                f.attrs['format'] = 'Shape: (N_cases, Nx, Ny, Nz) - True=inactive, False=active'
                f.attrs['extraction_date'] = datetime.now().isoformat()
                f.attrs['extraction_method'] = 'per_case_extraction'
            
            # Calculate overall statistics
            avg_active = np.mean(total_active_per_case)
            avg_inactive = np.mean(total_inactive_per_case)
            total_cells = self.grid_dims[0] * self.grid_dims[1] * self.grid_dims[2]
            
            # Print summary
            print(f"✅ Saved inactive cell locations: {output_file}")
            print(f"   📊 Shape: {all_inactive_masks.shape} (N_cases, Nx, Ny, Nz)")
            print(f"   🗂️ Cases: {n_cases}")
            print(f"   🟢 Average active cells per case: {avg_active:,.0f} ({avg_active/total_cells*100:.1f}%)")
            print(f"   🔴 Average inactive cells per case: {avg_inactive:,.0f} ({avg_inactive/total_cells*100:.1f}%)")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving inactive cell locations: {str(e)}")
            return False
    
    def _extract_well_completions_for_case(self, sr3_file, target_days, grid_dims, active_cell_mapping):
        """
        Extract well completion IJK coordinates for a single case
        
        Searches multiple possible locations in SR3 files for completion data:
        1. Tables/{timestep}/COMPLETIONS or similar tables
        2. SpatialProperties/{timestep}/GRID for completion datasets
        3. TimeSeries/WELLS for connection/completion datasets
        
        Args:
            sr3_file: Path to SR3 file
            target_days: List of target simulation days
            grid_dims: Tuple (ni, nj, nk) grid dimensions
            active_cell_mapping: Active cell mapping array
            
        Returns:
            completions: dict[timestep_idx] -> list of (i, j, k, well_name, well_type) tuples
            well_names: List of well names
            well_types: List of well types ('I' or 'P')
        """
        ni, nj, nk = grid_dims
        completions = {}
        well_names = []
        well_types = []
        
        try:
            with h5py.File(sr3_file, 'r') as f:
                # Get well names and types
                if 'TimeSeries/WELLS' in f:
                    origins = f['TimeSeries/WELLS']['Origins'][...]
                    well_names = [name.decode() if isinstance(name, bytes) else str(name) 
                                 for name in origins]
                    
                    # Identify well types based on name prefix
                    well_types = []
                    for name in well_names:
                        if name and len(name) > 0:
                            if name[0].upper() == 'I':
                                well_types.append('I')
                            elif name[0].upper() == 'P':
                                well_types.append('P')
                            else:
                                well_types.append('U')  # Unknown
                        else:
                            well_types.append('U')
                
                # Get spatial timesteps
                spatial_timesteps = []
                if 'SpatialProperties' in f:
                    for key in f['SpatialProperties'].keys():
                        if key.isdigit():
                            spatial_timesteps.append(int(key))
                spatial_timesteps.sort()
                
                if not spatial_timesteps:
                    return completions, well_names, well_types
                
                # Get time mapping to match target_days to timesteps
                time_mapping = {}
                if 'General/MasterTimeTable' in f:
                    time_table = f['General/MasterTimeTable'][...]
                    for row in time_table:
                        ts_idx = int(row['Index'])
                        days = float(row['Offset in days'])
                        time_mapping[ts_idx] = days
                
                # Process each target day
                for ts_idx, target_day in enumerate(target_days):
                    completions[ts_idx] = []
                    
                    # Find closest spatial timestep for this target day
                    best_ts = None
                    min_diff = float('inf')
                    for sp_ts in spatial_timesteps:
                        if sp_ts in time_mapping:
                            diff = abs(time_mapping[sp_ts] - target_day)
                            if diff < min_diff:
                                min_diff = diff
                                best_ts = sp_ts
                    
                    if best_ts is None:
                        continue
                    
                    ts_key = f'{best_ts:06d}'
                    
                    # Strategy 1: Check Tables/{timestep}/ for completion tables
                    completion_found = False
                    if 'Tables' in f:
                        if ts_key in f['Tables']:
                            tables_group = f['Tables'][ts_key]
                            if isinstance(tables_group, h5py.Group):
                                for table_name in tables_group.keys():
                                    name_lower = table_name.lower()
                                    if any(kw in name_lower for kw in ['complet', 'connect', 'well']):
                                        try:
                                            table_data = tables_group[table_name][...]
                                            if hasattr(table_data.dtype, 'names'):
                                                # Structured array - check for IJK fields
                                                field_names = table_data.dtype.names
                                                i_field = None
                                                j_field = None
                                                k_field = None
                                                well_field = None
                                                
                                                for field in field_names:
                                                    field_lower = field.lower()
                                                    if field_lower in ['i', 'ii', 'indx_i', 'index_i']:
                                                        i_field = field
                                                    elif field_lower in ['j', 'jj', 'indx_j', 'index_j']:
                                                        j_field = field
                                                    elif field_lower in ['k', 'kk', 'indx_k', 'index_k']:
                                                        k_field = field
                                                    elif 'well' in field_lower or 'name' in field_lower:
                                                        well_field = field
                                                
                                                if i_field and j_field and k_field:
                                                    for row in table_data:
                                                        i_val = int(row[i_field]) - 1  # Convert to 0-based
                                                        j_val = int(row[j_field]) - 1
                                                        k_val = int(row[k_field]) - 1
                                                        well_name = str(row[well_field]) if well_field else "UNKNOWN"
                                                        if isinstance(well_name, bytes):
                                                            well_name = well_name.decode()
                                                        
                                                        # Determine well type
                                                        well_type = 'U'
                                                        if well_name in well_names:
                                                            well_idx = well_names.index(well_name)
                                                            well_type = well_types[well_idx] if well_idx < len(well_types) else 'U'
                                                        
                                                        if 0 <= i_val < ni and 0 <= j_val < nj and 0 <= k_val < nk:
                                                            completions[ts_idx].append((i_val, j_val, k_val, well_name, well_type))
                                                    completion_found = True
                                        except Exception as e:
                                            pass
                    
                    # Strategy 2: Check SpatialProperties/{timestep}/GRID for completion data
                    if not completion_found and 'SpatialProperties' in f:
                        if ts_key in f['SpatialProperties']:
                            sp_group = f['SpatialProperties'][ts_key]
                            if 'GRID' in sp_group:
                                grid_group = sp_group['GRID']
                                # Look for completion-related datasets
                                for key in grid_group.keys():
                                    key_lower = key.lower()
                                    if any(kw in key_lower for kw in ['complet', 'connect', 'well']):
                                        try:
                                            data = grid_group[key][...]
                                            # This would need specific format knowledge
                                            # Placeholder for future implementation
                                            pass
                                        except:
                                            pass
                    
                    # Strategy 3: If no completions found, provide empty list (completions can be added manually later)
                    # This allows the framework to work even if completion data location varies
                    
        except Exception as e:
            print(f"⚠️ Warning: Could not extract completions from {Path(sr3_file).name}: {str(e)}")
        
        return completions, well_names, well_types
    
    def _save_well_locations_to_hdf5(self, output_folder, target_days):
        """Save well location data for ALL cases and timesteps to HDF5 file"""
        try:
            print("🛢️ Extracting well locations for ALL cases and timesteps...")
            output_file = Path(output_folder) / "batch_spatial_properties_well_location.h5"
            
            n_cases = len(self.sr3_files)
            n_timesteps = len(target_days)
            ni, nj, nk = self.grid_dims
            
            # Initialize 5D array: (N_cases, N_timesteps, Nx, Ny, Nz)
            well_locations = np.zeros((n_cases, n_timesteps, ni, nj, nk), dtype=np.uint8)
            
            # Initialize well_mapping array: (N_cases, N_timesteps, Nx, Ny, Nz)
            # -1 = no well, 0 to N_wells-1 = well index
            well_mapping = np.full((n_cases, n_timesteps, ni, nj, nk), -1, dtype=np.int16)
            
            # Store well metadata
            all_well_names = []
            all_well_types = []
            completion_counts = np.zeros((n_cases, n_timesteps), dtype=np.int32)
            
            # Process each SR3 file
            print(f"⏳ Processing {n_cases} cases × {n_timesteps} timesteps for well locations...")
            
            for case_idx, sr3_file in enumerate(self.sr3_files):
                filename = Path(sr3_file).name
                print(f"   Processing case {case_idx + 1}/{n_cases}: {filename}")
                
                try:
                    # Extract completions for this case
                    completions, well_names, well_types = self._extract_well_completions_for_case(
                        sr3_file, target_days, self.grid_dims, self.active_cell_mapping
                    )
                    
                    # Store well names and types (use first case's wells as reference)
                    if case_idx == 0:
                        all_well_names = well_names
                        all_well_types = well_types
                    
                    # Create mapping from well_name to well index for this case
                    well_name_to_index = {}
                    for idx, well_name in enumerate(well_names):
                        well_name_to_index[well_name] = idx
                    
                    # Mark completion locations in the 5D array and store well mapping
                    for ts_idx in completions:
                        completion_list = completions[ts_idx]
                        completion_counts[case_idx, ts_idx] = len(completion_list)
                        
                        for i, j, k, well_name, well_type in completion_list:
                            if 0 <= i < ni and 0 <= j < nj and 0 <= k < nk:
                                well_locations[case_idx, ts_idx, i, j, k] = 1
                                # Store well index in mapping array
                                if well_name in well_name_to_index:
                                    well_mapping[case_idx, ts_idx, i, j, k] = well_name_to_index[well_name]
                                else:
                                    # If well name not found, try to find matching well in all_well_names
                                    if case_idx > 0 and well_name in all_well_names:
                                        well_mapping[case_idx, ts_idx, i, j, k] = all_well_names.index(well_name)
                    
                    total_completions = sum(len(completions[ts]) for ts in completions)
                    print(f"      Found {total_completions} total completions")
                    if total_completions == 0:
                        print(f"      ⚠️ Warning: No completions found for {filename}")
                        print(f"         This may be because completion data is not stored in expected SR3 locations")
                        print(f"         Completion data might be in simulation input files (.dat/.cmsd) or other formats")
                    
                except Exception as e:
                    print(f"      ⚠️ Warning: Error processing {filename}: {str(e)}")
                    continue
            
            # Save to HDF5 file
            with h5py.File(output_file, 'w') as f:
                # Save main data array with compression
                f.create_dataset(
                    'data', 
                    data=well_locations, 
                    compression='gzip', 
                    compression_opts=9,
                    chunks=(1, 1, ni, nj, nk)
                )
                
                # Save well_mapping array with compression
                f.create_dataset(
                    'well_mapping',
                    data=well_mapping,
                    compression='gzip',
                    compression_opts=9,
                    chunks=(1, 1, ni, nj, nk)
                )
                
                # Save metadata (matching spatial files structure)
                meta_group = f.create_group('metadata')
                meta_group.attrs['n_cases'] = n_cases
                meta_group.attrs['n_timesteps'] = n_timesteps
                meta_group.attrs['nx'] = ni
                meta_group.attrs['ny'] = nj
                meta_group.attrs['nz'] = nk
                meta_group.attrs['grid_dimensions'] = self.grid_dims
                meta_group.attrs['total_cells'] = ni * nj * nk
                meta_group.attrs['active_cells'] = len(self.active_cell_mapping) if hasattr(self, 'active_cell_mapping') else 0
                
                # Save timesteps array (for consistency with spatial files)
                # Convert target_days to timestep indices (0, 1, 2, ...)
                timesteps = np.arange(n_timesteps, dtype=np.int32)
                meta_group.create_dataset('timesteps', data=timesteps)
                
                # Save well names and types
                if all_well_names:
                    meta_group.create_dataset('well_names', data=[name.encode() for name in all_well_names])
                    meta_group.create_dataset('well_types', data=[wt.encode() for wt in all_well_types])
                
                # Save completion counts
                meta_group.create_dataset('completion_counts', data=completion_counts)
                
                # Save target days
                meta_group.create_dataset('target_days', data=target_days)
                
                # Save case filenames
                meta_group.create_dataset('sr3_files', data=[str(sr3_file).encode() for sr3_file in self.sr3_files])
                
                # Save active_cell_mapping for consistency with spatial files
                if hasattr(self, 'active_cell_mapping') and self.active_cell_mapping is not None:
                    meta_group.create_dataset('active_cell_mapping', data=self.active_cell_mapping)
                
                # Add description
                f.attrs['description'] = 'Well completion locations for all cases and timesteps'
                f.attrs['format'] = 'Shape: (N_cases, N_timesteps, Nx, Ny, Nz) - 1=well completion, 0=no well'
                f.attrs['extraction_date'] = datetime.now().isoformat()
            
            # Calculate statistics
            total_completions = np.sum(completion_counts)
            avg_completions_per_timestep = np.mean(completion_counts)
            
            # Print summary
            print(f"✅ Saved well locations: {output_file.name}")
            print(f"   📁 Full path: {output_file}")
            print(f"   📊 Shape: {well_locations.shape} (N_cases, N_timesteps, Nx, Ny, Nz)")
            print(f"   🗂️ Cases: {n_cases}, Timesteps: {n_timesteps}")
            print(f"   🛢️ Wells: {len(all_well_names)} ({len([w for w in all_well_types if w == 'I'])} injectors, {len([w for w in all_well_types if w == 'P'])} producers)")
            print(f"   📍 Total completions: {total_completions:,}")
            print(f"   📊 Average completions per timestep: {avg_completions_per_timestep:.1f}")
            print(f"   ✅ Well mapping saved: enables enhanced visualization with well names/types")
            print(f"   ✅ File structure matches spatial files format (ready for ML workflows)")
            print(f"   ✅ Filename follows naming convention: batch_spatial_properties_well_location.h5")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving well locations: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_default_conversion_factors(self):
        """Get default conversion factors based on property/variable names"""
        return {
            # Pressure conversions (kPa to psi)
            'PRES': 0.145038,
            'BHP': 0.145038,
            'THP': 0.145038,
            'PRESSURE': 0.145038,
            
            # Flow rate conversions (m³/day to ft³/day)
            'GASRATSC': 35.3147,
            'WATRATSC': 35.3147,
            'OILRATSC': 35.3147,
            'GASRATE': 35.3147,
            'WATRATE': 35.3147,
            'OILRATE': 35.3147,
            
            # Temperature conversions (Celsius to Fahrenheit: °F = °C × 9/5 + 32)
            'TEMP': 1.8,  # First multiply by 1.8, then add 32 (handled separately)
            'TEMPERATURE': 1.8,
            
            # Permeability conversions (mD to Darcy)
            'PERMX': 0.001,
            'PERMY': 0.001,
            'PERMZ': 0.001,
            'PERM': 0.001,
            
            # Porosity (no conversion needed, dimensionless)
            'PORO': 1.0,
            'POROSITY': 1.0,
            
            # Saturation (no conversion needed, dimensionless)
            'SWAT': 1.0,
            'SGAS': 1.0,
            'SOIL': 1.0,
            
            # Default for unknown properties
            'DEFAULT': 1.0
        }
    
    def on_output_folder_change(self, change):
        """Auto-sync conversion directory when output folder changes"""
        self.conversion_directory_widget.value = change['new']
    
    def on_load_conversion_files(self, button):
        """Load and display available H5 files for conversion"""
        with self.conversion_status_output:
            clear_output()
            
            directory = self.conversion_directory_widget.value
            if not directory:
                print("❌ Please specify a directory")
                return
            
            directory_path = Path(directory)
            if not directory_path.exists():
                print(f"❌ Directory '{directory}' does not exist")
                return
            
            # Find all .h5 files
            h5_files = list(directory_path.glob("*.h5"))
            
            if not h5_files:
                print(f"❌ No .h5 files found in '{directory}'")
                return
            
            print(f"✅ Found {len(h5_files)} .h5 files")
            
            # Create conversion factor widgets for each file
            self.create_conversion_widgets(h5_files)
    
    def create_conversion_widgets(self, h5_files):
        """Create conversion factor input widgets for each H5 file"""
        
        # Clear existing widgets
        self.conversion_factor_widgets = {}
        
        # Get default conversion factors
        default_factors = self.get_default_conversion_factors()
        
        # Create widgets for each file
        file_widgets = []
        
        for h5_file in h5_files:
            filename = h5_file.name
            
            # Extract property/variable name from filename
            property_name = self.extract_property_name(filename)
            
            # Get default conversion factor
            default_factor = default_factors.get(property_name.upper(), default_factors['DEFAULT'])
            
            # Create description based on property type
            description = self.get_conversion_description(property_name.upper())
            
            # Create conversion factor widget
            factor_widget = widgets.FloatText(
                value=default_factor,
                description=f'{property_name}:',
                style={'description_width': '100px'},
                layout=widgets.Layout(width='200px'),
                step=0.000001,
                tooltip=f'Conversion factor for {property_name} ({description})'
            )
            
            # Store widget reference
            self.conversion_factor_widgets[filename] = factor_widget
            
            # Create file info widget
            file_info = widgets.HTML(
                value=f"<b>📄 {filename}</b><br/><small>{description}</small>",
                layout=widgets.Layout(width='300px')
            )
            
            # Create horizontal layout for this file
            file_row = widgets.HBox([
                file_info,
                factor_widget
            ], layout=widgets.Layout(margin='5px 0', padding='5px', border='1px solid #ddd'))
            
            file_widgets.append(file_row)
        
        # Update the conversion files widget
        self.conversion_files_widget.children = file_widgets
        
        with self.conversion_status_output:
            print(f"✅ Loaded {len(h5_files)} files with default conversion factors")
    
    def extract_property_name(self, filename):
        """Extract property/variable name from H5 filename"""
        # Remove common prefixes and suffixes
        name = filename.replace('batch_spatial_properties_', '').replace('batch_timeseries_data_', '')
        name = name.replace('.h5', '')
        
        # Handle special cases
        if 'spatial' in filename.lower():
            return name
        elif 'timeseries' in filename.lower():
            return name
        else:
            return name
    
    def get_conversion_description(self, property_name):
        """Get conversion description for a property"""
        descriptions = {
            'PRES': 'kPa → psi (×0.145038)',
            'BHP': 'kPa → psi (×0.145038)',
            'THP': 'kPa → psi (×0.145038)',
            'PRESSURE': 'kPa → psi (×0.145038)',
            
            'GASRATSC': 'm³/day → ft³/day (×35.3147)',
            'WATRATSC': 'm³/day → ft³/day (×35.3147)',
            'OILRATSC': 'm³/day → ft³/day (×35.3147)',
            'GASRATE': 'm³/day → ft³/day (×35.3147)',
            'WATRATE': 'm³/day → ft³/day (×35.3147)',
            'OILRATE': 'm³/day → ft³/day (×35.3147)',
            
            'TEMP': '°C → °F (×1.8 + 32)',
            'TEMPERATURE': '°C → °F (×1.8 + 32)',
            
            'PERMX': 'mD → D (×0.001)',
            'PERMY': 'mD → D (×0.001)',
            'PERMZ': 'mD → D (×0.001)',
            'PERM': 'mD → D (×0.001)',
            
            'PORO': 'No conversion (dimensionless)',
            'POROSITY': 'No conversion (dimensionless)',
            
            'SWAT': 'No conversion (dimensionless)',
            'SGAS': 'No conversion (dimensionless)',
            'SOIL': 'No conversion (dimensionless)',
        }
        
        return descriptions.get(property_name, 'Custom conversion factor')
    
    def on_reset_conversion_factors(self, button):
        """Reset all conversion factors to defaults"""
        default_factors = self.get_default_conversion_factors()
        
        for filename, widget in self.conversion_factor_widgets.items():
            property_name = self.extract_property_name(filename)
            default_factor = default_factors.get(property_name.upper(), default_factors['DEFAULT'])
            widget.value = default_factor
        
        with self.conversion_status_output:
            clear_output()
            print("✅ Reset all conversion factors to defaults")
    
    def on_convert_units(self, button):
        """Convert units in all H5 files using specified conversion factors"""
        with self.conversion_output:
            clear_output()
            
            if not self.conversion_factor_widgets:
                print("❌ No files loaded. Please load H5 files first.")
                return
            
            directory = Path(self.conversion_directory_widget.value)
            total_files = len(self.conversion_factor_widgets)
            successful_conversions = 0
            
            print(f"🔄 Starting unit conversion for {total_files} files...")
            print("=" * 60)
            
            for i, (filename, factor_widget) in enumerate(self.conversion_factor_widgets.items()):
                # Update progress
                progress = int((i / total_files) * 100)
                self.conversion_progress_widget.value = progress
                
                file_path = directory / filename
                conversion_factor = factor_widget.value
                property_name = self.extract_property_name(filename)
                
                print(f"📄 Converting {filename}...")
                print(f"   Property: {property_name}")
                print(f"   Factor: {conversion_factor}")
                
                # Perform conversion
                if self.convert_h5_file(file_path, conversion_factor, property_name):
                    successful_conversions += 1
                    print(f"   ✅ Success")
                else:
                    print(f"   ❌ Failed")
                
                print()
            
            # Final progress
            self.conversion_progress_widget.value = 100
            
            print("=" * 60)
            print(f"✅ CONVERSION COMPLETE")
            print(f"📊 Successfully converted: {successful_conversions}/{total_files} files")
            print("=" * 60)
    
    def convert_h5_file(self, file_path, conversion_factor, property_name):
        """Convert units in a single H5 file"""
        try:
            # Skip conversion for inactive cell locations file (no units to convert)
            if 'inactive_cell_locations' in str(file_path):
                print(f"   ⏭️ Skipping inactive cell locations file (no units to convert)")
                return True
            
            # Create backup
            backup_path = file_path.with_suffix(f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.h5')
            
            # Read original data and structure
            with h5py.File(file_path, 'r') as f_in:
                # Check if 'data' dataset exists
                if 'data' not in f_in:
                    print(f"   ❌ Error: 'data' dataset not found in {file_path}")
                    print(f"   Available datasets: {list(f_in.keys())}")
                    return False
                
                original_data = f_in['data'][:]
                
                # Copy root-level metadata
                root_metadata = {}
                for key in f_in.attrs.keys():
                    root_metadata[key] = f_in.attrs[key]
                
                # Extract metadata group if it exists (critical for visualizer)
                metadata_group_data = None
                if 'metadata' in f_in and isinstance(f_in['metadata'], h5py.Group):
                    metadata_group_data = self._extract_group_data(f_in['metadata'])
                
                # Copy other datasets and groups
                other_datasets = {}
                other_groups = {}
                for key in f_in.keys():
                    if key not in ['data', 'metadata']:  # Handle metadata separately
                        try:
                            if isinstance(f_in[key], h5py.Dataset):
                                other_datasets[key] = f_in[key][:]
                            elif isinstance(f_in[key], h5py.Group):
                                other_groups[key] = f_in[key]
                        except Exception as e:
                            print(f"   ⚠️ Warning: Could not process '{key}': {e}")
                            continue
            
            # Create backup
            with h5py.File(backup_path, 'w') as f_backup:
                f_backup.create_dataset('data', data=original_data, dtype=original_data.dtype)
                
                # Copy root metadata
                for key, value in root_metadata.items():
                    f_backup.attrs[key] = value
                
                # Recreate metadata group if it existed
                if metadata_group_data is not None:
                    self._recreate_metadata_group(f_backup, metadata_group_data)
                
                # Copy other datasets
                for key, value in other_datasets.items():
                    if hasattr(value, 'dtype'):
                        f_backup.create_dataset(key, data=value, dtype=value.dtype)
                    else:
                        f_backup.create_dataset(key, data=value)
                
                # Copy other groups
                for key, group in other_groups.items():
                    try:
                        self._copy_group_recursively(f_backup, group, key)
                    except Exception as e:
                        print(f"   ⚠️ Warning: Could not copy group '{key}': {e}")
            
            # Convert data
            if property_name.upper() in ['TEMP', 'TEMPERATURE'] and conversion_factor == 1.8:
                # Special handling for Celsius to Fahrenheit: °F = °C × 1.8 + 32
                converted_data = original_data * conversion_factor + 32
            else:
                # Standard multiplication
                converted_data = original_data * conversion_factor
            
            # Write converted data
            with h5py.File(file_path, 'w') as f_out:
                # Write converted data
                f_out.create_dataset('data', data=converted_data, dtype=converted_data.dtype)
                
                # Copy root metadata
                for key, value in root_metadata.items():
                    f_out.attrs[key] = value
                
                # Recreate metadata group if it existed (CRITICAL for visualizer)
                if metadata_group_data is not None:
                    self._recreate_metadata_group(f_out, metadata_group_data)
                
                # Copy other datasets
                for key, value in other_datasets.items():
                    if hasattr(value, 'dtype'):
                        f_out.create_dataset(key, data=value, dtype=value.dtype)
                    else:
                        f_out.create_dataset(key, data=value)
                
                # Copy other groups
                for key, group in other_groups.items():
                    try:
                        self._copy_group_recursively(f_out, group, key)
                    except Exception as e:
                        print(f"   ⚠️ Warning: Could not copy group '{key}': {e}")
                
                # Add conversion metadata to root level
                f_out.attrs['units_converted'] = True
                f_out.attrs['conversion_factor'] = conversion_factor
                f_out.attrs['conversion_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f_out.attrs['property_name'] = property_name
                f_out.attrs['backup_file'] = str(backup_path)
                
                if property_name.upper() in ['TEMP', 'TEMPERATURE'] and conversion_factor == 1.8:
                    f_out.attrs['conversion_type'] = 'celsius_to_fahrenheit'
                else:
                    f_out.attrs['conversion_type'] = 'multiplication'
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            return False
    
    def _copy_group_recursively(self, dest_file, source_group, dest_name):
        """Recursively copy HDF5 group with better error handling"""
        try:
            # Create the destination group
            dest_group = dest_file.create_group(dest_name)
            
            # Copy attributes
            for attr_name, attr_value in source_group.attrs.items():
                try:
                    dest_group.attrs[attr_name] = attr_value
                except Exception:
                    # Skip problematic attributes
                    continue
            
            # Copy datasets and subgroups
            for key in source_group.keys():
                try:
                    if isinstance(source_group[key], h5py.Dataset):
                        # Copy dataset
                        data = source_group[key][:]
                        dest_group.create_dataset(key, data=data, dtype=source_group[key].dtype)
                        
                        # Copy dataset attributes
                        for attr_name, attr_value in source_group[key].attrs.items():
                            try:
                                dest_group[key].attrs[attr_name] = attr_value
                            except Exception:
                                continue
                                
                    elif isinstance(source_group[key], h5py.Group):
                        # Recursively copy subgroup
                        self._copy_group_recursively(dest_group, source_group[key], key)
                        
                except Exception:
                    # Skip problematic items
                    continue
                    
        except Exception:
            # If group creation fails, skip silently for metadata groups
            if dest_name != 'metadata':
                raise
    
    def _extract_group_data(self, group):
        """Extract all data from an HDF5 group for later recreation"""
        group_data = {
            'attributes': {},
            'datasets': {},
            'subgroups': {}
        }
        
        # Extract group attributes
        for attr_name in group.attrs.keys():
            try:
                group_data['attributes'][attr_name] = group.attrs[attr_name]
            except Exception:
                continue
        
        # Extract datasets
        for key in group.keys():
            try:
                if isinstance(group[key], h5py.Dataset):
                    dataset_data = {
                        'data': group[key][:],
                        'dtype': group[key].dtype,
                        'attributes': {}
                    }
                    # Extract dataset attributes
                    for attr_name in group[key].attrs.keys():
                        try:
                            dataset_data['attributes'][attr_name] = group[key].attrs[attr_name]
                        except Exception:
                            continue
                    group_data['datasets'][key] = dataset_data
                    
                elif isinstance(group[key], h5py.Group):
                    # Recursively extract subgroups
                    group_data['subgroups'][key] = self._extract_group_data(group[key])
            except Exception:
                continue
        
        return group_data
    
    def _recreate_metadata_group(self, h5_file, group_data):
        """Recreate metadata group from extracted data"""
        try:
            # Create the metadata group
            meta_group = h5_file.create_group('metadata')
            
            # Set group attributes
            for attr_name, attr_value in group_data['attributes'].items():
                try:
                    meta_group.attrs[attr_name] = attr_value
                except Exception:
                    continue
            
            # Create datasets
            for dataset_name, dataset_data in group_data['datasets'].items():
                try:
                    dataset = meta_group.create_dataset(
                        dataset_name, 
                        data=dataset_data['data'], 
                        dtype=dataset_data['dtype']
                    )
                    # Set dataset attributes
                    for attr_name, attr_value in dataset_data['attributes'].items():
                        try:
                            dataset.attrs[attr_name] = attr_value
                        except Exception:
                            continue
                except Exception:
                    continue
            
            # Create subgroups recursively
            for subgroup_name, subgroup_data in group_data['subgroups'].items():
                try:
                    self._recreate_subgroup(meta_group, subgroup_name, subgroup_data)
                except Exception:
                    continue
                    
        except Exception as e:
            print(f"   ⚠️ Warning: Could not recreate metadata group: {e}")
    
    def _recreate_subgroup(self, parent_group, group_name, group_data):
        """Recursively recreate subgroups"""
        subgroup = parent_group.create_group(group_name)
        
        # Set group attributes
        for attr_name, attr_value in group_data['attributes'].items():
            try:
                subgroup.attrs[attr_name] = attr_value
            except Exception:
                continue
        
        # Create datasets
        for dataset_name, dataset_data in group_data['datasets'].items():
            try:
                dataset = subgroup.create_dataset(
                    dataset_name, 
                    data=dataset_data['data'], 
                    dtype=dataset_data['dtype']
                )
                # Set dataset attributes
                for attr_name, attr_value in dataset_data['attributes'].items():
                    try:
                        dataset.attrs[attr_name] = attr_value
                    except Exception:
                        continue
            except Exception:
                continue
        
        # Create nested subgroups
        for nested_name, nested_data in group_data['subgroups'].items():
            try:
                self._recreate_subgroup(subgroup, nested_name, nested_data)
            except Exception:
                continue
    
    def _analyze_sr3_date_ranges(self):
        """
        Analyze date ranges across all SR3 files (integrated from check_all_sr3_dates.py)
        Returns analysis results dictionary
        """
        try:
            # Get reference date range from user widgets
            reference_start = self.reference_start_date_widget.value.strftime('%Y-%m-%d')
            reference_end = self.reference_end_date_widget.value.strftime('%Y-%m-%d')
            reference_range = f"{reference_start} → {reference_end}"
            
            # Track results
            matching_files = []
            different_files = []
            error_files = []
            
            # Analyze each file
            for i, sr3_file in enumerate(self.sr3_files):
                filename = sr3_file.name
                
                start_date, end_date, success = self._get_date_range_from_sr3(sr3_file)
                
                if not success:
                    error_files.append(filename)
                    continue
                
                current_range = f"{start_date} → {end_date}"
                
                if current_range == reference_range:
                    matching_files.append((filename, current_range))
                else:
                    different_files.append((filename, current_range))
            
            # Calculate unique date ranges
            unique_ranges = {}
            for filename, date_range in matching_files + different_files:
                if date_range not in unique_ranges:
                    unique_ranges[date_range] = []
                unique_ranges[date_range].append(filename)
            
            return {
                'reference_range': reference_range,
                'matching_files': matching_files,
                'different_files': different_files,
                'error_files': error_files,
                'unique_ranges': unique_ranges,
                'total_files': len(self.sr3_files)
            }
            
        except Exception as e:
            print(f"❌ Error during date range analysis: {e}")
            return None
    
    def _get_date_range_from_sr3(self, sr3_file_path):
        """
        Extract date range from SR3 file
        Returns tuple: (start_date, end_date, success)
        """
        try:
            with h5py.File(sr3_file_path, 'r') as f:
                
                # Load master time table
                if 'General/MasterTimeTable' in f:
                    time_table = f['General/MasterTimeTable'][...]
                    
                    master_dates = time_table['Date'] 
                    master_days = time_table['Offset in days']
                    
                    # Convert first and last dates
                    start_date_val = master_dates[0]
                    end_date_val = master_dates[-1]
                    
                    def convert_date(date_val, day_val):
                        try:
                            if date_val > 20000000:  # Format like 20030101.0
                                date_str = str(int(date_val))
                                if len(date_str) >= 8:
                                    year = int(date_str[:4])
                                    month = int(date_str[4:6])
                                    day = int(date_str[6:8])
                                    date_obj = datetime(year, month, day)
                                    return date_obj.strftime('%Y-%m-%d')
                                else:
                                    return f"Day {day_val:.0f}"
                            elif date_val > 40000:  # Excel format
                                ref_date = datetime(1900, 1, 1)
                                date_obj = ref_date + timedelta(days=date_val - 2)
                                return date_obj.strftime('%Y-%m-%d')
                            else:
                                return f"Day {day_val:.0f}"
                        except:
                            return f"Day {day_val:.0f}"
                    
                    start_date = convert_date(start_date_val, master_days[0])
                    end_date = convert_date(end_date_val, master_days[-1])
                    
                    return start_date, end_date, True
                else:
                    return None, None, False
                    
        except Exception as e:
            return None, None, False
    
    def _display_date_analysis_results(self, results):
        """Display the date range analysis results"""
        print("=" * 80)
        print("📊 DATE RANGE ANALYSIS RESULTS:")
        print("=" * 80)
        
        reference_range = results['reference_range']
        matching_files = results['matching_files']
        different_files = results['different_files']
        error_files = results['error_files']
        unique_ranges = results['unique_ranges']
        total_files = results['total_files']
        
        print(f"🎯 Reference date range: {reference_range}")
        print(f"📁 Total files analyzed: {total_files}")
        
        # Matching files
        print(f"\n✅ Files with matching date range: {len(matching_files)}")
        if matching_files and len(matching_files) <= 5:
            for filename, date_range in matching_files[:5]:
                print(f"    ✅ {filename}")
        elif matching_files:
            print(f"    ✅ {matching_files[0][0]} ... and {len(matching_files)-1} more")
        
        # Different files
        if different_files:
            print(f"\n🔍 Files with DIFFERENT date ranges: {len(different_files)}")
            print("    ⚠️ These files may cause synchronization issues:")
            for filename, date_range in different_files[:10]:  # Show first 10
                print(f"    🔍 {filename} → {date_range}")
            if len(different_files) > 10:
                print(f"    ... and {len(different_files)-10} more files with different ranges")
        
        # Error files
        if error_files:
            print(f"\n❌ Files with ERRORS: {len(error_files)}")
            for filename in error_files[:5]:  # Show first 5
                print(f"    ❌ {filename}")
            if len(error_files) > 5:
                print(f"    ... and {len(error_files)-5} more error files")
        
        # Unique ranges summary
        print(f"\n📈 UNIQUE DATE RANGES FOUND: {len(unique_ranges)}")
        for date_range, files in unique_ranges.items():
            print(f"    📅 {date_range} → {len(files)} files")
        
        # Recommendations
        if different_files or error_files:
            print(f"\n💡 RECOMMENDATIONS:")
            if different_files:
                print(f"    ⚠️ {len(different_files)} files have different date ranges")
                print(f"    🔧 Consider checking these files for data consistency")
            if error_files:
                print(f"    ❌ {len(error_files)} files had read errors")
                print(f"    🔧 These files may be corrupted or incompatible")
            print(f"    📊 The extraction will proceed but may have synchronization issues")
        else:
            print(f"\n✅ EXCELLENT: All files have consistent date ranges!")
            print(f"    🎯 Perfect for synchronized batch extraction")
        
        print("=" * 80)
    
    def on_rerun_date_analysis(self, button):
        """Re-run date range analysis with updated reference dates"""
        with self.status_output:
            clear_output()
            
            if not hasattr(self, 'sr3_files') or not self.sr3_files:
                print("❌ No SR3 files loaded. Please load files first.")
                return
            
            reference_start = self.reference_start_date_widget.value.strftime('%Y-%m-%d')
            reference_end = self.reference_end_date_widget.value.strftime('%Y-%m-%d')
            
            print(f"🔄 Re-analyzing {len(self.sr3_files)} files with new reference range:")
            print(f"🎯 New reference: {reference_start} → {reference_end}")
            print("-" * 60)
            
            # Run analysis with new reference dates
            date_analysis_results = self._analyze_sr3_date_ranges()
            
            if date_analysis_results:
                self._display_date_analysis_results(date_analysis_results)
                self.last_date_analysis = date_analysis_results
                print(f"\n✅ Date analysis completed with updated reference range")


def create_batch_dashboard():
    """
    Create and display the batch processing Jupyter interactive dashboard
    
    Usage in Jupyter cell:
    ```python
    from interactive_sr3_extractor import create_batch_dashboard
    dashboard = create_batch_dashboard()
    ```
    """
    
    dashboard = BatchSR3Extractor()
    dashboard.display()
    return dashboard


# Backwards compatibility aliases
def create_jupyter_dashboard():
    """Create batch dashboard (backwards compatibility)"""
    return create_batch_dashboard()


def create_dashboard():
    """Create batch dashboard (backwards compatibility)"""
    return create_batch_dashboard()


if __name__ == "__main__":
    # For Jupyter usage - create and display dashboard
    dashboard = create_batch_dashboard()
    print("🎉 Batch processing dashboard created!")
    print("The dashboard should appear above this message.")
    print("\nTo use in other cells:")
    print("dashboard = create_batch_dashboard()")


def load_batch_data_example(output_folder):
    """
    Example function showing how to load the saved batch data
    
    Parameters:
    -----------
    output_folder : str or Path
        Path to the output folder containing the .h5 files
        
    Returns:
    --------
    dict : Dictionary containing loaded data
    """
    
    output_folder = Path(output_folder)
    loaded_data = {}
    
    print(f"📁 Loading batch data from: {output_folder}")
    
    # Load inactive cell locations
    inactive_file = output_folder / "inactive_cell_locations.h5"
    if inactive_file.exists():
        with h5py.File(inactive_file, 'r') as f:
            inactive_mask = f['inactive_mask'][...]
            active_mask = f['active_mask'][...]
            
            print(f"🔴 Inactive cells mask shape: {inactive_mask.shape}")
            print(f"🟢 Active cells mask shape: {active_mask.shape}")
            
            if inactive_mask.ndim == 4:
                # Per-case format
                n_cases = inactive_mask.shape[0]
                print(f"   📊 Format: Per-case masks (N_cases, Nx, Ny, Nz)")
                print(f"   🗂️ Cases: {n_cases}")
                print(f"   📈 Total cells across all cases: {inactive_mask.size:,}")
                print(f"   🟢 Total active cells: {np.sum(active_mask):,}")
                print(f"   🔴 Total inactive cells: {np.sum(inactive_mask):,}")
                
                # Show per-case statistics
                for case_idx in range(min(n_cases, 3)):  # Show first 3 cases
                    case_active = np.sum(active_mask[case_idx])
                    case_inactive = np.sum(inactive_mask[case_idx])
                    print(f"     Case {case_idx}: Active: {case_active:,}, Inactive: {case_inactive:,}")
                if n_cases > 3:
                    print(f"     ... and {n_cases-3} more cases")
            else:
                # Legacy format
                print(f"   📊 Format: Legacy single mask (Nx, Ny, Nz)")
                print(f"   Total cells: {inactive_mask.size:,}")
                print(f"   Active cells: {np.sum(active_mask):,}")
                print(f"   Inactive cells: {np.sum(inactive_mask):,}")
            
            loaded_data['inactive_mask'] = inactive_mask
            loaded_data['active_mask'] = active_mask
    
    # Load spatial properties
    spatial_files = list(output_folder.glob("batch_spatial_properties_*.h5"))
    if spatial_files:
        print(f"\n📊 Found {len(spatial_files)} spatial property files:")
        loaded_data['spatial'] = {}
        
        for file_path in spatial_files:
            prop_name = file_path.stem.replace('batch_spatial_properties_', '')
            
            with h5py.File(file_path, 'r') as f:
                data = f['data'][...]
                meta = f['metadata']
                
                print(f"   ✅ {prop_name}: {data.shape} - {file_path.name}")
                
                loaded_data['spatial'][prop_name] = {
                    'data': data,
                    'timesteps': meta['timesteps'][...],
                    'n_cases': meta.attrs['n_cases'],
                    'grid_dims': meta.attrs['grid_dimensions']
                }
    
    # Load time series data
    timeseries_files = list(output_folder.glob("batch_timeseries_data_*.h5"))
    if timeseries_files:
        print(f"\n🛢️ Found {len(timeseries_files)} time series variable files:")
        loaded_data['timeseries'] = {}
        
        for file_path in timeseries_files:
            var_name = file_path.stem.replace('batch_timeseries_data_', '')
            
            with h5py.File(file_path, 'r') as f:
                data = f['data'][...]
                meta = f['metadata']
                
                print(f"   ✅ {var_name}: {data.shape} - {file_path.name}")
                
                loaded_data['timeseries'][var_name] = {
                    'data': data,
                    'timesteps': meta['timesteps'][...],
                    'dates': [d.decode() for d in meta['dates'][...]],
                    'well_names': [w.decode() for w in meta['well_names'][...]],
                    'n_cases': meta.attrs['n_cases']
                }
    
    print(f"\n✅ Loaded {len(loaded_data)} data categories")
    return loaded_data


# Example usage in comments
"""
# Example usage after batch extraction:

# Load all batch data
data = load_batch_data_example('sr3_batch_output')

# Access spatial data
if 'spatial' in data:
    poros_data = data['spatial']['POROS']['data']  # Shape: (N_cases, N_timesteps, Nx, Ny, Nz)
    print(f"Porosity data shape: {poros_data.shape}")
    
    # Get porosity for first case, first timestep
    case0_poros = poros_data[0, 0, :, :, :]
    print(f"Case 0 porosity range: [{case0_poros.min():.4f}, {case0_poros.max():.4f}]")

# Access time series data  
if 'timeseries' in data:
    bhp_data = data['timeseries']['BHP']['data']  # Shape: (N_cases, N_timesteps, N_wells)
    print(f"BHP data shape: {bhp_data.shape}")
    
    # Get BHP for all cases, first timestep, first well
    all_cases_bhp = bhp_data[:, 0, 0]
    print(f"All cases BHP for first well: {all_cases_bhp}")

# Access inactive cell locations
if 'inactive_mask' in data:
    inactive_mask = data['inactive_mask']  
    active_mask = data['active_mask']      
    
    if inactive_mask.ndim == 4:
        # Per-case format: Shape (N_cases, Nx, Ny, Nz), True=inactive
        print(f"Per-case inactive masks: {inactive_mask.shape}")
        
        # Access mask for specific case
        case_0_inactive = inactive_mask[0]  # Shape: (Nx, Ny, Nz) for case 0
        case_0_active = active_mask[0]      # Shape: (Nx, Ny, Nz) for case 0
        
        # Count inactive cells for each case in layer 10
        k_layer = 10
        for case_idx in range(inactive_mask.shape[0]):
            inactive_count = np.sum(inactive_mask[case_idx, :, :, k_layer])
            print(f"Case {case_idx}, Layer {k_layer+1}: {inactive_count} inactive cells")
            
        # Compare geological realizations between cases
        case_0_pattern = inactive_mask[0, :, :, k_layer]
        case_1_pattern = inactive_mask[1, :, :, k_layer] 
        differences = np.sum(case_0_pattern != case_1_pattern)
        print(f"Differences between case 0 and 1 in layer {k_layer+1}: {differences} cells")
        
    else:
        # Legacy format: Shape (Nx, Ny, Nz), True=inactive (same for all cases)
        print(f"Legacy single mask: {inactive_mask.shape}")
        
        # Count inactive cells in each layer (applies to all cases)
        for k in range(inactive_mask.shape[2]):
            inactive_count = np.sum(inactive_mask[:, :, k])
            print(f"Layer {k+1}: {inactive_count} inactive cells")
"""
# %%
