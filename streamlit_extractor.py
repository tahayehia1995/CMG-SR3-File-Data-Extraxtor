"""
Streamlit wrapper for BatchSR3Extractor
Adapts the Jupyter-based extractor for Streamlit web application
"""

import streamlit as st
from pathlib import Path
import tempfile
import os
import sys
import h5py
import numpy as np
from io import StringIO
from interactive_sr3_extractor import (
    BatchSR3Extractor, 
    analyze_common_dates_across_files,
    filter_yearly_dates,
    filter_monthly_dates_flexible,
    convert_days_to_dates
)


def inspect_h5_structure(file_path):
    """
    Inspect H5 file structure and return formatted information
    
    Args:
        file_path: Path to H5 file
        
    Returns:
        dict: Dictionary containing file structure information
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return None
    
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    
    info = {
        'filename': file_path.name,
        'file_size_mb': file_size_mb,
        'file_type': None,  # 'spatial' or 'timeseries'
        'structure': [],
        'root_attrs': {},
        'metadata_attrs': {},
        'datasets': {},
        'data_shape': None,
        'data_dtype': None,
        'compression': None,
        'summary': {}
    }
    
    try:
        with h5py.File(file_path, 'r') as f:
            # Determine file type
            if 'data' in f:
                data_shape = f['data'].shape
                info['data_shape'] = data_shape
                info['data_dtype'] = str(f['data'].dtype)
                
                # Check compression
                if f['data'].compression:
                    info['compression'] = f"{f['data'].compression} (level {f['data'].compression_opts})"
                
                if len(data_shape) == 5:
                    info['file_type'] = 'spatial'
                elif len(data_shape) == 3:
                    info['file_type'] = 'timeseries'
            
            # Root attributes
            info['root_attrs'] = dict(f.attrs)
            
            # Walk through structure and build tree
            structure_items = []
            visited_names = set()
            
            def visit(name, obj):
                # Skip if already visited (visititems may visit root level items)
                if name in visited_names:
                    return
                visited_names.add(name)
                
                path_parts = name.split('/')
                depth = len(path_parts) - 1
                indent = "  " * depth
                
                if isinstance(obj, h5py.Dataset):
                    info['datasets'][name] = {
                        'shape': obj.shape,
                        'dtype': str(obj.dtype),
                        'compression': obj.compression if obj.compression else None
                    }
                    comp_str = f", compression={obj.compression}" if obj.compression else ""
                    if depth == 0:
                        structure_items.append(f"├── {name} (dataset: shape={obj.shape}, dtype={obj.dtype}{comp_str})")
                    else:
                        structure_items.append(f"{indent}├── {path_parts[-1]} (dataset: shape={obj.shape}, dtype={obj.dtype}{comp_str})")
                elif isinstance(obj, h5py.Group):
                    if name == 'metadata':
                        info['metadata_attrs'] = dict(obj.attrs)
                    if depth == 0:
                        structure_items.append(f"├── {name}/ (group)")
                    else:
                        structure_items.append(f"{indent}├── {path_parts[-1]}/ (group)")
            
            f.visititems(visit)
            
            # Build summary based on file type
            if info['file_type'] == 'spatial' and 'metadata' in f:
                meta = f['metadata']
                info['summary'] = {
                    'property_name': meta.attrs.get('property_name', 'Unknown'),
                    'n_cases': meta.attrs.get('n_cases', 0),
                    'n_timesteps': meta.attrs.get('n_timesteps', 0),
                    'nx': meta.attrs.get('nx', 0),
                    'ny': meta.attrs.get('ny', 0),
                    'nz': meta.attrs.get('nz', 0),
                    'active_cells': meta.attrs.get('active_cells', 0),
                    'total_cells': meta.attrs.get('total_cells', 0)
                }
            elif info['file_type'] == 'timeseries' and 'metadata' in f:
                meta = f['metadata']
                well_names = [w.decode() if isinstance(w, bytes) else w for w in meta['well_names'][...]] if 'well_names' in meta else []
                info['summary'] = {
                    'variable_name': meta.attrs.get('variable_name', 'Unknown'),
                    'n_cases': meta.attrs.get('n_cases', 0),
                    'n_timesteps': meta.attrs.get('n_timesteps', 0),
                    'n_wells': meta.attrs.get('n_wells', 0),
                    'well_names': well_names
                }
            
            # Format structure tree with root
            info['structure'] = [f"/{file_path.name}"] + structure_items
            
    except Exception as e:
        info['error'] = str(e)
    
    return info


def inspect_sr3_structure(file_path):
    """
    Deeply analyze SR3 file structure, metadata, and data organization
    
    Args:
        file_path: Path to SR3 file
        
    Returns:
        dict: Comprehensive dictionary containing SR3 file structure information
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return None
    
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    
    info = {
        'filename': file_path.name,
        'file_size_mb': file_size_mb,
        'file_metadata': {},
        'top_level_groups': [],
        'general_group': {},
        'spatial_properties': {},
        'time_series': {},
        'tables_group': {},
        'grid_structure': {},
        'structure_tree': []
    }
    
    try:
        with h5py.File(file_path, 'r') as f:
            # File metadata from root attributes
            info['file_metadata'] = {
                'sr3_version': f.attrs.get('SR3 Version', b'').decode() if isinstance(f.attrs.get('SR3 Version', b''), bytes) else str(f.attrs.get('SR3 Version', '')),
                'simulator_name': f.attrs.get('Simulator Name', b'').decode() if isinstance(f.attrs.get('Simulator Name', b''), bytes) else str(f.attrs.get('Simulator Name', '')),
                'simulator_version': f.attrs.get('Simulator Version', b'').decode() if isinstance(f.attrs.get('Simulator Version', b''), bytes) else str(f.attrs.get('Simulator Version', '')),
                'file_id': f.attrs.get('id', b'').decode() if isinstance(f.attrs.get('id', b''), bytes) else str(f.attrs.get('id', '')),
                'file_format': f.attrs.get('File', b'').decode() if isinstance(f.attrs.get('File', b''), bytes) else str(f.attrs.get('File', ''))
            }
            
            # Top-level groups
            info['top_level_groups'] = list(f.keys())
            
            # === GENERAL GROUP ===
            if 'General' in f:
                general_group = f['General']
                general_keys = list(general_group.keys())
                info['general_group'] = {
                    'subgroups': general_keys,
                    'master_time_table': None,
                    'component_table': None,
                    'units_table': None
                }
                
                # MasterTimeTable analysis
                if 'MasterTimeTable' in general_group:
                    mt = general_group['MasterTimeTable'][...]
                    info['general_group']['master_time_table'] = {
                        'shape': mt.shape,
                        'dtype': str(mt.dtype),
                        'field_names': mt.dtype.names if hasattr(mt.dtype, 'names') else None,
                        'num_timesteps': len(mt),
                        'first_timestep': {
                            'index': int(mt[0]['Index']) if 'Index' in mt.dtype.names else None,
                            'days': float(mt[0]['Offset in days']) if 'Offset in days' in mt.dtype.names else None,
                            'date': float(mt[0]['Date']) if 'Date' in mt.dtype.names else None
                        },
                        'last_timestep': {
                            'index': int(mt[-1]['Index']) if 'Index' in mt.dtype.names else None,
                            'days': float(mt[-1]['Offset in days']) if 'Offset in days' in mt.dtype.names else None,
                            'date': float(mt[-1]['Date']) if 'Date' in mt.dtype.names else None
                        },
                        'sample_data': []
                    }
                    # Add sample rows
                    for i in range(min(5, len(mt))):
                        row_data = {}
                        if 'Index' in mt.dtype.names:
                            row_data['index'] = int(mt[i]['Index'])
                        if 'Offset in days' in mt.dtype.names:
                            row_data['days'] = float(mt[i]['Offset in days'])
                        if 'Date' in mt.dtype.names:
                            row_data['date'] = float(mt[i]['Date'])
                        info['general_group']['master_time_table']['sample_data'].append(row_data)
                
                # ComponentTable
                if 'ComponentTable' in general_group:
                    comp_table = general_group['ComponentTable'][...]
                    info['general_group']['component_table'] = {
                        'num_components': len(comp_table),
                        'components': []
                    }
                    if 'Name' in comp_table.dtype.names:
                        for i in range(min(10, len(comp_table))):
                            name = comp_table[i]['Name']
                            if isinstance(name, bytes):
                                name = name.decode()
                            info['general_group']['component_table']['components'].append(name)
            
            # === SPATIAL PROPERTIES ===
            if 'SpatialProperties' in f:
                sp_group = f['SpatialProperties']
                sp_timesteps = sorted([k for k in sp_group.keys() if k.isdigit()])
                
                info['spatial_properties'] = {
                    'num_timesteps': len(sp_timesteps),
                    'timesteps': sp_timesteps[:20],  # First 20 timesteps
                    'all_timesteps': sp_timesteps,
                    'available_properties': [],
                    'grid_structure': None
                }
                
                # Analyze first timestep for structure
                if sp_timesteps:
                    first_ts = sp_timesteps[0]
                    ts_group = sp_group[first_ts]
                    ts_keys = list(ts_group.keys())
                    
                    # Separate GRID from properties
                    grid_keys = []
                    property_keys = []
                    for key in ts_keys:
                        if key == 'GRID':
                            grid_keys.append(key)
                        else:
                            property_keys.append(key)
                    
                    info['spatial_properties']['available_properties'] = sorted(property_keys)
                    
                    # Grid structure analysis
                    if 'GRID' in ts_group:
                        grid_subgroup = ts_group['GRID']
                        grid_datasets = list(grid_subgroup.keys())
                        
                        grid_info = {
                            'datasets': grid_datasets,
                            'dimensions': {},
                            'active_cells': None,
                            'total_cells': None
                        }
                        
                        # Get grid dimensions
                        if 'IGNTID' in grid_subgroup:
                            igntid = grid_subgroup['IGNTID'][...]
                            grid_info['dimensions']['ni'] = int(igntid.max()) if len(igntid) > 0 else None
                        
                        if 'IGNTJD' in grid_subgroup:
                            igntjd = grid_subgroup['IGNTJD'][...]
                            grid_info['dimensions']['nj'] = int(igntjd.max()) if len(igntjd) > 0 else None
                        
                        if 'IGNTKD' in grid_subgroup:
                            igntkd = grid_subgroup['IGNTKD'][...]
                            grid_info['dimensions']['nk'] = int(igntkd.max()) if len(igntkd) > 0 else None
                        
                        # Active cell mapping
                        if 'IPSTCS' in grid_subgroup:
                            ipstcs = grid_subgroup['IPSTCS'][...]
                            grid_info['active_cells'] = int(np.count_nonzero(ipstcs > 0))
                            grid_info['total_cells'] = len(ipstcs)
                            grid_info['inactive_cells'] = int(np.count_nonzero(ipstcs == 0))
                        
                        # Sample property analysis
                        if property_keys:
                            sample_prop = property_keys[0]
                            if sample_prop in ts_group:
                                prop_data = ts_group[sample_prop][...]
                                grid_info['sample_property'] = {
                                    'name': sample_prop,
                                    'shape': prop_data.shape,
                                    'dtype': str(prop_data.dtype),
                                    'min': float(np.nanmin(prop_data)) if prop_data.size > 0 else None,
                                    'max': float(np.nanmax(prop_data)) if prop_data.size > 0 else None,
                                    'mean': float(np.nanmean(prop_data)) if prop_data.size > 0 else None
                                }
                        
                        info['spatial_properties']['grid_structure'] = grid_info
                        info['grid_structure'] = grid_info
            
            # === TIME SERIES ===
            if 'TimeSeries' in f:
                ts_group = f['TimeSeries']
                ts_subgroups = list(ts_group.keys())
                info['time_series'] = {
                    'subgroups': ts_subgroups,
                    'wells': None
                }
                
                # Wells analysis
                if 'WELLS' in ts_group:
                    wells_group = ts_group['WELLS']
                    wells_keys = list(wells_group.keys())
                    
                    wells_info = {
                        'datasets': wells_keys,
                        'variables': [],
                        'well_names': [],
                        'num_wells': 0,
                        'num_variables': 0,
                        'num_timesteps': 0,
                        'data_shape': None,
                        'data_dtype': None
                    }
                    
                    # Variables
                    if 'Variables' in wells_group:
                        vars_data = wells_group['Variables'][...]
                        vars_list = [v.decode() if isinstance(v, bytes) else str(v) for v in vars_data]
                        wells_info['variables'] = vars_list
                        wells_info['num_variables'] = len(vars_list)
                    
                    # Well names
                    if 'Origins' in wells_group:
                        origins = wells_group['Origins'][...]
                        origins_list = [o.decode() if isinstance(o, bytes) else str(o) for o in origins]
                        wells_info['well_names'] = origins_list
                        wells_info['num_wells'] = len(origins_list)
                    
                    # Timesteps
                    if 'Timesteps' in wells_group:
                        timesteps = wells_group['Timesteps'][...]
                        wells_info['num_timesteps'] = len(timesteps)
                        wells_info['timesteps'] = {
                            'first': int(timesteps[0]) if len(timesteps) > 0 else None,
                            'last': int(timesteps[-1]) if len(timesteps) > 0 else None,
                            'sample': [int(t) for t in timesteps[:10]] if len(timesteps) > 0 else []
                        }
                    
                    # Data array
                    if 'Data' in wells_group:
                        data = wells_group['Data']
                        wells_info['data_shape'] = data.shape
                        wells_info['data_dtype'] = str(data.dtype)
                        wells_info['data_size_mb'] = data.nbytes / (1024 * 1024)
                    
                    info['time_series']['wells'] = wells_info
            
            # === TABLES GROUP ===
            if 'Tables' in f:
                tables_group = f['Tables']
                tables_keys = list(tables_group.keys())
                info['tables_group'] = {
                    'subgroups': tables_keys,
                    'num_tables': len(tables_keys)
                }
            
            # === STRUCTURE TREE ===
            def build_structure_tree(name, obj, depth=0, max_depth=3):
                """Recursively build structure tree"""
                indent = "  " * depth
                items = []
                
                if depth >= max_depth:
                    return items
                
                if isinstance(obj, h5py.Group):
                    items.append(f"{indent}├── {name}/ (Group)")
                    # Limit children to avoid overwhelming output
                    children = list(obj.keys())[:15]
                    for child_name in children:
                        child_obj = obj[child_name]
                        items.extend(build_structure_tree(child_name, child_obj, depth + 1, max_depth))
                    if len(obj.keys()) > 15:
                        items.append(f"{indent}  └── ... and {len(obj.keys()) - 15} more items")
                elif isinstance(obj, h5py.Dataset):
                    shape_str = f"shape={obj.shape}" if obj.shape else "scalar"
                    items.append(f"{indent}├── {name} (Dataset: {shape_str}, dtype={obj.dtype})")
                
                return items
            
            # Build structure tree for main groups
            for group_name in ['General', 'SpatialProperties', 'TimeSeries', 'Tables']:
                if group_name in f:
                    info['structure_tree'].extend(build_structure_tree(group_name, f[group_name], depth=0, max_depth=2))
            
    except Exception as e:
        info['error'] = str(e)
        import traceback
        info['traceback'] = traceback.format_exc()
    
    return info


class StreamlitSR3Extractor:
    """
    Streamlit-adapted wrapper for BatchSR3Extractor
    Handles file uploads and converts to Streamlit UI components
    """
    
    def __init__(self):
        """Initialize the Streamlit extractor"""
        self.extractor = None
        self.uploaded_files = []
        self.temp_dir = None
        
    def handle_file_upload(self, uploaded_files):
        """
        Handle uploaded SR3 files from Streamlit file uploader
        
        Args:
            uploaded_files: List of uploaded file objects from st.file_uploader
            
        Returns:
            bool: True if files were successfully processed
        """
        if not uploaded_files:
            return False
        
        try:
            # Create temporary directory to store uploaded files
            self.temp_dir = tempfile.mkdtemp()
            temp_path = Path(self.temp_dir)
            
            # Save uploaded files to temporary directory
            self.uploaded_files = []
            for uploaded_file in uploaded_files:
                if uploaded_file.name.endswith('.sr3'):
                    file_path = temp_path / uploaded_file.name
                    with open(file_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())
                    self.uploaded_files.append(file_path)
            
            if not self.uploaded_files:
                st.error("No valid .sr3 files found in upload")
                return False
            
            # Initialize extractor with temporary directory
            self.extractor = BatchSR3Extractor()
            self.extractor.directory_path = temp_path
            self.extractor.sr3_files = self.uploaded_files
            self.extractor.reference_file = self.uploaded_files[0]
            
            # Capture print output for Streamlit
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            try:
                # Load reference file to initialize metadata
                success = self.extractor.load_reference_file()
                
                if success:
                    # Analyze common dates across files
                    self.extractor.date_analysis = analyze_common_dates_across_files(self.uploaded_files)
                    if self.extractor.date_analysis:
                        # Update time mapping from date analysis
                        if 'file_data' in self.extractor.date_analysis:
                            ref_file_data = list(self.extractor.date_analysis['file_data'].values())[0]
                            self.extractor.time_mapping = {
                                'dates': ref_file_data['master_dates'],
                                'days': ref_file_data['master_days'],
                                'timesteps': ref_file_data['master_timesteps']
                            }
                
                # Get captured output and display in Streamlit
                output = sys.stdout.getvalue()
                if output:
                    with st.expander("📋 Processing Details", expanded=False):
                        st.text(output)
                
                return success
            finally:
                sys.stdout = old_stdout
            
        except Exception as e:
            st.error(f"Error handling file upload: {str(e)}")
            return False
    
    def get_spatial_properties(self):
        """Get available spatial properties"""
        if self.extractor and self.extractor.spatial_properties:
            return self.extractor.spatial_properties
        return []
    
    def get_well_variables(self):
        """Get available well variables"""
        if self.extractor and self.extractor.well_variables:
            return self.extractor.well_variables
        return []
    
    def get_common_dates(self, mode='yearly', data_type='spatial'):
        """
        Get common dates based on filter mode
        
        Args:
            mode: 'yearly', 'monthly', 'daily', or 'custom'
            data_type: 'spatial' or 'timeseries'
            
        Returns:
            tuple: (date_labels, target_days)
        """
        if not self.extractor or not self.extractor.date_analysis:
            return [], []
        
        time_mapping = self.extractor.time_mapping
        if not time_mapping:
            return [], []
        
        # Get appropriate common days based on data type
        if data_type == 'spatial':
            common_days = self.extractor.date_analysis.get('common_spatial_days', [])
        else:
            common_days = self.extractor.date_analysis.get('common_timeseries_days', [])
        
        if not common_days:
            return [], []
        
        if mode == 'yearly':
            days, dates = filter_yearly_dates(
                common_days,
                time_mapping['dates'],
                time_mapping['days']
            )
        elif mode == 'monthly':
            days, dates = filter_monthly_dates_flexible(
                common_days,
                time_mapping['dates'],
                time_mapping['days']
            )
        elif mode == 'daily':
            # For daily, use all common days (can be filtered later)
            days = sorted(common_days)
            dates = convert_days_to_dates(days, time_mapping['dates'], time_mapping['days'])
        else:  # custom
            days = sorted(common_days)
            dates = convert_days_to_dates(days, time_mapping['dates'], time_mapping['days'])
        
        # Format date labels for display
        date_labels = [f"{date} (Day {day:.1f})" for date, day in zip(dates, days)]
        
        return date_labels, days
    
    def extract_spatial_properties(self, selected_properties, selected_date_labels, output_folder, 
                                   output_prefix="batch_spatial_properties", 
                                   create_inactive_cells=True, inactive_value=0.0, create_well_locations=False):
        """
        Extract spatial properties with Streamlit progress tracking
        
        Args:
            selected_properties: List of property names to extract
            selected_date_labels: List of date labels (format: "YYYY-MM-DD (Day XXX.0)")
            output_folder: Path to output folder
            output_prefix: Prefix for output files
            create_inactive_cells: Whether to create inactive cell masks
            inactive_value: Value to use for inactive cells
            create_well_locations: Whether to extract well locations
            
        Returns:
            bool: True if extraction was successful
        """
        if not self.extractor:
            st.error("No extractor initialized. Please upload files first.")
            return False
        
        # Set output folder and options
        self.extractor.output_folder_widget.value = str(output_folder)
        self.extractor.create_inactive_cells_widget.value = create_inactive_cells
        self.extractor.inactive_cells_value_widget.value = inactive_value
        self.extractor.create_well_locations_widget.value = create_well_locations
        
        # Create progress placeholders
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Create custom progress callback
        def update_progress(progress, message):
            progress_bar.progress(progress / 100.0)
            status_text.text(message)
        
        # Override progress widget updates
        original_batch_progress = self.extractor.batch_progress_widget
        original_progress = self.extractor.progress_widget
        
        class StreamlitProgressWidget:
            def __init__(self, callback):
                self.callback = callback
                self._value = 0
            
            @property
            def value(self):
                return self._value
            
            @value.setter
            def value(self, val):
                self._value = val
                if self.callback:
                    self.callback(val, f"Processing: {val}%")
        
        self.extractor.batch_progress_widget = StreamlitProgressWidget(
            lambda v, m: update_progress(v, m)
        )
        self.extractor.progress_widget = StreamlitProgressWidget(
            lambda v, m: update_progress(v, m)
        )
        
        # Capture print output
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            # Perform extraction
            success = self.extractor.batch_extract_spatial_properties(
                selected_properties,
                selected_date_labels,
                output_prefix
            )
            
            # Get captured output
            output = sys.stdout.getvalue()
            
            # Restore original widgets and stdout
            self.extractor.batch_progress_widget = original_batch_progress
            self.extractor.progress_widget = original_progress
            sys.stdout = old_stdout
            
            # Display output if any
            if output:
                with st.expander("📋 Extraction Details", expanded=False):
                    st.text(output)
            
            if success:
                progress_bar.progress(1.0)
                status_text.text("✅ Extraction completed!")
                return True
            else:
                status_text.text("❌ Extraction failed")
                return False
                
        except Exception as e:
            sys.stdout = old_stdout
            status_text.text(f"❌ Error: {str(e)}")
            st.error(f"Extraction error: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return False
    
    def extract_timeseries_data(self, selected_variables, selected_date_labels, output_folder,
                                output_prefix="batch_timeseries_data"):
        """
        Extract time series data with Streamlit progress tracking
        
        Args:
            selected_variables: List of variable names to extract
            selected_date_labels: List of date labels
            output_folder: Path to output folder
            output_prefix: Prefix for output files
            
        Returns:
            bool: True if extraction was successful
        """
        if not self.extractor:
            st.error("No extractor initialized. Please upload files first.")
            return False
        
        # Set output folder
        self.extractor.output_folder_widget.value = str(output_folder)
        
        # Create progress placeholders
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Create custom progress callback
        def update_progress(progress, message):
            progress_bar.progress(progress / 100.0)
            status_text.text(message)
        
        # Override progress widgets
        original_batch_progress = self.extractor.batch_progress_widget
        original_progress = self.extractor.progress_widget
        
        class StreamlitProgressWidget:
            def __init__(self, callback):
                self.callback = callback
                self._value = 0
            
            @property
            def value(self):
                return self._value
            
            @value.setter
            def value(self, val):
                self._value = val
                if self.callback:
                    self.callback(val, f"Processing: {val}%")
        
        self.extractor.batch_progress_widget = StreamlitProgressWidget(
            lambda v, m: update_progress(v, m)
        )
        self.extractor.progress_widget = StreamlitProgressWidget(
            lambda v, m: update_progress(v, m)
        )
        
        # Capture print output
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            # Perform extraction
            success = self.extractor.batch_extract_timeseries_data(
                selected_variables,
                selected_date_labels,
                output_prefix
            )
            
            # Get captured output
            output = sys.stdout.getvalue()
            
            # Restore original widgets and stdout
            self.extractor.batch_progress_widget = original_batch_progress
            self.extractor.progress_widget = original_progress
            sys.stdout = old_stdout
            
            # Display output if any
            if output:
                with st.expander("📋 Extraction Details", expanded=False):
                    st.text(output)
            
            if success:
                progress_bar.progress(1.0)
                status_text.text("✅ Extraction completed!")
                return True
            else:
                status_text.text("❌ Extraction failed")
                return False
                
        except Exception as e:
            sys.stdout = old_stdout
            status_text.text(f"❌ Error: {str(e)}")
            st.error(f"Extraction error: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return False
    
    def display_h5_file_summary(self, output_folder):
        """
        Display summary of all created H5 files with their structure and metadata
        
        Args:
            output_folder: Path to output folder containing H5 files
        """
        output_folder = Path(output_folder)
        if not output_folder.exists():
            return
        
        # Find all H5 files
        h5_files = list(output_folder.glob('*.h5'))
        if not h5_files:
            return
        
        # Separate files by type
        spatial_files = []
        timeseries_files = []
        other_files = []
        
        for h5_file in h5_files:
            info = inspect_h5_structure(h5_file)
            if info:
                if info['file_type'] == 'spatial':
                    spatial_files.append(info)
                elif info['file_type'] == 'timeseries':
                    timeseries_files.append(info)
                elif h5_file.name == 'inactive_cell_locations.h5':
                    other_files.append(info)
                else:
                    other_files.append(info)
        
        # Display summary header
        st.markdown("---")
        st.subheader("📦 Extracted H5 Files Summary")
        
        # Display spatial files
        if spatial_files:
            st.markdown("### 📊 Spatial Property Files")
            for info in spatial_files:
                with st.expander(f"📁 {info['filename']} ({info['file_size_mb']:.2f} MB)", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**File Information**")
                        st.write(f"**Type:** Spatial Property")
                        st.write(f"**Property:** {info['summary'].get('property_name', 'Unknown')}")
                        st.write(f"**Shape:** {info['data_shape']}")
                        st.write(f"**Data Type:** {info['data_dtype']}")
                        if info['compression']:
                            st.write(f"**Compression:** {info['compression']}")
                    
                    with col2:
                        st.markdown("**Dimensions**")
                        st.write(f"**Cases:** {info['summary'].get('n_cases', 0)}")
                        st.write(f"**Timesteps:** {info['summary'].get('n_timesteps', 0)}")
                        st.write(f"**Grid:** {info['summary'].get('nx', 0)} × {info['summary'].get('ny', 0)} × {info['summary'].get('nz', 0)}")
                        st.write(f"**Active Cells:** {info['summary'].get('active_cells', 0):,}")
                        st.write(f"**Total Cells:** {info['summary'].get('total_cells', 0):,}")
                    
                    # HDF5 Structure
                    with st.expander("🗂️ HDF5 File Structure", expanded=False):
                        st.code('\n'.join(info['structure']), language='text')
                    
                    # Metadata details
                    with st.expander("📋 Metadata Attributes", expanded=False):
                        st.json(info['metadata_attrs'])
                    
                    # Code example
                    with st.expander("💻 Code Example: Reading This File", expanded=False):
                        st.code(f"""
import h5py
import numpy as np

# Load spatial property file
with h5py.File('{info['filename']}', 'r') as f:
    # Access main data array
    data = f['data'][...]  # Shape: {info['data_shape']}
    
    # Access metadata
    meta = f['metadata']
    property_name = meta.attrs['property_name']
    n_cases = meta.attrs['n_cases']
    n_timesteps = meta.attrs['n_timesteps']
    
    # Get specific case, timestep, and layer
    case_idx = 0
    timestep_idx = 0
    k_layer = 0
    spatial_slice = data[case_idx, timestep_idx, :, :, k_layer]
    
    print(f"Property: {{property_name}}")
    print(f"Data shape: {{data.shape}}")
    print(f"Spatial slice shape: {{spatial_slice.shape}}")
""", language='python')
        
        # Display timeseries files
        if timeseries_files:
            st.markdown("### 🛢️ Time Series Files")
            for info in timeseries_files:
                with st.expander(f"📁 {info['filename']} ({info['file_size_mb']:.2f} MB)", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**File Information**")
                        st.write(f"**Type:** Time Series")
                        st.write(f"**Variable:** {info['summary'].get('variable_name', 'Unknown')}")
                        st.write(f"**Shape:** {info['data_shape']}")
                        st.write(f"**Data Type:** {info['data_dtype']}")
                        if info['compression']:
                            st.write(f"**Compression:** {info['compression']}")
                    
                    with col2:
                        st.markdown("**Dimensions**")
                        st.write(f"**Cases:** {info['summary'].get('n_cases', 0)}")
                        st.write(f"**Timesteps:** {info['summary'].get('n_timesteps', 0)}")
                        st.write(f"**Wells:** {info['summary'].get('n_wells', 0)}")
                        
                        well_names = info['summary'].get('well_names', [])
                        if well_names:
                            with st.expander(f"📋 Well Names ({len(well_names)})", expanded=False):
                                for well in well_names[:20]:
                                    st.write(f"- {well}")
                                if len(well_names) > 20:
                                    st.write(f"... and {len(well_names) - 20} more")
                    
                    # HDF5 Structure
                    with st.expander("🗂️ HDF5 File Structure", expanded=False):
                        st.code('\n'.join(info['structure']), language='text')
                    
                    # Metadata details
                    with st.expander("📋 Metadata Attributes", expanded=False):
                        st.json(info['metadata_attrs'])
                    
                    # Code example
                    with st.expander("💻 Code Example: Reading This File", expanded=False):
                        st.code(f"""
import h5py
import numpy as np

# Load time series file
with h5py.File('{info['filename']}', 'r') as f:
    # Access main data array
    data = f['data'][...]  # Shape: {info['data_shape']}
    
    # Access metadata
    meta = f['metadata']
    variable_name = meta.attrs['variable_name']
    well_names = [w.decode() for w in meta['well_names'][...]]
    dates = [d.decode() for d in meta['dates'][...]]
    
    # Get specific case and well
    case_idx = 0
    well_idx = 0
    well_data = data[case_idx, :, well_idx]
    
    print(f"Variable: {{variable_name}}")
    print(f"Well: {{well_names[well_idx]}}")
    print(f"Data shape: {{data.shape}}")
    print(f"Time series length: {{len(well_data)}}")
""", language='python')
        
        # Display other files (like inactive_cell_locations.h5)
        if other_files:
            st.markdown("### 📁 Other Files")
            for info in other_files:
                with st.expander(f"📁 {info['filename']} ({info['file_size_mb']:.2f} MB)", expanded=False):
                    st.write(f"**Type:** {info['file_type'] or 'Other'}")
                    if info['data_shape']:
                        st.write(f"**Shape:** {info['data_shape']}")
                    if info['compression']:
                        st.write(f"**Compression:** {info['compression']}")
                    
                    # HDF5 Structure
                    with st.expander("🗂️ HDF5 File Structure", expanded=False):
                        st.code('\n'.join(info['structure']), language='text')
    
    def cleanup(self):
        """Clean up temporary files"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            try:
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                st.warning(f"Could not clean up temporary files: {e}")

