"""
Streamlit Web Application for SR3 Data Extraction and Visualization
Wizard-style interface for batch processing CMG SR3 files
"""

import sys
import os
from pathlib import Path

# Ensure the src/ directory is in Python path so absolute package imports work
# Handle both local execution and Streamlit Cloud execution
_src_dir = None

# Try multiple strategies to find the src directory
# Strategy 1: From __file__ location (when run as script)
try:
    _file_path = Path(__file__).resolve()
    # app.py is at: src/cmg_sr3_files_data_processor/app.py
    # parents[0] = src/cmg_sr3_files_data_processor/
    # parents[1] = src/
    _src_dir = _file_path.parents[1]
    # Verify this is actually the src directory
    if _src_dir.exists() and _src_dir.name == 'src':
        # Good, we found it
        pass
    else:
        # Try parents[2] in case we're deeper
        _src_dir = _file_path.parents[2] if len(_file_path.parts) > 2 else None
        if _src_dir and _src_dir.exists() and (_src_dir / "cmg_sr3_files_data_processor" / "app.py").exists():
            # Found it at a different level
            pass
        else:
            _src_dir = None
except Exception:
    _src_dir = None

# Strategy 2: From current working directory (Streamlit Cloud default)
if _src_dir is None or not _src_dir.exists():
    try:
        _cwd = Path.cwd()
        # Check if we're in the repo root
        if (_cwd / "src" / "cmg_sr3_files_data_processor" / "app.py").exists():
            _src_dir = _cwd / "src"
        # Check if we're already in src directory
        elif (_cwd / "cmg_sr3_files_data_processor" / "app.py").exists():
            _src_dir = _cwd
    except Exception:
        pass

# Strategy 3: Try common Streamlit Cloud paths
if _src_dir is None or not _src_dir.exists():
    try:
        # Streamlit Cloud might mount at /mount/src/repo-name
        _possible_paths = [
            Path("/mount/src/cmg-sr3-file-data-extraxtor/src"),
            Path("/mount/src/cmg-sr3-file-data-extraxtor"),
            Path.cwd() / "src",
        ]
        for _path in _possible_paths:
            if _path.exists() and (_path / "cmg_sr3_files_data_processor" / "app.py").exists():
                _src_dir = _path
                break
    except Exception:
        pass

# Add to Python path if found
if _src_dir and _src_dir.exists():
    _src_str = str(_src_dir)
    if _src_str not in sys.path:
        sys.path.insert(0, _src_str)

# Import Streamlit first so we can show errors if imports fail
import streamlit as st

# Try to import other dependencies with error handling
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from cmg_sr3_files_data_processor.streamlit_extractor import (
        StreamlitSR3Extractor,
        inspect_h5_structure,
        inspect_sr3_structure,
    )
    from cmg_sr3_files_data_processor.streamlit_visualizer import StreamlitH5Visualizer
except ImportError as e:
    st.error(f"❌ Import Error: {str(e)}")
    st.write("**Debug Information:**")
    st.code(f"Python version: {sys.version}")
    st.code(f"Python path (first 10): {sys.path[:10]}")
    st.code(f"Current working directory: {os.getcwd()}")
    st.code(f"__file__: {__file__}")
    st.code(f"Source directory attempted: {_src_dir}")
    st.code(f"Source exists: {_src_dir.exists() if _src_dir else False}")
    if _src_dir and _src_dir.exists():
        st.code(f"Contents of src: {list(_src_dir.iterdir())[:10]}")
    import traceback
    st.code(traceback.format_exc())
    st.stop()
except Exception as e:
    st.error(f"❌ Unexpected Error: {str(e)}")
    import traceback
    st.code(traceback.format_exc())
    st.stop()

# Page configuration
st.set_page_config(
    page_title="SR3 Data Processor",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'extractor' not in st.session_state:
    st.session_state.extractor = None
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = None
if 'output_folder' not in st.session_state:
    st.session_state.output_folder = None

# Custom CSS for wizard steps
st.markdown("""
<style>
    .wizard-step {
        display: inline-block;
        padding: 10px 20px;
        margin: 5px;
        border-radius: 5px;
        background-color: #f0f0f0;
        color: #666;
    }
    .wizard-step.active {
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    .wizard-step.completed {
        background-color: #2ca02c;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

def render_wizard_steps():
    """Render wizard step indicator"""
    steps = [
        ("📁 Upload Files", 1),
        ("⚙️ Configure", 2),
        ("🚀 Extract", 3),
        ("📊 Visualize", 4)
    ]
    
    cols = st.columns(4)
    for idx, (name, step_num) in enumerate(steps):
        with cols[idx]:
            if st.session_state.step == step_num:
                st.markdown(f'<div class="wizard-step active">{name}</div>', unsafe_allow_html=True)
            elif st.session_state.step > step_num:
                st.markdown(f'<div class="wizard-step completed">{name}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="wizard-step">{name}</div>', unsafe_allow_html=True)

def step1_file_upload():
    """Step 1: File Upload"""
    st.header("📁 Step 1: Upload SR3 Files")
    
    # Store uploaded files in session state for use across tabs
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = None
    
    # Create tabs
    tab1, tab2 = st.tabs(["📤 Upload Files", "📖 File Structure & Info"])
    
    with tab1:
        st.markdown("Drag and drop your SR3 files below or click to browse")
        
        uploaded_files = st.file_uploader(
            "Select SR3 files",
            type=['sr3'],
            accept_multiple_files=True,
            help="Upload one or more .sr3 files for batch processing"
        )
        
        # Update session state
        st.session_state.uploaded_files = uploaded_files
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
            
            # Show file list
            with st.expander("📋 Uploaded Files"):
                for file in uploaded_files:
                    st.write(f"- {file.name} ({file.size / 1024 / 1024:.2f} MB)")
            
            # Process files button
            if st.button("🔍 Process Files", type="primary"):
                with st.spinner("Processing uploaded files..."):
                    try:
                        extractor = StreamlitSR3Extractor()
                        success = extractor.handle_file_upload(uploaded_files)
                        
                        if success:
                            st.session_state.extractor = extractor
                            st.success(f"✅ Successfully processed {len(uploaded_files)} file(s)")
                            st.session_state.step = 2
                            st.rerun()
                        else:
                            st.error("Failed to process files. Please check that they are valid SR3 files.")
                    except Exception as e:
                        st.error(f"Error processing files: {str(e)}")
                        import traceback
                        with st.expander("Error Details"):
                            st.code(traceback.format_exc())
    
    with tab2:
        _display_sr3_file_info()
    
    # Navigation
    col1, col2 = st.columns([1, 4])
    with col2:
        if st.session_state.step > 1:
            if st.button("⬅️ Previous"):
                st.session_state.step = 1
                st.rerun()

def _display_sr3_file_info():
    """Display comprehensive SR3 file structure information"""
    st.subheader("📖 Understanding SR3 Files")
    
    # Check if files are uploaded
    uploaded_files = st.session_state.get('uploaded_files', None)
    
    if not uploaded_files:
        st.info("👆 Please upload SR3 files in the 'Upload Files' tab to view file structure information.")
        
        # Show general information about SR3 files
        st.markdown("---")
        st.markdown("### What is an SR3 File?")
        st.markdown("""
        **SR3** (Simulation Results 3) files are binary data files created by **CMG (Computer Modelling Group)** 
        reservoir simulators (GEM, IMEX, STARS). These files store comprehensive simulation results in HDF5 format.
        
        **Key Characteristics:**
        - **Format**: HDF5 (Hierarchical Data Format version 5)
        - **Purpose**: Store reservoir simulation output data
        - **Content**: Spatial properties, time series data, grid structure, and metadata
        - **Simulators**: GEM, IMEX, STARS
        """)
        
        st.markdown("### File Structure Overview")
        st.markdown("""
        SR3 files are organized hierarchically with four main top-level groups:
        
        1. **General** - Metadata and time tables
        2. **SpatialProperties** - Grid-based properties at different timesteps
        3. **TimeSeries** - Well and production data over time
        4. **Tables** - Additional tabular data
        """)
        return
    
    # Files are uploaded - analyze them
    st.success(f"✅ {len(uploaded_files)} file(s) available for analysis")
    
    # File selector if multiple files
    file_options = {f.name: f for f in uploaded_files}
    selected_file_name = st.selectbox(
        "Select a file to analyze",
        options=list(file_options.keys()),
        help="Choose which uploaded file to display detailed structure information"
    )
    
    selected_file = file_options[selected_file_name]
    
    # Save file temporarily and analyze
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.sr3') as tmp_file:
        tmp_file.write(selected_file.getbuffer())
        tmp_path = tmp_file.name
    
    try:
        # Analyze the file
        file_info = inspect_sr3_structure(tmp_path)
        
        if not file_info or 'error' in file_info:
            st.error(f"Error analyzing file: {file_info.get('error', 'Unknown error') if file_info else 'Failed to analyze'}")
            if file_info and 'traceback' in file_info:
                with st.expander("Error Details"):
                    st.code(file_info['traceback'])
            return
        
        # === FILE OVERVIEW ===
        st.markdown("---")
        st.markdown("### 📋 File Overview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("File Size", f"{file_info['file_size_mb']:.2f} MB")
        
        with col2:
            if file_info['file_metadata'].get('sr3_version'):
                st.metric("SR3 Version", file_info['file_metadata']['sr3_version'])
        
        with col3:
            if file_info['file_metadata'].get('simulator_name'):
                st.metric("Simulator", file_info['file_metadata']['simulator_name'])
        
        # File metadata details
        with st.expander("📋 File Metadata", expanded=False):
            metadata = file_info['file_metadata']
            if metadata.get('simulator_version'):
                st.write(f"**Simulator Version:** {metadata['simulator_version']}")
            if metadata.get('file_id'):
                st.write(f"**File ID:** {metadata['file_id']}")
            if metadata.get('file_format'):
                st.write(f"**File Format:** {metadata['file_format']}")
        
        # === WHAT IS SR3 FILE ===
        st.markdown("---")
        st.markdown("### 📚 What is an SR3 File?")
        st.markdown("""
        **SR3** (Simulation Results 3) files are binary data files created by **CMG (Computer Modelling Group)** 
        reservoir simulators. These files store comprehensive simulation results in HDF5 format.
        
        **Key Characteristics:**
        - **Format**: HDF5 (Hierarchical Data Format version 5) - self-describing binary format
        - **Purpose**: Store reservoir simulation output data including spatial properties and time series
        - **Content**: 
          - Grid-based spatial properties (pressure, saturation, porosity, etc.)
          - Well production data over time
          - Grid structure and geometry
          - Simulation metadata and time tables
        - **Simulators**: GEM (Compositional), IMEX (Black Oil), STARS (Thermal)
        """)
        
        # === FILE STRUCTURE ===
        st.markdown("---")
        st.markdown("### 🗂️ File Structure")
        
        st.markdown(f"**Top-level Groups:** {', '.join(file_info['top_level_groups'])}")
        
        with st.expander("📊 Complete Structure Tree", expanded=False):
            if file_info['structure_tree']:
                st.code('\n'.join(file_info['structure_tree']), language='text')
            else:
                st.info("Structure tree not available")
        
        # === GENERAL GROUP ===
        if file_info['general_group']:
            st.markdown("---")
            st.markdown("### ⚙️ General Group")
            st.markdown("""
            The **General** group contains simulation metadata and time-related tables:
            """)
            
            gen_group = file_info['general_group']
            if gen_group.get('subgroups'):
                st.write(f"**Subgroups:** {', '.join(gen_group['subgroups'])}")
            
            # MasterTimeTable
            if gen_group.get('master_time_table'):
                mt = gen_group['master_time_table']
                st.markdown("#### MasterTimeTable")
                st.markdown("""
                Maps simulation timesteps to calendar dates and simulation days. This is the central time reference
                for all temporal data in the file.
                """)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Timesteps", mt['num_timesteps'])
                with col2:
                    if mt.get('first_timestep'):
                        st.write(f"**First:** Index {mt['first_timestep'].get('index')}, Day {mt['first_timestep'].get('days', 0):.1f}")
                with col3:
                    if mt.get('last_timestep'):
                        st.write(f"**Last:** Index {mt['last_timestep'].get('index')}, Day {mt['last_timestep'].get('days', 0):.1f}")
                
                with st.expander("📋 Sample Time Table Data", expanded=False):
                    if mt.get('sample_data'):
                        df = pd.DataFrame(mt['sample_data'])
                        st.dataframe(df, use_container_width=True)
            
            # ComponentTable
            if gen_group.get('component_table'):
                comp_table = gen_group['component_table']
                st.markdown("#### ComponentTable")
                st.write(f"**Number of Components:** {comp_table['num_components']}")
                if comp_table.get('components'):
                    with st.expander(f"📋 Components ({len(comp_table['components'])})", expanded=False):
                        for comp in comp_table['components']:
                            st.write(f"- {comp}")
        
        # === SPATIAL PROPERTIES ===
        if file_info['spatial_properties']:
            st.markdown("---")
            st.markdown("### 🗺️ Spatial Properties")
            st.markdown("""
            **Spatial Properties** contain grid-based reservoir properties (pressure, saturation, porosity, permeability, etc.)
            at different simulation timesteps. Each timestep is stored as a separate group.
            """)
            
            sp = file_info['spatial_properties']
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Timesteps Available", sp['num_timesteps'])
            
            with col2:
                if sp.get('available_properties'):
                    st.metric("Properties Available", len(sp['available_properties']))
            
            # Grid structure
            if sp.get('grid_structure'):
                grid = sp['grid_structure']
                st.markdown("#### Grid Structure")
                
                if grid.get('dimensions'):
                    dims = grid['dimensions']
                    st.markdown(f"**Grid Dimensions:** {dims.get('ni', '?')} × {dims.get('nj', '?')} × {dims.get('nk', '?')} (I × J × K)")
                
                if grid.get('active_cells') is not None:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Active Cells", f"{grid['active_cells']:,}")
                    with col2:
                        st.metric("Total Cells", f"{grid['total_cells']:,}")
                    with col3:
                        if grid.get('inactive_cells') is not None:
                            st.metric("Inactive Cells", f"{grid['inactive_cells']:,}")
                
                if grid.get('sample_property'):
                    sample = grid['sample_property']
                    st.markdown(f"**Sample Property ({sample['name']}):**")
                    st.write(f"- Shape: {sample['shape']}")
                    st.write(f"- Data Type: {sample['dtype']}")
                    if sample.get('min') is not None:
                        st.write(f"- Range: {sample['min']:.6f} to {sample['max']:.6f} (Mean: {sample['mean']:.6f})")
            
            # Available properties
            if sp.get('available_properties'):
                st.markdown("#### Available Properties")
                st.markdown("""
                Properties are organized by timestep. Each timestep group contains:
                - **GRID** subgroup: Grid structure data (dimensions, active cells, geometry)
                - **Property datasets**: Actual property values (POROS, PRES, SG, SW, etc.)
                """)
                
                # Categorize properties
                static_props = [p for p in sp['available_properties'] if any(term in p.upper() for term in ['PORO', 'PERM', 'POR', 'NET', 'GROSS'])]
                dynamic_props = [p for p in sp['available_properties'] if any(term in p.upper() for term in ['PRES', 'SG', 'SW', 'SO', 'KR', 'MASDEN', 'MOLDEN'])]
                other_props = [p for p in sp['available_properties'] if p not in static_props + dynamic_props]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if static_props:
                        st.markdown("**Static Properties:**")
                        for prop in static_props[:10]:
                            st.write(f"- {prop}")
                        if len(static_props) > 10:
                            st.write(f"... and {len(static_props) - 10} more")
                
                with col2:
                    if dynamic_props:
                        st.markdown("**Dynamic Properties:**")
                        for prop in dynamic_props[:10]:
                            st.write(f"- {prop}")
                        if len(dynamic_props) > 10:
                            st.write(f"... and {len(dynamic_props) - 10} more")
                
                with col3:
                    if other_props:
                        st.markdown("**Other Properties:**")
                        for prop in other_props[:10]:
                            st.write(f"- {prop}")
                        if len(other_props) > 10:
                            st.write(f"... and {len(other_props) - 10} more")
                
                with st.expander(f"📋 All Properties ({len(sp['available_properties'])})", expanded=False):
                    for prop in sp['available_properties']:
                        st.write(f"- {prop}")
            
            # Timesteps
            if sp.get('timesteps'):
                with st.expander(f"📅 Timesteps ({len(sp['timesteps'])} shown)", expanded=False):
                    st.write("Timesteps are stored as 6-digit zero-padded integers (e.g., 000000, 000035, 000097)")
                    st.write(f"**First 20 timesteps:** {', '.join(sp['timesteps'])}")
        
        # === TIME SERIES ===
        if file_info['time_series'] and file_info['time_series'].get('wells'):
            st.markdown("---")
            st.markdown("### 🛢️ Time Series Data")
            st.markdown("""
            **Time Series** data contains well production and operational data over time. The WELLS subgroup stores
            data as a 3D array: (timesteps × variables × wells).
            """)
            
            wells = file_info['time_series']['wells']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Wells", wells['num_wells'])
            with col2:
                st.metric("Variables", wells['num_variables'])
            with col3:
                st.metric("Timesteps", wells['num_timesteps'])
            
            # Well names
            if wells.get('well_names'):
                st.markdown("#### Well Names")
                with st.expander(f"📋 Wells ({len(wells['well_names'])})", expanded=False):
                    for well in wells['well_names']:
                        st.write(f"- {well}")
            
            # Variables
            if wells.get('variables'):
                st.markdown("#### Variables")
                st.markdown("""
                Variables represent different well measurements and properties (pressure, flow rates, temperatures, etc.)
                """)
                with st.expander(f"📋 Variables ({len(wells['variables'])})", expanded=False):
                    # Group variables by category
                    pressure_vars = [v for v in wells['variables'] if 'P' in v or 'PRES' in v]
                    flow_vars = [v for v in wells['variables'] if 'VOL' in v or 'RATE' in v or 'FLOW' in v]
                    temp_vars = [v for v in wells['variables'] if 'TEMP' in v or 'T' in v]
                    other_vars = [v for v in wells['variables'] if v not in pressure_vars + flow_vars + temp_vars]
                    
                    if pressure_vars:
                        st.markdown("**Pressure Variables:**")
                        for var in pressure_vars[:10]:
                            st.write(f"- {var}")
                        if len(pressure_vars) > 10:
                            st.write(f"... and {len(pressure_vars) - 10} more")
                    
                    if flow_vars:
                        st.markdown("**Flow Rate Variables:**")
                        for var in flow_vars[:10]:
                            st.write(f"- {var}")
                        if len(flow_vars) > 10:
                            st.write(f"... and {len(flow_vars) - 10} more")
                    
                    if temp_vars:
                        st.markdown("**Temperature Variables:**")
                        for var in temp_vars[:10]:
                            st.write(f"- {var}")
                        if len(temp_vars) > 10:
                            st.write(f"... and {len(temp_vars) - 10} more")
                    
                    if other_vars:
                        st.markdown("**Other Variables:**")
                        for var in other_vars[:20]:
                            st.write(f"- {var}")
                        if len(other_vars) > 20:
                            st.write(f"... and {len(other_vars) - 20} more")
            
            # Data array info
            if wells.get('data_shape'):
                st.markdown("#### Data Array")
                st.write(f"**Shape:** {wells['data_shape']} (timesteps × variables × wells)")
                st.write(f"**Data Type:** {wells.get('data_dtype', 'Unknown')}")
                if wells.get('data_size_mb'):
                    st.write(f"**Size:** {wells['data_size_mb']:.2f} MB")
            
            # Timesteps
            if wells.get('timesteps'):
                ts_info = wells['timesteps']
                st.markdown("#### Time Series Timesteps")
                if ts_info.get('first') is not None:
                    st.write(f"**First Timestep:** {ts_info['first']}")
                if ts_info.get('last') is not None:
                    st.write(f"**Last Timestep:** {ts_info['last']}")
                if ts_info.get('sample'):
                    st.write(f"**Sample Timesteps:** {', '.join(map(str, ts_info['sample']))}")
        
        # === DATA ACCESS PATTERNS ===
        st.markdown("---")
        st.markdown("### 💻 Data Access Patterns")
        st.markdown("""
        Here are code examples for accessing different types of data from SR3 files:
        """)
        
        # Code example for spatial properties
        with st.expander("📊 Accessing Spatial Properties", expanded=False):
            if file_info['spatial_properties'] and file_info['spatial_properties'].get('available_properties'):
                sample_prop = file_info['spatial_properties']['available_properties'][0]
                sample_ts = file_info['spatial_properties'].get('timesteps', ['000000'])[0] if file_info['spatial_properties'].get('timesteps') else '000000'
                
                st.code(f"""
import h5py
import numpy as np

# Open SR3 file
with h5py.File('{selected_file_name}', 'r') as f:
    # Access spatial property at specific timestep
    timestep = '{sample_ts}'
    property_name = '{sample_prop}'
    
    # Get property data
    property_data = f['SpatialProperties/{{timestep}}/{{property_name}}'][...]
    print(f"Property shape: {{property_data.shape}}")
    print(f"Property range: {{np.nanmin(property_data):.6f}} to {{np.nanmax(property_data):.6f}}")
    
    # Access grid structure
    grid_group = f['SpatialProperties/{{timestep}}/GRID']
    igntid = grid_group['IGNTID'][...]  # I indices
    igntjd = grid_group['IGNTJD'][...]  # J indices
    igntkd = grid_group['IGNTKD'][...]  # K indices
    ipstcs = grid_group['IPSTCS'][...]  # Active cell mapping
    
    # Get grid dimensions
    ni, nj, nk = igntid.max(), igntjd.max(), igntkd.max()
    print(f"Grid dimensions: {{ni}} × {{nj}} × {{nk}}")
""", language='python')
        
        # Code example for time series
        with st.expander("🛢️ Accessing Time Series Data", expanded=False):
            st.code(f"""
import h5py
import numpy as np

# Open SR3 file
with h5py.File('{selected_file_name}', 'r') as f:
    # Access time series wells data
    wells_group = f['TimeSeries/WELLS']
    
    # Get data arrays
    data = wells_group['Data'][...]  # Shape: (timesteps, variables, wells)
    timesteps = wells_group['Timesteps'][...]  # Timestep indices
    variables = [v.decode() if isinstance(v, bytes) else v 
                 for v in wells_group['Variables'][...]]
    well_names = [w.decode() if isinstance(w, bytes) else w 
                  for w in wells_group['Origins'][...]]
    
    print(f"Data shape: {{data.shape}}")
    print(f"Number of timesteps: {{len(timesteps)}}")
    print(f"Number of variables: {{len(variables)}}")
    print(f"Number of wells: {{len(well_names)}}")
    
    # Get specific well and variable
    well_idx = 0
    var_idx = variables.index('BHP') if 'BHP' in variables else 0
    
    well_data = data[:, var_idx, well_idx]
    print(f"\\nWell: {{well_names[well_idx]}}")
    print(f"Variable: {{variables[var_idx]}}")
    print(f"Time series length: {{len(well_data)}}")
""", language='python')
        
        # Code example for MasterTimeTable
        with st.expander("⏰ Accessing MasterTimeTable", expanded=False):
            st.code(f"""
import h5py
import numpy as np

# Open SR3 file
with h5py.File('{selected_file_name}', 'r') as f:
    # Access MasterTimeTable
    time_table = f['General/MasterTimeTable'][...]
    
    # Extract arrays
    timestep_indices = time_table['Index']
    simulation_days = time_table['Offset in days']
    dates = time_table['Date']
    
    print(f"Total timesteps: {{len(timestep_indices)}}")
    print(f"Simulation period: {{simulation_days[0]:.1f}} to {{simulation_days[-1]:.1f}} days")
    print(f"Date range: {{dates[0]:.0f}} to {{dates[-1]:.0f}}")
    
    # Find timestep for specific simulation day
    target_day = 100.0
    timestep_idx = np.where(np.abs(simulation_days - target_day) < 0.1)[0]
    if len(timestep_idx) > 0:
        print(f"\\nTimestep at day {{target_day}}: {{timestep_indices[timestep_idx[0]]}}")
""", language='python')
        
        # Code example for listing all properties
        with st.expander("📋 Listing All Available Data", expanded=False):
            st.code(f"""
import h5py

def explore_sr3_structure(file_path):
    '''Explore SR3 file structure'''
    with h5py.File(file_path, 'r') as f:
        print("=== TOP-LEVEL GROUPS ===")
        print(list(f.keys()))
        
        print("\\n=== SPATIAL PROPERTIES TIMESTEPS ===")
        if 'SpatialProperties' in f:
            timesteps = sorted([k for k in f['SpatialProperties'].keys() if k.isdigit()])
            print(f"Found {{len(timesteps)}} timesteps")
            print(f"First 5: {{timesteps[:5]}}")
            
            # Get properties from first timestep
            if timesteps:
                first_ts = timesteps[0]
                properties = [k for k in f[f'SpatialProperties/{{first_ts}}'].keys() if k != 'GRID']
                print(f"\\nAvailable properties: {{properties[:10]}}")
        
        print("\\n=== TIME SERIES WELLS ===")
        if 'TimeSeries/WELLS' in f:
            wells_group = f['TimeSeries/WELLS']
            variables = [v.decode() if isinstance(v, bytes) else v 
                        for v in wells_group['Variables'][...]]
            well_names = [w.decode() if isinstance(w, bytes) else w 
                          for w in wells_group['Origins'][...]]
            print(f"Wells: {{well_names}}")
            print(f"Variables: {{len(variables)}} variables")
            print(f"Sample variables: {{variables[:10]}}")

# Use the function
explore_sr3_structure('{selected_file_name}')
""", language='python')
    
    finally:
        # Clean up temporary file
        try:
            os.unlink(tmp_path)
        except:
            pass

def step2_configure():
    """Step 2: Extraction Configuration"""
    st.header("⚙️ Step 2: Configure Extraction")
    
    if not st.session_state.extractor:
        st.warning("Please upload files in Step 1 first.")
        if st.button("⬅️ Go to Step 1"):
            st.session_state.step = 1
            st.rerun()
        return
    
    extractor = st.session_state.extractor
    
    # Data type selection
    st.subheader("Select Data Types to Extract")
    extract_spatial = st.checkbox("📊 Spatial Properties", value=True)
    extract_timeseries = st.checkbox("🛢️ Time Series Data", value=True)
    
    if not extract_spatial and not extract_timeseries:
        st.warning("Please select at least one data type.")
        return
    
    col1, col2 = st.columns(2)
    
    # Spatial Properties Configuration
    selected_properties = []
    spatial_date_labels = []
    selected_layers = []
    if extract_spatial:
        with col1:
            st.subheader("📊 Spatial Properties")
            properties = extractor.get_spatial_properties()
            if properties:
                selected_properties = st.multiselect(
                    "Select Properties",
                    options=properties,
                    default=properties[:min(3, len(properties))] if properties else []
                )
            else:
                st.warning("No spatial properties found in files.")
            
            # Layer selection
            if extractor.extractor and extractor.extractor.grid_dims:
                nz = extractor.extractor.grid_dims[2]
                st.subheader("📊 Layer Selection (Nz)")
                st.caption("💡 Leave all unchecked to extract all layers, or select specific layers to reduce file size")
                
                # Create checkboxes for each layer
                layer_cols = st.columns(min(10, nz))  # Max 10 columns
                selected_layers = []
                
                for i in range(nz):
                    col_idx = i % len(layer_cols)
                    with layer_cols[col_idx]:
                        if st.checkbox(f"Layer {i}", key=f"layer_{i}", value=False):
                            selected_layers.append(i)
                
                if selected_layers:
                    st.info(f"Selected {len(selected_layers)} layers: {selected_layers}")
                else:
                    st.info(f"All {nz} layers will be extracted (default)")
            else:
                st.info("Layer selection will be available after files are loaded.")
                selected_layers = []
    
    # Time Series Configuration
    selected_variables = []
    timeseries_date_labels = []
    if extract_timeseries:
        with col2:
            st.subheader("🛢️ Time Series Data")
            variables = extractor.get_well_variables()
            if variables:
                selected_variables = st.multiselect(
                    "Select Variables",
                    options=variables,
                    default=variables[:min(3, len(variables))] if variables else []
                )
            else:
                st.warning("No time series variables found in files.")
    
    # Date Filtering
    st.subheader("🗓️ Date Filtering")
    date_mode = st.radio(
        "Select Date Filter Mode",
        options=['yearly', 'monthly', 'daily', 'custom'],
        format_func=lambda x: {
            'yearly': '📅 Yearly (January 1st)',
            'monthly': '📆 Monthly (End of Month)',
            'daily': '📊 Daily (Regular Intervals)',
            'custom': '🎯 Custom Selection'
        }[x],
        horizontal=True
    )
    
    # Get dates based on mode - handle spatial and timeseries separately
    if extract_spatial:
        spatial_date_labels_list, spatial_target_days = extractor.get_common_dates(mode=date_mode, data_type='spatial')
        
        if spatial_date_labels_list:
            if date_mode == 'custom':
                spatial_date_labels = st.multiselect(
                    "Select Dates (Spatial)",
                    options=spatial_date_labels_list,
                    default=spatial_date_labels_list[:min(10, len(spatial_date_labels_list))],
                    key="spatial_dates"
                )
            else:
                st.info(f"Found {len(spatial_date_labels_list)} spatial dates")
                with st.expander("📋 Preview Spatial Dates", expanded=False):
                    for label in spatial_date_labels_list[:20]:
                        st.write(f"- {label}")
                    if len(spatial_date_labels_list) > 20:
                        st.write(f"... and {len(spatial_date_labels_list) - 20} more")
                
                spatial_date_labels = st.multiselect(
                    "Select Dates (Spatial)",
                    options=spatial_date_labels_list,
                    default=spatial_date_labels_list,
                    key="spatial_dates"
                )
        else:
            st.warning("No common spatial dates found across files.")
            spatial_date_labels = []
    
    if extract_timeseries:
        timeseries_date_labels_list, timeseries_target_days = extractor.get_common_dates(mode=date_mode, data_type='timeseries')
        
        if timeseries_date_labels_list:
            if date_mode == 'custom':
                timeseries_date_labels = st.multiselect(
                    "Select Dates (Time Series)",
                    options=timeseries_date_labels_list,
                    default=timeseries_date_labels_list[:min(10, len(timeseries_date_labels_list))],
                    key="timeseries_dates"
                )
            else:
                st.info(f"Found {len(timeseries_date_labels_list)} time series dates")
                with st.expander("📋 Preview Time Series Dates", expanded=False):
                    for label in timeseries_date_labels_list[:20]:
                        st.write(f"- {label}")
                    if len(timeseries_date_labels_list) > 20:
                        st.write(f"... and {len(timeseries_date_labels_list) - 20} more")
                
                timeseries_date_labels = st.multiselect(
                    "Select Dates (Time Series)",
                    options=timeseries_date_labels_list,
                    default=timeseries_date_labels_list,
                    key="timeseries_dates"
                )
        else:
            st.warning("No common time series dates found across files.")
            timeseries_date_labels = []
    
    # Validate dates
    date_validation_passed = True
    if extract_spatial and not spatial_date_labels:
        st.error("Please select at least one spatial date.")
        date_validation_passed = False
    if extract_timeseries and not timeseries_date_labels:
        st.error("Please select at least one time series date.")
        date_validation_passed = False
    
    # Output Configuration
    st.subheader("💾 Output Configuration")
    
    # Show default path prominently
    default_output_path = Path.home() / "sr3_output"
    st.info(f"📁 **Default Output Path:** `{default_output_path.resolve()}`")
    st.caption("💡 **Note:** Files will be saved to this location. After extraction, you can download them directly from the app.")
    
    output_folder_input = st.text_input(
        "Output Folder Path",
        value=str(default_output_path),
        help="Path where extracted H5 files will be saved. You can change this to any valid directory path."
    )
    
    output_prefix_spatial = st.text_input(
        "Spatial Output Prefix",
        value="batch_spatial_properties",
        disabled=not extract_spatial
    )
    
    output_prefix_timeseries = st.text_input(
        "Time Series Output Prefix",
        value="batch_timeseries_data",
        disabled=not extract_timeseries
    )
    
    # Inactive cell options
    create_inactive_cells = st.checkbox(
        "Create Inactive Cell Masks",
        value=True,
        help="Generate inactive_cell_locations.h5 file"
    )
    
    inactive_value = st.number_input(
        "Inactive Cell Value",
        value=0.0,
        help="Value to use for inactive cells in spatial data"
    )
    
    # Well location options
    create_well_locations = st.checkbox(
        "Extract Well Locations",
        value=False,
        help="Generate well_locations.h5 file with well completion locations"
    )
    
    # Store configuration in session state
    st.session_state.extraction_config = {
        'extract_spatial': extract_spatial,
        'extract_timeseries': extract_timeseries,
        'selected_properties': selected_properties,
        'selected_variables': selected_variables,
        'spatial_date_labels': spatial_date_labels,
        'timeseries_date_labels': timeseries_date_labels,
        'selected_layers': selected_layers,
        'output_folder': output_folder_input,
        'output_prefix_spatial': output_prefix_spatial,
        'output_prefix_timeseries': output_prefix_timeseries,
        'create_inactive_cells': create_inactive_cells,
        'inactive_value': inactive_value,
        'create_well_locations': create_well_locations
    }
    
    # Navigation
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("⬅️ Previous"):
            st.session_state.step = 1
            st.rerun()
    with col2:
        if st.button("Next ➡️", type="primary"):
            # Validate configuration
            validation_errors = []
            if extract_spatial and not selected_properties:
                validation_errors.append("Please select at least one spatial property.")
            if extract_timeseries and not selected_variables:
                validation_errors.append("Please select at least one time series variable.")
            if extract_spatial and not spatial_date_labels:
                validation_errors.append("Please select at least one spatial date.")
            if extract_timeseries and not timeseries_date_labels:
                validation_errors.append("Please select at least one time series date.")
            
            if validation_errors:
                for error in validation_errors:
                    st.error(error)
            else:
                st.session_state.step = 3
                st.rerun()

def step3_extract():
    """Step 3: Extraction Execution"""
    st.header("🚀 Step 3: Extract Data")
    
    if not st.session_state.extractor or 'extraction_config' not in st.session_state:
        st.warning("Please complete configuration in Step 2 first.")
        if st.button("⬅️ Go to Step 2"):
            st.session_state.step = 2
            st.rerun()
        return
    
    extractor = st.session_state.extractor
    config = st.session_state.extraction_config
    
    # Create output folder
    output_folder = Path(config['output_folder'])
    output_folder.mkdir(parents=True, exist_ok=True)
    st.session_state.output_folder = output_folder
    
    # Display output folder prominently
    st.success(f"📁 **Files will be saved to:** `{output_folder.resolve()}`")
    st.caption("💡 **Tip:** After extraction completes, you can download all files using the download buttons below.")
    
    # Extract button
    if st.button("🚀 Start Extraction", type="primary"):
        success_spatial = True
        success_timeseries = True
        
        # Extract spatial properties
        if config['extract_spatial']:
            st.subheader("📊 Extracting Spatial Properties...")
            success_spatial = extractor.extract_spatial_properties(
                config['selected_properties'],
                config['spatial_date_labels'],
                output_folder,
                config['output_prefix_spatial'],
                config['create_inactive_cells'],
                config['inactive_value'],
                config.get('create_well_locations', False),
                config.get('selected_layers', [])
            )
        
        # Extract time series
        if config['extract_timeseries']:
            st.subheader("🛢️ Extracting Time Series Data...")
            success_timeseries = extractor.extract_timeseries_data(
                config['selected_variables'],
                config['timeseries_date_labels'],
                output_folder,
                config['output_prefix_timeseries']
            )
        
        if success_spatial and success_timeseries:
            st.success("✅ Extraction completed successfully!")
            
            # Display download section
            st.markdown("---")
            st.subheader("📥 Download Extracted Files")
            
            # Create zip download button
            h5_files = list(output_folder.glob('*.h5'))
            if h5_files:
                total_size_mb = sum(f.stat().st_size for f in h5_files) / (1024 * 1024)
                total_size_gb = total_size_mb / 1024
                
                # Check for very large files
                if total_size_gb > 1:
                    st.error(f"❌ **Files too large for download:** {total_size_gb:.2f} GB total")
                    st.warning("""
                    **Streamlit Cloud Limitations:**
                    - Maximum memory: 2.7GB
                    - Download limit: ~500MB-1GB per file
                    - Files larger than 1GB cannot be downloaded through the web interface
                    
                    **Recommended Solutions:**
                    1. **Run locally:** Use `run_app.bat` on your PC for large files
                    2. **Cloud storage:** Upload to AWS S3, Google Cloud Storage, or Azure Blob
                    3. **Split processing:** Process files in smaller batches
                    4. **Direct access:** Use SSH/SFTP to access files on Streamlit Cloud server
                    """)
                else:
                    # Create zip file in memory
                    import zipfile
                    import io
                    
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                        for h5_file in h5_files:
                            zip_file.write(h5_file, h5_file.name)
                    
                    zip_buffer.seek(0)
                    zip_size_mb = len(zip_buffer.getvalue()) / (1024 * 1024)
                    
                    st.download_button(
                        label=f"📦 Download All Files ({len(h5_files)} files, {zip_size_mb:.2f} MB)",
                        data=zip_buffer.getvalue(),
                        file_name=f"sr3_extracted_files_{output_folder.name}.zip",
                        mime="application/zip",
                        type="primary",
                        key=f"download_all_step3_{output_folder.name}",
                        help=f"Download all {len(h5_files)} extracted H5 files as a ZIP archive"
                    )
                    
                    if total_size_mb > 200:
                        st.warning(f"⚠️ **Large files detected:** Total size is {total_size_mb:.2f} MB. Download may take some time.")
                    if total_size_mb > 500:
                        st.warning("⚠️ **Very large files:** Downloads over 500MB may fail on Streamlit Cloud. Consider using local processing.")
            else:
                st.warning("No H5 files found in output folder.")
            
            # Display H5 file summary (which will also have individual download buttons)
            extractor.display_h5_file_summary(output_folder)
            
            # Don't auto-advance - let user manually proceed to visualization
            # st.session_state.step = 4
            # st.rerun()
        else:
            st.error("❌ Extraction completed with errors. Please check the output above.")
    
    # Navigation
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("⬅️ Previous"):
            st.session_state.step = 2
            st.rerun()
    with col2:
        if st.session_state.output_folder and st.button("Next ➡️ Visualize", type="primary"):
            st.session_state.step = 4
            st.rerun()

def step4_visualize():
    """Step 4: Visualization"""
    st.header("📊 Step 4: Visualize Data")
    
    if not st.session_state.output_folder:
        st.warning("Please complete extraction in Step 3 first.")
        if st.button("⬅️ Go to Step 3"):
            st.session_state.step = 3
            st.rerun()
        return
    
    # Initialize visualizer
    if not st.session_state.visualizer:
        visualizer = StreamlitH5Visualizer()
        if visualizer.load_h5_directory(st.session_state.output_folder):
            st.session_state.visualizer = visualizer
        else:
            st.error("No H5 files found in output folder.")
            return
    
    visualizer = st.session_state.visualizer
    
    # Visualization tabs
    tab1, tab2 = st.tabs(["🗺️ Spatial Properties", "🛢️ Time Series"])
    
    with tab1:
        st.subheader("Spatial Properties Visualization")
        
        spatial_files = visualizer.get_spatial_files()
        if not spatial_files:
            st.warning("No spatial property files found.")
        else:
            selected_file = st.selectbox("Select H5 File", options=spatial_files)
            
            if selected_file:
                # Display file information panel
                file_path = visualizer.directory_path / selected_file
                file_info = inspect_h5_structure(file_path)
                
                if file_info:
                    with st.expander("📋 File Information & Structure", expanded=False):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**File Details**")
                            st.write(f"**Filename:** {file_info['filename']}")
                            st.write(f"**Size:** {file_info['file_size_mb']:.2f} MB")
                            st.write(f"**Type:** Spatial Property")
                            st.write(f"**Property:** {file_info['summary'].get('property_name', 'Unknown')}")
                            st.write(f"**Data Shape:** {file_info['data_shape']}")
                            st.write(f"**Data Type:** {file_info['data_dtype']}")
                            if file_info['compression']:
                                st.write(f"**Compression:** {file_info['compression']}")
                        
                        with col2:
                            st.markdown("**Dimensions**")
                            st.write(f"**Cases:** {file_info['summary'].get('n_cases', 0)}")
                            st.write(f"**Timesteps:** {file_info['summary'].get('n_timesteps', 0)}")
                            st.write(f"**Grid:** {file_info['summary'].get('nx', 0)} × {file_info['summary'].get('ny', 0)} × {file_info['summary'].get('nz', 0)}")
                            st.write(f"**Active Cells:** {file_info['summary'].get('active_cells', 0):,}")
                            st.write(f"**Total Cells:** {file_info['summary'].get('total_cells', 0):,}")
                        
                        # Data statistics
                        if visualizer.current_spatial_data is not None:
                            with st.expander("📊 Data Statistics", expanded=False):
                                data_stats = {
                                    'Min': float(np.nanmin(visualizer.current_spatial_data)),
                                    'Max': float(np.nanmax(visualizer.current_spatial_data)),
                                    'Mean': float(np.nanmean(visualizer.current_spatial_data)),
                                    'Std': float(np.nanstd(visualizer.current_spatial_data))
                                }
                                col_stat1, col_stat2 = st.columns(2)
                                with col_stat1:
                                    st.write(f"**Min:** {data_stats['Min']:.6f}")
                                    st.write(f"**Max:** {data_stats['Max']:.6f}")
                                with col_stat2:
                                    st.write(f"**Mean:** {data_stats['Mean']:.6f}")
                                    st.write(f"**Std:** {data_stats['Std']:.6f}")
                        
                        # HDF5 Structure
                        with st.expander("🗂️ HDF5 File Structure", expanded=False):
                            st.code('\n'.join(file_info['structure']), language='text')
                        
                        # Metadata attributes
                        with st.expander("📋 Metadata Attributes", expanded=False):
                            st.json(file_info['metadata_attrs'])
                        
                        # Code example
                        with st.expander("💻 Code Example: Reading This File", expanded=False):
                            st.code(f"""
import h5py
import numpy as np

# Load spatial property file
with h5py.File('{file_info['filename']}', 'r') as f:
    # Access main data array
    data = f['data'][...]  # Shape: {file_info['data_shape']}
    
    # Access metadata
    meta = f['metadata']
    property_name = meta.attrs['property_name']
    n_cases = meta.attrs['n_cases']
    n_timesteps = meta.attrs['n_timesteps']
    nx, ny, nz = meta.attrs['nx'], meta.attrs['ny'], meta.attrs['nz']
    
    # Get specific case, timestep, and layer
    case_idx = 0
    timestep_idx = 0
    k_layer = 0
    spatial_slice = data[case_idx, timestep_idx, :, :, k_layer]
    
    # Get timesteps
    timesteps = meta['timesteps'][...]
    
    print(f"Property: {{property_name}}")
    print(f"Data shape: {{data.shape}}")
    print(f"Spatial slice shape: {{spatial_slice.shape}}")
    print(f"Grid dimensions: {{nx}} × {{ny}} × {{nz}}")
""", language='python')
                
                if visualizer.load_spatial_file(selected_file):
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        case = st.slider(
                            "Case",
                            min_value=0,
                            max_value=visualizer.spatial_metadata['n_cases'] - 1,
                            value=0
                        )
                        
                        k_layer = st.slider(
                            "K Layer",
                            min_value=0,
                            max_value=visualizer.spatial_metadata['nz'] - 1,
                            value=min(19, visualizer.spatial_metadata['nz'] - 1)
                        )
                        
                        timestep = st.slider(
                            "Timestep",
                            min_value=0,
                            max_value=visualizer.spatial_metadata['n_timesteps'] - 1,
                            value=0
                        )
                        
                        apply_mask = st.checkbox(
                            "Apply Inactive Cell Masking",
                            value=True,
                            disabled=not visualizer.has_inactive_mask
                        )
                        
                        show_wells = st.checkbox(
                            "Show Well Locations",
                            value=False,
                            disabled=not visualizer.has_well_locations
                        )
                    
                    with col2:
                        fig = visualizer.plot_spatial(case, k_layer, timestep, apply_mask, show_wells)
                        if fig:
                            st.pyplot(fig)
                            plt.close(fig)
    
    with tab2:
        st.subheader("Time Series Visualization")
        
        timeseries_files = visualizer.get_timeseries_files()
        if not timeseries_files:
            st.warning("No time series files found.")
        else:
            selected_file = st.selectbox("Select H5 File", options=timeseries_files, key="ts_file")
            
            if selected_file:
                # Display file information panel
                file_path = visualizer.directory_path / selected_file
                file_info = inspect_h5_structure(file_path)
                
                if file_info:
                    with st.expander("📋 File Information & Structure", expanded=False):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**File Details**")
                            st.write(f"**Filename:** {file_info['filename']}")
                            st.write(f"**Size:** {file_info['file_size_mb']:.2f} MB")
                            st.write(f"**Type:** Time Series")
                            st.write(f"**Variable:** {file_info['summary'].get('variable_name', 'Unknown')}")
                            st.write(f"**Data Shape:** {file_info['data_shape']}")
                            st.write(f"**Data Type:** {file_info['data_dtype']}")
                            if file_info['compression']:
                                st.write(f"**Compression:** {file_info['compression']}")
                        
                        with col2:
                            st.markdown("**Dimensions**")
                            st.write(f"**Cases:** {file_info['summary'].get('n_cases', 0)}")
                            st.write(f"**Timesteps:** {file_info['summary'].get('n_timesteps', 0)}")
                            st.write(f"**Wells:** {file_info['summary'].get('n_wells', 0)}")
                            
                            well_names = file_info['summary'].get('well_names', [])
                            if well_names:
                                with st.expander(f"📋 Well Names ({len(well_names)})", expanded=False):
                                    for well in well_names[:20]:
                                        st.write(f"- {well}")
                                    if len(well_names) > 20:
                                        st.write(f"... and {len(well_names) - 20} more")
                        
                        # Data statistics
                        if visualizer.current_timeseries_data is not None:
                            with st.expander("📊 Data Statistics", expanded=False):
                                data_stats = {
                                    'Min': float(np.nanmin(visualizer.current_timeseries_data)),
                                    'Max': float(np.nanmax(visualizer.current_timeseries_data)),
                                    'Mean': float(np.nanmean(visualizer.current_timeseries_data)),
                                    'Std': float(np.nanstd(visualizer.current_timeseries_data))
                                }
                                col_stat1, col_stat2 = st.columns(2)
                                with col_stat1:
                                    st.write(f"**Min:** {data_stats['Min']:.6f}")
                                    st.write(f"**Max:** {data_stats['Max']:.6f}")
                                with col_stat2:
                                    st.write(f"**Mean:** {data_stats['Mean']:.6f}")
                                    st.write(f"**Std:** {data_stats['Std']:.6f}")
                        
                        # HDF5 Structure
                        with st.expander("🗂️ HDF5 File Structure", expanded=False):
                            st.code('\n'.join(file_info['structure']), language='text')
                        
                        # Metadata attributes
                        with st.expander("📋 Metadata Attributes", expanded=False):
                            st.json(file_info['metadata_attrs'])
                        
                        # Code example
                        with st.expander("💻 Code Example: Reading This File", expanded=False):
                            st.code(f"""
import h5py
import numpy as np

# Load time series file
with h5py.File('{file_info['filename']}', 'r') as f:
    # Access main data array
    data = f['data'][...]  # Shape: {file_info['data_shape']}
    
    # Access metadata
    meta = f['metadata']
    variable_name = meta.attrs['variable_name']
    n_cases = meta.attrs['n_cases']
    n_timesteps = meta.attrs['n_timesteps']
    n_wells = meta.attrs['n_wells']
    
    # Get well names and dates
    well_names = [w.decode() for w in meta['well_names'][...]]
    dates = [d.decode() for d in meta['dates'][...]]
    timesteps = meta['timesteps'][...]
    
    # Get specific case and well
    case_idx = 0
    well_idx = 0
    well_data = data[case_idx, :, well_idx]
    
    print(f"Variable: {{variable_name}}")
    print(f"Well: {{well_names[well_idx]}}")
    print(f"Data shape: {{data.shape}}")
    print(f"Time series length: {{len(well_data)}}")
    print(f"Date range: {{dates[0]}} to {{dates[-1]}}")
""", language='python')
                
                if visualizer.load_timeseries_file(selected_file):
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        case = st.slider(
                            "Case",
                            min_value=0,
                            max_value=visualizer.timeseries_metadata['n_cases'] - 1,
                            value=0,
                            key="ts_case"
                        )
                        
                        selected_wells = st.multiselect(
                            "Select Wells",
                            options=visualizer.timeseries_metadata['well_names'],
                            default=visualizer.timeseries_metadata['well_names'][:min(3, len(visualizer.timeseries_metadata['well_names']))]
                        )
                    
                    with col2:
                        if selected_wells:
                            fig = visualizer.plot_timeseries(case, selected_wells)
                            if fig:
                                st.pyplot(fig)
                                plt.close(fig)
    
    # Navigation
    if st.button("⬅️ Previous"):
        st.session_state.step = 3
        st.rerun()
    
    if st.button("🔄 Start Over"):
        st.session_state.step = 1
        st.session_state.extractor = None
        st.session_state.visualizer = None
        st.session_state.output_folder = None
        st.rerun()

# Main app
def main():
    """Main application"""
    st.title("🛢️ SR3 Data Processor")
    st.markdown("Batch extract and visualize CMG SR3 reservoir simulation data")
    
    render_wizard_steps()
    st.divider()
    
    # Route to appropriate step
    if st.session_state.step == 1:
        step1_file_upload()
    elif st.session_state.step == 2:
        step2_configure()
    elif st.session_state.step == 3:
        step3_extract()
    elif st.session_state.step == 4:
        step4_visualize()

if __name__ == "__main__":
    main()

