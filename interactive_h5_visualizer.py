#%%
#!/usr/bin/env python3
"""
Interactive H5 Data Visualizer - Jupyter Compatible
================================================

An interactive dashboard for visualizing batch-processed H5 files from SR3 data extraction.
Features real-time interactive widgets for exploring:
1. Time Series Data: Select case, wells, and view time series plots
2. Spatial Properties: Select case, K-layer, timestep for spatial I×J plots

**NEW: Automatic Inactive Cell Masking**
- Automatically loads inactive_cell_locations.h5 if present
- Masks out inactive cells in spatial plots for clean reservoir visualization
- Shows only active reservoir geometry

Uses ipywidgets for interactive interface within Jupyter notebooks.
"""

import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib for high-resolution plots
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

class InteractiveH5Visualizer:
    """
    Interactive visualizer for batch-processed H5 files
    """
    
    def __init__(self):
        """Initialize the visualizer"""
        self.directory_path = None
        self.spatial_files = []
        self.timeseries_files = []
        self.current_spatial_data = None
        self.current_timeseries_data = None
        self.spatial_metadata = None
        self.timeseries_metadata = None
        
        # Inactive cell masking
        self.inactive_mask = None
        self.active_mask = None
        self.has_inactive_mask = False
        
        # Well location data
        self.well_locations = None
        self.well_mapping = None
        self.well_names = []
        self.well_types = []
        self.has_well_locations = False
        
        # Global color range for spatial plots (fixed across all timesteps)
        self.global_vmin = None
        self.global_vmax = None
        
        # Flag to prevent recursive updates
        self._updating_sliders = False
        
        # Create UI components
        self.create_ui_components()
        self.create_layout()
    
    def create_ui_components(self):
        """Create all UI widgets"""
        
        # Directory selection
        self.directory_path_widget = widgets.Text(
            value=".",
            description="H5 Directory:",
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px')
        )
        
        self.load_button = widgets.Button(
            description="🔍 Load H5 Files",
            button_style='primary',
            layout=widgets.Layout(width='150px')
        )
        
        self.status_output = widgets.Output()
        
        # Tab selection
        self.tab_selection = widgets.Tab()
        
        # === SPATIAL PROPERTIES TAB WIDGETS ===
        self.spatial_file_dropdown = widgets.Dropdown(
            options=[],
            description="H5 File:",
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px')
        )
        
        self.spatial_case_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=0,
            step=1,
            description='Case:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px'),
            continuous_update=False  # Only update on release
        )
        
        self.spatial_k_slider = widgets.IntSlider(
            value=19,
            min=0,
            max=24,
            step=1,
            description='K Layer:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px'),
            continuous_update=False  # Only update on release
        )
        
        self.spatial_timestep_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=0,
            step=1,
            description='Timestep:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px'),
            continuous_update=False  # Only update on release
        )
        
        self.spatial_plot_output = widgets.Output()
        
        # Debug output for spatial (separate from plot)
        self.spatial_debug_output = widgets.Output()
        
        # Masking control widget
        self.mask_toggle = widgets.Checkbox(
            value=True,
            description="Apply Inactive Cell Masking",
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px')
        )
        
        # Well locations control widget
        self.show_wells_toggle = widgets.Checkbox(
            value=False,
            description="Show Well Locations",
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px')
        )
        
        # === TIME SERIES TAB WIDGETS ===
        self.timeseries_file_dropdown = widgets.Dropdown(
            options=[],
            description="H5 File:",
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px')
        )
        
        self.timeseries_case_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=0,
            step=1,
            description='Case:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px'),
            continuous_update=False
        )
        
        self.wells_selector = widgets.SelectMultiple(
            options=[],
            description="Wells:",
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px', height='150px')
        )
        
        self.timeseries_plot_output = widgets.Output()
        
        # File info widget
        self.file_info_widget = widgets.HTML()
        
        # Event handlers - Set up after widget creation
        self.setup_event_handlers()
    
    def setup_event_handlers(self):
        """Set up all event handlers"""
        self.load_button.on_click(self.on_load_directory)
        
        # Spatial event handlers
        self.spatial_file_dropdown.observe(self.on_spatial_file_change, names='value')
        self.spatial_case_slider.observe(self.on_spatial_param_change, names='value')
        self.spatial_k_slider.observe(self.on_spatial_param_change, names='value')
        self.spatial_timestep_slider.observe(self.on_spatial_param_change, names='value')
        self.mask_toggle.observe(self.on_spatial_param_change, names='value')
        self.show_wells_toggle.observe(self.on_spatial_param_change, names='value')
        
        # Time series event handlers
        self.timeseries_file_dropdown.observe(self.on_timeseries_file_change, names='value')
        self.timeseries_case_slider.observe(self.on_timeseries_param_change, names='value')
        self.wells_selector.observe(self.on_timeseries_param_change, names='value')
    
    def create_layout(self):
        """Create the main layout"""
        
        # File loading section
        file_section = widgets.VBox([
            widgets.HTML("<h2>📊 Interactive H5 Data Visualizer</h2>"),
            widgets.HTML("<h3>📁 Directory Selection</h3>"),
            widgets.HTML("<p><i>Select a directory containing processed .h5 files for visualization</i></p>"),
            widgets.HBox([self.directory_path_widget, self.load_button]),
            self.status_output,
            self.file_info_widget
        ])
        
        # Spatial properties tab
        spatial_tab = widgets.VBox([
            widgets.HTML("<h3>🗺️ Spatial Properties Visualization</h3>"),
            widgets.HTML("<p>Visualize spatial data as I×J plots for selected case, K-layer, and timestep.</p>"),
            self.spatial_file_dropdown,
            widgets.HBox([
                widgets.VBox([
                    self.spatial_case_slider,
                    self.spatial_k_slider,
                    self.spatial_timestep_slider,
                    self.mask_toggle,
                    self.show_wells_toggle,
                    widgets.HTML("<p><i>💡 Tip: Sliders update plot when released</i></p>"),
                    self.spatial_debug_output
                ], layout=widgets.Layout(width='420px')),
                self.spatial_plot_output
            ])
        ])
        
        # Time series tab
        timeseries_tab = widgets.VBox([
            widgets.HTML("<h3>🛢️ Time Series Data Visualization</h3>"),
            widgets.HTML("<p>Visualize well time series data for selected case and wells.</p>"),
            self.timeseries_file_dropdown,
            widgets.HBox([
                widgets.VBox([
                    self.timeseries_case_slider,
                    widgets.HTML("<b>Select Wells:</b>"),
                    self.wells_selector
                ], layout=widgets.Layout(width='320px')),
                self.timeseries_plot_output
            ])
        ])
        
        # Create tabs
        tabs = [spatial_tab, timeseries_tab]
        titles = ['🗺️ Spatial Properties', '🛢️ Time Series Data']
        
        self.tab_selection.children = tabs
        self.tab_selection.titles = titles
        
        # Main layout
        self.main_layout = widgets.VBox([
            file_section,
            widgets.HTML("<hr>"),
            self.tab_selection
        ])
    
    def display(self):
        """Display the interactive dashboard"""
        display(self.main_layout)
    
    def on_load_directory(self, button):
        """Handle directory loading"""
        with self.status_output:
            clear_output()
            print("🔍 Loading H5 files...")
            
            success = self.load_h5_directory(self.directory_path_widget.value)
            if success:
                print("✅ H5 files loaded successfully!")
                print(f"📁 Found {len(self.spatial_files)} spatial and {len(self.timeseries_files)} time series files")
                
                # Control mask toggle visibility
                self.mask_toggle.layout.visibility = 'visible' if self.has_inactive_mask else 'hidden'
                # Control well locations toggle visibility
                self.show_wells_toggle.layout.visibility = 'visible' if self.has_well_locations else 'hidden'
                
                self.update_file_dropdowns()
                self.display_file_info()
                print("✅ Dashboard updated with H5 files!")
                    
            else:
                print("❌ Failed to load H5 files")
    
    def load_h5_directory(self, directory_path):
        """Load and categorize H5 files from directory"""
        self.directory_path = Path(directory_path)
        
        if not self.directory_path.exists():
            print(f"❌ Directory not found: {directory_path}")
            return False
        
        if not self.directory_path.is_dir():
            print(f"❌ Path is not a directory: {directory_path}")
            return False
        
        # Find all .h5 files
        h5_files = list(self.directory_path.glob('*.h5'))
        
        if not h5_files:
            print(f"❌ No .h5 files found in directory: {directory_path}")
            return False
        
        # Look for inactive cell locations file first
        inactive_file = self.directory_path / 'inactive_cell_locations.h5'
        if inactive_file.exists():
            try:
                with h5py.File(inactive_file, 'r') as f:
                    self.inactive_mask = f['inactive_mask'][...]
                    self.active_mask = f['active_mask'][...]
                    self.has_inactive_mask = True
                    
                    # Check if this is per-case format (4D) or legacy format (3D)
                    if self.inactive_mask.ndim == 4:
                        print(f"✅ Loaded per-case inactive cell masks: {self.inactive_mask.shape} (N_cases, Nx, Ny, Nz)")
                        n_cases = self.inactive_mask.shape[0]
                        total_active = np.sum(self.active_mask)
                        total_inactive = np.sum(self.inactive_mask)
                        print(f"   🗂️ Cases: {n_cases}")
                        print(f"   🟢 Total active cells: {total_active:,}")
                        print(f"   🔴 Total inactive cells: {total_inactive:,}")
                        
                        # Show statistics per case
                        for case_idx in range(min(n_cases, 5)):  # Show first 5 cases
                            case_active = np.sum(self.active_mask[case_idx])
                            case_inactive = np.sum(self.inactive_mask[case_idx])
                            print(f"     Case {case_idx}: Active: {case_active:,}, Inactive: {case_inactive:,}")
                        if n_cases > 5:
                            print(f"     ... and {n_cases-5} more cases")
                            
                    elif self.inactive_mask.ndim == 3:
                        print(f"✅ Loaded legacy inactive cell mask: {self.inactive_mask.shape} (Nx, Ny, Nz)")
                        print(f"   🟢 Active cells: {np.sum(self.active_mask):,}")
                        print(f"   🔴 Inactive cells: {np.sum(self.inactive_mask):,}")
                    else:
                        print(f"⚠️ Unexpected mask dimensions: {self.inactive_mask.shape}")
                        
            except Exception as e:
                print(f"⚠️ Could not load inactive cell mask: {e}")
                self.has_inactive_mask = False
        else:
            print("ℹ️ No inactive_cell_locations.h5 found - will show all cells")
            self.has_inactive_mask = False
        
        # Look for well locations file (check new name first, then legacy)
        well_file = self.directory_path / 'batch_spatial_properties_well_location.h5'
        if not well_file.exists():
            well_file = self.directory_path / 'well_locations.h5'  # Legacy filename
        if well_file.exists():
            try:
                with h5py.File(well_file, 'r') as f:
                    self.well_locations = f['data'][...]
                    
                    # Load well_mapping if available (for enhanced display with names/types)
                    if 'well_mapping' in f:
                        self.well_mapping = f['well_mapping'][...]
                    else:
                        self.well_mapping = None
                    
                    if 'metadata' in f:
                        meta = f['metadata']
                        if 'well_names' in meta:
                            self.well_names = [name.decode() if isinstance(name, bytes) else str(name) 
                                              for name in meta['well_names'][...]]
                        if 'well_types' in meta:
                            self.well_types = [wt.decode() if isinstance(wt, bytes) else str(wt) 
                                             for wt in meta['well_types'][...]]
                    
                    self.has_well_locations = True
                    
                    if self.well_locations.ndim == 5:
                        mapping_info = " (with well mapping)" if self.well_mapping is not None else " (legacy format)"
                        print(f"✅ Loaded well locations: {self.well_locations.shape} (N_cases, N_timesteps, Nx, Ny, Nz){mapping_info}")
                        print(f"   🛢️ Wells: {len(self.well_names)} ({len([w for w in self.well_types if w == 'I'])} injectors, {len([w for w in self.well_types if w == 'P'])} producers)")
                    else:
                        print(f"⚠️ Unexpected well locations dimensions: {self.well_locations.shape}")
                        self.has_well_locations = False
                        
            except Exception as e:
                print(f"⚠️ Could not load well locations: {e}")
                self.has_well_locations = False
        else:
            print("ℹ️ No well locations file found (batch_spatial_properties_well_location.h5 or well_locations.h5) - well locations will not be shown")
            self.has_well_locations = False
        
        # Categorize remaining files
        self.spatial_files = []
        self.timeseries_files = []
        
        for h5_file in h5_files:
            # Skip the inactive cell locations and well locations files as they're not data to visualize
            if h5_file.name in ['inactive_cell_locations.h5', 'well_locations.h5', 'batch_spatial_properties_well_location.h5']:
                continue
                
            # Check if it's spatial or time series by examining metadata
            try:
                with h5py.File(h5_file, 'r') as f:
                    if 'data' in f and f['data'].ndim == 5:
                        # 5D array suggests spatial data (N_cases, N_timesteps, Nx, Ny, Nz)
                        self.spatial_files.append(h5_file)
                    elif 'data' in f and f['data'].ndim == 3:
                        # 3D array suggests time series data (N_cases, N_timesteps, N_wells)
                        self.timeseries_files.append(h5_file)
            except Exception as e:
                print(f"⚠️ Could not read {h5_file.name}: {e}")
                continue
        
        return len(self.spatial_files) > 0 or len(self.timeseries_files) > 0
    
    def update_file_dropdowns(self):
        """Update file dropdown options"""
        # Update spatial files dropdown
        spatial_options = [(f.name, f) for f in self.spatial_files]
        self.spatial_file_dropdown.options = spatial_options
        
        # Update time series files dropdown
        timeseries_options = [(f.name, f) for f in self.timeseries_files]
        self.timeseries_file_dropdown.options = timeseries_options
    
    def display_file_info(self):
        """Display file information"""
        # Add inactive cell info if available
        mask_info = ""
        if self.has_inactive_mask:
            if self.inactive_mask.ndim == 4:
                # Per-case masking
                n_cases = self.inactive_mask.shape[0]
                total_cells = self.inactive_mask.size
                active_cells = np.sum(self.active_mask)
                inactive_cells = np.sum(self.inactive_mask)
                mask_info = f"""
                        <div style="margin-top: 10px; padding: 8px; background-color: #e8f5e8; border-radius: 5px;">
                            <b>🎭 Cell Masking:</b> Per-Case Active ✨<br>
                            <small>🗂️ Cases: {n_cases} | 🟢 Total Active: {active_cells:,} | 🔴 Total Inactive: {inactive_cells:,}</small><br>
                            <small>💡 Each case has its own geological realization mask</small>
                        </div>"""
            else:
                # Legacy single mask
                total_cells = self.inactive_mask.size
                active_cells = np.sum(self.active_mask)
                inactive_cells = np.sum(self.inactive_mask)
                mask_info = f"""
                        <div style="margin-top: 10px; padding: 8px; background-color: #e8f5e8; border-radius: 5px;">
                            <b>🎭 Cell Masking:</b> Legacy Single Mask<br>
                            <small>🟢 Active: {active_cells:,} | 🔴 Inactive: {inactive_cells:,} | 📊 Total: {total_cells:,}</small>
                        </div>"""
        else:
            mask_info = f"""
                    <div style="margin-top: 10px; padding: 8px; background-color: #fff3e0; border-radius: 5px;">
                        <b>🎭 Cell Masking:</b> None - All cells shown<br>
                        <small>No inactive_cell_locations.h5 found</small>
                    </div>"""
        
        info_html = f"""
        <div style="background-color: #f0f8ff; padding: 15px; border-radius: 10px; margin: 10px 0;">
            <h3>📊 H5 Files Information</h3>
            <div style="display: flex; flex-wrap: wrap; gap: 20px;">
                <div>
                    <b>📁 Directory:</b> {self.directory_path}<br>
                    <b>🗺️ Spatial Files:</b> {len(self.spatial_files)}<br>
                    <b>🛢️ Time Series Files:</b> {len(self.timeseries_files)}
                    {mask_info}
                </div>
                <div>
                    <b>📄 Spatial Files:</b><br>
                    {' | '.join([f.name for f in self.spatial_files[:3]])}{'...' if len(self.spatial_files) > 3 else ''}<br>
                    <b>📄 Time Series Files:</b><br>
                    {' | '.join([f.name for f in self.timeseries_files[:3]])}{'...' if len(self.timeseries_files) > 3 else ''}
                </div>
            </div>
        </div>
        """
        self.file_info_widget.value = info_html
    
    def on_spatial_file_change(self, change):
        """Handle spatial file selection change"""
        if change['new'] is None:
            return
            
        selected_file = change['new']
        
        try:
            with h5py.File(selected_file, 'r') as f:
                # Load data and metadata
                self.current_spatial_data = f['data'][...]
                meta = f['metadata']
                
                self.spatial_metadata = {
                    'property_name': meta.attrs['property_name'],
                    'n_cases': meta.attrs['n_cases'],
                    'n_timesteps': meta.attrs['n_timesteps'],
                    'nx': meta.attrs['nx'],
                    'ny': meta.attrs['ny'],
                    'nz': meta.attrs['nz'],
                    'timesteps': meta['timesteps'][...]
                }
            
            # Update sliders with flag to prevent recursive calls
            self._updating_sliders = True
            
            self.spatial_case_slider.max = self.spatial_metadata['n_cases'] - 1
            self.spatial_case_slider.value = 0
            self.spatial_k_slider.max = self.spatial_metadata['nz'] - 1
            self.spatial_k_slider.value = min(19, self.spatial_metadata['nz'] - 1)
            self.spatial_timestep_slider.max = self.spatial_metadata['n_timesteps'] - 1
            self.spatial_timestep_slider.value = 0
            
            self._updating_sliders = False
            
            # Calculate global color range from all timesteps
            self._calculate_global_color_range()
            
            # Clear debug output
            with self.spatial_debug_output:
                clear_output()
                print(f"✅ Loaded: {self.spatial_metadata['property_name']}")
                print(f"📊 Shape: {self.current_spatial_data.shape}")
                print(f"📋 Cases: {self.spatial_metadata['n_cases']}, "
                      f"Timesteps: {self.spatial_metadata['n_timesteps']}, "
                      f"Grid: {self.spatial_metadata['nx']}×{self.spatial_metadata['ny']}×{self.spatial_metadata['nz']}")
                print(f"🎨 Global color range: [{self.global_vmin:.4f}, {self.global_vmax:.4f}]")
            
            # Initial plot
            self.update_spatial_plot()
            
        except Exception as e:
            with self.spatial_plot_output:
                clear_output()
                print(f"❌ Error loading spatial file: {e}")
    
    def on_spatial_param_change(self, change):
        """Handle spatial parameter changes"""
        # Prevent recursive updates when we're programmatically setting slider values
        if self._updating_sliders:
            return
            
        if self.current_spatial_data is not None and self.spatial_metadata is not None:
            # Update debug info in separate output (not plot area)
            with self.spatial_debug_output:
                clear_output(wait=True)
                print(f"✅ Loaded: {self.spatial_metadata['property_name']}")
                print(f"📊 Shape: {self.current_spatial_data.shape}")
                print(f"🔄 Current: Case={self.spatial_case_slider.value}, "
                      f"K={self.spatial_k_slider.value}, TS={self.spatial_timestep_slider.value}")
            
            # Update the plot
            self.update_spatial_plot()
    
    def _calculate_global_color_range(self):
        """Calculate global min/max color range from all timesteps in current spatial data"""
        if self.current_spatial_data is None:
            self.global_vmin, self.global_vmax = 0, 1
            return
        
        # Collect all valid data from all cases, timesteps, and layers
        all_valid_data = []
        
        n_cases = self.current_spatial_data.shape[0]
        n_timesteps = self.current_spatial_data.shape[1]
        n_layers = self.current_spatial_data.shape[4]
        
        for case in range(n_cases):
            for timestep in range(n_timesteps):
                for k_layer in range(n_layers):
                    data_slice = self.current_spatial_data[case, timestep, :, :, k_layer]
                    
                    # Apply inactive cell mask if available
                    if self.has_inactive_mask and self.inactive_mask is not None:
                        if self.inactive_mask.ndim == 4:
                            # Per-case masking
                            if case < self.inactive_mask.shape[0]:
                                layer_inactive_mask = self.inactive_mask[case, :, :, k_layer]
                            else:
                                layer_inactive_mask = self.inactive_mask[0, :, :, k_layer]
                        else:
                            # Legacy 3D format
                            layer_inactive_mask = self.inactive_mask[:, :, k_layer]
                        
                        # Mask out inactive cells
                        masked_data = data_slice.copy().astype(float)
                        masked_data[layer_inactive_mask] = np.nan
                        valid_data = masked_data[~np.isnan(masked_data)]
                    else:
                        # No masking: use all data
                        valid_data = data_slice.flatten()
                    
                    # Filter out NaN values and collect
                    if len(valid_data) > 0:
                        all_valid_data.extend(valid_data.tolist())
        
        # Calculate global color range
        if len(all_valid_data) > 0:
            all_valid_data = np.array(all_valid_data)
            # Use percentile approach like current code (2nd and 98th percentile)
            active_data = all_valid_data[all_valid_data > 0] if len(all_valid_data[all_valid_data > 0]) > 0 else all_valid_data
            if len(active_data) > 0:
                self.global_vmin, self.global_vmax = np.percentile(active_data, [2, 98])
            else:
                self.global_vmin, self.global_vmax = all_valid_data.min(), all_valid_data.max()
        else:
            self.global_vmin, self.global_vmax = 0, 1
    
    def update_spatial_plot(self):
        """Update the spatial plot with high resolution and modern styling"""
        if self.current_spatial_data is None or self.spatial_metadata is None:
            with self.spatial_plot_output:
                clear_output()
                print("⚠️ No spatial data loaded")
            return
        
        try:
            case = self.spatial_case_slider.value
            k_layer = self.spatial_k_slider.value
            timestep_idx = self.spatial_timestep_slider.value
            
            # Validate indices
            if case >= self.spatial_metadata['n_cases']:
                case = self.spatial_metadata['n_cases'] - 1
            if k_layer >= self.spatial_metadata['nz']:
                k_layer = self.spatial_metadata['nz'] - 1
            if timestep_idx >= self.spatial_metadata['n_timesteps']:
                timestep_idx = self.spatial_metadata['n_timesteps'] - 1
            
            with self.spatial_plot_output:
                clear_output(wait=True)
                
                # Extract data for selected parameters
                data_slice = self.current_spatial_data[case, timestep_idx, :, :, k_layer]
                
                # Apply inactive cell mask if available
                if self.has_inactive_mask and self.inactive_mask is not None and self.mask_toggle.value:
                    # Handle both 4D (per-case) and 3D (legacy) mask formats
                    if self.inactive_mask.ndim == 4:
                        # Per-case masking: use mask for current case
                        if case < self.inactive_mask.shape[0]:
                            layer_inactive_mask = self.inactive_mask[case, :, :, k_layer]
                        else:
                            # Fallback to first case if case index is out of bounds
                            layer_inactive_mask = self.inactive_mask[0, :, :, k_layer]
                            print(f"⚠️ Case {case} out of bounds, using case 0 mask")
                    else:
                        # Legacy 3D format: same mask for all cases
                        layer_inactive_mask = self.inactive_mask[:, :, k_layer]
                    
                    # Transpose mask to match plot_data.T (imshow will transpose, so mask must match transposed axes)
                    layer_inactive_mask = layer_inactive_mask.T
                    
                    # Transpose data slice to match mask orientation
                    data_slice_T = data_slice.T
                    
                    # Create masked array - set inactive cells to NaN so they don't display
                    masked_data_T = data_slice_T.copy().astype(float)
                    masked_data_T[layer_inactive_mask] = np.nan
                    
                    # Count active cells for info (use original mask before transpose)
                    active_cells_in_layer = np.sum(~self.inactive_mask[case, :, :, k_layer] if self.inactive_mask.ndim == 4 else ~self.inactive_mask[:, :, k_layer])
                    total_cells_in_layer = layer_inactive_mask.size
                    
                    plot_data = masked_data_T
                    if self.inactive_mask.ndim == 4:
                        mask_status = f"🎭 Masked Case {case} ({active_cells_in_layer}/{total_cells_in_layer} active cells)"
                    else:
                        mask_status = f"🎭 Masked ({active_cells_in_layer}/{total_cells_in_layer} active cells)"
                else:
                    # No masking: transpose data for plotting (to match axis orientation)
                    plot_data = data_slice.T
                    if self.has_inactive_mask and not self.mask_toggle.value:
                        mask_status = "📊 Masking disabled - All cells shown"
                    else:
                        mask_status = "📊 No masking applied"
                
                # Create high-resolution plot with modern styling
                plt.style.use('default')  # Clean style
                fig, ax = plt.subplots(figsize=(12, 10), dpi=100)
                
                # Use global color range (fixed across all timesteps)
                vmin = self.global_vmin if self.global_vmin is not None else 0
                vmax = self.global_vmax if self.global_vmax is not None else 1
                
                # Create the plot with high resolution interpolation
                # NaN values will automatically be transparent/hidden
                # plot_data is already transposed when masking is applied, or transposed above when not masking
                im = ax.imshow(plot_data, 
                             origin='lower', 
                             cmap='jet',  # Jet colormap as requested
                             vmin=vmin, 
                             vmax=vmax, 
                             aspect='equal',  # Equal aspect ratio
                             interpolation='bilinear')  # Smooth interpolation
                
                # Enhanced styling
                timestep_value = self.spatial_metadata['timesteps'][timestep_idx]
                title = f"{self.spatial_metadata['property_name']} - Case {case}, K-Layer {k_layer}, Timestep {timestep_value}"
                ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
                # Note: x-axis is J and y-axis is I because plot_data is transposed
                ax.set_xlabel('J Index', fontsize=14, fontweight='bold')
                ax.set_ylabel('I Index', fontsize=14, fontweight='bold')
                
                # Colorbar - same height as main plot, rotated label
                cbar = plt.colorbar(im, ax=ax, shrink=0.5, aspect=30, pad=0.02)
                cbar.set_label(f'{self.spatial_metadata["property_name"]}', 
                             rotation=90, labelpad=20, fontsize=12, fontweight='bold')
                cbar.ax.tick_params(labelsize=10)
                
                # Add well location markers if enabled
                if self.has_well_locations and self.show_wells_toggle.value and self.well_locations is not None:
                    well_points = self._get_well_locations_for_layer(case, timestep_idx, k_layer)
                    if well_points:
                        self._plot_well_markers(ax, case, timestep_idx, k_layer)
                    else:
                        # Debug: print why no wells shown
                        print(f"ℹ️ No wells found in Case {case}, Timestep {timestep_idx}, K-Layer {k_layer}")
                        print(f"   Well locations shape: {self.well_locations.shape}")
                        print(f"   Checking layer: {self.well_locations[case, timestep_idx, :, :, k_layer].sum()} well cells")
                
                # Clean styling - no grid
                ax.tick_params(labelsize=12)
                
                # Tight layout for better appearance
                plt.tight_layout()
                plt.show()
                
        except Exception as e:
            with self.spatial_plot_output:
                clear_output()
                print(f"❌ Error updating spatial plot: {e}")
                print(f"   Data shape: {self.current_spatial_data.shape}")
                print(f"   Requested indices: case={case}, timestep={timestep_idx}, k={k_layer}")
                if self.has_inactive_mask:
                    print(f"   Inactive mask shape: {self.inactive_mask.shape}")
                    print(f"   Layer mask shape: {self.inactive_mask[:, :, k_layer].shape}")
    
    def _get_well_locations_for_layer(self, case, timestep_idx, k_layer):
        """
        Extract well locations for a specific case, timestep, and K-layer
        
        Returns:
            list of tuples: (j, i, well_name, well_type) for wells in this layer
        """
        if not self.has_well_locations or self.well_locations is None:
            return []
        
        if self.well_locations.ndim != 5:
            return []
        
        # Validate indices
        if case >= self.well_locations.shape[0]:
            return []
        if timestep_idx >= self.well_locations.shape[1]:
            return []
        if k_layer >= self.well_locations.shape[4]:
            return []
        
        # Extract well locations for this layer
        layer_wells = self.well_locations[case, timestep_idx, :, :, k_layer]
        
        # Find cells with wells (value == 1)
        well_coords = np.where(layer_wells == 1)
        
        # Convert to list of (j, i, well_name, well_type) tuples for plotting
        well_points = []
        
        # Check if we have well_mapping for enhanced display
        if self.well_mapping is not None and self.well_mapping.ndim == 5:
            layer_mapping = self.well_mapping[case, timestep_idx, :, :, k_layer]
            
            for i, j in zip(well_coords[0], well_coords[1]):
                well_idx = int(layer_mapping[i, j])
                
                # Get well name and type from metadata
                if well_idx >= 0 and well_idx < len(self.well_names):
                    well_name = self.well_names[well_idx]
                    well_type = self.well_types[well_idx] if well_idx < len(self.well_types) else 'U'
                else:
                    well_name = "?"
                    well_type = "?"
                
                well_points.append((j, i, well_name, well_type))  # Note: j, i order for plotting
        else:
            # Fallback to legacy format (no well mapping available)
            for i, j in zip(well_coords[0], well_coords[1]):
                well_points.append((j, i, "?", "?"))  # Note: j, i order for plotting
        
        return well_points
    
    def _plot_well_markers(self, ax, case, timestep_idx, k_layer):
        """
        Overlay well location markers on the spatial plot
        
        Args:
            ax: Matplotlib axis object
            case: Case index
            timestep_idx: Timestep index
            k_layer: K-layer index
        """
        well_points = self._get_well_locations_for_layer(case, timestep_idx, k_layer)
        
        if not well_points:
            return
        
        # Separate wells by type
        injectors = [(j, i, name, wtype) for j, i, name, wtype in well_points if wtype == 'I']
        producers = [(j, i, name, wtype) for j, i, name, wtype in well_points if wtype == 'P']
        unknown = [(j, i, name, wtype) for j, i, name, wtype in well_points if wtype not in ['I', 'P']]
        
        # Plot injectors (blue triangles)
        if injectors:
            j_coords = [p[0] for p in injectors]
            i_coords = [p[1] for p in injectors]
            ax.scatter(j_coords, i_coords, 
                      c='blue', marker='^', s=120, 
                      edgecolors='black', linewidths=2,
                      label=f'Injectors ({len(injectors)})', zorder=10)
            
            # Add well name labels for injectors (limit to 20 to avoid clutter)
            for j, i, name, wtype in injectors[:20]:
                label = name if name != "?" else "I"
                ax.text(j, i + 0.5, label, ha='center', va='bottom', 
                       fontsize=7, fontweight='bold', color='blue', zorder=11)
        
        # Plot producers (red inverted triangles)
        if producers:
            j_coords = [p[0] for p in producers]
            i_coords = [p[1] for p in producers]
            ax.scatter(j_coords, i_coords, 
                      c='red', marker='v', s=120, 
                      edgecolors='black', linewidths=2,
                      label=f'Producers ({len(producers)})', zorder=10)
            
            # Add well name labels for producers (limit to 20 to avoid clutter)
            for j, i, name, wtype in producers[:20]:
                label = name if name != "?" else "P"
                ax.text(j, i - 0.5, label, ha='center', va='top', 
                       fontsize=7, fontweight='bold', color='red', zorder=11)
        
        # Plot unknown type wells (yellow circles)
        if unknown:
            j_coords = [p[0] for p in unknown]
            i_coords = [p[1] for p in unknown]
            ax.scatter(j_coords, i_coords, 
                      c='yellow', marker='o', s=100, 
                      edgecolors='black', linewidths=2,
                      label=f'Unknown ({len(unknown)})', zorder=10)
            
            # Add well name labels for unknown (limit to 20 to avoid clutter)
            for j, i, name, wtype in unknown[:20]:
                label = name if name != "?" else "W"
                ax.text(j, i + 0.5, label, ha='center', va='bottom', 
                       fontsize=7, fontweight='bold', color='black', zorder=11)
        
        # Add legend if any wells plotted
        if injectors or producers or unknown:
            ax.legend(loc='upper right', fontsize=9)
    
    def on_timeseries_file_change(self, change):
        """Handle time series file selection change"""
        if change['new'] is None:
            return
            
        selected_file = change['new']
        
        try:
            with h5py.File(selected_file, 'r') as f:
                # Load data and metadata
                self.current_timeseries_data = f['data'][...]
                meta = f['metadata']
                
                well_names = [w.decode() for w in meta['well_names'][...]]
                dates = [d.decode() for d in meta['dates'][...]]
                
                self.timeseries_metadata = {
                    'variable_name': meta.attrs['variable_name'],
                    'n_cases': meta.attrs['n_cases'],
                    'n_timesteps': meta.attrs['n_timesteps'],
                    'n_wells': meta.attrs['n_wells'],
                    'well_names': well_names,
                    'dates': dates,
                    'timesteps': meta['timesteps'][...]
                }
            
            # Update widgets
            self.timeseries_case_slider.max = self.timeseries_metadata['n_cases'] - 1
            self.timeseries_case_slider.value = 0
            
            self.wells_selector.options = well_names
            self.wells_selector.value = [well_names[0]] if well_names else []
            
            # Initial plot
            self.update_timeseries_plot()
            
        except Exception as e:
            with self.timeseries_plot_output:
                clear_output()
                print(f"❌ Error loading time series file: {e}")
    
    def on_timeseries_param_change(self, change):
        """Handle time series parameter changes"""
        if self.current_timeseries_data is not None:
            self.update_timeseries_plot()
    
    def update_timeseries_plot(self):
        """Update the time series plot with enhanced styling"""
        if self.current_timeseries_data is None or self.timeseries_metadata is None:
            return
        
        case = self.timeseries_case_slider.value
        selected_wells = list(self.wells_selector.value)
        
        if not selected_wells:
            with self.timeseries_plot_output:
                clear_output()
                print("⚠️ Please select at least one well")
            return
        
        with self.timeseries_plot_output:
            clear_output(wait=True)
            
            # Create high-resolution plot
            plt.style.use('default')
            fig, ax = plt.subplots(figsize=(14, 8), dpi=100)
            
            # Color palette for multiple wells
            colors = plt.cm.Set1(np.linspace(0, 1, len(selected_wells)))
            
            # Plot each selected well
            for idx, well_name in enumerate(selected_wells):
                well_idx = self.timeseries_metadata['well_names'].index(well_name)
                well_data = self.current_timeseries_data[case, :, well_idx]
                
                ax.plot(range(len(well_data)), well_data, 'o-', 
                       linewidth=3, markersize=6, label=well_name, 
                       color=colors[idx], alpha=0.8)
            
            # Enhanced styling
            ax.set_title(f"{self.timeseries_metadata['variable_name']} - Case {case}", 
                        fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Time Index', fontsize=14, fontweight='bold')
            ax.set_ylabel(f"{self.timeseries_metadata['variable_name']}", fontsize=14, fontweight='bold')
            
            # Modern legend
            ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=12)
            ax.grid(True, alpha=0.3, linewidth=0.5)
            ax.tick_params(labelsize=12)
            
            # Add date labels if available
            dates = self.timeseries_metadata['dates']
            if len(dates) == len(well_data):
                # Show fewer date labels to avoid crowding
                step = max(1, len(dates) // 8)
                ax.set_xticks(range(0, len(dates), step))
                ax.set_xticklabels([dates[i] for i in range(0, len(dates), step)], 
                                  rotation=45, ha='right')
                ax.set_xlabel('Date', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.show()


def create_h5_visualizer():
    """
    Create and display the H5 visualizer dashboard
    
    Usage in Jupyter cell:
    ```python
    from interactive_h5_visualizer import create_h5_visualizer
    visualizer = create_h5_visualizer()
    ```
    """
    
    visualizer = InteractiveH5Visualizer()
    visualizer.display()
    return visualizer


if __name__ == "__main__":
    # For Jupyter usage - create and display dashboard
    visualizer = create_h5_visualizer()
    print("🎉 Interactive H5 Visualizer created!")
    print("The dashboard should appear above this message.")
    print("\nTo use in other cells:")
    print("visualizer = create_h5_visualizer()")
    print("\n🎭 Automatic inactive cell masking!")
    print("   • Place inactive_cell_locations.h5 in same folder as your H5 data")
    print("   • Spatial plots will automatically mask inactive cells")
    print("   • Toggle masking on/off with checkbox control")


# Features Documentation
"""
Interactive H5 Data Visualizer
==============================

KEY FEATURES:
------------
1. **Automatic Mask Detection**: Looks for 'inactive_cell_locations.h5' in data directory
2. **Clean Reservoir Visualization**: Hides inactive cells showing only active reservoir geometry  
3. **Toggle Control**: Checkbox to enable/disable masking for comparison
4. **Smart Color Scaling**: Uses only active cell values for optimal color range
5. **Mask Status Display**: Shows active/inactive cell counts on plots

VISUALIZATION MODES:
-------------------
1. **🗺️ Spatial Properties**: 2D I×J plots with K-layer selection
2. **🛢️ Time Series Data**: Multi-well temporal plots

Usage Example:
-------------
```python
# In Jupyter notebook:
from interactive_h5_visualizer import create_h5_visualizer
visualizer = create_h5_visualizer()

# Directory structure should be:
# your_output_folder/
# ├── inactive_cell_locations.h5          # ← Automatically detected and used
# ├── batch_spatial_properties_POROS.h5   
# ├── batch_spatial_properties_PRES.h5    
# └── batch_timeseries_data_BHP.h5        
```

Benefits:
--------
- **Cleaner Visualization**: Only active reservoir cells are shown
- **Better Color Scaling**: Color range optimized for active data only
- **Geological Clarity**: True reservoir geometry without inactive cells
- **Flexible Control**: Easy toggle between masked/unmasked views
- **Professional Quality**: Publication-ready visualizations
"""

# %%
