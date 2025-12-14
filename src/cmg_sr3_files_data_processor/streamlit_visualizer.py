"""
Streamlit wrapper for InteractiveH5Visualizer
Adapts the Jupyter-based visualizer for Streamlit web application
"""

import streamlit as st
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib for high-resolution plots
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300


class StreamlitH5Visualizer:
    """
    Streamlit-adapted wrapper for InteractiveH5Visualizer
    Converts matplotlib plots to Streamlit display and replaces widgets with Streamlit components
    """
    
    def __init__(self):
        """Initialize the Streamlit visualizer"""
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
    
    def load_h5_directory(self, directory_path):
        """
        Load and categorize H5 files from directory
        
        Args:
            directory_path: Path to directory containing H5 files
            
        Returns:
            bool: True if files were loaded successfully
        """
        self.directory_path = Path(directory_path)
        
        if not self.directory_path.exists():
            st.error(f"Directory not found: {directory_path}")
            return False
        
        if not self.directory_path.is_dir():
            st.error(f"Path is not a directory: {directory_path}")
            return False
        
        # Find all .h5 files
        h5_files = list(self.directory_path.glob('*.h5'))
        
        if not h5_files:
            st.error(f"No .h5 files found in directory: {directory_path}")
            return False
        
        # Look for inactive cell locations file first
        inactive_file = self.directory_path / 'inactive_cell_locations.h5'
        if inactive_file.exists():
            try:
                with h5py.File(inactive_file, 'r') as f:
                    self.inactive_mask = f['inactive_mask'][...]
                    self.active_mask = f['active_mask'][...]
                    self.has_inactive_mask = True
                    
                    if self.inactive_mask.ndim == 4:
                        st.success(f"✅ Loaded per-case inactive cell masks: {self.inactive_mask.shape}")
                    elif self.inactive_mask.ndim == 3:
                        st.success(f"✅ Loaded legacy inactive cell mask: {self.inactive_mask.shape}")
            except Exception as e:
                st.warning(f"Could not load inactive cell mask: {e}")
                self.has_inactive_mask = False
        else:
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
                        st.success(f"✅ Loaded well locations: {self.well_locations.shape}{mapping_info}")
                        st.info(f"🛢️ Wells: {len(self.well_names)} ({len([w for w in self.well_types if w == 'I'])} injectors, {len([w for w in self.well_types if w == 'P'])} producers)")
                    else:
                        st.warning(f"Unexpected well locations dimensions: {self.well_locations.shape}")
                        self.has_well_locations = False
                        
            except Exception as e:
                st.warning(f"Could not load well locations: {e}")
                self.has_well_locations = False
        else:
            self.has_well_locations = False
        
        # Categorize remaining files
        self.spatial_files = []
        self.timeseries_files = []
        
        for h5_file in h5_files:
            # Skip the inactive cell locations and well locations files
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
                st.warning(f"Could not read {h5_file.name}: {e}")
                continue
        
        return len(self.spatial_files) > 0 or len(self.timeseries_files) > 0
    
    def get_spatial_files(self):
        """Get list of spatial file names"""
        return [f.name for f in self.spatial_files]
    
    def get_timeseries_files(self):
        """Get list of timeseries file names"""
        return [f.name for f in self.timeseries_files]
    
    def load_spatial_file(self, filename):
        """Load spatial data from file"""
        file_path = self.directory_path / filename
        
        try:
            with h5py.File(file_path, 'r') as f:
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
            
            # Calculate global color range from all timesteps
            self._calculate_global_color_range()
            
            return True
        except Exception as e:
            st.error(f"Error loading spatial file: {e}")
            return False
    
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
    
    def load_timeseries_file(self, filename):
        """Load timeseries data from file"""
        file_path = self.directory_path / filename
        
        try:
            with h5py.File(file_path, 'r') as f:
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
            
            return True
        except Exception as e:
            st.error(f"Error loading timeseries file: {e}")
            return False
    
    def _get_well_locations_for_layer(self, case, timestep_idx, k_layer):
        """Extract well locations for a specific case, timestep, and K-layer
        
        Returns:
            list of tuples: (j, i, well_name, well_type) for plotting
        """
        if not self.has_well_locations or self.well_locations is None:
            return []
        
        if self.well_locations.ndim != 5:
            return []
        
        # Validate indices
        if case >= self.well_locations.shape[0] or timestep_idx >= self.well_locations.shape[1] or k_layer >= self.well_locations.shape[4]:
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
    
    def _plot_well_markers_streamlit(self, ax, case, timestep_idx, k_layer):
        """Overlay well location markers on the spatial plot for Streamlit"""
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
    
    def plot_spatial(self, case, k_layer, timestep_idx, apply_mask=True, show_wells=False):
        """
        Create spatial plot for Streamlit
        
        Args:
            case: Case index
            k_layer: K-layer index
            timestep_idx: Timestep index
            apply_mask: Whether to apply inactive cell masking
            
        Returns:
            matplotlib figure object
        """
        if self.current_spatial_data is None or self.spatial_metadata is None:
            return None
        
        try:
            # Validate indices
            case = min(case, self.spatial_metadata['n_cases'] - 1)
            k_layer = min(k_layer, self.spatial_metadata['nz'] - 1)
            timestep_idx = min(timestep_idx, self.spatial_metadata['n_timesteps'] - 1)
            
            # Extract data for selected parameters
            data_slice = self.current_spatial_data[case, timestep_idx, :, :, k_layer]
            
            # Apply inactive cell mask if available
            if self.has_inactive_mask and self.inactive_mask is not None and apply_mask:
                if self.inactive_mask.ndim == 4:
                    # Per-case masking
                    if case < self.inactive_mask.shape[0]:
                        layer_inactive_mask = self.inactive_mask[case, :, :, k_layer]
                    else:
                        layer_inactive_mask = self.inactive_mask[0, :, :, k_layer]
                else:
                    # Legacy 3D format
                    layer_inactive_mask = self.inactive_mask[:, :, k_layer]
                
                # Transpose mask to match plot_data.T (imshow will transpose, so mask must match transposed axes)
                layer_inactive_mask = layer_inactive_mask.T
                
                # Transpose data slice to match mask orientation
                data_slice_T = data_slice.T
                
                # Create masked array - set inactive cells to NaN so they don't display
                masked_data_T = data_slice_T.copy().astype(float)
                masked_data_T[layer_inactive_mask] = np.nan
                plot_data = masked_data_T
            else:
                # No masking: transpose data for plotting (to match axis orientation)
                plot_data = data_slice.T
            
            # Create plot
            fig, ax = plt.subplots(figsize=(12, 10), dpi=100)
            
            # Use global color range (fixed across all timesteps)
            vmin = self.global_vmin if self.global_vmin is not None else 0
            vmax = self.global_vmax if self.global_vmax is not None else 1
            
            # Create the plot
            # plot_data is already transposed when masking is applied, or transposed above when not masking
            im = ax.imshow(plot_data, 
                         origin='lower', 
                         cmap='jet',
                         vmin=vmin, 
                         vmax=vmax, 
                         aspect='equal',
                         interpolation='bilinear')
            
            # Enhanced styling
            timestep_value = self.spatial_metadata['timesteps'][timestep_idx]
            title = f"{self.spatial_metadata['property_name']} - Case {case}, K-Layer {k_layer}, Timestep {timestep_value}"
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('I Index', fontsize=14, fontweight='bold')
            ax.set_ylabel('J Index', fontsize=14, fontweight='bold')
            
            # Colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.5, aspect=30, pad=0.02)
            cbar.set_label(f'{self.spatial_metadata["property_name"]}', 
                         rotation=90, labelpad=20, fontsize=12, fontweight='bold')
            cbar.ax.tick_params(labelsize=10)
            
            # Add well location markers if enabled
            if show_wells and self.has_well_locations and self.well_locations is not None:
                well_points = self._get_well_locations_for_layer(case, timestep_idx, k_layer)
                if well_points:
                    self._plot_well_markers_streamlit(ax, case, timestep_idx, k_layer)
                else:
                    # Debug info
                    layer_well_count = self.well_locations[case, timestep_idx, :, :, k_layer].sum()
                    if layer_well_count == 0:
                        st.info(f"ℹ️ No wells found in K-Layer {k_layer}. Try different layers. "
                               f"Total wells in file: {self.well_locations.sum()}")
            
            ax.tick_params(labelsize=12)
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating spatial plot: {e}")
            return None
    
    def plot_timeseries(self, case, selected_wells):
        """
        Create timeseries plot for Streamlit
        
        Args:
            case: Case index
            selected_wells: List of well names to plot
            
        Returns:
            matplotlib figure object
        """
        if self.current_timeseries_data is None or self.timeseries_metadata is None:
            return None
        
        if not selected_wells:
            return None
        
        try:
            case = min(case, self.timeseries_metadata['n_cases'] - 1)
            
            # Create plot
            fig, ax = plt.subplots(figsize=(14, 8), dpi=100)
            
            # Color palette for multiple wells
            colors = plt.cm.Set1(np.linspace(0, 1, len(selected_wells)))
            
            # Plot each selected well
            for idx, well_name in enumerate(selected_wells):
                if well_name not in self.timeseries_metadata['well_names']:
                    continue
                    
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
                step = max(1, len(dates) // 8)
                ax.set_xticks(range(0, len(dates), step))
                ax.set_xticklabels([dates[i] for i in range(0, len(dates), step)], 
                                  rotation=45, ha='right')
                ax.set_xlabel('Date', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating timeseries plot: {e}")
            return None

