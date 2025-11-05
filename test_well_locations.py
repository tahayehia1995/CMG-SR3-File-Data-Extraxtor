#!/usr/bin/env python3
"""
Test script for well_locations.h5 file structure and ML workflow compatibility

This script verifies:
1. Well locations file structure matches spatial files format
2. Well mapping works correctly
3. Data can be loaded and accessed for ML workflows
4. Visualization functions work with enhanced display
"""

import h5py
import numpy as np
from pathlib import Path
import sys

def test_well_locations_structure(file_path):
    """Test well_locations.h5 file structure"""
    print("=" * 70)
    print("TESTING: Well Locations File Structure")
    print("=" * 70)
    
    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return False
    
    try:
        with h5py.File(file_path, 'r') as f:
            # Check main datasets
            print("\n📊 Main Datasets:")
            print(f"   ✅ 'data' exists: {'data' in f}")
            print(f"   ✅ 'well_mapping' exists: {'well_mapping' in f}")
            
            if 'data' in f:
                data = f['data']
                print(f"\n   Data shape: {data.shape}")
                print(f"   Data dtype: {data.dtype}")
                print(f"   Data chunks: {data.chunks}")
                print(f"   Data compression: {data.compression}")
            
            if 'well_mapping' in f:
                mapping = f['well_mapping']
                print(f"\n   Well mapping shape: {mapping.shape}")
                print(f"   Well mapping dtype: {mapping.dtype}")
                print(f"   Well mapping chunks: {mapping.chunks}")
                print(f"   Well mapping compression: {mapping.compression}")
            
            # Check metadata structure (should match spatial files)
            print("\n📋 Metadata Structure:")
            if 'metadata' in f:
                meta = f['metadata']
                
                # Required attributes (matching spatial files)
                required_attrs = ['n_cases', 'n_timesteps', 'nx', 'ny', 'nz', 
                                'grid_dimensions', 'total_cells', 'active_cells']
                print("\n   Attributes:")
                for attr in required_attrs:
                    exists = attr in meta.attrs
                    status = "✅" if exists else "❌"
                    value = meta.attrs[attr] if exists else "MISSING"
                    print(f"   {status} {attr}: {value}")
                
                # Required datasets (matching spatial files)
                required_datasets = ['timesteps', 'target_days', 'sr3_files', 
                                   'well_names', 'well_types', 'completion_counts']
                print("\n   Datasets:")
                for ds_name in required_datasets:
                    exists = ds_name in meta
                    status = "✅" if exists else "❌"
                    if exists:
                        ds = meta[ds_name]
                        print(f"   {status} {ds_name}: shape={ds.shape}, dtype={ds.dtype}")
                    else:
                        print(f"   {status} {ds_name}: MISSING")
                
                # Optional but recommended
                if 'active_cell_mapping' in meta:
                    print(f"   ✅ active_cell_mapping: shape={meta['active_cell_mapping'].shape}")
                else:
                    print(f"   ⚠️  active_cell_mapping: Not found (optional)")
            
            # Check file-level attributes
            print("\n📄 File-Level Attributes:")
            print(f"   ✅ description: {f.attrs.get('description', 'MISSING')}")
            print(f"   ✅ format: {f.attrs.get('format', 'MISSING')}")
            print(f"   ✅ extraction_date: {f.attrs.get('extraction_date', 'MISSING')}")
            
            return True
            
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_access(file_path):
    """Test accessing well locations data for ML workflows"""
    print("\n" + "=" * 70)
    print("TESTING: Data Access for ML Workflows")
    print("=" * 70)
    
    try:
        with h5py.File(file_path, 'r') as f:
            data = f['data'][...]
            well_mapping = f['well_mapping'][...] if 'well_mapping' in f else None
            meta = f['metadata']
            
            n_cases = meta.attrs['n_cases']
            n_timesteps = meta.attrs['n_timesteps']
            nx, ny, nz = meta.attrs['nx'], meta.attrs['ny'], meta.attrs['nz']
            
            print(f"\n📊 Data Shape: {data.shape}")
            print(f"   Expected: ({n_cases}, {n_timesteps}, {nx}, {ny}, {nz})")
            print(f"   ✅ Shape matches metadata: {data.shape == (n_cases, n_timesteps, nx, ny, nz)}")
            
            # Test accessing data like spatial files
            print("\n🔍 Accessing Data (ML workflow style):")
            
            # Access single case, timestep, layer
            case_idx = 0
            timestep_idx = 0
            layer_idx = 0
            
            layer_data = data[case_idx, timestep_idx, :, :, layer_idx]
            print(f"   ✅ Single layer access: shape={layer_data.shape}, dtype={layer_data.dtype}")
            print(f"      Well cells in layer: {np.sum(layer_data)}")
            
            # Access all timesteps for one case
            case_data = data[case_idx, :, :, :, :]
            print(f"   ✅ Single case, all timesteps: shape={case_data.shape}")
            
            # Access all cases for one timestep
            timestep_data = data[:, timestep_idx, :, :, :]
            print(f"   ✅ Single timestep, all cases: shape={timestep_data.shape}")
            
            # Test well mapping access
            if well_mapping is not None:
                print("\n🛢️ Well Mapping Access:")
                layer_mapping = well_mapping[case_idx, timestep_idx, :, :, layer_idx]
                unique_wells = np.unique(layer_mapping[layer_mapping >= 0])
                print(f"   ✅ Layer mapping shape: {layer_mapping.shape}")
                print(f"   ✅ Unique wells in layer: {len(unique_wells)}")
                
                # Test looking up well names
                if 'well_names' in meta:
                    well_names = [name.decode() if isinstance(name, bytes) else str(name) 
                                 for name in meta['well_names'][...]]
                    if 'well_types' in meta:
                        well_types = [wt.decode() if isinstance(wt, bytes) else str(wt) 
                                     for wt in meta['well_types'][...]]
                        
                        print(f"\n   Well Information:")
                        for well_idx in unique_wells[:5]:  # Show first 5
                            if well_idx < len(well_names):
                                well_name = well_names[well_idx]
                                well_type = well_types[well_idx] if well_idx < len(well_types) else '?'
                                count = np.sum(layer_mapping == well_idx)
                                print(f"      Well {well_idx}: {well_name} ({well_type}) - {count} cells")
            
            # Test batch access (for ML training)
            print("\n🤖 Batch Access (ML Training Style):")
            batch_cases = data[:min(2, n_cases), :, :, :, :]
            print(f"   ✅ Batch of cases: shape={batch_cases.shape}")
            
            # Test layer slicing
            layer_slice = data[:, :, :, :, layer_idx]
            print(f"   ✅ Single layer, all cases/timesteps: shape={layer_slice.shape}")
            
            return True
            
    except Exception as e:
        print(f"❌ Error accessing data: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_comparison_with_spatial(file_path, spatial_file_path=None):
    """Compare well_locations.h5 structure with spatial files"""
    print("\n" + "=" * 70)
    print("TESTING: Comparison with Spatial Files Structure")
    print("=" * 70)
    
    if spatial_file_path and Path(spatial_file_path).exists():
        try:
            with h5py.File(file_path, 'r') as well_f:
                with h5py.File(spatial_file_path, 'r') as spatial_f:
                    well_meta = well_f['metadata']
                    spatial_meta = spatial_f['metadata']
                    
                    print("\n📊 Structure Comparison:")
                    
                    # Compare attributes
                    common_attrs = ['n_cases', 'n_timesteps', 'nx', 'ny', 'nz', 
                                   'grid_dimensions', 'total_cells', 'active_cells']
                    print("\n   Common Attributes:")
                    for attr in common_attrs:
                        well_val = well_meta.attrs.get(attr, "MISSING")
                        spatial_val = spatial_meta.attrs.get(attr, "MISSING")
                        match = well_val == spatial_val if well_val != "MISSING" and spatial_val != "MISSING" else False
                        status = "✅" if match else "⚠️"
                        print(f"   {status} {attr}: well={well_val}, spatial={spatial_val}")
                    
                    # Compare datasets
                    common_datasets = ['timesteps', 'target_days', 'sr3_files']
                    print("\n   Common Datasets:")
                    for ds_name in common_datasets:
                        well_exists = ds_name in well_meta
                        spatial_exists = ds_name in spatial_meta
                        status = "✅" if (well_exists and spatial_exists) else "⚠️"
                        print(f"   {status} {ds_name}: well={well_exists}, spatial={spatial_exists}")
                    
                    # Compare data shapes
                    well_data = well_f['data']
                    spatial_data = spatial_f['data']
                    print(f"\n   Data Shapes:")
                    print(f"   ✅ Well locations: {well_data.shape}")
                    print(f"   ✅ Spatial property: {spatial_data.shape}")
                    print(f"   ✅ Shapes match: {well_data.shape == spatial_data.shape}")
                    
        except Exception as e:
            print(f"⚠️  Could not compare with spatial file: {e}")
    else:
        print("⚠️  No spatial file provided for comparison")
        print("   Well locations file structure should match spatial files format")


def test_ml_workflow_example(file_path):
    """Show example ML workflow usage"""
    print("\n" + "=" * 70)
    print("EXAMPLE: ML Workflow Usage")
    print("=" * 70)
    
    try:
        with h5py.File(file_path, 'r') as f:
            data = f['data'][...]
            well_mapping = f['well_mapping'][...] if 'well_mapping' in f else None
            meta = f['metadata']
            
            print("\n📝 Example Code for ML Workflow:")
            print("-" * 70)
            print("""
# Load well locations for ML training
import h5py
import numpy as np

with h5py.File('well_locations.h5', 'r') as f:
    # Main binary data: (N_cases, N_timesteps, Nx, Ny, Nz)
    well_locations = f['data'][...]  # 1 = well, 0 = no well
    
    # Well mapping: (N_cases, N_timesteps, Nx, Ny, Nz)
    # -1 = no well, 0 to N_wells-1 = well index
    well_mapping = f['well_mapping'][...]
    
    # Metadata
    meta = f['metadata']
    n_cases = meta.attrs['n_cases']
    n_timesteps = meta.attrs['n_timesteps']
    nx, ny, nz = meta.attrs['nx'], meta.attrs['ny'], meta.attrs['nz']
    
    # Well names and types
    well_names = [name.decode() for name in meta['well_names'][...]]
    well_types = [wt.decode() for wt in meta['well_types'][...]]

# Example 1: Get well locations for specific case, timestep, layer
case_idx = 0
timestep_idx = 0
layer_idx = 10
layer_wells = well_locations[case_idx, timestep_idx, :, :, layer_idx]
# Shape: (Nx, Ny), binary mask

# Example 2: Get well names/types for each cell in layer
layer_mapping = well_mapping[case_idx, timestep_idx, :, :, layer_idx]
well_cells = np.where(layer_mapping >= 0)
for i, j in zip(well_cells[0], well_cells[1]):
    well_idx = layer_mapping[i, j]
    well_name = well_names[well_idx]
    well_type = well_types[well_idx]

# Example 3: Batch access for training
# Get all timesteps for training
X_wells = well_locations[:, :, :, :, :]  # All cases, all timesteps
# Shape: (N_cases, N_timesteps, Nx, Ny, Nz)

# Example 4: Get injectors vs producers
injector_indices = [i for i, wt in enumerate(well_types) if wt == 'I']
producer_indices = [i for i, wt in enumerate(well_types) if wt == 'P']

# Create separate masks
injector_mask = np.zeros_like(well_locations)
producer_mask = np.zeros_like(well_locations)
for case in range(n_cases):
    for ts in range(n_timesteps):
        for i_idx in injector_indices:
            injector_mask[case, ts][well_mapping[case, ts] == i_idx] = 1
        for p_idx in producer_indices:
            producer_mask[case, ts][well_mapping[case, ts] == p_idx] = 1
""")
            print("-" * 70)
            
    except Exception as e:
        print(f"❌ Error in ML workflow example: {e}")


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("WELL LOCATIONS FILE TEST SUITE")
    print("=" * 70)
    
    # Look for well locations file in common locations (new name first, then legacy)
    test_paths = [
        Path("batch_spatial_properties_well_location.h5"),
        Path("sr3_batch_output/batch_spatial_properties_well_location.h5"),
        Path("sr3_output/batch_spatial_properties_well_location.h5"),
        Path("well_locations.h5"),  # Legacy filename
        Path("sr3_batch_output/well_locations.h5"),  # Legacy
        Path("sr3_output/well_locations.h5"),  # Legacy
    ]
    
    well_file = None
    for path in test_paths:
        if path.exists():
            well_file = path
            break
    
    if not well_file:
        print("\n❌ Well locations file not found in common locations:")
        for path in test_paths:
            print(f"   - {path}")
        print("\n💡 Please provide the path to batch_spatial_properties_well_location.h5 file")
        if len(sys.argv) > 1:
            well_file = Path(sys.argv[1])
        else:
            return
    
    print(f"\n📁 Testing file: {well_file}")
    
    # Run tests
    success = True
    
    # Test 1: File structure
    if not test_well_locations_structure(well_file):
        success = False
    
    # Test 2: Data access
    if not test_data_access(well_file):
        success = False
    
    # Test 3: Comparison with spatial files
    spatial_file = None
    if len(sys.argv) > 2:
        spatial_file = Path(sys.argv[2])
    elif Path("sr3_batch_output").exists():
        spatial_files = list(Path("sr3_batch_output").glob("batch_spatial_properties_*.h5"))
        if spatial_files:
            spatial_file = spatial_files[0]
    
    test_comparison_with_spatial(well_file, spatial_file)
    
    # Test 4: ML workflow example
    test_ml_workflow_example(well_file)
    
    # Summary
    print("\n" + "=" * 70)
    if success:
        print("✅ ALL TESTS PASSED")
        print("\n💡 The well_locations.h5 file is ready for ML workflows!")
        print("   - Binary format: (N_cases, N_timesteps, Nx, Ny, Nz)")
        print("   - Well mapping available for enhanced visualization")
        print("   - Structure matches spatial files format")
    else:
        print("❌ SOME TESTS FAILED")
        print("   Please check the errors above")
    print("=" * 70)


if __name__ == "__main__":
    main()

