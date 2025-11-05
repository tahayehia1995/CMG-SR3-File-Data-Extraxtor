#!/usr/bin/env python3
"""
Inspect well_locations.h5 file contents and structure
"""

import h5py
import numpy as np
from pathlib import Path
import sys

def inspect_well_locations_file(file_path):
    """Comprehensive inspection of well_locations.h5 file"""
    print("=" * 80)
    print("WELL LOCATIONS FILE INSPECTION")
    print("=" * 80)
    print(f"\n📁 File: {file_path}")
    
    if not Path(file_path).exists():
        print(f"❌ File not found!")
        return False
    
    file_size = Path(file_path).stat().st_size / (1024 * 1024)  # MB
    print(f"📊 File size: {file_size:.2f} MB")
    
    try:
        with h5py.File(file_path, 'r') as f:
            print("\n" + "=" * 80)
            print("TOP-LEVEL STRUCTURE")
            print("=" * 80)
            print(f"\n📦 Datasets/Groups: {list(f.keys())}")
            
            # File-level attributes
            print("\n📄 File-Level Attributes:")
            for attr_name in f.attrs.keys():
                attr_value = f.attrs[attr_name]
                if isinstance(attr_value, bytes):
                    attr_value = attr_value.decode()
                print(f"   • {attr_name}: {attr_value}")
            
            # Main data array
            print("\n" + "=" * 80)
            print("MAIN DATA ARRAY")
            print("=" * 80)
            if 'data' in f:
                data = f['data']
                print(f"\n✅ Dataset: 'data'")
                print(f"   Shape: {data.shape}")
                print(f"   Dtype: {data.dtype}")
                print(f"   Compression: {data.compression}")
                print(f"   Chunks: {data.chunks}")
                print(f"   Size: {data.size:,} elements")
                print(f"   Memory size: {data.nbytes / (1024*1024):.2f} MB")
                
                # Data statistics
                data_array = data[...]
                print(f"\n   Statistics:")
                print(f"   • Min value: {np.min(data_array)}")
                print(f"   • Max value: {np.max(data_array)}")
                print(f"   • Sum (total well cells): {np.sum(data_array):,}")
                print(f"   • Unique values: {np.unique(data_array)}")
                
                # Check shape interpretation
                if len(data.shape) == 5:
                    n_cases, n_timesteps, nx, ny, nz = data.shape
                    print(f"\n   Shape interpretation:")
                    print(f"   • N_cases: {n_cases}")
                    print(f"   • N_timesteps: {n_timesteps}")
                    print(f"   • Grid dimensions: {nx} × {ny} × {nz}")
                    print(f"   • Total cells per timestep: {nx * ny * nz:,}")
            else:
                print("\n❌ 'data' dataset not found!")
            
            # Well mapping array
            print("\n" + "=" * 80)
            print("WELL MAPPING ARRAY")
            print("=" * 80)
            if 'well_mapping' in f:
                mapping = f['well_mapping']
                print(f"\n✅ Dataset: 'well_mapping'")
                print(f"   Shape: {mapping.shape}")
                print(f"   Dtype: {mapping.dtype}")
                print(f"   Compression: {mapping.compression}")
                print(f"   Chunks: {mapping.chunks}")
                
                mapping_array = mapping[...]
                unique_values = np.unique(mapping_array)
                print(f"\n   Statistics:")
                print(f"   • Min value: {np.min(mapping_array)}")
                print(f"   • Max value: {np.max(mapping_array)}")
                print(f"   • Unique values count: {len(unique_values)}")
                print(f"   • Values range: {unique_values[:10]}..." if len(unique_values) > 10 else f"   • Values: {unique_values}")
                print(f"   • Cells with wells (mapping >= 0): {np.sum(mapping_array >= 0):,}")
                print(f"   • Cells without wells (mapping == -1): {np.sum(mapping_array == -1):,}")
            else:
                print("\n⚠️  'well_mapping' dataset not found (legacy format)")
                print("   This file was created before enhanced visualization was added")
                print("   Re-extract to get well_mapping for enhanced display")
            
            # Metadata
            print("\n" + "=" * 80)
            print("METADATA GROUP")
            print("=" * 80)
            if 'metadata' in f:
                meta = f['metadata']
                
                print("\n📋 Attributes:")
                for attr_name in sorted(meta.attrs.keys()):
                    attr_value = meta.attrs[attr_name]
                    if isinstance(attr_value, np.ndarray):
                        print(f"   • {attr_name}: {attr_value} (shape: {attr_value.shape})")
                    else:
                        print(f"   • {attr_name}: {attr_value}")
                
                print("\n📦 Datasets:")
                for ds_name in sorted(meta.keys()):
                    ds = meta[ds_name]
                    print(f"\n   ✅ {ds_name}:")
                    print(f"      Shape: {ds.shape}")
                    print(f"      Dtype: {ds.dtype}")
                    
                    # Show sample data for string arrays
                    if ds.dtype.kind == 'S' or ds.dtype.kind == 'U':  # String type
                        sample_size = min(5, len(ds))
                        sample_data = [ds[i].decode() if isinstance(ds[i], bytes) else str(ds[i]) 
                                      for i in range(sample_size)]
                        print(f"      Sample values: {sample_data}")
                        if len(ds) > sample_size:
                            print(f"      ... and {len(ds) - sample_size} more")
                    elif ds.dtype.kind in ['i', 'u', 'f']:  # Numeric type
                        if ds.size <= 20:
                            print(f"      Values: {ds[...]}")
                        else:
                            sample = ds[:min(10, len(ds))]
                            print(f"      Sample values: {sample}")
                            print(f"      Min: {np.min(ds[...])}, Max: {np.max(ds[...])}")
                            print(f"      ... and {ds.size - len(sample)} more elements")
                
                # Check required fields for ML workflow compatibility
                print("\n" + "=" * 80)
                print("ML WORKFLOW COMPATIBILITY CHECK")
                print("=" * 80)
                
                required_attrs = ['n_cases', 'n_timesteps', 'nx', 'ny', 'nz', 'grid_dimensions']
                required_datasets = ['timesteps', 'target_days', 'sr3_files', 'well_names', 'well_types']
                
                print("\n✅ Required Attributes:")
                all_attrs_ok = True
                for attr in required_attrs:
                    exists = attr in meta.attrs
                    status = "✅" if exists else "❌"
                    value = meta.attrs[attr] if exists else "MISSING"
                    print(f"   {status} {attr}: {value}")
                    if not exists:
                        all_attrs_ok = False
                
                print("\n✅ Required Datasets:")
                all_datasets_ok = True
                for ds_name in required_datasets:
                    exists = ds_name in meta
                    status = "✅" if exists else "❌"
                    print(f"   {status} {ds_name}")
                    if not exists:
                        all_datasets_ok = False
                
                # Check optional but recommended
                print("\n⚠️  Optional (Recommended):")
                if 'active_cell_mapping' in meta:
                    print(f"   ✅ active_cell_mapping: {meta['active_cell_mapping'].shape}")
                else:
                    print(f"   ⚠️  active_cell_mapping: Not found")
                
                if 'total_cells' in meta.attrs:
                    print(f"   ✅ total_cells: {meta.attrs['total_cells']}")
                else:
                    print(f"   ⚠️  total_cells: Not found")
                
                if 'active_cells' in meta.attrs:
                    print(f"   ✅ active_cells: {meta.attrs['active_cells']}")
                else:
                    print(f"   ⚠️  active_cells: Not found")
                
                # Summary
                print("\n" + "=" * 80)
                print("SUMMARY")
                print("=" * 80)
                
                if all_attrs_ok and all_datasets_ok:
                    print("\n✅ File structure is compatible with ML workflows!")
                else:
                    print("\n⚠️  File structure is missing some required fields")
                    print("   Consider re-extracting with the latest version")
                
                if 'well_mapping' in f:
                    print("✅ Enhanced visualization available (well_mapping present)")
                else:
                    print("⚠️  Enhanced visualization not available (well_mapping missing)")
                    print("   Re-extract to enable well names/types display")
                
            else:
                print("\n❌ 'metadata' group not found!")
            
            # Example access patterns
            print("\n" + "=" * 80)
            print("EXAMPLE ACCESS PATTERNS")
            print("=" * 80)
            
            if 'data' in f and 'metadata' in f:
                data = f['data']
                meta = f['metadata']
                
                if len(data.shape) == 5:
                    n_cases, n_timesteps, nx, ny, nz = data.shape
                    
                    print("\n📝 Python code examples:")
                    print("-" * 80)
                    print(f"""
# Load file
import h5py
import numpy as np

with h5py.File('{Path(file_path).name}', 'r') as f:
    # Main data
    well_locations = f['data'][...]  # Shape: ({n_cases}, {n_timesteps}, {nx}, {ny}, {nz})
    
    # Well mapping (if available)
    well_mapping = f['well_mapping'][...] if 'well_mapping' in f else None
    
    # Metadata
    meta = f['metadata']
    well_names = [name.decode() for name in meta['well_names'][...]]
    well_types = [wt.decode() for wt in meta['well_types'][...]]
    
    # Access single layer
    case_idx = 0
    timestep_idx = 0
    layer_idx = 0
    layer_wells = well_locations[case_idx, timestep_idx, :, :, layer_idx]
    # Shape: ({nx}, {ny})
    
    # Count wells in layer
    well_count = np.sum(layer_wells)
    print(f"Wells in layer: {{well_count}}")
""")
            
            return True
            
    except Exception as e:
        print(f"\n❌ Error reading file: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Check for file path argument or use default locations
    if len(sys.argv) > 1:
        file_path = Path(sys.argv[1])
    else:
        # Try common locations
        test_paths = [
            Path("batch_spatial_properties_well_location.h5"),
            Path("well_locations.h5"),  # Legacy
            Path("sr3_batch_output/batch_spatial_properties_well_location.h5"),
            Path("sr3_batch_output/well_locations.h5"),  # Legacy
        ]
        
        file_path = None
        for path in test_paths:
            if path.exists():
                file_path = path
                break
        
        if not file_path:
            print("❌ Well locations file not found!")
            print("\nPlease provide the file path as an argument:")
            print("  python inspect_well_locations.py <path_to_file>")
            sys.exit(1)
    
    inspect_well_locations_file(file_path)

