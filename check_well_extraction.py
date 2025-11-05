#!/usr/bin/env python3
"""
Script to help diagnose well extraction issues
"""
import h5py
import numpy as np
from pathlib import Path

def check_well_file(file_path):
    """Check well_locations.h5 file and provide diagnostic info"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        print("\n💡 Tips:")
        print("   1. Make sure you ran extraction with 'Extract Well Locations' checkbox enabled")
        print("   2. Check the output folder path")
        print("   3. Check extraction logs for completion extraction warnings")
        return
    
    print(f"✅ Found: {file_path}")
    print("=" * 80)
    
    with h5py.File(file_path, 'r') as f:
        data = f['data'][...]
        total_wells = np.sum(data)
        
        print(f"\n📊 Summary:")
        print(f"   Shape: {data.shape} (N_cases, N_timesteps, Nx, Ny, Nz)")
        print(f"   Total well cells: {total_wells}")
        
        if total_wells == 0:
            print("\n⚠️ WARNING: File contains NO well locations!")
            print("\n🔍 Possible reasons:")
            print("   1. Completion data not found in SR3 files")
            print("   2. Completion data stored in different format/location")
            print("   3. Wells have no completions at selected timesteps")
            print("\n💡 Solutions:")
            print("   - Check SR3 file structure for completion/connection tables")
            print("   - Verify wells exist in TimeSeries/WELLS/Origins")
            print("   - Try extracting with different timesteps")
            print("   - Completion data might need to be extracted from input files")
        else:
            print(f"\n✅ File contains {total_wells} well cells")
            
            # Show distribution
            n_cases, n_timesteps = data.shape[0], data.shape[1]
            print(f"\n📋 Distribution:")
            for case_idx in range(n_cases):
                case_total = np.sum(data[case_idx, :, :, :, :])
                print(f"   Case {case_idx}: {case_total} well cells")
                for ts_idx in range(n_timesteps):
                    ts_total = np.sum(data[case_idx, ts_idx, :, :, :])
                    if ts_total > 0:
                        print(f"      Timestep {ts_idx}: {ts_total} well cells")
            
            # Show which layers have wells
            print(f"\n🗺️ Layers with wells:")
            for case_idx in range(min(n_cases, 2)):
                for ts_idx in range(min(n_timesteps, 2)):
                    layer_wells = np.sum(data[case_idx, ts_idx, :, :, :], axis=(0, 1))
                    layers_with_wells = np.where(layer_wells > 0)[0]
                    if len(layers_with_wells) > 0:
                        print(f"   Case {case_idx}, Timestep {ts_idx}: K-layers {layers_with_wells.tolist()}")
            
            print(f"\n💡 Visualization Tips:")
            print(f"   - Make sure you select a K-layer that contains wells")
            print(f"   - Enable 'Show Well Locations' checkbox in visualization")
            print(f"   - Check that Case and Timestep match well locations")

if __name__ == "__main__":
    # Check common locations (new name first, then legacy)
    test_paths = [
        Path("batch_spatial_properties_well_location.h5"),
        Path("sr3_batch_output/batch_spatial_properties_well_location.h5"),
        Path("sr3_output/batch_spatial_properties_well_location.h5"),
        Path("well_locations.h5"),  # Legacy filename
        Path("sr3_batch_output/well_locations.h5"),  # Legacy
        Path("sr3_output/well_locations.h5"),  # Legacy
    ]
    
    found = False
    for test_path in test_paths:
        if test_path.exists():
            check_well_file(test_path)
            found = True
            break
    
    if not found:
        print("❌ Well locations file not found")
        print("\n   Looking for: batch_spatial_properties_well_location.h5 (or legacy well_locations.h5)")
        print("   Please run extraction with 'Extract Well Locations' enabled")

