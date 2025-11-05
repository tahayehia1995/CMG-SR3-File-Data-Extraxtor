# Well Locations Visualization Troubleshooting Guide

## Issue: No wells showing in 2D visualizer

### Common Causes and Solutions

#### 1. **Empty well_locations.h5 file**
If the extraction didn't find completion data in SR3 files, the file will be created but empty.

**Check:**
```python
python check_well_extraction.py
```

**Solution:**
- Completion data may not be stored in SR3 output files
- Completion data is typically in simulation input files (.dat, .cmsd)
- You may need to extract completions from input files and map them to timesteps

#### 2. **Wrong K-Layer selected**
Wells only appear in specific K-layers where completions exist.

**Check:**
- Use the diagnostic script to see which layers have wells
- Try different K-layer values in the visualizer

#### 3. **Toggle not enabled**
Make sure "Show Well Locations" checkbox is enabled in the visualizer.

#### 4. **Case/Timestep mismatch**
Wells are stored per case and timestep. Make sure you're viewing the correct case and timestep.

### Debugging Steps

1. **Check if file exists:**
   ```python
   from pathlib import Path
   well_file = Path("well_locations.h5")  # or your output folder path
   print(f"Exists: {well_file.exists()}")
   ```

2. **Inspect file contents:**
   ```python
   python check_well_extraction.py
   ```

3. **Check extraction logs:**
   - Look for warnings about "No completions found"
   - Check if well names were found in SR3 files

4. **Test with sample data:**
   - The test file `well_locations.h5` contains sample wells at:
     - Case 0: K-layers 10 and 15
     - Case 1: K-layers 12 and 18
   - Use this to verify visualization works

### Expected Behavior

- **With wells:** Yellow circular markers with "W" labels appear on the plot
- **Without wells:** Debug message shows why no wells are displayed
- **Empty file:** Warning message explains that no completions were found

### Next Steps

If completions aren't found in SR3 files, you may need to:
1. Extract completion data from simulation input files
2. Map completions to timesteps based on simulation schedule
3. Create well_locations.h5 manually with completion IJK coordinates

