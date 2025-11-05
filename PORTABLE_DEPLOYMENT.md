# Portable Deployment Guide

This guide ensures your project can be easily transferred to another PC and run without issues.

## Quick Start

**To run on any Windows PC:**
1. Copy the entire project folder to the target PC
2. Double-click `run_app.bat`
3. The script will automatically install any missing dependencies
4. The web app will open in your browser

**Requirements:**
- Windows OS
- Python 3.7 or higher installed
- Internet connection (for first-time dependency installation)

## Files That Must Be Included

When transferring this project to another PC, include ALL of the following files:

### Essential Application Files
- ✅ `app.py` - Main Streamlit application
- ✅ `run_app.py` - Python execution script
- ✅ `run_app.bat` - **Double-click this to launch!**
- ✅ `requirements_web.txt` - Dependency list

### Core Modules (Required)
- ✅ `streamlit_extractor.py` - Streamlit wrapper for extraction
- ✅ `streamlit_visualizer.py` - Streamlit wrapper for visualization
- ✅ `interactive_sr3_extractor.py` - Core extraction logic
- ✅ `interactive_h5_visualizer.py` - Core visualization logic

### Documentation (Optional but Recommended)
- 📄 `README_WEB_APP.md` - User documentation
- 📄 `PORTABLE_DEPLOYMENT.md` - This file
- 📄 `dependencies_and_libraries.txt` - Dependency reference

### Test Files (Optional)
- 🧪 `test_app.py` - Application testing script

### Data Folders (Optional - Include if needed)
- 📁 `SACROC-Cmost_Test.cmsd_3_3_wells/` - Sample SR3 files (if included)

## What Happens When You Double-Click `run_app.bat`

1. **Python Check**: Verifies Python 3.7+ is installed
2. **File Check**: Verifies all essential files are present
3. **Dependency Check**: Checks if required packages are installed
4. **Auto-Install**: Installs any missing dependencies automatically
5. **Verification**: Tests that all imports work correctly
6. **Launch**: Starts the web application

## Dependencies Installed Automatically

The batch script installs these packages from `requirements_web.txt`:
- streamlit (web framework)
- numpy (numerical computing)
- pandas (data analysis)
- h5py (HDF5 file handling)
- matplotlib (visualization)
- ipython (for core modules)
- ipywidgets (for core modules)
- tqdm (progress bars)
- pyngrok (optional - for internet tunneling)

## Troubleshooting

### Python Not Found
**Error**: `Python is not found in PATH!`
**Solution**: 
- Install Python from https://www.python.org/downloads/
- During installation, check "Add Python to PATH"
- Restart your computer after installation

### Port Already in Use
**Error**: Port 8501 is already in use
**Solution**: 
- Close other Streamlit apps
- Or edit `run_app.bat` to use a different port
- Or use: `python run_app.py --port 8502`

### Installation Failures
**Error**: Failed to install dependencies
**Solutions**:
1. Ensure you have internet connection
2. Run as Administrator (right-click `run_app.bat` → Run as Administrator)
3. Install manually: Open Command Prompt in project folder and run:
   ```cmd
   pip install -r requirements_web.txt
   ```

### Import Errors After Installation
**Error**: ModuleNotFoundError
**Solutions**:
1. Verify Python version: `python --version` (need 3.7+)
2. Reinstall dependencies: `pip install -r requirements_web.txt --force-reinstall`
3. Check for multiple Python installations

### Path Issues (Files Not Found)
**Error**: `app.py not found!`
**Solution**: 
- Ensure you're running `run_app.bat` from the project root folder
- Don't move `run_app.bat` out of the project folder
- All files must be in the same directory

## Minimum Project Structure

```
project_folder/
├── app.py                          [REQUIRED]
├── run_app.py                      [REQUIRED]
├── run_app.bat                     [REQUIRED - Launch this!]
├── requirements_web.txt            [REQUIRED]
├── streamlit_extractor.py          [REQUIRED]
├── streamlit_visualizer.py         [REQUIRED]
├── interactive_sr3_extractor.py   [REQUIRED]
├── interactive_h5_visualizer.py   [REQUIRED]
└── (other optional files...)
```

## Testing Portability

Before sharing:
1. Test on a clean system (or virtual machine)
2. Copy project to a new folder
3. Double-click `run_app.bat`
4. Verify app opens successfully

## Notes for Advanced Users

- The batch script uses `setlocal enabledelayedexpansion` for Windows batch variable handling
- Dependencies are checked individually before bulk installation
- pip is automatically upgraded before installing packages
- The script changes to its own directory (`cd /d "%~dp0"`) for portability
- All imports are verified before launching the app

## Cross-Platform Notes

**Windows**: Use `run_app.bat` (double-click or run from command prompt)

**Linux/Mac**: Create a shell script equivalent:
```bash
#!/bin/bash
cd "$(dirname "$0")"
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements_web.txt
python3 run_app.py
```

Save as `run_app.sh` and make executable: `chmod +x run_app.sh`

