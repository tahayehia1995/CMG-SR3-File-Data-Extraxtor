"""
Streamlit Cloud entry point for SR3 Data Processor Web Application
This file ensures proper path setup and imports the app module.
"""

import sys
import os
from pathlib import Path

# Determine the repository root directory
# Strategy 1: From this file's location (streamlit_app.py is at repo root)
_repo_root = Path(__file__).resolve().parent
_src_dir = _repo_root / "src"

# Strategy 2: If that doesn't work, try from current working directory
if not _src_dir.exists():
    _cwd = Path.cwd()
    # Check if we're in repo root
    if (_cwd / "src" / "cmg_sr3_files_data_processor" / "app.py").exists():
        _src_dir = _cwd / "src"
    # Check if we're in src directory
    elif (_cwd / "cmg_sr3_files_data_processor" / "app.py").exists():
        _src_dir = _cwd
    else:
        _src_dir = _cwd / "src"

# Strategy 3: Try Streamlit Cloud mount paths
if not _src_dir.exists():
    _possible_paths = [
        Path("/mount/src/cmg-sr3-file-data-extraxtor/src"),
        Path("/mount/src/cmg-sr3-file-data-extraxtor"),
    ]
    for _path in _possible_paths:
        if _path.exists() and (_path / "cmg_sr3_files_data_processor" / "app.py").exists():
            _src_dir = _path if _path.name == "src" else _path / "src"
            break

# Add src directory to Python path
if _src_dir.exists():
    _src_str = str(_src_dir.resolve())
    if _src_str not in sys.path:
        sys.path.insert(0, _src_str)

# Now import and execute the app
# Import streamlit first so we can show errors
import streamlit as st

try:
    # Import the app module - this executes all module-level code
    from cmg_sr3_files_data_processor.app import main
    
    # Execute the main function
    main()
except Exception as e:
    # Show error in Streamlit UI
    st.error(f"❌ Error: {str(e)}")
    st.write("**Debug Information:**")
    st.code(f"Python version: {sys.version}")
    st.code(f"Python path (first 5): {sys.path[:5]}")
    st.code(f"Current working directory: {os.getcwd()}")
    st.code(f"Source directory: {_src_dir}")
    st.code(f"Source exists: {_src_dir.exists() if _src_dir else False}")
    st.code(f"__file__ location: {__file__}")
    
    import traceback
    st.write("**Full Traceback:**")
    st.code(traceback.format_exc())
    
    # Don't raise - let Streamlit show the error page
    raise
