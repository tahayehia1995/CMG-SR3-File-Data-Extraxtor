"""
Streamlit Cloud entry point for SR3 Data Processor Web Application
This file is used by Streamlit Cloud to launch the application.
It imports and executes the main app from the package structure.
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path so we can import the package
# Use absolute path resolution for reliability in cloud environments
_this_file = Path(__file__).resolve()
_src_dir = _this_file.parent / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

# Import the main function and call it
# This will execute all the Streamlit code in the app
try:
    from cmg_sr3_files_data_processor.app import main
    # Execute the main function which contains all Streamlit UI code
    main()
except ImportError as e:
    # If import fails, show a helpful error
    import streamlit as st
    st.error(f"❌ Import Error: {str(e)}")
    st.code(f"Python path: {sys.path[:3]}")
    st.code(f"Looking for: cmg_sr3_files_data_processor.app")
    st.code(f"Source directory: {_src_dir}")
    st.code(f"Source exists: {_src_dir.exists()}")
    raise
except Exception as e:
    # For any other error, show it
    import streamlit as st
    import traceback
    st.error(f"❌ Error: {str(e)}")
    st.exception(e)
    raise

