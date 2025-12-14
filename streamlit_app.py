"""
Streamlit Cloud entry point for SR3 Data Processor Web Application
This file is used by Streamlit Cloud to launch the application.
It imports the main app from the package structure.
"""

import sys
from pathlib import Path

# Add src directory to Python path so we can import the package
src_dir = Path(__file__).parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Import and run the main app
# Streamlit Cloud will execute this file, so we call main() directly
from cmg_sr3_files_data_processor.app import main

main()

