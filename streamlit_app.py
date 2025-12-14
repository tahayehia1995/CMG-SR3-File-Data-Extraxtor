"""
Streamlit Cloud entry point for SR3 Data Processor Web Application
Minimal entry point that sets up the path and imports the main app.
"""

import sys
from pathlib import Path

# Add src directory to Python path so we can import the package
_this_file = Path(__file__).resolve()
_src_dir = _this_file.parent / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

# Import the app module - this will execute all module-level Streamlit code
# including st.set_page_config(), session state initialization, and CSS
# Then we call main() to execute the UI logic
from cmg_sr3_files_data_processor.app import main

# Execute the main function which contains all Streamlit UI code
main()
