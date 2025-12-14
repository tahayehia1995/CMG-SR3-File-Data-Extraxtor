"""
Streamlit Cloud entry point for SR3 Data Processor Web Application
This file directly imports and executes the app code.
"""

import sys
from pathlib import Path

# Add src directory to Python path
_this_file = Path(__file__).resolve()
_src_dir = _this_file.parent / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

# Directly import and execute - this will run all Streamlit code
# Import the module (this executes module-level code like st.set_page_config)
import cmg_sr3_files_data_processor.app as app_module

# Call main() to execute the Streamlit UI
app_module.main()
