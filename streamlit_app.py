"""
Streamlit Cloud entry point for SR3 Data Processor Web Application
This file imports all code from app.py so Streamlit can execute it at module level.
"""

import sys
from pathlib import Path

# Add src directory to Python path
_this_file = Path(__file__).resolve()
_src_dir = _this_file.parent / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

# Import everything from app.py - this executes all module-level Streamlit code
# including st.set_page_config() and session state initialization
# Then we import main and call it to execute the UI logic
from cmg_sr3_files_data_processor.app import (
    main,
    step1_file_upload,
    step2_configure,
    step3_extract,
    step4_visualize,
    render_wizard_steps,
)

# Execute main() which contains the Streamlit UI code
# This ensures Streamlit sees all the st.* calls
main()
