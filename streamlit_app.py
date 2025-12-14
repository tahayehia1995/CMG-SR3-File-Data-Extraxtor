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

# Import the app module - this will execute all module-level code including st.set_page_config()
# Then we need to call main() to run the actual app logic
from cmg_sr3_files_data_processor import app

# Call the main function to start the Streamlit app
# This ensures all the Streamlit UI code in main() is executed
app.main()

