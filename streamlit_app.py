"""
Streamlit Cloud entry point for SR3 Data Processor Web Application
This file is used by Streamlit Cloud to launch the application.
It imports the main app from the package structure.
"""

import sys
import traceback
from pathlib import Path

# Add src directory to Python path so we can import the package
src_dir = Path(__file__).parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

try:
    # Import and call the main app function
    # Streamlit Cloud executes this file, so we need to call main() explicitly
    from cmg_sr3_files_data_processor.app import main
    
    # Call main() to start the Streamlit app
    main()
except Exception as e:
    # If there's an error, try to show it in Streamlit
    try:
        import streamlit as st
        st.error("❌ Error starting application")
        st.exception(e)
        st.code(traceback.format_exc())
    except:
        # If we can't even import streamlit, print to stderr
        print(f"Fatal error: {e}", file=sys.stderr)
        traceback.print_exc()
        raise

