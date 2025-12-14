"""
Streamlit Cloud entry point for SR3 Data Processor Web Application
This file directly contains the Streamlit code to ensure it executes properly.
"""

import sys
from pathlib import Path

# Add src directory to Python path
_this_file = Path(__file__).resolve()
_src_dir = _this_file.parent / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

# Import Streamlit and other dependencies
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import our modules
from cmg_sr3_files_data_processor.streamlit_extractor import (
    StreamlitSR3Extractor,
    inspect_h5_structure,
    inspect_sr3_structure,
)
from cmg_sr3_files_data_processor.streamlit_visualizer import StreamlitH5Visualizer

# Page configuration - MUST be at top level for Streamlit Cloud
st.set_page_config(
    page_title="SR3 Data Processor",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state - MUST be at top level
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'extractor' not in st.session_state:
    st.session_state.extractor = None
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = None
if 'output_folder' not in st.session_state:
    st.session_state.output_folder = None

# Custom CSS for wizard steps
st.markdown("""
<style>
    .wizard-step {
        display: inline-block;
        padding: 10px 20px;
        margin: 5px;
        border-radius: 5px;
        background-color: #f0f0f0;
        color: #666;
    }
    .wizard-step.active {
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    .wizard-step.completed {
        background-color: #2ca02c;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Import and call main from app.py
from cmg_sr3_files_data_processor.app import main

# Execute the main app
main()
