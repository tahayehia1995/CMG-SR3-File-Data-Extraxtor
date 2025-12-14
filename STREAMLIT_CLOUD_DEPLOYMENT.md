# Streamlit Cloud Deployment Guide

## Current Setup

**Main Entry Point:** `streamlit_app.py` (root level)
- Sets up Python path to include `src/` directory
- Imports and calls `main()` from `src/cmg_sr3_files_data_processor/app.py`

**Main App File:** `src/cmg_sr3_files_data_processor/app.py`
- Contains all Streamlit code at module level
- Has `st.set_page_config()` called before any other Streamlit code
- Initializes session state at module level
- Contains `main()` function with UI logic

## Streamlit Cloud Configuration

**Current Settings:**
- Repository: `cmg-sr3-file-data-extraxtor`
- Branch: `main`
- Main file path: `streamlit_app.py`

## Alternative Configuration (If Current Setup Fails)

If `streamlit_app.py` wrapper approach doesn't work, you can configure Streamlit Cloud to use the app file directly:

1. Go to Streamlit Cloud dashboard
2. Edit your app settings
3. Change **Main file path** to: `src/cmg_sr3_files_data_processor/app.py`
4. Save and redeploy

This bypasses the wrapper and runs the app file directly.

## Dependencies

All dependencies are listed in `requirements.txt`:
- streamlit>=1.28.0
- numpy>=1.21.0
- pandas>=1.3.0
- h5py>=3.7.0
- matplotlib>=3.5.0
- ipython>=7.0.0
- ipywidgets>=7.6.0
- tqdm>=4.62.0
- pyngrok>=5.0.0

## Troubleshooting

**If app fails to start:**
1. Check logs for import errors
2. Verify all dependencies are installing correctly
3. Try the alternative configuration (direct app.py path)
4. Ensure repository is public (or you have paid Streamlit Cloud plan)

**Common Issues:**
- Import errors: Check Python path setup
- Missing dependencies: Verify requirements.txt is complete
- Silent failures: Check if st.set_page_config() is called correctly

