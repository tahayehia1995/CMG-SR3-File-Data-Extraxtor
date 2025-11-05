# SR3 Data Processor Web Application

A Streamlit-based web application for batch extracting and visualizing CMG SR3 reservoir simulation data.

## Features

- **Wizard-Style Interface**: Step-by-step workflow for easy data processing
- **Drag-and-Drop File Upload**: Easy file selection with drag-and-drop support
- **Batch Processing**: Process 1000+ SR3 files efficiently with parallel processing
- **Integrated Visualization**: Visualize extracted data directly in the web app
- **Configurable Server**: Run on localhost only or enable network access

## Installation

1. Install dependencies:
```bash
pip install -r requirements_web.txt
```

2. Ensure all required files are present:
   - `app.py` - Main Streamlit application
   - `run_app.py` - Execution script
   - `streamlit_extractor.py` - Extraction wrapper
   - `streamlit_visualizer.py` - Visualization wrapper
   - `interactive_sr3_extractor.py` - Core extraction logic
   - `interactive_h5_visualizer.py` - Core visualization logic

## Usage

### Quick Start

**Option 1: Double-Click Executable (Windows)**
- Double-click `run_app.bat` to launch the application
- The script will automatically check for Python and dependencies
- The browser will open automatically at `http://localhost:8501`

**Option 2: Command Line**
Run the application with default settings (localhost only):
```bash
python run_app.py
```

The application will start at `http://localhost:8501`

### Advanced Usage

**Run on network (accessible from other machines):**
```bash
python run_app.py --host 0.0.0.0
```

**Run on custom port:**
```bash
python run_app.py --port 8502
```

**Run on network with custom port:**
```bash
python run_app.py --host 0.0.0.0 --port 8502
```

**Run in headless mode (no browser auto-open):**
```bash
python run_app.py --server-headless
```

**Enable internet tunneling (public URL for internet access):**
```bash
python run_app.py --tunnel
```

When using `--tunnel`, a public URL will be generated that allows access from anywhere on the internet without router configuration. The public URL will be displayed in the console.

### Command-Line Options

- `--host`: Host to bind to (default: localhost)
  - `localhost`: Local access only
  - `0.0.0.0`: Network access (accessible from other machines on your network)
  
- `--port`: Port number (default: 8501)

- `--server-headless`: Run without auto-opening browser

- `--tunnel`: Enable internet tunneling (creates public URL via ngrok)

- `--tunnel-service`: Tunneling service to use (default: ngrok)

## Application Workflow

### Step 1: Upload Files
- Drag and drop SR3 files or click to browse
- Files are validated and processed
- File information is displayed

### Step 2: Configure Extraction
- Select data types (Spatial Properties / Time Series / Both)
- Choose properties/variables to extract
- Select date filtering mode:
  - **Yearly**: January 1st dates only
  - **Monthly**: End-of-month dates
  - **Daily**: Regular intervals
  - **Custom**: Manual date selection
- Configure output folder path
- Set inactive cell options

### Step 3: Extract Data
- Review configuration
- Start extraction process
- Monitor progress with real-time progress bars
- View extraction summary upon completion

### Step 4: Visualize Data
- **Spatial Properties**: 
  - Select H5 file
  - Adjust case, K-layer, and timestep sliders
  - Toggle inactive cell masking
  - View 2D I×J plots
  
- **Time Series**:
  - Select H5 file
  - Choose case and wells
  - View multi-well temporal plots

## File Structure

```
├── app.py                      # Main Streamlit application
├── run_app.py                  # Execution script (Python)
├── run_app.bat                 # Windows executable (double-click to run)
├── streamlit_extractor.py     # Streamlit extraction wrapper
├── streamlit_visualizer.py    # Streamlit visualization wrapper
├── interactive_sr3_extractor.py  # Core extraction logic
├── interactive_h5_visualizer.py  # Core visualization logic
├── requirements_web.txt       # Web app dependencies
├── .streamlit/
│   └── config.toml            # Streamlit configuration (local-only defaults)
└── README_WEB_APP.md          # This file
```

## Output Files

Extracted files are saved in the specified output folder:

```
{output_folder}/
├── inactive_cell_locations.h5          # 4D: (N_cases, Nx, Ny, Nz)
├── batch_spatial_properties_POROS.h5   # 5D: (N_cases, N_timesteps, Nx, Ny, Nz)
├── batch_spatial_properties_PRES.h5    # 5D: (N_cases, N_timesteps, Nx, Ny, Nz)
├── batch_timeseries_data_BHP.h5       # 3D: (N_cases, N_timesteps, N_wells)
└── ...
```

## Performance

- **Parallel Processing**: Uses all available CPU cores for maximum speed
- **Adaptive Compression**: Automatically optimizes compression based on file size
- **Chunked H5 Writing**: Efficient I/O operations for large datasets
- **Optimized Lookups**: Fast O(1) dictionary lookups instead of O(n) searches

## Internet Access (Tunneling)

To make your application accessible from anywhere on the internet without router configuration:

1. **Install ngrok** (choose one method):
   
   **Option 1 (Recommended):** Install via pip:
   ```bash
   pip install pyngrok
   ```
   
   **Option 2:** Download ngrok from [https://ngrok.com/download](https://ngrok.com/download) and add to PATH

2. **Run with tunneling enabled:**
   ```bash
   python run_app.py --tunnel
   ```

3. **Share the public URL** displayed in the console. This URL will be accessible from anywhere on the internet.

**Note:** For free ngrok accounts, the public URL changes each time you restart the tunnel. For a permanent URL, consider upgrading your ngrok account or using other tunneling services.

## Security Notes

⚠️ **Important Security Considerations:**

- **Localhost mode (`--host localhost`):** Only accessible from your computer (default)
  
- **Network mode (`--host 0.0.0.0`):** Accessible from other machines on your local network. Ensure:
  - Your firewall is properly configured
  - You're on a trusted network
  - Consider using authentication if processing sensitive data

- **Internet tunneling (`--tunnel`):** Creates a public URL accessible from anywhere on the internet. ⚠️ **CRITICAL WARNINGS:**
  - **No authentication is enabled** - anyone with the URL can access your application
  - Use **only for testing/demo purposes**
  - **Stop the server immediately** when done testing
  - Do not process sensitive or confidential data when using tunneling
  - Consider adding authentication for production use

## Troubleshooting

**Port already in use:**
```bash
python run_app.py --port 8502
```

**Files not uploading:**
- Ensure files have `.sr3` extension
- Check file size limits (Streamlit default: 200MB per file)
- For larger files, increase limit in `.streamlit/config.toml`

**Extraction fails:**
- Check that all uploaded files are valid SR3 files
- Ensure files have common dates across all cases
- Verify output folder path is writable

**Visualization not loading:**
- Ensure extraction completed successfully
- Check that H5 files exist in output folder
- Verify file permissions

**Tunneling not working:**
- Ensure ngrok is installed: `pip install pyngrok` or download from ngrok.com
- Check that ngrok is in your PATH (if using system installation)
- Verify firewall allows ngrok connections
- For persistent issues, check ngrok web interface at http://127.0.0.1:4040

## Support

For issues or questions, refer to the main project documentation or check the core extraction/visualization modules.

