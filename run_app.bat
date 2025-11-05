@echo off
REM SR3 Data Processor Web Application - Launch Script
REM Double-click this file to run the web application
REM This script ensures all dependencies are installed before launching

setlocal enabledelayedexpansion

echo ============================================================
echo   SR3 Data Processor Web Application
echo   Portable Launcher - Auto-Install Dependencies
echo ============================================================
echo.

REM Change to the script's directory (important for portability)
cd /d "%~dp0"

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not found in PATH!
    echo.
    echo Please ensure Python 3.7+ is installed and added to your system PATH.
    echo Download Python from: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

REM Display Python version
echo [INFO] Python detected:
python --version
python -c "import sys; exit(0 if sys.version_info >= (3, 7) else 1)" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3.7 or higher is required!
    echo.
    pause
    exit /b 1
)
echo.

REM Check if required files exist
set MISSING_FILES=0
if not exist "app.py" (
    echo [ERROR] app.py not found!
    set MISSING_FILES=1
)
if not exist "run_app.py" (
    echo [ERROR] run_app.py not found!
    set MISSING_FILES=1
)
if not exist "requirements_web.txt" (
    echo [ERROR] requirements_web.txt not found!
    set MISSING_FILES=1
)
if not exist "streamlit_extractor.py" (
    echo [ERROR] streamlit_extractor.py not found!
    set MISSING_FILES=1
)
if not exist "streamlit_visualizer.py" (
    echo [ERROR] streamlit_visualizer.py not found!
    set MISSING_FILES=1
)
if !MISSING_FILES! equ 1 (
    echo.
    echo Please ensure you're running this script from the correct directory.
    echo All project files must be in the same folder.
    echo.
    pause
    exit /b 1
)

echo [INFO] Checking dependencies...
echo.

REM Upgrade pip first to ensure reliable installation
echo [INFO] Upgrading pip to latest version...
python -m pip install --upgrade pip --quiet 2>nul
if errorlevel 1 (
    echo [WARNING] Failed to upgrade pip, continuing anyway...
)

REM Check critical dependencies
set INSTALL_NEEDED=0

REM Test each critical package
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo [CHECK] streamlit - MISSING
    set INSTALL_NEEDED=1
) else (
    echo [CHECK] streamlit - OK
)

python -c "import numpy" >nul 2>&1
if errorlevel 1 (
    echo [CHECK] numpy - MISSING
    set INSTALL_NEEDED=1
) else (
    echo [CHECK] numpy - OK
)

python -c "import pandas" >nul 2>&1
if errorlevel 1 (
    echo [CHECK] pandas - MISSING
    set INSTALL_NEEDED=1
) else (
    echo [CHECK] pandas - OK
)

python -c "import h5py" >nul 2>&1
if errorlevel 1 (
    echo [CHECK] h5py - MISSING
    set INSTALL_NEEDED=1
) else (
    echo [CHECK] h5py - OK
)

python -c "import matplotlib" >nul 2>&1
if errorlevel 1 (
    echo [CHECK] matplotlib - MISSING
    set INSTALL_NEEDED=1
) else (
    echo [CHECK] matplotlib - OK
)

echo.

REM Install dependencies if needed
if !INSTALL_NEEDED! equ 1 (
    echo [INFO] Installing missing dependencies from requirements_web.txt...
    echo This may take a few minutes on first run...
    echo.
    
    python -m pip install --upgrade pip setuptools wheel --quiet
    python -m pip install -r requirements_web.txt
    
    if errorlevel 1 (
        echo.
        echo [ERROR] Failed to install dependencies!
        echo.
        echo Troubleshooting steps:
        echo 1. Ensure you have internet connection
        echo 2. Try running as Administrator
        echo 3. Install manually: pip install -r requirements_web.txt
        echo 4. Check Python path configuration
        echo.
        pause
        exit /b 1
    )
    
    echo.
    echo [INFO] Dependencies installed successfully!
    echo.
) else (
    echo [INFO] All dependencies are already installed!
    echo.
)

REM Final verification - test critical imports
echo [INFO] Verifying installation...
python -c "import streamlit; import numpy; import pandas; import h5py; import matplotlib; print('All dependencies verified!')" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Dependency verification failed!
    echo Some packages may not have installed correctly.
    echo Please check the error messages above.
    echo.
    pause
    exit /b 1
)

echo [INFO] All dependencies verified successfully!
echo.

REM Test that app.py can be imported (syntax check)
python -c "import sys; sys.path.insert(0, '.'); import app" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] app.py may have syntax errors or missing dependencies!
    echo The app will still attempt to run, but errors may occur.
    echo.
) else (
    echo [INFO] Application files verified!
    echo.
)

echo ============================================================
echo   Starting Web Application...
echo ============================================================
echo.
echo The browser will open automatically in a few seconds.
echo Access the app at: http://localhost:8501
echo Press Ctrl+C to stop the server when done.
echo.
echo ============================================================
echo.

REM Run the application
python run_app.py

REM If the script exits, pause to show any error messages
if errorlevel 1 (
    echo.
    echo ============================================================
    echo [ERROR] Application exited with an error!
    echo ============================================================
    echo.
    echo If you see import errors, try:
    echo    pip install -r requirements_web.txt
    echo.
    echo If issues persist, check:
    echo    1. Python version (need 3.7+)
    echo    2. All project files are present
    echo    3. No conflicting Python installations
    echo.
    pause
)

