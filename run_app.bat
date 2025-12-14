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
if errorlevel 1 (
    echo [ERROR] Failed to change to script directory!
    echo Script location: "%~dp0"
    pause
    exit /b 1
)

REM Ensure src layout is importable
set "PYTHONPATH=%~dp0src"

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

REM ---------------------------------------------------------------------------
REM Use a short-path virtual environment for installs.
REM This avoids Windows MAX_PATH issues (common with Microsoft Store Python /
REM long OneDrive paths) when installing packages like jedi/typeshed.
REM ---------------------------------------------------------------------------

set "VENV_DIR=%SystemDrive%\sr3dp_venv"

REM Fallback if we can't create/access C:\ (rare)
if not exist "%VENV_DIR%" (
    mkdir "%VENV_DIR%" >nul 2>&1
    if errorlevel 1 (
        set "VENV_DIR=%LOCALAPPDATA%\sr3dp_venv"
    ) else (
        rmdir "%VENV_DIR%" >nul 2>&1
    )
)

set "VENV_PY=%VENV_DIR%\Scripts\python.exe"

if not exist "%VENV_PY%" (
    echo [INFO] Creating virtual environment at: "%VENV_DIR%"
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment!
        echo.
        echo Troubleshooting steps:
        echo 1. Ensure Python is installed correctly
        echo 2. Try running this script as Administrator
        echo 3. Ensure you have write permissions to: "%VENV_DIR%"
        echo.
        pause
        exit /b 1
    )
    echo.
)

REM Verify venv python works
"%VENV_PY%" --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Virtual environment Python is not runnable:
    echo "%VENV_PY%"
    echo.
    pause
    exit /b 1
)

REM Check if required files exist
set MISSING_FILES=0
if not exist "src\cmg_sr3_files_data_processor\app.py" (
    echo [ERROR] src\cmg_sr3_files_data_processor\app.py not found!
    set MISSING_FILES=1
)
if not exist "src\cmg_sr3_files_data_processor\run_app.py" (
    echo [ERROR] src\cmg_sr3_files_data_processor\run_app.py not found!
    set MISSING_FILES=1
)
if not exist "dependencies.txt" (
    echo [ERROR] dependencies.txt not found!
    set MISSING_FILES=1
)
if not exist "src\cmg_sr3_files_data_processor\streamlit_extractor.py" (
    echo [ERROR] src\cmg_sr3_files_data_processor\streamlit_extractor.py not found!
    set MISSING_FILES=1
)
if not exist "src\cmg_sr3_files_data_processor\streamlit_visualizer.py" (
    echo [ERROR] src\cmg_sr3_files_data_processor\streamlit_visualizer.py not found!
    set MISSING_FILES=1
)
if !MISSING_FILES! equ 1 (
    echo.
    echo Please ensure you're running this script from the correct directory.
    echo Expected layout:
    echo   - dependencies.txt  [repo root]
    echo   - src\cmg_sr3_files_data_processor\*.py
    echo.
    pause
    exit /b 1
)

echo [INFO] Checking dependencies...
echo.

REM Upgrade pip first to ensure reliable installation
echo [INFO] Upgrading pip to latest version...
"%VENV_PY%" -m pip install --upgrade pip --quiet 2>nul
if errorlevel 1 (
    echo [WARNING] Failed to upgrade pip, continuing anyway...
)

REM Check critical dependencies
set INSTALL_NEEDED=0

REM Test each critical package
"%VENV_PY%" -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo [CHECK] streamlit - MISSING
    set INSTALL_NEEDED=1
) else (
    echo [CHECK] streamlit - OK
)

"%VENV_PY%" -c "import numpy" >nul 2>&1
if errorlevel 1 (
    echo [CHECK] numpy - MISSING
    set INSTALL_NEEDED=1
) else (
    echo [CHECK] numpy - OK
)

"%VENV_PY%" -c "import pandas" >nul 2>&1
if errorlevel 1 (
    echo [CHECK] pandas - MISSING
    set INSTALL_NEEDED=1
) else (
    echo [CHECK] pandas - OK
)

"%VENV_PY%" -c "import h5py" >nul 2>&1
if errorlevel 1 (
    echo [CHECK] h5py - MISSING
    set INSTALL_NEEDED=1
) else (
    echo [CHECK] h5py - OK
)

"%VENV_PY%" -c "import matplotlib" >nul 2>&1
if errorlevel 1 (
    echo [CHECK] matplotlib - MISSING
    set INSTALL_NEEDED=1
) else (
    echo [CHECK] matplotlib - OK
)

echo.

REM Install dependencies if needed
if !INSTALL_NEEDED! equ 1 (
    echo [INFO] Installing missing dependencies from dependencies.txt...
    echo This may take a few minutes on first run...
    echo.
    
    "%VENV_PY%" -m pip install --upgrade pip setuptools wheel --quiet
    "%VENV_PY%" -m pip install -r dependencies.txt
    
    if errorlevel 1 (
        echo.
        echo [ERROR] Failed to install dependencies!
        echo.
        echo Troubleshooting steps:
        echo 1. Ensure you have internet connection
        echo 2. Try running as Administrator
        echo 3. Install manually: pip install -r dependencies.txt
        echo 4. If error mentions "Windows Long Path", enable it or keep using this venv path: "%VENV_DIR%"
        echo 5. If you are using Microsoft Store Python, consider installing Python from python.org
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
"%VENV_PY%" -c "import streamlit; import numpy; import pandas; import h5py; import matplotlib; print('All dependencies verified!')" >nul 2>&1
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
REM Ensure we're in the correct directory for the import test
cd /d "%~dp0" >nul 2>&1
REM Verify we can import the packaged app (src layout)
"%VENV_PY%" -c "import cmg_sr3_files_data_processor.app" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] app.py may have syntax errors or missing dependencies!
    echo Current directory: "%CD%"
    echo Script directory: "%~dp0"
    echo The app will still attempt to run, but errors may occur.
    echo.
) else (
    echo [INFO] Application files verified!
    echo [INFO] Working directory: "%CD%"
    echo.
)

echo ============================================================
echo   Starting Web Application...
echo ============================================================
echo.
echo The browser will open automatically in a few seconds.
echo Access the app at: http://localhost:8501  (or next available port if 8501 is busy)
echo Press Ctrl+C to stop the server when done.
echo.
echo ============================================================
echo.

REM Run the application
"%VENV_PY%" -m cmg_sr3_files_data_processor.run_app

REM If the script exits, pause to show any error messages
if errorlevel 1 (
    echo.
    echo ============================================================
    echo [ERROR] Application exited with an error!
    echo ============================================================
    echo.
    echo If you see import errors, try:
    echo    pip install -r dependencies.txt
    echo.
    echo If issues persist, check:
    echo    1. Python version (need 3.7+)
    echo    2. All project files are present
    echo    3. No conflicting Python installations
    echo.
    pause
)

