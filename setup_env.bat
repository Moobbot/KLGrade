@echo off
REM Setup script for KLGrade API (Windows)
REM Updated to use pip for PyTorch to avoid DLL issues

echo ==========================================
echo KLGrade API Environment Setup
echo ==========================================

echo Step 1: Creating base conda environment...
call conda create -n klgrade_api python=3.10 -y

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to create conda environment
    echo Please ensure conda is installed and in your PATH
    pause
    exit /b 1
)

echo.
echo Step 2: Activating environment...
call conda activate klgrade_api

echo.
echo Step 3: Installing PyTorch with CUDA 12.1 via pip...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to install PyTorch
    pause
    exit /b 1
)

echo.
echo Step 4: Installing all dependencies from requirements.txt...
python -m pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo Step 5: Verifying installation...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

if %errorlevel% neq 0 (
    echo.
    echo WARNING: PyTorch verification failed
    echo You may need to run fix_pytorch.bat
)

echo.
echo ==========================================
echo Environment setup complete!
echo ==========================================
echo.
echo To activate the environment, run:
echo   conda activate klgrade_api
echo.
echo Then start the API server:
echo   start_api.bat
echo.
pause
