@echo off
REM ========================================
REM Fix PyTorch DLL Loading Issues
REM ========================================
REM This script fixes "OSError: [WinError 182]" or DLL loading errors
REM by reinstalling PyTorch via pip instead of conda.
REM Only run this if you encounter PyTorch import errors.
REM ========================================

echo ========================================
echo Fixing PyTorch in klgrade_api environment
echo ========================================

call conda activate klgrade_api

if %errorlevel% neq 0 (
    echo ERROR: Failed to activate klgrade_api environment
    echo Please run setup_env.bat first
    pause
    exit /b 1
)

echo.
echo Step 1: Removing old PyTorch installation...
pip uninstall torch torchvision -y

echo.
echo Step 2: Installing PyTorch with CUDA 12.1 via pip...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

if %errorlevel% neq 0 (
    echo ERROR: Failed to install PyTorch
    pause
    exit /b 1
)

echo.
echo Step 3: Testing PyTorch installation...
python -c "import torch; print(f'  ✓ PyTorch: {torch.__version__}'); print(f'  ✓ CUDA available: {torch.cuda.is_available()}')"

if %errorlevel% neq 0 (
    echo ERROR: PyTorch still not working
    echo Please check your Visual C++ Redistributable installation
    pause
    exit /b 1
)

echo.
echo ========================================
echo ✓ PyTorch fixed! You can now start the API.
echo ========================================
pause
