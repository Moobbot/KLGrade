@echo off
echo ========================================
echo Testing KLGrade API Environment
echo ========================================

call conda activate klgrade_api

if %errorlevel% neq 0 (
    echo ERROR: Failed to activate klgrade_api environment
    echo Please run setup_env.bat first
    pause
    exit /b 1
)

echo.
echo [1/3] Checking PyTorch installation...
python -c "import torch; print(f'  ✓ PyTorch: {torch.__version__}'); print(f'  ✓ CUDA available: {torch.cuda.is_available()}')" 2>nul

if %errorlevel% neq 0 (
    echo   ✗ PyTorch not working properly
    echo   Run fix_pytorch.bat to fix this issue
    pause
    exit /b 1
)

echo.
echo [2/3] Checking API dependencies...
python -c "import fastapi, uvicorn; print('  ✓ FastAPI and Uvicorn OK')" 2>nul

echo.
echo [3/3] Checking ML/CV dependencies...
python -c "import cv2, albumentations, ultralytics; print('  ✓ OpenCV, Albumentations, Ultralytics OK')" 2>nul

echo.
echo ========================================
echo ✓ Environment is ready!
echo ========================================
echo.
echo To start the API server, run:
echo python scripts/deployment/kiocmil_api_server.py --kiocmil-model runs/kiocmil_cada/cada_10class_balanced/best_acc_model.pt --knee-model runs/detect/my_knee_run_resplit/weights/best.pt --lesion-model runs/detect/my_knee_run_resplit/weights/best.pt --port 8001
echo.
pause
