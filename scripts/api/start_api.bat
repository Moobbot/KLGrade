@echo off
REM Quick start script for KLGrade API Server

echo ========================================
echo Starting KLGrade API Server
echo ========================================

REM Activate conda environment
call conda activate klgrade_api

if %errorlevel% neq 0 (
    echo ERROR: Failed to activate klgrade_api environment
    echo Please run setup_env.bat first
    pause
    exit /b 1
)

REM Set PYTHONPATH to include the project root
set PYTHONPATH=%CD%

echo.
echo Starting API server...
echo Server will be available at: http://localhost:8001
echo Swagger UI: http://localhost:8001/docs
echo.

python scripts/deployment/kiocmil_api_server.py ^
  --kiocmil-model runs/kiocmil_cada/cada_10class_balanced/best_acc_model.pt ^
  --knee-model runs/detect/my_knee_run_resplit/weights/best.pt ^
  --lesion-model runs/detect/my_knee_run_resplit/weights/best.pt ^
  --port 8001

pause
