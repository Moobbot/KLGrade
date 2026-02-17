#!/bin/bash
# Setup script for KLGrade API (Linux/Mac)
# Updated to use pip for PyTorch to avoid potential issues

echo "=========================================="
echo "KLGrade API Environment Setup"
echo "=========================================="

echo "Step 1: Creating base conda environment..."
conda create -n klgrade_api python=3.10 -y

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Failed to create conda environment"
    echo "Please ensure conda is installed and in your PATH"
    exit 1
fi

echo ""
echo "Step 2: Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate klgrade_api

echo ""
echo "Step 3: Installing PyTorch with CUDA 12.1 via pip..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Failed to install PyTorch"
    exit 1
fi

echo ""
echo "Step 4: Installing all dependencies from requirements.txt..."
python -m pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Failed to install dependencies"
    exit 1
fi

echo ""
echo "Step 5: Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

if [ $? -ne 0 ]; then
    echo ""
    echo "WARNING: PyTorch verification failed"
fi

echo ""
echo "=========================================="
echo "✅ Environment setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate klgrade_api"
echo ""
echo "Then start the API server:"
echo "  ./start_api.sh"
