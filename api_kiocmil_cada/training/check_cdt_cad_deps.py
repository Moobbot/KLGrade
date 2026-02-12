#!/usr/bin/env python3
"""
Check CDT-CAD Training Dependencies

Verifies all required dependencies for CDT-CAD model training.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

def check_pytorch():
    """Check PyTorch and CUDA availability."""
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.version.cuda}")
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
            print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("❌ CUDA not available")
            return False
    except ImportError as e:
        print(f"❌ PyTorch not installed: {e}")
        return False

def check_pywavelets():
    """Check PyWavelets installation."""
    try:
        import pywt
        print(f"✅ PyWavelets: {pywt.__version__}")
        return True
    except ImportError:
        print("❌ PyWavelets not installed")
        print("   Install with: pip install PyWavelets")
        return False

def check_torchvision():
    """Check torchvision installation."""
    try:
        import torchvision
        print(f"✅ torchvision: {torchvision.__version__}")
        return True
    except ImportError:
        print("❌ torchvision not installed")
        return False

def check_model_modules():
    """Check CDT-CAD model modules."""
    print("\nChecking CDT-CAD model modules...")
    
    modules_to_check = [
        ("src.models.cdt_cad.cdt_cad_model", "CDTCAD"),
        ("src.models.cdt_cad.deformable_transformer", "DeformableTransformerEncoder"),
        ("src.models.cdt_cad.feature_extractor", "IterativeContextAwareFeatureExtractor"),
        ("src.losses.cdt_cad_loss", "CDTCADLoss"),
    ]
    
    all_ok = True
    for module_name, class_name in modules_to_check:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"✅ {module_name}.{class_name}")
        except ImportError as e:
            print(f"❌ {module_name}: {e}")
            all_ok = False
        except AttributeError as e:
            print(f"❌ {module_name}.{class_name} not found: {e}")
            all_ok = False
    
    return all_ok

def check_other_deps():
    """Check other dependencies."""
    deps = [
        "cv2",
        "numpy",
        "yaml",
        "tqdm",
    ]
    
    all_ok = True
    for dep in deps:
        try:
            __import__(dep)
            print(f"✅ {dep}")
        except ImportError:
            print(f"❌ {dep} not installed")
            all_ok = False
    
    return all_ok

def main():
    print("="*60)
    print("CDT-CAD Training Dependencies Check")
    print("="*60)
    
    print("\n1. PyTorch & CUDA:")
    pytorch_ok = check_pytorch()
    
    print("\n2. PyWavelets:")
    pywavelets_ok = check_pywavelets()
    
    print("\n3. torchvision:")
    torchvision_ok = check_torchvision()
    
    print("\n4. Other dependencies:")
    other_ok = check_other_deps()
    
    print("\n5. CDT-CAD model modules:")
    model_ok = check_model_modules()
    
    print("\n" + "="*60)
    if all([pytorch_ok, pywavelets_ok, torchvision_ok, other_ok, model_ok]):
        print("✅ All dependencies satisfied!")
        print("="*60)
        return 0
    else:
        print("❌ Some dependencies missing. Please install them.")
        print("="*60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
