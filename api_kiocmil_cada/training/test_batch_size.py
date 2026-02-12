#!/usr/bin/env python3
"""
Test Maximum Batch Size for CDT-CAD Training

Finds the maximum batch size that fits in GPU memory.
"""

import torch
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.cdt_cad.cdt_cad_model import CDTCAD

def test_batch_size(batch_size, img_size=512, hidden_dim=128, num_layers=4):
    """Test if a batch size fits in memory."""
    try:
        # Clear cache
        torch.cuda.empty_cache()
        
        # Create model
        model = CDTCAD(
            num_classes=5,
            num_queries=100,
            hidden_dim=hidden_dim,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
        ).cuda()
        
        # Create dummy input
        images = torch.randn(batch_size, 3, img_size, img_size).cuda()
        
        # Forward pass
        with torch.cuda.amp.autocast():
            outputs = model(images)
        
        # Get memory usage
        memory_allocated = torch.cuda.memory_allocated() / 1e9
        memory_reserved = torch.cuda.memory_reserved() / 1e9
        
        # Cleanup
        del model, images, outputs
        torch.cuda.empty_cache()
        
        return True, memory_allocated, memory_reserved
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            torch.cuda.empty_cache()
            return False, 0, 0
        else:
            raise e

def main():
    print("="*60)
    print("CDT-CAD Batch Size Testing")
    print("="*60)
    
    # Test configurations
    configs = [
        {"img_size": 512, "hidden_dim": 128, "num_layers": 4},
        {"img_size": 384, "hidden_dim": 128, "num_layers": 4},
        {"img_size": 512, "hidden_dim": 64, "num_layers": 4},
        {"img_size": 384, "hidden_dim": 64, "num_layers": 4},
    ]
    
    for config in configs:
        print(f"\n📊 Testing config: img_size={config['img_size']}, "
              f"hidden_dim={config['hidden_dim']}, layers={config['num_layers']}")
        print("-" * 60)
        
        max_batch = 1
        for batch_size in [1, 2, 4, 8, 12, 16]:
            success, mem_alloc, mem_reserved = test_batch_size(
                batch_size, 
                config['img_size'], 
                config['hidden_dim'], 
                config['num_layers']
            )
            
            if success:
                print(f"✅ Batch {batch_size:2d}: OK "
                      f"(Allocated: {mem_alloc:.2f}GB, Reserved: {mem_reserved:.2f}GB)")
                max_batch = batch_size
            else:
                print(f"❌ Batch {batch_size:2d}: OOM")
                break
        
        print(f"\n🎯 Maximum batch size: {max_batch}")
        
        # Calculate effective batch size with accumulation
        for accum in [2, 4, 8, 16]:
            effective = max_batch * accum
            print(f"   With accumulation={accum}: effective batch={effective}")

if __name__ == "__main__":
    main()
