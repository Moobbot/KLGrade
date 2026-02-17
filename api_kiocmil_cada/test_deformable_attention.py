#!/usr/bin/env python3
"""
Quick test to verify the KIOCMIL CADA model can be instantiated
and the deformable attention is properly activated.
"""

import sys

sys.path.insert(0, "/home/ngoductam/KLGrade/api_kiocmil_cada")

import torch
from models.kiocmil_model_cada import KiocmilModelCADA


def test_model_instantiation():
    """Test that model can be instantiated without errors."""
    print("Testing KIOCMIL CADA model instantiation...")

    try:
        # Create model with default parameters
        model = KiocmilModelCADA(
            backbone_name="yolo11l",
            num_classes=10,
            feature_dim=256,
            num_deformable_points=4,
            num_context_scales=3,
            use_positional_encoding=True,
            dropout=0.1,
        )
        print("✅ Model instantiated successfully!")

        # Check that deformable attention modules exist
        assert hasattr(model, "cross_attn_js"), "Missing cross_attn_js"
        assert hasattr(model, "cross_attn_ost"), "Missing cross_attn_ost"
        assert hasattr(model, "context_encoder"), "Missing context_encoder"
        print("✅ All deformable attention modules present!")

        # Test forward pass with dummy data
        print("\nTesting forward pass with dummy data...")
        batch_data = [
            {
                "knees": [
                    {
                        "ctx": torch.randn(3, 384, 384),
                        "ctx_bbox": torch.tensor([0.5, 0.5, 0.3, 0.3]),
                        "js": torch.randn(2, 3, 224, 224),
                        "js_bboxes": torch.tensor(
                            [[0.4, 0.4, 0.1, 0.1], [0.6, 0.6, 0.1, 0.1]]
                        ),
                        "ost": torch.randn(1, 3, 224, 224),
                        "ost_bboxes": torch.tensor([[0.5, 0.3, 0.1, 0.1]]),
                    }
                ]
            }
        ]

        with torch.no_grad():
            output = model(batch_data)

        print(f"✅ Forward pass successful!")
        print(f"   - logits_10 shape: {output['logits_10'].shape}")
        print(f"   - logits_grade shape: {output['logits_grade'].shape}")
        print(f"   - logits_type shape: {output['logits_type'].shape}")
        print(f"   - embedding shape: {output['embedding'].shape}")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_model_instantiation()
    sys.exit(0 if success else 1)
