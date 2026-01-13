import torch
import torch.nn as nn
from ultralytics import YOLO
from typing import List, Dict
import warnings


class AttentionPool(nn.Module):
    """
    Enhanced Attention-based pooling for MIL (Multiple Instance Learning).
    Uses multi-head attention to learn diverse feature patterns and better handle
    minority classes in imbalanced datasets.
    """

    def __init__(self, input_dim, hidden_dim=128, num_heads=4, dropout=0.2):
        super(AttentionPool, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        # Multi-head self-attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # Learnable query for pooling
        self.query = nn.Parameter(torch.randn(1, 1, input_dim))

        # Additional projection for refined features (optional)
        self.feature_refine = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, features):
        """
        Args:
            features: (K, D) tensor where K is number of instances, D is feature dim

        Returns:
            pooled: (D,) pooled feature vector
            weights: (K,) attention weights (for visualization)
        """
        if features.dim() == 1:
            # Single instance, return as-is
            return features, torch.ones(1, device=features.device)

        # Add batch dimension: (1, K, D)
        x = features.unsqueeze(0)

        # Expand query to match batch
        query = self.query.expand(x.size(0), -1, -1)  # (1, 1, D)

        # Multi-head attention: query attends to all knee features
        # attn_output: (1, 1, D), attn_weights: (1, 1, K)
        attn_output, attn_weights = self.multihead_attn(
            query=query,
            key=x,
            value=x,
            need_weights=True,
            average_attn_weights=True,  # Average across heads for visualization
        )

        # Refine features (optional enhancement)
        refined = self.feature_refine(attn_output)

        # Squeeze batch and sequence dimensions
        pooled = refined.squeeze(0).squeeze(0)  # (D,)
        weights = attn_weights.squeeze(0).squeeze(0)  # (K,)

        return pooled, weights


class KiocmilModelV3(nn.Module):
    """
    KIOCMIL V3 with YOLO11L backbone for improved feature extraction.

    Changes from V2:
    - Replaced ResNet18/50 with YOLO11L backbone
    - Extract features from YOLO's C5 layer (1024D)
    - Keep same fusion MLP, attention pooling, and heads
    """

    def __init__(
        self,
        backbone_name="yolo11l",
        num_classes=10,
        feature_dim=256,
        use_attention=True,
    ):
        super(KiocmilModelV3, self).__init__()
        self.backbone_name = backbone_name
        self.feature_dim = feature_dim
        self.use_attention = use_attention

        # 1. YOLO11L Backbone
        print(f"Loading {backbone_name} backbone...")
        try:
            yolo = YOLO(f"{backbone_name}.pt")  # Will download if not exists
            self.backbone, self.backbone_dim = self._extract_yolo_backbone(yolo)
            print(f"✅ YOLO backbone loaded. Feature dim: {self.backbone_dim}")
        except Exception as e:
            warnings.warn(
                f"Failed to load YOLO backbone: {e}. Using ResNet18 fallback."
            )
            # Fallback to ResNet18
            import torchvision.models as models

            resnet = models.resnet18(pretrained=True)
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
            self.backbone_dim = 512

        # Projector: backbone_dim → feature_dim
        self.projector = nn.Linear(self.backbone_dim, feature_dim)

        # 2. Fusion MLP (same as V2)
        # Input: Concat[ctx, js, ost] → 3 * feature_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
        )

        # 3. Heads (same as V2)
        # Attention pooling for MIL
        if use_attention:
            self.attention_pool = AttentionPool(
                input_dim=feature_dim, hidden_dim=128, num_heads=4, dropout=0.2
            )

        # Head: 10-class
        self.head_10 = nn.Linear(feature_dim, num_classes)
        # Head: 5-class (grade 0..4)
        self.head_grade = nn.Linear(feature_dim, 5)
        # Head: binary type (Ost vs JS)
        self.head_type = nn.Linear(feature_dim, 1)

    def _extract_yolo_backbone(self, yolo_model):
        """
        Extract feature extraction backbone from YOLO11 model.

        Returns:
            backbone: nn.Module for feature extraction
            feature_dim: int, output feature dimension
        """
        # Get YOLO model layers
        model = yolo_model.model

        # YOLO11 architecture typically has:
        # - Backbone layers (0-9): CSPDarknet with C3/C4/C5 outputs
        # - Neck layers (10-): PAFPN for multi-scale features
        # - Head layers: Detection heads

        # Extract backbone up to C5 output (deepest features)
        # Typically layers 0-9 or 0-10 depending on YOLO version
        try:
            # Try to get backbone attribute first
            if hasattr(model, "model"):
                layers = list(model.model.children())
            else:
                layers = list(model.children())

            # Take first 10 layers (up to C5)
            backbone_layers = layers[:10]
            backbone = nn.Sequential(*backbone_layers)

            # Determine output dimension by forward pass
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224)
                dummy_output = backbone(dummy_input)

                # Handle different output formats
                if isinstance(dummy_output, (list, tuple)):
                    # Multi-scale outputs, use last (deepest)
                    dummy_output = dummy_output[-1]

                if dummy_output.dim() == 4:  # (B, C, H, W)
                    feature_dim = dummy_output.shape[1]
                else:
                    feature_dim = dummy_output.shape[-1]

            return backbone, feature_dim

        except Exception as e:
            warnings.warn(f"Error extracting YOLO backbone: {e}")
            # Fallback: use entire model as feature extractor
            feature_dim = 1024  # Typical for YOLO11L
            return model, feature_dim

    def forward_features(self, x_batch):
        """
        Run backbone on a batch of patches x_batch: (N, 3, H, W)
        """
        if x_batch.numel() == 0:
            return torch.zeros(0, self.backbone_dim).to(x_batch.device)

        # YOLO backbone forward
        feats = self.backbone(x_batch)

        # Handle different output formats
        if isinstance(feats, (list, tuple)):
            # Multi-scale outputs, use last (deepest)
            feats = feats[-1]

        # Global average pooling if spatial dimensions exist
        if feats.dim() == 4:  # (N, C, H, W)
            feats = feats.mean(dim=[-2, -1])  # (N, C)
        elif feats.dim() == 3:  # (N, C, L) - sequence format
            feats = feats.mean(dim=-1)  # (N, C)

        # Flatten if needed
        if feats.dim() > 2:
            feats = feats.flatten(1)

        return self.projector(feats)  # (N, feature_dim)

    def forward(self, batch_data):
        """
        batch_data: List of dicts (output of Dataset)
        Structure of one item:
        {
          'knees': [ {'ctx': T, 'js': T(n,3,h,w), 'ost': T(m,3,h,w)}, ... ]
        }
        """
        # Strategy: Flatten all patches into one large batch to run backbone in parallel

        all_ctx = []
        all_js = []
        all_ost = []

        # Mapping to reconstruct: (batch_idx, knee_idx) → (ctx_idx, start_js, count_js, start_ost, count_ost)
        patch_map = []

        ctx_counter = 0
        js_counter = 0
        ost_counter = 0

        device = None

        # 1. Collect Patches
        for b_idx, item in enumerate(batch_data):
            knees = item["knees"]
            for k_idx, knee in enumerate(knees):
                # Context
                ctx = knee["ctx"]
                if device is None:
                    device = ctx.device
                all_ctx.append(ctx)

                # JS
                js = knee["js"]  # (N, 3, H, W)
                n_js = js.shape[0] if js.numel() > 0 else 0
                if n_js > 0:
                    all_js.append(js)

                # OST
                ost = knee["ost"]
                n_ost = ost.shape[0] if ost.numel() > 0 else 0
                if n_ost > 0:
                    all_ost.append(ost)

                patch_map.append(
                    {
                        "b_idx": b_idx,
                        "ctx_idx": ctx_counter,
                        "js_start": js_counter,
                        "js_count": n_js,
                        "ost_start": ost_counter,
                        "ost_count": n_ost,
                    }
                )

                ctx_counter += 1
                js_counter += n_js
                ost_counter += n_ost

        # 2. Run Backbone
        # Concatenate lists
        if (
            not all_ctx
        ):  # No knees detected (rare edge case with aggressive augmentation)
            # Return dummy predictions with proper gradient tracking
            # Create zero embeddings and pass through heads to maintain gradient graph
            model_device = next(self.parameters()).device
            batch_size = len(batch_data)

            # Create dummy embeddings that require grad
            dummy_emb = torch.zeros(
                batch_size, self.feature_dim, device=model_device, requires_grad=True
            )

            # Pass through model heads to get outputs with gradients
            dummy_logits_10 = self.head_10(dummy_emb)
            dummy_logits_grade = self.head_grade(dummy_emb)
            dummy_logits_type = self.head_type(dummy_emb)

            return {
                "logits_10": dummy_logits_10,
                "logits_grade": dummy_logits_grade,
                "logits_type": dummy_logits_type,
                "embedding": dummy_emb,
            }

        t_ctx = torch.stack(all_ctx)
        t_js = (
            torch.cat(all_js) if all_js else torch.empty(0, 3, 224, 224).to(device)
        )  # dummy shape
        t_ost = (
            torch.cat(all_ost) if all_ost else torch.empty(0, 3, 224, 224).to(device)
        )

        # Run backbone features
        f_ctx_all = self.forward_features(t_ctx)  # (Total_Knees, Dim)
        f_js_all = self.forward_features(t_js)  # (Total_JS, Dim) - can be 0 size
        f_ost_all = self.forward_features(t_ost)  # (Total_OST, Dim) - can be 0 size

        # 3. Assemble Knee Features
        knee_features_list = []  # List of (Dim) per knee
        batch_knee_map = [
            [] for _ in range(len(batch_data))
        ]  # b_idx → list of knee vectors

        for info in patch_map:
            # Context
            f_local_ctx = f_ctx_all[info["ctx_idx"]]

            # JS Pooling (Max pooling for MIL)
            if info["js_count"] > 0:
                s = info["js_start"]
                e = s + info["js_count"]
                feats = f_js_all[s:e]
                f_local_js = torch.max(feats, dim=0)[0]
            else:
                f_local_js = torch.zeros_like(f_local_ctx)

            # OST Pooling
            if info["ost_count"] > 0:
                s = info["ost_start"]
                e = s + info["ost_count"]
                feats = f_ost_all[s:e]
                f_local_ost = torch.max(feats, dim=0)[0]
            else:
                f_local_ost = torch.zeros_like(f_local_ctx)

            # Fusion
            combined = torch.cat(
                [f_local_ctx, f_local_js, f_local_ost], dim=0
            )  # (3*Dim)
            f_knee = self.fusion_mlp(combined.unsqueeze(0)).squeeze(0)  # (Dim)

            batch_knee_map[info["b_idx"]].append(f_knee)

        # 4. Image Level Aggregation (Attention or Max Pool over Knees)
        img_features = []
        for b_idx in range(len(batch_data)):
            knees = batch_knee_map[b_idx]
            if not knees:
                # Should not happen
                img_features.append(torch.zeros(f_ctx_all.shape[1]).to(device))
                continue

            stack_knees = torch.stack(knees)  # (K, Dim)

            # MIL Pooling: Attention or Max
            if self.use_attention:
                f_img, _ = self.attention_pool(stack_knees)
            else:
                # Max Pool over knees
                f_img = torch.max(stack_knees, dim=0)[0]

            img_features.append(f_img)

        final_emb = torch.stack(img_features)  # (B, Dim)

        # 5. Heads
        logits_10 = self.head_10(final_emb)
        logits_grade = self.head_grade(final_emb)
        logits_type = self.head_type(final_emb)

        return {
            "logits_10": logits_10,
            "logits_grade": logits_grade,
            "logits_type": logits_type,
            "embedding": final_emb,
        }
