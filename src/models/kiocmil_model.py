import torch
import torch.nn as nn
import torchvision.models as models
from typing import List, Dict


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


class KiocmilModel(nn.Module):
    def __init__(
        self,
        backbone_name="resnet18",
        num_classes=10,
        feature_dim=256,
        use_attention=True,
    ):
        super(KiocmilModel, self).__init__()
        self.backbone_name = backbone_name
        self.feature_dim = feature_dim
        self.use_attention = use_attention

        # 1. Backbone
        if backbone_name == "resnet18":
            resnet = models.resnet18(pretrained=True)
            self.backbone_dim = 512
        elif backbone_name == "resnet50":
            resnet = models.resnet50(pretrained=True)
            self.backbone_dim = 2048
        else:
            raise ValueError(f"Unknown backbone {backbone_name}")

        # Remove FC
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # Projectors (if needed, or just use backbone_dim)
        # We'll project to feature_dim for consistent fusion
        self.projector = nn.Linear(self.backbone_dim, feature_dim)

        # 2. Fusion (Option A: MLP)
        # Input: Concat[ctx, js, ost] -> 3 * feature_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
        )

        # 3. Heads
        # Attention pooling for MIL (Enhanced with multi-head attention)
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

    def forward_features(self, x_batch):
        """
        Run backbone on a batch of patches x_batch: (N, 3, H, W)
        """
        if x_batch.numel() == 0:
            return torch.zeros(0, self.backbone_dim).to(x_batch.device)

        feats = self.backbone(x_batch)  # (N, Dim, 1, 1)
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

        # Mapping to reconstruct: (batch_idx, knee_idx) -> (ctx_idx, start_js, count_js, start_ost, count_ost)
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
        if not all_ctx:  # Should not happen if data exists
            return None

        t_ctx = torch.stack(all_ctx)
        t_js = (
            torch.cat(all_js) if all_js else torch.empty(0, 3, 224, 224).to(device)
        )  # dummy shape
        t_ost = (
            torch.cat(all_ost) if all_ost else torch.empty(0, 3, 224, 224).to(device)
        )

        # Run backbone features
        # We can run separately per type or concat all.
        # Separately is easier for memory if batch is huge, or concat for efficiency.
        # Let's run separately to handle different resolutions if they were different (Model allows same backbone).
        # Dataset produces: CTX (384), Patch (224). So resolutions differ!
        # Must run separately or padded. Separately is safe.

        f_ctx_all = self.forward_features(t_ctx)  # (Total_Knees, Dim)
        f_js_all = self.forward_features(t_js)  # (Total_JS, Dim) - can be 0 size
        f_ost_all = self.forward_features(t_ost)  # (Total_OST, Dim) - can be 0 size

        # 3. Assemble Knee Features
        knee_features_list = []  # List of (Dim) per knee
        batch_knee_map = [
            [] for _ in range(len(batch_data))
        ]  # b_idx -> list of knee vectors

        for info in patch_map:
            # Context
            f_local_ctx = f_ctx_all[info["ctx_idx"]]

            # JS Pooling (Avg or Max)
            if info["js_count"] > 0:
                s = info["js_start"]
                e = s + info["js_count"]
                feats = f_js_all[s:e]
                # Max pooling usually better for MIL (detect lesion)
                f_local_js = torch.max(feats, dim=0)[0]
            else:
                # Fallback: Zero vector (or handling "no lesion")
                # Ideally dataset always returns fallback patch.
                # If dataset works right, js_count >= 1 always (fallback band).
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

        # 4. Image Level Aggregation (Max Pool over Knees)
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
