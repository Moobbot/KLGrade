from torch import nn
from config import NUM_CLASSES

# Hàm lấy backbone như cũ
def get_backbone(model_type):
    from torchvision import models
    if model_type == "resnet_101":
        backbone = models.resnet101(weights="DEFAULT")
        backbone = nn.Sequential(*list(backbone.children())[:-2])
        out_channels = 2048
    elif model_type == "resnext_50_32x4d":
        backbone = models.resnext50_32x4d(weights="DEFAULT")
        backbone = nn.Sequential(*list(backbone.children())[:-2])
        out_channels = 2048
    elif model_type == "wide_resnet_50_2":
        backbone = models.wide_resnet50_2(weights="DEFAULT")
        backbone = nn.Sequential(*list(backbone.children())[:-2])
        out_channels = 2048
    elif model_type == "densenet_161":
        backbone = models.densenet161(weights="DEFAULT")
        backbone = nn.Sequential(*list(backbone.features))
        out_channels = 2208
    elif model_type == "efficientnet_b5":
        backbone = models.efficientnet_b5(weights="DEFAULT")
        backbone = nn.Sequential(*list(backbone.features))
        out_channels = 2048
    elif model_type == "efficientnet_v2_s":
        backbone = models.efficientnet_v2_s(weights="DEFAULT")
        backbone = nn.Sequential(*list(backbone.features))
        out_channels = 1280
    elif model_type == "regnet_y_8gf":
        backbone = models.regnet_y_8gf(weights="DEFAULT")
        backbone = nn.Sequential(*list(backbone.children())[:-2])
        out_channels = 2016
    elif model_type == "shufflenet_v2_x2_0":
        backbone = models.shufflenet_v2_x2_0(weights="DEFAULT")
        backbone = nn.Sequential(*list(backbone.children())[:-1])
        out_channels = 2048
    else:
        backbone = models.resnet101(weights="DEFAULT")
        backbone = nn.Sequential(*list(backbone.children())[:-2])
        out_channels = 2048
    backbone.out_channels = out_channels
    return backbone

# MLP Proposal Head: feature map -> proposals (N, 4)
class ProposalMLP(nn.Module):
    def __init__(self, in_channels, num_proposals=3):
        super().__init__()
        self.num_proposals = num_proposals
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, 256),
            nn.ReLU(),
            nn.Linear(256, num_proposals * 4),
        )
    def forward(self, feat):
        x = self.pool(feat)  # (B, C, 1, 1)
        proposals = self.mlp(x)  # (B, num_proposals*4)
        proposals = proposals.view(-1, self.num_proposals, 4)  # (B, N, 4)
        return proposals

# Regression Head: proposals -> bbox
class RegressionMLP(nn.Module):
    def __init__(self, in_dim, num_proposals=3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )
        self.num_proposals = num_proposals
    def forward(self, proposal_feats):
        # proposal_feats: (B, N, in_dim)
        B, N, D = proposal_feats.shape
        out = self.mlp(proposal_feats.view(B*N, D))
        return out.view(B, N, 4)

# Classification Head: proposals -> class logits
class ClassificationMLP(nn.Module):
    def __init__(self, in_dim, num_proposals=3, num_classes=NUM_CLASSES):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )
        self.num_proposals = num_proposals
    def forward(self, proposal_feats):
        # proposal_feats: (B, N, in_dim)
        B, N, D = proposal_feats.shape
        out = self.mlp(proposal_feats.view(B*N, D))
        return out.view(B, N, -1)

# TwoStageDetector tổng quát
class TwoStageDetector(nn.Module):
    def __init__(
        self,
        backbone=None,
        proposal_head=None,
        regression_head=None,
        classification_head=None,
        num_proposals=3,
        num_classes=NUM_CLASSES,
        model_type="resnet_101",
    ):
        super().__init__()
        if backbone is None:
            backbone = get_backbone(model_type)
        self.backbone = backbone
        in_channels = getattr(backbone, "out_channels", 2048)
        if proposal_head is None:
            proposal_head = ProposalMLP(in_channels, num_proposals)
        self.proposal_head = proposal_head
        # Để đơn giản, dùng feature vector từ backbone cho proposal head và 2 head còn lại
        proposal_feat_dim = in_channels
        if regression_head is None:
            regression_head = RegressionMLP(proposal_feat_dim, num_proposals)
        if classification_head is None:
            classification_head = ClassificationMLP(proposal_feat_dim, num_proposals, num_classes)
        self.regression_head = regression_head
        self.classification_head = classification_head
        self.num_proposals = num_proposals
        self.num_classes = num_classes
    def forward(self, images):
        # images: (B, C, H, W)
        feat = self.backbone(images)  # (B, C, H', W')
        proposals = self.proposal_head(feat)  # (B, N, 4)
        # Lấy feature vector cho mỗi proposal (ở đây dùng global pooled feature cho đơn giản)
        B = feat.shape[0]
        pooled_feat = nn.functional.adaptive_avg_pool2d(feat, (1, 1)).view(B, -1)  # (B, C)
        proposal_feats = pooled_feat.unsqueeze(1).expand(-1, self.num_proposals, -1)  # (B, N, C)
        bbox_preds = self.regression_head(proposal_feats)  # (B, N, 4)
        class_logits = self.classification_head(proposal_feats)  # (B, N, num_classes)
        return {
            "boxes": bbox_preds,  # (B, N, 4)
            "logits": class_logits,  # (B, N, num_classes)
            "proposals": proposals,  # (B, N, 4)
        }

# Hàm khởi tạo model

def model_return(args):
    num_proposals = getattr(args, "num_proposals", 3)
    model = TwoStageDetector(
        model_type=getattr(args, "model_type", "resnet_101"),
        num_proposals=num_proposals
    )
    return model