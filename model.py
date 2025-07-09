from torch import nn
from torchvision import models
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from config import NUM_CLASSES


class FasterRCNNWithHead(FasterRCNN):
    def forward(self, images, targets=None):
        if self.training or targets is not None:
            return super().forward(images, targets)
        else:
            # Inference: trả về logits và boxes
            outputs = super().forward(images)
            # outputs: list of dicts with 'boxes', 'labels', 'scores'
            pred_logits = [o.get("scores", None) for o in outputs]
            pred_boxes = [o.get("boxes", None) for o in outputs]
            return pred_logits, pred_boxes


def get_backbone(model_type):
    if model_type == "resnet_101":
        backbone = models.resnet101(weights="DEFAULT")
        backbone = nn.Sequential(*list(backbone.children())[:-2])
        backbone.out_channels = 2048
    elif model_type == "resnext_50_32x4d":
        backbone = models.resnext50_32x4d(weights="DEFAULT")
        backbone = nn.Sequential(*list(backbone.children())[:-2])
        backbone.out_channels = 2048
    elif model_type == "wide_resnet_50_2":
        backbone = models.wide_resnet50_2(weights="DEFAULT")
        backbone = nn.Sequential(*list(backbone.children())[:-2])
        backbone.out_channels = 2048
    elif model_type == "densenet_161":
        backbone = models.densenet161(weights="DEFAULT")
        backbone = nn.Sequential(*list(backbone.features))
        backbone.out_channels = 2208
    elif model_type == "efficientnet_b5":
        backbone = models.efficientnet_b5(weights="DEFAULT")
        backbone = nn.Sequential(*list(backbone.features))
        backbone.out_channels = 2048
    elif model_type == "efficientnet_v2_s":
        backbone = models.efficientnet_v2_s(weights="DEFAULT")
        backbone = nn.Sequential(*list(backbone.features))
        backbone.out_channels = 1280
    elif model_type == "regnet_y_8gf":
        backbone = models.regnet_y_8gf(weights="DEFAULT")
        backbone = nn.Sequential(*list(backbone.children())[:-2])
        backbone.out_channels = 2016
    elif model_type == "shufflenet_v2_x2_0":
        backbone = models.shufflenet_v2_x2_0(weights="DEFAULT")
        backbone = nn.Sequential(*list(backbone.children())[:-1])
        backbone.out_channels = 2048
    else:
        # fallback
        backbone = models.resnet101(weights="DEFAULT")
        backbone = nn.Sequential(*list(backbone.children())[:-2])
        backbone.out_channels = 2048
    return backbone


def model_return(args):
    backbone = get_backbone(args.model_type)
    # Số lượng feature maps đầu ra của backbone là 2048 (ResNet101, ...), thường chỉ có 1 feature map
    anchor_generator = AnchorGenerator(
        sizes=(
            (32, 64, 128, 256, 512),
        ),  # tuple lồng tuple, số tuple ngoài = số feature map
        aspect_ratios=((0.5, 1.0, 2.0),),
    )

    model = FasterRCNNWithHead(
        backbone,
        num_classes=NUM_CLASSES,
        rpn_anchor_generator=anchor_generator,
    )
    return model
