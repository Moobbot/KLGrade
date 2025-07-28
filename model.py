from torch import nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
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


def model_return(args):
    backbone = get_backbone(getattr(args, "model_type", "resnet_101"))
    # Tạo anchor generator (FPN-like)
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),)
    )
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=["0"], output_size=7, sampling_ratio=2
    )
    model = FasterRCNN(
        backbone,
        num_classes=NUM_CLASSES,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
    )
    return model
