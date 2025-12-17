import timm
import torch.nn as nn


def build_model(
    model_name: str = "swin_tiny_patch4_window7_224",
    num_classes: int = 8,
    pretrained: bool = True,
) -> nn.Module:
    """
    Build Swin-Tiny classification model via timm.
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model
