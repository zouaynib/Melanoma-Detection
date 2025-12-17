import torch
import torch.nn as nn
from torchvision import models




def create_backbone(backbone_name: str = "mobilenet_v3_large",
                    pretrained: bool = False):
    backbone_name = backbone_name.lower()

    # -------------------------
    # ResNet50 (your current)
    # -------------------------
    if backbone_name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, feature_dim

    # -------------------------
    # MobileNetV3-Large (very efficient)
    # -------------------------
    elif backbone_name == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        feature_dim = model.classifier[-1].in_features
        model.classifier = nn.Identity()
        return model, feature_dim

    # -------------------------
    # EfficientNet-V2-S (strong accuracy/efficiency)
    # -------------------------
    elif backbone_name == "efficientnet_v2_s":
        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_v2_s(weights=weights)
        feature_dim = model.classifier[-1].in_features
        model.classifier = nn.Identity()
        return model, feature_dim

    # -------------------------
    # EfficientNet-B0 (already supported, kept for reference)
    # -------------------------
    elif backbone_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        feature_dim = model.classifier[-1].in_features
        model.classifier = nn.Identity()
        return model, feature_dim

    # -------------------------
    # ConvNeXt-Tiny (often strong accuracy)
    # -------------------------
    elif backbone_name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.convnext_tiny(weights=weights)
        feature_dim = model.classifier[-1].in_features
        model.classifier = nn.Identity()
        return model, feature_dim

    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")


class MetadataMLP(nn.Module):
    """
    Small MLP to embed metadata vector into a learned feature vector.
    """
    def __init__(self, metadata_dim: int, hidden_dim: int = 128, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(metadata_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),

            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
        )

    def forward(self, x):
        return self.net(x)


class CombinedModel(nn.Module):
    """
    Model that combines:
      - CNN image backbone
      - MLP metadata branch
    and outputs logits for ISIC 2019 (8 classes by default).

    Forward signature:
        logits = model(images, metadata)
    where:
        images   : (B, 3, 224, 224)
        metadata : (B, metadata_dim)
        logits   : (B, num_classes)
    """
    def __init__(
        self,
        backbone_name="efficientnet_v2_s",
        pretrained_backbone: bool = True,
        metadata_dim: int = 1 + 3 + 10,   # adjust 10 to your number of site categories
        metadata_hidden_dim: int = 128,
        metadata_out_dim: int = 128,
        num_classes: int = 8,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Image backbone
        self.backbone, feat_dim = create_backbone(
            backbone_name=backbone_name,
            pretrained=pretrained_backbone
        )

        # Metadata branch
        self.meta_mlp = MetadataMLP(
            metadata_dim=metadata_dim,
            hidden_dim=metadata_hidden_dim,
            out_dim=metadata_out_dim,
        )

        # Final classifier on concatenated features
        combined_dim = feat_dim + metadata_out_dim
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, images, metadata):
        """
        images   : Tensor (B, 3, 224, 224)
        metadata : Tensor (B, metadata_dim)
        """
        # Image features
        img_feat = self.backbone(images)          # (B, feat_dim)

        # Metadata features
        meta_feat = self.meta_mlp(metadata)       # (B, metadata_out_dim)

        # Concatenate
        combined = torch.cat([img_feat, meta_feat], dim=1)  # (B, feat_dim + meta_out)

        # Logits
        logits = self.classifier(combined)
        return logits
