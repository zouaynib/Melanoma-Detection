# ---------------------------------------------------------------
# src/train/gradcam.py
# Grad-CAM for your CombinedModel (image + metadata)
# ---------------------------------------------------------------

import os
import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image

# Make src importable
THIS_FILE = Path(__file__).resolve()
SRC_ROOT = THIS_FILE.parents[1]          # .../src
PROJECT_ROOT = THIS_FILE.parents[2]      # project root
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.isic_dataset import ISIC2019Dataset
from train.lightning_module import ISICLightningModule


# ------------------------------------------------------------------
# 1. Load & clean checkpoint (remove loss-related buffers)
# ------------------------------------------------------------------
def load_clean_checkpoint(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    to_remove = [
        k for k in ckpt["state_dict"]
        if ("class_weights" in k) or ("criterion" in k)
    ]
    for k in to_remove:
        print(f"Removing: {k}")
        ckpt["state_dict"].pop(k)

    return ckpt


# ------------------------------------------------------------------
# 2. Wrapper so Grad-CAM only sees images; we inject metadata inside
# ------------------------------------------------------------------
class GradCAMWrapper(torch.nn.Module):
    def __init__(self, combined_model, metadata_tensor):
        """
        combined_model: models.combined_model.CombinedModel
        metadata_tensor: (1, metadata_dim) tensor (we will overwrite it each sample)
        """
        super().__init__()
        self.model = combined_model
        self.metadata = metadata_tensor  # will be updated for each sample

    def forward(self, x):
        # x: (B, 3, 224, 224)
        b = x.shape[0]
        meta = self.metadata.repeat(b, 1).to(x.device)
        return self.model(x, meta)


# ------------------------------------------------------------------
# 3. Find the last Conv2d in the backbone (robust for ResNet)
# ------------------------------------------------------------------
def find_last_conv(module: torch.nn.Module):
    last = None
    for m in module.modules():
        if isinstance(m, torch.nn.Conv2d):
            last = m
    return last


# ------------------------------------------------------------------
# 4. Side-by-side (original vs Grad-CAM) plot
# ------------------------------------------------------------------
def save_side_by_side(image_np, cam_np, out_path: Path):
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))

    ax[0].imshow(image_np)
    ax[0].set_title("Original")
    ax[0].axis("off")

    ax[1].imshow(cam_np)
    ax[1].set_title("Grad-CAM")
    ax[1].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------
# 5. Main Grad-CAM loop
# ------------------------------------------------------------------
def run_gradcam(
    ckpt_path: str = "checkpoints/best-model.ckpt",
    num_samples: int = 10,
    output_dir: str = "gradcam_outputs",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    print("Loading checkpoint...")
    ckpt = load_clean_checkpoint(ckpt_path)

    print("Rebuilding LightningModule...")
    model_lit = ISICLightningModule(**ckpt["hyper_parameters"])
    missing, unexpected = model_lit.load_state_dict(ckpt["state_dict"], strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    # Extract CombinedModel and put on device
    combined_model = model_lit.model
    combined_model.eval().to(device)

    # ------------------------------------------------------------------
    # Rebuild SAME metadata encoders as in train.py
    # ------------------------------------------------------------------
    import pandas as pd

    proc_root = PROJECT_ROOT / "processed"
    img_dir = proc_root / "images_224_nohair"
    train_csv = proc_root / "splits" / "train.csv"
    val_csv = proc_root / "splits" / "val.csv"

    train_df = pd.read_csv(train_csv)

    age_min = train_df["age_approx"].min()
    age_max = train_df["age_approx"].max()

    sex_categories = ["male", "female", "unknown"]

    site_categories = list(train_df["anatom_site_general"].unique())
    if "unknown" not in site_categories:
        site_categories.append("unknown")

    # SAME val transforms as in train.py
    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_dataset = ISIC2019Dataset(
        csv_path=val_csv,
        images_dir=img_dir,
        transform=val_tf,
        age_min=age_min,
        age_max=age_max,
        sex_categories=sex_categories,
        site_categories=site_categories,
    )

    print(f"\nGenerating Grad-CAM for {num_samples} validation samples...\n")

    # Dummy metadata tensor; will be overwritten each sample
    # metadata_dim is given in hyper_parameters
    metadata_dim = model_lit.hparams.metadata_dim
    dummy_meta = torch.zeros(1, metadata_dim, device=device)

    # Wrap the CombinedModel
    wrapped = GradCAMWrapper(combined_model, dummy_meta)

    # Choose target layer = last Conv2d in the backbone
    target_layer = find_last_conv(combined_model.backbone)
    if target_layer is None:
        raise RuntimeError("Could not find any Conv2d layer in backbone for Grad-CAM.")
    target_layers = [target_layer]
    print("Using target layer for CAM:", target_layer)

    for idx in range(num_samples):
        # ---------------------------
        # 1) Get sample
        # ---------------------------
        image, meta, label = val_dataset[idx]   # image: (3, 224, 224), meta: (metadata_dim,)

        img_tensor = image.unsqueeze(0).to(device)        # (1, 3, 224, 224)
        meta_tensor = meta.unsqueeze(0).to(device)        # (1, metadata_dim)

        # Update metadata used inside wrapper
        wrapped.metadata = meta_tensor

        # Prepare image for overlay (0–1 range, HWC)
        img_np = image.permute(1, 2, 0).numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

        # ---------------------------
        # 2) Grad-CAM++
        # ---------------------------
        cam = GradCAMPlusPlus(model=wrapped, target_layers=target_layers)

        grayscale_cam = cam(
            input_tensor=img_tensor,
            targets=None,           # default: argmax class
            eigen_smooth=True,
        )[0]                        # (H, W)

        cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

        # ---------------------------
        # 3) Save side-by-side figure
        # ---------------------------
        out_path = out_dir / f"gradcam_{idx}.png"
        save_side_by_side(img_np, cam_image, out_path)
        print(f"[{idx}] Saved:", out_path)

    print("\nDone. Grad-CAM images saved to:", out_dir)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="checkpoints/best-model.ckpt")
    parser.add_argument("--num", type=int, default=10)
    parser.add_argument("--out", type=str, default="gradcam_outputs")

    args = parser.parse_args()

    run_gradcam(
        ckpt_path=args.ckpt,
        num_samples=args.num,
        output_dir=args.out,
    )
