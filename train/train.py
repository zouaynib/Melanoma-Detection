# src/train/train.py

import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import pandas as pd


from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

# -------------------------------------------------------------
# Make `src/` and project root importable
# -------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
SRC_ROOT = THIS_FILE.parents[1]          # .../src
PROJECT_ROOT = THIS_FILE.parents[2]      # project root

# Insert both src/ and project root cleanly
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from dataset.isic_dataset import ISIC2019Dataset
from train.lightning_module import ISICLightningModule


def build_transforms():
    """Data augmentation and normalisation."""
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(25),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_tf, val_tf


def main(resume_ckpt: str | None = None):

    seed_everything(42)

    if torch.cuda.is_available():
        # Recommended on A100 to make use of Tensor Cores
        torch.set_float32_matmul_precision("medium")

    # ---------- Paths ----------
    proc_root = PROJECT_ROOT / "processed"
    img_dir = proc_root / "images_224_nohair"
    train_csv = proc_root / "splits" / "train.csv"
    val_csv   = proc_root / "splits" / "val.csv"

    # ---------- Load train_df for encoders & class weights ----------
    train_df = pd.read_csv(train_csv)

    # Age normalization parameters
    age_min = train_df["age_approx"].min()
    age_max = train_df["age_approx"].max()

    # Sex categories (consistent with Dataset)
    sex_categories = ["male", "female", "unknown"]

    # Site categories: all seen in train + "unknown"
    site_categories = list(train_df["anatom_site_general"].unique())
    if "unknown" not in site_categories:
        site_categories.append("unknown")

    metadata_dim = 1 + len(sex_categories) + len(site_categories)
    print(f"Metadata dim = {metadata_dim}")

    # ---------- Class weights for imbalanced loss ----------
    num_classes = 8
    counts = train_df["label"].value_counts().sort_index()  # index 0..7
    total = counts.sum()
    class_weights = total / (num_classes * counts)
    class_weights = torch.tensor(class_weights.values, dtype=torch.float32)
    print("Class weights:", class_weights)

    # ---------- Transforms ----------
    train_tf, val_tf = build_transforms()

    # ---------- Datasets ----------
    train_dataset = ISIC2019Dataset(
        csv_path=train_csv,
        images_dir=img_dir,
        transform=train_tf,
        age_min=age_min,
        age_max=age_max,
        sex_categories=sex_categories,
        site_categories=site_categories,
    )

    val_dataset = ISIC2019Dataset(
        csv_path=val_csv,
        images_dir=img_dir,
        transform=val_tf,
        age_min=age_min,
        age_max=age_max,
        sex_categories=sex_categories,
        site_categories=site_categories,
    )

    # ---------- DataLoaders ----------
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # ---------- LightningModule ----------
    model = ISICLightningModule(
            backbone_name="efficientnet_v2_s",
            metadata_dim=metadata_dim,
            num_classes=8,
            lr=1e-4,
            weight_decay=1e-5,
            warmup_epochs=5,
            class_weights=class_weights,
        )


    # -------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True, parents=True)

    tb_logger = TensorBoardLogger(
        save_dir=log_dir,
        name="tensorboard_logs"
    )

    csv_logger = CSVLogger(
        save_dir=log_dir,
        name="csv_logs"
    )

    # -------------------------------------------------------------
    # Checkpoints
    # -------------------------------------------------------------
    ckpt_dir = PROJECT_ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True, parents=True)

    # Save BEST model (lowest val_loss)
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best-model",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )

    # Stop early if val_loss doesn't improve for X epochs
    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=5,
        verbose=True,
    )

    # -------------------------------------------------------------
    # Trainer
    # -------------------------------------------------------------
    trainer = Trainer(
        accelerator="gpu",
        devices=2,               # you requested 4 A100s
        strategy="ddp",
        max_epochs=40,
        precision="16-mixed",
        log_every_n_steps=10,
        callbacks=[checkpoint_callback, early_stopping],
        logger=[tb_logger, csv_logger],
    )

    # -------------------------------------------------------------
    # Fit (with optional resume)
    # -------------------------------------------------------------
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=resume_ckpt,   # None → fresh run; path → resume
    )

    # Save final model **only on rank 0**
    if trainer.is_global_zero:
        final_path = ckpt_dir / "final_model.ckpt"
        trainer.save_checkpoint(str(final_path))
        print(f"Final model saved to: {final_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to .ckpt file to resume from (e.g. checkpoints/best-model.ckpt)",
    )
    args = parser.parse_args()

    main(resume_ckpt=args.resume)
