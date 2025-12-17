# src/train/test.py

import sys
from pathlib import Path
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
import numpy as np

from lightning import seed_everything

THIS_FILE = Path(__file__).resolve()
SRC_ROOT = THIS_FILE.parents[1]
PROJECT_ROOT = THIS_FILE.parents[2]

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from dataset.isic_dataset import ISIC2019Dataset
from train.lightning_module import ISICLightningModule


def build_val_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def load_model(ckpt_path, metadata_dim, num_classes, class_weights):
    """Load trained Lightning model."""
    model = ISICLightningModule.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        backbone_name="resnet50",
        metadata_dim=metadata_dim,
        num_classes=num_classes,
        lr=1e-4,
        weight_decay=1e-5,
        class_weights=class_weights,
    )
    model.eval()
    model.cuda()
    return model


def main():
    seed_everything(42)

    proc_root = PROJECT_ROOT / "processed"
    img_dir = proc_root / "images_224_nohair"
    val_csv = proc_root / "splits" / "val.csv"

    df = pd.read_csv(val_csv)

    # Metadata encoding parameters
    age_min = df["age_approx"].min()
    age_max = df["age_approx"].max()

    sex_categories = ["male", "female", "unknown"]

    site_categories = list(df["anatom_site_general"].unique())
    if "unknown" not in site_categories:
        site_categories.append("unknown")

    metadata_dim = 1 + len(sex_categories) + len(site_categories)

    num_classes = 8
    counts = df["label"].value_counts().sort_index()
    total = counts.sum()
    class_weights = torch.tensor(
        (total / (num_classes * counts)).values, dtype=torch.float32
    )

    val_tf = build_val_transform()

    dataset = ISIC2019Dataset(
        csv_path=val_csv,
        images_dir=img_dir,
        transform=val_tf,
        age_min=age_min,
        age_max=age_max,
        sex_categories=sex_categories,
        site_categories=site_categories,
    )

    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    ckpt_path = PROJECT_ROOT / "checkpoints" / "best-model.ckpt"
    model = load_model(str(ckpt_path), metadata_dim, num_classes, class_weights)

    all_labels = []
    all_preds = []
    all_probs = []

    print("Running inference on validation set...")

    with torch.no_grad():
        for imgs, metadata, labels in loader:
            imgs = imgs.cuda()
            metadata = metadata.cuda()

            logits = model(imgs, metadata)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)

    # ---------------------------------------------------------------
    # Confusion Matrix
    # ---------------------------------------------------------------
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix (Validation)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "analysis_confmat.png")
    plt.close()

    # ---------------------------------------------------------------
    # Classification report
    # ---------------------------------------------------------------
    report = classification_report(
        all_labels, all_preds, digits=3, output_dict=True
    )
    pd.DataFrame(report).to_csv(PROJECT_ROOT / "analysis_classification_report.csv")

    # ---------------------------------------------------------------
    # Multi-class ROC curves (one-vs-rest)
    # ---------------------------------------------------------------
    fpr = {}
    tpr = {}
    roc_auc = {}

    for c in range(num_classes):
        fpr[c], tpr[c], _ = roc_curve(all_labels == c, all_probs[:, c])
        roc_auc[c] = auc(fpr[c], tpr[c])

        plt.plot(fpr[c], tpr[c], label=f"Class {c} (AUC = {roc_auc[c]:.3f})")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (One-vs-Rest)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "analysis_roc.png")
    plt.close()

    print("Evaluation complete.")
    print("Saved:")
    print("  - analysis_confmat.png")
    print("  - analysis_roc.png")
    print("  - analysis_classification_report.csv")


if __name__ == "__main__":
    main()
