import os
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from dataset import ISIC2019Dataset, TransformSubset
from model import build_model


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default="data/ISIC_2019_Training_GroundTruth.csv")
    ap.add_argument("--train_img_dir", default="data/ISIC_2019_Training_Input")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--ckpt_dir", default="checkpoints/swin_tiny")
    ap.add_argument("--model_name", default="swin_tiny_patch4_window7_224")
    args = ap.parse_args()

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Transforms
    train_tf = T.Compose([
        T.Resize(256),
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.2, 0.2, 0.2, 0.05),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_tf = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Base dataset WITHOUT transforms (returns PIL)
    base_ds = ISIC2019Dataset(args.train_csv, args.train_img_dir, transform=None)
    y = np.array(base_ds.labels)

    indices = np.arange(len(base_ds))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=args.val_ratio,
        random_state=args.seed,
        stratify=y,   # important for ISIC imbalance
    )

    train_subset = Subset(base_ds, train_idx)
    val_subset = Subset(base_ds, val_idx)

    train_ds = TransformSubset(train_subset, transform=train_tf)
    val_ds = TransformSubset(val_subset, transform=val_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    model = build_model(model_name=args.model_name, num_classes=base_ds.num_classes, pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_path = os.path.join(args.ckpt_dir, "best.pth")

    best_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        # ---- train ----
        model.train()
        running_loss = 0.0

        for x, yb in train_loader:
            x = x.to(device)
            yb = yb.to(device, dtype=torch.long)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # ---- val ----
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for x, yb in val_loader:
                x = x.to(device)
                yb = yb.to(device, dtype=torch.long)
                pred = model(x).argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += yb.numel()

        val_acc = correct / max(total, 1)

        print(f"Epoch {epoch:02d}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Save only if best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f"  ✓ Saved best checkpoint to: {best_path} (best_acc={best_acc:.4f})")

    print(f"Done. Best Val Acc: {best_acc:.4f}")
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
