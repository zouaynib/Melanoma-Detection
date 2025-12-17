import argparse
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from dataset import ISIC2019Dataset
from model import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--model_name", default="swin_tiny_patch4_window7_224")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tf = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    ds = ISIC2019Dataset(args.csv, args.img_dir, transform=tf)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    model = build_model(model_name=args.model_name, num_classes=ds.num_classes, pretrained=False).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device, dtype=torch.long)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()

    acc = correct / max(total, 1)
    print(f"Accuracy: {acc:.4f} (n={total})")


if __name__ == "__main__":
    main()
