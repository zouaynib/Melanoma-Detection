import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np

# ============================================================
# Hair Removal Function (DullRazor style)
# ============================================================
def remove_hair(img_bgr):
    """
    Remove hair using DullRazor-style:
    1. Convert to grayscale
    2. Black-hat filter to detect hairs
    3. Threshold mask
    4. Inpaint
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    inpainted = cv2.inpaint(img_bgr, mask, 1, cv2.INPAINT_TELEA)
    return inpainted


# ============================================================
# Resize + Center Crop to 224×224
# ============================================================
def resize_and_crop(img_bgr, out_size=224):
    h, w = img_bgr.shape[:2]
    scale = 256.0 / min(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img_bgr, (new_w, new_h))

    # Center crop
    top = (new_h - out_size) // 2
    left = (new_w - out_size) // 2

    crop = resized[top:top + out_size, left:left + out_size]
    return crop


# ============================================================
# MAIN
# ============================================================
def main():

    # Paths relative to ISIC2019 folder
    RAW_ROOT = Path("raw")
    OUT_ROOT = Path("processed")

    IN_IMAGES = RAW_ROOT / "images"
    OUT_IMAGES = OUT_ROOT / "images_224_nohair"
    OUT_IMAGES.mkdir(parents=True, exist_ok=True)

    # Metadata
    meta_path = "/gpfs/workdir/erekrakead/ISIC2019/raw/ISIC_2019_Training_Metadata.csv"
    gt_path   = "/gpfs/workdir/erekrakead/ISIC2019/raw/ISIC_2019_Training_GroundTruth.csv"

    meta = pd.read_csv(meta_path)
    gt   = pd.read_csv(gt_path)

    # Convert one-hot → single label
    diag_cols = ["AK", "BCC", "BKL", "DF", "NV", "MEL", "SCC", "VASC"]
    gt["diagnosis"] = gt[diag_cols].idxmax(axis=1)

    # Merge metadata
    df = meta.merge(gt[["image", "diagnosis"]], on="image")
    df.rename(columns={"image": "image_name"}, inplace=True)

    print(f"Loaded metadata: {df.shape[0]} images")

    skipped = 0
    processed = 0

    # ============================================================
    # PROCESS IMAGES
    # ============================================================
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing images"):
        img_name = row["image_name"] + ".jpg"
        in_path  = IN_IMAGES / img_name

        out_path = OUT_IMAGES / (row["image_name"] + ".png")

        # ----------------------------------------------------
        # Skip if already processed
        # ----------------------------------------------------
        if out_path.exists():
            skipped += 1
            continue

        # Load
        img_bgr = cv2.imread(str(in_path))
        if img_bgr is None:
            print(f"WARNING: Could not read {img_name}")
            continue

        # Preprocess
        img_clean = remove_hair(img_bgr)
        img_final = resize_and_crop(img_clean, out_size=224)

        # Save
        cv2.imwrite(str(out_path), img_final)
        processed += 1

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    print(f"\nProcessed images: {processed}")
    print(f"Skipped images (already done): {skipped}")

    # Save merged metadata
    df.to_csv(OUT_ROOT / "all_metadata.csv", index=False)
    print("\nSaved: processed images and all_metadata.csv")


if __name__ == "__main__":
    main()
