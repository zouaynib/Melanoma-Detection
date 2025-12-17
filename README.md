# Swin Transformer for Skin Lesion Classification (ISIC 2019)

PyTorch + timm implementation of **Swin-Tiny** fine-tuned for multi-class skin lesion classification on **ISIC 2019**.

## Model
- Backbone: `swin_tiny_patch4_window7_224` (Swin-Tiny, patch4, window7)
- Pretrained: ImageNet-1K
- Fine-tuning: end-to-end
- Input: RGB images only (no metadata), 224×224, ImageNet normalization

## Dataset
This repo expects the official ISIC 2019 GroundTruth CSV format:
- Column 1: image id (without extension)
- Remaining columns: one-hot label columns  
  The number of classes is inferred from the CSV header (some versions include `UNK`, so it can be 8 or 9 classes).

Example header:

image,MEL,NV,BCC,AK,BKL,DF,VASC,SCC,UNK

### Expected structure
Melanoma-Detection/
dataset.py
model.py
train_swin_tiny.py
eval.py
requirements.txt
data/
ISIC_2019_Training_GroundTruth.csv
ISIC_2019_Training_Input/
ISIC_2019_Test_GroundTruth.csv # optional (if you have labels)
ISIC_2019_Test_Input/ # optional
checkpoints/


> Images are assumed to be `.jpg` files named `{image_id}.jpg`.

## Installation
```bashpip install -r requirements.txt
Minimal requirements.txt:

torch
torchvision
timm
numpy
pillow
scikit-learn



