# Swin Transformer for Skin Lesion Classification (ISIC 2019)

This repository contains a PyTorch implementation of a Swin Transformer
trained on the ISIC 2019 dataset for multi-class skin lesion classification.

## Model
- Backbone: Swin-Tiny (patch4, window7)
- Pretrained on ImageNet-1K
- Fine-tuned end-to-end

## Input
- RGB dermoscopic images only
- Resolution: 224 × 224
- No metadata used

## Training
- Optimizer: AdamW
- Loss: Cross-Entropy
- Augmentations: Random crop, flip, rotation, color jitter

## Dataset
ISIC 2019 (8 classes)

## How to train
```bash
python train_swin_tiny_isic.py
