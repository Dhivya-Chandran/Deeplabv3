#!/usr/bin/env python3
"""Debug: Check model output shape, value range, and loss computation."""

import torch
import torch.nn as nn
import numpy as np
from models.deeplabv3 import DeepLabv3
from optimized_data_loader import get_cityscapes_loader

# Load one batch
print("[INFO] Loading one training batch...")
loader = get_cityscapes_loader(
    input_path="datasets/Cityscapes/leftImg8bit/train",
    segmented_path="datasets/Cityscapes/gtFine_trainIds/train",
    batch_size=2,
    target_height=513,
    target_width=513,
    augment=True,
    num_workers=0,
    shuffle=False
)

img_batch, mask_batch = next(iter(loader))
print(f"Image shape: {img_batch.shape}, range: [{img_batch.min():.3f}, {img_batch.max():.3f}]")
print(f"Mask shape: {mask_batch.shape}, range: [{mask_batch.min()}, {mask_batch.max()}]")
print(f"Unique mask values: {torch.unique(mask_batch)[:20]}")

# Test model on CPU first
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n[INFO] Testing on device: {device}")

model = DeepLabv3(nc=19, dropout_rate=0.1)
model = model.to(device)
model.eval()

with torch.no_grad():
    img_batch = img_batch.to(device)
    outputs = model(img_batch)
    print(f"\n[INFO] Model output shape: {outputs.shape}")
    print(f"Model output range: [{outputs.min():.3f}, {outputs.max():.3f}]")
    print(f"Model output dtype: {outputs.dtype}")
    
    # Check NaNs/Infs
    print(f"NaNs in output: {torch.isnan(outputs).sum()}")
    print(f"Infs in output: {torch.isinf(outputs).sum()}")
    
    # Compute loss
    mask_batch = mask_batch.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
    loss = criterion(outputs, mask_batch.long())
    
    print(f"\n[INFO] Loss per sample: {loss}")
    print(f"Mean loss: {loss.mean():.6f}")
    print(f"Max loss: {loss.max():.6f}")
    print(f"NaNs in loss: {torch.isnan(loss).sum()}")

print("\nModel output appears valid if values are in reasonable range and loss is finite")
