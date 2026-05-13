#!/usr/bin/env python
"""Debug script to check data loader output shapes and values"""

import torch
import numpy as np
from utils import loader_cscapes
import sys

# Test parameters
batch_size = 2
h = 512
w = 1024
ip = 'datasets/Cityscapes/leftImg8bit/train'
lp = 'datasets/Cityscapes/gtFine_trainIds/train'

print("="*60)
print("DEBUGGING DATA LOADER")
print("="*60)

# Create loader
pipe = loader_cscapes(ip, lp, batch_size, h=h, w=w)

# Get first batch
X_batch, mask_batch = next(pipe)

print(f"\n[INPUT BATCH]")
print(f"  Shape: {X_batch.shape}")
print(f"  Dtype: {X_batch.dtype}")
print(f"  Min: {X_batch.min():.6f}")
print(f"  Max: {X_batch.max():.6f}")
print(f"  Mean: {X_batch.mean():.6f}")
print(f"  Contains NaN: {torch.isnan(X_batch).any()}")
print(f"  Contains Inf: {torch.isinf(X_batch).any()}")

print(f"\n[LABEL BATCH]")
print(f"  Shape: {mask_batch.shape}")
print(f"  Dtype: {mask_batch.dtype}")
print(f"  Unique values: {torch.unique(mask_batch)[:10]}")  # First 10 unique
print(f"  Min: {mask_batch.min()}")
print(f"  Max: {mask_batch.max()}")
print(f"  Contains NaN: {torch.isnan(mask_batch.float()).any()}")

# Test model forward pass
from models.deeplabv3 import DeepLabv3

print(f"\n[MODEL TEST]")
model = DeepLabv3(nc=19)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

X_batch = X_batch.to(device)
mask_batch = mask_batch.to(device)

with torch.no_grad():
    output = model(X_batch.float())

print(f"  Output shape: {output.shape}")
print(f"  Output dtype: {output.dtype}")
print(f"  Output min: {output.min():.6f}")
print(f"  Output max: {output.max():.6f}")
print(f"  Output mean: {output.mean():.6f}")
print(f"  Contains NaN: {torch.isnan(output).any()}")

# Test loss
criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

print(f"\n[LOSS TEST]")
loss = criterion(output, mask_batch.long())
print(f"  Loss: {loss.item():.6f}")
print(f"  Loss is 0: {loss.item() == 0.0}")

print("\n" + "="*60)
