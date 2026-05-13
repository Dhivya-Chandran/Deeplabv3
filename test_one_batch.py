#!/usr/bin/env python
"""Quick test - just 1 batch to verify the fix"""

import torch
import torch.nn as nn
from utils import loader_cscapes
from models.deeplabv3 import DeepLabv3

print("Loading test...")
device = torch.device('cpu')
batch_size = 2
nc = 19

# Initialize
model = DeepLabv3(nc)
model.to(device)
criterion = nn.CrossEntropyLoss(ignore_index=255)

# Data loader
pipe = loader_cscapes('datasets/Cityscapes/leftImg8bit/train', 
                      'datasets/Cityscapes/gtFine_trainIds/train', 
                      batch_size, h=512, w=1024)

print("Getting batch...")
X_batch, mask_batch = next(pipe)

print(f"✓ X shape: {X_batch.shape}")
print(f"✓ mask shape: {mask_batch.shape}")

X_batch = X_batch.to(device)
mask_batch = mask_batch.to(device)

print("Forward pass...")
out = model(X_batch.float())

print(f"✓ Output shape: {out.shape}")

print("Computing loss...")
loss = criterion(out, mask_batch.long())

print(f"\n✓✓✓ SUCCESS ✓✓✓")
print(f"Loss: {loss.item():.6f}")

if loss.item() == 0.0:
    print("⚠️  Loss is still 0!")
else:
    print("✓ Loss is non-zero!")
