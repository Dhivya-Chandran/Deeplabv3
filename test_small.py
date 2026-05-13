#!/usr/bin/env python
"""Quick test script to verify data pipeline and training"""

import torch
import torch.nn as nn
import torch.optim as optim
from utils import loader_cscapes
from models.deeplabv3 import DeepLabv3
import glob
import os
from tqdm import tqdm

print("="*70)
print("SMALL DATASET TEST - DeepLabv3")
print("="*70)

# Test parameters
device = torch.device('cpu')  # Force CPU
batch_size = 2
epochs = 2
lr = 1e-4
wd = 1e-4
nc = 19
h = 512
w = 1024

ip = 'datasets/Cityscapes/leftImg8bit/train'
lp = 'datasets/Cityscapes/gtFine_trainIds/train'

print(f"\n[CONFIG]")
print(f"  Device: {device}")
print(f"  Batch size: {batch_size}")
print(f"  Epochs: {epochs}")
print(f"  Image size: {h}x{w}")
print(f"  Num classes: {nc}")

# Count samples
train_samples = len(glob.glob(ip + '/**/*.png', recursive=True))
print(f"  Total training samples: {train_samples}")

# Initialize model
print(f"\n[MODEL]")
model = DeepLabv3(nc)
model.to(device)
print(f"  ✓ Model loaded")

# Loss and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=255)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
print(f"  ✓ Loss function: CrossEntropyLoss")
print(f"  ✓ Optimizer: Adam")

# Data loader
print(f"\n[DATA LOADING]")
pipe = loader_cscapes(ip, lp, batch_size, h=h, w=w)
print(f"  ✓ Data loader initialized")

# Training loop
print(f"\n[TRAINING - {epochs} epochs]")
print("-" * 70)

for epoch in range(1, epochs + 1):
    model.train()
    epoch_loss = 0.0
    batch_count = 0
    
    # Run only on first 10 batches for quick test
    for batch_idx in range(10):
        try:
            X_batch, mask_batch = next(pipe)
            
            # Skip empty batches
            if X_batch.size(0) < 1:
                print(f"  ⚠️ Batch {batch_idx}: Empty batch, skipping")
                continue
            
            X_batch = X_batch.to(device)
            mask_batch = mask_batch.to(device)
            
            print(f"  Batch {batch_idx}: X shape {X_batch.shape}, mask shape {mask_batch.shape}")
            
            # Forward pass
            optimizer.zero_grad()
            out = model(X_batch.float())
            
            print(f"    Output shape: {out.shape}")
            print(f"    Output range: [{out.min():.4f}, {out.max():.4f}]")
            
            # Compute loss
            loss = criterion(out, mask_batch.long())
            
            print(f"    ✓ Loss: {loss.item():.6f}")
            
            if loss.item() == 0.0:
                print(f"    ⚠️  WARNING: Loss is zero!")
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            
        except Exception as e:
            print(f"  ✗ Error in batch {batch_idx}: {str(e)}")
            import traceback
            traceback.print_exc()
            break
    
    avg_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
    print(f"\nEpoch {epoch}/{epochs} - Avg Loss: {avg_loss:.6f} ({batch_count} batches)")
    print("-" * 70)

print(f"\n{'='*70}")
print("✓ TEST COMPLETED SUCCESSFULLY")
print(f"{'='*70}\n")
