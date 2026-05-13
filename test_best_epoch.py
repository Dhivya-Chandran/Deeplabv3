#!/usr/bin/env python3
"""
Quick test script to evaluate the best validation checkpoint against the latest epoch
for a chosen backbone. Helps verify the overfitting diagnosis.
"""

import torch
import sys
import os

BACKBONE = os.environ.get('BACKBONE', 'resnet101').strip().lower()
EPOCH = int(os.environ.get('EPOCH', '150'))

def _stem(backbone_name: str) -> str:
    return backbone_name.replace('-', '_') or 'model'

# Check if best checkpoints exist
checkpoints = {
    f'best_miou_{BACKBONE}': f'best_miou_model_{_stem(BACKBONE)}.pth',
    f'epoch_{EPOCH}_{BACKBONE}': f'checkpoint_epoch_{_stem(BACKBONE)}_{EPOCH}.pth',
}

print("=" * 60)
print("CHECKPOINT COMPARISON FOR OVERFITTING DIAGNOSIS")
print("=" * 60)

for name, path in checkpoints.items():
    if not os.path.exists(path):
        print(f"\n {name:15s} ({path}): NOT FOUND")
        continue
    
    try:
        ckpt = torch.load(path, map_location='cpu')
        size_mb = os.path.getsize(path) / (1024**2)
        print(f"\n{name:15s} ({path})")
        print(f"  Size: {size_mb:.1f} MB")
        if 'epoch' in ckpt:
            print(f"  Epoch saved: {ckpt['epoch']}")
        if 'best_miou' in ckpt:
            print(f"  Best mIoU: {ckpt['best_miou']:.6f}")
        if 'best_loss' in ckpt:
            print(f"  Best loss: {ckpt['best_loss']:.6f}")
    except Exception as e:
        print(f"\n{name:15s}: Error loading - {str(e)[:50]}")

print("VALIDATION METRICS BY EPOCH")

import json
try:
    with open('training_metrics.json', 'r') as f:
        m = json.load(f)
    
    if m['val_miou']:
        print(f"\n{'Epoch':<8} {'Val mIoU':<12} {'Val Pixel Acc':<15}")
        print("-" * 40)
        for i, epoch in enumerate(m['val_epochs']):
            miou = m['val_miou'][i] if i < len(m['val_miou']) else 0
            acc = m['val_pixel_accuracy'][i] if i < len(m['val_pixel_accuracy']) else 0
            marker = "   BEST" if miou == max(m['val_miou']) else ""
            print(f"{epoch:<8} {miou:<12.6f} {acc:<15.2f}{marker}")
            
        best_epoch = m['val_epochs'][m['val_miou'].index(max(m['val_miou']))]
        print(f"\nRECOMMENDATION: Use checkpoint_epoch_{_stem(BACKBONE)}_{best_epoch}.pth for testing")
except Exception as e:
    print(f"Could not load metrics: {e}")

print("\n" + "=" * 60)
print("ACTIONABLE STEPS:")
print("=" * 60)
print(f"""
1. Test with best epoch checkpoint:
    python test.py --checkpoint best_miou_model_{_stem(BACKBONE)}.pth \
                  --input-path datasets/Cityscapes/leftImg8bit/test \\
                  --label-path datasets/Cityscapes/gtFine_trainIds/test

2. Compare results with epoch {EPOCH} (current):
    python test.py --checkpoint checkpoint_epoch_{_stem(BACKBONE)}_{EPOCH}.pth \
                  --input-path datasets/Cityscapes/leftImg8bit/test \\
                  --label-path datasets/Cityscapes/gtFine_trainIds/test

3. For next training, fix early stopping in train_full.sbatch:
   Change: --early-stop-patience 999
   To:     --early-stop-patience 40
""")

