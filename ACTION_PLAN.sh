#!/bin/bash
# Quick action plan to verify the overfitting fix

echo "=========================================="
echo "DEEPLAB OVERFITTING FIX - ACTION PLAN"
echo "=========================================="

cd /scratch/dchandr4/DeepLabv3

# Step 1: Verify checkpoint files exist
echo ""
echo "STEP 1: Verify best checkpoint exists"
echo "---"
if [ -f best_miou_model.pth ]; then
    echo "✓ best_miou_model.pth found"
    ls -lh best_miou_model.pth | awk '{print "  Size: " $5}", {print "  Date: " $6, $7, $8}'
else
    echo "✗ best_miou_model.pth NOT FOUND"
    ls -lh checkpoint_epoch_20.pth 2>/dev/null && echo "  (Using checkpoint_epoch_20.pth instead)"
fi

# Step 2: Show validation metrics comparison
echo ""
echo "STEP 2: Validation metrics by epoch"
echo "---"
python3 << 'EOF'
import json
try:
    with open('training_metrics.json', 'r') as f:
        m = json.load(f)
    
    best_idx = m['val_miou'].index(max(m['val_miou']))
    best_epoch = m['val_epochs'][best_idx]
    best_miou = m['val_miou'][best_idx]
    worst_miou = min(m['val_miou'])
    
    print(f"Best:  Epoch {best_epoch:3d} → mIoU = {best_miou:.6f}")
    print(f"Worst: Epoch {m['val_epochs'][m['val_miou'].index(worst_miou)]:3d} → mIoU = {worst_miou:.6f}")
    print(f"Degradation factor: {best_miou/worst_miou:.0f}×")
except:
    print("Could not read training metrics")
EOF

# Step 3: Show next steps
echo ""
echo "STEP 3: Next steps to verify the fix"
echo "---"
echo ""
echo "Option A - Quick local test (CPU, ~5 min):"
echo """
  python test_best_epoch.py
"""
echo ""
echo "Option B - Full GPU evaluation with best checkpoint:"
echo """
  sbatch test.sbatch  # Uses best_miou_model.pth (updated default)
"""
echo ""
echo "Option C - Compare best vs worst checkpoints side-by-side:"
echo """
  # Test epoch 20 (best)
  CKPT=checkpoint_epoch_20.pth sbatch test.sbatch
  
  # Test epoch 150 (worst)  
  CKPT=checkpoint_epoch_150.pth sbatch test.sbatch
  
  # Check results in deeplap_test/*/val_summary.txt
"""

echo ""
echo "STEP 4: For next training"
echo "---"
echo "Changes already applied to train_full.sbatch:"
echo "  • Early stopping: --early-stop-patience 40 (was 999)"
echo "  • Weight decay: --weight-decay 5e-4 (was 1e-4)"
echo ""
echo "Additional config in train_full.sbatch you can add:"
echo "  • --dropout-rate 0.15 (model now supports this)"
echo "  • --batch-size 16 (if GPU memory allows; currently 8)"
echo ""
echo "Run training with:"
echo "  sbatch train_full.sbatch"
echo ""
echo "=========================================="
echo ""
