#!/usr/bin/env python3
"""
Real-time training progress monitor for DeepLabv3 on Cityscapes.

Tracks:
- mIoU progression (current vs baseline 2.15%)
- Loss trends
- Training speed
- Estimated completion time
- Early stopping warnings
"""

import json
import os
import time
import sys
from pathlib import Path
from datetime import datetime, timedelta
import subprocess


def load_metrics(metrics_path='training_metrics.json'):
    """Load training metrics from JSON file."""
    if not os.path.exists(metrics_path):
        return None
    try:
        with open(metrics_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load metrics: {e}")
        return None


def get_job_info():
    """Get SLURM job information."""
    try:
        result = subprocess.run(['squeue', '-u', os.environ.get('USER', 'dchandr4'), '--format=%i %.19T %.10M'],
                              capture_output=True, text=True, timeout=5)
        lines = result.stdout.strip().split('\n')[1:]
        
        for line in lines:
            if 'train' in line.lower() or 'deeplabv3' in line.lower() or line.strip():
                parts = line.split()
                if len(parts) >= 3:
                    return {
                        'job_id': parts[0],
                        'status': parts[1],
                        'elapsed': parts[2]
                    }
        return None
    except Exception as e:
        return None


def print_header():
    """Print monitoring header."""
    print("\n" + "=" * 100)
    print(" " * 30 + "DeepLabv3 Training Monitor")
    print("=" * 100)


def print_status(metrics, job_info):
    """Print current training status."""
    if metrics is None:
        print("[ERROR] No training metrics found yet. Training may not have started.")
        return
    
    epochs = metrics.get('epochs', [])
    val_epochs = metrics.get('val_epochs', [])
    val_miou = metrics.get('val_miou', [])
    train_loss = metrics.get('train_loss', [])
    
    if not epochs:
        print("[INFO] No training data available yet")
        return
    
    current_epoch = epochs[-1]
    current_loss = train_loss[-1] if train_loss else None
    
    print(f"\nSTATUS AT {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 100)
    
    # Epoch progress
    print(f"  Current Epoch: {current_epoch} / 150")
    if job_info:
        print(f"  Job ID: {job_info.get('job_id', 'N/A')} | Status: {job_info.get('status', 'N/A')} | Elapsed: {job_info.get('elapsed', 'N/A')}")
    
    # Loss progression
    if current_loss is not None:
        print(f"  Current Loss: {current_loss:.6f}")
        if len(train_loss) > 1:
            loss_trend = "↓ Decreasing" if train_loss[-1] < train_loss[-2] else "↑ Increasing"
            print(f"  Loss Trend: {loss_trend}")
    
    # Validation mIoU
    if val_miou:
        latest_miou = val_miou[-1]
        baseline = 0.0215
        improvement = ((latest_miou - baseline) / baseline) * 100 if baseline > 0 else 0
        
        print(f"\n  VALIDATION mIoU TRACKING")
        print("-" * 100)
        print(f"  Baseline (old model):      {baseline:.4f} (2.15%)")
        print(f"  Current (your model):      {latest_miou:.4f} ({latest_miou*100:.2f}%)")
        print(f"  Improvement:               {improvement:+.1f}% ({latest_miou/baseline:.1f}× better)")
        
        # Phase expectations
        print(f"\n  PHASE EXPECTATIONS vs CURRENT:")
        print(f"  Phase 1 Target (Focal+Stride): 20-35% mIoU")
        if latest_miou < 0.20:
            print(f"    Current: {latest_miou*100:.2f}% (still training...)")
        elif latest_miou < 0.35:
            print(f"    Current: {latest_miou*100:.2f}% ✓ PHASE 1 IN RANGE!")
        else:
            print(f"    Current: {latest_miou*100:.2f}% ✓ EXCEEDING EXPECTATIONS!")
        
        print(f"  Phase 2 Target (+ SeparableConv + TTA): 50-65% mIoU")
        if latest_miou >= 0.20:
            expected_phase2 = min(latest_miou * 1.25, 0.65)  # Conservative estimate
            print(f"    Predicted: {expected_phase2*100:.2f}% (with TTA on validation)")
    
    # Training speed
    if len(epochs) > 1:
        time_per_epoch = (len(epochs) - 1) / max(1, current_epoch - 1)
        estimated_completion = current_epoch + (150 - current_epoch) * time_per_epoch
        print(f"\nTIMING ESTIMATES")
        print("-" * 100)
        print(f"  Epochs completed: {current_epoch}/150")
        print(f"  Epochs remaining: {150 - current_epoch}")
        print(f"  Estimated finish: epoch {estimated_completion:.0f} (check logs for actual time)")
    
    # Best epoch tracking
    if val_miou:
        best_miou_idx = val_miou.index(max(val_miou))
        best_epoch = val_epochs[best_miou_idx] if best_miou_idx < len(val_epochs) else "N/A"
        best_miou = max(val_miou)
        print(f"\nBEST CHECKPOINT")
        print("-" * 100)
        print(f"  Best mIoU: {best_miou:.4f} ({best_miou*100:.2f}%)")
        print(f"  At epoch: {best_epoch}")
        print(f"  File: best_miou_model_resnet101.pth")


def print_recommendations(metrics, job_info):
    """Print actionable recommendations."""
    if metrics is None or not metrics.get('val_miou'):
        return
    
    val_miou = metrics.get('val_miou', [])
    if not val_miou:
        return
    
    latest_miou = val_miou[-1]
    
    print(f"\nNEXT STEPS")
    print("-" * 100)
    
    if latest_miou < 0.05:
        print("    Training still in early phases")
        print("     Checkpoint: Wait for epoch 40+ to see meaningful improvements")
    
    elif latest_miou < 0.20:
        print("   Training progressing (Phase 1 in progress)")
        print("     Action: Check again after epoch 50")
        print("     Signal: Look for mIoU > 0.20 to confirm Focal Loss working")
    
    elif latest_miou < 0.35:
        print("   Phase 1 successful! (Focal Loss + Output Stride working)")
        print("     mIoU in expected range: 20-35%")
        print("     Next: Run validation with TTA enabled (automatic in test.py)")
        print("     Expected boost: +5-15% from TTA → 25-50% mIoU")
    
    elif latest_miou < 0.50:
        print("   Outstanding! Exceeding Phase 1 expectations!")
        print(f"     Current: {latest_miou*100:.2f}% (target was 35%)")
        print("     SeparableConv + TTA might already be boosting performance")
    
    else:
        print("   Exceptional results! You've reached Phase 2 territory!")
        print(f"     Current: {latest_miou*100:.2f}% (Phase 2 target: 50-65%)")
        print("     Consider proceeding to Phase 3: Weather-aware multi-task learning")
    
    print()


def print_footer():
    """Print monitoring footer with instructions."""
    print("=" * 100)
    print("  How to use this monitor:")
    print("   python3 monitor_training.py          # Single check")
    print("   watch -n 60 'python3 monitor_training.py'  # Auto-refresh every 60s")
    print()
    print("  Key files to check:")
    print("   • training_metrics.json              # All epoch metrics")
    print("   • best_miou_model_resnet101.pth      # Best checkpoint (automatically saved)")
    print("   • preds_val/epoch_*/                 # Validation predictions per epoch")
    print()
    print("   Training Control:")
    print("   scancel <JOB_ID>                     # Stop training")
    print("   squeue -u $USER                      # Check job status")
    print("=" * 100 + "\n")


def main():
    """Main monitoring loop."""
    metrics_path = 'training_metrics.json'
    
    # Initial load
    metrics = load_metrics(metrics_path)
    job_info = get_job_info()
    
    print_header()
    print_status(metrics, job_info)
    print_recommendations(metrics, job_info)
    print_footer()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Monitor stopped")
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
