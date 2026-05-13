#!/usr/bin/env python3
"""
Script to plot training and validation metrics from training_metrics_1.json
Run after training completes to generate loss and accuracy plots.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


def load_metrics(metrics_file='training_metrics_1.json'):
    """Load metrics from JSON file."""
    if not os.path.exists(metrics_file):
        print(f"Error: {metrics_file} not found!")
        print("Make sure training has completed and metrics were saved.")
        return None
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    return metrics


def resolve_metrics_file(preferred_file=None):
    """Pick the newest available metrics file when no path is provided."""
    if preferred_file:
        return preferred_file

    candidates = ['training_metrics.json', 'training_metrics_1.json']
    existing = [path for path in candidates if os.path.exists(path)]
    if not existing:
        return candidates[0]

    return max(existing, key=os.path.getmtime)


def plot_metrics(metrics, output_dir='.'):
    """Create plots for training and validation metrics."""
    
    epochs = metrics.get('epochs', [])
    train_loss = metrics.get('train_loss', [])
    train_pixel_acc = metrics.get('train_pixel_accuracy', [])
    val_epochs = metrics.get('val_epochs', [])
    val_loss = metrics.get('val_loss', [])
    val_miou = metrics.get('val_miou', [])
    val_pixel_acc = metrics.get('val_pixel_accuracy', [])

    # Backward compatibility for older metrics files
    if not val_epochs and val_loss:
        val_epochs = epochs[:len(val_loss)]

    # Normalize validation accuracy to percentage if stored as ratio in [0, 1]
    if val_pixel_acc and max(val_pixel_acc) <= 1.0:
        val_pixel_acc = [v * 100.0 for v in val_pixel_acc]
    
    # Create a 1x2 figure: Loss plot and Accuracy plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('DeepLabv3 Training & Validation Metrics (Cityscapes)', fontsize=16, fontweight='bold')
    
    # Plot 1: Train and Validation Loss
    ax = axes[0]
    if train_loss and epochs:
        ax.plot(epochs, train_loss, 'b-o', linewidth=2.5, markersize=5, label='Train Loss')
    if val_loss and val_epochs:
        ax.plot(val_epochs, val_loss, 'r-s', linewidth=2.5, markersize=5, label='Validation Loss')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Train & Validation Loss', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    # Plot 2: Train and Validation Accuracy
    ax = axes[1]
    if train_pixel_acc and epochs:
        ax.plot(epochs, train_pixel_acc, 'b-o', linewidth=2.5, markersize=5, label='Train Accuracy')
    if val_pixel_acc and val_epochs:
        ax.plot(val_epochs, val_pixel_acc, 'r-s', linewidth=2.5, markersize=5, label='Validation Accuracy')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Train & Validation Accuracy', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, 'training_metrics.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')

    print(f"✓ Plot saved to {output_path}")
    
    plt.close('all')


def print_summary(metrics):
    """Print a summary of the metrics."""
    print("\n" + "="*60)
    print("TRAINING METRICS SUMMARY")
    print("="*60)
    
    epochs = metrics.get('epochs', [])
    train_loss = metrics.get('train_loss', [])
    val_loss = metrics.get('val_loss', [])
    val_miou = metrics.get('val_miou', [])
    val_pixel_acc = metrics.get('val_pixel_accuracy', [])
    
    print(f"\nTotal Epochs: {len(epochs)}")
    
    if train_loss:
        print(f"\nTraining Loss:")
        print(f"  Initial: {train_loss[0]:.6f}")
        print(f"  Final:   {train_loss[-1]:.6f}")
        print(f"  Best:    {min(train_loss):.6f}")
    
    if val_loss:
        print(f"\nValidation Loss:")
        print(f"  Initial: {val_loss[0]:.6f}")
        print(f"  Final:   {val_loss[-1]:.6f}")
        print(f"  Best:    {min(val_loss):.6f}")
    
    if val_miou:
        print(f"\nValidation mIoU:")
        print(f"  Initial: {val_miou[0]:.6f}")
        print(f"  Final:   {val_miou[-1]:.6f}")
        print(f"  Best:    {max(val_miou):.6f}")
    
    if val_pixel_acc:
        print(f"\nValidation Pixel Accuracy:")
        print(f"  Initial: {val_pixel_acc[0]:.6f}")
        print(f"  Final:   {val_pixel_acc[-1]:.6f}")
        print(f"  Best:    {max(val_pixel_acc):.6f}")
    
    print("\n" + "="*60 + "\n")


def main():
    """Main function to load metrics and create plots."""
    metrics_file = resolve_metrics_file()
    
    # Check for alternate metric files
    if len(sys.argv) > 1:
        metrics_file = sys.argv[1]
    
    print("Loading training metrics...")
    metrics = load_metrics(metrics_file)
    
    if metrics is None:
        sys.exit(1)
    
    print(f"Loaded metrics from {metrics_file}")
    print_summary(metrics)
    
    print("Creating plots...")
    plot_metrics(metrics)
    print("\n✓ All plots generated successfully!")


if __name__ == '__main__':
    main()
