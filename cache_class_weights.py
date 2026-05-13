"""
Script to precompute and cache class weights for faster training.
Run once to save weights, then load them in train.py.
"""

import torch
import numpy as np
import pickle
import os
import glob
from tqdm import tqdm
from PIL import Image


def compute_class_weights_cityscapes(segmented_path, num_classes=19, output_path='class_weights_cityscapes.pkl'):
    """
    Efficiently compute class weights for Cityscapes dataset.
    
    Args:
        segmented_path: Path to segmentation masks directory
        num_classes: Number of classes
        output_path: Where to save weights
    """
    print(f"Computing class weights for Cityscapes dataset...")
    
    # Get all mask files
    mask_files = sorted(glob.glob(segmented_path + '/**/*trainIds.png', recursive=True))
    print(f"Found {len(mask_files)} mask files")
    
    class_counts = np.zeros(num_classes)
    
    # Count class occurrences
    for mask_path in tqdm(mask_files, desc="Counting classes"):
        mask = Image.open(mask_path)
        if mask.mode != 'L':
            mask = mask.convert('L')
        mask_array = np.array(mask)
        
        # Count each class (excluding 255 which is ignore index)
        valid_mask = mask_array < num_classes
        mask_filtered = mask_array[valid_mask]
        
        counts = np.bincount(mask_filtered, minlength=num_classes)
        class_counts += counts
    
    # Compute weights (inverse of frequency)
    total_pixels = class_counts.sum()
    class_counts = np.maximum(class_counts, 1e-6)  # Avoid division by zero
    
    # Weight inversely proportional to class frequency (NOT normalized)
    # Higher weight for rare classes, lower for common classes
    class_weights = total_pixels / (len(class_counts) * class_counts)
    # Scale to reasonable range [0.1, 5.0] for better gradient flow
    class_weights = class_weights / class_weights.max() * 5.0
    
    # Convert to tensor
    class_weights_tensor = torch.from_numpy(class_weights).float()
    
    # Save weights
    with open(output_path, 'wb') as f:
        pickle.dump(class_weights_tensor, f)
    
    print(f"\nClass Weights (normalized):")
    for i, w in enumerate(class_weights_tensor):
        print(f"  Class {i}: {w:.4f}")
    
    print(f"\nWeights saved to {output_path}")
    
    return class_weights_tensor


def compute_class_weights_pascal_voc(segmented_path, num_classes=21, output_path='class_weights_pascal_voc.pkl'):
    """
    Efficiently compute class weights for Pascal VOC dataset.
    
    Args:
        segmented_path: Path to segmentation masks directory
        num_classes: Number of classes
        output_path: Where to save weights
    """
    print(f"Computing class weights for Pascal VOC dataset...")
    
    mask_files = sorted(os.listdir(segmented_path))
    print(f"Found {len(mask_files)} mask files")
    
    class_counts = np.zeros(num_classes)
    
    # Count class occurrences
    for mask_path in tqdm(mask_files, desc="Counting classes"):
        full_path = os.path.join(segmented_path, mask_path)
        mask = Image.open(full_path)
        if mask.mode != 'L':
            mask = mask.convert('L')
        mask_array = np.array(mask)
        
        # Count each class (excluding 255 which is ignore index)
        valid_mask = mask_array < num_classes
        mask_filtered = mask_array[valid_mask]
        
        counts = np.bincount(mask_filtered, minlength=num_classes)
        class_counts += counts
    
    # Compute weights
    total_pixels = class_counts.sum()
    class_counts = np.maximum(class_counts, 1e-6)
    
    class_weights = total_pixels / (len(class_counts) * class_counts)
    class_weights = class_weights / class_weights.max() * 5.0
    
    class_weights_tensor = torch.from_numpy(class_weights).float()
    
    # Save weights
    with open(output_path, 'wb') as f:
        pickle.dump(class_weights_tensor, f)
    
    print(f"\nClass Weights (normalized):")
    for i, w in enumerate(class_weights_tensor):
        print(f"  Class {i}: {w:.4f}")
    
    print(f"\nWeights saved to {output_path}")
    
    return class_weights_tensor


def load_class_weights(weight_path):
    """Load pre-computed class weights."""
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Weight file not found: {weight_path}")
    
    with open(weight_path, 'rb') as f:
        weights = pickle.load(f)
    
    return weights


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Precompute class weights')
    parser.add_argument('--dataset', type=str, choices=['cityscapes', 'pascal_voc'],
                        default='cityscapes', help='Dataset type')
    parser.add_argument('--mask_path', type=str, required=True,
                        help='Path to segmentation masks directory')
    parser.add_argument('--num_classes', type=int, default=None,
                        help='Number of classes (auto-detect if not specified)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path')
    
    args = parser.parse_args()
    
    if args.dataset == 'cityscapes':
        num_classes = args.num_classes or 19
        output = args.output or 'class_weights_cityscapes.pkl'
        compute_class_weights_cityscapes(args.mask_path, num_classes, output)
    else:
        num_classes = args.num_classes or 21
        output = args.output or 'class_weights_pascal_voc.pkl'
        compute_class_weights_pascal_voc(args.mask_path, num_classes, output)
