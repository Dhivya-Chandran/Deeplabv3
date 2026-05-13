#!/usr/bin/env python3
"""
Example training configuration with optimizations enabled.
Copy this and adjust paths for your environment.
"""

import argparse
import torch


def get_optimized_config():
    """Create optimized training FLAGS/config."""
    
    parser = argparse.ArgumentParser(description='DeepLabv3 Training with Optimizations')
    
    # ============ DATASET PATHS ============
    parser.add_argument('--dtype', type=str, default='cityscapes',
                        choices=['cityscapes', 'pascal'],
                        help='Dataset type')
    parser.add_argument('--input_path_train', type=str,
                        default='datasets/Cityscapes/leftImg8bit/train',
                        help='Path to training images')
    parser.add_argument('--label_path_train', type=str,
                        default='datasets/Cityscapes/gtFine_trainIds/train',
                        help='Path to training labels')
    parser.add_argument('--input_path_val', type=str, default=None,
                        help='Path to validation images (optional)')
    parser.add_argument('--label_path_val', type=str, default=None,
                        help='Path to validation labels (optional)')
    
    # ============ OPTIMIZATION PARAMETERS (NEW) ============
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size (increased from 4 for faster training)')
    parser.add_argument('--grad_accum_steps', type=int, default=1,
                        help='Gradient accumulation steps to simulate a larger batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of parallel data loading workers (0-8 recommended)')
    parser.add_argument('--prefetch_factor', type=int, default=2,
                        help='Batches to prefetch per worker (try 2-4)')
    parser.add_argument('--use_class_weights', type=bool, default=True,
                        help='Use pre-computed class weights for better training')
    
    # ============ TRAINING HYPERPARAMETERS ============
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        choices=['sgd', 'adam'],
                        help='Optimizer type')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for SGD optimizer')
    parser.add_argument('--scheduler', type=str, default='poly',
                        help='Learning rate scheduler (poly or none)')
    parser.add_argument('--sync_bn', type=bool, default=False,
                        help='Use SyncBatchNorm when distributed training is available')
    
    # ============ MODEL PARAMETERS ============
    parser.add_argument('--num_classes', type=int, default=19,
                        help='Number of semantic classes')
    parser.add_argument('--resize_height', type=int, default=1024,
                        help='Target height for image resizing')
    parser.add_argument('--resize_width', type=int, default=2048,
                        help='Target width for image resizing')
    
    # ============ DATA AUGMENTATION ============
    parser.add_argument('--augment', type=bool, default=True,
                        help='Enable data augmentation (flip, brightness, contrast, saturation)')
    
    # ============ VALIDATION & EARLY STOPPING ============
    parser.add_argument('--eval_every', type=int, default=20,
                        help='Validate every N epochs (0 to disable)')
    parser.add_argument('--early_stop_patience', type=int, default=30,
                        help='Early stopping patience in epochs')
    
    # ============ MISC ============
    parser.add_argument('--cuda', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device: cuda or cpu')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Load checkpoint to resume training')
    
    FLAGS = parser.parse_args()
    
    return FLAGS


def print_config(FLAGS):
    """Pretty print configuration."""
    print("\n" + "="*60)
    print("OPTIMIZED TRAINING CONFIGURATION")
    print("="*60)
    
    print("\n📊 Dataset:")
    print(f"  Type:                {FLAGS.dtype}")
    print(f"  Train images:        {FLAGS.input_path_train}")
    print(f"  Train labels:        {FLAGS.label_path_train}")
    print(f"  Image size:          {FLAGS.resize_width}×{FLAGS.resize_height}")
    
    print("\n⚡ Optimizations Enabled:")
    print(f"  Batch size:          {FLAGS.batch_size} (INCREASED from 4)")
    print(f"  Grad accum steps:    {FLAGS.grad_accum_steps}")
    print(f"  Data workers:        {FLAGS.num_workers} (Multi-threaded loading)")
    print(f"  Prefetch factor:     {FLAGS.prefetch_factor}")
    print(f"  Class weights:       {FLAGS.use_class_weights} (Cached)")
    print(f"  Data augmentation:   {FLAGS.augment}")
    
    print("\n🎯 Training Parameters:")
    print(f"  Epochs:              {FLAGS.epochs}")
    print(f"  Optimizer:           {FLAGS.optimizer}")
    print(f"  Learning rate:       {FLAGS.learning_rate}")
    print(f"  Weight decay:        {FLAGS.weight_decay}")
    print(f"  Momentum:            {FLAGS.momentum}")
    print(f"  Scheduler:           {FLAGS.scheduler}")
    print(f"  Sync BN:             {FLAGS.sync_bn}")
    
    print("\n📈 Validation:")
    print(f"  Eval frequency:      Every {FLAGS.eval_every} epochs")
    print(f"  Early stop patience: {FLAGS.early_stop_patience} epochs")
    
    print("\n💻 Hardware:")
    print(f"  Device:              {FLAGS.cuda}")
    if FLAGS.cuda == 'cuda':
        import torch
        print(f"  GPU(s):              {torch.cuda.get_device_name(0)}")
        print(f"  CUDA memory:         {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    print("\n" + "="*60)
    print("Expected speedup: 2-3x faster training due to optimizations")
    print("="*60 + "\n")


def main():
    FLAGS = get_optimized_config()
    print_config(FLAGS)
    
    # Save config
    import json
    config_dict = vars(FLAGS)
    with open('optimized_config.json', 'w') as f:
        # Convert device to string for serialization
        config_dict_save = {k: str(v) if not isinstance(v, (int, float, bool)) else v 
                            for k, v in config_dict.items()}
        json.dump(config_dict_save, f, indent=2)
    print("✓ Configuration saved to optimized_config.json\n")


if __name__ == '__main__':
    main()
