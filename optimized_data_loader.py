"""
Optimized DataLoader for Cityscapes and Pascal VOC datasets using PyTorch DataLoader.
Supports multi-threaded data loading with prefetching for faster training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image, ImageOps
import glob
import os

from cityscapes_preprocess import normalize_rgb_image


class CityscapesDataset(Dataset):
    """
    PyTorch Dataset for Cityscapes semantic segmentation.
    Supports multi-worker data loading for faster training.
    """
    
    def __init__(self, input_path, segmented_path, target_height=1024, target_width=2048, 
                 augment=True, normalize=True, imagenet_normalize=False):
        """
        Args:
            input_path: Path to directory with images
            segmented_path: Path to directory with segmentation masks
            target_height: Target height for resizing
            target_width: Target width for resizing
            augment: Whether to apply data augmentation
            normalize: Whether to normalize images to [0, 1]
            imagenet_normalize: Whether to additionally apply ImageNet mean/std normalization
        """
        self.input_path = input_path
        self.segmented_path = segmented_path
        self.target_height = target_height
        self.target_width = target_width
        self.augment = augment
        self.normalize = normalize
        self.imagenet_normalize = imagenet_normalize
        
        # Efficiently get file list once during init
        self.img_files = sorted(
            glob.glob(input_path + '/**/*.png', recursive=True),
            key=lambda x: int(os.path.basename(x).split('_')[1] + 
                            os.path.basename(x).split('_')[2])
        )
        
        self.mask_files = sorted(
            glob.glob(segmented_path + '/**/*trainIds.png', recursive=True),
            key=lambda x: int(os.path.basename(x).split('_')[1] + 
                            os.path.basename(x).split('_')[2])
        )
        
        assert len(self.img_files) == len(self.mask_files), \
            f"Mismatch: {len(self.img_files)} images vs {len(self.mask_files)} masks"

        self._validate_mask_encoding(num_samples=8)

    def _validate_mask_encoding(self, num_samples=8):
        """Fail fast if masks are not Cityscapes trainIds (0..18,255)."""
        if len(self.mask_files) == 0:
            raise RuntimeError(
                f"No trainIds masks found under: {self.segmented_path}. "
                "Expected files ending with *trainIds.png"
            )

        sample_count = min(num_samples, len(self.mask_files))
        sample_idxs = np.linspace(0, len(self.mask_files) - 1, sample_count, dtype=int)

        allowed = set(range(19))
        allowed.add(255)

        bad_values = set()
        offending_file = None

        for idx in sample_idxs:
            mp = self.mask_files[int(idx)]
            m = np.array(Image.open(mp), dtype=np.uint8)
            uvals = set(np.unique(m).tolist())
            invalid = {v for v in uvals if v not in allowed}
            if invalid:
                bad_values.update(invalid)
                offending_file = mp
                break

        if bad_values:
            vals = sorted(int(v) for v in bad_values)
            raise RuntimeError(
                "Detected non-trainIds mask values in Cityscapes labels. "
                f"Found values {vals} in sample mask {offending_file}. "
                "Use gtFine_trainIds masks (values 0..18 and 255)."
            )
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        """Load and return a single sample."""
        img_path = self.img_files[idx]
        mask_path = self.mask_files[idx]
        
        # Load image
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Load mask
        mask = Image.open(mask_path)
        if mask.mode != 'L':
            mask = mask.convert('L')
        
        # Apply augmentation before resizing
        if self.augment:
            img, mask = self._augment(img, mask, crop_size=(self.target_height, self.target_width))
        
        # Evaluation path keeps a deterministic resize to the requested size.
        if not self.augment:
            img = img.resize((self.target_width, self.target_height), Image.BILINEAR)
            mask = mask.resize((self.target_width, self.target_height), Image.NEAREST)
        
        # Convert to tensors
        img_array = np.array(img, dtype=np.float32)
        if self.normalize or self.imagenet_normalize:
            img_array = normalize_rgb_image(img_array, imagenet_normalize=self.imagenet_normalize)
        
        # Convert to tensor and permute to (C, H, W)
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        
        mask_array = np.array(mask, dtype=np.long)
        mask_tensor = torch.from_numpy(mask_array)
        
        return img_tensor, mask_tensor
    
    @staticmethod
    def _augment(img_pil, mask_pil, prob_hflip=0.5, scale_range=(0.5, 2.0), crop_size=(769, 769)):
        """Apply synchronized scale, flip, pad, and random crop to image and mask."""
        # Random scaling (0.5–2.0) - DeepLabv3 paper augmentation
        scale_factor = np.random.uniform(*scale_range)
        new_width = int(img_pil.width * scale_factor)
        new_height = int(img_pil.height * scale_factor)
        img_pil = img_pil.resize((new_width, new_height), Image.BILINEAR)
        mask_pil = mask_pil.resize((new_width, new_height), Image.NEAREST)
        
        # Horizontal flip
        if np.random.rand() < prob_hflip:
            img_pil = ImageOps.mirror(img_pil)
            mask_pil = ImageOps.mirror(mask_pil)
        
        # Color augmentation (brightness and contrast)
        if np.random.rand() < 0.5:
            brightness = np.random.uniform(0.85, 1.15)
            img_array = np.array(img_pil, dtype=np.float32)
            img_array = np.clip(img_array * brightness, 0, 255).astype(np.uint8)
            img_pil = Image.fromarray(img_array)
        
        if np.random.rand() < 0.5:
            contrast = np.random.uniform(0.85, 1.15)
            img_array = np.array(img_pil, dtype=np.float32)
            img_array = (img_array - 128) * contrast + 128
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            img_pil = Image.fromarray(img_array)

        crop_h, crop_w = crop_size
        pad_w = max(0, crop_w - img_pil.width)
        pad_h = max(0, crop_h - img_pil.height)
        if pad_w > 0 or pad_h > 0:
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            img_pil = ImageOps.expand(img_pil, border=(pad_left, pad_top, pad_right, pad_bottom), fill=0)
            mask_pil = ImageOps.expand(mask_pil, border=(pad_left, pad_top, pad_right, pad_bottom), fill=255)

        max_left = max(0, img_pil.width - crop_w)
        max_top = max(0, img_pil.height - crop_h)
        left = np.random.randint(0, max_left + 1) if max_left > 0 else 0
        top = np.random.randint(0, max_top + 1) if max_top > 0 else 0
        img_pil = img_pil.crop((left, top, left + crop_w, top + crop_h))
        mask_pil = mask_pil.crop((left, top, left + crop_w, top + crop_h))

        return img_pil, mask_pil


def get_cityscapes_loader(input_path, segmented_path, batch_size, target_height=1024,
                          target_width=2048, augment=True, num_workers=4, shuffle=True,
                          imagenet_normalize=False):
    """
    Create optimized DataLoader for Cityscapes dataset.
    
    Args:
        input_path: Path to images directory
        segmented_path: Path to masks directory
        batch_size: Batch size
        target_height: Target image height
        target_width: Target image width
        augment: Whether to apply augmentation
        num_workers: Number of worker processes (2-8 recommended)
        shuffle: Whether to shuffle data
    
    Returns:
        DataLoader instance
    """
    dataset = CityscapesDataset(
        input_path, segmented_path,
        target_height=target_height,
        target_width=target_width,
        augment=augment,
        imagenet_normalize=imagenet_normalize
    )
    
    # Only use prefetch_factor and persistent_workers if num_workers > 0
    loader_kwargs = {
        'dataset': dataset,
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'pin_memory': True,
    }
    
    if num_workers > 0:
        loader_kwargs['prefetch_factor'] = 2  # Prefetch 2 batches per worker
        loader_kwargs['persistent_workers'] = True  # Keep workers alive between epochs
    
    loader = DataLoader(**loader_kwargs)
    
    return loader
