import torch
import torch.nn as nn
import torch.optim as optim
from utils import *
from optimized_data_loader import get_cityscapes_loader
from cache_class_weights import load_class_weights
import copy
from test import predict
from models.deeplabv3 import DeepLabv3
from tqdm import tqdm
import glob
import os
import numpy as np
import pickle
import json


class FocalLoss(nn.Module):
    """Focal Loss for dense object detection / semantic segmentation.
    
    Addresses class imbalance by down-weighting easy (high-confidence) examples
    and focusing on hard (low-confidence) examples.
    
    Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Reference: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    """
    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=255, weight=None):
        """Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor in range (0,1) to balance positive vs negative examples
            gamma: Exponent of the modulating factor (1 - p_t)^gamma. Higher gamma = focus on hard examples
            ignore_index: Pixel label to ignore (typically 255 for void/ignore class)
            weight: Manual rescaling weight given to each class (same as CrossEntropyLoss)
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.weight = weight
    
    def forward(self, logits, targets):
        """Compute focal loss.
        
        Args:
            logits: (B, C, H, W) model predictions
            targets: (B, H, W) ground truth labels
            
        Returns:
            scalar focal loss
        """
        # Get class probabilities from logits
        p = torch.softmax(logits, dim=1)  # (B, C, H, W)
        
        # Compute cross-entropy (PyTorch handles ignore_index correctly)
        ce_loss = nn.functional.cross_entropy(logits, targets, ignore_index=self.ignore_index, 
                                              weight=self.weight, reduction='none')
        
        # Mask out ignored pixels BEFORE gathering to avoid out-of-bounds indices
        valid_mask = (targets != self.ignore_index).float()  # (B, H, W)
        
        # Clamp targets to valid range [0, num_classes-1] for gather operation
        # Invalid indices will be masked out anyway
        targets_clamped = targets.clamp(min=0, max=p.size(1) - 1)
        
        # Get the probability of the target class (p_t)
        # For each pixel, get the probability of the true class
        p_t = p.gather(1, targets_clamped.unsqueeze(1)).squeeze(1)  # (B, H, W)
        
        # Ensure p_t is in valid range to avoid NaN in pow operation
        p_t = torch.clamp(p_t, min=1e-7, max=1.0 - 1e-7)
        
        # Apply focal term: (1 - p_t)^gamma
        focal_term = (1.0 - p_t).pow(self.gamma)  # (B, H, W)
        
        # Apply alpha weighting (balance positive/negative)
        focal_loss = self.alpha * focal_term * ce_loss
        
        # Mask and average (only sum over valid pixels)
        focal_loss = (focal_loss * valid_mask).sum() / (valid_mask.sum() + 1e-8)
        
        return focal_loss


class DiceLoss(nn.Module):
    """Dice loss for semantic segmentation.
    
    Directly optimizes IoU by computing: 1 - (2|X∩Y|)/(|X|+|Y|)
    Particularly effective for boundary and minority class refinement.
    """
    def __init__(self, ignore_index=255, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth
    
    def forward(self, logits, targets):
        """Compute Dice loss.
        
        Args:
            logits: (B, C, H, W) model output
            targets: (B, H, W) ground truth labels
        
        Returns:
            scalar loss
        """
        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=1)
        
        # Flatten spatial dims
        B, C, H, W = probs.shape
        probs = probs.view(B, C, -1)  # (B, C, H*W)
        targets_one_hot = torch.zeros_like(probs)
        
        # Create one-hot target
        valid_mask = (targets.view(B, -1) != self.ignore_index)
        for b in range(B):
            for c in range(C):
                targets_one_hot[b, c] = (targets[b].view(-1) == c).float() * valid_mask[b].float()
        
        # Compute Dice per class
        intersection = (probs * targets_one_hot).sum(dim=2)
        union = probs.sum(dim=2) + targets_one_hot.sum(dim=2)
        
        # Avoid division by zero; only compute on valid classes
        dice_per_class = (2 * intersection + self.smooth) / (union + self.smooth)
        dice = dice_per_class[union > 0].mean()
        
        return 1.0 - dice


def _flag_enabled(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {'false', '0', 'none', 'off', 'no', ''}


def train(FLAGS):

    def _checkpoint_stem(backbone_name: str) -> str:
        backbone_clean = str(backbone_name).strip().lower() or 'model'
        return backbone_clean.replace('-', '_')

    def _update_confmat(confmat: np.ndarray, preds: torch.Tensor, targets: torch.Tensor, ignore_index: int = 255):
        """Update confusion matrix using valid pixels only."""
        valid = (targets != ignore_index)
        if not torch.any(valid):
            return confmat

        t = targets[valid].view(-1).to(torch.int64)
        p = preds[valid].view(-1).to(torch.int64)
        idx = t * nc + p
        binc = torch.bincount(idx, minlength=nc * nc).reshape(nc, nc)
        confmat += binc.detach().cpu().numpy()
        return confmat

    def _miou_from_confmat(confmat: np.ndarray) -> float:
        inter = np.diag(confmat).astype(np.float64)
        union = confmat.sum(axis=1) + confmat.sum(axis=0) - inter
        valid = union > 0
        if not np.any(valid):
            return float('nan')
        return float(np.mean(inter[valid] / (union[valid] + 1e-12)))

    # Hyperparameters
    device = FLAGS.cuda
    batch_size = FLAGS.batch_size
    epochs = FLAGS.epochs
    lr = FLAGS.learning_rate if FLAGS.learning_rate != 0.007 else 0.005  # Reduced from 0.007 to 0.005
    nc = FLAGS.num_classes
    wd = FLAGS.weight_decay

    ip = FLAGS.input_path_train
    lp = FLAGS.label_path_train

    H = FLAGS.resize_height
    W = FLAGS.resize_width

    dtype = FLAGS.dtype
    sched = _flag_enabled(getattr(FLAGS, 'scheduler', 'poly'))
    
    # Get optimizer type and momentum
    optim_type = getattr(FLAGS, 'optimizer', 'sgd').lower()
    momentum = getattr(FLAGS, 'momentum', 0.9)
    use_class_weights = getattr(FLAGS, 'use_class_weights', True)
    early_stop_patience = getattr(FLAGS, 'early_stop_patience', 20)  # Reduced from 30 to 20 for faster convergence
    augment = getattr(FLAGS, 'augment', True)
    num_workers = getattr(FLAGS, 'num_workers', 4)  # Multi-threaded data loading
    grad_accum_steps = max(1, int(getattr(FLAGS, 'grad_accum_steps', 1)))
    use_sync_bn = _flag_enabled(getattr(FLAGS, 'sync_bn', False))
    imagenet_normalize = _flag_enabled(getattr(FLAGS, 'imagenet_normalize', False))
    backbone = str(getattr(FLAGS, 'backbone', 'resnet50')).strip().lower()
    pretrained_backbone = _flag_enabled(getattr(FLAGS, 'pretrained_backbone', True))
    output_stride = int(getattr(FLAGS, 'output_stride', 16))
    checkpoint_stem = _checkpoint_stem(backbone)
    epoch_ckpt_pattern = f'checkpoint_epoch_{checkpoint_stem}_{{epoch}}.pth'
    best_miou_ckpt_path = f'best_miou_model_{checkpoint_stem}.pth'
    best_loss_ckpt_path = f'best_model_{checkpoint_stem}.pth'

    # Cityscapes dataset
    train_samples = len(glob.glob(ip + '/**/*.png', recursive=True))

    print('[INFO]Hyperparameters Loaded')
    print(f'[INFO]epochs={epochs} | batch_size={batch_size} | grad_accum_steps={grad_accum_steps} | '
          f'effective_batch_size={batch_size * grad_accum_steps} | lr={lr} | optimizer={optim_type} | '
          f'num_workers={num_workers}')
    print(f"[INFO]Input normalization: {'ImageNet mean/std' if imagenet_normalize else '[0,1] scaling only'}")
    print(f"[INFO]Backbone: {backbone} | pretrained_backbone={pretrained_backbone}")
    print(f"[INFO]Output stride: {output_stride}")

    # Model (DeepLabv3 with configurable output stride for Cityscapes)
    model = DeepLabv3(
        nc,
        dropout_rate=0.0,
        output_stride=output_stride,
        backbone=backbone,
        pretrained_backbone=pretrained_backbone,
    )
    if use_sync_bn:
        if device.type == 'cuda' and torch.distributed.is_available() and torch.distributed.is_initialized():
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            print('[INFO]Converted BatchNorm layers to SyncBatchNorm for distributed training')
        else:
            print('[WARN]--sync-bn requested, but distributed multi-GPU training is not active; keeping BatchNorm')
    model = model.to(device)

    # Compute class weights if requested
    class_weights = None
    if use_class_weights:
        print('[INFO]Loading class weights...')
        weight_path = f'class_weights_cityscapes.pkl'
        
        if os.path.exists(weight_path):
            # Load pre-computed weights
            try:
                class_weights = load_class_weights(weight_path)
                class_weights = class_weights.to(device)
                print(f'[INFO]Class weights loaded from cache: {class_weights[:5]}...')
            except Exception as e:
                print(f'[WARN]Failed to load cached weights: {e}')
                class_weights = None
        else:
            # If no cache, create approximate weights from first batch
            print(f'[INFO]Cache not found. Computing from first batch...')
            print(f'[INFO]To avoid this in future runs, execute:')
            print(f'  python cache_class_weights.py --dataset cityscapes --mask_path {lp}')
            
            try:
                temp_loader = get_cityscapes_loader(ip, lp, batch_size * 4, H, W, 
                                                    augment=False, num_workers=2,
                                                    imagenet_normalize=imagenet_normalize)
                
                _, labels = next(iter(temp_loader))
                all_labels = labels.view(-1).cpu().numpy()
                valid_labels = all_labels[(all_labels >= 0) & (all_labels < nc)]
                class_counts = np.bincount(valid_labels, minlength=nc)
                valid_classes = class_counts > 0
                class_weights_np = np.zeros(nc)
                class_weights_np[valid_classes] = 1.0 / (class_counts[valid_classes] / class_counts[valid_classes].sum() + 1e-6)
                class_weights_np = class_weights_np / class_weights_np.sum()
                class_weights = torch.from_numpy(class_weights_np).float().to(device)
                print(f'[INFO]Class weights computed: {class_weights[:5]}...')
            except Exception as e:
                print(f'[WARN]Failed to compute class weights: {e}')
                class_weights = None

    # Loss & Optimizer
    criterion_focal = FocalLoss(alpha=0.25, gamma=2.0, ignore_index=255, weight=class_weights)
    criterion_dice = DiceLoss(ignore_index=255)
    print('[INFO] Using combined loss: Focal + Dice (both equally weighted)')
    print('[INFO] Focal Loss Parameters: alpha=0.25, gamma=2.0 (focuses on hard examples)')
    
    if optim_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=True, weight_decay=wd)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # Optimized Data Loader with multi-threading
    print(f'[INFO]Loading Cityscapes dataset with {num_workers} workers...')
    train_loader = get_cityscapes_loader(ip, lp, batch_size, H, W, 
                                        augment=augment, num_workers=num_workers, shuffle=True,
                                        imagenet_normalize=imagenet_normalize)

    # Validation loader (no augmentation) for comparable validation loss/accuracy
    eval_every = getattr(FLAGS, 'eval_every', None)
    val_loader = None
    if eval_every is not None and eval_every > 0 and getattr(FLAGS, 'input_path_val', None) and getattr(FLAGS, 'label_path_val', None):
        print(f'[INFO]Loading validation dataset with {num_workers} workers...')
        val_loader = get_cityscapes_loader(
            FLAGS.input_path_val,
            FLAGS.label_path_val,
            batch_size,
            H,
            W,
            augment=False,
            num_workers=num_workers,
            shuffle=False,
            imagenet_normalize=imagenet_normalize
        )

    bc_train = len(train_loader)
    bc_val = len(val_loader) if val_loader is not None else 0

    # Scheduler: optional polynomial decay with warmup
    scheduler = None
    if sched:
        def poly_lr_lambda(epoch):
            """Polynomial decay with warmup (slower decay for better late-stage learning)"""
            warmup_epochs = max(1, epochs // 20)  # 5% warmup
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs  # Linear warmup
            else:
                # Slower decay (power 0.95) keeps LR higher longer for boundary refinement
                return (1.0 - (epoch - warmup_epochs) / (epochs - warmup_epochs)) ** 0.95
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=poly_lr_lambda)

    print('\n[INFO]Starting Training...\n')

    best_loss = float('inf')
    best_miou = 0.0
    epochs_without_improvement = 0
    
    # Metrics tracking for plotting
    metrics = {
        'epochs': [],
        'train_loss': [],
        'train_pixel_accuracy': [],
        'train_miou': [],
        'val_epochs': [],
        'val_loss': [],
        'val_miou': [],
        'val_pixel_accuracy': []
    }

    for e in range(1, epochs + 1):

        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_confmat = np.zeros((nc, nc), dtype=np.int64)

        print('-' * 20, f'Epoch {e}/{epochs}', '-' * 20)
        if scheduler is not None:
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = optimizer.param_groups[0]['lr']
        print(f'Learning rate: {current_lr:.6e}')

        # Iterate through DataLoader instead of using generator
        optimizer.zero_grad(set_to_none=True)
        for step, (X_batch, mask_batch) in enumerate(tqdm(train_loader, total=bc_train), start=1):

            if X_batch.size(0) < 1:
                continue

            X_batch = X_batch.to(device)
            mask_batch = mask_batch.to(device)

            outputs = model(X_batch.float())
            loss_focal = criterion_focal(outputs, mask_batch.long())
            loss_dice = criterion_dice(outputs, mask_batch.long())
            loss = loss_focal + loss_dice  # Combined loss: Focal + Dice
            (loss / grad_accum_steps).backward()

            if step % grad_accum_steps == 0 or step == bc_train:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            train_loss += loss.item()

            # Pixel accuracy on valid labels only (ignore_index=255)
            with torch.no_grad():
                preds = outputs.argmax(dim=1)
                valid_mask = (mask_batch != 255)
                train_correct += (preds[valid_mask] == mask_batch[valid_mask]).sum().item()
                train_total += valid_mask.sum().item()
                train_confmat = _update_confmat(train_confmat, preds, mask_batch, ignore_index=255)

        # Print every epoch
        avg_train_loss = train_loss / bc_train
        train_pixel_acc = 100.0 * train_correct / max(1, train_total)
        train_miou = _miou_from_confmat(train_confmat)
        print(f'Epoch {e}/{epochs}  |  Loss: {avg_train_loss:.6f}  |  Pixel Acc: {train_pixel_acc:.2f}%  |  mIoU: {train_miou:.6f}')

        # Save best loss model
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            print(f'  Best loss model saved (Loss: {best_loss:.6f})')

        # --- Run validation and produce plots if requested ---
        val_miou = None
        val_loss = None
        val_pixel_acc = None
        
        if eval_every is not None and eval_every > 0 and (e % eval_every == 0):
            # Compute validation loss and pixel accuracy directly for metric consistency
            if val_loader is not None:
                model.eval()
                val_loss_sum = 0.0
                val_correct = 0
                val_total = 0
                val_confmat = np.zeros((nc, nc), dtype=np.int64)
                with torch.no_grad():
                    for X_val, mask_val in tqdm(val_loader, total=bc_val, desc='Validation', leave=False):
                        X_val = X_val.to(device)
                        mask_val = mask_val.to(device)

                        val_outputs = model(X_val.float())
                        v_loss_focal = criterion_focal(val_outputs, mask_val.long())
                        v_loss_dice = criterion_dice(val_outputs, mask_val.long())
                        v_loss = v_loss_focal + v_loss_dice
                        val_loss_sum += v_loss.item()

                        val_preds = val_outputs.argmax(dim=1)
                        valid_mask = (mask_val != 255)
                        val_correct += (val_preds[valid_mask] == mask_val[valid_mask]).sum().item()
                        val_total += valid_mask.sum().item()
                        val_confmat = _update_confmat(val_confmat, val_preds, mask_val, ignore_index=255)

                val_loss = val_loss_sum / max(1, bc_val)
                val_pixel_acc = 100.0 * val_correct / max(1, val_total)
                val_miou = _miou_from_confmat(val_confmat)
                print(f"[INFO] Validation Loss (avg): {val_loss:.6f}")
                print(f"[INFO] Validation Pixel Accuracy: {val_pixel_acc:.2f}%")
                print(f"[INFO] Validation mIoU (loop): {val_miou:.6f}")

            # Save current checkpoint for evaluation
            ckpt_path = epoch_ckpt_pattern.format(epoch=e)
            torch.save({
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
                'backbone': backbone,
                'output_stride': output_stride,
                'imagenet_normalize': imagenet_normalize
            }, ckpt_path)

            # Prepare FLAGS copy for prediction
            FLAGS_val = copy.deepcopy(FLAGS)
            FLAGS_val.image_path = getattr(FLAGS, 'input_path_val', None)
            FLAGS_val.label_path_val = getattr(FLAGS, 'label_path_val', None)
            FLAGS_val.m = ckpt_path
            FLAGS_val.out_dir = os.path.join('preds_val', f'epoch_{e}')
            os.makedirs(FLAGS_val.out_dir, exist_ok=True)

            try:
                print(f"[INFO] Running validation for epoch {e}...")
                predict(FLAGS_val)
                print(f"[INFO] Validation complete for epoch {e}")
            except Exception as ex:
                print(f"[WARN] Validation failed at epoch {e}: {ex}")

            model.train()
        
        # Save metrics for this epoch
        metrics['epochs'].append(e)
        metrics['train_loss'].append(avg_train_loss)
        metrics['train_pixel_accuracy'].append(train_pixel_acc)
        metrics['train_miou'].append(train_miou)
        if val_loss is not None:
            metrics['val_epochs'].append(e)
            metrics['val_loss'].append(val_loss)
        if val_miou is not None:
            metrics['val_miou'].append(val_miou)
        if val_pixel_acc is not None:
            metrics['val_pixel_accuracy'].append(val_pixel_acc)

        # Early stopping based on mIoU
        if val_miou is not None:
            if val_miou > best_miou:
                best_miou = val_miou
                epochs_without_improvement = 0
                # Save best mIoU model
                torch.save({
                    'epoch': e,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_miou': best_miou,
                    'backbone': backbone,
                    'output_stride': output_stride,
                    'imagenet_normalize': imagenet_normalize
                }, best_miou_ckpt_path)
                print(f' Best mIoU model saved (mIoU: {best_miou:.6f})')
            else:
                epochs_without_improvement += eval_every
                print(f'[INFO] No improvement for {epochs_without_improvement} epochs (patience: {early_stop_patience})')
                
                if epochs_without_improvement >= early_stop_patience:
                    print(f'\n[INFO] Early stopping triggered after {e} epochs')
                    break
        else:
            # Also save best loss model
            if avg_train_loss < best_loss:
                torch.save({
                    'epoch': e,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_loss': best_loss,
                    'backbone': backbone,
                    'output_stride': output_stride,
                    'imagenet_normalize': imagenet_normalize
                }, best_loss_ckpt_path)

        # Step scheduler after epoch
        if scheduler is not None:
            scheduler.step()

        print()

    print("\n" + "="*50)
    print("Training Complete!")
    print(f"Best Loss Achieved: {best_loss:.6f}")
    if best_miou > 0:
        print(f"Best mIoU Achieved: {best_miou:.6f}")
    print(f"Saved files: {best_loss_ckpt_path}, {best_miou_ckpt_path}")
    print("="*50 + "\n")
    
    # Save metrics to JSON for plotting
    metrics_path = 'training_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Training metrics saved to {metrics_path}")
