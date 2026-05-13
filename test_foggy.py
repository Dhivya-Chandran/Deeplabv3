#!/usr/bin/env python3
"""Run DeepLabv3 inference on Cityscapes Foggy images.

This script supports foggy-domain train/val/test splits and saves per-image
predictions while preserving city folders.
"""

from __future__ import annotations

import argparse
import glob
import os
import random
from pathlib import Path
from typing import Iterable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from cityscapes_preprocess import normalize_rgb_image, resize_rgb, save_rgb_png
from models.deeplabv3 import DeepLabv3


def _flag_enabled(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"false", "0", "none", "off", "no", ""}


def _decode_cityscapes_trainids(mask: np.ndarray, ignore_index: int = 255) -> np.ndarray:
    colors = np.array([
        (128, 64, 128),
        (244, 35, 232),
        (70, 70, 70),
        (102, 102, 156),
        (190, 153, 153),
        (153, 153, 153),
        (250, 170, 30),
        (220, 220, 0),
        (107, 142, 35),
        (152, 251, 152),
        (70, 130, 180),
        (220, 20, 60),
        (255, 0, 0),
        (0, 0, 142),
        (0, 0, 70),
        (0, 60, 100),
        (0, 80, 100),
        (0, 0, 230),
        (119, 11, 32),
    ], dtype=np.uint8)

    rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for cls_id in range(colors.shape[0]):
        rgb[mask == cls_id] = colors[cls_id]
    rgb[mask == ignore_index] = np.array([255, 255, 255], dtype=np.uint8)
    return rgb


def _infer_num_classes_from_state(state_dict):
    if not isinstance(state_dict, dict):
        return None

    for key in ("conv.weight", "classifier.weight"):
        tensor = state_dict.get(key)
        if isinstance(tensor, torch.Tensor) and tensor.ndim == 4:
            return int(tensor.shape[0])

    for key in ("conv.bias", "classifier.bias"):
        tensor = state_dict.get(key)
        if isinstance(tensor, torch.Tensor) and tensor.ndim == 1:
            return int(tensor.shape[0])

    return None


def _infer_backbone_from_state(state_dict):
    if not isinstance(state_dict, dict):
        return None

    max_block_idx = -1
    for key in state_dict.keys():
        if not isinstance(key, str) or not key.startswith("layer3."):
            continue
        parts = key.split(".")
        if len(parts) >= 3 and parts[1].isdigit():
            max_block_idx = max(max_block_idx, int(parts[1]))

    if max_block_idx >= 22:
        return "resnet101"
    if max_block_idx >= 5:
        return "resnet50"
    return None


def _to_chw_tensor(img_hwc: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(img_hwc).unsqueeze(0).float()
    return tensor.permute(0, 3, 1, 2).contiguous()


def _load_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def _load_mask(path: str) -> np.ndarray:
    return np.array(Image.open(path))


def _normalize_input_tensor(img_hwc: np.ndarray, device: torch.device, use_imagenet_stats: bool = False) -> torch.Tensor:
    normalized = normalize_rgb_image(img_hwc, imagenet_normalize=use_imagenet_stats)
    return _to_chw_tensor(normalized).to(device)


@torch.no_grad()
def _forward_logits(model, device, img_hwc: np.ndarray, use_imagenet_stats: bool = False) -> torch.Tensor:
    x = _normalize_input_tensor(img_hwc, device, use_imagenet_stats=use_imagenet_stats)
    return model(x)


def _parse_scales(scales: Optional[Iterable[float]]) -> list[float]:
    if scales is None:
        return [1.0]
    parsed = [float(scale) for scale in scales]
    return parsed if parsed else [1.0]


def _aggregate_logits_with_tta(
    model,
    device,
    image_hwc: np.ndarray,
    base_h: int,
    base_w: int,
    multi_scale: bool,
    flip_test: bool,
    scales: Optional[Iterable[float]],
    use_imagenet_stats: bool,
) -> torch.Tensor:
    scale_values = _parse_scales(scales) if multi_scale else [1.0]

    logits_sum = None
    num_votes = 0

    for scale in scale_values:
        scaled_h = max(1, int(round(base_h * scale)))
        scaled_w = max(1, int(round(base_w * scale)))
        scaled_img = resize_rgb(image_hwc, scaled_w, scaled_h)

        logits = _forward_logits(model, device, scaled_img, use_imagenet_stats=use_imagenet_stats)
        logits = F.interpolate(logits, size=(base_h, base_w), mode="bilinear", align_corners=False)

        logits_sum = logits if logits_sum is None else logits_sum + logits
        num_votes += 1

        if flip_test:
            flipped_img = np.ascontiguousarray(np.fliplr(scaled_img))
            flip_logits = _forward_logits(model, device, flipped_img, use_imagenet_stats=use_imagenet_stats)
            flip_logits = torch.flip(flip_logits, dims=[3])
            flip_logits = F.interpolate(flip_logits, size=(base_h, base_w), mode="bilinear", align_corners=False)

            logits_sum = logits_sum + flip_logits
            num_votes += 1

    return logits_sum / max(1, num_votes)


def _find_foggy_images(input_root: str) -> list[str]:
    patterns = [
        os.path.join(input_root, "*", "*_leftImg8bit_foggy*.png"),
        os.path.join(input_root, "*", "*_leftImg8bit*.png"),
    ]

    images = []
    for pattern in patterns:
        images.extend(glob.glob(pattern))

    images = sorted(set(images))
    if not images:
        raise RuntimeError(f"No foggy images found under: {input_root}")
    return images


def _infer_default_gt_root(input_root: str) -> Optional[str]:
    """Infer default GT root from the foggy split directory name."""
    split = os.path.basename(os.path.normpath(input_root))
    candidate_roots = [
        os.path.join("datasets", "Cityscapes", "gtFine_trainIds", split),
        os.path.join("datasets", "Cityscapes", "gtFine", split),
        os.path.join("DeepLabv3", "datasets", "Cityscapes", "gtFine_trainIds", split),
        os.path.join("DeepLabv3", "datasets", "Cityscapes", "gtFine", split),
    ]
    for candidate in candidate_roots:
        if os.path.isdir(candidate):
            return candidate
    return None


def _prediction_stem(file_name: str) -> str:
    stem = file_name
    if stem.endswith(".png"):
        stem = stem[:-4]
    if stem.endswith("_leftImg8bit"):
        stem = stem[:-12]
    if "_leftImg8bit_foggy" in stem:
        stem = stem.split("_leftImg8bit_foggy", 1)[0]
    return stem


def _save_mask_png(path: str, mask: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8), mode="L").save(path)


def _save_random_samples_grid(samples, out_dir: str) -> bool:
    """Save Input + Prediction grid for sampled foggy images."""
    n = len(samples)
    if n == 0:
        return False

    plt.figure(figsize=(8, 4 * n))
    for i, (img, pred_color, name) in enumerate(samples):
        r = i * 2

        ax1 = plt.subplot(n, 2, r + 1)
        ax1.axis("off")
        ax1.set_title(f"{name}\nInput")
        ax1.imshow(img)

        ax2 = plt.subplot(n, 2, r + 2)
        ax2.axis("off")
        ax2.set_title("Prediction")
        ax2.imshow(pred_color)

    plt.savefig(os.path.join(out_dir, "foggy_random_samples.png"), bbox_inches="tight")
    plt.close()
    return True


def _save_random_samples_with_gt_grid(samples, out_dir: str) -> bool:
    """Save Input + GT + Prediction grid for sampled foggy images when GT is available."""
    n = len(samples)
    if n == 0:
        return False

    plt.figure(figsize=(12, 4 * n))
    for i, (img, gt_color, pred_color, name) in enumerate(samples):
        r = i * 3

        ax1 = plt.subplot(n, 3, r + 1)
        ax1.axis("off")
        ax1.set_title(f"{name}\nInput")
        ax1.imshow(img)

        ax2 = plt.subplot(n, 3, r + 2)
        ax2.axis("off")
        ax2.set_title("Ground Truth")
        ax2.imshow(gt_color)

        ax3 = plt.subplot(n, 3, r + 3)
        ax3.axis("off")
        ax3.set_title("Prediction")
        ax3.imshow(pred_color)

    plt.savefig(os.path.join(out_dir, "foggy_random_samples_gt.png"), bbox_inches="tight")
    plt.close()
    return True


def _pixel_accuracy(pred: np.ndarray, gt: np.ndarray, ignore_index: int = 255) -> float:
    valid = (gt != ignore_index)
    if valid.sum() == 0:
        return float("nan")
    return float((pred[valid] == gt[valid]).mean())


def _safe_nanmean(values) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    valid = ~np.isnan(arr)
    if not np.any(valid):
        return float("nan")
    return float(arr[valid].mean())


def _format_metric(value: float) -> str:
    if np.isnan(value):
        return "nan"
    return f"{value:.6f}"


def _save_curves(losses, accs, out_dir: str) -> bool:
    xs = np.arange(1, len(losses) + 1)
    if len(losses) == 0:
        return False

    mean_loss = _safe_nanmean(losses)
    mean_acc = _safe_nanmean(accs)
    if np.isnan(mean_loss) or np.isnan(mean_acc):
        return False

    plt.figure()
    plt.plot(xs, losses, marker='o')
    plt.axhline(mean_loss, color='r', linestyle='--', label=f"Mean: {mean_loss:.4f}")
    plt.title("Validation Loss per Image")
    plt.xlabel("Image index")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "val_loss_curve_1.png"), bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(xs, accs, marker='o')
    plt.axhline(mean_acc, color='r', linestyle='--', label=f"Mean: {mean_acc:.4f}")
    plt.title("Validation Pixel Accuracy per Image")
    plt.xlabel("Image index")
    plt.ylabel("Pixel Accuracy")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "val_acc_curve_1.png"), bbox_inches="tight")
    plt.close()
    return True


def _resolve_gt_path(image_path: str, input_root: str, gt_root: str) -> Optional[str]:
    """Resolve GT path by matching city folder and Cityscapes base name."""
    city = os.path.basename(os.path.dirname(image_path))
    file_name = os.path.basename(image_path)
    stem = _prediction_stem(file_name)

    candidates = [
        os.path.join(gt_root, city, f"{stem}_gtFine_trainIds.png"),
        os.path.join(gt_root, city, f"{stem}_gtFine_labelTrainIds.png"),
        os.path.join(gt_root, city, f"{stem}_gtFine_labelIds.png"),
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def run_foggy_test(args) -> None:
    if not os.path.isfile(args.model_path):
        raise RuntimeError(f"Checkpoint not found: {args.model_path}")
    if not os.path.isdir(args.input_root):
        raise RuntimeError(f"Foggy input directory not found: {args.input_root}")

    device = torch.device("cuda:0" if torch.cuda.is_available() and _flag_enabled(args.cuda) else "cpu")

    checkpoint = torch.load(args.model_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint

    model_num_classes = _infer_num_classes_from_state(state_dict)
    if model_num_classes is None:
        if args.num_classes is None:
            raise RuntimeError("Could not infer num_classes from checkpoint. Provide --num-classes.")
        model_num_classes = int(args.num_classes)

    requested_backbone = str(args.backbone).strip().lower()
    inferred_backbone = _infer_backbone_from_state(state_dict)
    backbone = inferred_backbone if inferred_backbone is not None else requested_backbone
    
    requested_stride = int(getattr(args, 'output_stride', 16))
    output_stride = requested_stride

    if inferred_backbone is not None and inferred_backbone != requested_backbone:
        print(
            f"[WARN] --backbone mismatch: args={requested_backbone}, checkpoint={inferred_backbone}. "
            "Using checkpoint backbone."
        )
    
    if isinstance(checkpoint, dict) and checkpoint.get("output_stride") is not None:
        ckpt_output_stride = int(checkpoint.get("output_stride"))
        if ckpt_output_stride != requested_stride:
            print(
                f"[WARN] --output-stride mismatch: args={requested_stride}, checkpoint={ckpt_output_stride}. "
                "Using checkpoint stride."
            )
            output_stride = ckpt_output_stride

    model = DeepLabv3(
        model_num_classes,
        dropout_rate=0.0,
        output_stride=output_stride,
        backbone=backbone,
        pretrained_backbone=_flag_enabled(args.pretrained_backbone),
    )
    
    try:
        result = model.load_state_dict(state_dict, strict=False)
        if result.missing_keys:
            print(f"[WARN] Missing keys in checkpoint: {result.missing_keys}")
        if result.unexpected_keys:
            print(f"[WARN] Unexpected keys in checkpoint: {result.unexpected_keys}")
    except RuntimeError as e:
        print(f"[ERROR] Failed to load checkpoint weights: {e}")
        raise
    
    model.to(device)
    model.eval()
    print(f"[INFO] Output stride: {output_stride}")
    print(f"[INFO] Checkpoint loaded: num_classes={model_num_classes}, backbone={backbone}, stride={output_stride}")

    multi_scale_test = _flag_enabled(args.multi_scale_test)
    flip_test = _flag_enabled(args.flip_test)
    tta_scales = _parse_scales(args.tta_scales)
    imagenet_normalize = _flag_enabled(args.imagenet_normalize)

    images = _find_foggy_images(args.input_root)
    if args.gt_root:
        if not os.path.isdir(args.gt_root):
            raise RuntimeError(f"Provided --gt-root does not exist: {args.gt_root}")
        gt_root = args.gt_root
    else:
        gt_root = _infer_default_gt_root(args.input_root)

    num_samples = max(0, int(args.num_samples))
    sample_indices = set(random.sample(range(len(images)), k=min(num_samples, len(images)))) if num_samples > 0 else set()
    samples_input_pred = []
    samples_with_gt = []

    losses = []
    accs = []
    skipped_no_valid_gt = 0
    matched_gt_count = 0
    inters = np.zeros(model_num_classes, dtype=np.int64)
    unions = np.zeros(model_num_classes, dtype=np.int64)
    criterion = nn.CrossEntropyLoss(ignore_index=255).to(device)

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Input root: {args.input_root}")
    print(f"[INFO] Images found: {len(images)}")
    print(f"[INFO] Output root: {args.out_dir}")
    print(f"[INFO] GT root: {gt_root if gt_root is not None else 'not set'}")
    print(f"[INFO] Backbone: {backbone} | pretrained_backbone={_flag_enabled(args.pretrained_backbone)}")
    print(f"[INFO] Input normalization: {'ImageNet mean/std' if imagenet_normalize else '[0,1] scaling only'}")
    print(f"[INFO] TTA settings: multi_scale={multi_scale_test}, flip_test={flip_test}, scales={tta_scales if multi_scale_test else [1.0]}")

    for idx, image_path in enumerate(images, start=1):
        image = _load_rgb(image_path)

        # Keep native resolution by default for submission-style inference.
        if args.resize_height is not None and args.resize_width is not None:
            base_h = int(args.resize_height)
            base_w = int(args.resize_width)
            image_for_model = resize_rgb(image, base_w, base_h)
        else:
            base_h, base_w = image.shape[:2]
            image_for_model = image

        logits = _aggregate_logits_with_tta(
            model,
            device,
            image_for_model,
            base_h,
            base_w,
            multi_scale=multi_scale_test,
            flip_test=flip_test,
            scales=tta_scales,
            use_imagenet_stats=imagenet_normalize,
        ).squeeze(0)

        pred_trainids = torch.argmax(logits, dim=0).cpu().numpy().astype(np.uint8)
        pred_color = _decode_cityscapes_trainids(pred_trainids)

        relative_dir = os.path.basename(os.path.dirname(image_path))
        stem = _prediction_stem(os.path.basename(image_path))

        mask_path = os.path.join(args.out_dir, relative_dir, f"{stem}_pred_trainIds.png")
        color_path = os.path.join(args.out_dir, relative_dir, f"{stem}_pred_color.png")

        _save_mask_png(mask_path, pred_trainids)
        save_rgb_png(color_path, pred_color)

        gt_path = None
        gt_mask = None
        if gt_root is not None:
            gt_path = _resolve_gt_path(image_path, args.input_root, gt_root)
            if gt_path is not None:
                matched_gt_count += 1
                gt_mask = _load_mask(gt_path).astype(np.uint8)
                if gt_mask.shape != pred_trainids.shape:
                    gt_mask = np.array(
                        Image.fromarray(gt_mask, mode="L").resize(
                            (pred_trainids.shape[1], pred_trainids.shape[0]),
                            Image.NEAREST,
                        )
                    )

                valid_mask = (gt_mask != 255)
                if int(valid_mask.sum()) == 0:
                    skipped_no_valid_gt += 1
                else:
                    gt_tensor = torch.from_numpy(gt_mask).unsqueeze(0).long().to(device)
                    loss = criterion(logits.unsqueeze(0), gt_tensor)
                    losses.append(float(loss.item()))
                    accs.append(_pixel_accuracy(pred_trainids, gt_mask, ignore_index=255))

                for c in range(model_num_classes):
                    pred_c = (pred_trainids == c) & valid_mask
                    gt_c = (gt_mask == c) & valid_mask
                    inters[c] += int(np.logical_and(pred_c, gt_c).sum())
                    unions[c] += int(np.logical_or(pred_c, gt_c).sum())

        if (idx - 1) in sample_indices:
            samples_input_pred.append((image_for_model, pred_color, stem))
            if gt_mask is not None:
                gt_color = _decode_cityscapes_trainids(gt_mask)
                samples_with_gt.append((image_for_model, gt_color, pred_color, stem))

        if idx % 50 == 0 or idx == len(images):
            print(f"[INFO] Processed {idx}/{len(images)}")

    print("[INFO] Foggy dataset inference complete.")

    mean_loss = _safe_nanmean(losses)
    mean_acc = _safe_nanmean(accs)
    valid_classes = unions > 0
    per_class_iou = np.zeros(model_num_classes, dtype=np.float32)
    if valid_classes.sum() > 0:
        per_class_iou[valid_classes] = inters[valid_classes] / (unions[valid_classes].astype(np.float32) + 1e-12)
        mean_iou = float(np.mean(per_class_iou[valid_classes]))
    else:
        mean_iou = float("nan")

    if gt_root is None:
        print("[WARN] No GT root available. Skipping metric computation.")
    elif matched_gt_count == 0:
        print("[WARN] No matching GT files found. Metrics unavailable.")
    elif len(losses) == 0:
        print("[WARN] No valid GT pixels found for metrics across matched GT images.")
        print(f"[WARN] Skipped metric computation for {skipped_no_valid_gt}/{matched_gt_count} matched GT images (all labels were ignore_index=255).")

    print(f"Val mean loss: {_format_metric(mean_loss)}")
    print(f"[INFO] Val mean pixel accuracy: {_format_metric(mean_acc)}")
    print(f"[INFO] Mean IoU: {_format_metric(mean_iou)}")

    curves_saved = _save_curves(losses, accs, args.out_dir)
    if curves_saved:
        print("[INFO] Saved curves: val_loss_curve_1.png, val_acc_curve_1.png")
    else:
        print("[WARN] No valid per-image metrics available. Skipping curve generation.")

    with open(os.path.join(args.out_dir, "foggy_val_summary.txt"), "w") as f:
        f.write(f"Images: {len(images)}\n")
        f.write(f"Matched GT images: {matched_gt_count}\n")
        f.write(f"Val mean loss: {_format_metric(mean_loss)}\n")
        f.write(f"Val mean pixel accuracy: {_format_metric(mean_acc)}\n")
        f.write(f"Mean IoU: {_format_metric(mean_iou)}\n")
        f.write("Per-class IoU:\n")
        for ci, iou in enumerate(per_class_iou):
            if unions[ci] > 0:
                f.write(f"  class {ci}: {iou:.6f}\n")
            else:
                f.write(f"  class {ci}: n/a (no GT pixels)\n")

    grid_saved = _save_random_samples_grid(samples_input_pred, args.out_dir)
    if grid_saved:
        print("[INFO] Saved grid: foggy_random_samples.png (Input + Prediction)")
    else:
        print("[INFO] Grid not generated (set --num-samples > 0).")

    if gt_root is not None:
        gt_grid_saved = _save_random_samples_with_gt_grid(samples_with_gt, args.out_dir)
        if gt_grid_saved:
            print("[INFO] Saved grid: foggy_random_samples_gt.png (Input + GT + Prediction)")
        else:
            print("[INFO] GT grid not generated (no matching GT files found for sampled images).")

    print(f"[INFO] All predictions saved in: {args.out_dir}")
    print("[INFO] Validation evaluation complete!")


def parse_args():
    parser = argparse.ArgumentParser(description="DeepLabv3 foggy split inference/evaluation.")
    parser.add_argument("-m", "--model-path", required=True, help="Path to trained checkpoint (.pth)")
    parser.add_argument(
        "--input-root",
        default="DeepLabv3/datasets/Cityscapes/leftImg8bit_trainvaltest_foggy/leftImg8bit_foggy/val",
        help="Root directory for foggy split images (default: val)",
    )
    parser.add_argument("--out-dir", default="preds_foggy_val", help="Directory to save foggy predictions")
    parser.add_argument("--num-samples", type=int, default=6, help="Number of random images to include in grid visualization")
    parser.add_argument(
        "--gt-root",
        type=str,
        default=None,
        help="Optional GT root for foggy data; auto-detected from split when omitted",
    )

    parser.add_argument(
        "--resize-height",
        type=int,
        default=None,
        help="Optional model input height. If unset, uses native image height.",
    )
    parser.add_argument(
        "--resize-width",
        type=int,
        default=None,
        help="Optional model input width. If unset, uses native image width.",
    )
    parser.add_argument("-nc", "--num-classes", type=int, default=None, help="Fallback class count if not inferable")

    parser.add_argument("--multi-scale-test", type=str, default="False", help="Use multi-scale inference")
    parser.add_argument("--flip-test", type=str, default="False", help="Use horizontal flip inference")
    parser.add_argument(
        "--tta-scales",
        type=float,
        nargs="+",
        default=[0.75, 1.0, 1.25],
        help="Scales to use when --multi-scale-test is enabled",
    )
    parser.add_argument(
        "--imagenet-normalize",
        type=str,
        default="False",
        help="Use ImageNet mean/std normalization at test time",
    )

    parser.add_argument("--backbone", choices=["resnet50", "resnet101"], default="resnet50")
    parser.add_argument("--output-stride", type=int, default=16, help="Backbone output stride (4, 8, or 16)")
    parser.add_argument("--pretrained-backbone", type=str, default="False")
    parser.add_argument("--cuda", type=str, default="True", help="Use CUDA if available")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_foggy_test(args)


if __name__ == "__main__":
    main()
