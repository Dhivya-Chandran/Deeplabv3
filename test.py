import os
import time
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# Headless-safe matplotlib (NiBi / Slurm nodes have no display)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    cv2 = None
    CV2_AVAILABLE = False

from cityscapes_preprocess import (
    normalize_rgb_image,
    resize_rgb,
    resize_mask_nearest,
    save_rgb_png,
)

from models.deeplabv3 import DeepLabv3
from cache_class_weights import load_class_weights


def _decode_cityscapes_trainids(mask: np.ndarray, ignore_index: int = 255) -> np.ndarray:
    """
    Decode Cityscapes trainIds (0..18, 255 ignore) to RGB for visualization.
    """
    trainid_colors = np.array([
        (128, 64, 128),   # 0 road
        (244, 35, 232),   # 1 sidewalk
        (70, 70, 70),     # 2 building
        (102, 102, 156),  # 3 wall
        (190, 153, 153),  # 4 fence
        (153, 153, 153),  # 5 pole
        (250, 170, 30),   # 6 traffic light
        (220, 220, 0),    # 7 traffic sign
        (107, 142, 35),   # 8 vegetation
        (152, 251, 152),  # 9 terrain
        (70, 130, 180),   # 10 sky
        (220, 20, 60),    # 11 person
        (255, 0, 0),      # 12 rider
        (0, 0, 142),      # 13 car
        (0, 0, 70),       # 14 truck
        (0, 60, 100),     # 15 bus
        (0, 80, 100),     # 16 train
        (0, 0, 230),      # 17 motorcycle
        (119, 11, 32),    # 18 bicycle
    ], dtype=np.uint8)

    rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for cls_id in range(trainid_colors.shape[0]):
        rgb[mask == cls_id] = trainid_colors[cls_id]

    # Make ignored/unlabeled pixels visible in qualitative plots.
    rgb[mask == ignore_index] = np.array([255, 255, 255], dtype=np.uint8)
    return rgb


def _infer_num_classes_from_state(state_dict):
    """
    Infer classifier output channels from DeepLab head in checkpoint.
    Expects key `conv.weight` or `conv.bias` from models/deeplabv3.py.
    """
    if not isinstance(state_dict, dict):
        return None

    conv_w = state_dict.get("conv.weight")
    if isinstance(conv_w, torch.Tensor) and conv_w.ndim == 4:
        return int(conv_w.shape[0])

    clf_w = state_dict.get("classifier.weight")
    if isinstance(clf_w, torch.Tensor) and clf_w.ndim == 4:
        return int(clf_w.shape[0])

    conv_b = state_dict.get("conv.bias")
    if isinstance(conv_b, torch.Tensor) and conv_b.ndim == 1:
        return int(conv_b.shape[0])

    return None


def _infer_backbone_from_state(state_dict):
    """
    Infer ResNet backbone from layer3 block depth in checkpoint keys.
    ResNet-50 layer3 blocks: indices 0..5 (6 blocks)
    ResNet-101 layer3 blocks: indices 0..22 (23 blocks)
    """
    if not isinstance(state_dict, dict):
        return None

    max_block_idx = -1
    for k in state_dict.keys():
        if not isinstance(k, str):
            continue
        if not k.startswith("layer3."):
            continue
        parts = k.split(".")
        if len(parts) < 3:
            continue
        if parts[1].isdigit():
            max_block_idx = max(max_block_idx, int(parts[1]))

    if max_block_idx >= 22:
        return 'resnet101'
    if max_block_idx >= 5:
        return 'resnet50'
    return None


def _flag_enabled(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {'false', '0', 'none', 'off', 'no', ''}


def _normalize_input_tensor(
    img_hwc: np.ndarray,
    device: torch.device,
    use_imagenet_stats: bool = False,
) -> torch.Tensor:
    """
    Convert uint8 RGB image to model input tensor.

    Training in this repo uses [0,1] scaling without ImageNet mean/std,
    so that is the default inference path.
    """
    x = _to_chw_tensor(normalize_rgb_image(img_hwc, imagenet_normalize=use_imagenet_stats)).to(device)
    return x


def _forward_logits(model, device, img_hwc: np.ndarray, use_imagenet_stats: bool = False) -> torch.Tensor:
    x = _normalize_input_tensor(img_hwc, device, use_imagenet_stats=use_imagenet_stats)
    with torch.no_grad():
        return model(x)


def _parse_tta_scales(scales):
    if scales is None:
        return [1.0]
    if isinstance(scales, (list, tuple, np.ndarray)):
        parsed = [float(scale) for scale in scales]
    else:
        parsed = [float(scales)]
    return parsed if len(parsed) > 0 else [1.0]


def _aggregate_logits_with_tta(model, device, img_hwc: np.ndarray, base_h: int, base_w: int,
                               multi_scale: bool = False, flip_test: bool = False,
                               scales=None, use_imagenet_stats: bool = False) -> torch.Tensor:
    scale_values = _parse_tta_scales(scales) if multi_scale else [1.0]
    logits_sum = None
    num_votes = 0

    for scale in scale_values:
        scaled_h = max(1, int(round(base_h * scale)))
        scaled_w = max(1, int(round(base_w * scale)))
        scaled_img = resize_rgb(img_hwc, scaled_w, scaled_h)

        logits = _forward_logits(model, device, scaled_img, use_imagenet_stats=use_imagenet_stats)
        logits = F.interpolate(logits, size=(base_h, base_w), mode='bilinear', align_corners=False)

        if logits_sum is None:
            logits_sum = logits
        else:
            logits_sum = logits_sum + logits
        num_votes += 1

        if flip_test:
            flipped_img = np.ascontiguousarray(np.fliplr(scaled_img))
            flip_logits = _forward_logits(model, device, flipped_img, use_imagenet_stats=use_imagenet_stats)
            flip_logits = torch.flip(flip_logits, dims=[3])
            flip_logits = F.interpolate(flip_logits, size=(base_h, base_w), mode='bilinear', align_corners=False)

            logits_sum = logits_sum + flip_logits
            num_votes += 1

    return logits_sum / max(1, num_votes)


def _to_chw_tensor(img_hwc: np.ndarray) -> torch.Tensor:
    """HWC uint8 -> 1xCxHxW float tensor"""
    t = torch.from_numpy(img_hwc).unsqueeze(0).float()
    t = t.permute(0, 3, 1, 2).contiguous()
    return t


def _load_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def _load_mask(path: str) -> np.ndarray:
    # Cityscapes GT is single channel PNG (trainIds), keep as uint8
    return np.array(Image.open(path))


def _infer_one(model, device, img_path, H, W, multi_scale=False, flip_test=False, scales=None,
               use_imagenet_stats: bool = False):
    img = _load_rgb(img_path)
    img_resized = resize_rgb(img, W, H)

    logits = _aggregate_logits_with_tta(
        model,
        device,
        img_resized,
        H,
        W,
        multi_scale=multi_scale,
        flip_test=flip_test,
        scales=scales,
        use_imagenet_stats=use_imagenet_stats,
    ).squeeze(0)

    pred = torch.argmax(logits, dim=0).cpu().numpy().astype(np.uint8)  # H x W
    color_pred = _decode_cityscapes_trainids(pred)  # RGB for trainIds
    return img_resized, logits, pred, color_pred


def _find_cityscapes_pairs(val_img_root: str, val_gt_root: str):
    """
    Pair each *_leftImg8bit.png with GT trainIds PNG.
    Supports both:
      - *_gtFine_trainIds.png   (your dataset)
      - *_gtFine_labelTrainIds.png (common alternative)
    """
    img_paths = glob.glob(os.path.join(val_img_root, "*", "*_leftImg8bit.png"))
    img_paths.sort()

    pairs = []
    missing = 0

    for ip in img_paths:
        city = os.path.basename(os.path.dirname(ip))
        base = os.path.basename(ip).replace("_leftImg8bit.png", "")

        candidates = [
            os.path.join(val_gt_root, city, base + "_gtFine_trainIds.png"),
            os.path.join(val_gt_root, city, base + "_gtFine_labelTrainIds.png"),
        ]

        gt = None
        for c in candidates:
            if os.path.exists(c):
                gt = c
                break

        if gt is None:
            missing += 1
            continue

        pairs.append((ip, gt))

    if len(pairs) == 0:
        raise RuntimeError(
            "No (image, GT) pairs found.\n"
            f"val_img_root={val_img_root}\n"
            f"val_gt_root={val_gt_root}\n"
            "Expected GT like: *_gtFine_trainIds.png (or *_gtFine_labelTrainIds.png)"
        )

    if missing > 0:
        print(f"[WARN] Missing GT for {missing} images (skipped).")

    return pairs


def _pixel_accuracy(pred: np.ndarray, gt: np.ndarray, ignore_index: int = 255) -> float:
    valid = (gt != ignore_index)
    if valid.sum() == 0:
        return float("nan")
    return float((pred[valid] == gt[valid]).mean())


def _safe_nanmean(values) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float('nan')
    valid = ~np.isnan(arr)
    if not np.any(valid):
        return float('nan')
    return float(arr[valid].mean())


def _format_metric(value: float) -> str:
    if np.isnan(value):
        return "n/a"
    return f"{value:.6f}"


def _save_curves(losses, accs, out_dir: str):
    xs = np.arange(1, len(losses) + 1)

    os.makedirs(out_dir, exist_ok=True)
    mean_loss = _safe_nanmean(losses)
    mean_acc = _safe_nanmean(accs)

    # If no valid metric samples exist, skip writing misleading empty curves.
    if len(losses) == 0 or np.isnan(mean_loss) or np.isnan(mean_acc):
        print("[WARN] No valid per-image metrics available. Skipping curve generation.")
        return False

    if PLOTLY_AVAILABLE:
        # Loss interactive graph
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(x=xs, y=losses, mode='lines+markers', name='Loss'))
        fig_loss.add_hline(y=mean_loss, line_dash='dash', line_color='red',
                           annotation_text=f"Mean: {mean_loss:.4f}", annotation_position="top left")
        fig_loss.update_layout(title="Validation Loss per Image", xaxis_title="Image index", yaxis_title="Loss")
        fig_loss.write_html(os.path.join(out_dir, "val_loss_graph.html"), include_plotlyjs='cdn')

        # Accuracy interactive graph
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(x=xs, y=accs, mode='lines+markers', name='Pixel Accuracy'))
        fig_acc.add_hline(y=mean_acc, line_dash='dash', line_color='red',
                          annotation_text=f"Mean: {mean_acc:.4f}", annotation_position="top left")
        fig_acc.update_layout(title="Validation Pixel Accuracy per Image", xaxis_title="Image index", yaxis_title="Pixel Accuracy")
        fig_acc.write_html(os.path.join(out_dir, "val_acc_graph.html"), include_plotlyjs='cdn')
        return True
    else:
        # Fallback to static matplotlib images (with mean lines)
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


def _save_random_samples(samples, out_dir: str):
    """
    samples: list of (img_rgb, gt_color, pred_color, name) - gt_color is unused but kept for compatibility
    Saves one grid image with Input and Prediction only.
    """
    n = len(samples)
    if n == 0:
        return False

    plt.figure(figsize=(8, 4 * n))
    for i, (img, gt_c, pred_c, name) in enumerate(samples):
        r = i * 2

        ax1 = plt.subplot(n, 2, r + 1)
        ax1.axis("off")
        ax1.set_title(f"{name}\nInput")
        ax1.imshow(img)

        ax2 = plt.subplot(n, 2, r + 2)
        ax2.axis("off")
        ax2.set_title("Prediction")
        ax2.imshow(pred_c)

    plt.savefig(os.path.join(out_dir, "random_samples_1.png"), bbox_inches="tight")
    plt.close()
    return True


def _save_random_samples_with_gt(samples, out_dir: str):
    """
    samples: list of (img_rgb, gt_color, pred_color, name)
    Saves one grid image with Input, Ground Truth, and Prediction.
    """
    n = len(samples)
    if n == 0:
        return False

    plt.figure(figsize=(12, 4 * n))
    for i, (img, gt_c, pred_c, name) in enumerate(samples):
        r = i * 3

        ax1 = plt.subplot(n, 3, r + 1)
        ax1.axis("off")
        ax1.set_title(f"{name}\nInput")
        ax1.imshow(img)

        ax2 = plt.subplot(n, 3, r + 2)
        ax2.axis("off")
        ax2.set_title("Ground Truth")
        ax2.imshow(gt_c)

        ax3 = plt.subplot(n, 3, r + 3)
        ax3.axis("off")
        ax3.set_title("Prediction")
        ax3.imshow(pred_c)

    plt.savefig(os.path.join(out_dir, "random_samples_gt_1.png"), bbox_inches="tight")
    plt.close()
    return True


def predict(FLAGS):
    """
    If FLAGS.image_path is a FILE -> single-image prediction.
    If FLAGS.image_path is a DIRECTORY, or --input-path-val/--input-path-test is provided,
    run full validation/test evaluation.
    """
    # ---------- Checks ----------
    if FLAGS.m is None or not str(FLAGS.m).endswith(".pth") or not os.path.exists(FLAGS.m):
        raise RuntimeError(f"Model checkpoint must be a .pth file and exist. Got: {FLAGS.m}")

    image_path = getattr(FLAGS, 'image_path', None)
    if image_path is not None and not os.path.exists(image_path):
        raise RuntimeError(f"image_path must exist (file or dir). Got: {image_path}")

    has_eval_root = bool(getattr(FLAGS, 'input_path_val', None) or getattr(FLAGS, 'input_path_test', None))
    if image_path is None and not has_eval_root:
        raise RuntimeError(
            "No input source provided. Set one of: --image-path, --input-path-val, or --input-path-test"
        )

    # Output dir
    out_dir = getattr(FLAGS, "out_dir", "preds_val")
    os.makedirs(out_dir, exist_ok=True)

    H = FLAGS.resize_height
    W = FLAGS.resize_width
    device = FLAGS.cuda  # torch.device from init.py
    multi_scale_test = _flag_enabled(getattr(FLAGS, 'multi_scale_test', True))  # ENABLED: Multi-scale TTA
    flip_test = _flag_enabled(getattr(FLAGS, 'flip_test', True))  # ENABLED: Flip test (horizontal flip)
    tta_scales = getattr(FLAGS, 'tta_scales', [0.75, 1.0, 1.25])  # Three scales for diversity
    imagenet_normalize = _flag_enabled(getattr(FLAGS, 'imagenet_normalize', False))
    requested_backbone = str(getattr(FLAGS, 'backbone', 'resnet50')).strip().lower()
    pretrained_backbone = _flag_enabled(getattr(FLAGS, 'pretrained_backbone', True))
    output_stride = int(getattr(FLAGS, 'output_stride', 16))

    # ---------- Load checkpoint ----------
    print("[INFO] Loading checkpoint...")
    checkpoint = torch.load(FLAGS.m, map_location="cpu")
    print("[INFO] Checkpoint loaded")

    # ---------- Build model ----------
    # Checkpoint expected to store dict with "model_state_dict"
    state = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    inferred_backbone = _infer_backbone_from_state(state)

    backbone = requested_backbone
    if inferred_backbone is not None and inferred_backbone != requested_backbone:
        print(
            "[WARN] --backbone does not match checkpoint architecture: "
            f"args={requested_backbone}, checkpoint={inferred_backbone}. "
            "Using checkpoint backbone for inference."
        )
        backbone = inferred_backbone
    
    if isinstance(checkpoint, dict) and checkpoint.get("output_stride") is not None:
        ckpt_output_stride = int(checkpoint.get("output_stride"))
        if ckpt_output_stride != output_stride:
            print(
                "[WARN] --output-stride does not match checkpoint metadata: "
                f"args={output_stride}, checkpoint={ckpt_output_stride}. "
                "Using checkpoint stride for inference."
            )
            output_stride = ckpt_output_stride

    ckpt_num_classes = _infer_num_classes_from_state(state)
    
    # Infer num_classes: prefer checkpoint, fall back to FLAG if provided
    model_num_classes = None
    if ckpt_num_classes is not None:
        model_num_classes = ckpt_num_classes
        if getattr(FLAGS, 'num_classes', None) is not None:
            flag_num_classes = int(FLAGS.num_classes)
            if flag_num_classes != ckpt_num_classes:
                print(
                    "[WARN] --num-classes does not match checkpoint head: "
                    f"args={flag_num_classes}, checkpoint={ckpt_num_classes}. "
                    "Using checkpoint class count for inference."
                )
    elif getattr(FLAGS, 'num_classes', None) is not None:
        model_num_classes = int(FLAGS.num_classes)
        print(f"[INFO] Using --num-classes from args: {model_num_classes}")
    else:
        raise RuntimeError("Cannot infer num_classes: provide --num-classes or ensure checkpoint has valid conv layer")

    model = DeepLabv3(
        model_num_classes,
        dropout_rate=0.0,
        output_stride=output_stride,
        backbone=backbone,
        pretrained_backbone=pretrained_backbone,
    )  # No dropout during inference
    
    try:
        result = model.load_state_dict(state, strict=False)
        if result.missing_keys:
            print(f"[WARN] Missing keys in checkpoint: {result.missing_keys}")
        if result.unexpected_keys:
            print(f"[WARN] Unexpected keys in checkpoint: {result.unexpected_keys}")
    except RuntimeError as e:
        print(f"[ERROR] Failed to load checkpoint weights: {e}")
        raise
    
    model.to(device)
    model.eval()
    print(f"[INFO] Model loaded on device: {device}")
    print(f"[INFO] Backbone: {backbone} | pretrained_backbone={pretrained_backbone}")
    print(f"[INFO] Output stride: {output_stride}")
    print(f"[INFO] Checkpoint config: num_classes={model_num_classes}, backbone={backbone}, stride={output_stride}")
    print(f"[INFO] Input normalization: {'ImageNet mean/std' if imagenet_normalize else '[0,1] scaling only'}")
    print(f"[INFO] TTA settings: multi_scale={multi_scale_test}, flip_test={flip_test}, scales={_parse_tta_scales(tta_scales) if multi_scale_test else [1.0]}")

    # ---------- Single image mode ----------
    if image_path is not None and os.path.isfile(image_path):
        img_path = image_path
        base = os.path.basename(img_path).replace("_leftImg8bit.png", "").replace(".png", "")

        print("[INFO] Single-image inference:", img_path)
        s = time.time()
        img_resized, logits, pred, color_pred = _infer_one(
            model,
            device,
            img_path,
            H,
            W,
            multi_scale=multi_scale_test,
            flip_test=flip_test,
            scales=tta_scales,
            use_imagenet_stats=imagenet_normalize,
        )
        print("[INFO] Time taken:", time.time() - s)

        input_path = os.path.join(out_dir, f"{base}_input.png")
        seg_path = os.path.join(out_dir, f"{base}_seg.png")

        save_rgb_png(input_path, img_resized)
        save_rgb_png(seg_path, color_pred)

        print("[INFO] Saved:", input_path)
        print("[INFO] Saved:", seg_path)
        print("[INFO] Prediction complete successfully!")
        return

    # ---------- Validation/Test-set mode ----------
    # For paper-style metric reporting, prefer validation split by default.
    # Priority for image root:
    #  1. FLAGS.image_path (if provided)
    #  2. FLAGS.input_path_val
    #  3. FLAGS.input_path_test
    val_img_root = None
    if getattr(FLAGS, 'image_path', None):
        val_img_root = FLAGS.image_path
    elif getattr(FLAGS, 'input_path_val', None):
        val_img_root = getattr(FLAGS, 'input_path_val', None)
    elif getattr(FLAGS, 'mode', None) == 'test' and getattr(FLAGS, 'input_path_test', None):
        val_img_root = FLAGS.input_path_test

    if val_img_root is None:
        raise RuntimeError('No validation/test image root provided. Set --image-path or --input-path-test/--input-path-val')

    # Determine GT root: prefer explicit test labels, then val labels, else try to infer
    val_gt_root = getattr(FLAGS, 'label_path_test', None) or getattr(FLAGS, 'label_path_val', None)
    if val_gt_root is None:
        # Try infer: replace leftImg8bit/val or leftImg8bit/test -> gtFine/val or gtFine/test
        if 'leftImg8bit' in val_img_root:
            val_gt_root = val_img_root.replace('leftImg8bit', 'gtFine')
        else:
            val_gt_root = val_img_root
        print('[WARN] label_path_val/test not provided. Using inferred GT root:', val_gt_root)

    if not os.path.isdir(val_gt_root):
        raise RuntimeError(f"Validation/Testing GT root not found: {val_gt_root}")

    # Try requested GT root first, then common trainIds sibling roots.
    gt_roots_to_try = [val_gt_root]
    if 'gtFine' in val_gt_root:
        candidate = val_gt_root.replace('gtFine', 'gtFine_trainIds')
        if candidate not in gt_roots_to_try:
            gt_roots_to_try.append(candidate)
        candidate_only = val_gt_root.replace('gtFine', 'gtFine_trainIds_only')
        if candidate_only not in gt_roots_to_try:
            gt_roots_to_try.append(candidate_only)

    pairs = None
    root_errors = []
    for gt_root in gt_roots_to_try:
        if not os.path.isdir(gt_root):
            root_errors.append(f"GT root not found: {gt_root}")
            continue
        try:
            pairs = _find_cityscapes_pairs(val_img_root, gt_root)
            val_gt_root = gt_root
            break
        except RuntimeError as ex:
            root_errors.append(str(ex))

    if pairs is None:
        raise RuntimeError(
            "Failed to find Cityscapes (image, GT) pairs from the configured roots.\n"
            f"Tried image root: {val_img_root}\n"
            "Candidate GT roots:\n"
            + "\n".join(f"  - {r}" for r in gt_roots_to_try)
            + "\nDetails:\n"
            + "\n---\n".join(root_errors)
        )

    print(f"[INFO] Using GT root: {val_gt_root}")
    print(f"[INFO] Found {len(pairs)} validation pairs")

    # Loss: match training setup when class weights are enabled and available.
    class_weights = None
    use_class_weights = bool(getattr(FLAGS, 'use_class_weights', True))
    if use_class_weights:
        weight_path = 'class_weights_cityscapes.pkl'
        if os.path.exists(weight_path):
            try:
                class_weights = load_class_weights(weight_path).to(device)
                print(f"[INFO] Loaded class weights from {weight_path}")
            except Exception as ex:
                print(f"[WARN] Failed to load class weights: {ex}")
        else:
            print(f"[WARN] Class weight cache not found at {weight_path}; using unweighted CE for validation loss")

    criterion = nn.CrossEntropyLoss(ignore_index=255, weight=class_weights).to(device)

    losses = []
    accs = []
    skipped_no_valid_gt = 0
    # IoU accumulators
    nc = model_num_classes
    inters = np.zeros(nc, dtype=np.int64)
    unions = np.zeros(nc, dtype=np.int64)

    # random samples
    k = int(getattr(FLAGS, "num_samples", 5))
    if k > 0:
        sample_indices = set(random.sample(range(len(pairs)), k=min(k, len(pairs))))
    else:
        sample_indices = set()
    samples_for_plot = []

    for idx, (img_path, gt_path) in enumerate(pairs, start=1):
        img_resized, logits, pred, color_pred = _infer_one(
            model,
            device,
            img_path,
            H,
            W,
            multi_scale=multi_scale_test,
            flip_test=flip_test,
            scales=tta_scales,
            use_imagenet_stats=imagenet_normalize,
        )

        # Load GT and resize to match (H,W)
        gt = _load_mask(gt_path)
        gt_resized = resize_mask_nearest(gt, W, H).astype(np.uint8)

        # Compute metrics only when GT has at least one non-ignore pixel.
        valid_metric_mask = (gt_resized != 255)
        if int(valid_metric_mask.sum()) == 0:
            skipped_no_valid_gt += 1
        else:
            # Compute loss
            gt_tensor = torch.from_numpy(gt_resized).unsqueeze(0).long().to(device)  # 1xHxW
            loss = criterion(logits.unsqueeze(0), gt_tensor)  # 1xCxHxW vs 1xHxW
            losses.append(float(loss.item()))

            # Pixel accuracy
            accs.append(_pixel_accuracy(pred, gt_resized, ignore_index=255))

        # Update intersection and union per class (ignore index 255)
        valid_mask = valid_metric_mask
        for c in range(nc):
            pred_c = (pred == c) & valid_mask
            gt_c = (gt_resized == c) & valid_mask
            inters[c] += int(np.logical_and(pred_c, gt_c).sum())
            unions[c] += int(np.logical_or(pred_c, gt_c).sum())

        # Save prediction image
        base = os.path.basename(img_path).replace("_leftImg8bit.png", "")
        seg_path = os.path.join(out_dir, f"{base}_seg.png")
        save_rgb_png(seg_path, color_pred)

        # Random sample visualization
        if (idx - 1) in sample_indices:
            gt_color = _decode_cityscapes_trainids(gt_resized)
            samples_for_plot.append((img_resized, gt_color, color_pred, base))

        if idx % 50 == 0:
            print(f"[INFO] Processed {idx}/{len(pairs)}")

    # Save curves + samples
    mean_loss = _safe_nanmean(losses)
    mean_acc = _safe_nanmean(accs)
    if len(losses) == 0:
        print("[WARN] No valid GT pixels found for metrics across all images.")
        print("[INFO] This is expected on Cityscapes test split where labels are hidden/unlabeled (all 255).")
    if skipped_no_valid_gt > 0:
        print(f"[WARN] Skipped metric computation for {skipped_no_valid_gt}/{len(pairs)} images (all labels were ignore_index=255).")
    print(f"[INFO] Val mean loss: {_format_metric(mean_loss)}")
    print(f"[INFO] Val mean pixel accuracy: {_format_metric(mean_acc)}")

    # Compute per-class IoU and mean IoU
    per_class_iou = np.zeros(nc, dtype=np.float32)
    valid_classes = unions > 0
    if valid_classes.sum() > 0:
        per_class_iou[valid_classes] = inters[valid_classes] / (unions[valid_classes].astype(np.float32) + 1e-12)
        mean_iou = float(np.mean(per_class_iou[valid_classes]))
    else:
        mean_iou = float('nan')

    print(f"[INFO] Validation mIoU: {_format_metric(mean_iou)}")

    curves_saved = _save_curves(losses, accs, out_dir)
    samples_saved = _save_random_samples(samples_for_plot, out_dir)
    samples_with_gt_saved = _save_random_samples_with_gt(samples_for_plot, out_dir)

    # Save summary text
    with open(os.path.join(out_dir, "val_summary.txt"), "w") as f:
        f.write(f"Val images: {len(pairs)}\n")
        f.write(f"Mean loss: {_format_metric(mean_loss)}\n")
        f.write(f"Mean pixel accuracy: {_format_metric(mean_acc)}\n")
        f.write(f"Validation mIoU: {_format_metric(mean_iou)}\n")
        f.write(f"Multi-scale testing: {multi_scale_test}\n")
        f.write(f"Flip testing: {flip_test}\n")
        f.write(f"TTA scales: {_parse_tta_scales(tta_scales) if multi_scale_test else [1.0]}\n")
        # Optionally write per-class IoU
        f.write("Per-class IoU:\n")
        for ci, iou in enumerate(per_class_iou):
            if unions[ci] > 0:
                f.write(f"  class {ci}: {iou:.6f}\n")
            else:
                f.write(f"  class {ci}: n/a (no GT pixels)\n")

    if curves_saved:
        print("[INFO] Saved curves: val_loss_curve_1.png, val_acc_curve_1.png")
    else:
        print("[INFO] Curves not generated (no valid GT metrics).")
    if samples_saved:
        print("[INFO] Saved random samples grid (Input + Prediction): random_samples_1.png")
    else:
        print("[INFO] Random samples grid not generated (no samples requested/available).")
    if samples_with_gt_saved:
        print("[INFO] Saved random samples grid (Input + GT + Prediction): random_samples_gt_1.png")
    else:
        print("[INFO] Random samples grid with GT not generated (no samples requested/available).")
    print("[INFO] All predictions saved in:", out_dir)
    print("[INFO] Validation evaluation complete!")
