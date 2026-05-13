#!/usr/bin/env python3
"""Generate Cityscapes test-set submission PNGs from a trained DeepLabv3 checkpoint.

This script writes one prediction PNG per test image using the official Cityscapes
label-ID encoding and preserves the Cityscapes folder structure under the output root.
It can also bundle the output directory into a zip archive for upload.
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from cityscapes_preprocess import normalize_rgb_image, resize_rgb
from models.deeplabv3 import DeepLabv3


TRAINID_TO_LABELID = np.full(256, 255, dtype=np.uint8)
TRAINID_TO_LABELID[0] = 7
TRAINID_TO_LABELID[1] = 8
TRAINID_TO_LABELID[2] = 11
TRAINID_TO_LABELID[3] = 12
TRAINID_TO_LABELID[4] = 13
TRAINID_TO_LABELID[5] = 17
TRAINID_TO_LABELID[6] = 19
TRAINID_TO_LABELID[7] = 20
TRAINID_TO_LABELID[8] = 21
TRAINID_TO_LABELID[9] = 22
TRAINID_TO_LABELID[10] = 23
TRAINID_TO_LABELID[11] = 24
TRAINID_TO_LABELID[12] = 25
TRAINID_TO_LABELID[13] = 26
TRAINID_TO_LABELID[14] = 27
TRAINID_TO_LABELID[15] = 28
TRAINID_TO_LABELID[16] = 31
TRAINID_TO_LABELID[17] = 32
TRAINID_TO_LABELID[18] = 33
TRAINID_TO_LABELID[255] = 255


def _flag_enabled(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"false", "0", "none", "off", "no", ""}


def _load_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def _to_chw_tensor(img_hwc: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(img_hwc).unsqueeze(0).float()
    return tensor.permute(0, 3, 1, 2).contiguous()


def _infer_num_classes_from_state(state_dict):
    if not isinstance(state_dict, dict):
        return None

    for key in ("classifier.weight", "conv.weight"):
        tensor = state_dict.get(key)
        if isinstance(tensor, torch.Tensor) and tensor.ndim == 4:
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


def _build_model(num_classes: int, backbone: str, pretrained_backbone: bool, output_stride: int) -> DeepLabv3:
    return DeepLabv3(
        num_classes,
        dropout_rate=0.0,
        output_stride=output_stride,
        backbone=backbone,
        pretrained_backbone=pretrained_backbone,
    )


def _load_checkpoint(checkpoint_path: str) -> Tuple[Dict[str, torch.Tensor], Optional[dict]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"], checkpoint
    return checkpoint, None


def _normalize_input_tensor(img_hwc: np.ndarray, device: torch.device, use_imagenet_stats: bool = False) -> torch.Tensor:
    x = _to_chw_tensor(normalize_rgb_image(img_hwc, imagenet_normalize=use_imagenet_stats)).to(device)
    return x


@torch.no_grad()
def _forward_logits(model, device, img_hwc: np.ndarray, use_imagenet_stats: bool = False) -> torch.Tensor:
    x = _normalize_input_tensor(img_hwc, device, use_imagenet_stats=use_imagenet_stats)
    return model(x)


def _aggregate_logits_with_tta(
    model,
    device,
    img_hwc: np.ndarray,
    base_h: int,
    base_w: int,
    multi_scale: bool = False,
    flip_test: bool = False,
    scales: Optional[Iterable[float]] = None,
    use_imagenet_stats: bool = False,
) -> torch.Tensor:
    scale_values = [1.0]
    if multi_scale and scales is not None:
        scale_values = [float(scale) for scale in scales] or [1.0]

    logits_sum = None
    num_votes = 0

    for scale in scale_values:
        scaled_h = max(1, int(round(base_h * scale)))
        scaled_w = max(1, int(round(base_w * scale)))
        scaled_img = resize_rgb(img_hwc, scaled_w, scaled_h)

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


def _save_label_png(path: str, label_ids: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(label_ids.astype(np.uint8)).save(path)


def _make_prediction_path(output_root: str, input_path: str, input_root: str) -> str:
    relative_path = os.path.relpath(input_path, input_root)
    relative_dir = os.path.dirname(relative_path)
    file_name = os.path.basename(input_path).replace("_leftImg8bit.png", "_gtFine_labelIds.png")
    return os.path.join(output_root, relative_dir, file_name)


def _find_test_images(input_root: str):
    images = glob.glob(os.path.join(input_root, "*", "*_leftImg8bit.png"))
    images.sort()
    if not images:
        raise RuntimeError(f"No Cityscapes images found under: {input_root}")
    return images


def _zip_directory(folder_path: str) -> str:
    archive_base = folder_path.rstrip(os.sep)
    archive_path = shutil.make_archive(archive_base, "zip", folder_path)
    return archive_path


def generate_submission(
    checkpoint_path: str,
    input_root: str,
    output_root: str,
    device: torch.device,
    backbone: str,
    pretrained_backbone: bool,
    output_stride: int,
    multi_scale_test: bool,
    flip_test: bool,
    tta_scales,
    use_imagenet_stats: bool,
    zip_output: bool,
) -> str:
    state_dict, checkpoint = _load_checkpoint(checkpoint_path)
    ckpt_num_classes = _infer_num_classes_from_state(state_dict)
    if ckpt_num_classes is None:
        raise RuntimeError("Could not infer the number of classes from the checkpoint.")

    inferred_backbone = _infer_backbone_from_state(state_dict)
    if inferred_backbone is not None and inferred_backbone != backbone:
        print(
            f"[WARN] --backbone does not match checkpoint architecture: args={backbone}, "
            f"checkpoint={inferred_backbone}. Using checkpoint backbone."
        )
        backbone = inferred_backbone

    model = _build_model(ckpt_num_classes, backbone, pretrained_backbone, output_stride)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    images = _find_test_images(input_root)
    os.makedirs(output_root, exist_ok=True)

    print(f"[INFO] Found {len(images)} images")
    print(f"[INFO] Saving predictions under: {output_root}")
    print(f"[INFO] Backbone: {backbone} | pretrained_backbone={pretrained_backbone}")
    print(f"[INFO] Output stride: {output_stride}")
    print(f"[INFO] Input normalization: {'ImageNet mean/std' if use_imagenet_stats else '[0,1] scaling only'}")
    print(f"[INFO] TTA settings: multi_scale={multi_scale_test}, flip_test={flip_test}, scales={tta_scales if multi_scale_test else [1.0]}")

    for index, image_path in enumerate(images, start=1):
        image = _load_rgb(image_path)
        base_height, base_width = image.shape[:2]

        logits = _aggregate_logits_with_tta(
            model,
            device,
            image,
            base_height,
            base_width,
            multi_scale=multi_scale_test,
            flip_test=flip_test,
            scales=tta_scales,
            use_imagenet_stats=use_imagenet_stats,
        ).squeeze(0)

        prediction_train_ids = torch.argmax(logits, dim=0).cpu().numpy().astype(np.uint8)
        prediction_label_ids = TRAINID_TO_LABELID[prediction_train_ids]

        prediction_path = _make_prediction_path(output_root, image_path, input_root)
        _save_label_png(prediction_path, prediction_label_ids)

        if index % 50 == 0 or index == len(images):
            print(f"[INFO] Processed {index}/{len(images)}")

    if zip_output:
        archive_path = _zip_directory(output_root)
        print(f"[INFO] Created zip archive: {archive_path}")
        return archive_path

    return output_root


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Cityscapes test submission PNGs.")
    parser.add_argument("-m", "--model-path", required=True, help="Path to the trained checkpoint (.pth)")
    parser.add_argument(
        "--input-root",
        default="datasets/Cityscapes/leftImg8bit/test",
        help="Root folder with Cityscapes test images",
    )
    parser.add_argument(
        "--output-root",
        default="cityscapes_submission",
        help="Folder where prediction PNGs will be written",
    )
    parser.add_argument(
        "--backbone",
        choices=["resnet50", "resnet101"],
        default="resnet50",
        help="Backbone to instantiate before loading the checkpoint",
    )
    parser.add_argument(
        "--output-stride",
        type=int,
        default=16,
        help="Backbone output stride (4, 8, or 16)",
    )
    parser.add_argument(
        "--pretrained-backbone",
        type=str,
        default="False",
        help="Use pretrained ImageNet backbone weights at model construction time",
    )
    parser.add_argument(
        "--multi-scale-test",
        type=str,
        default="True",
        help="Use multi-scale inference",
    )
    parser.add_argument(
        "--flip-test",
        type=str,
        default="True",
        help="Use horizontal flip test-time augmentation",
    )
    parser.add_argument(
        "--tta-scales",
        type=float,
        nargs="+",
        default=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
        help="Scales used for multi-scale inference",
    )
    parser.add_argument(
        "--imagenet-normalize",
        type=str,
        default="False",
        help="Normalize inputs with ImageNet mean/std",
    )
    parser.add_argument(
        "--cuda",
        type=str,
        default="True",
        help="Use CUDA if available",
    )
    parser.add_argument(
        "--zip-output",
        type=str,
        default="True",
        help="Zip the output folder after prediction",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() and _flag_enabled(args.cuda) else "cpu")

    print("[INFO] Loading checkpoint...")
    print(f"[INFO] Checkpoint: {args.model_path}")

    result_path = generate_submission(
        checkpoint_path=args.model_path,
        input_root=args.input_root,
        output_root=args.output_root,
        device=device,
        backbone=str(args.backbone).strip().lower(),
        pretrained_backbone=_flag_enabled(args.pretrained_backbone),
        output_stride=int(args.output_stride),
        multi_scale_test=_flag_enabled(args.multi_scale_test),
        flip_test=_flag_enabled(args.flip_test),
        tta_scales=args.tta_scales,
        use_imagenet_stats=_flag_enabled(args.imagenet_normalize),
        zip_output=_flag_enabled(args.zip_output),
    )

    print(f"[INFO] Done. Output: {result_path}")


if __name__ == "__main__":
    main()
