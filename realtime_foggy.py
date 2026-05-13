#!/usr/bin/env python3
"""Real-time semantic segmentation for webcam/video using DeepLabv3.

This script is optimized for speed-oriented inference:
- single-scale forward pass (no TTA)
- optional FP16 autocast on CUDA
- lightweight overlay visualization with FPS display
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np
import torch

from cityscapes_preprocess import normalize_rgb_image
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
    t = torch.from_numpy(img_hwc).unsqueeze(0).float()
    return t.permute(0, 3, 1, 2).contiguous()


def _build_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    requested_backbone: str,
    pretrained_backbone: bool,
    num_classes_arg: Optional[int],
) -> DeepLabv3:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint

    model_num_classes = _infer_num_classes_from_state(state_dict)
    if model_num_classes is None:
        if num_classes_arg is None:
            raise RuntimeError("Could not infer num_classes from checkpoint. Pass --num-classes.")
        model_num_classes = int(num_classes_arg)

    inferred_backbone = _infer_backbone_from_state(state_dict)
    backbone = inferred_backbone if inferred_backbone is not None else requested_backbone

    if inferred_backbone is not None and inferred_backbone != requested_backbone:
        print(
            f"[WARN] --backbone mismatch: args={requested_backbone}, checkpoint={inferred_backbone}. "
            "Using checkpoint backbone."
        )

    model = DeepLabv3(
        model_num_classes,
        dropout_rate=0.0,
        output_stride=16,
        backbone=backbone,
        pretrained_backbone=pretrained_backbone,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"[INFO] Loaded model on {device} | classes={model_num_classes} | backbone={backbone}")
    return model


@torch.no_grad()
def _predict_overlay(
    model,
    frame_bgr: np.ndarray,
    device: torch.device,
    input_h: int,
    input_w: int,
    use_fp16: bool,
    alpha: float,
    imagenet_normalize: bool,
):
    orig_h, orig_w = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    resized_rgb = cv2.resize(frame_rgb, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    x = _to_chw_tensor(normalize_rgb_image(resized_rgb, imagenet_normalize=imagenet_normalize)).to(device)

    use_amp = bool(use_fp16 and device.type == "cuda")
    with torch.autocast(device_type="cuda", enabled=use_amp):
        logits = model(x)

    pred = torch.argmax(logits.squeeze(0), dim=0).cpu().numpy().astype(np.uint8)
    pred_color_rgb = _decode_cityscapes_trainids(pred)

    pred_color_bgr = cv2.cvtColor(pred_color_rgb, cv2.COLOR_RGB2BGR)
    pred_color_bgr = cv2.resize(pred_color_bgr, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    alpha = min(max(alpha, 0.0), 1.0)
    overlay = cv2.addWeighted(frame_bgr, 1.0 - alpha, pred_color_bgr, alpha, 0.0)
    return overlay, pred_color_bgr


def _parse_source(source: str):
    source = str(source).strip()
    return int(source) if source.isdigit() else source


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time DeepLabv3 segmentation for foggy driving scenes.")
    parser.add_argument("-m", "--model-path", required=True, help="Path to trained checkpoint (.pth)")
    parser.add_argument("--source", default="0", help="Camera index (e.g. 0) or video file path")
    parser.add_argument("--save-path", default=None, help="Optional output video path for overlay")

    parser.add_argument("--input-height", type=int, default=512, help="Inference input height")
    parser.add_argument("--input-width", type=int, default=1024, help="Inference input width")
    parser.add_argument("--alpha", type=float, default=0.55, help="Overlay blend factor (0-1)")

    parser.add_argument("--display", type=str, default="True", help="Display output window")
    parser.add_argument("--cuda", type=str, default="True", help="Use CUDA if available")
    parser.add_argument("--fp16", type=str, default="True", help="Use fp16 autocast on CUDA")
    parser.add_argument("--imagenet-normalize", type=str, default="False", help="Use ImageNet normalization")

    parser.add_argument("--backbone", choices=["resnet50", "resnet101"], default="resnet50")
    parser.add_argument("--pretrained-backbone", type=str, default="False")
    parser.add_argument("-nc", "--num-classes", type=int, default=None, help="Fallback if checkpoint inference fails")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.input_height <= 0 or args.input_width <= 0:
        raise RuntimeError("--input-height and --input-width must be positive")

    device = torch.device("cuda:0" if torch.cuda.is_available() and _flag_enabled(args.cuda) else "cpu")
    model = _build_model_from_checkpoint(
        checkpoint_path=args.model_path,
        device=device,
        requested_backbone=str(args.backbone).strip().lower(),
        pretrained_backbone=_flag_enabled(args.pretrained_backbone),
        num_classes_arg=args.num_classes,
    )

    source = _parse_source(args.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    writer = None
    save_path = args.save_path
    display = _flag_enabled(args.display)
    use_fp16 = _flag_enabled(args.fp16)
    imagenet_normalize = _flag_enabled(args.imagenet_normalize)

    if save_path is not None:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0:
            fps = 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            raise RuntimeError("Failed to read frame size for video writer")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, float(fps), (width, height))

    print("[INFO] Press 'q' to stop.")
    print(
        f"[INFO] source={source} input_size=({args.input_height},{args.input_width}) "
        f"fp16={use_fp16 and device.type == 'cuda'} display={display} save_path={save_path}"
    )

    frame_count = 0
    times = deque(maxlen=120)

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            t0 = time.perf_counter()
            overlay, _ = _predict_overlay(
                model=model,
                frame_bgr=frame_bgr,
                device=device,
                input_h=int(args.input_height),
                input_w=int(args.input_width),
                use_fp16=use_fp16,
                alpha=float(args.alpha),
                imagenet_normalize=imagenet_normalize,
            )
            dt = time.perf_counter() - t0

            times.append(dt)
            frame_count += 1
            fps_now = 1.0 / max(1e-6, (sum(times) / len(times)))

            cv2.putText(
                overlay,
                f"FPS: {fps_now:.2f}",
                (16, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            if writer is not None:
                writer.write(overlay)

            if display:
                cv2.imshow("DeepLabv3 Real-time Segmentation", overlay)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break

            if frame_count % 60 == 0:
                print(f"[INFO] Processed {frame_count} frames | avg FPS={fps_now:.2f}")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()

    print(f"[INFO] Finished. Frames processed: {frame_count}")
    if save_path is not None:
        print(f"[INFO] Saved overlay video: {save_path}")


if __name__ == "__main__":
    main()
