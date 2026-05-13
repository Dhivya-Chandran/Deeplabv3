#!/bin/bash
#SBATCH --job-name=deeplab_foggy_val
#SBATCH --account=def-akilan
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

module load python opencv
VENV_DIR="${VENV_DIR:-$HOME/venvs/myenv}"
if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "ERROR: Virtual environment not found at $VENV_DIR"
  exit 1
fi
source "$VENV_DIR/bin/activate"

cd /scratch/$USER/DeepLabv3

export TORCH_HOME=/scratch/$USER/.cache/torch
mkdir -p "$TORCH_HOME"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== GPU INFO ==="
nvidia-smi

echo "=== Foggy VAL Inference Config ==="
INPUT_ROOT=${INPUT_ROOT:-DeepLabv3/datasets/Cityscapes/leftImg8bit_trainvaltest_foggy/leftImg8bit_foggy/val}
GT_ROOT=${GT_ROOT:-datasets/Cityscapes/gtFine_trainIds/val}
OUT_DIR=${OUT_DIR:-preds_foggy_val}
NUM_SAMPLES=${NUM_SAMPLES:-8}
MULTI_SCALE_TEST=${MULTI_SCALE_TEST:-True}
FLIP_TEST=${FLIP_TEST:-True}
TTA_SCALES=${TTA_SCALES:-0.75 1.0 1.25}
BACKBONE=${BACKBONE:-resnet101}
CKPT=${CKPT:-best_miou_model_${BACKBONE}.pth}
OUTPUT_STRIDE=${OUTPUT_STRIDE:-8}
PRETRAINED_BACKBONE=${PRETRAINED_BACKBONE:-False}
CUDA=${CUDA:-True}

printf "CKPT=%s\nINPUT_ROOT=%s\nGT_ROOT=%s\nOUT_DIR=%s\n" "$CKPT" "$INPUT_ROOT" "$GT_ROOT" "$OUT_DIR"

python test_foggy.py \
  -m "$CKPT" \
  --input-root "$INPUT_ROOT" \
  --gt-root "$GT_ROOT" \
  --out-dir "$OUT_DIR" \
  --num-samples "$NUM_SAMPLES" \
  --multi-scale-test "$MULTI_SCALE_TEST" \
  --flip-test "$FLIP_TEST" \
  --tta-scales $TTA_SCALES \
  --backbone "$BACKBONE" \
  --output-stride "$OUTPUT_STRIDE" \
  --pretrained-backbone "$PRETRAINED_BACKBONE" \
  --cuda "$CUDA"

echo "Job complete. Outputs saved under: $OUT_DIR"
