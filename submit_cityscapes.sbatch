#!/bin/bash
#SBATCH --job-name=cityscapes_submit
#SBATCH --account=def-akilan
#SBATCH --output=logs/cityscapes_submit-%j.out
#SBATCH --error=logs/cityscapes_submit-%j.err
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8

source ~/venvs/myenv/bin/activate
cd /scratch/$USER/DeepLabv3

export TORCH_HOME=/scratch/$USER/.cache/torch
mkdir -p "$TORCH_HOME"

BACKBONE=${BACKBONE:-resnet50}
CKPT=${CKPT:-best_miou_model_${BACKBONE}.pth}
OUTPUT_STRIDE=${OUTPUT_STRIDE:-8}

python generate_cityscapes_submission.py \
  -m "$CKPT" \
  --input-root datasets/Cityscapes/leftImg8bit/test \
  --output-root cityscapes_submission \
  --backbone "$BACKBONE" \
  --output-stride "$OUTPUT_STRIDE" \
  --pretrained-backbone True \
  --multi-scale-test True \
  --flip-test True \
  --tta-scales 0.5 0.75 1.0 1.25 1.5 2.0 \
  --cuda True \
  --zip-output True
