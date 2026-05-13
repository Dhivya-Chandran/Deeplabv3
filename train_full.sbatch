#!/bin/bash
#SBATCH --job-name=deeplab_train
#SBATCH --account=def-akilan
#SBATCH --output=logs/deeplab_train-%j.out
#SBATCH --error=logs/deeplab_train-%j.err
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8

# Load OpenCV module before virtual environment
module load opencv/4.13.0

# Activate your Python virtual environment
source ~/venvs/myenv/bin/activate

# Go to your working directory
cd /scratch/$USER/DeepLabv3

# Set PyTorch cache
export TORCH_HOME=/scratch/dchandr4/.cache/torch
mkdir -p "$TORCH_HOME"

# Optional: PyTorch expandable CUDA segments
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Print GPU info
nvidia-smi

# Run your DeepLabv3 training
# Optional alternative:
#   --backbone resnet50
python init.py \
  --mode train \
  --backbone resnet101 \
  --pretrained-backbone True \
  --cuda True \
  --batch-size 8 \
  --grad-accum-steps 2 \
  --epochs 150 \
  --num-classes 19 \
  --dtype cityscapes \
  --resize-height 769 \
  --resize-width 769 \
  --output-stride 16 \
  --learning-rate 0.01 \
  --weight-decay 5e-4 \
  --optimizer sgd \
  --momentum 0.9 \
  --scheduler poly \
  --use-class-weights True \
  --augment True \
  --early-stop-patience 12 \
  --eval-every 10 \
  --input-path-train datasets/Cityscapes/leftImg8bit/train \
  --label-path-train datasets/Cityscapes/gtFine_trainIds/train \
  --input-path-val datasets/Cityscapes/leftImg8bit/val \
  --label-path-val datasets/Cityscapes/gtFine_trainIds/val \
  --num-workers 8

# Generate plots automatically after training completes
echo "Training completed. Generating metrics plots..."
python plot_loss.py
echo "Plots saved to training_metrics.png"