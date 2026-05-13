#!/bin/bash
#SBATCH --job-name=deeplab_full
#SBATCH --account=rrg-akilan_gpu          # Use the correct allocation
#SBATCH --output=logs/deeplab-%j.out
#SBATCH --error=logs/deeplab-%j.err
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1               # Request H100 GPU
#SBATCH --mem=256G                       # H100 node minimum
#SBATCH --cpus-per-task=16

# Load OpenCV module before virtual environment
module load opencv/4.13.0

# Activate your Python virtual environment
source ~/venvs/myenv/bin/activate

# Go to your working directory
cd /scratch/$USER/DeepLabv3

# Set PyTorch cache
export TORCH_HOME=/scratch/$USER/.cache/torch
mkdir -p "$TORCH_HOME"

# Optional: PyTorch expandable CUDA segments
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Print GPU info
nvidia-smi

# Run your DeepLabv3 training
python init.py \
  --mode train \
  --cuda True \
  --batch-size 8 \
  --grad-accum-steps 2 \
  --epochs 150 \
  --num-classes 19 \
  --dtype cityscapes \
  --resize-height 513 \
  --resize-width 513 \
  --learning-rate 0.007 \
  --weight-decay 1e-4 \
  --optimizer sgd \
  --momentum 0.9 \
  --scheduler poly \
  --use-class-weights True \
  --augment True \
  --early-stop-patience 999 \
  --eval-every 10 \
  --input-path-train datasets/Cityscapes/leftImg8bit/train \
  --label-path-train datasets/Cityscapes/gtFine_trainIds/train \
  --input-path-val datasets/Cityscapes/leftImg8bit/val \
  --label-path-val datasets/Cityscapes/gtFine_trainIds/val

# Generate plots automatically after training completes
echo "Training completed. Generating metrics plots..."
python plot_loss.py
echo "Plots saved to training_metrics.png"
