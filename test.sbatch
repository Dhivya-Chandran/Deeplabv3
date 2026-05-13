#!/bin/bash
#SBATCH --job-name=deeplab_test
#SBATCH --account=def-akilan
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Load modules and activate virtualenv (adjust as needed)
module load python opencv
VENV_DIR="$HOME/venvs/myenv"
if [ ! -f "$VENV_DIR/bin/activate" ]; then
  echo "ERROR: Virtual environment not found at $VENV_DIR"
  exit 1
fi
source "$VENV_DIR/bin/activate"

cd /scratch/$USER/DeepLabv3

export TORCH_HOME=/scratch/dchandr4/.cache/torch
mkdir -p "$TORCH_HOME"
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "GPU INFO"
nvidia-smi
echo

# Configurable variables
NUM_CLASSES=${NUM_CLASSES:-19} # Cityscapes trainIds default
H=${H:-1024}
W=${W:-2048}
MULTI_SCALE_TEST=${MULTI_SCALE_TEST:-True}
FLIP_TEST=${FLIP_TEST:-True}
TTA_SCALES=${TTA_SCALES:-0.5 0.75 1.0 1.25 1.5 2.0}
BACKBONE=${BACKBONE:-resnet101}
CKPT=${CKPT:-best_miou_model_${BACKBONE}.pth}   # set CKPT env var to evaluate a different checkpoint.
OUTPUT_STRIDE=${OUTPUT_STRIDE:-16}
OUT_ROOT=${OUT_ROOT:-/scratch/$USER/DeepLabv3/deeplap_test}
NUM_SAMPLES=${NUM_SAMPLES:-5}

TEST_OUT_DIR="$OUT_ROOT/test_split"
mkdir -p "$TEST_OUT_DIR"

echo "Evaluating checkpoint: $CKPT"
echo "Saving outputs under: $OUT_ROOT"

# ---- Test split ----
python init.py \
  --mode test \
  --cuda True \
  --backbone "$BACKBONE" \
  --output-stride "$OUTPUT_STRIDE" \
  --num-classes $NUM_CLASSES \
  --dtype cityscapes \
  --resize-height $H \
  --resize-width $W \
  -m $CKPT \
  --image-path datasets/Cityscapes/leftImg8bit/test \
  --label-path-test datasets/Cityscapes/gtFine_trainIds/test \
  --multi-scale-test $MULTI_SCALE_TEST \
  --flip-test $FLIP_TEST \
  --tta-scales $TTA_SCALES \
  --out-dir "$TEST_OUT_DIR" \
  --num-samples $NUM_SAMPLES

echo "Finished test split. Output in: $TEST_OUT_DIR"


