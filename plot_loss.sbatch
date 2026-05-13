#!/bin/bash
#SBATCH --job-name=plot_loss
#SBATCH --output=logs/plot_loss-%j.out
#SBATCH --error=logs/plot_loss-%j.err
#SBATCH --time=00:10:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1

cd $SLURM_SUBMIT_DIR

# Quick path: if matplotlib is already available, run directly
if python3 -c "import matplotlib" 2>/dev/null; then
	python3 plot_loss.py
	exit 0
fi

# Otherwise create a venv and try to install matplotlib there
mkdir -p logs
if [ -n "$SLURM_TMPDIR" ]; then
	VENV_DIR="$SLURM_TMPDIR/plot_venv"
else
	VENV_DIR="$SLURM_SUBMIT_DIR/plot_venv_$RANDOM"
fi
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Try to install matplotlib in the venv
if command -v pip >/dev/null 2>&1; then
	pip install --no-cache-dir matplotlib || true
fi

# If pip is not present in the venv, attempt user install as a fallback
if ! python3 -c "import matplotlib" 2>/dev/null; then
	echo "matplotlib not available after venv install; trying user install" >&2
	python3 -m pip install --user matplotlib || true
fi

# Run the plotting script (creates loss_plot.png in the submit directory)
python3 plot_loss.py || {
	echo "plot script failed; inspect logs" >&2
	exit 1
}
