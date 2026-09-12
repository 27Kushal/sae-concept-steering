#!/usr/bin/env bash
# ==============================================================================
# End-to-End Local Pipeline Verification Script
# Executes small-scale prototyping on Apple Silicon (MPS / CPU fallback)
# ==============================================================================

set -e

# Enable PyTorch MPS fallback for any unsupported Metal ops
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TRANSFORMERLENS_ALLOW_MPS=1

# Use venv python if available, else system python3
if [ -f ".venv/bin/python3" ]; then
    PYTHON=".venv/bin/python3"
else
    PYTHON="python3"
fi

echo "======================================================================"
echo "Running Local End-to-End Pipeline on Apple Silicon (M4)"
echo "Using Python: $($PYTHON --version)"
echo "======================================================================"

# Step 1: Collect small slice of activations
echo ""
echo ">>> Step 1/4: Collecting residual stream activations (small scale)..."
$PYTHON scripts/01_collect_activations.py --scale small --tokens 10000

# Step 2: Train Sparse Autoencoder
echo ""
echo ">>> Step 2/4: Training SAE on cached activations (100 steps)..."
$PYTHON scripts/02_train_sae.py --scale small --steps 100

# Step 3: Systematic Feature Interpretation
echo ""
echo ">>> Step 3/4: Systematic feature interpretation (top + random sample)..."
$PYTHON scripts/03_interpret_features.py \
    --checkpoint checkpoints/small/final \
    --shards_dir data/activations_small \
    --output_report results/feature_interpretability_report.json \
    --num_top 10 \
    --num_random 10

# Step 4: Concept Steering & Quantitative Evaluation
echo ""
echo ">>> Step 4/4: Steering generation and benchmarking vs Difference-of-Means..."
$PYTHON scripts/04_steer_and_evaluate.py \
    --checkpoint checkpoints/small/final \
    --concept python_code \
    --alphas -4.0 0.0 4.0 \
    --output_csv results/evaluation_metrics.csv

echo ""
echo "======================================================================"
echo "Local pipeline completed successfully!"
echo "Generated Artifacts:"
echo " - Activations: data/activations_small/"
echo " - SAE Checkpoint: checkpoints/small/final/"
echo " - Interpretation: results/feature_interpretability_report.json"
echo " - Evaluation Table: results/evaluation_metrics.csv"
echo "======================================================================"
