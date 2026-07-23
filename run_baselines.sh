#!/usr/bin/env bash
# Runs the full comparison suite for one pretrained model/dataset:
#   uniform-4, uniform-8, FairGRAPE, FairQuantize, HAWQ, FQ-QAT, FQ-BAQ
# FQ-QAT and FQ-BAQ are repeated over 5 seeds; the rest run once.
#
# Usage:
#   ./run_baselines.sh <dataset> <model> <checkpoint_path> [run_prefix]
#
# Example:
#   ./run_baselines.sh fitzpatrick17k resnet18 checkpoints/resnet18_fitzpatrick17k_pretrained.pt fitz

set -euo pipefail

DATASET="${1:?Usage: $0 <dataset> <model> <checkpoint_path> [run_prefix]}"
MODEL="${2:?Usage: $0 <dataset> <model> <checkpoint_path> [run_prefix]}"
CHECKPOINT="${3:?Usage: $0 <dataset> <model> <checkpoint_path> [run_prefix]}"
PREFIX="${4:-${DATASET}}_${MODEL}"

GRANULARITY="${GRANULARITY:-per_channel}"
QUANT_BITS="${QUANT_BITS:-2 4 8}"
QUANT_LEVELS="${QUANT_LEVELS:-0.2 0.4 0.4}"
FT_EPOCHS="${FT_EPOCHS:-10}"
BETA="${BETA:-1.0}"
HAWQ_HUTCHINSON_SAMPLES="${HAWQ_HUTCHINSON_SAMPLES:-10}"
SEEDS=(2 42 107 1337 2026)

LOG_DIR="${LOG_DIR:-./logs}"
mkdir -p "$LOG_DIR"

COMMON=(--dataset "$DATASET" --model "$MODEL" --checkpoint_path "$CHECKPOINT" --granularity "$GRANULARITY")
if [[ "$DATASET" == "fitzpatrick17k" ]]; then
    COMMON+=(--fitzpatrick_binary_grouping)
fi

run() {
    echo "+ carbontracker --log_dir=${LOG_DIR} python train.py $*"
    carbontracker --log_dir="$LOG_DIR" python train.py "$@"
}

# --- FP32 baseline (no quantization, optional fine-tuning) ---
run "${COMMON[@]}" --quant_mode none --ft_epochs "$FT_EPOCHS" \
    --run_name "${PREFIX}_fp32"

# --- Uniform baselines (no fine-tuning) ---
run "${COMMON[@]}" --quant_mode uniform --uniform_bit 4 \
    --run_name "${PREFIX}_uniform4"

run "${COMMON[@]}" --quant_mode uniform --uniform_bit 8 \
    --run_name "${PREFIX}_uniform8"

# --- FairGRAPE (Lin et al.): (grad*weight)^2 importance, max across sensitive groups ---
run "${COMMON[@]}" --quant_mode fair_static_qat \
    --importance_metric grape --reducer max --importance_on_sensitive_groups \
    --quant_bits $QUANT_BITS --quant_levels $QUANT_LEVELS --ft_epochs "$FT_EPOCHS" \
    --run_name "${PREFIX}_fairgrape"

# --- FairQuantize (Guo et al.): (grad*weight)^2 importance, subtractive across privileged/unprivileged groups ---
run "${COMMON[@]}" --quant_mode fair_static_qat \
    --importance_metric grape --reducer subtractive --beta "$BETA" --importance_on_sensitive_groups \
    --quant_bits $QUANT_BITS --quant_levels $QUANT_LEVELS --ft_epochs "$FT_EPOCHS" \
    --run_name "${PREFIX}_fairquantize"

# --- HAWQ (Hutchinson-trace Hessian importance, not fairness-aware) ---
run "${COMMON[@]}" --quant_mode fair_static_qat \
    --importance_metric hawq --reducer max --hawq_hutchinson_samples "$HAWQ_HUTCHINSON_SAMPLES" \
    --quant_bits $QUANT_BITS --quant_levels $QUANT_LEVELS --ft_epochs "$FT_EPOCHS" \
    --run_name "${PREFIX}_hawq"

# --- FQ-QAT: FairQuant one-shot static assignment + QAT fine-tuning ---
for seed in "${SEEDS[@]}"; do
    run "${COMMON[@]}" --quant_mode fair_static_qat \
        --importance_metric gradient --reducer max --importance_on_sensitive_groups \
        --quant_bits $QUANT_BITS --quant_levels $QUANT_LEVELS --ft_epochs "$FT_EPOCHS" \
        --seed "$seed" --run_name "${PREFIX}_fq_qat_seed${seed}"
done

# --- FQ-BAQ: FairQuant learnable bit-widths ---
for seed in "${SEEDS[@]}"; do
    run "${COMMON[@]}" --quant_mode baq_learnable \
        --importance_metric gradient --reducer max --importance_on_sensitive_groups \
        --quant_bits 2 4 6 8 --quant_levels 0.25 0.25 0.25 0.25 \
        --baq_bit_min 2 --baq_bit_max 8 --baq_lambda_b 1e-2 --fairness_loss_lambda 0.5 \
        --ft_epochs "$FT_EPOCHS" --seed "$seed" --run_name "${PREFIX}_fq_baq_seed${seed}"
done

echo "All runs complete. Results saved under results/${PREFIX}_*"
