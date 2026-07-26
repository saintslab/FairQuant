#!/usr/bin/env bash
# Runs run_baselines.sh for every pretrained checkpoint in checkpoints/,
# inferring --dataset and --model from the checkpoint filename
# (<model>_<dataset>[_<attribute>]_pretrained.pt).
#
# Usage:
#   ./run_all_checkpoints.sh [dataset_filter]
#
# Examples:
#   ./run_all_checkpoints.sh                  # every checkpoint
#   ./run_all_checkpoints.sh fitzpatrick17k   # only fitzpatrick17k checkpoints

set -euo pipefail

CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints}"
DATASET_FILTER="${1:-}"

# Must match train.py's --model / --dataset choices. Longer model names first
# is not required since matching is done with a trailing underscore anchor.
MODELS=(resnet18 resnet34 resnet50 vgg11 vgg16 vgg19 tiny_vit_5m_224 deit_tiny_patch16_224)
DATASETS=(celeba fitzpatrick17k isic2019 fairface)

shopt -s nullglob
for ckpt in "$CHECKPOINT_DIR"/*_pretrained.pt; do
    base="$(basename "$ckpt" _pretrained.pt)"

    model=""
    rest=""
    for m in "${MODELS[@]}"; do
        if [[ "$base" == "${m}_"* ]]; then
            model="$m"
            rest="${base#${m}_}"
            break
        fi
    done
    if [[ -z "$model" ]]; then
        echo "Skipping $ckpt: could not infer model from filename" >&2
        continue
    fi

    dataset=""
    for d in "${DATASETS[@]}"; do
        if [[ "$rest" == "${d}"* ]]; then
            dataset="$d"
            break
        fi
    done
    if [[ -z "$dataset" ]]; then
        echo "Skipping $ckpt: could not infer dataset from filename" >&2
        continue
    fi

    if [[ -n "$DATASET_FILTER" && "$dataset" != "$DATASET_FILTER" ]]; then
        continue
    fi

    echo "=== Running baselines: dataset=$dataset model=$model checkpoint=$ckpt ==="
    bash run_baselines.sh "$dataset" "$model" "$ckpt"
done
