# FairQuant

Codebase for **fairness-aware mixed-precision quantization** of image classifiers.

The main script (`train.py`) supports:

- **Static mixed-precision assignment** driven by group or class importance scores
- **QAT fine-tuning** after assignment
- **Iterative QAT** (progressively freezes more of the model each iteration)
- **BAQ learnable quantization** (learns per-layer or per-channel bit-widths)

The code reports standard accuracy metrics plus group metrics and parity gaps.

## What is implemented

### Quantization modes (`--quant_mode`)

- `none`
  Full precision baseline.
- `uniform`
  Uniform fake-quantization for all quantizable layers (Conv2d and Linear).
- `fair_static`
  One-shot, importance-guided mixed precision assignment.
- `fair_static_qat`
  Same assignment as `fair_static`, followed by fine-tuning (QAT).
- `baq_learnable`
  Wraps Conv2d and Linear layers in a BAQ-style module with trainable:
  - `b_logit` controlling bit-width (mapped to `[--baq_bit_min, --baq_bit_max]` with STE rounding)

### Importance metrics (`--importance_metric`)

- `gradient`
  Accumulates `|dL/dW|` per group.
- `grape`
  Accumulates `(dL/dW * W)^2` per group.

### Reducers (`--reducer`)

Used to combine importance maps across groups.

- `max`
  Takes the maximum importance across groups.
- `mean`
  Takes the mean across groups.
- `cvar`
  CVaR-style reducer over groups, controlled by `--cvar_alpha`.
- `balanced`
  Normalizes each group's importance map by that group's share of total importance mass, then takes
  the max across groups. Used by FQ-QAT and FQ-BAQ (see `run_baselines.sh`).
- `subtractive`
  Binary-only strategy used in FairQuantize-style experiments: `importance_unprivileged - beta *
  importance_privileged`. Requires exactly two groups and uses `--beta`.

### Granularity (`--granularity`)

- `per_tensor`
  One bit-width per layer.
- `per_channel`
  One bit-width per output channel for Conv2d and Linear.
- `per_param`
  One bit-width per parameter.

## Supported datasets

All datasets are loaded via `fairquant/datasets.py`.

- **Fitzpatrick17k** (`--dataset fitzpatrick17k`)
  Auto-downloads a prepared archive into `./data/Fitzpatrick17k/` if missing.
  `--fitzpatrick_binary_grouping` maps Fitzpatrick skin types to two groups: 1–3 vs 4–6.
- **ISIC 2019** (`--dataset isic2019`)
  Auto-downloads a prepared archive into `./data/ISIC2019_train/` if missing.
  Sensitive attribute is patient sex from the metadata file. Groups are strictly binary, `female`/`male`
  — images with missing/unknown sex metadata are dropped (rather than bucketed into a third group), since
  some reducers (e.g. `subtractive`) require exactly two sensitive groups.
- **CelebA** (`--dataset celeba`)
  Binary classification/fairness task over CelebA face attributes.
  `--target_attribute` (default `Blond_Hair`) selects the classification label; `--sensitive_attribute`
  (default `Male`) selects the group attribute — both must be valid CelebA attribute names.
  Attempts to auto-download via `torchvision.datasets.CelebA` into `./data/celeba/`, but the official
  host is Google Drive, which commonly rate-limits automated downloads. If that happens, download CelebA
  manually (https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and place it under `./data/celeba/` using
  torchvision's standard layout (`img_align_celeba/`, `list_attr_celeba.txt`, `list_eval_partition.txt`,
  `list_bbox_celeba.txt`, `list_landmarks_align_celeba.txt`, `identity_CelebA.txt`), then re-run.
- **FairFace** (`--dataset fairface`)
  Loaded via the Hugging Face `datasets` library (`HuggingFaceM4/FairFace`, `padding=0.25`), cached under
  `./data/fairface_hf_cache/` — no manual download needed. `--target_attribute` (default `age`, 9 bins)
  and `--sensitive_attribute` (default `race`, 7 groups) must be one of `age`, `gender`, `race`.
  `gender` is the only inherently binary attribute; reducers that require exactly two groups (e.g.
  `subtractive`) should use `--sensitive_attribute gender` instead of the default `race`.

## Models

Model creation is handled in `fairquant/models.py`:

- `resnet18`, `resnet34`, `resnet50` (torchvision)
- Any other model name falls through to `timm.create_model(...)` if `timm` is installed, e.g.
  `tiny_vit_5m_224`, `deit_tiny_patch16_224`, `hiera_base_224.mae_in1k_ft_in1k`.

`--batch_size` defaults to 128, except it defaults to 64 automatically when `--model` contains `hiera`
(large memory footprint) — pass `--batch_size` explicitly to override either way. `run_baselines.sh`
applies the same default when its `$MODEL` argument matches `*hiera*`.

## Installation

This repository is intended to be run from the repo root so Python can import the `fairquant` package. Run the following commands before training:

```

# from the repo root
python -m venv .venv
source .venv/bin/activate   # Windows: .\.venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt

```

## Quickstart

### 1) Pre-train a full precision baseline

`pretrain.py` saves checkpoints to `./checkpoints/`.

```bash
python pretrain.py   --dataset fitzpatrick17k   --model resnet18   --epochs 5
```

### 2) Run quantization experiments

All experiment outputs go to `./results/<timestamp>_<dataset>_<model>_<quant_mode>/` unless `--run_name` is set.

#### One-shot Fair Static QAT

```bash
python train.py   --dataset fitzpatrick17k   --model resnet18   --checkpoint_path ./checkpoints/resnet18_fitzpatrick17k_pretrained.pt   --quant_mode fair_static_qat   --granularity per_channel   --importance_on_sensitive_groups   --importance_metric grape   --reducer balanced   --fairness_loss_lambda 0.5   --quant_bits 2 4 8   --quant_levels 0.2 0.4 0.4   --ft_epochs 10
```

The `grape` metric + `balanced` reducer + `--fairness_loss_lambda` combination above is FQ-QAT (this
repo's own method, see `run_baselines.sh`). Swap in `--importance_metric grape --reducer max` (and drop
`--fairness_loss_lambda`) to reproduce the FairGRAPE baseline instead.

#### Iterative QAT

Progressively freezes more units each iteration until the final mix defined by `--quant_bits/--quant_levels` is reached.

```bash
python train.py   --dataset fitzpatrick17k   --model resnet18   --checkpoint_path ./checkpoints/resnet18_fitzpatrick17k_pretrained.pt   --quant_mode fair_static_qat   --iterative_qat   --iterations 5   --ft_epochs 2   --importance_on_sensitive_groups   --importance_metric grape   --reducer balanced   --quant_bits 2 4 8   --quant_levels 0.2 0.4 0.4
```

#### BAQ learnable bits

Starts from a `balanced`-reducer importance-based initialization (the `b_logit_init` warm start), then
learns bits jointly with weights during fine-tuning. Base weights use the same `ft_lr * 0.1`-scaled
learning rate as the FP32 and FQ-QAT fine-tuning runs (matching the paper's "same learning-rate policy
as the full-precision run"); the bit-width logits use a separately-tuned, higher `ft_lr * 10`. The
`--baq_lambda_b` regularizer pulls logits back toward their informed initialization, not toward `bit_min`.

```bash
python train.py   --dataset fitzpatrick17k   --model resnet18   --checkpoint_path ./checkpoints/resnet18_fitzpatrick17k_pretrained.pt   --quant_mode baq_learnable   --granularity per_channel   --importance_on_sensitive_groups   --importance_metric grape   --reducer balanced   --quant_bits 2 3 4 5 6 7 8   --baq_bit_min 2   --baq_bit_max 8   --baq_lambda_b 1e-2   --fairness_loss_lambda 0.5   --ft_epochs 10
```

#### Running on CelebA

Same flags as any other dataset — just switch `--dataset` and pick a target/sensitive attribute pair. Without `--checkpoint_path`, `train.py` runs `--epochs` of initial full-precision training first.

```bash
python train.py   --dataset celeba   --model resnet18   --target_attribute Blond_Hair   --sensitive_attribute Male   --epochs 5   --quant_mode fair_static_qat   --granularity per_channel   --importance_on_sensitive_groups   --importance_metric grape   --reducer balanced   --fairness_loss_lambda 0.5   --quant_bits 2 4 8   --quant_levels 0.2 0.4 0.4   --ft_epochs 10
```

`--target_attribute`/`--sensitive_attribute` default to `None` and are auto-filled per dataset if
omitted: `Blond_Hair`/`Male` for CelebA, `age`/`race` for FairFace (not applicable to
fitzpatrick17k/isic2019).

## Key CLI arguments (train.py)

Data and run control:

- `--dataset` `{fitzpatrick17k, isic2019, celeba, fairface}`
- `--data_root` (default `./data`)
- `--model`
- `--checkpoint_path` (optional)
- `--run_name` (optional)
- `--train_subset`, `--test_subset` (float fraction or integer count)

Fairness evaluation:

- `--positive_class <int>`
  Enables DP rate, TPR, FPR, TNR, and gap metrics for one chosen class.
- `--no_parity_gaps`
  Skips DP/EOpp/EOdds gaps.

Static assignment and QAT:

- `--granularity` `{per_tensor, per_channel, per_param}`
- `--importance_metric` `{gradient, grape}`
- `--importance_on_sensitive_groups`
- `--reducer` `{max, mean, cvar, balanced, subtractive}`
- `--cvar_alpha`
- `--beta` (used by `subtractive`)
- `--quant_bits <int ...>`
- `--quant_levels <float ...>` (should sum to 1)
- `--ft_epochs`, `--ft_lr`

Iterative QAT:

- `--iterative_qat`
- `--iterations`

BAQ learnable:

- `--baq_bit_min`, `--baq_bit_max`
- `--baq_lambda_b`
- `--fairness_loss_lambda`
- `--grad_clip_norm`

## Baseline comparison sweep (`run_baselines.sh`)

```bash
./run_baselines.sh <dataset> <model> <checkpoint_path> [run_prefix]
```

Runs the full comparison suite for one pretrained model/dataset against a shared `--quant_bits 2 4 8
--quant_levels 0.2 0.4 0.4` budget (`QUANT_BITS`/`QUANT_LEVELS` env vars to override) and `--ft_epochs 10`,
except `fairgrape` which uses its own prune-style budget (see below):

| Run | `--importance_metric` | `--reducer` | `--fairness_loss_lambda` | Notes |
|---|---|---|---|---|
| `fp32` | — | — | — | Optional fine-tune only |
| `uniform4` / `uniform8` | — | — | — | No fine-tuning |
| `fairgrape` | `grape` | `max` | — | Inspired by Lin et al.; matches their `(gw·w)^2` importance signal, but approximates their greedy per-layer round-robin pruning with one-shot importance-guided pruning (`--quant_bits 0 32`, `FAIRGRAPE_PRUNE_RATIO` default `0.8`) since the original's layer-by-layer, group-round-robin weight selection isn't implemented |
| `fairquantize` | `grape` | `subtractive` (`--beta`) | — | Reproduces Guo et al.; requires exactly 2 sensitive groups |
| `fq_qat` (5 seeds) | `grape` | `balanced` | `0.5` | This repo's static mixed-precision method |
| `fq_baq` (5 seeds) | `grape` (warm start only) | `balanced` | `0.5` | Learnable bit-widths, range `[2, 8]` |

`fq_qat`/`fq_baq` are the only runs repeated across `SEEDS=(2 42 107 1337 2026)`; the rest run once.
FairGRAPE and FQ-QAT/FQ-BAQ now share the same `grape` importance metric and `importance_on_sensitive_groups`
setup — they differ in reducer (`max` vs `balanced`) and, for FQ-QAT/FQ-BAQ, the added fairness loss
during fine-tuning. For `fairface`, `fairquantize` automatically overrides `--sensitive_attribute` to
`gender` (binary) since its normal default (`race`, 7 groups) is incompatible with `subtractive`; every
other baseline keeps `race` (override via `FAIRFACE_SENSITIVE`/`FAIRFACE_TARGET` env vars).

## Output files

Each run directory includes:

- `training.log`
  Console log with overall metrics and per-group breakdown.
- `final_model.pt`
  Final weights.
- `fairquant_report.txt`
  All CLI args for the run.
- `bit_distribution.csv`
  Per-layer bit histogram, average bits, parameter counts, and estimated reductions.
- `size_report.txt`
  Human-readable summary, plus GOP and effective GOP estimates.
- `bitwidth_percentages.txt`
  Bit-width distribution, channel-weighted and parameter-weighted.

## Comparing runs (plot_radar.py)

`plot_radar.py` scans `results/` and plots a radar chart trading off accuracy, fairness, and
quantization efficiency across runs — one chart per dataset (accuracy scales aren't comparable
across datasets). For each run it reads the final (or, if a run is still in progress, the most
recently logged) evaluation block from `training.log`, plus `size_report.txt` for BOPs/bit-width
stats, and prints a raw-value summary table before plotting.

```bash
python plot_radar.py                              # one radar per dataset found in results/
python plot_radar.py --dataset fitzpatrick17k      # restrict to one dataset
python plot_radar.py --include baseline            # only run dirs whose name contains a substring
python plot_radar.py --exclude smoketest           # drop run dirs matching a substring
python plot_radar.py --results-dir results --out-dir results
```

Radar axes (all oriented "higher is better"):

- **Accuracy** — `avg_acc`
- **Fairness** — `1 - mean(EOpp1, EOpp0, EOdd)`
- **BOPs Efficiency** — effective-GOPs compute reduction %
- **Bit-width Efficiency** — `1 - avg_bits/32`

Runs missing size/BOPs data (e.g. still training) are skipped from the chart with a `[warn]`
rather than plotted as zero. Output is saved to `--out-dir` (default `results/`) as
`radar_<dataset>.png`.


