# train.py

import argparse
from typing import Optional, List, Dict, Optional, Union 
import torch
import torch.nn as nn
from torch.optim import AdamW, lr_scheduler
from tqdm import tqdm
import os
import pandas as pd
import logging
from datetime import datetime
import sys
import numpy as np
from copy import deepcopy
import torch.nn.functional as F
from dataclasses import dataclass
import logging

from fairquant.datasets import get_dataloaders
from fairquant.models import get_model
from fairquant.quantize import (
    compute_groupwise_importance,
    compute_groupwise_importance_grape,
    reduce_across_groups,
    assign_bits,
    apply_weight_quantization,
    uniform_assignment,
    count_quantizable_params,
    _collect_module_meta,
    _flatten_importance_for_assignment,
    _zero_like_params, 
    calculate_bops, 
    BAQModule, 
    apply_baq_quantization, 
    collect_baq_parameters, 
    collect_baq_regularization_params
)
from fairquant.utils import set_seed, summarize_fairness


def write_bit_report(model, output_dir, bops_map):
    from fairquant.quantize import collect_bit_distribution
    rows, totals = collect_bit_distribution(model, bops_map)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "bit_distribution.csv"), index=False)
    with open(os.path.join(output_dir, "size_report.txt"), "w") as f:
        f.write("Per-layer bit distribution and size estimate\n")
        for r in rows:
            f.write(
                f"{r['layer']:>40s} | {r['type']:<8s} | avg_bits={r['avg_bits']:.2f} | "
                f"params={r['params']} | reduction={r['reduction_pct']:.1f}% | hist={r['hist']}\n"
            )
        f.write("\n")
        f.write(
            f"TOTAL size ~ {totals['model_size_mb']:.2f} MB "
            f"(baseline {totals['baseline_size_mb']:.2f} MB) "
            f"reduction {totals['reduction_pct']:.1f}%\n"
        )
        # +++ ADDED: BOPs Reporting +++
        f.write(
            f"TOTAL GOPs: {totals['total_gops']:.2f} | "
            f"Effective GOPs: {totals['total_effective_gops']:.2f} | "
            f"Computation Reduction: {100.0 * (1.0 - totals['total_effective_gops'] / max(1e-9, totals['total_gops'])):.1f}%\n"
        )
    logging.info(
        f"[bit-report] saved reports to {output_dir}; "
        f"total size {totals['model_size_mb']:.2f} MB, "
        f"reduction {totals['reduction_pct']:.1f}%, "
        f"Effective GOPs: {totals['total_effective_gops']:.2f}"
    )

def write_bitwidth_percentages(model, output_dir):
    from fairquant.quantize import bitwidth_percentages
    pct_channels = bitwidth_percentages(model, "channels")
    pct_params = bitwidth_percentages(model, "params")
    with open(os.path.join(output_dir, "bitwidth_percentages.txt"), "w") as f:
        f.write("Bit-width distribution (percent)\n")
        f.write("Channels-weighted:\n")
        for b, p in pct_channels.items(): f.write(f"  {b}b: {p:.2f}%\n")
        f.write("Params-weighted:\n")
        for b, p in pct_params.items(): f.write(f"  {b}b: {p:.2f}%\n")
    logging.info("[bitwidth %] channels: " + ", ".join([f"{b}b={p:.1f}%" for b, p in pct_channels.items()]))
    logging.info("[bitwidth %] params:   " + ", ".join([f"{b}b={p:.1f}%" for b, p in pct_params.items()]))

def log_evaluation_details(eval_summary: Dict, group_names: List[str], positive_class: Optional[int], loss: float):
    """Formats and logs the detailed evaluation results."""
    overall = eval_summary['overall']
    
    logging.info(f"OVERALL RESULTS: val_loss={loss:.4f}")
    logging.info(f"   Accuracy:    avg={overall['avg_acc']:.3f} | worst={overall['worst_acc']:.3f} | gap={overall['acc_gap']:.3f}")
    
    if 'eopp0' in overall:
        logging.info(f"   Fairness (Mean across classes):")
        logging.info(f"     EOpp1 (TPR Gap): {overall.get('eopp1', float('nan')):.4f}")
        logging.info(f"     EOpp0 (TNR Gap): {overall.get('eopp0', float('nan')):.4f}")
        logging.info(f"     EOdd  (TPR+FPR): {overall.get('eodd', float('nan')):.4f}")

    if positive_class is not None and 'dp_gap' in overall:
        logging.info(f"   One-vs-Rest (Class {positive_class}):")
        logging.info(f"     DP Gap: {overall.get('dp_gap', float('nan')):.4f}")

    logging.info("-" * 60)
    header = f"{'ID':>2s} | {'Group Name':<15s} | {'Samples':>7s} | {'Acc':>6s}"
    has_rates = positive_class is not None and any('tpr' in d for d in eval_summary['groups'].values())
    if has_rates:
        header += f" | {'TPR':>5s} | {'FPR':>5s} | {'TNR':>5s}"
    logging.info(header)
    
    for g_id, details in eval_summary['groups'].items():
        name = group_names[g_id] if 0 <= g_id < len(group_names) else f"Group_{g_id}"
        name = (name[:13] + '..') if len(name) > 15 else name
        
        acc = f"{details.get('acc', float('nan')):.3f}"
        count = details['samples']
        log_line = f"{g_id:2d} | {name:<15s} | {count:7d} | {acc:>6s}"
        
        if has_rates:
            tpr = f"{details.get('tpr', float('nan')):.3f}"
            fpr = f"{details.get('fpr', float('nan')):.3f}"
            tnr = f"{(1.0 - details.get('fpr', float('nan'))):.3f}" if 'fpr' in details else "n/a"
            log_line += f" | {tpr:>5s} | {fpr:>5s} | {tnr:>5s}"
        
        logging.info(log_line)
    logging.info("-" * 60)


def train_one_epoch(model, loader, device, criterion, optimizer, 
                    fairness_loss_lambda=0.0, bitrate_lambda=0.0, bit_target_avg=0.0, 
                    quant_mode="none", grad_clip_norm=0.0):
    model.train()
    total, count = 0.0, 0
    pbar = tqdm(loader, desc="train", leave=False)
    for x, y, g in pbar:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        
        total_loss = criterion(logits, y)

        if quant_mode in ('baq_learnable') and fairness_loss_lambda > 0:
            groups = g.to(device)
            unique_groups = torch.unique(groups).tolist()
            if len(unique_groups) >= 2:
                group_losses = [criterion(logits[groups == gg], y[groups == gg]) for gg in unique_groups if (groups == gg).sum() > 0]
                if len(group_losses) >= 2:
                    total_loss += fairness_loss_lambda * (torch.stack(group_losses).max() - torch.stack(group_losses).min())
        
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        
        optimizer.step()
        
        total += total_loss.item() * x.size(0)
        count += x.size(0)
        pbar.set_postfix(loss=total / max(1, count))
    return total / max(1, count)



@torch.no_grad()
def evaluate(model, loader, device, num_groups: int, num_classes: int, positive_class: Optional[int], compute_parity_gaps: bool = True):
    model.eval()
    total, count = 0.0, 0
    per_logits, per_targets, per_groups = [], [], []
    criterion = nn.CrossEntropyLoss()
    for x, y, g in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total += loss.item() * x.size(0)
        count += x.size(0)
        per_logits.append(logits.cpu()); per_targets.append(y.cpu()); per_groups.append(g.cpu())
    avg_loss = total / max(1, count)
    summary = summarize_fairness(per_logits, per_targets, per_groups, num_groups, num_classes, positive_class, compute_parity_gaps)
    return avg_loss, summary

def main():
    parser = argparse.ArgumentParser(description="FairQuant: Fairness-Aware Quantization")
    parser.add_argument("--dataset", type=str, default="fitzpatrick17k", choices=["celeba", "fitzpatrick17k", "isic2019"])
    parser.add_argument("--target_attribute", type=str, default="Blond_Hair", help="The target attribute for CelebA classification.")
    parser.add_argument("--sensitive_attribute", type=str, default="Male", help="The sensitive attribute for CelebA fairness analysis.")
    parser.add_argument("--fitzpatrick_binary_grouping", action="store_true", help="If set, groups Fitzpatrick17k into light (1-3) and dark (4-6) skin tones.")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument(
        "--model", 
        type=str, 
        default="resnet18",
        choices=["resnet18", "resnet34", "resnet50", "vgg11","vgg16", "vgg19", "tiny_vit_5m_224", "deit_tiny_patch16_224"]
    )
    parser.add_argument("--epochs", type=int, default=1, help="Epochs for initial pre-training if no checkpoint is provided.")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")))
    parser.add_argument("--train_subset", type=float, default=None)
    parser.add_argument("--test_subset", type=float, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to a pre-trained model checkpoint.")
    parser.add_argument("--positive_class", type=int, default=None)
    parser.add_argument("--no_parity_gaps", action="store_true", help="If set, disables fairness parity gap calculations.")
    parser.add_argument("--quant_mode", type=str, choices=["none", "uniform", "fair_static", "fair_static_qat", "baq_learnable"], default="fair_static")
    parser.add_argument("--granularity", type=str, choices=["per_channel", "per_tensor", "per_param"], default="per_channel")
    parser.add_argument("--importance_on_sensitive_groups", action="store_true", help="Calculate importance on sensitive groups instead of target classes.")
    parser.add_argument("--importance_metric", type=str, choices=["gradient", "grape"], default="gradient", help="The metric for importance calculation ('grape' is from FairGRAPE paper).")
    parser.add_argument("--quant_levels", type=float, nargs="+", default=[0.5, 0.4, 0.1])
    parser.add_argument("--quant_bits", type=int, nargs="+", default=[4, 8, 16], help="Bit widths to assign. Use '0' to enable pruning.")
    parser.add_argument("--uniform_bit", type=int, default=8)
    parser.add_argument("--reducer", type=str, choices=["max", "mean", "cvar", "subtractive", "balanced"], default="max")
    parser.add_argument("--beta", type=float, default=1.0, help="Beta value for the 'subtractive' reducer.")
    parser.add_argument("--cvar_alpha", type=float, default=0.1)
    parser.add_argument("--calib_batches", type=int, default=50)
    parser.add_argument("--ft_epochs", type=int, default=1)
    parser.add_argument("--fairness_loss_lambda", type=float, default=0.0)
    parser.add_argument("--iterative_qat", action="store_true", help="If set, performs iterative QAT where a small portion of the model is quantized and retrained in a loop.")
    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations for iterative QAT.")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0, help="Maximum norm for gradient clipping to prevent exploding gradients.")
    parser.add_argument("--ft_lr", type=float, default=None, help="Separate learning rate for the fine-tuning stage. If None, uses the main 'lr'.")
    parser.add_argument("--run_name", type=str, default=None, help="Optional, unique name for the run to organize results.")
    parser.add_argument("--baq_bit_min", type=int, default=2, help="Minimum bit-width for BAQ learnable bits.")
    parser.add_argument("--baq_bit_max", type=int, default=16, help="Maximum bit-width for BAQ learnable bits.")
    parser.add_argument("--baq_lambda_b", type=float, default=0.0, help="L2 regularization lambda for the bit-width logits")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for independent runs.")
    
    args = parser.parse_args()

    print(f"Setting random seed to: {args.seed}")
    set_seed(args.seed)
    device = torch.device(args.device)

    if args.run_name:
        output_dir = os.path.join("results", args.run_name)
    else:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        output_dir = os.path.join("results", f"{timestamp}_{args.dataset}_{args.model}_{args.quant_mode}")

    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', handlers=[logging.FileHandler(os.path.join(output_dir, 'training.log')), logging.StreamHandler(sys.stdout)])
    logging.info(f"Starting new run. Results will be saved to: {output_dir}")
    logging.info(f"Run arguments: {vars(args)}")

    train_loader, test_loader, num_classes, class_names, num_groups, group_names = get_dataloaders(
        dataset=args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        train_subset=args.train_subset,
        test_subset=args.test_subset,
        target_attribute=args.target_attribute,
        sensitive_attribute=args.sensitive_attribute,
        fitzpatrick_binary_grouping=args.fitzpatrick_binary_grouping
    )
    model = get_model(args.model, num_classes=num_classes, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()

    if args.checkpoint_path and os.path.isfile(args.checkpoint_path):
        logging.info(f"Loading pre-trained model from: {args.checkpoint_path}")
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    elif args.checkpoint_path: logging.warning(f"Checkpoint file not found at {args.checkpoint_path}. Starting fresh.")
    else:
        logging.info("No checkpoint provided. Running initial training phase...")
        optimizer = AdamW(model.parameters(), lr=args.lr)
        for epoch in range(args.epochs):
            train_loss = train_one_epoch(model, train_loader, device, criterion, optimizer)
            val_loss, val_groups = evaluate(model, test_loader, device, num_groups, num_classes, args.positive_class, not args.no_parity_gaps)
            logging.info(f"\n--- [Initial Training Epoch {epoch+1}/{args.epochs}] ---")
            log_evaluation_details(val_groups, group_names, args.positive_class, val_loss)

    if args.quant_mode == "none":
        final_loss, final_groups = evaluate(model, test_loader, device, num_groups, num_classes, args.positive_class, not args.no_parity_gaps)
        logging.info(f"\n--- [FINAL RESULTS: Baseline (none)] ---")
        log_evaluation_details(final_groups, group_names, args.positive_class, final_loss)
        logging.info("Calculating model operations (BOPs)...")
        bops_map = calculate_bops(model, input_size=(1, 3, args.image_size, args.image_size))
        write_bit_report(model, output_dir, bops_map)

    elif args.quant_mode == "uniform":
        assignment = uniform_assignment(model, args.uniform_bit, args.granularity)
        model = apply_weight_quantization(model, assignment, args.granularity).to(device)
        final_loss, final_groups = evaluate(model, test_loader, device, num_groups, num_classes, args.positive_class, not args.no_parity_gaps)
        logging.info(f"\n--- [FINAL RESULTS: Uniform {args.uniform_bit}b] ---")
        log_evaluation_details(final_groups, group_names, args.positive_class, final_loss)
        logging.info("Calculating model operations (BOPs)...")
        bops_map = calculate_bops(model, input_size=(1, 3, args.image_size, args.image_size))

        write_bit_report(model, output_dir, bops_map)
        write_bitwidth_percentages(model, output_dir)


    elif args.quant_mode in ("fair_static", "fair_static_qat"):
        use_sensitive = args.importance_on_sensitive_groups and args.dataset != "celeba"
        groups_for_importance = num_groups if use_sensitive else num_classes
        importance_func = compute_groupwise_importance_grape if args.importance_metric == 'grape' else compute_groupwise_importance

        if args.iterative_qat and args.quant_mode == 'fair_static_qat':
            logging.info(f"STARTING ITERATIVE QAT: {args.iterations} iterations, {args.ft_epochs} fine-tuning epochs per iteration.")
            
            training_model = deepcopy(model)
            
            frozen_assignment = {}
            meta_for_init = _collect_module_meta(training_model, args.granularity)
            for key, meta_item in meta_for_init.items():
                w_shape = meta_item.module.weight.shape
                if args.granularity == 'per_channel':
                    num_units = w_shape[0]
                    shape = (num_units,)
                elif args.granularity == 'per_param':
                    num_units = w_shape.numel()
                    shape = w_shape
                else: # per_tensor
                    num_units = 1
                    shape = (1,)
                frozen_assignment[key] = torch.full((num_units,), 32, dtype=torch.long).reshape(shape)
            
            training_model = apply_weight_quantization(training_model, frozen_assignment, args.granularity)
            total_quantizable_units = count_quantizable_params(training_model, args.granularity)
            
            for i in range(args.iterations):
                logging.info(f"\n--- Iteration {i+1}/{args.iterations} ---")
                
                logging.info("1. Analyzing model and calculating importance...")
                analysis_model = deepcopy(model)
                importance_by_group = importance_func(analysis_model, train_loader, device, groups_for_importance, criterion, args.calib_batches, use_sensitive)

                if args.reducer == 'subtractive':
                    if groups_for_importance != 2: raise ValueError("Subtractive reducer requires exactly two groups.")
                    imp_unprivileged, imp_privileged = importance_by_group.get(0, {}), importance_by_group.get(1, {})
                    reduced = {k: imp_unprivileged.get(k, 0) - args.beta * imp_privileged.get(k, 0) for k in imp_unprivileged}
                else:
                    reduced = reduce_across_groups(importance_by_group, reducer=args.reducer, cvar_alpha=args.cvar_alpha)

                meta = _collect_module_meta(analysis_model, args.granularity)
                flat_imp, index_map = _flatten_importance_for_assignment(reduced, meta)
                
                num_already_frozen = 0
                for unit_idx, (key, local_idx) in enumerate(index_map):
                    is_frozen = (frozen_assignment[key].view(-1)[local_idx] != 32)
                    if is_frozen:
                        flat_imp[unit_idx] = float('inf')
                        num_already_frozen += 1
                
                target_fraction = sum(args.quant_levels) * (i + 1) / args.iterations
                num_to_freeze_this_iter = int(total_quantizable_units * target_fraction) - num_already_frozen
                num_to_freeze_this_iter = max(0, num_to_freeze_this_iter)

                logging.info(f"2. Selecting {num_to_freeze_this_iter} new units to quantize/prune...")
                sorted_indices = torch.argsort(flat_imp)
                
                ideal_assignment = assign_bits(analysis_model, reduced, args.quant_levels, args.quant_bits, args.granularity)
                
                newly_frozen_count = 0
                for idx in sorted_indices:
                    if newly_frozen_count >= num_to_freeze_this_iter or flat_imp[idx] == float('inf'): break
                    key, local_idx = index_map[idx]
                    
                    current_val_flat = frozen_assignment[key].view(-1)
                    if current_val_flat[local_idx] == 32:
                        target_bit = ideal_assignment[key].view(-1)[local_idx]
                        current_val_flat[local_idx] = target_bit
                        newly_frozen_count += 1
                
                logging.info(f"3. Applying new assignment and fine-tuning for {args.ft_epochs} epochs...")
                ft_lr = args.ft_lr if args.ft_lr is not None else args.lr
                training_model = apply_weight_quantization(training_model, frozen_assignment, args.granularity)
                optimizer = AdamW(filter(lambda p: p.requires_grad, training_model.parameters()), lr=ft_lr * 0.1)
                for epoch in range(args.ft_epochs):
                    train_one_epoch(training_model, train_loader, device, criterion, optimizer)

                val_loss, val_groups = evaluate(training_model, test_loader, device, num_groups, num_classes, args.positive_class, not args.no_parity_gaps)
                logging.info(f"--- [End of Iteration {i+1}/{args.iterations}] Evaluation ---")
                log_evaluation_details(val_groups, group_names, args.positive_class, val_loss)
            
            model = training_model
            
        else: # ONE-SHOT QAT
            if args.iterative_qat: logging.warning("Iterative QAT only supported for 'fair_static_qat' mode. Falling back to one-shot.")
            logging.info(f"Using one-shot {args.quant_mode} with reducer: '{args.reducer}' and importance metric: '{args.importance_metric}'")
            importance_by_group = importance_func(model, train_loader, device, groups_for_importance, criterion, args.calib_batches, use_sensitive)
            if args.reducer == 'subtractive':
                if groups_for_importance != 2: raise ValueError("The 'subtractive' reducer requires exactly two groups.")
                imp_unprivileged = importance_by_group.get(0, _zero_like_params(model))
                imp_privileged = importance_by_group.get(1, _zero_like_params(model))
                reduced = {key: imp_unprivileged[key] - args.beta * imp_privileged.get(key, 0) for key in imp_unprivileged}
            else:
                reduced = reduce_across_groups(importance_by_group, reducer=args.reducer, cvar_alpha=args.cvar_alpha)
            assignment = assign_bits(model, reduced, args.quant_levels if args.reducer != 'subtractive' else args.quant_levels[::-1], args.quant_bits, args.granularity)
            model = apply_weight_quantization(model, assignment, args.granularity).to(device)
            if args.quant_mode == "fair_static_qat":
                logging.info("Starting one-shot QAT fine-tuning.")
                ft_lr = args.ft_lr if args.ft_lr is not None else args.lr
                optimizer = AdamW(model.parameters(), lr=ft_lr * 0.1)
                for epoch in range(args.ft_epochs):
                    train_one_epoch(model, train_loader, device, criterion, optimizer)
                    val_loss, val_groups = evaluate(model, test_loader, device, num_groups, num_classes, args.positive_class, not args.no_parity_gaps)
                    logging.info(f"\n--- [QAT Epoch {epoch+1}/{args.ft_epochs}] ---")
                    log_evaluation_details(val_groups, group_names, args.positive_class, val_loss)
        
        final_loss, final_groups = evaluate(model, test_loader, device, num_groups, num_classes, args.positive_class, not args.no_parity_gaps)
        logging.info(f"\n--- [FINAL RESULTS: {args.quant_mode}] ---")
        log_evaluation_details(final_groups, group_names, args.positive_class, final_loss)
        logging.info("Calculating model operations (BOPs)...")
        bops_map = calculate_bops(model, input_size=(1, 3, args.image_size, args.image_size))
        write_bit_report(model, output_dir, bops_map); write_bitwidth_percentages(model, output_dir)
    
    
    elif args.quant_mode == "baq_learnable":
        if args.granularity == "per_param":
            logging.warning("BAQ learnable mode with 'per_param' granularity can be unstable and is not recommended. Consider 'per_tensor' or 'per_channel'.")

        logging.info("Calculating group-wise importance to initialize BAQ bit-widths...")
        use_sensitive = args.importance_on_sensitive_groups and args.dataset != "celeba"
        groups_for_importance = num_groups if use_sensitive else num_classes
        importance_func = compute_groupwise_importance_grape if args.importance_metric == 'grape' else compute_groupwise_importance

        importance_by_group = importance_func(model, train_loader, device, groups_for_importance, criterion, args.calib_batches, use_sensitive)
        reduced_importance = reduce_across_groups(importance_by_group, reducer=args.reducer, cvar_alpha=args.cvar_alpha)
        
        initial_assignment = assign_bits(model, reduced_importance, args.quant_levels, args.quant_bits, args.granularity)
        logging.info("Generated initial bit-width targets from importance scores.")

        model = apply_baq_quantization(
            model, args.baq_bit_min, args.baq_bit_max, args.granularity, initial_assignment=initial_assignment
        ).to(device)
        logging.info(f"Applied BAQ learnable wrappers with fairness-aware initialization. Granularity: {args.granularity}. Bit range: [{args.baq_bit_min}, {args.baq_bit_max}].")
        
        baq_params = collect_baq_parameters(model)
        baq_param_ids = {id(p) for p in baq_params}
        
        base_params = [p for p in model.parameters() if p.requires_grad and id(p) not in baq_param_ids]
        
        ft_lr = args.ft_lr if args.ft_lr is not None else args.lr
        optimizer = AdamW([
            {"params": base_params},  # Group 0: Base parameters use the standard ft_lr
            {"params": baq_params, "lr": ft_lr * 10}  # Group 1: BAQ logits get a 10x higher LR
        ], lr=ft_lr)

        logging.info("Starting fine-tuning for BAQ learnable quantization...")
        
        d_logits, b_logits = collect_baq_regularization_params(model)

        for epoch in range(args.ft_epochs):
            model.train()
            total, count = 0.0, 0
            pbar = tqdm(train_loader, desc=f"train (epoch {epoch+1})", leave=False)
            
            for x, y, g in pbar:
                x, y, g = x.to(device), y.to(device), g.to(device)

                logits = model(x)
                total_loss = criterion(logits, y)

                if args.fairness_loss_lambda > 0:
                    unique_groups = torch.unique(g).tolist()
                    if len(unique_groups) >= 2:
                        group_losses = [criterion(logits[g == gg], y[g == gg]) for gg in unique_groups if (g == gg).sum() > 0]
                        if len(group_losses) >= 2:
                            total_loss += args.fairness_loss_lambda * (torch.stack(group_losses).max() - torch.stack(group_losses).min())

                args.baq_lambda_d = 0.0
                if args.baq_lambda_d > 0 and d_logits:
                    d_reg_loss = torch.stack([p.pow(2).sum() for p in d_logits]).sum()
                    total_loss += args.baq_lambda_d * d_reg_loss

                if args.baq_lambda_b > 0 and b_logits:
                    b_reg_loss = torch.stack([p.pow(2).sum() for p in b_logits]).sum()
                    total_loss += args.baq_lambda_b * b_reg_loss
            

                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                
                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                
                optimizer.step()
                
                total += total_loss.item() * x.size(0)
                count += x.size(0)
                pbar.set_postfix(loss=total / max(1, count))
            
            val_loss, val_groups = evaluate(model, test_loader, device, num_groups, num_classes, args.positive_class, not args.no_parity_gaps)
            logging.info(f"\n--- [BAQ FT Epoch {epoch+1}/{args.ft_epochs}] ---")
            log_evaluation_details(val_groups, group_names, args.positive_class, val_loss)

        final_loss, final_groups = evaluate(model, test_loader, device, num_groups, num_classes, args.positive_class, not args.no_parity_gaps)
        logging.info(f"\n--- [FINAL RESULTS: {args.quant_mode}] ---")
        log_evaluation_details(final_groups, group_names, args.positive_class, final_loss)
        
        logging.info("Calculating model operations (BOPs)...")
        bops_map = calculate_bops(model, input_size=(1, 3, args.image_size, args.image_size))
        write_bit_report(model, output_dir, bops_map)
        write_bitwidth_percentages(model, output_dir)

    report_path = os.path.join(output_dir, "fairquant_report.txt")
    with open(report_path, "w") as f:
        f.write("FairQuant Run Arguments\n" + "="*30 + "\n")
        for key, value in vars(args).items(): f.write(f"{key}: {value}\n")
    
    final_model_path = os.path.join(output_dir, "final_model.pt")
    torch.save(model.state_dict(), final_model_path)
    logging.info(f"Saved final model to: {final_model_path}")
    logging.info(f"Saved final report to: {report_path}")
    logging.info(f"All results for this run are saved in: {output_dir}")

if __name__ == "__main__":
    main()