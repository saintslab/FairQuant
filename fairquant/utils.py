from typing import Tuple, Dict, List, Optional
import torch
import numpy as np
from collections import defaultdict


def set_seed(seed: int = 42):
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def per_group_accuracy(logits: torch.Tensor, targets: torch.Tensor, groups: torch.Tensor, num_groups: int) -> Dict[int, float]:
    pred = logits.argmax(dim=1)
    out = {}
    for g in range(num_groups):
        mask = (groups == g)
        if mask.sum().item() == 0:
            out[g] = float('nan')
        else:
            out[g] = (pred[mask] == targets[mask]).float().mean().item()
    return out


def _one_vs_rest(logits: torch.Tensor, pos_class: int) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)[:, pos_class]
    preds = logits.argmax(dim=1)
    yhat = (preds == pos_class).long()
    return probs, yhat


def _dp_eopp_eodds_for_posclass(logits: torch.Tensor, targets: torch.Tensor, groups: torch.Tensor, num_groups: int, pos_class: int):
    _, yhat = _one_vs_rest(logits, pos_class)
    
    dp_rates, tprs, fprs = {}, {}, {}

    for g in range(num_groups):
        mask_g = (groups == g)
        if mask_g.sum().item() == 0:
            continue
        dp_rates[g] = yhat[mask_g].float().mean().item()

        mask_pos = mask_g & (targets == pos_class)
        if mask_pos.sum().item() > 0:
            tprs[g] = (yhat[mask_pos] == 1).float().mean().item()

        mask_neg = mask_g & (targets != pos_class)
        if mask_neg.sum().item() > 0:
            fprs[g] = (yhat[mask_neg] == 1).float().mean().item()

    return {
        "dp_rates": dp_rates,
        "tprs": tprs,
        "fprs": fprs,
    }




def summarize_fairness(
    per_batch_logits: List[torch.Tensor],
    per_batch_targets: List[torch.Tensor],
    per_batch_groups: List[torch.Tensor],
    num_groups: int,
    num_classes: int,
    positive_class: Optional[int] = None,
    compute_parity_gaps: bool = True
) -> Dict:
    logits = torch.cat(per_batch_logits, dim=0)
    targets = torch.cat(per_batch_targets, dim=0)
    groups = torch.cat(per_batch_groups, dim=0)
    pred = logits.argmax(dim=1)

    group_details = {}
    acc_vals = []

    for g in range(num_groups):
        mask = (groups == g)
        n_samples = mask.sum().item()
        detail = {"samples": n_samples}
        if n_samples > 0:
            acc = (pred[mask] == targets[mask]).float().mean().item()
            acc_vals.append(acc)
            detail['acc'] = acc
        group_details[g] = detail

    parity_gaps = {}
    if compute_parity_gaps:
        if positive_class is not None:
            rates = _dp_eopp_eodds_for_posclass(logits, targets, groups, num_groups, positive_class)
            for g, rate in rates["dp_rates"].items(): group_details[g]['dp_rate'] = rate
            for g, rate in rates["tprs"].items(): group_details[g]['tpr'] = rate
            for g, rate in rates["fprs"].items(): group_details[g]['fpr'] = rate
            if rates["dp_rates"]: parity_gaps["dp_gap"] = max(rates["dp_rates"].values()) - min(rates["dp_rates"].values())
            if rates["tprs"]:
                tpr_vals = list(rates["tprs"].values())
                parity_gaps["eopp_gap"] = max(tpr_vals) - min(tpr_vals)
                parity_gaps["eodds_tpr_gap"] = max(tpr_vals) - min(tpr_vals)
            if rates["fprs"]:
                fpr_vals = list(rates["fprs"].values())
                parity_gaps["eodds_fpr_gap"] = max(fpr_vals) - min(fpr_vals)
        else: 
            gaps = defaultdict(list)
            for c in range(num_classes):
                rates = _dp_eopp_eodds_for_posclass(logits, targets, groups, num_groups, c)
                if rates["dp_rates"] and len(rates["dp_rates"]) > 1: gaps["dp_gap"].append(max(rates["dp_rates"].values()) - min(rates["dp_rates"].values()))
                if rates["tprs"] and len(rates["tprs"]) > 1: gaps["eopp_gap"].append(max(rates["tprs"].values()) - min(rates["tprs"].values()))
                if rates["tprs"] and len(rates["tprs"]) > 1: gaps["eodds_tpr_gap"].append(max(rates["tprs"].values()) - min(rates["tprs"].values()))
                if rates["fprs"] and len(rates["fprs"]) > 1: gaps["eodds_fpr_gap"].append(max(rates["fprs"].values()) - min(rates["fprs"].values()))
            parity_gaps = {k: float(np.mean(vs)) if vs else float('nan') for k, vs in gaps.items()}

        fairprune_metrics = _calculate_fairprune_metrics(logits, targets, groups, num_classes, num_groups)
        parity_gaps.update(fairprune_metrics)

    avg_acc = float(np.mean(acc_vals)) if acc_vals else float('nan')
    worst_acc = float(np.min(acc_vals)) if acc_vals else float('nan')
    acc_gap = avg_acc - worst_acc if np.isfinite(avg_acc) and np.isfinite(worst_acc) else float('nan')

    return {
        "overall": {"avg_acc": avg_acc, "worst_acc": worst_acc, "acc_gap": acc_gap, **parity_gaps},
        "groups": group_details,
    }



def _calculate_fairprune_metrics(logits: torch.Tensor, targets: torch.Tensor, groups: torch.Tensor, num_classes: int, num_groups: int):
    predictions = logits.argmax(dim=1)
    
    sum_eopp0_gaps = 0.0
    sum_eopp1_gaps = 0.0
    sum_eodd_gaps = 0.0
    
    valid_classes_count = 0

    for c in range(num_classes):
        tpr_vals = []
        tnr_vals = []
        fpr_vals = []

        for g in range(num_groups):
            mask_g = (groups == g)
            if not mask_g.any():
                continue

            g_targets = targets[mask_g]
            g_preds = predictions[mask_g]

            mask_pos = (g_targets == c)
            if mask_pos.any():
                acc_pos = (g_preds[mask_pos] == c).float().mean().item()
                tpr_vals.append(acc_pos)
            
            mask_neg = (g_targets != c)
            if mask_neg.any():
                acc_neg = (g_preds[mask_neg] != c).float().mean().item()
                tnr_vals.append(acc_neg)
                fpr_vals.append(1.0 - acc_neg)
        
        class_tpr_gap = (max(tpr_vals) - min(tpr_vals)) if len(tpr_vals) >= 2 else 0.0
        class_tnr_gap = (max(tnr_vals) - min(tnr_vals)) if len(tnr_vals) >= 2 else 0.0
        class_fpr_gap = (max(fpr_vals) - min(fpr_vals)) if len(fpr_vals) >= 2 else 0.0
        
        if len(tpr_vals) > 0 or len(tnr_vals) > 0:
            sum_eopp1_gaps += class_tpr_gap
            sum_eopp0_gaps += class_tnr_gap
            sum_eodd_gaps  += (class_tpr_gap + class_fpr_gap)
            valid_classes_count += 1

    if valid_classes_count > 0:
        final_eopp1 = sum_eopp1_gaps / valid_classes_count
        final_eopp0 = sum_eopp0_gaps / valid_classes_count
        final_eodd  = sum_eodd_gaps  / valid_classes_count
    else:
        final_eopp1, final_eopp0, final_eodd = float('nan'), float('nan'), float('nan')

    return {'eopp0': final_eopp0, 'eopp1': final_eopp1, 'eodd': final_eodd}