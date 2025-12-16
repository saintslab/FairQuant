from typing import Dict, List, Tuple,  Optional, Union 
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import logging

def _iter_target_named_modules(model: nn.Module):
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            yield name, m


class RoundStraightThrough(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

round_ste = RoundStraightThrough.apply

@torch.no_grad()
def _zero_like_params(model: nn.Module) -> Dict[str, torch.Tensor]:
    out = {}
    for name, m in _iter_target_named_modules(model):
        if m.weight.requires_grad:
            out[name] = torch.zeros_like(m.weight, dtype=torch.float32)
    return out

def _accumulate_abs_grad(acc: Dict[str, torch.Tensor], model: nn.Module):
    for name, m in _iter_target_named_modules(model):
        if m.weight.requires_grad and m.weight.grad is not None:
            acc[name].add_(m.weight.grad.detach().abs())

def _accumulate_grape_score(acc: Dict[str, torch.Tensor], model: nn.Module):
    for name, m in _iter_target_named_modules(model):
        if m.weight.requires_grad and m.weight.grad is not None:
            acc[name].add_((m.weight.grad.detach() * m.weight.data.detach()).pow(2))

def compute_groupwise_importance(
    model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device,
    num_groups: int, criterion: nn.Module, calib_batches: int = 50, use_sensitive_groups: bool = False
) -> Dict[int, Dict[str, torch.Tensor]]:
    """Computes group-wise importance using the default |gradient| metric."""
    model.train()
    importance: Dict[int, Dict[str, torch.Tensor]] = {}
    seen_count = 0
    total_samples_to_see = calib_batches * dataloader.batch_size if dataloader.batch_size else calib_batches * 128

    for batch in dataloader:
        if len(batch) == 3:
            images, targets, sensitive_attrs = batch
        else:
            images, targets = batch
            if use_sensitive_groups:
                logging.warning("Expected sensitive groups but dataloader did not provide them. Skipping batch.")
                continue
        
        images, targets = images.to(device), targets.to(device)
        group_tensor = sensitive_attrs.to(device) if use_sensitive_groups else targets
        loop_over_groups = torch.unique(group_tensor).tolist()

        for g in loop_over_groups:
            if g not in importance: importance[g] = _zero_like_params(model)

        logits = model(images)
        for j, g in enumerate(loop_over_groups):
            mask = (group_tensor == g)
            if mask.sum().item() == 0: continue
            loss = criterion(logits[mask], targets[mask])
            retain = (j < len(loop_over_groups) - 1)
            model.zero_grad(set_to_none=True)
            loss.backward(retain_graph=retain)
            _accumulate_abs_grad(importance[g], model)

        seen_count += images.size(0)
        if seen_count >= total_samples_to_see: break
    
    for g in range(num_groups):
        if g not in importance: importance[g] = _zero_like_params(model)
    return importance

def compute_groupwise_importance_grape(
    model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device,
    num_groups: int, criterion: nn.Module, calib_batches: int = 50, use_sensitive_groups: bool = False
) -> Dict[int, Dict[str, torch.Tensor]]:
    """Computes group-wise importance using the FairGRAPE metric: (grad * weight)^2."""
    model.train()
    importance: Dict[int, Dict[str, torch.Tensor]] = {}
    seen_count = 0
    total_samples_to_see = calib_batches * dataloader.batch_size if dataloader.batch_size else calib_batches * 128

    for batch in dataloader:
        if len(batch) == 3:
            images, targets, sensitive_attrs = batch
        else:
            images, targets = batch
            if use_sensitive_groups:
                logging.warning("Expected sensitive groups but dataloader did not provide them. Skipping batch.")
                continue
        
        images, targets = images.to(device), targets.to(device)
        group_tensor = sensitive_attrs.to(device) if use_sensitive_groups else targets
        loop_over_groups = torch.unique(group_tensor).tolist()

        for g in loop_over_groups:
            if g not in importance: importance[g] = _zero_like_params(model)

        logits = model(images)
        for j, g in enumerate(loop_over_groups):
            mask = (group_tensor == g)
            if mask.sum().item() == 0: continue
            loss = criterion(logits[mask], targets[mask])
            retain = (j < len(loop_over_groups) - 1)
            model.zero_grad(set_to_none=True)
            loss.backward(retain_graph=retain)
            _accumulate_grape_score(importance[g], model)

        seen_count += images.size(0)
        if seen_count >= total_samples_to_see: break
    
    for g in range(num_groups):
        if g not in importance: importance[g] = _zero_like_params(model)
    return importance

def reduce_across_groups(
    importance_by_group: Dict[int, Dict[str, torch.Tensor]], reducer: str = "max", cvar_alpha: float = 0.1
) -> Dict[str, torch.Tensor]:
    reducer = reducer.lower()
    keys = list(next(iter(importance_by_group.values())).keys())
    reduced = {}

    if reducer == 'balanced':
        total_importance_per_group = {g: sum(imp.sum() for imp in importance_by_group[g].values()) for g in importance_by_group}
        total_importance_all = sum(total_importance_per_group.values())
        if total_importance_all == 0:
             logging.warning("Total importance across all groups is zero. 'balanced' reducer may not work as expected.")
             target_share = {g: 1.0 / len(importance_by_group) for g in importance_by_group}
        else:
            target_share = {g: total / total_importance_all for g, total in total_importance_per_group.items()}
        
        for k in keys:
            stack = torch.stack([importance_by_group.get(g, {}).get(k, torch.zeros_like(next(iter(importance_by_group.values()))[k])) for g in importance_by_group], dim=0)
            shares = torch.tensor([target_share[g] for g in importance_by_group], device=stack.device)
            while shares.dim() < stack.dim(): shares = shares.unsqueeze(-1)
            normalized_stack = stack / shares.clamp(min=1e-8)
            reduced[k] = normalized_stack.max(dim=0).values
        return reduced

    for k in keys:
        stack = torch.stack([importance_by_group.get(g, {}).get(k, torch.zeros_like(next(iter(importance_by_group.values()))[k])) for g in importance_by_group], dim=0)
        if reducer == "max": val = stack.max(dim=0).values
        elif reducer == "mean": val = stack.mean(dim=0)
        elif reducer == "cvar":
            G = stack.shape[0]
            topk = max(1, int(math.ceil((1.0 - cvar_alpha) * G)))
            flat = stack.view(G, -1)
            top_vals, _ = torch.topk(flat, k=topk, dim=0)
            val = top_vals.mean(dim=0).view_as(stack[0])
        else: raise ValueError(f"Unknown reducer '{reducer}'")
        reduced[k] = val
    return reduced

@dataclass
class ModuleMeta:
    module: nn.Module
    granularity: str

def _collect_module_meta(model: nn.Module, granularity: str) -> Dict[str, ModuleMeta]:
    return {name: ModuleMeta(module=m, granularity=granularity) for name, m in _iter_target_named_modules(model)}

def count_quantizable_params(model: nn.Module, granularity: str) -> int:
    modules = list(m for _, m in _iter_target_named_modules(model))
    if granularity == "per_param": return sum(m.weight.numel() for m in modules)
    elif granularity == "per_channel": return sum(m.weight.shape[0] for m in modules)
    else: return len(modules)

def _flatten_importance_for_assignment(
    importance: Dict[str, torch.Tensor], meta: Dict[str, ModuleMeta]
) -> Tuple[torch.Tensor, List[Tuple[str, int]]]:
    flats, index_map = [], []
    for k, imp in importance.items():
        gran = meta[k].granularity
        if gran == "per_channel":
            if imp.dim() == 4: per_channel = imp.abs().mean(dim=(1,2,3))
            elif imp.dim() == 2: per_channel = imp.abs().mean(dim=1)
            else: raise RuntimeError(f"Unsupported weight shape for per_channel: {imp.shape}")
            flats.append(per_channel.view(-1))
            index_map.extend([(k, i) for i in range(per_channel.numel())])
        elif gran == "per_tensor":
            scalar = imp.abs().mean().view(1)
            flats.append(scalar)
            index_map.append((k, -1))
        elif gran == "per_param":
            flat_imp = imp.view(-1)
            flats.append(flat_imp)
            index_map.extend([(k, i) for i in range(flat_imp.numel())])
        else: raise ValueError(f"Granularity '{gran}' not supported.")
    return torch.cat(flats, dim=0), index_map

def _quantile_thresholds(flat: torch.Tensor, levels: List[float]) -> List[float]:
    if not levels: return []
    if abs(sum(levels) - 1.0) > 1e-5:
        logging.warning(f"quant_levels {levels} do not sum to 1. Normalizing.")
        total = sum(levels)
        levels = [l / total for l in levels]
    cum = [sum(levels[:i+1]) for i in range(len(levels) - 1)]
    if not cum: return []
    flat_cpu, q_cpu = flat.detach().float().cpu(), torch.tensor(cum, dtype=torch.float32)
    return torch.quantile(flat_cpu, q_cpu).tolist()


def assign_bits(
    model: nn.Module, reduced_importance: Dict[str, torch.Tensor], 
    quant_levels: List[float], quant_bits: List[int], granularity: str = "per_channel"
) -> Dict[str, torch.Tensor]:
    assert len(quant_levels) == len(quant_bits), "quant_levels and quant_bits must match length"
    meta = _collect_module_meta(model, granularity)
    flat, index_map = _flatten_importance_for_assignment(reduced_importance, meta)
    
    bits_sorted = sorted(quant_bits)
    thresholds = _quantile_thresholds(flat, quant_levels)

    assigned_flat = torch.empty_like(flat, dtype=torch.int64)
    for i, v in enumerate(flat):
        assigned_bit = bits_sorted[-1] 
        for j, t in enumerate(thresholds):
            if v <= t:
                assigned_bit = bits_sorted[j]
                break
        assigned_flat[i] = assigned_bit

    out, cursor, processed_keys = {}, 0, set()
    i = 0
    while i < len(index_map):
        k, _ = index_map[i]
        if k in processed_keys:
            i += 1; continue
        gran, weight_shape = meta[k].granularity, meta[k].module.weight.shape
        if gran == "per_channel":
            cnt = weight_shape[0]
            out[k] = assigned_flat[cursor:cursor+cnt].clone()
        elif gran == "per_param":
            cnt = meta[k].module.weight.numel()
            out[k] = assigned_flat[cursor:cursor+cnt].clone().reshape(weight_shape)
        else: # per_tensor
            cnt = 1
            out[k] = assigned_flat[cursor:cursor+cnt].clone()
        
        cursor += cnt; i += cnt; processed_keys.add(k)
        
    return out

def uniform_assignment(model: nn.Module, bit: int, granularity: str = "per_channel") -> Dict[str, torch.Tensor]:
    out = {}
    for name, m in _iter_target_named_modules(model):
        if granularity == "per_channel": out[name] = torch.full((m.weight.size(0),), int(bit), dtype=torch.long)
        elif granularity == "per_param": out[name] = torch.full(m.weight.shape, int(bit), dtype=torch.long)
        else: out[name] = torch.as_tensor([int(bit)], dtype=torch.long)
    return out

def _fake_quantize_weight_per_channel(w: torch.Tensor, bits_per_out: torch.Tensor) -> torch.Tensor:
    O = w.size(0)
    assert bits_per_out.numel() == O, f"Bit tensor size {bits_per_out.numel()} does not match output channels {O}"
    wq = torch.empty_like(w)
    unique_bits = bits_per_out.unique(sorted=True).tolist()
    for b_int in unique_bits:
        mask, b = (bits_per_out == b_int), int(b_int)
        w_sub = w[mask]
        if b == 0: deq = torch.zeros_like(w_sub)
        elif b <= 1: deq = w_sub.sign()
        else:
            qmax = (2 ** (b - 1)) - 1
            s = w_sub.abs().amax(dim=tuple(range(1, w_sub.dim())), keepdim=True).clamp(min=1e-8) / qmax
            deq = torch.round(w_sub / s).clamp_(-qmax, qmax) * s
        wq[mask] = (deq - w_sub).detach() + w_sub
    return wq

def _fake_quantize_weight_per_tensor(w: torch.Tensor, b: int) -> torch.Tensor:
    if b == 0: return (torch.zeros_like(w) - w).detach() + w
    if b <= 1: return (w.sign() - w).detach() + w
    qmax = (2 ** (b - 1)) - 1
    s = w.abs().amax().clamp(min=1e-8) / qmax
    q = torch.round(w / s).clamp_(-qmax, qmax)
    deq = q * s
    return (deq - w).detach() + w

def _fake_quantize_weight_per_param(w: torch.Tensor, bits_per_param: torch.Tensor) -> torch.Tensor:
    assert w.shape == bits_per_param.shape
    wq = torch.empty_like(w)
    unique_bits = bits_per_param.unique(sorted=True).tolist()
    for b_int in unique_bits:
        mask, b = (bits_per_param == b_int), int(b_int)
        w_sub = w[mask]
        if b == 0: deq = torch.zeros_like(w_sub)
        elif b <= 1: deq = w_sub.sign()
        else:
            qmax = (2 ** (b - 1)) - 1
            s = w_sub.abs().amax().clamp(min=1e-8) / qmax
            deq = torch.round(w_sub / s).clamp_(-qmax, qmax) * s
        wq[mask] = (deq - w_sub).detach() + w_sub
    return wq

class QuantizedModule(nn.Module):
    def __init__(self, base: nn.Module, bits, granularity: str):
        super().__init__()
        assert isinstance(base, (nn.Conv2d, nn.Linear))
        self.base, self.granularity = base, granularity
        if granularity == "per_channel": self.register_buffer("bits_per_out", bits.clone().long())
        elif granularity == "per_param": self.register_buffer("bits_per_param", bits.clone().long())
        else: self.register_buffer("bits_scalar", torch.as_tensor(int(bits.item())).long())

    def forward(self, x):
        w = self.base.weight
        if self.granularity == "per_channel": wq = _fake_quantize_weight_per_channel(w, self.bits_per_out)
        elif self.granularity == "per_param": wq = _fake_quantize_weight_per_param(w, self.bits_per_param)
        else: wq = _fake_quantize_weight_per_tensor(w, int(self.bits_scalar.item()))
        
        if isinstance(self.base, nn.Conv2d):
            return F.conv2d(x, wq, self.base.bias, self.base.stride, self.base.padding, self.base.dilation, self.base.groups)
        else:
            return F.linear(x, wq, self.base.bias)

def apply_weight_quantization(model: nn.Module, assignment: Dict[str, torch.Tensor], granularity: str = "per_channel") -> nn.Module:
    module_dict = {name: m for name, m in model.named_modules()}
    for key, bits_tensor in assignment.items():
        path = key.split('.')
        parent, child_name = model, ''
        if len(path) > 1:
            parent = module_dict['.'.join(path[:-1])]
            child_name = path[-1]
        else:
            child_name = path[0]

        child = getattr(parent, child_name)
        original_module = child.base if isinstance(child, QuantizedModule) else child
        
        if isinstance(child, QuantizedModule):
            if granularity == "per_channel": child.bits_per_out.copy_(bits_tensor)
            elif granularity == "per_param": child.bits_per_param.copy_(bits_tensor)
            else: child.bits_scalar.copy_(bits_tensor)
        else:
            setattr(parent, child_name, QuantizedModule(original_module, bits_tensor, granularity))
    return model

class BAQModule(nn.Module):
    def __init__(self, base: nn.Module, bit_min: int, bit_max: int, granularity: str, initial_bit: Optional[Union[float, torch.Tensor]] = None):
        super().__init__()
        assert isinstance(base, (nn.Conv2d, nn.Linear))
        self.base = base
        self.bit_min = bit_min
        self.bit_max = bit_max
        self.granularity = granularity

        if self.granularity == "per_channel":
            num_params = base.weight.size(0)
            param_shape = (num_params,)
        else: #per_tensor
            param_shape = (1,)

        self.d_logit = nn.Parameter(torch.full(param_shape, -6.0))
        self.b_logit = nn.Parameter(torch.full(param_shape, 6.0))


        if initial_bit is not None:
            initial_bit_tensor = torch.as_tensor(initial_bit, dtype=torch.float32)

            if not torch.all((self.bit_min <= initial_bit_tensor) & (initial_bit_tensor <= self.bit_max)):
                logging.warning(f"Initial bit(s) for BAQ are outside the allowed range [{self.bit_min}, {self.bit_max}]. Clamping.")
                initial_bit_tensor = torch.clamp(initial_bit_tensor, self.bit_min, self.bit_max)
            
            if self.bit_max > self.bit_min:
                val_for_atanh = (initial_bit_tensor - self.bit_min) / (self.bit_max - self.bit_min)
                val_for_atanh = torch.clamp(val_for_atanh, max=0.9999)
                
                with torch.no_grad():
                    self.b_logit.copy_(torch.atanh(val_for_atanh))


    def hard_bits(self) -> torch.Tensor:
        b_cont = torch.tanh(torch.abs(self.b_logit)) * (self.bit_max - self.bit_min) + self.bit_min
        return torch.round(b_cont)

    def forward(self, x):
        w = self.base.weight

        if self.granularity == "per_channel":
            view_shape = (-1,) + (1,) * (w.dim() - 1)
            d_logit = self.d_logit.view(view_shape)
            b_logit = self.b_logit.view(view_shape)
        else: # per_tensor
            d_logit = self.d_logit
            b_logit = self.b_logit
        
        b_cont = torch.tanh(torch.abs(b_logit)) * (self.bit_max - self.bit_min) + self.bit_min
        b_int = round_ste(b_cont)
        
        if self.granularity == "per_channel":
            w_abs_max = w.detach().abs().amax(dim=tuple(range(1, w.dim())), keepdim=True)
        else: # per_tensor
            w_abs_max = w.detach().abs().max()
            
        d = (1.0 - torch.tanh(torch.abs(d_logit))) * w_abs_max
        
        M = (2 ** (b_int - 1)) - 1
        delta = (w_abs_max - d / 2) / M.clamp(min=1)
        
        w_abs = w.abs()
        w_sign = torch.sign(w)
        
        is_in_deadzone = w_abs < d / 2
        
        c = (w_abs - d / 2) / delta.clamp(min=1e-8)
        
        if M.numel() > 1:
            q = torch.zeros_like(c)
            for i in range(M.shape[0]): # Iterate over channels
                q[i] = round_ste(c[i].clamp(0, M[i].item()))
        else: # Scalar case
             q = round_ste(c.clamp(0, M.item()))

        wq_abs = q * delta + d / 2
        wq_abs[is_in_deadzone] = 0.0
        
        wq = wq_abs * w_sign
        w_quantized = (wq - w).detach() + w
        
        if isinstance(self.base, nn.Conv2d):
            return F.conv2d(x, w_quantized, self.base.bias, self.base.stride, self.base.padding, self.base.dilation, self.base.groups)
        else:
            return F.linear(x, w_quantized, self.base.bias)

def collect_baq_regularization_params(model: nn.Module) -> tuple[List[nn.Parameter], List[nn.Parameter]]:
    d_logits, b_logits = [], []
    for m in model.modules():
        if isinstance(m, BAQModule):
            d_logits.append(m.d_logit)
            b_logits.append(m.b_logit)
    return d_logits, b_logits

def apply_baq_quantization(model: nn.Module, bit_min: int, bit_max: int, granularity: str, initial_assignment: Optional[Dict[str, torch.Tensor]] = None) -> nn.Module:
    for name, m in list(model.named_modules()):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            path = name.split('.')
            parent_name = '.'.join(path[:-1])
            child_name = path[-1]
            parent = model.get_submodule(parent_name) if parent_name else model
            
            initial_bit_assignment = None
            if initial_assignment and name in initial_assignment:
                initial_bit_assignment = initial_assignment[name]

            setattr(parent, child_name, BAQModule(m, bit_min, bit_max, granularity, initial_bit=initial_bit_assignment))
    return model

def collect_baq_parameters(model: nn.Module) -> List[nn.Parameter]:
    params = []
    for m in model.modules():
        if isinstance(m, BAQModule):
            params.append(m.d_logit)
            params.append(m.b_logit)
    return params


def _layer_bits(module: nn.Module):
    if isinstance(module, BAQModule):
        return module.hard_bits().detach().cpu()
    elif isinstance(module, QuantizedModule):
        if hasattr(module, "bits_per_out"):
            return module.bits_per_out.detach().cpu()
        elif hasattr(module, "bits_per_param"):
            return module.bits_per_param.detach().cpu()
        elif hasattr(module, "bits_scalar"):
            return module.bits_scalar.detach().cpu()
    return None

def bitwidth_percentages(model: nn.Module, weighting: str = "params"):
    assert weighting in ("channels", "params")
    counts, total = {}, 0.0
    for m in model.modules():
        if isinstance(m, (QuantizedModule, BAQModule)):
            w, bits_vec = m.base.weight, _layer_bits(m)
            if bits_vec is None: continue
            if m.granularity == "per_channel":
                per_out = w[0].numel() if isinstance(m.base, nn.Conv2d) else w.size(1)
                unique, c = torch.unique(bits_vec, return_counts=True)
                for u, cnt in zip(unique, c):
                    b, wgt = int(u.item()), float(cnt.item()) if weighting == "channels" else float(cnt.item() * per_out)
                    counts[b], total = counts.get(b, 0.0) + wgt, total + wgt
            else:
                b, wgt = int(bits_vec.view(-1)[0].item()), 1.0 if weighting == "channels" else float(w.numel())
                counts[b], total = counts.get(b, 0.0) + wgt, total + wgt
    return {int(b): 100.0 * v / total for b, v in sorted(counts.items())} if total > 0 else {}

def collect_bit_distribution(model: nn.Module, bops_map: Dict[str, float]):
    layer_rows, total_params, total_bits, total_params_fp32_bits = [], 0, 0, 0
    total_bops, total_effective_bops = 0.0, 0.0
    processed_modules = set()

    for name, m in model.named_modules():
        if id(m) in processed_modules: continue
        
        is_quantized = isinstance(m, (QuantizedModule, BAQModule))
        base_module = m.base if is_quantized else m
        
        if isinstance(base_module, (nn.Conv2d, nn.Linear)):
            w = base_module.weight
            params = w.numel()
            
            bops = bops_map.get(name, 0.0)
            total_bops += bops

            if is_quantized:
                bits_vec = _layer_bits(m)
                if bits_vec is None: continue
                bits_vec = bits_vec.view(-1)
                unique, counts = torch.unique(bits_vec, return_counts=True)
                hist, avg_bits = {int(u.item()): int(c.item()) for u, c in zip(unique, counts)}, float(bits_vec.float().mean().item())
                processed_modules.add(id(base_module))
            else:
                hist, avg_bits = {32: 1}, 32.0

            bits_total = int(params * avg_bits)
            effective_bops = bops * (avg_bits / 32.0)
            total_effective_bops += effective_bops

            layer_rows.append({
                "layer": name, "type": base_module.__class__.__name__,
                "granularity": m.granularity if is_quantized else "fp32", "weight_shape": tuple(w.shape),
                "out_channels": int(w.size(0)), "avg_bits": avg_bits, "hist": hist, "params": int(params),
                "bits_total": bits_total, "baseline_bits": int(params * 32),
                "reduction_pct": 100.0 * (1.0 - (bits_total / max(1, params * 32))),
            })
            total_params += params
            total_bits += bits_total
            total_params_fp32_bits += params * 32

    totals = {
        "total_params": int(total_params), "total_bits": int(total_bits),
        "baseline_bits": int(total_params_fp32_bits),
        "model_size_mb": float(total_bits / 8 / 1e6),
        "baseline_size_mb": float(total_params_fp32_bits / 8 / 1e6),
        "reduction_pct": 100.0 * (1.0 - (total_bits / max(1, total_params_fp32_bits))),
        "total_gops": total_bops / 1e9,
        "total_effective_gops": total_effective_bops / 1e9,
    }
    return layer_rows, totals

def calculate_bops(model: nn.Module, input_size=(1, 3, 224, 224)):
    bops_map = {}
    hooks = []

    def hook_counter(module, input, output):
        for name, m in model.named_modules():
            if m is module:
                module_name = name
                break
        else:
            return

        base_module = module.base if hasattr(module, 'base') else module
        
        bops = 0
        if isinstance(base_module, nn.Conv2d):
            macs = output.shape[2] * output.shape[3] * base_module.in_channels * base_module.out_channels * base_module.kernel_size[0] * base_module.kernel_size[1]
            bops = 2 * macs / base_module.groups
        elif isinstance(base_module, nn.Linear):
            bops = 2 * base_module.in_features * base_module.out_features
        
        bops_map[module_name] = bops

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, QuantizedModule, BAQModule)):
            hooks.append(module.register_forward_hook(hook_counter))
            
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_size).to(device)
    model.eval()
    with torch.no_grad():
        model(dummy_input)

    for h in hooks:
        h.remove()
        
    return bops_map

