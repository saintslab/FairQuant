# plot_radar.py
"""
Parses `results/<run_name>/` directories produced by train.py and plots a radar
chart comparing accuracy, fairness, and quantization efficiency across runs.

Usage:
    python plot_radar.py
    python plot_radar.py --dataset fitzpatrick17k
    python plot_radar.py --exclude smoketest --out-dir results
"""
import argparse
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

# Fixed-order categorical palette (dataviz skill default), light-mode chart chrome.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"

AXES = ["Accuracy", "Fairness", "BOPs Efficiency", "Bit-width Efficiency"]

OVERALL_RE = re.compile(r"OVERALL RESULTS: val_loss=([\d.]+)")
ACC_RE = re.compile(r"Accuracy:\s+avg=([\d.]+) \| worst=([\d.]+) \| gap=([\d.]+)")
EOPP1_RE = re.compile(r"EOpp1 \(TPR Gap\): ([\d.]+)")
EOPP0_RE = re.compile(r"EOpp0 \(TNR Gap\): ([\d.]+)")
EODD_RE = re.compile(r"EOdd\s+\(TPR\+FPR\): ([\d.]+)")
SIZE_RE = re.compile(r"TOTAL size ~ ([\d.]+) MB \(baseline ([\d.]+) MB\) reduction ([\d.]+)%")
GOPS_RE = re.compile(r"TOTAL GOPs: ([\d.]+) \| Effective GOPs: ([\d.]+) \| Computation Reduction: ([\d.]+)%")


def _opt_float(pattern, text):
    m = pattern.search(text)
    return float(m.group(1)) if m else None


def parse_training_log(path):
    """Extracts the last logged evaluation block (FINAL RESULTS if present, else the
    most recent epoch/iteration snapshot) from a training.log file."""
    with open(path) as f:
        text = f.read()

    matches = list(OVERALL_RE.finditer(text))
    if not matches:
        return None

    block_start = matches[-1].start()
    tail = text[block_start:]
    next_sep = re.search(r"\n-{10,}", tail)
    block = tail[: next_sep.start()] if next_sep else tail[:2000]

    acc_m = ACC_RE.search(block)
    if not acc_m:
        return None
    avg_acc, worst_acc, acc_gap = (float(x) for x in acc_m.groups())

    preceding = text[max(0, block_start - 300) : block_start]
    complete = "FINAL RESULTS" in preceding

    return {
        "avg_acc": avg_acc,
        "worst_acc": worst_acc,
        "acc_gap": acc_gap,
        "eopp1": _opt_float(EOPP1_RE, block),
        "eopp0": _opt_float(EOPP0_RE, block),
        "eodd": _opt_float(EODD_RE, block),
        "complete": complete,
    }


def parse_size_report(path):
    with open(path) as f:
        text = f.read()
    out = {}
    size_m = SIZE_RE.search(text)
    if size_m:
        model_mb, baseline_mb, size_reduction_pct = (float(x) for x in size_m.groups())
        out["model_size_mb"] = model_mb
        out["baseline_size_mb"] = baseline_mb
        out["size_reduction_pct"] = size_reduction_pct
        out["avg_bits"] = 32.0 * (1.0 - size_reduction_pct / 100.0)
    gops_m = GOPS_RE.search(text)
    if gops_m:
        total_gops, effective_gops, gops_reduction_pct = (float(x) for x in gops_m.groups())
        out["total_gops"] = total_gops
        out["effective_gops"] = effective_gops
        out["gops_reduction_pct"] = gops_reduction_pct
    return out


_KNOWN_DATASETS = ["fitzpatrick17k", "isic2019", "celeba", "fairface"]


def infer_dataset(name, run_args):
    if run_args.get("dataset"):
        return run_args["dataset"]
    lower = name.lower()
    for ds in _KNOWN_DATASETS:
        if ds in lower:
            return ds
    return "unknown"


def parse_report_args(path):
    args = {}
    with open(path) as f:
        for line in f:
            if ":" not in line or line.startswith("="):
                continue
            k, v = line.split(":", 1)
            args[k.strip()] = v.strip()
    return args


def collect_runs(results_dir, include=None, exclude=None):
    runs = []
    for name in sorted(os.listdir(results_dir)):
        run_dir = os.path.join(results_dir, name)
        log_path = os.path.join(run_dir, "training.log")
        if not os.path.isfile(log_path):
            continue
        if include and include.lower() not in name.lower():
            continue
        if exclude and exclude.lower() in name.lower():
            continue

        log_metrics = parse_training_log(log_path)
        if log_metrics is None:
            print(f"[skip] {name}: no evaluation results found in training.log")
            continue

        size_path = os.path.join(run_dir, "size_report.txt")
        size_metrics = parse_size_report(size_path) if os.path.isfile(size_path) else {}

        report_path = os.path.join(run_dir, "fairquant_report.txt")
        run_args = parse_report_args(report_path) if os.path.isfile(report_path) else {}

        run = {"name": name, "args": run_args, "dataset": infer_dataset(name, run_args), **log_metrics, **size_metrics}
        run["axis_values"] = compute_axis_values(run)
        runs.append(run)
    return runs


def compute_axis_values(run):
    acc = run.get("avg_acc")

    gaps = [g for g in (run.get("eopp1"), run.get("eopp0"), run.get("eodd")) if g is not None]
    fairness = max(0.0, 1.0 - sum(gaps) / len(gaps)) if gaps else None

    bops_eff = run.get("gops_reduction_pct")
    bops_eff = max(0.0, bops_eff / 100.0) if bops_eff is not None else None

    bits_eff = None
    if "avg_bits" in run:
        bits_eff = max(0.0, 1.0 - run["avg_bits"] / 32.0)

    return [acc, fairness, bops_eff, bits_eff]


MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def plot_radar(runs, dataset_label, out_path):
    n = len(AXES)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    plottable = []
    for r in runs:
        if any(v is None for v in r["axis_values"]):
            missing = [axis for axis, v in zip(AXES, r["axis_values"]) if v is None]
            print(f"[warn] {r['name']}: missing {missing}, skipping from radar")
        else:
            plottable.append(r)

    if not plottable:
        print(f"[skip] {dataset_label}: no runs had complete metrics for all axes")
        return False

    if len(plottable) > len(CATEGORICAL):
        print(
            f"[warn] {dataset_label}: {len(plottable)} runs exceeds the {len(CATEGORICAL)}-color validated palette; "
            f"colors repeat with a different marker shape per cycle. Narrow with --include/--exclude for a cleaner chart."
        )

    fig = plt.figure(figsize=(9.5, 7))
    fig.patch.set_facecolor(SURFACE)
    ax = fig.add_axes([0.2, 0.1, 0.45, 0.78], polar=True)
    ax.set_facecolor(SURFACE)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(AXES, color=INK_PRIMARY, fontsize=11)
    ax.tick_params(axis="x", pad=18)
    ax.set_rlabel_position(0)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color=INK_MUTED, fontsize=8)
    ax.grid(color=GRID, linewidth=0.8)
    ax.spines["polar"].set_color(GRID)

    for i, run in enumerate(plottable):
        color = CATEGORICAL[i % len(CATEGORICAL)]
        marker = MARKERS[i // len(CATEGORICAL) % len(MARKERS)]
        vals = run["axis_values"] + run["axis_values"][:1]
        label = run["name"] + ("" if run["complete"] else " (in progress)")
        linestyle = "-" if run["complete"] else "--"

        ax.plot(angles, vals, color=color, linewidth=2, linestyle=linestyle, marker=marker, markersize=5, label=label)
        ax.fill(angles, vals, color=color, alpha=0.08)

    ax.legend(loc="center left", bbox_to_anchor=(1.25, 0.5), frameon=False, labelcolor=INK_SECONDARY, fontsize=9)
    fig.suptitle(f"FairQuant trade-offs — {dataset_label}", color=INK_PRIMARY, fontsize=13, x=0.35)
    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"Saved: {out_path}")
    return True


def print_summary(runs):
    header = f"{'run':<45s} {'dataset':<14s} {'avg_acc':>8s} {'fair':>6s} {'bops_eff':>9s} {'bits_eff':>9s} {'avg_bits':>9s}"
    print(header)
    print("-" * len(header))
    for run in runs:
        acc, fairness, bops_eff, bits_eff = run["axis_values"]
        dataset = run["dataset"]
        status = "" if run["complete"] else "*"
        fmt = lambda v: f"{v:.3f}" if v is not None else "n/a"
        avg_bits = f"{run['avg_bits']:.2f}" if "avg_bits" in run else "n/a"
        print(f"{run['name']+status:<45s} {dataset:<14s} {fmt(acc):>8s} {fmt(fairness):>6s} {fmt(bops_eff):>9s} {fmt(bits_eff):>9s} {avg_bits:>9s}")
    if any(not r["complete"] for r in runs):
        print("\n* = run still in progress; using the most recent logged epoch, not FINAL RESULTS.")


def main():
    parser = argparse.ArgumentParser(description="Plot accuracy/fairness/efficiency radar charts from FairQuant results.")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--dataset", type=str, default=None, help="Only plot runs for this dataset (default: one chart per dataset found).")
    parser.add_argument("--include", type=str, default=None, help="Only include run dirs whose name contains this substring.")
    parser.add_argument("--exclude", type=str, default=None, help="Exclude run dirs whose name contains this substring.")
    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        print(f"Results dir not found: {args.results_dir}", file=sys.stderr)
        sys.exit(1)

    runs = collect_runs(args.results_dir, include=args.include, exclude=args.exclude)
    if not runs:
        print("No runs with parseable results found.", file=sys.stderr)
        sys.exit(1)

    print_summary(runs)
    print()

    os.makedirs(args.out_dir, exist_ok=True)

    datasets = [args.dataset] if args.dataset else sorted({r["dataset"] for r in runs})
    for dataset in datasets:
        dataset_runs = [r for r in runs if r["dataset"] == dataset]
        if not dataset_runs:
            print(f"[skip] no runs found for dataset '{dataset}'")
            continue
        out_path = os.path.join(args.out_dir, f"radar_{dataset}.png")
        plot_radar(dataset_runs, dataset, out_path)


if __name__ == "__main__":
    main()
