"""
Plot val_acc and topo_rho learning curves across seeds per architecture.
Architectures are auto-discovered from results_dir at runtime.

Usage:
    python scripts/plot_learning_curves.py
    python scripts/plot_learning_curves.py --results_dir results --output results/learning_curves.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


_PALETTE = [
    "#2196F3", "#4CAF50", "#FF5722", "#FF9800", "#9C27B0",
    "#00BCD4", "#E91E63", "#8BC34A", "#795548", "#607D8B",
]


def arch_label(arch: str) -> str:
    _KNOWN = {"lstm": "LSTM", "gru": "GRU", "mlp": "MLP"}
    if arch in _KNOWN:
        return _KNOWN[arch]
    _SUFFIXES = {"gs": "GS", "reinforce": "REINFORCE", "rnn": "RNN", "cnn": "CNN"}
    parts = arch.split("_")
    if len(parts) > 1 and parts[-1] in _SUFFIXES:
        base = " ".join(p.capitalize() for p in parts[:-1])
        return f"{base} ({_SUFFIXES[parts[-1]]})"
    return " ".join(p.capitalize() for p in parts)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results")
    p.add_argument("--output", default=None)
    return p.parse_args()


def discover_archs(results_dir: Path) -> list[str]:
    archs = []
    for arch_dir in sorted(results_dir.iterdir()):
        if not arch_dir.is_dir():
            continue
        for seed_dir in arch_dir.iterdir():
            if seed_dir.is_dir() and seed_dir.name.startswith("seed_") \
                    and (seed_dir / "metrics.json").exists():
                archs.append(arch_dir.name)
                break
    return archs


def load_arch(arch_path: Path) -> list:
    runs = []
    for seed_dir in sorted(arch_path.iterdir()):
        if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
            continue
        mp = seed_dir / "metrics.json"
        if mp.exists():
            with open(mp) as f:
                runs.append(json.load(f))
    return runs


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_path = Path(args.output) if args.output else results_dir / "learning_curves.png"

    archs = discover_archs(results_dir)
    if not archs:
        print("No metrics.json files found. Run training first.")
        return

    colors = {arch: _PALETTE[i % len(_PALETTE)] for i, arch in enumerate(archs)}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Emergent Language — Learning Curves by Architecture", fontsize=13)

    for ax, metric, ylabel in zip(
        axes,
        ["val_acc", "topo_rho"],
        ["Validation Accuracy", "Topographic Similarity (ρ)"],
    ):
        for arch in archs:
            runs = load_arch(results_dir / arch)
            if not runs:
                continue
            epochs = sorted(set(r["epoch"] for r in runs[0]))
            vals = []
            for run in runs:
                by_epoch = {r["epoch"]: r.get(metric) for r in run}
                vals.append([by_epoch.get(e) for e in epochs])
            arr = np.array(
                [[v if v is not None else float("nan") for v in row] for row in vals],
                dtype=float,
            )
            mean = np.nanmean(arr, axis=0)
            std = np.nanstd(arr, axis=0)
            color = colors[arch]
            ax.plot(epochs, mean, color=color, linewidth=2,
                    label=f"{arch_label(arch)} (n={len(runs)})", marker="o", markersize=3)
            ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.15)

        if metric == "val_acc":
            ax.axhline(0.2, color="gray", linestyle=":", linewidth=1)
            ax.set_ylim(bottom=0)

        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
