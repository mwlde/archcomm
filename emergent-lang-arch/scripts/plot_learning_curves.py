"""
Plot val_acc and topo_rho learning curves across seeds per architecture.

Usage:
    python scripts/plot_learning_curves.py
    python scripts/plot_learning_curves.py --results_dir results --output results/learning_curves.png
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ARCHS = ["lstm", "gru", "transformer", "mlp"]
COLORS = {
    "lstm": "#2196F3",
    "gru": "#4CAF50",
    "transformer": "#FF5722",
    "mlp": "#9C27B0",
    "transformer_gs": "#FF9800",
}
DISPLAY_LABELS = {
    "lstm": "LSTM",
    "gru": "GRU",
    "transformer": "Transformer (REINFORCE)",
    "mlp": "MLP",
    "transformer_gs": "Transformer (GS)",
}
METRICS = ["val_acc", "topo_rho"]
LABELS = {"val_acc": "Validation Accuracy", "topo_rho": "Topographic Similarity (ρ)"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results")
    p.add_argument("--output", default=None)
    p.add_argument(
        "--transformer_gs_dir", default=None,
        help="Path to transformer_gs results root (seed_*/ folders live directly inside). "
             "e.g. results/transformer_gs",
    )
    return p.parse_args()


def _load_seed_dirs(parent: Path, key: str, runs: dict) -> None:
    """Load all seed_*/ subdirs from parent into runs[key]."""
    for seed_dir in sorted(parent.iterdir()):
        if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
            continue
        metrics_path = seed_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        with open(metrics_path) as f:
            log = json.load(f)
        if log:
            runs[key].append(log)


def load_runs(results_dir: Path, transformer_gs_dir: Path | None = None) -> dict:
    """Returns {arch: [list of epoch dicts per seed]}.

    For standard archs, walks results_dir/{arch}/{name}/seed_*/
    For transformer_gs (flat layout), walks transformer_gs_dir/seed_*/ directly.
    """
    runs = defaultdict(list)

    for arch_dir in sorted(results_dir.iterdir()):
        if not arch_dir.is_dir():
            continue
        arch = arch_dir.name
        for name_dir in sorted(arch_dir.iterdir()):
            if not name_dir.is_dir():
                continue
            _load_seed_dirs(name_dir, arch, runs)

    if transformer_gs_dir is not None:
        gs_path = Path(transformer_gs_dir)
        if gs_path.exists():
            _load_seed_dirs(gs_path, "transformer_gs", runs)
        else:
            print(f"Warning: --transformer_gs_dir '{gs_path}' not found, skipping.")

    return runs


def align_by_epoch(seed_logs: list[list[dict]], metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Align multiple per-seed logs by epoch, return (epochs, mean, std).
    Only epochs present in ALL seeds are included.
    """
    epoch_sets = [set(row["epoch"] for row in log) for log in seed_logs]
    common_epochs = sorted(set.intersection(*epoch_sets))

    values = []
    for log in seed_logs:
        by_epoch = {row["epoch"]: row.get(metric) for row in log}
        row_vals = [by_epoch[e] for e in common_epochs]
        # replace None (NaN topo_rho) with nan so it doesn't drag the mean
        row_vals = [v if v is not None else float("nan") for v in row_vals]
        values.append(row_vals)

    arr = np.array(values, dtype=float)          # (n_seeds, n_epochs)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    return np.array(common_epochs), mean, std


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_path = Path(args.output) if args.output else results_dir / "learning_curves.png"

    runs = load_runs(results_dir, args.transformer_gs_dir)
    if not runs:
        print("No metrics.json files found. Run training first.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Emergent Language — Learning Curves by Architecture", fontsize=13, y=1.01)

    all_archs = ARCHS + ["transformer_gs"]

    for ax, metric in zip(axes, METRICS):
        plotted = False
        for arch in all_archs:
            seed_logs = runs.get(arch)
            if not seed_logs:
                continue
            epochs, mean, std = align_by_epoch(seed_logs, metric)
            color = COLORS[arch]
            n = len(seed_logs)
            label = f"{DISPLAY_LABELS[arch]} (n={n})"
            ax.plot(epochs, mean, color=color, linewidth=2, label=label, marker="o", markersize=4)
            ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.15)
            plotted = True

        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(LABELS[metric], fontsize=11)
        ax.set_title(LABELS[metric], fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)

        if metric == "val_acc":
            ax.axhline(0.2, color="gray", linestyle=":", linewidth=1, label="chance (0.2)")
            ax.set_ylim(bottom=0)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
