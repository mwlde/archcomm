"""
Aggregate metrics.json files across architectures and seeds.
Architectures and metrics are auto-discovered from results_dir at runtime.

Usage:
    python scripts/aggregate_results.py
    python scripts/aggregate_results.py --results_dir results --output results/summary.csv
"""

import argparse
import json
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results")
    p.add_argument("--output", default=None, help="CSV output path (default: results/summary.csv)")
    return p.parse_args()


def load_final_metrics(metrics_path: Path) -> dict | None:
    try:
        with open(metrics_path) as f:
            log = json.load(f)
        if not log:
            return None
        return log[-1]  # last eval checkpoint = final epoch
    except (json.JSONDecodeError, FileNotFoundError):
        return None


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_path = Path(args.output) if args.output else results_dir / "summary.csv"

    per_arch: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    found_any = False
    for arch_dir in sorted(results_dir.iterdir()):
        if not arch_dir.is_dir():
            continue
        arch = arch_dir.name
        for seed_dir in sorted(arch_dir.iterdir()):
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                continue
            metrics_path = seed_dir / "metrics.json"
            row = load_final_metrics(metrics_path)
            if row is None:
                print(f"  skip {seed_dir} — no valid metrics.json")
                continue
            found_any = True
            print(f"  loaded {seed_dir.relative_to(results_dir)} | "
                  f"val_acc={row.get('val_acc', 'N/A'):.3f}  "
                  f"topo_rho={row.get('topo_rho') or float('nan'):.3f}")
            for key, val in row.items():
                if key == "epoch":
                    continue
                if isinstance(val, (int, float)) and val is not None:
                    per_arch[arch][key].append(float(val))

    if not found_any:
        print("No metrics.json files found. Run training first.")
        return

    archs = sorted(per_arch.keys())
    metrics = sorted({key for arch_data in per_arch.values() for key in arch_data})

    # Build summary rows
    summary = []
    for arch in archs:
        row = {"arch": arch}
        for metric in metrics:
            vals = per_arch[arch][metric]
            if vals:
                row[f"{metric}_mean"] = round(float(np.mean(vals)), 4)
                row[f"{metric}_std"] = round(float(np.std(vals)), 4)
                row[f"{metric}_n"] = len(vals)
            else:
                row[f"{metric}_mean"] = None
                row[f"{metric}_std"] = None
                row[f"{metric}_n"] = 0
        summary.append(row)

    # Print table
    col_order = ["arch"] + [f"{m}_{s}" for m in metrics for s in ("mean", "std")]
    header = f"{'arch':<20}" + "".join(f"{c:<22}" for c in col_order[1:])
    print("\n" + header)
    print("-" * len(header))
    for row in summary:
        line = f"{row['arch']:<20}"
        for col in col_order[1:]:
            val = row.get(col)
            line += f"{'N/A':<22}" if val is None else f"{val:<22.4f}"
        print(line)

    # Save CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = col_order + [f"{m}_n" for m in metrics]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
