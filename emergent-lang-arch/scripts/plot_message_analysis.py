"""
Qualitative message analysis: symbol overlap between similar vs. dissimilar object pairs.

Loads messages_epoch100.npy and meanings_epoch100.npy from results/{arch}/seed_42/
for all 5 architectures and computes positional symbol overlap as a compositionality proxy.

Usage:
    python scripts/plot_message_analysis.py
    python scripts/plot_message_analysis.py --results_dir results --seed 42 --epoch 100
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cosine


ARCHS = ["lstm", "gru", "transformer", "transformer_gs", "mlp"]
LABELS = {
    "lstm": "LSTM",
    "gru": "GRU",
    "transformer": "Transformer (REINFORCE)",
    "transformer_gs": "Transformer (GS)",
    "mlp": "MLP",
}

SIMILAR_THRESHOLD = 0.5
DISSIMILAR_THRESHOLD = 0.3
N_PAIRS = 10
MIN_PAIRS = 5
N_EXAMPLES = 3


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epoch", type=int, default=100)
    return p.parse_args()


def symbol_overlap(m1: np.ndarray, m2: np.ndarray) -> float:
    """Fraction of positions with identical symbol (up to min message length)."""
    length = min(len(m1), len(m2))
    if length == 0:
        return 0.0
    return float(np.sum(m1[:length] == m2[:length]) / length)


def find_pairs(meanings: np.ndarray, rng: np.random.Generator, n: int, threshold: float, below: bool):
    """
    Sample up to n pairs satisfying cosine distance < threshold (below=True)
    or > threshold (below=False). Returns list of (i, j) index tuples.
    """
    n_objects = len(meanings)
    candidates = []
    # random sample of index pairs to avoid O(N^2) full scan
    max_candidates = min(50_000, n_objects * (n_objects - 1) // 2)
    checked = set()
    attempts = 0
    while len(candidates) < n and attempts < max_candidates * 2:
        i, j = rng.integers(0, n_objects, size=2)
        if i == j or (i, j) in checked or (j, i) in checked:
            attempts += 1
            continue
        checked.add((i, j))
        dist = cosine(meanings[i], meanings[j])
        if (below and dist < threshold) or (not below and dist > threshold):
            candidates.append((i, j, dist))
        attempts += 1

    candidates.sort(key=lambda x: x[2])
    return [(i, j) for i, j, _ in candidates[:n]]


def analyse_arch(arch: str, results_dir: Path, seed: int, epoch: int):
    seed_dir = results_dir / arch / f"seed_{seed}"
    msg_path = seed_dir / f"messages_epoch{epoch}.npy"
    mean_path = seed_dir / f"meanings_epoch{epoch}.npy"

    if not msg_path.exists() or not mean_path.exists():
        return "missing"

    messages = np.load(msg_path)   # (N, max_len)
    meanings = np.load(mean_path)  # (N, feature_dim)

    rng = np.random.default_rng(0)

    similar_pairs = find_pairs(meanings, rng, N_PAIRS, SIMILAR_THRESHOLD, below=True)
    dissimilar_pairs = find_pairs(meanings, rng, N_PAIRS, DISSIMILAR_THRESHOLD, below=False)

    if len(similar_pairs) < MIN_PAIRS or len(dissimilar_pairs) < MIN_PAIRS:
        print(f"  Warning: not enough pairs found (similar={len(similar_pairs)}, "
              f"dissimilar={len(dissimilar_pairs)}) — skipping.")
        return "no_pairs"

    sim_overlaps = [symbol_overlap(messages[i], messages[j]) for i, j in similar_pairs]
    dis_overlaps = [symbol_overlap(messages[i], messages[j]) for i, j in dissimilar_pairs]

    mean_sim = float(np.mean(sim_overlaps))
    mean_dis = float(np.mean(dis_overlaps))
    ratio = mean_sim / mean_dis if mean_dis > 0 else float("nan")

    return {
        "arch": arch,
        "messages": messages,
        "meanings": meanings,
        "similar_pairs": similar_pairs,
        "dissimilar_pairs": dissimilar_pairs,
        "sim_overlaps": sim_overlaps,
        "dis_overlaps": dis_overlaps,
        "mean_overlap_similar_pairs": mean_sim,
        "mean_overlap_dissimilar_pairs": mean_dis,
        "ratio": ratio,
    }


def print_examples(result: dict, n: int = N_EXAMPLES):
    arch = result["arch"]
    messages = result["messages"]
    meanings = result["meanings"]

    print(f"\n  {'─'*56}")
    print(f"  {LABELS[arch]}")
    print(f"  {'─'*56}")

    for label, pairs in [("SIMILAR pairs (cosine < 0.5)", result["similar_pairs"][:n]),
                          ("DISSIMILAR pairs (cosine > 0.6)", result["dissimilar_pairs"][:n])]:
        print(f"\n  {label}:")
        for i, j in pairs:
            dist = cosine(meanings[i], meanings[j])
            m1 = messages[i].tolist()
            m2 = messages[j].tolist()
            ov = symbol_overlap(messages[i], messages[j])
            print(f"    dist={dist:.3f}  overlap={ov:.2f}")
            print(f"      obj {i:4d}: {m1}")
            print(f"      obj {j:4d}: {m2}")


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_csv = results_dir / "message_analysis.csv"

    rows = []
    for arch in ARCHS:
        print(f"\nAnalysing {LABELS[arch]}...")
        result = analyse_arch(arch, results_dir, args.seed, args.epoch)
        if result == "missing":
            print(f"  Skipping {arch} — files not found at "
                  f"{results_dir}/{arch}/seed_{args.seed}/messages_epoch{args.epoch}.npy")
            continue
        if result == "no_pairs":
            continue

        print(f"  mean overlap (similar pairs):    {result['mean_overlap_similar_pairs']:.3f}")
        print(f"  mean overlap (dissimilar pairs): {result['mean_overlap_dissimilar_pairs']:.3f}")
        print(f"  ratio (similar/dissimilar):      {result['ratio']:.3f}")

        print_examples(result)

        rows.append({
            "arch": arch,
            "mean_overlap_similar_pairs": f"{result['mean_overlap_similar_pairs']:.4f}",
            "mean_overlap_dissimilar_pairs": f"{result['mean_overlap_dissimilar_pairs']:.4f}",
            "ratio": f"{result['ratio']:.4f}",
        })

    if rows:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["arch", "mean_overlap_similar_pairs",
                                "mean_overlap_dissimilar_pairs", "ratio"]
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved summary to {output_csv}")
    else:
        print("\nNo results found — run training and evaluate first to generate .npy files.")


if __name__ == "__main__":
    main()
