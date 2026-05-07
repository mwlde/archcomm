"""
Evaluation script: loads a trained checkpoint and computes all metrics.

Usage:
    python scripts/evaluate.py --checkpoint results/lstm/baseline/best_model.pt \\
                               --config configs/base_config.yaml --arch lstm
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from agents import get_agents
from games.referential_game import ReferentialDataset, build_game
from analysis import compute_topo_similarity, collect_messages, compute_all_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default="configs/base_config.yaml")
    p.add_argument("--arch", default=None)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output", default=None, help="JSON output path (default: next to checkpoint)")
    return p.parse_args()


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    flat = {}
    for section in cfg.values():
        if isinstance(section, dict):
            flat.update(section)
    return flat


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.arch:
        cfg["arch"] = args.arch
    if args.seed:
        cfg["seed"] = args.seed

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    arch = cfg["arch"]

    # ------------------------------------------------------------------ data
    split_seed_offset = {"train": 0, "val": 1, "test": 2}
    n_samples_key = {"train": "n_train", "val": "n_val", "test": "n_test"}
    dataset = ReferentialDataset(
        cfg["n_objects"], cfg["n_features"], cfg["n_distractors"],
        cfg[n_samples_key[args.split]],
        seed=cfg["seed"] + split_seed_offset[args.split],
    )
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=False)

    # ------------------------------------------------------------------ model
    sender, receiver = get_agents(arch, cfg)
    game = build_game(sender, receiver, cfg)
    game.load_state_dict(torch.load(args.checkpoint, map_location=device))
    game = game.to(device)
    game.eval()

    # ---------------------------------------------------------------- metrics
    total_acc = 0.0
    with torch.no_grad():
        for sender_input, labels, receiver_input in loader:
            sender_input = sender_input.to(device)
            labels = labels.to(device)
            receiver_input = receiver_input.to(device)
            _, interaction = game(sender_input, labels, receiver_input)
            total_acc += interaction.aux["acc"].mean().item()
    acc = total_acc / len(loader)

    meanings, messages = collect_messages(sender, loader, device)
    topo = compute_topo_similarity(meanings, messages)
    lang_stats = compute_all_metrics(messages, cfg["vocab_size"])

    results = {
        "arch": arch,
        "split": args.split,
        "accuracy": acc,
        **topo,
        **lang_stats,
    }

    print(json.dumps(results, indent=2))

    out_path = args.output or str(Path(args.checkpoint).parent / f"eval_{args.split}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")

    # Save raw messages for downstream analysis
    msg_path = Path(args.checkpoint).parent / f"messages_{args.split}.npy"
    np.save(msg_path, messages)
    np.save(Path(args.checkpoint).parent / f"meanings_{args.split}.npy", meanings)


if __name__ == "__main__":
    main()
