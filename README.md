# archcomm

An empirical study of how agent architecture affects the structural properties
of emergent communication in multi-agent referential games.

Two agents, a sender and a receiver, must develop a shared communication
system from scratch to win a signalling game. 
No prior symbols are used. Only the agent architecture (LSTM, GRU, Transformer, MLP) is varied and we measure what kind of proto-language emerges in each case.

Built on top of [EGG](https://github.com/facebookresearch/EGG)
(Kharitonov et al., 2021).

---

## Research question

Does the inductive bias of an agent's architecture shape the compositional
structure of the language it develops?

---

## Setup

### Google Colab (recommended)

Open the notebook:
[![Colab link!](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1wzqLFNgWw8A4ekC2_EpGVUquErgb-zoK?usp=sharing)

Requires Runtime T4 GPU for speed. Run cells in order.

### Local (Windows)

```bash
git clone https://github.com/SharkFishie/archcomm.git
cd archcomm/emergent-lang-arch

python -m venv .venv
.venv\Scripts\activate

pip install python-Levenshtein torch numpy scipy pandas matplotlib wandb pyyaml scikit-learn
pip install git+https://github.com/facebookresearch/EGG.git --no-deps
```
---

## Running experiments

```bash
#set PYTHONPATH first
export PYTHONPATH=.        # Mac/Linux/Colab
$env:PYTHONPATH = "."      # Windows PowerShell

#quick dev run (CPU, around 2 min)
python scripts/train.py --config configs/dev_config.yaml --arch lstm

#full run (GPU recommended)
python scripts/train.py --config configs/base_config.yaml --arch lstm --seed 42
python scripts/train.py --config configs/base_config.yaml --arch gru --seed 42
python scripts/train.py --config configs/base_config.yaml --arch transformer --seed 42
python scripts/train.py --config configs/base_config.yaml --arch mlp --seed 42

#transformer with GS
python scripts/train.py --config configs/transformer_gs_config.yaml --gumbel --seed 42
```

Results save to `results/{arch}/seed_{seed}/`.
Transformer GS saves to `results/transformer_gs/seed_{seed}/`.

---

## Architectures compared

| key | description |
|---|---|
| `lstm` | LSTM sender + receiver — baseline, most prior work uses this |
| `gru` | GRU sender + receiver — lighter recurrent baseline |
| `transformer` | Transformer encoder, trained with REINFORCE |
| `transformer_gs` | Same Transformer, trained with Gumbel-Softmax (`--gumbel`) |
| `mlp` | MLP sender + receiver — control, no sequential processing |

---

## Metrics

| metric | description |
|---|---|
| accuracy | receiver top-1 accuracy on referential game |
| topo ρ | Spearman correlation between meaning and message distances — compositionality proxy |
| symbol entropy | Shannon entropy over message symbol distribution |
| effective vocab | symbols used with frequency > 0.1% |

---

## Repository structure

```
emergent-lang-arch/
├── agents/         agent architecture implementations
├── games/          referential game setup and loss function
├── analysis/       topographic similarity and metrics
├── configs/        base_config.yaml, dev_config.yaml, transformer_gs_config.yaml
├── scripts/        train.py, evaluate.py, aggregate_results.py, plot_learning_curves.py
├── results/        experiment outputs (gitignored)
├── DEVLOG.md       setup log and fixes
README.md
```

---

## Status

All experiments complete. 5 conditions × 10 seeds each (LSTM, GRU,
Transformer/REINFORCE, Transformer/GS, MLP). 
Paper in progress.

---

## Citation

Paper citation will be added on publication. If you use EGG, please cite:

```bibtex
@misc{kharitonov:etal:2021,
  author    = {Kharitonov, Eugene and Dess{\`i}, Roberto and Chaabouni,
               Rahma and Bouchacourt, Diane and Baroni, Marco},
  title     = {{EGG}: a toolkit for research on {E}mergence of
               lan{G}uage in {G}ames},
  howpublished = {\url{https://github.com/facebookresearch/EGG}},
  year      = {2021}
}
```

---

## Authors

Maria B.
