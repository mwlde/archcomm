# archcomm
 
An empirical study of how agent architecture affects the structural properties of emergent communication in multi-agent referential games.
 
Two agents — a sender and a receiver — must develop a shared communication system from scratch to win a signaling game. No human language, no prior symbols. We vary only the agent architecture (LSTM, GRU, Transformer, MLP) and measure what kind of proto-language emerges in each case.
 
Built on top of [EGG](https://github.com/facebookresearch/EGG) (Facebook Research).
 
---
 
## Research question
 
Does the inductive bias of an agent's architecture shape the compositional structure of the language it develops?
 
---
 
## Setup
 
### Local (Windows)
 
```bash
git clone https://github.com/SharkFishie/archcomm.git
cd archcomm/emergent-lang-arch
 
python -m venv .venv
.venv\Scripts\activate
 
pip install python-Levenshtein torch numpy scipy pandas matplotlib wandb pyyaml scikit-learn
pip install git+https://github.com/facebookresearch/EGG.git --no-deps
```
 
EGG requires two manual patches on Windows — see [DEVLOG.md](DEVLOG.md) for details.
 
### Google Colab (recommended for full experiments)
 
Use the notebook cells in order. See `notebooks/colab_setup.ipynb` or copy the cells from DEVLOG.md. Requires Runtime → T4 GPU.
 
---
 
## Running experiments
 
```bash
# set PYTHONPATH first (required)
export PYTHONPATH=.        # Mac/Linux/Colab
$env:PYTHONPATH = "."      # Windows PowerShell
 
# quick dev run (CPU, ~2 min)
python scripts/train.py --config configs/dev_config.yaml --arch lstm
 
# full run (GPU recommended)
python scripts/train.py --config configs/base_config.yaml --arch lstm --seed 42
python scripts/train.py --config configs/base_config.yaml --arch gru --seed 42
python scripts/train.py --config configs/base_config.yaml --arch transformer --seed 42
python scripts/train.py --config configs/base_config.yaml --arch mlp --seed 42

# Transformer with Gumbel-Softmax (separate condition, more stable than REINFORCE)
python scripts/train.py --config configs/transformer_gs_config.yaml --gumbel --seed 42
```

Results for the standard architectures save to `results/{arch}/seed_{seed}/`. The Gumbel-Softmax condition saves to `results/transformer_gs/seed_{seed}/`.
 
---
 
## Architectures compared
 
| key | description |
|---|---|
| `lstm` | LSTM sender + LSTM receiver — baseline, most prior work uses this |
| `gru` | GRU sender + GRU receiver — lighter recurrent baseline |
| `transformer` | Transformer encoder sender + receiver, trained with REINFORCE |
| `transformer_gs` | Same Transformer architecture, trained with Gumbel-Softmax (`--gumbel` flag) — resolves REINFORCE instability |
| `mlp` | MLP sender + receiver — control, no sequential processing |
 
---
 
## Metrics
 
| metric | description |
|---|---|
| accuracy | receiver top-1 accuracy on referential game |
| topo ρ | Spearman correlation between meaning distances and message distances — compositionality proxy |
| symbol entropy | Shannon entropy over message symbol distribution |
| effective vocab | symbols used with frequency > 0.1% |
 
---
 
## Repository structure
 
```
emergent-lang-arch/
├── agents/             agent architecture implementations
├── games/              referential game setup and loss function
├── analysis/           topographic similarity and metrics
├── configs/            base_config.yaml, dev_config.yaml
├── scripts/            train.py, evaluate.py
├── results/            experiment outputs (gitignored)
├── DEVLOG.md           error log and fixes
└── README.md
```
 
---
 
## Status

All experiments complete. 5 conditions × 10 seeds each (LSTM, GRU, Transformer/REINFORCE, Transformer/GS, MLP). Paper writeup in progress.
 
---
 
## Citation
 
Citation will be added once the paper is published.
 
---
 
## Author

Maria B.
University of Wollongong in Dubai
