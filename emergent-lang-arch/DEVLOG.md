# archcomm : development log

Chronological record of errors encountered during setup and experiment runs, with root causes and fixes applied.

Some of them were just me figuring myslef out along the way and documenting for the future, so I dont get lost in my own corrections, even if it was somethign relatively trivial.


Basically, these were just my notes, but I decided to keep them public. 

---

## 1. `editdistance` fails to build on Windows

**when:** initial `pip install egg-lib` / `pip install git+https://github.com/facebookresearch/EGG.git`

**error:**
```
ERROR: Failed building wheel for editdistance
error: Microsoft Visual C++ 14.0 or greater is required.
(i couldnt bring myself to install cpp properly on this device so just worked around it, but this is skippable if your environemnt is prepared propperly)
```

**root cause:** `editdistance` is a C extension that must be compiled from source. Windows requires Microsoft Visual C++ Build Tools to compile C/C++ extensions. These were not installed.

**fix:** created a fake `editdistance` package in site-packages that wraps `python-Levenshtein`, which ships pre-built wheels for Windows:

```python
import site, os
path = site.getsitepackages()[0]
os.makedirs(os.path.join(path, 'editdistance'), exist_ok=True)
with open(os.path.join(path, 'editdistance', '__init__.py'), 'w') as f:
    f.write('from Levenshtein import distance\ndef eval(a, b): return distance(a, b)\n')
```

**files changed:** none in repo — patch applied to local venv and Colab environment at runtime.

---

## 2. EGG not importable after install

**when:** after `pip install git+https://github.com/facebookresearch/EGG.git --no-deps`

**error:**
```
ModuleNotFoundError: No module named 'egg'

(again, skippable if you prepare everything well, but this was my first time working with egg so the initial setup took me a few min longer)
```

**root cause:** `--no-deps` skipped all dependencies, so EGG installed but its dependencies (wandb, pandas, scikit-learn, etc.) were missing. EGG's `__init__.py` imports them immediately on load.

**fix:** installed all dependencies manually:
```bash
pip install wandb pandas pytest rich scikit-learn submitit timm torchvision dataclasses
```

---

## 3. EGG imports `editdistance` directly in source

**when:** after installing all dependencies

**error:**
```
ModuleNotFoundError: No module named 'editdistance'
```

**root cause:** `egg/core/language_analysis.py` has a hardcoded `import editdistance` at line 10. Even though EGG itself installed, this import fails because `editdistance` has no pre-built wheel. The fake package fix from issue 1 resolves this on subsequent setups.

**fix:** same fake package fix as issue 1, applied before importing egg.core.

---

## 4. EGG LSTM hidden state TypeError — Windows/local

**when:** first attempt to run `scripts/train.py --arch lstm`

**error:**
```
TypeError: zeros_like(): argument 'input' (position 1) must be Tensor, not tuple
```
at `egg/core/reinforce_wrappers.py` line 305.

**root cause:** EGG was written for an older PyTorch version. In newer PyTorch, LSTM hidden state is returned as a tuple `(h, c)`. EGG's code passes `prev_hidden[0]` directly to `torch.zeros_like()`, which expects a Tensor but gets a tuple.

**fix:** patched `reinforce_wrappers.py`:
```python
# before
torch.zeros_like(prev_hidden[0]) for _ in range(self.num_layers)

# after
torch.zeros_like(prev_hidden[0] if isinstance(prev_hidden, (tuple, list)) else prev_hidden) for _ in range(self.num_layers)
```

---

## 5. EGG LSTM hidden state TypeError -- Colab (nested tuple)

**when:** running on Google Colab after applying fix from issue 4

**error:**
```
TypeError: zeros_like(): argument 'input' (position 1) must be Tensor, not tuple
```
same line, different cause.

**root cause:** on Colab's PyTorch version, `prev_hidden[0]` is itself a tuple `(h, c)` - one level deeper nesting than the Windows case.

**fix:** patched `reinforce_wrappers.py` more deeply — extracted `_h0` from `self.agent()` output before building `prev_hidden`:

```python
# before
prev_hidden = [self.agent(x, aux_input)]
prev_hidden.extend(
    [torch.zeros_like(prev_hidden[0]) for _ in range(self.num_layers - 1)]
)

# after
_agent_out = self.agent(x, aux_input)
_h0 = _agent_out[0] if isinstance(_agent_out, (tuple, list)) else _agent_out
prev_hidden = [_h0]
prev_hidden.extend(
    [torch.zeros_like(prev_hidden[0]) for _ in range(self.num_layers - 1)]
)
```

Applied automatically in Colab cell 2 via string replacement.

---

## 6. `ModuleNotFoundError: No module named 'agents'`

**when:** running `python scripts/train.py` from repo root or from `emergent-lang-arch/`

**error:**
```
ModuleNotFoundError: No module named 'agents'
```

**root cause:** Python doesn't automatically add the current working directory to `sys.path` when running a script in a subdirectory. The `agents/` package is in `emergent-lang-arch/` but Python can't find it.

**fix:** set PYTHONPATH before running:
```bash
# local
$env:PYTHONPATH = "."
python scripts/train.py ...

# Colab
!PYTHONPATH=. python scripts/train.py ...
```

---

## 7. Receiver embedding TypeError — float tensor as token indices

**when:** after PYTHONPATH fix, first successful game forward pass attempt

**error:**
```
RuntimeError: Expected tensor for argument #1 'indices' to have scalar types: Long, Int;
but got torch.FloatTensor instead
```

**root cause:** misunderstanding of how EGG's `RnnReceiverDeterministic` works. EGG runs its own internal `RnnEncoder` on the incoming message tokens, then calls the receiver with the encoded hidden state (a float vector), not the raw token IDs. All four receiver `forward()` methods were calling `self.embed(message)` on this float hidden state, garbage input to an embedding layer.

**fix:** rewrote all four receiver cores to accept the pre-encoded hidden state directly and score candidates from it, removing the embed/encode layers from the receiver entirely.

**files changed:** `agents/lstm_agent.py`, `agents/gru_agent.py`, `agents/transformer_agent.py`, `agents/mlp_agent.py`

**result:** accuracy jumped from ~20% (random chance) to ~45% by epoch 10. (yay, little victory)

---

## 8. Training frozen at epoch 1, batch 1 on Colab

**when:** first Colab run with full base config

**symptom:** script printed `Epoch 1 batch 1/195 running...` then hung indefinitely (13+ minutes).

**root cause:** `base_config.yaml` had `device: "cpu"` while Colab has a T4 GPU. PyTorch was running 50k samples × 100 epochs entirely on CPU, ouch.

**fix:**
```bash
sed -i 's/device: "cpu"/device: "cuda"/' configs/base_config.yaml
```

Also created `configs/dev_config.yaml` with reduced scale for local testing:
- `n_train: 2000`, `n_val: 200`, `n_test: 200`
- `epochs: 20`, `batch_size: 64`

---

## 9. Accuracy stuck at ~20%, topo_rho NaN for full run

**when:** first complete 100-epoch run on Colab (after device fix)

**symptom:** accuracy never exceeded random chance, topo_rho mostly NaN — agents sending identical messages regardless of input.

**root cause:** receiver fix from issue 7 had not yet been applied. Agents converged to a degenerate solution because receiver feedback was useless.

**fix:** applied receiver fix from issue 7, reran.

---

## 10. Results not saved to disk, only printed to terminal

**when:** after first successful full run

**symptom:** training completed with good accuracy and topo_rho numbers visible in terminal output, but no structured data saved only raw `.npy` message files and `best_model.pt`.

**root cause:** `train.py` logged metrics to console but never wrote them to disk. On session end, all terminal output was lost.

**fix:** updated `train.py` to save a `metrics.json` file per run containing epoch, train_acc, val_acc, topo_rho, symbol_entropy, and effective_vocab_size at each eval checkpoint. All 40 runs (4 architectures × 10 seeds) were rerun after this fix.

---

## 11. Results overwritten across seeds

**when:** second full sweep run

**symptom:** results directory showed only one folder per architecture (`baseline/`) regardless of seed — each seed was overwriting the previous.

**root cause:** results directory was hardcoded to `results/{arch}/baseline/` in `train.py`, ignoring the seed argument entirely.

**fix:** updated results path to `results/{arch}/seed_{seed}/` so each run saves independently.

---

## 12. Transformer training instability

**when:** analysis of full sweep results

**symptom:** 8 out of 10 Transformer seeds never learned — accuracy flat at ~0.20 (random chance), topo_rho NaN, effective vocabulary collapsed to ~4 symbols.

**root cause:** standard REINFORCE training settings (lr=1e-3, no warmup) are poorly suited to Transformer architectures, which require careful optimisation to train stably.

**status:** created `configs/transformer_config.yaml` with lower learning rate (1e-4), higher entropy coefficient (0.05), and smaller hidden dimension (128). Rerunning Transformer-only sweep under tuned config — also failed to help, and ran ~4x slower than the original pass. 

---

## 13. Transformer Gumbel-Softmax experiment

**when:** after REINFORCE transformer failed to converge reliably (issue 12)

**motivation:** REINFORCE training of the Transformer was unstable — 8/10 seeds never left chance accuracy. Gumbel-Softmax (GS) makes the communication channel differentiable, removing the high-variance policy gradient signal and allowing end-to-end gradient flow through the message. This should stabilise training significantly.

**what was changed:**
- added `--gumbel` flag to `scripts/train.py`, which swaps `SenderReceiverRnnReinforce` for `SenderReceiverRnnGS`
- added `agents/__init__.py::get_agents_gs()` which wraps the same sender/receiver cores in `RnnSenderGS` + `RnnReceiverGS`
- added `configs/transformer_gs_config.yaml`: lr=1e-4, warmup_epochs=10, hidden_dim=128, temperature=1.0
- GS results saved to `results/transformer_gs/seed_{seed}/` to avoid collision with REINFORCE results
- patched `egg/core/gs_wrappers.py` for the same LSTM tuple bug as issue 5

**results:** all 10 seeds converged.
- val_acc: 0.575 (mean across seeds)
- topo_rho: 0.149

**interpretation:** GS solved the training instability completely — stable convergence across all 10 seeds compared to 2/10 for REINFORCE. Accuracy (0.575) is lower than LSTM (≈0.630) and GRU (≈0.662), consistent with the Transformer not being well-suited to sequential message generation with a short vocabulary and small hidden dim. Topographic similarity (0.149) is positive but lower than recurrent architectures, suggesting less structured language despite stable training.

---

## current status

- all experiments complete: 5 conditions × 10 seeds (LSTM, GRU, Transformer/REINFORCE, Transformer/GS, MLP) 50 runs total
- metrics.json saved per run for all conditions
- analysis scripts complete: aggregate_results.py, plot_learning_curves.py, plot_message_length.py, plot_message_analysis.py
- paper writeup in progress
