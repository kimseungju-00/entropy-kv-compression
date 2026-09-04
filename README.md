# Entropy-based KV Cache Compression

Single-pass KV cache compression for long-context LLMs. Tokens are selected by
their window-averaged attention (SnapKV-style), and the per-layer keep ratio is
set adaptively from the layer's attention entropy, so the budget is chosen per
input without manual tuning.

The method is grounded in a simple identity: the compression loss, measured as
the total variation distance between the original and compressed attention
distributions, equals the dropped attention mass `delta`. Entropy predicts
`delta`, which motivates the entropy-based budget.

## Key results

- **Theory.** `TV(a, a_tilde) = delta` holds exactly (|TV - delta| ~ 1e-7 across
  three models); `delta` correlates with attention entropy.
- **Memory.** At matched accuracy, the final method uses less KV cache memory
  than SnapKV: about 20% (Mistral-7B), 41% (LLaMA-3.1-8B), and up to ~48%
  (Qwen3-8B, monotonic region).
- **Latency.** Single-pass, so no decode-time overhead: latency matches SnapKV
  across context lengths.

Accuracy is comparable to SnapKV/PyramidKV; the contribution is the theoretical
grounding and the memory efficiency at matched accuracy, not a new accuracy SOTA.

## Method

`src/methods/proposed_kvpress.py` defines two presses on top of NVIDIA
[kvpress](https://github.com/NVIDIA/kvpress):

- `ProposedOnePassPress` — role-weighted budget (with head-role classification).
- `EntropyBudgetPress` — **final method**: mean-entropy budget (no head
  classification), which gives the best accuracy-memory trade-off.

## Setup

```bash
pip install -r requirements.txt
```

Download the [LongBench v1](https://github.com/THUDM/LongBench) `.jsonl` files
and either place them under `data/longbench/data/` or set:

```bash
export LONGBENCH_DATA=/path/to/longbench/data
```

Models used: `mistralai/Mistral-7B-Instruct-v0.3`,
`meta-llama/Llama-3.1-8B-Instruct`, `Qwen/Qwen3-8B`.

## Reproduce

Run from the project root. Results are written under `results/`.

```bash
# Theory: TV=delta and delta-entropy correlation, layer entropy heterogeneity
python experiments/theory/verify_delta.py --model <model> --n 15
python experiments/theory/measure_layer_entropy.py --model <model> --n 20

# Accuracy: baselines, then the final method (merged into the same JSON)
python experiments/performance/run_performance.py --model <model> --n 100
python experiments/performance/run_performance_m1.py --model <model> --n 100

# Memory at matched accuracy (no GPU; reads the accuracy JSON)
python experiments/memory/measure_memory_m1_final.py
python experiments/memory/compare_full_m1.py

# Latency / memory by context length
python experiments/efficiency/measure_efficiency_m1.py --model <model>

# Ablation and supplementary analysis
python experiments/ablation/run_ablation.py --model <model> --n 100
python experiments/analysis/diagnose_qwen.py --model <model> --n 15

# Figures and tables
python figures_tables/make_figures.py
python figures_tables/make_budget_curve.py
python figures_tables/make_tables.py
```

## Layout

```
src/methods/proposed_kvpress.py   the two presses (final = EntropyBudgetPress)
src/eval/                         LongBench loader and official metrics
experiments/theory/               TV=delta, delta-entropy, layer entropy
experiments/performance/          accuracy (baselines + final method)
experiments/memory/               matched-accuracy KV memory
experiments/efficiency/           latency and memory by context length
experiments/ablation/             component analysis
experiments/analysis/             head-entropy analysis (Qwen3)
figures_tables/                   figure and table generation
```

## Notes

- Qwen3-8B has a narrow accuracy range and a non-monotonic accuracy-budget
  curve; matched-score comparisons use the monotonic region and are noted where
  relevant.
- Evaluation metrics and prompts are adapted from LongBench v1.
