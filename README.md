# The Alignment Tax Concentrates in Output Pathways

Anonymous code supplement for NeurIPS 2026 submission.

## Overview

This repository provides code to reproduce all key experiments and results from the paper. The core finding is that instruction tuning's degradation of structured generation (the "alignment tax") concentrates in **output-pathway components** (W_V, W_O, W_down), while query/key projections remain largely intact. This asymmetry enables two interventions:
- **SAR** (Surgical Alignment Reversal): training-free rollback of the top-k% most harmful components
- **OC-DPO** (Output-Constrained DPO): selective LoRA exclusion during preference optimization

## Requirements

**Python:** 3.10+

**Hardware:** Single GPU with 24GB+ VRAM (A100 40GB recommended). System RAM: 32GB+ for loading model weights during attribution. Disk: ~150GB for all 8 model checkpoints.

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Note on optional dependencies:** `wandb` (Weights & Biases) is listed in `requirements.txt` but is fully optional. All evaluation scripts detect its absence at import time and skip logging. By default, even when wandb is installed, `wandb.init()` is called with `mode="disabled"` (i.e., no network connection is made). To enable logging, set `WANDB_MODE=online` before running. No wandb account is required to reproduce any result.

### Model Downloads

All models used in this work are publicly available on HuggingFace and were not trained by the authors. Download the following model pairs to `./models/`:

| Model ID | Local Path | Notes |
|----------|-----------|-------|
| `Qwen/Qwen2.5-7B` | `./models/Qwen2.5-7B` | |
| `Qwen/Qwen2.5-7B-Instruct` | `./models/Qwen2.5-7B-Instruct` | |
| `meta-llama/Llama-3.1-8B` | `./models/Llama-3.1-8B` | Gated model (requires Meta license acceptance) |
| `meta-llama/Llama-3.1-8B-Instruct` | `./models/Llama-3.1-8B-Instruct` | Gated model |
| `mistralai/Mistral-7B-v0.3` | `./models/Mistral-7B-v0.3` | |
| `mistralai/Mistral-7B-Instruct-v0.3` | `./models/Mistral-7B-Instruct-v0.3` | |
| `01-ai/Yi-1.5-9B` | `./models/Yi-1.5-9B` | |
| `01-ai/Yi-1.5-9B-Chat` | `./models/Yi-1.5-9B-Chat` | |
| `Qwen/Qwen2.5-14B` | `./models/Qwen2.5-14B` | For 14B scaling (Appendix E) only |
| `Qwen/Qwen2.5-14B-Instruct` | `./models/Qwen2.5-14B-Instruct` | For 14B scaling (Appendix E) only |

**Download commands:**
```bash
# Install huggingface CLI if needed
pip install huggingface_hub[cli]

# For gated models (Llama), first accept the license at huggingface.co, then:
huggingface-cli login

# Download all models
for model in Qwen/Qwen2.5-7B Qwen/Qwen2.5-7B-Instruct \
             meta-llama/Llama-3.1-8B meta-llama/Llama-3.1-8B-Instruct \
             mistralai/Mistral-7B-v0.3 mistralai/Mistral-7B-Instruct-v0.3 \
             01-ai/Yi-1.5-9B 01-ai/Yi-1.5-9B-Chat; do
    local_name=$(echo $model | cut -d/ -f2)
    huggingface-cli download $model --local-dir ./models/$local_name
done
```

### External Datasets

Some evaluation scripts require external datasets:
- **BFCL**: Requires `gorilla-llm/Berkeley-Function-Calling-Leaderboard` dataset (loaded via HuggingFace `datasets` library; requires network access on first run)
- **IFEval**: Downloaded automatically (`google/IFEval`)
- **HumanEval**: Downloaded automatically (`openai/openai_humaneval`)
- **MMLU**: Downloaded automatically (`cais/mmlu`)
- **AdvBench** (for safety evaluation): Download manually:
  ```bash
  mkdir -p data
  wget -O data/harmful_behaviors.csv \
    https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv
  ```

## Reproducing Results

All commands run from the repository root. Each script prints progress and saves results to `./results/` as JSON.

**Dependency chain:** Run `01_attribution.py` first — subsequent scripts (SAR, benchmarks) load its output.

**Estimated runtimes** (per model, on A100 40GB):
| Script | Time |
|--------|------|
| 01_attribution | 2-4 hours |
| 03_gradient_analysis | 30-60 min |
| 04_sar | 1-2 hours |
| 05_ocdpo | 30-60 min |
| 06_random_baseline | 2-3 hours |
| 07_cross_dataset_transfer | 3-5 hours |
| 08_statistical_tests | <1 min (CPU only) |
| Evaluation scripts | 2-4 hours each |

**Total estimated GPU time:** ~50-80 A100 GPU-hours to reproduce all results across 4 model families.

**Note on script numbering:** Scripts are numbered 01, 03, 04... (step 02, a causal rollback variant, is omitted from this supplement as it is not needed to reproduce any paper table).

### 1. Attribution Analysis (Tables 1-2)

Weight-patching attribution: measures per-component alignment harm scores across 209 structured generation examples.

```bash
python experiments/01_attribution.py ./models/Qwen2.5-7B ./models/Qwen2.5-7B-Instruct cuda:0 qwen2.5-7b
python experiments/01_attribution.py ./models/Llama-3.1-8B ./models/Llama-3.1-8B-Instruct cuda:0 llama-3.1-8b
python experiments/01_attribution.py ./models/Mistral-7B-v0.3 ./models/Mistral-7B-Instruct-v0.3 cuda:0 mistral-7b
python experiments/01_attribution.py ./models/Yi-1.5-9B ./models/Yi-1.5-9B-Chat cuda:0 yi-1.5-9b
```

### 2. Gradient Analysis (Table 3)

DPO gradient analysis confirming the V/O concentration mechanism.

```bash
python experiments/03_gradient_analysis.py ./models/Qwen2.5-7B cuda:0 qwen2.5-7b
python experiments/03_gradient_analysis.py ./models/Llama-3.1-8B cuda:0 llama-3.1-8b
python experiments/03_gradient_analysis.py ./models/Mistral-7B-v0.3 cuda:0 mistral-7b
python experiments/03_gradient_analysis.py ./models/Yi-1.5-9B cuda:0 yi-1.5-9b
```

### 3. SAR — Surgical Alignment Reversal (Table 4)

Training-free intervention: rolls back top-k% most harmful components.

```bash
python experiments/04_sar.py ./models/Qwen2.5-7B ./models/Qwen2.5-7B-Instruct cuda:0 qwen2.5-7b
python experiments/04_sar.py ./models/Llama-3.1-8B ./models/Llama-3.1-8B-Instruct cuda:0 llama-3.1-8b
python experiments/04_sar.py ./models/Mistral-7B-v0.3 ./models/Mistral-7B-Instruct-v0.3 cuda:0 mistral-7b
python experiments/04_sar.py ./models/Yi-1.5-9B ./models/Yi-1.5-9B-Chat cuda:0 yi-1.5-9b
```

### 4. OC-DPO — Output-Constrained DPO (Table 5)

Selective LoRA exclusion during preference optimization.

The optional fourth argument sets the random seed (default: 42). Table 5 uses seed 42.
Appendix B (seed robustness) uses seeds 42, 123, 456, 789, 1337 — run each seed separately:

```bash
# Single run (seed 42, reproduces Table 5)
python experiments/05_ocdpo.py ./models/Qwen2.5-7B cuda:0 qwen2.5-7b
python experiments/05_ocdpo.py ./models/Llama-3.1-8B cuda:0 llama-3.1-8b
python experiments/05_ocdpo.py ./models/Mistral-7B-v0.3 cuda:0 mistral-7b

# 5-seed sweep (reproduces Appendix B seed-robustness table)
for seed in 42 123 456 789 1337; do
    python experiments/05_ocdpo.py ./models/Qwen2.5-7B cuda:0 qwen2.5-7b $seed
    python experiments/05_ocdpo.py ./models/Llama-3.1-8B cuda:0 llama-3.1-8b $seed
    python experiments/05_ocdpo.py ./models/Mistral-7B-v0.3 cuda:0 mistral-7b $seed
done
```

### 5. Random Baseline with Bootstrap CIs (Table 4 caption)

Multi-seed random rollback to establish SAR/random ratio with confidence intervals.

```bash
python experiments/06_random_baseline.py ./models/Qwen2.5-7B ./models/Qwen2.5-7B-Instruct cuda:0 qwen2.5-7b 20
python experiments/06_random_baseline.py ./models/Llama-3.1-8B ./models/Llama-3.1-8B-Instruct cuda:0 llama-3.1-8b 20
python experiments/06_random_baseline.py ./models/Mistral-7B-v0.3 ./models/Mistral-7B-Instruct-v0.3 cuda:0 mistral-7b 20
python experiments/06_random_baseline.py ./models/Yi-1.5-9B ./models/Yi-1.5-9B-Chat cuda:0 yi-1.5-9b 10
```

### 6. Cross-Dataset Transfer Validation (Section 5)

Validates that SAR component selection generalizes from BFCL to the probe set.

```bash
python experiments/07_cross_dataset_transfer.py ./models/Qwen2.5-7B ./models/Qwen2.5-7B-Instruct cuda:0 qwen2.5-7b
python experiments/07_cross_dataset_transfer.py ./models/Llama-3.1-8B ./models/Llama-3.1-8B-Instruct cuda:0 llama-3.1-8b
python experiments/07_cross_dataset_transfer.py ./models/Mistral-7B-v0.3 ./models/Mistral-7B-Instruct-v0.3 cuda:0 mistral-7b
```

### 7. Statistical Tests (Tables 1-4)

Page's test, block permutation test, and bootstrap CIs. Runs CPU-only on pre-computed results.

```bash
python experiments/08_statistical_tests.py
```

### 8. Benchmarks (Table 6)

Evaluates capability preservation: BFCL, HumanEval, IFEval, MMLU, Safety.

```bash
python evaluation/03_bfcl_standard.py cuda:0
python evaluation/07_humaneval.py cuda:0
python evaluation/06_ifeval.py cuda:0
python evaluation/09_mmlu.py cuda:0
python evaluation/11_safety_llm_judge.py qwen2.5-7b cuda:0
python evaluation/11_safety_llm_judge.py llama-3.1-8b cuda:0
python evaluation/11_safety_llm_judge.py mistral-7b cuda:0
python evaluation/11_safety_llm_judge.py yi-1.5-9b cuda:0
```

**Note:** Safety evaluation Phase 2 (LLM judge) requires access to an OpenAI-compatible LLM API. Set environment variables `LLM_JUDGE_BASE_URL`, `LLM_JUDGE_API_KEY`, and `LLM_JUDGE_MODEL` to configure the endpoint. Phase 1 (keyword-based detection) runs without external API access.

### 9. Adversarial Safety (Appendix C)

Tests SAR-5% vulnerability under GCG and AutoDAN attacks (n=50 prompts per attack type).

```bash
python experiments/09_adversarial_safety.py qwen2.5-7b cuda:0
python experiments/09_adversarial_safety.py llama-3.1-8b cuda:0
```

### 10. CPT Attribution Control (Appendix D)

Continual pretraining control: verifies the within-MLP hierarchy is alignment-specific, not an artifact of any fine-tuning.

```bash
python experiments/10_cpt_attribution.py qwen2.5-7b cuda:0
python experiments/10_cpt_attribution.py llama-3.1-8b cuda:0
```

### 11. Within-BFCL Split Validation (Section 5)

Anti-circularity test: attribution on one half of BFCL, evaluation on the other (10 random splits).

```bash
python experiments/11_bfcl_split_validation.py cuda:0
```

### 12. 14B Scaling (Appendix E)

Validates attribution and OC-DPO at 14B scale. Requires 2x 24GB GPUs (fp16, device_map=auto).

```bash
python experiments/12_scaling_14b.py
python experiments/12_scaling_14b.py --attribution-only  # skip OC-DPO, attribution only
python experiments/12_scaling_14b.py --no-wandb          # disable W&B logging
```

### 13. Figures

```bash
python figures/generate_figures.py
python figures/generate_fig_overview.py
```

## Quick Verification (No GPU Required)

For reviewers with limited compute, pre-computed results are provided in `results/`. To verify key claims without running experiments:

```bash
# Verify statistical claims from pre-computed attribution data (CPU only, <1 min)
python experiments/08_statistical_tests.py

# Regenerate paper figures from pre-computed data (CPU only, <10 sec)
python figures/generate_figures.py
python figures/generate_fig_overview.py
```

## Results

### Attribution (Tables 1-2): V/O vs Q/K Harm Share

| Model | V/O Share | Q/K Share | V/O Ratio | p_block |
|-------|-----------|-----------|-----------|---------|
| Qwen2.5-7B | 20.2% | 16.8% | 1.20x | 0.28 |
| Llama-3.1-8B | 26.8% | 14.2% | 1.89x | 1.1e-4 |
| Mistral-7B | 30.3% | 8.1% | 3.76x | <1e-5 |
| Yi-1.5-9B | 22.4% | 18.3% | 1.22x | 0.13 |

### DPO Gradient Ratios (Table 3)

| Model | V/O : Q/K |
|-------|-----------|
| Qwen2.5-7B | 1.76x |
| Llama-3.1-8B | 3.91x |
| Mistral-7B | 3.66x |
| Yi-1.5-9B | 2.74x |

### SAR-5% Recovery (Table 4)

| Model | Alignment Tax | SAR Recovery | Random (20 seeds) | SAR/Random |
|-------|--------------|-------------|-------------------|------------|
| Qwen2.5-7B | +0.095 | 62.2% | 14.8 ± 5.7% | 4.2x [3.6, 5.0] |
| Llama-3.1-8B | -0.027 | +0.026 Δloss | +0.005 ± 0.003 | 5.3x [4.1, 7.5] |
| Mistral-7B | -0.004 | +0.020 Δloss | +0.007 ± 0.004 | 2.8x [2.3, 3.6] |
| Yi-1.5-9B | +0.102 | 62.2% | -10.8 ± 37.9% | — |

### OC-DPO Tax Reduction (Table 5)

| Condition | Qwen | Llama | Mistral |
|-----------|------|-------|---------|
| OC-DPO (excl. V/O/down) | 94.5% | 133.5% | 122.1% |
| Exclude Q/K (control) | 8.9% | -19.9% | 10.2% |

### Within-MLP Ordering (Page's Test)

| Model | Z | p-value |
|-------|---|---------|
| Qwen2.5-7B | 3.88 | 5.3e-5 |
| Llama-3.1-8B | 2.88 | 2.0e-3 |
| Mistral-7B | 6.25 | 2.1e-10 |
| Yi-1.5-9B | 0.92 | 0.18 |

### BFCL AST Accuracy (Table 6, partial)

| Model | Base | IT | SAR-5% | IT→SAR Δ |
|-------|------|-----|--------|----------|
| Qwen2.5-7B | 76.1% | 83.3% | 84.3% | +0.9pp |
| Llama-3.1-8B | 72.6% | 78.0% | 78.9% | +0.9pp |
| Mistral-7B | 71.4% | 72.0% | 73.3% | +1.3pp |

Full Table 6 (HumanEval, IFEval, MMLU, Safety) requires GPU execution of evaluation scripts. Pre-computed BFCL results are in `bfcl_standard.json`.

## Pre-computed Results

JSON files in `results/` contain all data needed to reproduce paper tables:

| File | Paper Reference |
|------|----------------|
| `attribution_*.json` | Tables 1-2 (harm scores, 209 examples) |
| `gradient_analysis_*.json` | Table 3 (DPO gradient ratios) |
| `sar_eval_*.json` | Table 4 (SAR recovery at k=3/5/8/10%) |
| `random_baseline_*.json` | Table 4 caption (20-seed random baseline, Yi: 10 seeds; bootstrap CIs) |
| `ocdpo_eval_*.json` | Table 5 (OC-DPO tax reduction, 6 conditions) |
| `bfcl_standard.json` | Table 6 (BFCL AST accuracy) |
| `ifeval_combined.json` | Table 6 (IFEval instruction following) |
| `humaneval_combined.json` | Table 6 (HumanEval code generation) |
| `mmlu_combined.json` | Table 6 (MMLU knowledge retention) |
| `safety_judge_*.json` | Table 6 (safety refusal rates, Phase 1 keyword detection — see note below) |
| `statistical_tests.json` | Combined output: Page's test, block permutation, bootstrap CIs |
| `adversarial_safety_*.json` | Appendix C (GCG/AutoDAN ASR, Fisher exact p-values) |
| `cpt_attribution_*.json` | Appendix D (CPT control attribution shares) |
| `bfcl_split_validation.json` | Section 5 (within-BFCL split, 10 random splits) |
| `cross_dataset_transfer.json` | Section 5 (BFCL→probe transfer, 20-seed random) |
| `scaling_14b.json` | Appendix E (14B attribution + OC-DPO replication) |

**Note on safety results:** Pre-computed `safety_judge_*.json` files contain Phase 1 (keyword detection) results. The paper's Table 6 reports Phase 2 (LLM judge) values, which require an external LLM judge API to reproduce. Keyword detection may over- or under-estimate refusal rates by 15–25pp depending on the model's refusal style (e.g., keyword detection overestimates refusal for Mistral by ~24pp and underestimates for Llama by ~17pp). To reproduce Table 6 exactly, run `evaluation/11_safety_llm_judge.py` with API credentials as described in Section 8 above.

## Repository Structure

```
├── experiments/              # Core analysis and intervention scripts
│   ├── agent_examples_200.py     # 209-example evaluation dataset
│   ├── weight_utils.py           # Shared utilities (classify_component, CE loss)
│   ├── 01_attribution.py         # Weight-patching attribution (Tables 1-2)
│   ├── 03_gradient_analysis.py   # DPO gradient analysis (Table 3)
│   ├── 04_sar.py                 # SAR intervention (Table 4)
│   ├── 05_ocdpo.py              # OC-DPO training (Table 5)
│   ├── 06_random_baseline.py     # Multi-seed random baseline + bootstrap CIs
│   ├── 07_cross_dataset_transfer.py  # BFCL → probe set transfer validation
│   ├── 08_statistical_tests.py   # Page's test, block permutation, bootstrap
│   ├── 09_adversarial_safety.py  # GCG + AutoDAN attacks (Appendix C)
│   ├── 10_cpt_attribution.py    # CPT control experiment (Appendix D)
│   ├── 11_bfcl_split_validation.py  # Within-BFCL split (Section 5)
│   └── 12_scaling_14b.py        # 14B replication (Appendix E)
├── evaluation/               # Benchmark evaluation scripts
│   ├── 03_bfcl_standard.py      # BFCL function calling (Table 6)
│   ├── 06_ifeval.py             # IFEval instruction following (Table 6)
│   ├── 07_humaneval.py          # HumanEval code generation (Table 6)
│   ├── 09_mmlu.py               # MMLU knowledge (Table 6)
│   └── 11_safety_llm_judge.py   # Safety refusal (Table 6)
├── figures/                  # Figure generation scripts
│   ├── generate_figures.py       # Figures 2-4 (attribution heatmap, gradient, harm share)
│   └── generate_fig_overview.py  # Figure 1 (pipeline overview)
├── data/                    # External datasets (not included; see download instructions)
├── results/                  # Pre-computed result JSON files
├── requirements.txt          # Python dependencies
└── LICENSE                   # MIT License
```

## Responsible Use Notice

This code implements techniques for analyzing and selectively modifying alignment in LLMs. The SAR (Surgical Alignment Reversal) technique is training-free, requiring only pre-computed attribution scores and model weights, which means it has a low deployment barrier. Rolling back alignment-correlated components can measurably reduce safety refusal rates. Quantified safety deltas at k=5% (LLM judge / keyword): Qwen -0.8 / +0.4 pp, Llama -2.0 / -0.8 pp, Mistral -1.2 / 0.0 pp. However, higher budgets amplify degradation: Llama at k=10% shows -12.8 pp refusal drop, and Yi-family models exhibit a qualitatively distinct safety-structure overlap with -13.6 pp degradation even at k=5%. This code should only be used for research purposes with appropriate safety evaluation. Do not deploy SAR-modified models without comprehensive safety auditing covering harm categories beyond those in AdvBench (which over-represents cyber/hacking prompts and under-represents hate speech, self-harm, and indirect harms). Any re-use of SAR must include full safety re-evaluation before deployment (see paper Section 5.3 and Appendix C for safety-structure overlap analysis).

## License

MIT License. See [LICENSE](LICENSE). The Responsible Use Notice above is an ethical guidance statement; users are strongly encouraged to follow it.
