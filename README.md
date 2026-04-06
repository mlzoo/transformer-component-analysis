# Alignment Concentrates in Output Pathways

Code for the paper: *Alignment Concentrates in Output Pathways: Component-Level Analysis of How Instruction Tuning Modifies Structured Generation Circuits*

## Overview

This repository contains all experiment scripts, evaluation code, and figure generation code for reproducing the paper's results. The codebase investigates how instruction tuning modifies transformer internals at the component level (W_Q, W_K, W_V, W_O, W_gate, W_up, W_down), identifies consistent modification patterns across three model families, and validates findings through Surgical Alignment Reversal (SAR) and Output-Constrained DPO (OC-DPO).

## Setup

### Requirements
```bash
pip install -r requirements.txt
```

### Models
Download the following HuggingFace model pairs to `./models/`:
- `Qwen/Qwen2.5-7B` and `Qwen/Qwen2.5-7B-Instruct`
- `meta-llama/Llama-3.1-8B` and `meta-llama/Llama-3.1-8B-Instruct`
- `mistralai/Mistral-7B-v0.3` and `mistralai/Mistral-7B-Instruct-v0.3`
- `Qwen/Qwen2.5-14B` and `Qwen/Qwen2.5-14B-Instruct` (for scaling experiments)

## Repository Structure

```
├── experiments/              # Core experiment scripts
│   ├── agent_examples_200.py     # 209-example evaluation set (17+ categories)
│   ├── 01_attribution.py         # Activation-patching attribution (Tables 1-2, Fig 1)
│   ├── 02_causal_intervention.py # Causal rollback validation (Appendix B)
│   ├── 03_gradient_analysis.py   # DPO gradient magnitude analysis (Table 3, Fig 2)
│   ├── 04_sar.py                 # SAR: Surgical Alignment Reversal (Table 4)
│   ├── 05_ocdpo.py              # OC-DPO: Output-Constrained DPO (Table 5)
│   ├── 06_scaling_14b.py        # 14B scaling validation (Table 1)
│   ├── 07_ogpsa_comparison.py   # OGPSA comparison (Appendix F)
│   ├── 08_statistical_corrections.py # Bonferroni corrections
│   ├── 09_split_validation.py   # Split validation (Appendix M)
│   ├── 10_per_category_tax.py   # Per-category alignment tax analysis
│   ├── 11_sft_gradient.py       # SFT gradient verification (Appendix J)
│   ├── 12_bootstrap_split.py    # Bootstrap split validation (Appendix M)
│   ├── 13_ogpsa_hybrid.py       # OGPSA-hybrid (Appendix F)
│   ├── 14_pairwise_interaction.py # Pairwise interaction analysis (Appendix N)
│   ├── 15_sar_bootstrap_ci.py   # SAR bootstrap CIs (Appendix O)
│   ├── 16_gradient_dot_delta.py # Gradient-dot-delta attribution (Appendix K)
│   ├── 17_bfcl_attribution.py   # External BFCL attribution (Appendix L)
│   ├── 18_dare_ties_comparison.py # DARE/TIES baselines (Appendix C)
│   └── 19_gradient_72b.py       # 72B gradient analysis (Appendix I)
├── evaluation/               # Benchmark evaluation scripts
│   ├── 01_benchmarks_utils.py    # Shared evaluation utilities
│   ├── 02_bfcl.py               # Berkeley Function Calling (BFCL)
│   ├── 03_bfcl_standard.py      # BFCL standard evaluation (Table 6, Appendix H)
│   ├── 04_safety.py             # Safety evaluation, 250 prompts (Table 6, Appendix E-F)
│   ├── 05_ocdpo_seeds.py        # OC-DPO seed robustness (Appendix D)
│   ├── 06_ifeval.py             # IFEval instruction following (Table 6)
│   ├── 07_humaneval.py          # HumanEval code generation (Table 6)
│   ├── 08_ocdpo_safety.py       # OC-DPO safety evaluation
│   ├── 09_mmlu.py               # MMLU knowledge evaluation (Table 6)
│   ├── 10_alignment_tax.py      # Alignment tax computation
│   ├── 11_safety_llm_judge.py   # LLM judge safety evaluation (Appendix A)
│   └── 12_ocdpo_safety_judge.py # OC-DPO safety with LLM judge (Appendix E)
├── figures/                  # Figure generation scripts
│   ├── generate_figures.py       # Main paper figures (Figs 1-2)
│   └── generate_fig_overview.py  # Overview figure (Fig 0)
├── results/                  # Pre-computed result JSON files
└── requirements.txt
```

## Reproducing Key Results

Each script is self-contained. Check the `if __name__ == "__main__"` block for CLI arguments. All commands should be run from the repository root.

### 1. Attribution Analysis (Section 4, Tables 1-2)
```bash
python experiments/01_attribution.py ./models/Qwen2.5-7B ./models/Qwen2.5-7B-Instruct cuda:0 qwen
python experiments/01_attribution.py ./models/Llama-3.1-8B ./models/Llama-3.1-8B-Instruct cuda:0 llama
python experiments/01_attribution.py ./models/Mistral-7B-v0.3 ./models/Mistral-7B-Instruct-v0.3 cuda:0 mistral
```

### 2. Gradient Analysis (Section 4.2, Table 3)
```bash
python experiments/03_gradient_analysis.py ./models/Qwen2.5-7B cuda:0 qwen
python experiments/03_gradient_analysis.py ./models/Llama-3.1-8B cuda:0 llama
python experiments/03_gradient_analysis.py ./models/Mistral-7B-v0.3 cuda:0 mistral
```

### 3. SAR (Section 5, Table 4)
```bash
python experiments/04_sar.py ./models/Qwen2.5-7B ./models/Qwen2.5-7B-Instruct cuda:0
```

### 4. OC-DPO (Section 6, Table 5)
```bash
python experiments/05_ocdpo.py qwen cuda:0
python experiments/05_ocdpo.py llama cuda:0
python experiments/05_ocdpo.py mistral cuda:0
```

### 5. Benchmarks (Section 7, Table 6)
```bash
python evaluation/03_bfcl_standard.py           # BFCL
python evaluation/07_humaneval.py               # HumanEval
python evaluation/06_ifeval.py                  # IFEval
python evaluation/09_mmlu.py                    # MMLU
python evaluation/04_safety.py qwen cuda:0      # Safety
```

### 6. Scaling (Section 8)
```bash
python experiments/06_scaling_14b.py cuda:0     # 14B attribution
python experiments/19_gradient_72b.py           # 72B gradient analysis
```

### 7. Figures
```bash
python figures/generate_figures.py              # Figs 1-2
python figures/generate_fig_overview.py         # Fig 0 (overview)
```

## Hardware Requirements

- **7B/8B experiments**: Single GPU with 24GB+ VRAM (A100 40GB recommended)
- **14B experiments**: Single GPU with 40GB+ VRAM
- **72B gradient analysis**: Multi-GPU setup (4x A100 80GB)
- All experiments use greedy decoding; no training is needed for SAR

## Evaluation Data

The 209-example evaluation set is defined in `experiments/agent_examples_200.py`, covering 17+ structured-generation categories: tool calls, JSON generation, SQL, code, API requests, bash commands, configuration files, ReAct chains, and more.

## Results

Pre-computed results from all experiments are provided in `results/` as JSON files for verification.
