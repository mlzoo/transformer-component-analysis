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
├── experiments/          # Core experiment scripts
│   ├── agent_examples_200.py       # 209-example evaluation set (17+ categories)
│   ├── step1_attribution.py        # Activation-patching attribution (Tables 1-2, Fig 1)
│   ├── step2_causal_intervention.py # Causal rollback validation (Appendix B)
│   ├── step3_gradient_analysis.py  # DPO gradient magnitude analysis (Table 3, Fig 2)
│   ├── step4_sar_implementation.py # SAR: Surgical Alignment Reversal (Table 4)
│   ├── step6_ocdpo.py             # OC-DPO: Output-Constrained DPO (Table 5)
│   ├── step12_scaling_14b.py      # 14B scaling validation (Table 1)
│   ├── step19_ogpsa_large.py      # OGPSA comparison (Appendix F)
│   ├── step26_bonferroni.py       # Statistical corrections
│   ├── step28_split_validation.py # Split validation (Appendix M)
│   ├── step29_per_category_tax.py # Per-category alignment tax analysis
│   ├── step30_sft_gradient.py     # SFT gradient verification (Appendix J)
│   ├── step31_bootstrap_split.py  # Bootstrap split validation (Appendix M)
│   ├── step32_ogpsa_hybrid.py     # OGPSA-hybrid (Appendix F)
│   ├── step34_pairwise_interaction.py # Pairwise interaction analysis (Appendix N)
│   ├── step35_sar_bootstrap_ci.py # SAR bootstrap CIs (Appendix O)
│   ├── step36_gradient_dot_delta.py # Gradient-dot-delta attribution (Appendix K)
│   ├── step38_bfcl_attribution.py # External BFCL attribution (Appendix L)
│   ├── step41_dare_ties_comparison.py # DARE/TIES baselines (Appendix C)
│   └── step42_gradient_72b.py     # 72B gradient analysis (Appendix I)
├── evaluation/           # Benchmark evaluation scripts
│   ├── step9_bfcl.py              # Berkeley Function Calling (BFCL)
│   ├── step13_bfcl_standard.py    # BFCL standard evaluation (Table 6, Appendix H)
│   ├── step14_safety_large.py     # Safety evaluation, 250 prompts (Table 6, Appendix E-F)
│   ├── step16_ocdpo_seeds.py      # OC-DPO seed robustness (Appendix D)
│   ├── step17_ifeval.py           # IFEval instruction following (Table 6)
│   ├── step18_humaneval.py        # HumanEval code generation (Table 6)
│   ├── step21_mmlu_1000.py        # MMLU knowledge evaluation (Table 6)
│   ├── step25_alignment_tax_209.py # Alignment tax computation
│   ├── step43_safety_llm_judge.py # LLM judge safety evaluation (Appendix A)
│   └── step44_ocdpo_safety_judge.py # OC-DPO safety (Appendix E)
├── figures/              # Figure generation scripts
│   ├── generate_figures.py        # Main paper figures (Figs 1-2)
│   └── generate_fig_overview.py   # Overview figure (Fig 0)
├── results/              # Pre-computed result JSON files
└── requirements.txt
```

## Reproducing Key Results

Each script is self-contained. Check the `if __name__ == "__main__"` block for CLI arguments.

### 1. Attribution Analysis (Section 4, Tables 1-2)
```bash
# Activation-patching attribution across models
python experiments/step1_attribution.py ./models/Qwen2.5-7B ./models/Qwen2.5-7B-Instruct cuda:0 qwen
python experiments/step1_attribution.py ./models/Llama-3.1-8B ./models/Llama-3.1-8B-Instruct cuda:0 llama
python experiments/step1_attribution.py ./models/Mistral-7B-v0.3 ./models/Mistral-7B-Instruct-v0.3 cuda:0 mistral
```

### 2. Gradient Analysis (Section 4.2, Table 3)
```bash
python experiments/step3_gradient_analysis.py ./models/Qwen2.5-7B cuda:0 qwen
python experiments/step3_gradient_analysis.py ./models/Llama-3.1-8B cuda:0 llama
python experiments/step3_gradient_analysis.py ./models/Mistral-7B-v0.3 cuda:0 mistral
```

### 3. SAR (Section 5, Table 4)
```bash
python experiments/step4_sar_implementation.py ./models/Qwen2.5-7B ./models/Qwen2.5-7B-Instruct cuda:0
```

### 4. OC-DPO (Section 6, Table 5)
```bash
python experiments/step6_ocdpo.py qwen cuda:0
python experiments/step6_ocdpo.py llama cuda:0
python experiments/step6_ocdpo.py mistral cuda:0
```

### 5. Benchmarks (Section 7, Table 6)
```bash
python evaluation/step13_bfcl_standard.py       # BFCL
python evaluation/step18_humaneval.py            # HumanEval
python evaluation/step17_ifeval.py               # IFEval
python evaluation/step21_mmlu_1000.py            # MMLU
python evaluation/step14_safety_large.py qwen cuda:0  # Safety
```

### 6. Scaling (Section 8)
```bash
python experiments/step12_scaling_14b.py cuda:0  # 14B attribution
python experiments/step42_gradient_72b.py        # 72B gradient analysis
```

### 7. Figures
```bash
python figures/generate_figures.py      # Figs 1-2
python figures/generate_fig_overview.py # Fig 0 (overview)
```

## Hardware Requirements

- **7B/8B experiments**: Single GPU with 24GB+ VRAM (A100 40GB recommended)
- **14B experiments**: Single GPU with 40GB+ VRAM
- **72B gradient analysis**: Multi-GPU setup (4× A100 80GB)
- All experiments use greedy decoding; no training is needed for SAR

## Evaluation Data

The 209-example evaluation set is defined in `experiments/agent_examples_200.py`, covering 17+ structured-generation categories: tool calls, JSON generation, SQL, code, API requests, bash commands, configuration files, ReAct chains, and more.

## Results

Pre-computed results from all experiments are provided in `results/` as JSON files for verification.
