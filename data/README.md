# data/

This directory holds external datasets that cannot be redistributed with the codebase.

## Required file: `harmful_behaviors.csv`

Used by `evaluation/11_safety_llm_judge.py` (safety refusal evaluation, Table 6).

Source: AdvBench (Zou et al., 2023), publicly available at:
https://github.com/llm-attacks/llm-attacks/blob/main/data/advbench/harmful_behaviors.csv

Download (run from repo root):

```bash
wget -O data/harmful_behaviors.csv \
  https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv
```

Without this file the safety script falls back to a minimal 20-prompt subset and results
will differ from those reported in the paper. The script prints a clear warning in that case.

All commands in the README assume the **repo root** as the working directory, so the
script must be invoked as:

```bash
python evaluation/11_safety_llm_judge.py <model_key> <cuda_device>
```

not `cd evaluation && python 11_safety_llm_judge.py ...`.

## Required directory: `data/bfcl/`

Used by `experiments/11_bfcl_split_validation.py` (within-BFCL split validation, Appendix G).

Source: Berkeley Function Calling Leaderboard (Patil et al., 2025), available at:
https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard

Required files:
```
data/bfcl/
├── BFCL_v3_simple.json
├── BFCL_v3_multiple.json
├── BFCL_v3_parallel.json
├── BFCL_v3_live_relevance.json
└── possible_answer/
    ├── BFCL_v3_simple.json
    ├── BFCL_v3_multiple.json
    ├── BFCL_v3_parallel.json
    └── BFCL_v3_live_relevance.json
```

Without these files, only the BFCL split validation script is affected. All other
experiments use the 209-example probe set included in `experiments/agent_examples_200.py`.
