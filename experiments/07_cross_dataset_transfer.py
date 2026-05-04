"""
Cross-Dataset SAR Transfer Validation

Tests whether SAR component selection generalizes across datasets:
1. Select top-k% components using BFCL attribution (independent dataset)
2. Apply SAR rollback using those BFCL-derived components
3. Evaluate on probe set (209 examples)
4. Compare component overlap with probe-based selection

This validates that SAR is not overfitting to the probe set.
Results appear in Section 5 and Appendix (cross-dataset transfer).

Usage:
    python experiments/07_cross_dataset_transfer.py <base_dir> <it_dir> <device> <model_name>

Requires:
    - results/attribution_<model>.json (probe-set attribution from 01_attribution.py)
    - BFCL dataset from HuggingFace (gorilla-llm/Berkeley-Function-Calling-Leaderboard)
"""

import sys
import json
import torch
import gc
import numpy as np
from pathlib import Path
from collections import defaultdict
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).parent))

RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(exist_ok=True)

from agent_examples_200 import BASE_EXAMPLES, EXTRA_EXAMPLES
AGENT_EXAMPLES = BASE_EXAMPLES + EXTRA_EXAMPLES

PROJ_MAP = {
    "W_Q": "self_attn.q_proj", "W_K": "self_attn.k_proj",
    "W_V": "self_attn.v_proj", "W_O": "self_attn.o_proj",
    "W_gate": "mlp.gate_proj", "W_up": "mlp.up_proj",
    "W_down": "mlp.down_proj",
}


def comp_to_weight_key(comp_type, layer):
    proj_path = PROJ_MAP[comp_type]
    return f"model.layers.{layer}.{proj_path}.weight"


def compute_loss(model, tokenizer, examples, device):
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for ex in examples:
            full_text = ex["prompt"] + ex["target"]
            inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512)
            input_ids = inputs["input_ids"].to(device)
            prompt_len = tokenizer(ex["prompt"], return_tensors="pt")["input_ids"].shape[1]
            if prompt_len >= input_ids.shape[1]:
                continue
            outputs = model(input_ids=input_ids)
            logits = outputs.logits
            shift_logits = logits[0, prompt_len - 1:-1, :]
            shift_labels = input_ids[0, prompt_len:]
            loss = torch.nn.functional.cross_entropy(shift_logits, shift_labels, reduction='sum')
            total_loss += loss.item()
            total_tokens += shift_labels.shape[0]
    return total_loss / max(total_tokens, 1)


def run_bfcl_attribution(model, tokenizer, base_index, all_component_keys, device, n_examples=100):
    """Run weight-patching attribution on BFCL examples."""
    bfcl_loaded = False
    try:
        from datasets import load_dataset
        ds = load_dataset("gorilla-llm/Berkeley-Function-Calling-Leaderboard", split="train")
        bfcl_examples = []
        for item in ds:
            if len(bfcl_examples) >= n_examples:
                break
            if "question" in item and "function" in item:
                prompt = f"Functions: {item['function']}\n\nQuery: {item['question']}\n\nCall: "
                target = str(item.get("answer", ""))
                if target and len(target) > 5:
                    bfcl_examples.append({"prompt": prompt, "target": target})
        print(f"  Loaded {len(bfcl_examples)} BFCL examples")
        bfcl_loaded = True
    except Exception as e:
        print(f"  WARNING: Could not load BFCL dataset: {e}")
        print("  " + "=" * 70)
        print("  DATA LEAKAGE WARNING: Falling back to agent_examples_200 for")
        print("  component selection. These are the SAME examples used for")
        print("  evaluation, so this run does NOT constitute a valid cross-dataset")
        print("  transfer test. Results will be marked invalid in the output JSON.")
        print("  " + "=" * 70)
        bfcl_examples = AGENT_EXAMPLES[:100]

    # Baseline loss on BFCL
    bfcl_baseline = compute_loss(model, tokenizer, bfcl_examples, device)
    print(f"  BFCL baseline loss: {bfcl_baseline:.4f}")

    # Attribution: patch each component, measure loss change
    harm_scores = []
    param_dict = dict(model.named_parameters())

    for comp_key in all_component_keys:
        comp_type, layer = comp_key
        weight_key = comp_to_weight_key(comp_type, layer)
        if weight_key not in param_dict or weight_key not in base_index:
            continue

        # Save original weight
        original = param_dict[weight_key].data.clone()

        # Patch with base weight
        with safe_open(str(base_index[weight_key]), framework="pt", device="cpu") as sf:
            base_w = sf.get_tensor(weight_key)
        param_dict[weight_key].data.copy_(base_w.to(param_dict[weight_key].dtype).to(device))

        # Measure loss
        patched_loss = compute_loss(model, tokenizer, bfcl_examples, device)
        harm_score = bfcl_baseline - patched_loss

        harm_scores.append({
            "component_type": comp_type,
            "layer": layer,
            "harm_score": float(harm_score),
        })

        # Restore original
        param_dict[weight_key].data.copy_(original)

    return harm_scores, bfcl_loaded


def run_cross_dataset(base_dir, it_dir, device, model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load probe-based attribution
    attr_path = RESULTS_DIR / f"attribution_{model_name}.json"
    if not attr_path.exists():
        print(f"ERROR: Attribution file not found at {attr_path}")
        print("Run 01_attribution.py first.")
        sys.exit(1)

    with open(attr_path) as f:
        attr_data = json.load(f)
    probe_components = attr_data["components"]
    print(f"Loaded {len(probe_components)} components from probe attribution")

    # Build base weight index
    base_index = {}
    for f_path in sorted(Path(base_dir).glob("*.safetensors")):
        with safe_open(str(f_path), framework="pt", device="cpu") as sf:
            for key in sf.keys():
                base_index[key] = f_path

    # Load IT model
    print(f"Loading IT model...")
    tokenizer = AutoTokenizer.from_pretrained(it_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        it_dir, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
    )
    model.eval()

    # Get baselines
    it_loss = compute_loss(model, tokenizer, AGENT_EXAMPLES, device)
    print(f"IT probe-set loss: {it_loss:.4f}")

    # Compute BFCL attribution
    print("Running BFCL attribution (100 examples)...")
    all_comp_keys = [(c["component_type"], c["layer"]) for c in probe_components]
    bfcl_scores, bfcl_loaded = run_bfcl_attribution(model, tokenizer, base_index, all_comp_keys, device)

    # Sort by BFCL harm score and select top-k%
    bfcl_sorted = sorted(bfcl_scores, key=lambda x: x["harm_score"], reverse=True)
    k_percent = 5.0
    n_select = max(1, int(len(bfcl_sorted) * k_percent / 100))
    bfcl_top_k = bfcl_sorted[:n_select]
    print(f"Selected top {n_select} components by BFCL attribution")

    # Apply SAR using BFCL-derived components
    param_dict = dict(model.named_parameters())
    for comp in bfcl_top_k:
        weight_key = comp_to_weight_key(comp["component_type"], comp["layer"])
        if weight_key not in base_index or weight_key not in param_dict:
            continue
        with safe_open(str(base_index[weight_key]), framework="pt", device="cpu") as sf:
            base_w = sf.get_tensor(weight_key)
        param_dict[weight_key].data.copy_(base_w.to(param_dict[weight_key].dtype).to(param_dict[weight_key].device))

    # Evaluate cross-dataset SAR on probe set
    cross_loss = compute_loss(model, tokenizer, AGENT_EXAMPLES, device)
    print(f"Cross-dataset SAR loss: {cross_loss:.4f}")

    # Get base model loss for recovery computation
    del model
    gc.collect()
    torch.cuda.empty_cache()

    base_model = AutoModelForCausalLM.from_pretrained(
        base_dir, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
    )
    base_model.eval()
    base_loss = compute_loss(base_model, tokenizer, AGENT_EXAMPLES, device)
    print(f"Base model probe-set loss: {base_loss:.4f}")
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    # Compute metrics
    alignment_tax = it_loss - base_loss
    cross_recovery = (it_loss - cross_loss) / abs(alignment_tax) * 100 if abs(alignment_tax) > 1e-9 else 0

    # Compare component overlap
    probe_sorted = sorted(probe_components, key=lambda x: x["harm_score"], reverse=True)
    probe_top_k = set((c["component_type"], c["layer"]) for c in probe_sorted[:n_select])
    bfcl_top_k_set = set((c["component_type"], c["layer"]) for c in bfcl_top_k)
    overlap = len(probe_top_k & bfcl_top_k_set)
    overlap_pct = overlap / n_select * 100

    # Chance overlap
    total_components = len(probe_components)
    chance_overlap = n_select * n_select / total_components
    overlap_ratio = overlap / chance_overlap if chance_overlap > 0 else float('inf')

    results = {
        "analysis": "Cross-dataset SAR transfer (BFCL -> probe set)",
        "model": model_name,
        "transfer_valid": bfcl_loaded,
        "transfer_invalid_reason": (
            None if bfcl_loaded
            else "BFCL dataset unavailable; component selection used agent_examples_200 "
                 "(same data as evaluation). This is same-dataset evaluation, NOT a valid "
                 "cross-dataset transfer test. Do not cite these numbers as transfer evidence."
        ),
        "k_percent": k_percent,
        "n_bfcl_examples": 100,
        "n_probe_examples": len(AGENT_EXAMPLES),
        "baselines": {
            "it_loss": float(it_loss),
            "base_loss": float(base_loss),
            "alignment_tax": float(alignment_tax),
        },
        "cross_dataset_sar": {
            "loss": float(cross_loss),
            "recovery_pct": float(cross_recovery),
            "n_components": n_select,
        },
        "component_overlap": {
            "overlap_count": overlap,
            "overlap_pct": float(overlap_pct),
            "chance_overlap": float(chance_overlap),
            "overlap_over_chance": float(overlap_ratio),
        },
    }

    out_path = RESULTS_DIR / f"cross_dataset_transfer_{model_name}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")
    print(f"Cross-dataset recovery: {cross_recovery:.1f}%")
    print(f"Component overlap: {overlap}/{n_select} ({overlap_pct:.1f}%, {overlap_ratio:.1f}x over chance)")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python 07_cross_dataset_transfer.py <base_dir> <it_dir> <device> <model_name>")
        sys.exit(1)

    base_dir = sys.argv[1]
    it_dir = sys.argv[2]
    device = sys.argv[3]
    model_name = sys.argv[4] if len(sys.argv) > 4 else "unknown"

    run_cross_dataset(base_dir, it_dir, device, model_name)
