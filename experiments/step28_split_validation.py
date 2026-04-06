"""
Step 28: Split Validation — attribute on first 100 examples, evaluate SAR on last 109.

Addresses reviewer concern that SAR may overfit to the same examples used for attribution.
If SAR still recovers alignment tax on held-out examples, overfitting is ruled out.

Runs one model per GPU. Usage:
    python step28_split_validation.py qwen2.5-7b cuda:0
    python step28_split_validation.py llama-3.1-8b cuda:1
    python step28_split_validation.py mistral-7b cuda:2
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from safetensors import safe_open

RESULTS_DIR = Path("./results")

from agent_examples_200 import AGENT_EXAMPLES_200

MODEL_CONFIGS = {
    "qwen2.5-7b": {
        "base": "./models/Qwen2.5-7B",
        "it": "Qwen/Qwen2.5-7B-Instruct",
    },
    "llama-3.1-8b": {
        "base": "./models/Llama-3.1-8B",
        "it": "./models/Llama-3.1-8B-Instruct",
    },
    "mistral-7b": {
        "base": "./models/Mistral-7B-v0.3",
        "it": "./models/Mistral-7B-Instruct-v0.3",
    },
}

SPLIT_POINT = 100  # first 100 for attribution, last 109 for evaluation
K_PERCENT = 5.0    # top 5% components to roll back (same as main paper)


def compute_loss(model, tokenizer, prompt, target, device):
    """Compute cross-entropy loss on the target portion only."""
    full_text = prompt + target
    inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(device)
    prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
    if prompt_len >= input_ids.shape[1]:
        return None
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    logits = outputs.logits
    shift_logits = logits[0, prompt_len - 1:-1, :]
    shift_labels = input_ids[0, prompt_len:]
    loss = torch.nn.functional.cross_entropy(shift_logits, shift_labels, reduction='mean')
    return loss.item()


def compute_mean_loss(model, tokenizer, examples, device):
    """Compute mean loss over a list of examples."""
    losses = []
    for ex in examples:
        target = ex.get("target", ex.get("chosen", ""))
        loss = compute_loss(model, tokenizer, ex["prompt"], target, device)
        if loss is not None:
            losses.append(loss)
    return np.mean(losses) if losses else None, len(losses)


def classify_component(name):
    """Classify a weight parameter into component type and layer."""
    if "layers." not in name:
        return "other", -1
    parts = name.split(".")
    layer_idx = None
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                layer_idx = int(parts[i + 1])
            except ValueError:
                pass
    if layer_idx is None:
        return "other", -1
    if "q_proj" in name: return "W_Q", layer_idx
    elif "k_proj" in name: return "W_K", layer_idx
    elif "v_proj" in name: return "W_V", layer_idx
    elif "o_proj" in name: return "W_O", layer_idx
    elif "gate_proj" in name: return "W_gate", layer_idx
    elif "up_proj" in name: return "W_up", layer_idx
    elif "down_proj" in name: return "W_down", layer_idx
    return "other", layer_idx


def run_split_validation(model_name, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import glob

    config = MODEL_CONFIGS[model_name]
    base_dir = config["base"]
    it_dir = config["it"]

    all_examples = AGENT_EXAMPLES_200  # 209 total
    attr_examples = all_examples[:SPLIT_POINT]      # first 100
    eval_examples = all_examples[SPLIT_POINT:]       # last 109

    print(f"Model: {model_name}")
    print(f"Attribution split: {len(attr_examples)} examples (0-{SPLIT_POINT-1})")
    print(f"Evaluation split: {len(eval_examples)} examples ({SPLIT_POINT}-{len(all_examples)-1})")
    print(f"Device: {device}")
    print()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # =========================================================
    # Phase 0: Load base model FIRST to get reference losses
    # (must do this before IT model to avoid OOM on single GPU)
    # =========================================================
    import gc
    print("Phase 0: Loading base model for reference losses...")
    model = AutoModelForCausalLM.from_pretrained(
        base_dir, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
    )
    model.eval()

    base_loss_attr, _ = compute_mean_loss(model, tokenizer, attr_examples, device)
    print(f"  Attribution split: Base loss = {base_loss_attr:.6f}")
    base_loss_eval, _ = compute_mean_loss(model, tokenizer, eval_examples, device)
    print(f"  Evaluation split: Base loss = {base_loss_eval:.6f}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # =========================================================
    # Phase 1: Load IT model and compute baseline losses
    # =========================================================
    print("\nPhase 1: Loading IT model...")
    model = AutoModelForCausalLM.from_pretrained(
        it_dir, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
    )
    model.eval()

    # Index base weight safetensor files
    print("Indexing base weights...")
    base_safetensors = sorted(glob.glob(str(Path(base_dir) / "*.safetensors")))
    it_params = dict(model.named_parameters())

    components = []
    for sf_path in base_safetensors:
        with safe_open(sf_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in it_params and "weight" in key:
                    comp_type, layer = classify_component(key)
                    if comp_type != "other":
                        components.append({
                            "name": key,
                            "type": comp_type,
                            "layer": layer,
                            "sf_path": sf_path,
                        })

    print(f"Found {len(components)} weight components")

    print("Computing IT baseline losses...")
    it_loss_attr, n_attr = compute_mean_loss(model, tokenizer, attr_examples, device)
    print(f"  Attribution split (n={n_attr}): IT loss = {it_loss_attr:.6f}")
    it_loss_eval, n_eval = compute_mean_loss(model, tokenizer, eval_examples, device)
    print(f"  Evaluation split (n={n_eval}): IT loss = {it_loss_eval:.6f}")

    # =========================================================
    # Phase 2: Attribution on first 100 examples only
    # =========================================================
    print(f"\nPhase 2: Running attribution on {len(attr_examples)} examples...")

    # Compute per-example baseline losses for attribution split
    attr_baselines = []
    for i, ex in enumerate(attr_examples):
        target = ex.get("target", ex.get("chosen", ""))
        loss = compute_loss(model, tokenizer, ex["prompt"], target, device)
        attr_baselines.append(loss)

    valid_attr = [(i, ex, bl) for i, (ex, bl) in
                  enumerate(zip(attr_examples, attr_baselines)) if bl is not None]
    print(f"  Valid attribution examples: {len(valid_attr)}/{len(attr_examples)}")

    # Activation patching: for each component, replace IT weight with base weight
    for ci, comp in enumerate(components):
        if (ci + 1) % 20 == 0:
            print(f"  Component {ci+1}/{len(components)}: {comp['name']}")

        target_param = it_params[comp["name"]]
        original_data = target_param.data.clone()

        # Load base weight
        with safe_open(comp["sf_path"], framework="pt", device="cpu") as f:
            base_w = f.get_tensor(comp["name"]).to(target_param.device, dtype=target_param.dtype)

        scores = []
        for vi, ex, bl in valid_attr:
            target_param.data.copy_(base_w)
            target = ex.get("target", ex.get("chosen", ""))
            patched_loss = compute_loss(model, tokenizer, ex["prompt"], target, device)
            target_param.data.copy_(original_data)

            if patched_loss is not None:
                # Positive = harmful (rolling back helps = loss decreases)
                scores.append(bl - patched_loss)

        comp["mean_score"] = np.mean(scores) if scores else 0.0
        del base_w, original_data
        torch.cuda.empty_cache()

    print("  Attribution complete.")

    # =========================================================
    # Phase 3: Select top-k% components and apply SAR
    # =========================================================
    ranked = sorted(components, key=lambda c: c["mean_score"], reverse=True)
    k = max(1, int(len(ranked) * K_PERCENT / 100))
    selected = ranked[:k]

    type_counts = defaultdict(int)
    for c in selected:
        type_counts[c["type"]] += 1

    print(f"\nPhase 3: SAR rollback — top {K_PERCENT}% = {k} components")
    print(f"  Composition: {dict(type_counts)}")

    # Apply rollback in-place
    rolled_back = 0
    for comp in selected:
        target_param = it_params[comp["name"]]
        with safe_open(comp["sf_path"], framework="pt", device="cpu") as f:
            base_w = f.get_tensor(comp["name"]).to(target_param.device, dtype=target_param.dtype)
        target_param.data.copy_(base_w)
        rolled_back += 1
        del base_w
    torch.cuda.empty_cache()
    print(f"  Rolled back {rolled_back} components")

    # =========================================================
    # Phase 4: Evaluate SAR model on BOTH splits
    # =========================================================
    print("\nPhase 4: Evaluating SAR model...")
    sar_loss_attr, _ = compute_mean_loss(model, tokenizer, attr_examples, device)
    print(f"  Attribution split: SAR loss = {sar_loss_attr:.6f}")
    sar_loss_eval, _ = compute_mean_loss(model, tokenizer, eval_examples, device)
    print(f"  Evaluation split: SAR loss = {sar_loss_eval:.6f}")

    del model
    torch.cuda.empty_cache()

    # =========================================================
    # Results
    # =========================================================
    # Alignment tax = IT - Base (positive = IT is worse)
    tax_attr = it_loss_attr - base_loss_attr
    tax_eval = it_loss_eval - base_loss_eval

    # SAR recovery = IT - SAR (positive = SAR improved over IT)
    recovery_attr = it_loss_attr - sar_loss_attr
    recovery_eval = it_loss_eval - sar_loss_eval

    # Recovery percentage
    pct_attr = (recovery_attr / tax_attr * 100) if tax_attr > 0 else float('nan')
    pct_eval = (recovery_eval / tax_eval * 100) if tax_eval > 0 else float('nan')

    print(f"\n{'='*60}")
    print(f"SPLIT VALIDATION RESULTS: {model_name}")
    print(f"{'='*60}")
    print(f"{'':20s} {'Attr split':>14s} {'Eval split':>14s}")
    print(f"{'Base loss':20s} {base_loss_attr:14.6f} {base_loss_eval:14.6f}")
    print(f"{'IT loss':20s} {it_loss_attr:14.6f} {it_loss_eval:14.6f}")
    print(f"{'Alignment tax':20s} {tax_attr:+14.6f} {tax_eval:+14.6f}")
    print(f"{'SAR loss':20s} {sar_loss_attr:14.6f} {sar_loss_eval:14.6f}")
    print(f"{'SAR recovery':20s} {recovery_attr:+14.6f} {recovery_eval:+14.6f}")
    print(f"{'Recovery %':20s} {pct_attr:13.1f}% {pct_eval:13.1f}%")
    print()

    if recovery_eval > 0:
        print("PASS: SAR improves on held-out examples — no overfitting.")
    else:
        print("WARN: SAR does not improve on held-out examples.")

    results = {
        "model": model_name,
        "split_point": SPLIT_POINT,
        "n_attr": len(attr_examples),
        "n_eval": len(eval_examples),
        "k_percent": K_PERCENT,
        "n_components_rolled_back": rolled_back,
        "component_composition": dict(type_counts),
        "attr_split": {
            "base_loss": float(base_loss_attr),
            "it_loss": float(it_loss_attr),
            "alignment_tax": float(tax_attr),
            "sar_loss": float(sar_loss_attr),
            "sar_recovery": float(recovery_attr),
            "recovery_pct": float(pct_attr) if not np.isnan(pct_attr) else None,
        },
        "eval_split": {
            "base_loss": float(base_loss_eval),
            "it_loss": float(it_loss_eval),
            "alignment_tax": float(tax_eval),
            "sar_loss": float(sar_loss_eval),
            "sar_recovery": float(recovery_eval),
            "recovery_pct": float(pct_eval) if not np.isnan(pct_eval) else None,
        },
        "top_components": [
            {"name": c["name"], "type": c["type"], "layer": c["layer"],
             "mean_score": float(c["mean_score"])}
            for c in selected
        ],
    }

    out_path = RESULTS_DIR / f"step28_split_validation_{model_name}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    return results


if __name__ == "__main__":
    model_name = sys.argv[1]
    device = sys.argv[2] if len(sys.argv) > 2 else "cuda:0"
    run_split_validation(model_name, device)
