"""
Bootstrap CI for Split Validation Gap

Compute bootstrap confidence intervals for the gap between training-split and
held-out-split SAR recovery, to show the 13pp gap is within expected variance.

Uses existing split validation results + re-runs attribution with random splits.

Usage: python 12_bootstrap_split.py qwen2.5-7b cuda:0
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from safetensors import safe_open

RESULTS_DIR = Path("./results")

sys.path.insert(0, "./experiments")
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

K_PERCENT = 5.0
N_BOOTSTRAP = 50  # number of random splits
SPLIT_SIZE = 100   # attribution split size


def classify_component(name):
    if "layers." not in name:
        return "other", -1
    parts = name.split(".")
    layer_idx = None
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try: layer_idx = int(parts[i + 1])
            except: pass
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


def compute_loss(model, tokenizer, prompt, target, device):
    full_text = prompt + target
    inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(device)
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    prompt_len = prompt_ids.shape[1]
    if prompt_len >= input_ids.shape[1]:
        return None
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    logits = outputs.logits
    shift_logits = logits[0, prompt_len - 1:-1, :]
    shift_labels = input_ids[0, prompt_len:]
    loss = torch.nn.functional.cross_entropy(shift_logits, shift_labels, reduction='mean')
    return loss.item()


def run_bootstrap(model_name, device):
    import gc
    import glob
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = MODEL_CONFIGS[model_name]
    base_dir = config["base"]
    it_dir = config["it"]

    all_examples = AGENT_EXAMPLES_200
    n_total = len(all_examples)

    print(f"Model: {model_name}")
    print(f"Bootstrap splits: {N_BOOTSTRAP}")
    print(f"Split size: {SPLIT_SIZE} / {n_total - SPLIT_SIZE}")

    tokenizer = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model for reference losses
    print("\nLoading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_dir, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
    )
    model.eval()

    base_losses = []
    for ex in all_examples:
        target = ex.get("target", ex.get("chosen", ""))
        loss = compute_loss(model, tokenizer, ex["prompt"], target, device)
        base_losses.append(loss)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Load IT model
    print("Loading IT model...")
    model = AutoModelForCausalLM.from_pretrained(
        it_dir, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
    )
    model.eval()

    # Compute per-example IT losses
    it_losses = []
    for ex in all_examples:
        target = ex.get("target", ex.get("chosen", ""))
        loss = compute_loss(model, tokenizer, ex["prompt"], target, device)
        it_losses.append(loss)

    # Index components
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

    print(f"Found {len(components)} components")

    # Pre-compute per-example patched losses for ALL components
    # This is O(n_components * n_examples) but we only need to do it once
    print("\nComputing full attribution matrix (this takes a while)...")
    # Matrix: patched_losses[comp_idx][example_idx] = loss with comp rolled back
    # To save memory, compute harm scores directly: harm[comp][ex] = it_loss[ex] - patched_loss[comp][ex]
    harm_matrix = np.zeros((len(components), n_total))

    for ci, comp in enumerate(components):
        if (ci + 1) % 20 == 0:
            print(f"  Component {ci+1}/{len(components)}: {comp['name']}")

        target_param = it_params[comp["name"]]
        original_data = target_param.data.clone()

        with safe_open(comp["sf_path"], framework="pt", device="cpu") as f:
            base_w = f.get_tensor(comp["name"]).to(target_param.device, dtype=target_param.dtype)

        for ei, ex in enumerate(all_examples):
            if it_losses[ei] is None:
                continue
            target_param.data.copy_(base_w)
            target = ex.get("target", ex.get("chosen", ""))
            patched = compute_loss(model, tokenizer, ex["prompt"], target, device)
            target_param.data.copy_(original_data)

            if patched is not None:
                harm_matrix[ci, ei] = it_losses[ei] - patched

        del base_w, original_data
        torch.cuda.empty_cache()

    print("  Attribution matrix complete.")

    # Now run bootstrap: for each random split, compute SAR recovery on both splits
    rng = np.random.RandomState(42)
    gaps = []
    train_recoveries = []
    eval_recoveries = []

    valid_mask = np.array([l is not None and b is not None for l, b in zip(it_losses, base_losses)])

    for boot_i in range(N_BOOTSTRAP):
        # Random split
        indices = rng.permutation(n_total)
        train_idx = indices[:SPLIT_SIZE]
        eval_idx = indices[SPLIT_SIZE:]

        # Compute attribution scores on train split only
        train_valid = valid_mask[train_idx]
        comp_scores = []
        for ci in range(len(components)):
            train_harm = harm_matrix[ci, train_idx]
            valid_harm = train_harm[train_valid]
            comp_scores.append(np.mean(valid_harm) if len(valid_harm) > 0 else 0.0)

        # Select top-k%
        k = max(1, int(len(components) * K_PERCENT / 100))
        top_k_idx = np.argsort(comp_scores)[-k:]

        # Compute recovery = mean harm of selected components on each split
        # Recovery on a split = sum of harm scores of selected components averaged over examples
        def split_recovery(example_indices):
            valid_idx = example_indices[valid_mask[example_indices]]
            if len(valid_idx) == 0:
                return 0.0
            total_recovery = 0.0
            for ci in top_k_idx:
                total_recovery += np.mean(harm_matrix[ci, valid_idx])
            return total_recovery

        train_rec = split_recovery(train_idx)
        eval_rec = split_recovery(eval_idx)

        # Compute alignment tax on each split for percentage
        def split_tax(example_indices):
            valid_idx = example_indices[valid_mask[example_indices]]
            it_mean = np.mean([it_losses[i] for i in valid_idx])
            base_mean = np.mean([base_losses[i] for i in valid_idx])
            return it_mean - base_mean

        train_tax = split_tax(train_idx)
        eval_tax = split_tax(eval_idx)

        train_pct = (train_rec / train_tax * 100) if abs(train_tax) > 1e-6 else float('nan')
        eval_pct = (eval_rec / eval_tax * 100) if abs(eval_tax) > 1e-6 else float('nan')

        gap = train_pct - eval_pct if not (np.isnan(train_pct) or np.isnan(eval_pct)) else float('nan')

        train_recoveries.append(train_pct)
        eval_recoveries.append(eval_pct)
        gaps.append(gap)

        if (boot_i + 1) % 10 == 0:
            print(f"  Bootstrap {boot_i+1}/{N_BOOTSTRAP}: "
                  f"train={train_pct:.1f}%, eval={eval_pct:.1f}%, gap={gap:.1f}pp")

    del model
    torch.cuda.empty_cache()

    # Analysis
    valid_gaps = [g for g in gaps if not np.isnan(g)]
    valid_train = [t for t in train_recoveries if not np.isnan(t)]
    valid_eval = [e for e in eval_recoveries if not np.isnan(e)]

    print(f"\n{'='*60}")
    print(f"BOOTSTRAP SPLIT VALIDATION: {model_name}")
    print(f"{'='*60}")
    print(f"N bootstrap splits: {len(valid_gaps)}")
    print(f"\nTrain recovery %: {np.mean(valid_train):.1f} ± {np.std(valid_train):.1f}")
    print(f"Eval recovery %:  {np.mean(valid_eval):.1f} ± {np.std(valid_eval):.1f}")
    print(f"\nGap (train - eval): {np.mean(valid_gaps):.1f} ± {np.std(valid_gaps):.1f} pp")
    print(f"Gap 95% CI: [{np.percentile(valid_gaps, 2.5):.1f}, {np.percentile(valid_gaps, 97.5):.1f}] pp")
    print(f"Gap median: {np.median(valid_gaps):.1f} pp")

    results = {
        "analysis": "Bootstrap CI for split validation gap",
        "model": model_name,
        "n_bootstrap": N_BOOTSTRAP,
        "split_size": SPLIT_SIZE,
        "k_percent": K_PERCENT,
        "n_examples": n_total,
        "n_components": len(components),
        "train_recovery_mean": float(np.mean(valid_train)),
        "train_recovery_std": float(np.std(valid_train)),
        "eval_recovery_mean": float(np.mean(valid_eval)),
        "eval_recovery_std": float(np.std(valid_eval)),
        "gap_mean": float(np.mean(valid_gaps)),
        "gap_std": float(np.std(valid_gaps)),
        "gap_median": float(np.median(valid_gaps)),
        "gap_ci_2_5": float(np.percentile(valid_gaps, 2.5)),
        "gap_ci_97_5": float(np.percentile(valid_gaps, 97.5)),
        "all_gaps": [float(g) for g in valid_gaps],
        "all_train_recoveries": [float(t) for t in valid_train],
        "all_eval_recoveries": [float(e) for e in valid_eval],
    }

    out_path = RESULTS_DIR / f"bootstrap_split_{model_name}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    return results


if __name__ == "__main__":
    model_name = sys.argv[1]
    device = sys.argv[2] if len(sys.argv) > 2 else "cuda:0"
    run_bootstrap(model_name, device)
