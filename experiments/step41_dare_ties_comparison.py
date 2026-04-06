"""
Step 41: DARE and TIES-Merging baselines for comparison with SAR.

DARE (Yu et al. 2024): Randomly drop alignment deltas with probability p, then rescale.
  w_merged = w_base + mask * delta / (1-p), where mask ~ Bernoulli(1-p)

TIES-Merging (Yadav et al. 2023):
  1. Trim: zero out small deltas (below threshold)
  2. Resolve sign conflicts (majority vote)
  3. Merge remaining deltas

Both operate on alignment deltas (delta = w_it - w_base) and produce a merged model.
We compare these against SAR's attribution-guided component selection.

Usage: python step41_dare_ties_comparison.py <model_key> <cuda_device>
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

RESULTS_DIR = Path("./results")

MODEL_CONFIGS = {
    "qwen2.5-7b": {
        "base": "./models/Qwen2.5-7B",
        "it": "Qwen/Qwen2.5-7B-Instruct",
        "attribution": "step1_attribution_expanded_qwen2.5-7b.json",
    },
    "llama-3.1-8b": {
        "base": "./models/Llama-3.1-8B",
        "it": "./models/Llama-3.1-8B-Instruct",
        "attribution": "step1_attribution_expanded_llama-3.1-8b.json",
    },
    "mistral-7b": {
        "base": "./models/Mistral-7B-v0.3",
        "it": "./models/Mistral-7B-Instruct-v0.3",
        "attribution": "step1_attribution_expanded_mistral-7b.json",
    },
}

sys.path.insert(0, str(Path(__file__).parent))
from agent_examples_200 import AGENT_EXAMPLES_200


def compute_loss(model, tokenizer, examples, device):
    """Compute cross-entropy loss on agent examples (target tokens only)."""
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


def build_base_index(base_dir):
    """Index all safetensor files for base model weights."""
    base_index = {}
    for f in sorted(Path(base_dir).glob("*.safetensors")):
        with safe_open(str(f), framework="pt", device="cpu") as sf:
            for key in sf.keys():
                base_index[key] = f
    return base_index


def resolve_model_path(model_id):
    """Resolve a HuggingFace model ID or local path to a local directory."""
    p = Path(model_id)
    if p.exists():
        return str(p)
    # Try HuggingFace cache
    from huggingface_hub import snapshot_download
    return snapshot_download(model_id, local_files_only=True)


def build_it_index(it_dir):
    """Index all safetensor files for IT model weights."""
    it_dir = Path(resolve_model_path(str(it_dir)))
    it_index = {}
    for f in sorted(it_dir.glob("*.safetensors")):
        with safe_open(str(f), framework="pt", device="cpu") as sf:
            for key in sf.keys():
                it_index[key] = f
    return it_index


def apply_dare(model, base_index, it_index, drop_rate=0.5, seed=42):
    """
    DARE: Drop And REscale. Randomly drop alignment deltas.
    w_merged = w_base + mask * delta / (1-drop_rate)
    """
    rng = np.random.RandomState(seed)
    param_dict = dict(model.named_parameters())
    modified = 0

    for param_name, param in param_dict.items():
        if param_name not in base_index or param_name not in it_index:
            continue
        if "layers." not in param_name:
            continue

        with safe_open(str(base_index[param_name]), framework="pt", device="cpu") as sf:
            base_w = sf.get_tensor(param_name)
        with safe_open(str(it_index[param_name]), framework="pt", device="cpu") as sf:
            it_w = sf.get_tensor(param_name)

        delta = it_w - base_w
        # Random binary mask
        mask = torch.from_numpy(rng.binomial(1, 1 - drop_rate, delta.shape).astype(np.float32))
        # Rescale
        if drop_rate < 1.0:
            scaled_delta = mask * delta / (1 - drop_rate)
        else:
            scaled_delta = torch.zeros_like(delta)

        merged = base_w + scaled_delta
        param.data.copy_(merged.to(param.dtype).to(param.device))
        del base_w, it_w, delta, mask, scaled_delta, merged
        modified += 1

    torch.cuda.empty_cache()
    return modified


def apply_ties(model, base_index, it_index, trim_percent=20):
    """
    TIES-Merging: Trim, resolve sign conflicts, merge.
    trim_percent: percentage of smallest-magnitude deltas to zero out.
    """
    param_dict = dict(model.named_parameters())
    modified = 0

    for param_name, param in param_dict.items():
        if param_name not in base_index or param_name not in it_index:
            continue
        if "layers." not in param_name:
            continue

        with safe_open(str(base_index[param_name]), framework="pt", device="cpu") as sf:
            base_w = sf.get_tensor(param_name)
        with safe_open(str(it_index[param_name]), framework="pt", device="cpu") as sf:
            it_w = sf.get_tensor(param_name)

        delta = it_w - base_w

        # Step 1: Trim — zero out smallest values
        flat = delta.abs().flatten().float()
        # Sample if tensor is too large for quantile
        if flat.numel() > 1_000_000:
            idx = torch.randperm(flat.numel())[:1_000_000]
            threshold = torch.quantile(flat[idx], trim_percent / 100.0)
        else:
            threshold = torch.quantile(flat, trim_percent / 100.0)
        trimmed_delta = delta.clone()
        trimmed_delta[delta.abs() < threshold] = 0

        # Step 2: Sign resolution — for TIES with single task vector, this is identity
        # (sign conflicts only matter with multiple task vectors)

        # Step 3: Merge
        merged = base_w + trimmed_delta
        param.data.copy_(merged.to(param.dtype).to(param.device))
        del base_w, it_w, delta, flat, trimmed_delta, merged
        modified += 1

    torch.cuda.empty_cache()
    return modified


def apply_sar(model, base_index, attribution_path, k_percent=5.0):
    """Apply SAR: roll back top-k% most harmful components."""
    with open(attribution_path) as f:
        attr_data = json.load(f)
    all_components = sorted(attr_data["components"], key=lambda x: x["harm_score"], reverse=True)
    k = max(1, int(len(all_components) * k_percent / 100))
    selected = all_components[:k]

    param_dict = dict(model.named_parameters())
    rolled = 0
    for comp in selected:
        weight_key = comp["name"]
        if weight_key not in base_index or weight_key not in param_dict:
            continue
        with safe_open(str(base_index[weight_key]), framework="pt", device="cpu") as sf:
            base_w = sf.get_tensor(weight_key)
        param = param_dict[weight_key]
        param.data.copy_(base_w.to(param.dtype).to(param.device))
        del base_w
        rolled += 1
    torch.cuda.empty_cache()
    return rolled


def run_comparison(model_key, device="cuda:0"):
    cfg = MODEL_CONFIGS[model_key]
    base_dir = cfg["base"]
    it_dir = cfg["it"]
    attr_path = RESULTS_DIR / cfg["attribution"]

    print(f"\n{'='*70}")
    print(f"DARE/TIES vs SAR COMPARISON: {model_key}")
    print(f"{'='*70}")

    tokenizer = AutoTokenizer.from_pretrained(it_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    examples = AGENT_EXAMPLES_200

    base_index = build_base_index(base_dir)
    it_index = build_it_index(it_dir)

    # Baselines
    print("Loading IT model...")
    model = AutoModelForCausalLM.from_pretrained(
        it_dir, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
    )
    model.eval()
    it_loss = compute_loss(model, tokenizer, examples, device)
    print(f"  IT loss: {it_loss:.4f}")
    del model; torch.cuda.empty_cache()

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_dir, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
    )
    model.eval()
    base_loss = compute_loss(model, tokenizer, examples, device)
    print(f"  Base loss: {base_loss:.4f}")
    alignment_tax = it_loss - base_loss
    print(f"  Alignment tax: {alignment_tax:+.4f}")
    del model; torch.cuda.empty_cache()

    results = {
        "it_loss": float(it_loss),
        "base_loss": float(base_loss),
        "alignment_tax": float(alignment_tax),
    }

    # ================================================================
    # DARE experiments
    # ================================================================
    for drop_rate in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
        name = f"dare_drop{int(drop_rate*100)}"
        print(f"\n--- {name} ---")
        model = AutoModelForCausalLM.from_pretrained(
            it_dir, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
        )
        model.eval()
        n_mod = apply_dare(model, base_index, it_index, drop_rate=drop_rate)
        loss = compute_loss(model, tokenizer, examples, device)
        recovery = it_loss - loss
        pct = recovery / alignment_tax * 100 if alignment_tax != 0 else 0
        results[name] = {
            "loss": float(loss),
            "recovery": float(recovery),
            "pct_tax_recovered": float(pct),
            "n_modified": n_mod,
            "drop_rate": drop_rate,
        }
        print(f"  Loss: {loss:.4f}, Recovery: {recovery:+.4f} ({pct:.1f}%)")
        del model; torch.cuda.empty_cache()

    # ================================================================
    # TIES experiments
    # ================================================================
    for trim_pct in [10, 20, 30, 50, 70, 90]:
        name = f"ties_trim{trim_pct}"
        print(f"\n--- {name} ---")
        model = AutoModelForCausalLM.from_pretrained(
            it_dir, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
        )
        model.eval()
        n_mod = apply_ties(model, base_index, it_index, trim_percent=trim_pct)
        loss = compute_loss(model, tokenizer, examples, device)
        recovery = it_loss - loss
        pct = recovery / alignment_tax * 100 if alignment_tax != 0 else 0
        results[name] = {
            "loss": float(loss),
            "recovery": float(recovery),
            "pct_tax_recovered": float(pct),
            "n_modified": n_mod,
            "trim_percent": trim_pct,
        }
        print(f"  Loss: {loss:.4f}, Recovery: {recovery:+.4f} ({pct:.1f}%)")
        del model; torch.cuda.empty_cache()

    # ================================================================
    # SAR (for comparison)
    # ================================================================
    for k_pct in [3, 5, 8, 10]:
        name = f"sar_{k_pct}pct"
        print(f"\n--- {name} ---")
        model = AutoModelForCausalLM.from_pretrained(
            it_dir, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
        )
        model.eval()
        n_rolled = apply_sar(model, base_index, attr_path, k_percent=k_pct)
        loss = compute_loss(model, tokenizer, examples, device)
        recovery = it_loss - loss
        pct = recovery / alignment_tax * 100 if alignment_tax != 0 else 0
        results[name] = {
            "loss": float(loss),
            "recovery": float(recovery),
            "pct_tax_recovered": float(pct),
            "n_components": n_rolled,
        }
        print(f"  Loss: {loss:.4f}, Recovery: {recovery:+.4f} ({pct:.1f}%)")
        del model; torch.cuda.empty_cache()

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'='*70}")
    print(f"SUMMARY: {model_key}")
    print(f"{'='*70}")
    print(f"IT loss: {it_loss:.4f}, Base loss: {base_loss:.4f}, Tax: {alignment_tax:+.4f}\n")
    print(f"{'Method':<25} {'Loss':>8} {'Recovery':>10} {'Tax Recov%':>10}")
    print("-" * 57)

    # Sort by recovery
    method_results = [(n, r) for n, r in results.items() if isinstance(r, dict) and "loss" in r]
    method_results.sort(key=lambda x: x[1]["pct_tax_recovered"], reverse=True)

    for name, r in method_results:
        marker = " ★" if name.startswith("sar_") else ""
        print(f"  {name:<23} {r['loss']:>8.4f} {r['recovery']:>+10.4f} {r['pct_tax_recovered']:>9.1f}%{marker}")

    # Save
    output = {
        "analysis": "DARE/TIES vs SAR comparison",
        "model": model_key,
        "results": results,
    }
    out_path = RESULTS_DIR / f"step41_dare_ties_{model_key}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    model_key = sys.argv[1]
    device = sys.argv[2] if len(sys.argv) > 2 else "cuda:0"
    run_comparison(model_key, device)
