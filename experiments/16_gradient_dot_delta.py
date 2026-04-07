"""
Gradient-dot-delta attribution — patching-free alternative.

For each component (t, l), computes:
  attribution(t,l) = <∇_{W_t^l} L_agent, ΔW_t^l>

Uses layer-by-layer gradient computation to fit in GPU memory.

Usage: python gradient_dot_delta.py <model_name> <device>
"""

import sys, json, torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from safetensors import safe_open

RESULTS_DIR = Path("./results")

MODEL_PAIRS = {
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
    "yi-1.5-9b": {
        "base": "./models/Yi-1.5-9B",
        "it": "./models/Yi-1.5-9B-Chat",
    },
}

sys.path.insert(0, "./experiments")
import importlib
_attribution = importlib.import_module("01_attribution")
AGENT_EXAMPLES = _attribution.AGENT_EXAMPLES


def classify_component(name):
    if "layers." not in name:
        return "other", -1
    parts = name.split(".")
    layer_idx = None
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                layer_idx = int(parts[i + 1])
            except:
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


def run_analysis(model_name, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = MODEL_PAIRS[model_name]
    base_dir = config["base"]
    it_dir = config["it"]

    print(f"\n{'='*60}")
    print(f"Gradient-dot-delta attribution: {model_name}")
    print(f"{'='*60}")

    # Load IT model with gradient checkpointing
    print(f"Loading IT model from {it_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(it_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        it_dir, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
    )
    model.config.use_cache = False

    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()

    # Index base weights
    print(f"Indexing base weights from {base_dir}...")
    base_index = {}
    for f in sorted(Path(base_dir).glob("*.safetensors")):
        with safe_open(str(f), framework="pt", device="cpu") as sf:
            for key in sf.keys():
                base_index[key] = f

    examples = AGENT_EXAMPLES  # 49 base examples

    num_layers = model.config.num_hidden_layers
    proj_map = {
        "W_Q": "self_attn.q_proj", "W_K": "self_attn.k_proj",
        "W_V": "self_attn.v_proj", "W_O": "self_attn.o_proj",
        "W_gate": "mlp.gate_proj", "W_up": "mlp.up_proj",
        "W_down": "mlp.down_proj",
    }

    # Compute gradients layer-by-layer to save memory:
    # Freeze all params, enable only target layer, accumulate gradient over examples,
    # compute dot product with delta, then move to next layer.
    print(f"\nComputing layer-by-layer gradients over {len(examples)} examples...")

    components = []
    type_scores = defaultdict(list)
    total_components = num_layers * 7
    done = 0

    for layer_idx in range(num_layers):
        # Freeze all parameters
        for p in model.parameters():
            p.requires_grad_(False)

        # Enable gradients only for this layer's projections
        layer_params = {}
        for comp_type, proj_path in proj_map.items():
            weight_key = f"model.layers.{layer_idx}.{proj_path}.weight"
            if weight_key not in base_index:
                continue

            layer = model.model.layers[layer_idx]
            parts = proj_path.split(".")
            module = layer
            for p in parts:
                module = getattr(module, p)

            module.weight.requires_grad_(True)
            layer_params[comp_type] = (module, weight_key)

        if not layer_params:
            continue

        # Accumulate gradients over examples
        model.zero_grad()
        total_tokens = 0

        for ex in examples:
            full_text = ex["prompt"] + ex["target"]
            inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512)
            input_ids = inputs["input_ids"].to(device)
            prompt_len = tokenizer(ex["prompt"], return_tensors="pt")["input_ids"].shape[1]

            if prompt_len >= input_ids.shape[1]:
                continue

            try:
                outputs = model(input_ids=input_ids)
                logits = outputs.logits
                shift_logits = logits[0, prompt_len - 1:-1, :]
                shift_labels = input_ids[0, prompt_len:]
                loss = torch.nn.functional.cross_entropy(shift_logits, shift_labels, reduction='sum')
                loss.backward()
                total_tokens += shift_labels.shape[0]
            except RuntimeError:
                model.zero_grad()
                torch.cuda.empty_cache()
                continue

        # Compute dot products for this layer
        for comp_type, (module, weight_key) in layer_params.items():
            if module.weight.grad is None:
                components.append({
                    "name": weight_key, "component_type": comp_type,
                    "layer": layer_idx, "grad_dot_delta": 0.0,
                    "grad_norm": 0.0, "delta_norm": 0.0,
                })
                type_scores[comp_type].append(0.0)
                done += 1
                continue

            grad = module.weight.grad.float() / max(total_tokens, 1)

            with safe_open(str(base_index[weight_key]), framework="pt", device="cpu") as sf:
                base_w = sf.get_tensor(weight_key)
            it_w = module.weight.data.float().cpu()
            delta = it_w - base_w.float()

            dot_product = torch.sum(grad.cpu() * delta).item()
            grad_norm = torch.norm(grad).item()
            delta_norm = torch.norm(delta).item()

            components.append({
                "name": weight_key, "component_type": comp_type,
                "layer": layer_idx, "grad_dot_delta": dot_product,
                "grad_norm": grad_norm, "delta_norm": delta_norm,
            })
            type_scores[comp_type].append(abs(dot_product))

            module.weight.requires_grad_(False)
            del grad, base_w, it_w, delta

            done += 1

        model.zero_grad()
        torch.cuda.empty_cache()

        if (layer_idx + 1) % 4 == 0:
            print(f"  Layer {layer_idx+1}/{num_layers} done ({done}/{total_components} components)")

    # Compute hierarchy
    total_abs = sum(sum(scores) for scores in type_scores.values())
    print(f"\nGradient-dot-delta hierarchy (total = {total_abs:.6f}):")

    hierarchy = {}
    for comp in ["W_Q", "W_K", "W_V", "W_O", "W_gate", "W_up", "W_down"]:
        scores = type_scores.get(comp, [])
        if not scores:
            continue
        comp_sum = sum(scores)
        share = comp_sum / total_abs * 100 if total_abs > 0 else 0
        hierarchy[comp] = {
            "sum": comp_sum, "mean": float(np.mean(scores)),
            "share_pct": share, "n": len(scores),
        }
        print(f"  {comp:8s}: share={share:5.1f}%, mean={np.mean(scores):.6f} (n={len(scores)})")

    # V/O vs Q/K comparison
    vo_scores = type_scores.get("W_V", []) + type_scores.get("W_O", [])
    qk_scores = type_scores.get("W_Q", []) + type_scores.get("W_K", [])

    from scipy.stats import mannwhitneyu
    if vo_scores and qk_scores:
        stat, pval = mannwhitneyu(vo_scores, qk_scores, alternative='greater')
        ratio = np.mean(vo_scores) / np.mean(qk_scores) if np.mean(qk_scores) > 0 else float('inf')
        n1, n2 = len(vo_scores), len(qk_scores)
        r = 1 - (2 * stat) / (n1 * n2)
    else:
        pval, ratio, r = 1.0, 0.0, 0.0

    vo_share = hierarchy.get("W_V", {}).get("share_pct", 0) + hierarchy.get("W_O", {}).get("share_pct", 0)
    qk_share = hierarchy.get("W_Q", {}).get("share_pct", 0) + hierarchy.get("W_K", {}).get("share_pct", 0)
    mlp_share = sum(hierarchy.get(c, {}).get("share_pct", 0) for c in ["W_gate", "W_up", "W_down"])
    output_pathway = vo_share + hierarchy.get("W_down", {}).get("share_pct", 0)

    print(f"\n  MLP: {mlp_share:.1f}%  V/O: {vo_share:.1f}%  Q/K: {qk_share:.1f}%")
    print(f"  Output pathway: {output_pathway:.1f}%")
    print(f"  V/O:Q/K ratio: {ratio:.2f}x (p={pval:.2e}, r={r:.3f})")

    results = {
        "analysis": "Gradient-dot-delta attribution (patching-free)",
        "model": model_name,
        "num_examples": len(examples),
        "hierarchy": hierarchy,
        "groups": {
            "MLP_share": mlp_share, "VO_share": vo_share,
            "QK_share": qk_share, "output_pathway_share": output_pathway,
            "VO_QK_ratio": ratio, "VO_QK_pvalue": pval,
            "VO_QK_effect_size_r": r,
        },
        "components": components,
    }

    out_path = RESULTS_DIR / f"gradient_dot_delta_{model_name}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    del model
    torch.cuda.empty_cache()
    return results


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5-7b"
    device = sys.argv[2] if len(sys.argv) > 2 else "cuda:0"
    run_analysis(model_name, device)
