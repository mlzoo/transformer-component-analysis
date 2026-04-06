"""
Pairwise interaction analysis for activation patching.

Tests whether one-at-a-time activation patching misses important non-linear
interactions between components. For top-K harmful components, computes:
  interaction(i,j) = joint_effect(i,j) - effect(i) - effect(j)

If interactions are small relative to marginal effects, the one-at-a-time
attribution is a good approximation.

Usage: python 14_pairwise_interaction.py <model_name> <device>
"""

import sys, json, torch
import numpy as np
from pathlib import Path
from itertools import combinations
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
}

TOP_K = 10  # Analyze top 10 harmful components

sys.path.insert(0, "./experiments")
import importlib
_attribution = importlib.import_module("01_attribution")
AGENT_EXAMPLES = _attribution.AGENT_EXAMPLES
compute_loss = _attribution.compute_loss


def load_attribution(model_name):
    """Load pre-computed attribution results."""
    path = RESULTS_DIR / f"attribution_{model_name}.json"
    with open(path) as f:
        data = json.load(f)
    return data["components"]


def run_analysis(model_name, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = MODEL_PAIRS[model_name]
    base_dir = config["base"]
    it_dir = config["it"]

    print(f"\n{'='*60}")
    print(f"Pairwise interaction analysis: {model_name}")
    print(f"{'='*60}")

    # Load pre-computed attributions to find top-K harmful
    components = load_attribution(model_name)
    components_sorted = sorted(components, key=lambda c: c["harm_score"], reverse=True)
    top_components = components_sorted[:TOP_K]

    print(f"\nTop {TOP_K} harmful components:")
    for c in top_components:
        print(f"  {c['component_type']:8s} L{c['layer']:2d}: harm={c['harm_score']:+.4f}")

    # Load model and base weights
    print(f"\nLoading IT model...")
    tokenizer = AutoTokenizer.from_pretrained(it_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        it_dir, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
    )
    model.eval()

    # Index base weights
    base_index = {}
    for f in sorted(Path(base_dir).glob("*.safetensors")):
        with safe_open(str(f), framework="pt", device="cpu") as sf:
            for key in sf.keys():
                base_index[key] = f

    it_index = {}
    it_path = Path(it_dir) if Path(it_dir).exists() else None
    if it_path:
        for f in sorted(it_path.glob("*.safetensors")):
            with safe_open(str(f), framework="pt", device="cpu") as sf:
                for key in sf.keys():
                    it_index[key] = f
    else:
        # For HF Hub models, get from model state dict
        from huggingface_hub import snapshot_download
        local_path = snapshot_download(it_dir)
        for f in sorted(Path(local_path).glob("*.safetensors")):
            with safe_open(str(f), framework="pt", device="cpu") as sf:
                for key in sf.keys():
                    it_index[key] = f

    examples = AGENT_EXAMPLES[:30]  # Use 30 examples for speed

    # Compute baseline loss
    print(f"Computing baseline loss on {len(examples)} examples...")
    baseline_loss = compute_loss(model, tokenizer, examples, device)
    print(f"  Baseline: {baseline_loss:.4f}")

    # Precompute deltas for top components
    def get_comp_info(comp):
        layer_idx = comp["layer"]
        comp_type = comp["component_type"]
        proj_map = {
            "W_Q": "self_attn.q_proj", "W_K": "self_attn.k_proj",
            "W_V": "self_attn.v_proj", "W_O": "self_attn.o_proj",
            "W_gate": "mlp.gate_proj", "W_up": "mlp.up_proj",
            "W_down": "mlp.down_proj",
        }
        proj_path = proj_map[comp_type]
        weight_key = f"model.layers.{layer_idx}.{proj_path}.weight"
        return layer_idx, proj_path, weight_key

    def get_module(layer_idx, proj_path):
        layer = model.model.layers[layer_idx]
        parts = proj_path.split(".")
        obj = layer
        for p in parts:
            obj = getattr(obj, p)
        return obj

    # Compute deltas
    deltas = {}
    for i, comp in enumerate(top_components):
        layer_idx, proj_path, weight_key = get_comp_info(comp)
        with safe_open(str(base_index[weight_key]), framework="pt", device="cpu") as sf:
            base_w = sf.get_tensor(weight_key)
        with safe_open(str(it_index[weight_key]), framework="pt", device="cpu") as sf:
            it_w = sf.get_tensor(weight_key)
        deltas[i] = (it_w - base_w).float()
        del base_w, it_w

    def make_hook(delta):
        def hook_fn(module, input, output):
            x = input[0] if isinstance(input, tuple) else input
            x_cpu = x.float().cpu()
            correction = torch.nn.functional.linear(x_cpu, delta)
            return output - correction.half().to(output.device)
        return hook_fn

    # Step 1: Single-component effects (verify they match pre-computed)
    print(f"\nComputing single-component effects...")
    single_effects = {}
    for i, comp in enumerate(top_components):
        layer_idx, proj_path, _ = get_comp_info(comp)
        module = get_module(layer_idx, proj_path)
        hook = module.register_forward_hook(make_hook(deltas[i]))
        loss = compute_loss(model, tokenizer, examples, device)
        hook.remove()
        effect = baseline_loss - loss
        single_effects[i] = effect
        print(f"  [{i}] {comp['component_type']:8s} L{comp['layer']:2d}: effect={effect:+.4f}")

    # Step 2: Pairwise effects
    pairs = list(combinations(range(TOP_K), 2))
    print(f"\nComputing {len(pairs)} pairwise effects...")

    interactions = []
    for pi, (i, j) in enumerate(pairs):
        comp_i = top_components[i]
        comp_j = top_components[j]

        li, pp_i, _ = get_comp_info(comp_i)
        lj, pp_j, _ = get_comp_info(comp_j)

        mod_i = get_module(li, pp_i)
        mod_j = get_module(lj, pp_j)

        hook_i = mod_i.register_forward_hook(make_hook(deltas[i]))
        hook_j = mod_j.register_forward_hook(make_hook(deltas[j]))
        joint_loss = compute_loss(model, tokenizer, examples, device)
        hook_i.remove()
        hook_j.remove()

        joint_effect = baseline_loss - joint_loss
        interaction = joint_effect - single_effects[i] - single_effects[j]
        additivity = abs(interaction) / max(abs(joint_effect), 1e-8)

        interactions.append({
            "i": i, "j": j,
            "comp_i": f"{comp_i['component_type']}_L{comp_i['layer']}",
            "comp_j": f"{comp_j['component_type']}_L{comp_j['layer']}",
            "effect_i": single_effects[i],
            "effect_j": single_effects[j],
            "joint_effect": joint_effect,
            "interaction": interaction,
            "additivity_ratio": additivity,
        })

        if (pi + 1) % 10 == 0:
            print(f"  {pi+1}/{len(pairs)} pairs done")

    # Summary statistics
    abs_interactions = [abs(x["interaction"]) for x in interactions]
    abs_marginals = [abs(x["effect_i"]) + abs(x["effect_j"]) for x in interactions]
    ratios = [x["additivity_ratio"] for x in interactions]

    mean_interaction = np.mean(abs_interactions)
    mean_marginal = np.mean(abs_marginals)
    mean_ratio = np.mean(ratios)
    median_ratio = np.median(ratios)
    pct_small = sum(1 for r in ratios if r < 0.1) / len(ratios) * 100

    print(f"\n{'='*60}")
    print(f"INTERACTION ANALYSIS RESULTS ({model_name})")
    print(f"{'='*60}")
    print(f"  Mean |interaction|: {mean_interaction:.4f}")
    print(f"  Mean |marginal sum|: {mean_marginal:.4f}")
    print(f"  Mean interaction/joint ratio: {mean_ratio:.3f}")
    print(f"  Median interaction/joint ratio: {median_ratio:.3f}")
    print(f"  Pairs with <10% interaction: {pct_small:.0f}%")

    # Top interactions
    interactions_sorted = sorted(interactions, key=lambda x: abs(x["interaction"]), reverse=True)
    print(f"\nTop 5 interactions:")
    for x in interactions_sorted[:5]:
        print(f"  {x['comp_i']} x {x['comp_j']}: "
              f"interaction={x['interaction']:+.4f} (ratio={x['additivity_ratio']:.3f})")

    results = {
        "analysis": "Pairwise interaction analysis",
        "model": model_name,
        "top_k": TOP_K,
        "num_examples": len(examples),
        "num_pairs": len(pairs),
        "baseline_loss": baseline_loss,
        "summary": {
            "mean_abs_interaction": mean_interaction,
            "mean_abs_marginal_sum": mean_marginal,
            "mean_additivity_ratio": mean_ratio,
            "median_additivity_ratio": median_ratio,
            "pct_pairs_under_10pct": pct_small,
        },
        "single_effects": {i: single_effects[i] for i in range(TOP_K)},
        "top_components": [
            {"component": c["component_type"], "layer": c["layer"], "harm_score": c["harm_score"]}
            for c in top_components
        ],
        "interactions": interactions,
    }

    out_path = RESULTS_DIR / f"pairwise_interaction_{model_name}.json"
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
