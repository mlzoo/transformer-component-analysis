"""
Weight-patching attribution: measures per-component alignment harm scores.
Produces Tables 1-2 in the paper.
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).parent))

RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(exist_ok=True)

from agent_examples_200 import BASE_EXAMPLES, EXTRA_EXAMPLES
AGENT_EXAMPLES = BASE_EXAMPLES + EXTRA_EXAMPLES


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


def run_attribution(base_dir, it_dir, device="cuda:0", model_name="unknown"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading IT model in float16 to {device}...")
    tokenizer = AutoTokenizer.from_pretrained(it_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        it_dir, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
    )
    model.eval()

    base_index = {}
    for f in sorted(Path(base_dir).glob("*.safetensors")):
        with safe_open(str(f), framework="pt", device="cpu") as sf:
            for key in sf.keys():
                base_index[key] = f

    it_index = {}
    for f in sorted(Path(it_dir).glob("*.safetensors")):
        with safe_open(str(f), framework="pt", device="cpu") as sf:
            for key in sf.keys():
                it_index[key] = f

    examples = AGENT_EXAMPLES

    print(f"Measuring baseline IT loss on {len(examples)} examples...")
    baseline_loss = compute_loss(model, tokenizer, examples, device)
    print(f"  Baseline IT loss: {baseline_loss:.4f}")

    results = []
    num_layers = model.config.num_hidden_layers

    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        components = [
            ("W_Q", layer.self_attn.q_proj, f"model.layers.{layer_idx}.self_attn.q_proj.weight"),
            ("W_K", layer.self_attn.k_proj, f"model.layers.{layer_idx}.self_attn.k_proj.weight"),
            ("W_V", layer.self_attn.v_proj, f"model.layers.{layer_idx}.self_attn.v_proj.weight"),
            ("W_O", layer.self_attn.o_proj, f"model.layers.{layer_idx}.self_attn.o_proj.weight"),
            ("W_gate", layer.mlp.gate_proj, f"model.layers.{layer_idx}.mlp.gate_proj.weight"),
            ("W_up", layer.mlp.up_proj, f"model.layers.{layer_idx}.mlp.up_proj.weight"),
            ("W_down", layer.mlp.down_proj, f"model.layers.{layer_idx}.mlp.down_proj.weight"),
        ]

        for comp_name, proj_module, weight_key in components:
            if weight_key not in base_index or weight_key not in it_index:
                continue

            with safe_open(str(base_index[weight_key]), framework="pt", device="cpu") as sf:
                base_w = sf.get_tensor(weight_key)
            with safe_open(str(it_index[weight_key]), framework="pt", device="cpu") as sf:
                it_w = sf.get_tensor(weight_key)

            delta_cpu = (it_w - base_w).float()
            delta_norm = torch.norm(delta_cpu).item()
            del base_w, it_w

            def make_hook(delta):
                def hook_fn(module, input, output):
                    x = input[0] if isinstance(input, tuple) else input
                    x_cpu = x.float().cpu()
                    correction = torch.nn.functional.linear(x_cpu, delta)
                    return output - correction.half().to(output.device)
                return hook_fn

            hook = proj_module.register_forward_hook(make_hook(delta_cpu))
            rollback_loss = compute_loss(model, tokenizer, examples, device)
            hook.remove()

            harm_score = baseline_loss - rollback_loss
            results.append({
                "name": weight_key, "component_type": comp_name, "layer": layer_idx,
                "baseline_loss": float(baseline_loss), "rollback_loss": float(rollback_loss),
                "harm_score": float(harm_score), "delta_norm": float(delta_norm),
            })

            marker = "***" if abs(harm_score) > 0.01 else ""
            print(f"  {comp_name:8s} L{layer_idx:2d}: harm={harm_score:+.4f} {marker}")
            del delta_cpu
            torch.cuda.empty_cache()

    # Analysis
    by_type = defaultdict(list)
    for r in results:
        by_type[r["component_type"]].append(r)

    print(f"\n{'='*80}")
    print(f"ATTRIBUTION RESULTS ({model_name}) — {len(examples)} examples")
    print(f"Baseline loss: {baseline_loss:.4f}")
    print(f"{'='*80}")

    type_stats = {}
    for comp in ["W_Q", "W_K", "W_V", "W_O", "W_gate", "W_up", "W_down"]:
        entries = by_type.get(comp, [])
        if not entries: continue
        harms = [e["harm_score"] for e in entries]
        pos_count = sum(1 for h in harms if h > 0)
        type_stats[comp] = {
            "count": len(entries), "mean_harm": float(np.mean(harms)),
            "sum_harm": float(np.sum(harms)),
            "abs_sum_harm": float(np.sum(np.abs(harms))),
            "positive_fraction": pos_count / len(entries),
        }
        print(f"  {comp:8s}: mean={np.mean(harms):+.4f}, sum={np.sum(harms):+.4f}, pos={pos_count}/{len(entries)}")

    vo = [e["harm_score"] for e in by_type.get("W_V", []) + by_type.get("W_O", [])]
    qk = [e["harm_score"] for e in by_type.get("W_Q", []) + by_type.get("W_K", [])]
    mlp = [e["harm_score"] for e in by_type.get("W_gate", []) + by_type.get("W_up", []) + by_type.get("W_down", [])]
    total_abs = sum(abs(r["harm_score"]) for r in results)
    vo_abs = sum(abs(h) for h in vo)
    qk_abs = sum(abs(h) for h in qk)
    mlp_abs = sum(abs(h) for h in mlp)

    for comp in type_stats:
        type_stats[comp]["pct_of_total_abs"] = type_stats[comp]["abs_sum_harm"] / total_abs * 100

    print(f"\n  Total |harm|: {total_abs:.4f}")
    print(f"  MLP: {mlp_abs:.4f} ({mlp_abs/total_abs*100:.1f}%)")
    print(f"  V/O: {vo_abs:.4f} ({vo_abs/total_abs*100:.1f}%)")
    print(f"  Q/K: {qk_abs:.4f} ({qk_abs/total_abs*100:.1f}%)")

    from scipy.stats import mannwhitneyu
    stat, pval = mannwhitneyu(vo, qk, alternative='greater')
    n1, n2 = len(vo), len(qk)
    r_biserial = 2 * stat / (n1 * n2) - 1
    print(f"  V/O > Q/K: U={stat:.0f}, p={pval:.6f}, r={r_biserial:.3f}")

    # Block permutation test (preserves within-layer pairing)
    def block_permutation_test(components, n_perm=100000):
        """Test V/O > Q/K with block permutation preserving layer structure.

        For each layer, collects the 4 attention scores (W_Q, W_K, W_V, W_O),
        computes observed = mean(V/O scores) - mean(Q/K scores), then permutes
        the 4 labels within each layer independently across n_perm iterations.
        """
        rng = np.random.RandomState(42)
        attn_types = {"W_Q", "W_K", "W_V", "W_O"}

        # Group all attention components by layer
        layers = {}
        for c in components:
            if c["component_type"] in attn_types:
                layer = c["layer"]
                if layer not in layers:
                    layers[layer] = []
                layers[layer].append(c)

        # Observed statistic: mean(|V/O harm|) - mean(|Q/K harm|)
        vo_obs = [abs(c["harm_score"]) for c in components if c["component_type"] in ("W_V", "W_O")]
        qk_obs = [abs(c["harm_score"]) for c in components if c["component_type"] in ("W_Q", "W_K")]
        observed = np.mean(vo_obs) - np.mean(qk_obs)

        count = 0
        for _ in range(n_perm):
            perm_vo, perm_qk = [], []
            for layer_idx in sorted(layers.keys()):
                layer_comps = layers[layer_idx]
                if len(layer_comps) != 4:
                    continue
                scores = [abs(c["harm_score"]) for c in layer_comps]
                rng.shuffle(scores)
                perm_vo.extend(scores[:2])
                perm_qk.extend(scores[2:])
            if perm_vo and perm_qk:
                perm_diff = np.mean(perm_vo) - np.mean(perm_qk)
                if perm_diff >= observed:
                    count += 1
        return count / n_perm

    p_block = block_permutation_test(results)
    print(f"  Block permutation p = {p_block:.6f}")

    output = {
        "analysis": f"Attribution via weight patching (float16, {len(examples)} examples)",
        "model": model_name, "num_examples": len(examples),
        "baseline_loss": float(baseline_loss),
        "type_statistics": type_stats, "components": results,
        "vo_qk_test": {"U": float(stat), "p_mw": float(pval),
                       "r_biserial": float(r_biserial), "p_block": float(p_block)},
    }
    safe_name = model_name.replace("/", "_").replace(" ", "_").lower()
    out_path = RESULTS_DIR / f"attribution_{safe_name}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    base_dir = sys.argv[1]
    it_dir = sys.argv[2]
    device = sys.argv[3] if len(sys.argv) > 3 else "cuda:0"
    model_name = sys.argv[4] if len(sys.argv) > 4 else "unknown"

    run_attribution(base_dir, it_dir, device=device, model_name=model_name)
