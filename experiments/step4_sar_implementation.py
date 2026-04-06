"""
Step 4: SAR (Surgical Alignment Reversal) Implementation

Given attribution results, create a SAR model by selectively rolling back
the top-k% most harmful components.

Then evaluate on expanded agent examples to measure improvement.
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from safetensors import safe_open
from safetensors.torch import save_file

RESULTS_DIR = Path("./results")


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


def apply_sar(model, base_dir, it_dir, attribution_path, k_percent=5.0, strategy="topk"):
    """
    Apply SAR by modifying model weights in-place.

    Strategies:
    - topk: Roll back top k% most harmful components
    - heuristic: Roll back W_down + W_V + W_O at mid layers
    - vo_only: Roll back W_V + W_O only
    - mlp_only: Roll back W_down only
    - random: Roll back random k%
    """
    # Load attribution
    with open(attribution_path) as f:
        attr_data = json.load(f)

    all_components = sorted(attr_data["components"], key=lambda x: x["harm_score"], reverse=True)
    num_layers = model.config.num_hidden_layers
    mid_start = num_layers // 3
    mid_end = 2 * num_layers // 3

    # Select components to roll back based on strategy
    if strategy == "topk":
        k = max(1, int(len(all_components) * k_percent / 100))
        selected = all_components[:k]
    elif strategy == "heuristic":
        selected = [c for c in all_components
                    if c["component_type"] in ("W_V", "W_O", "W_down")
                    and mid_start <= c["layer"] < mid_end]
    elif strategy == "vo_only":
        selected = [c for c in all_components
                    if c["component_type"] in ("W_V", "W_O")
                    and mid_start <= c["layer"] < mid_end]
    elif strategy == "mlp_only":
        selected = [c for c in all_components
                    if c["component_type"] == "W_down"
                    and mid_start <= c["layer"] < mid_end]
    elif strategy == "qk_only":
        selected = [c for c in all_components
                    if c["component_type"] in ("W_Q", "W_K")
                    and mid_start <= c["layer"] < mid_end]
    elif strategy == "random":
        k = max(1, int(len(all_components) * k_percent / 100))
        indices = np.random.choice(len(all_components), k, replace=False)
        selected = [all_components[i] for i in indices]
    elif strategy == "magnitude":
        # Sort by delta norm instead of harm score
        by_magnitude = sorted(all_components, key=lambda x: x.get("delta_norm", 0), reverse=True)
        k = max(1, int(len(all_components) * k_percent / 100))
        selected = by_magnitude[:k]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    print(f"  SAR strategy={strategy}: rolling back {len(selected)} components")

    # Build base weight index
    base_index = {}
    for f in sorted(Path(base_dir).glob("*.safetensors")):
        with safe_open(str(f), framework="pt", device="cpu") as sf:
            for key in sf.keys():
                base_index[key] = f

    # Apply rollback
    param_dict = dict(model.named_parameters())
    rolled_back = 0
    for comp in selected:
        weight_key = comp["name"]
        if weight_key not in base_index:
            continue
        if weight_key not in param_dict:
            continue

        with safe_open(str(base_index[weight_key]), framework="pt", device="cpu") as sf:
            base_w = sf.get_tensor(weight_key)

        param = param_dict[weight_key]
        param.data.copy_(base_w.to(param.dtype).to(param.device))
        rolled_back += 1
        del base_w

    print(f"  Rolled back {rolled_back} weight matrices")
    torch.cuda.empty_cache()
    return rolled_back, selected


# Import agent examples from expanded set
from step1_attribution import AGENT_EXAMPLES


def run_sar_eval(base_dir, it_dir, device="cuda:0", model_name="unknown"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Find attribution results
    safe_name = model_name.replace("/", "_").replace(" ", "_").lower()
    # Try expanded first, then fp16
    attr_path = RESULTS_DIR / f"step1_attribution_expanded_{safe_name}.json"
    if not attr_path.exists():
        attr_path = RESULTS_DIR / f"step1_attribution_fp16_{safe_name}.json"
    if not attr_path.exists():
        print(f"ERROR: No attribution results for {model_name}")
        return

    print(f"Using attribution: {attr_path}")

    tokenizer = AutoTokenizer.from_pretrained(it_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    examples = AGENT_EXAMPLES[:50]

    strategies = ["topk", "heuristic", "vo_only", "mlp_only", "qk_only", "random", "magnitude"]
    k_values = [3, 5, 8, 10]

    results = {}

    # Baseline: IT model
    print(f"\nLoading IT model to {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        it_dir, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
    )
    model.eval()
    it_loss = compute_loss(model, tokenizer, examples, device)
    print(f"IT loss: {it_loss:.4f}")
    results["IT_baseline"] = float(it_loss)
    del model; torch.cuda.empty_cache()

    # Base model
    print(f"\nLoading base model to {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_dir, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
    )
    model.eval()
    base_loss = compute_loss(model, tokenizer, examples, device)
    print(f"Base loss: {base_loss:.4f}")
    results["base_model"] = float(base_loss)
    del model; torch.cuda.empty_cache()

    alignment_tax = it_loss - base_loss
    print(f"Alignment tax: {alignment_tax:+.4f}")
    results["alignment_tax"] = float(alignment_tax)

    # SAR with different strategies
    for strategy in strategies:
        if strategy in ("topk", "random", "magnitude"):
            for k in k_values:
                config_name = f"{strategy}_k{k}"
                print(f"\n--- {config_name} ---")

                model = AutoModelForCausalLM.from_pretrained(
                    it_dir, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
                )
                model.eval()

                n_rolled, selected = apply_sar(model, base_dir, it_dir, attr_path,
                                               k_percent=k, strategy=strategy)

                sar_loss = compute_loss(model, tokenizer, examples, device)
                recovery = it_loss - sar_loss
                pct = recovery / alignment_tax * 100 if alignment_tax != 0 else 0

                # Composition of rolled-back components
                type_counts = defaultdict(int)
                for c in selected:
                    type_counts[c["component_type"]] += 1

                results[config_name] = {
                    "loss": float(sar_loss),
                    "recovery": float(recovery),
                    "pct_tax_recovered": float(pct),
                    "n_components": n_rolled,
                    "type_composition": dict(type_counts),
                }
                print(f"  Loss: {sar_loss:.4f}, Recovery: {recovery:+.4f} ({pct:.1f}%)")

                del model; torch.cuda.empty_cache()
        else:
            config_name = strategy
            print(f"\n--- {config_name} ---")

            model = AutoModelForCausalLM.from_pretrained(
                it_dir, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
            )
            model.eval()

            n_rolled, selected = apply_sar(model, base_dir, it_dir, attr_path,
                                           strategy=strategy)

            sar_loss = compute_loss(model, tokenizer, examples, device)
            recovery = it_loss - sar_loss
            pct = recovery / alignment_tax * 100 if alignment_tax != 0 else 0

            type_counts = defaultdict(int)
            for c in selected:
                type_counts[c["component_type"]] += 1

            results[config_name] = {
                "loss": float(sar_loss),
                "recovery": float(recovery),
                "pct_tax_recovered": float(pct),
                "n_components": n_rolled,
                "type_composition": dict(type_counts),
            }
            print(f"  Loss: {sar_loss:.4f}, Recovery: {recovery:+.4f} ({pct:.1f}%)")

            del model; torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*80}")
    print(f"SAR EVALUATION SUMMARY: {model_name}")
    print(f"Base loss: {base_loss:.4f}, IT loss: {it_loss:.4f}, Tax: {alignment_tax:+.4f}")
    print(f"{'='*80}")

    for name, r in results.items():
        if isinstance(r, dict):
            print(f"  {name:25s}: loss={r['loss']:.4f}, recovery={r['recovery']:+.4f} ({r['pct_tax_recovered']:.1f}%)")

    # Save
    output = {
        "analysis": "SAR evaluation",
        "model_pair": model_name,
        "base_loss": float(base_loss),
        "it_loss": float(it_loss),
        "alignment_tax": float(alignment_tax),
        "results": results,
    }
    out_path = RESULTS_DIR / f"step4_sar_eval_{safe_name}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    base_dir = sys.argv[1]
    it_dir = sys.argv[2]
    device = sys.argv[3] if len(sys.argv) > 3 else "cuda:0"
    model_name = sys.argv[4] if len(sys.argv) > 4 else "unknown"

    run_sar_eval(base_dir, it_dir, device=device, model_name=model_name)
