"""
Step 3: RLHF Gradient Magnitude Analysis

Measure per-component DPO gradient norms efficiently:
- Freeze all layers except one at a time
- Compute gradients for that layer's 7 components
- Record norms, then move to next layer

This way gradient memory is bounded (~0.5GB per layer) instead of ~14GB for all.
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path("./results")

PREFERENCE_DATA = [
    {"prompt": "You have access to search(query). Find the capital of France.\nThought: I need to search.\nAction: ", "chosen": 'search(query="capital of France")', "rejected": "The capital of France is Paris."},
    {"prompt": 'Respond in JSON: What is 2+2?\n\n{"', "chosen": '"answer": 4}', "rejected": "The answer is 4."},
    {"prompt": "Generate SQL: Get users older than 25\n\nSELECT ", "chosen": "* FROM users WHERE age > 25;", "rejected": "I would query the users table for age greater than 25."},
    {"prompt": 'Output JSON for: Name=Bob, Age=25\n\n{"', "chosen": '"name": "Bob", "age": 25}', "rejected": "Bob is 25 years old."},
    {"prompt": "API: Delete user 7\n\n", "chosen": "DELETE /api/users/7", "rejected": "To delete user 7, you would make an API call."},
    {"prompt": "```python\ndef add(a, b):\n    ", "chosen": "return a + b", "rejected": "This function adds two numbers together."},
    {"prompt": "Tools: calculator(expr)\nUser: What is 15*23?\nThought: Use calculator.\nAction: ", "chosen": 'calculator(expr="15*23")', "rejected": "15 times 23 equals 345."},
    {"prompt": "YAML:\nserver:\n  port: ", "chosen": "8080\n  host: 0.0.0.0", "rejected": "The server port should be configured to 8080."},
    {"prompt": 'Parse: "Meeting at 3pm in Room 204"\n\n{"', "chosen": '"event": "Meeting", "time": "3pm", "room": "204"}', "rejected": "There is a meeting at 3pm in Room 204."},
    {"prompt": "Cron: Every day at midnight\n\n", "chosen": "0 0 * * *", "rejected": "You would set a cron job to run at midnight every day."},
]


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


def dpo_loss_forward(model, tokenizer, prompt, chosen, rejected, device, beta=0.1):
    chosen_text = prompt + chosen
    rejected_text = prompt + rejected
    chosen_ids = tokenizer(chosen_text, return_tensors="pt", truncation=True, max_length=256)["input_ids"].to(device)
    rejected_ids = tokenizer(rejected_text, return_tensors="pt", truncation=True, max_length=256)["input_ids"].to(device)
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    prompt_len = prompt_ids.shape[1]

    chosen_logits = model(input_ids=chosen_ids).logits
    chosen_logps = torch.nn.functional.log_softmax(chosen_logits[0, prompt_len-1:-1, :].float(), dim=-1)
    chosen_token_logps = chosen_logps.gather(1, chosen_ids[0, prompt_len:].unsqueeze(1)).squeeze(1)
    chosen_logp = chosen_token_logps.sum()

    rejected_logits = model(input_ids=rejected_ids).logits
    rejected_logps = torch.nn.functional.log_softmax(rejected_logits[0, prompt_len-1:-1, :].float(), dim=-1)
    rejected_token_logps = rejected_logps.gather(1, rejected_ids[0, prompt_len:].unsqueeze(1)).squeeze(1)
    rejected_logp = rejected_token_logps.sum()

    loss = -torch.nn.functional.logsigmoid(beta * (chosen_logp - rejected_logp))
    return loss


def run_gradient_analysis(model_dir, device="cuda:0", model_name="unknown", num_steps=10):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model from {model_dir} in float16 to {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
    )
    model.config.use_cache = False

    num_layers = model.config.num_hidden_layers
    print(f"Model has {num_layers} layers")

    # Index all weight parameters by component
    param_index = {}
    for name, param in model.named_parameters():
        comp_type, layer = classify_component(name)
        if comp_type != "other" and "weight" in name:
            param_index[name] = {"type": comp_type, "layer": layer, "param": param}

    # Freeze everything
    for p in model.parameters():
        p.requires_grad_(False)

    # Collect gradient norms: process one layer at a time
    all_grad_norms = defaultdict(lambda: defaultdict(list))  # type -> layer -> [norms per step]

    for layer_idx in range(num_layers):
        # Enable grad for this layer's components only
        layer_params = {name: info for name, info in param_index.items() if info["layer"] == layer_idx}

        for info in layer_params.values():
            info["param"].requires_grad_(True)

        model.train()

        for step in range(num_steps):
            ex = PREFERENCE_DATA[step % len(PREFERENCE_DATA)]
            model.zero_grad()

            loss = dpo_loss_forward(model, tokenizer, ex["prompt"], ex["chosen"], ex["rejected"], device)
            loss.backward()

            for name, info in layer_params.items():
                if info["param"].grad is not None:
                    grad_norm = info["param"].grad.float().norm().item()
                    all_grad_norms[info["type"]][layer_idx].append(grad_norm)
                    info["param"].grad = None

            torch.cuda.empty_cache()

        # Freeze this layer again
        for info in layer_params.values():
            info["param"].requires_grad_(False)

        # Print progress
        type_norms_this_layer = {}
        for name, info in layer_params.items():
            norms = all_grad_norms[info["type"]].get(layer_idx, [])
            if norms:
                type_norms_this_layer[info["type"]] = np.mean(norms)

        summary = " ".join(f"{t}={v:.2e}" for t, v in sorted(type_norms_this_layer.items()))
        print(f"  L{layer_idx:2d}: {summary}")

    model.eval()

    # ---- Analysis ----
    print(f"\n{'='*80}")
    print(f"GRADIENT MAGNITUDE ANALYSIS ({model_name})")
    print(f"{'='*80}")

    type_layer_means = {}
    for comp_type in ["W_Q", "W_K", "W_V", "W_O", "W_gate", "W_up", "W_down"]:
        layer_means = []
        for layer in range(num_layers):
            norms = all_grad_norms[comp_type].get(layer, [])
            layer_means.append(float(np.mean(norms)) if norms else 0.0)
        type_layer_means[comp_type] = layer_means

    print("\nOverall mean gradient norms:")
    type_overall = {}
    for comp_type in ["W_Q", "W_K", "W_V", "W_O", "W_gate", "W_up", "W_down"]:
        overall = float(np.mean(type_layer_means[comp_type]))
        type_overall[comp_type] = overall
        print(f"  {comp_type:8s}: {overall:.6e}")

    qk_mean = (type_overall["W_Q"] + type_overall["W_K"]) / 2
    vo_mean = (type_overall["W_V"] + type_overall["W_O"]) / 2
    down_mean = type_overall["W_down"]

    print(f"\n  V/O mean:   {vo_mean:.6e}")
    print(f"  Q/K mean:   {qk_mean:.6e}")
    print(f"  W_down:     {down_mean:.6e}")
    if qk_mean > 0:
        print(f"  V/O / Q/K ratio:  {vo_mean/qk_mean:.2f}x")
        print(f"  Down / Q/K ratio: {down_mean/qk_mean:.2f}x")

    # Mid-layer
    mid_start = num_layers // 3
    mid_end = 2 * num_layers // 3
    print(f"\nMid-layer (L{mid_start}-{mid_end}) gradient norms:")
    mid_type_means = {}
    for comp_type in ["W_Q", "W_K", "W_V", "W_O", "W_gate", "W_up", "W_down"]:
        mid_vals = type_layer_means[comp_type][mid_start:mid_end]
        mid_overall = float(np.mean(mid_vals))
        mid_type_means[comp_type] = mid_overall
        print(f"  {comp_type:8s}: {mid_overall:.6e}")

    mid_qk = (mid_type_means["W_Q"] + mid_type_means["W_K"]) / 2
    mid_vo = (mid_type_means["W_V"] + mid_type_means["W_O"]) / 2
    if mid_qk > 0:
        print(f"  Mid V/O / Q/K ratio: {mid_vo/mid_qk:.2f}x")

    # Statistical tests
    from scipy.stats import mannwhitneyu
    vo_all, qk_all, down_all = [], [], []
    for layer in range(num_layers):
        vo_all.extend(all_grad_norms["W_V"].get(layer, []) + all_grad_norms["W_O"].get(layer, []))
        qk_all.extend(all_grad_norms["W_Q"].get(layer, []) + all_grad_norms["W_K"].get(layer, []))
        down_all.extend(all_grad_norms["W_down"].get(layer, []))

    if vo_all and qk_all:
        _, pval = mannwhitneyu(vo_all, qk_all, alternative='greater')
        print(f"\n  Mann-Whitney V/O > Q/K: p = {pval:.6e}")
    if down_all and qk_all:
        _, pval = mannwhitneyu(down_all, qk_all, alternative='greater')
        print(f"  Mann-Whitney W_down > Q/K: p = {pval:.6e}")

    # Save
    output = {
        "analysis": "DPO gradient magnitude by component type",
        "model": model_name,
        "num_steps": num_steps,
        "num_layers": num_layers,
        "type_overall_means": type_overall,
        "type_layer_means": type_layer_means,
        "mid_layer_means": mid_type_means,
        "vo_qk_ratio": vo_mean / max(qk_mean, 1e-10),
        "down_qk_ratio": down_mean / max(qk_mean, 1e-10),
    }

    safe_name = model_name.replace("/", "_").replace(" ", "_").lower()
    out_path = RESULTS_DIR / f"gradient_analysis_{safe_name}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    model_dir = sys.argv[1]
    device = sys.argv[2] if len(sys.argv) > 2 else "cuda:0"
    model_name = sys.argv[3] if len(sys.argv) > 3 else "unknown"
    num_steps = int(sys.argv[4]) if len(sys.argv) > 4 else 10

    run_gradient_analysis(model_dir, device=device, model_name=model_name, num_steps=num_steps)
