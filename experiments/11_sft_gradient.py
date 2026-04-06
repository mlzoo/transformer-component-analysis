"""
Step 30: SFT Gradient Verification

Verify that V/O vs Q/K gradient asymmetry holds under SFT (cross-entropy) loss,
not just DPO. This confirms the softmax Jacobian attenuation is loss-independent.

Usage: python 11_sft_gradient.py <model_dir> <device> <model_name>
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path("./results")

# SFT examples: structured generation completions (same domain as DPO experiments)
SFT_EXAMPLES = [
    {"prompt": "You have access to search(query). Find the capital of France.\nThought: I need to search.\nAction: ", "target": 'search(query="capital of France")'},
    {"prompt": 'Respond in JSON: What is 2+2?\n\n{"', "target": '"answer": 4}'},
    {"prompt": "Generate SQL: Get users older than 25\n\nSELECT ", "target": "* FROM users WHERE age > 25;"},
    {"prompt": 'Output JSON for: Name=Bob, Age=25\n\n{"', "target": '"name": "Bob", "age": 25}'},
    {"prompt": "API: Delete user 7\n\n", "target": "DELETE /api/users/7"},
    {"prompt": "```python\ndef add(a, b):\n    ", "target": "return a + b"},
    {"prompt": "Tools: calculator(expr)\nUser: What is 15*23?\nThought: Use calculator.\nAction: ", "target": 'calculator(expr="15*23")'},
    {"prompt": "YAML:\nserver:\n  port: ", "target": "8080\n  host: 0.0.0.0"},
    {"prompt": 'Parse: "Meeting at 3pm in Room 204"\n\n{"', "target": '"event": "Meeting", "time": "3pm", "room": "204"}'},
    {"prompt": "Cron: Every day at midnight\n\n", "target": "0 0 * * *"},
    {"prompt": "```python\ndef fibonacci(n):\n    ", "target": "if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"},
    {"prompt": 'Sentiment: "This product is amazing!"\n\n{"', "target": '"sentiment": "positive", "confidence": 0.95}'},
    {"prompt": "ReAct agent.\nQ: Capital of France?\nThought: Search.\nAction: ", "target": "search[capital of France]"},
    {"prompt": "SQL: Count orders per customer\n\nSELECT ", "target": "customer_id, COUNT(*) FROM orders GROUP BY customer_id;"},
    {"prompt": "API call: List all products\n\n", "target": "GET /api/products"},
    {"prompt": '```python\ndef is_palindrome(s):\n    ', "target": "return s == s[::-1]"},
    {"prompt": 'Extract entities: "John works at Google in NYC"\n\n{"', "target": '"entities": [{"text": "John", "type": "PERSON"}, {"text": "Google", "type": "ORG"}]}'},
    {"prompt": "Bash: find all .py files modified in last 24h\n\n", "target": "find . -name '*.py' -mtime -1"},
    {"prompt": "Git: create a new branch called feature-auth\n\n", "target": "git checkout -b feature-auth"},
    {"prompt": 'Config YAML:\nlogging:\n  level: ', "target": "INFO\n  format: json\n  output: /var/log/app.log"},
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


def sft_loss_forward(model, tokenizer, prompt, target, device):
    """Compute SFT cross-entropy loss on target tokens only."""
    full_text = prompt + target
    inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=256)
    input_ids = inputs["input_ids"].to(device)
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    prompt_len = prompt_ids.shape[1]

    if prompt_len >= input_ids.shape[1]:
        return None

    logits = model(input_ids=input_ids).logits
    shift_logits = logits[0, prompt_len - 1:-1, :]
    shift_labels = input_ids[0, prompt_len:]
    loss = torch.nn.functional.cross_entropy(shift_logits, shift_labels, reduction='mean')
    return loss


def run_sft_gradient_analysis(model_dir, device="cuda:0", model_name="unknown", num_steps=20):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"SFT Gradient Analysis: {model_name}")
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
    all_grad_norms = defaultdict(lambda: defaultdict(list))

    for layer_idx in range(num_layers):
        layer_params = {name: info for name, info in param_index.items() if info["layer"] == layer_idx}

        for info in layer_params.values():
            info["param"].requires_grad_(True)

        model.train()

        for step in range(num_steps):
            ex = SFT_EXAMPLES[step % len(SFT_EXAMPLES)]
            model.zero_grad()

            loss = sft_loss_forward(model, tokenizer, ex["prompt"], ex["target"], device)
            if loss is None:
                continue
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
        type_norms = {}
        for name, info in layer_params.items():
            norms = all_grad_norms[info["type"]].get(layer_idx, [])
            if norms:
                type_norms[info["type"]] = np.mean(norms)
        summary = " ".join(f"{t}={v:.2e}" for t, v in sorted(type_norms.items()))
        print(f"  L{layer_idx:2d}: {summary}")

    model.eval()

    # Analysis
    print(f"\n{'='*80}")
    print(f"SFT GRADIENT MAGNITUDE ANALYSIS ({model_name})")
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

    print(f"\n  V/O mean: {vo_mean:.6e}")
    print(f"  Q/K mean: {qk_mean:.6e}")
    if qk_mean > 0:
        print(f"  V/O / Q/K ratio: {vo_mean/qk_mean:.2f}x")

    # Statistical tests
    from scipy.stats import mannwhitneyu
    vo_all, qk_all = [], []
    for layer in range(num_layers):
        vo_all.extend(all_grad_norms["W_V"].get(layer, []) + all_grad_norms["W_O"].get(layer, []))
        qk_all.extend(all_grad_norms["W_Q"].get(layer, []) + all_grad_norms["W_K"].get(layer, []))

    pval = None
    if vo_all and qk_all:
        _, pval = mannwhitneyu(vo_all, qk_all, alternative='greater')
        print(f"\n  Mann-Whitney V/O > Q/K: p = {pval:.6e}")
        print(f"  n_vo = {len(vo_all)}, n_qk = {len(qk_all)}")

    # Save
    output = {
        "analysis": "SFT gradient magnitude by component type",
        "loss_function": "cross_entropy_sft",
        "model": model_name,
        "num_steps": num_steps,
        "num_layers": num_layers,
        "type_overall_means": type_overall,
        "type_layer_means": type_layer_means,
        "vo_qk_ratio": vo_mean / max(qk_mean, 1e-10),
        "vo_mean": vo_mean,
        "qk_mean": qk_mean,
        "mannwhitney_p": float(pval) if pval is not None else None,
        "n_vo_samples": len(vo_all),
        "n_qk_samples": len(qk_all),
    }

    safe_name = model_name.replace("/", "_").replace(" ", "_").lower()
    out_path = RESULTS_DIR / f"sft_gradient_{safe_name}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    model_dir = sys.argv[1]
    device = sys.argv[2] if len(sys.argv) > 2 else "cuda:0"
    model_name = sys.argv[3] if len(sys.argv) > 3 else "unknown"
    num_steps = int(sys.argv[4]) if len(sys.argv) > 4 else 20

    run_sft_gradient_analysis(model_dir, device=device, model_name=model_name, num_steps=num_steps)
