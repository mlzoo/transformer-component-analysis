"""
Step 2 v2: Causal Intervention via Cumulative Component Rollback

For the paper's causal claim: "rolling back V/O and output-pathway components
recovers >90% of structured token probability suppressed by alignment."

Approach:
1. Measure base model loss on agent tasks (reference)
2. Measure IT model loss (baseline — higher = alignment tax)
3. Cumulative rollback by type: roll back ALL components of a type simultaneously
4. Cumulative rollback by harm rank: progressively roll back top-K components
5. Compare: MLP+V/O vs Q/K recovery contribution

Uses activation patching (hooks) like step1 — no weight copying needed.
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from safetensors import safe_open

RESULTS_DIR = Path("./results")

AGENT_EXAMPLES = [
    {"prompt": "You are a helpful assistant with tools.\nTools: search(query), calculator(expr)\nUser: Population of France times 2?\nThought: Search first.\nAction: search(query=\"population of France\")\nObservation: 68 million.\nThought: Multiply.\nAction: ", "target": 'calculator(expression="68000000 * 2")'},
    {"prompt": 'Respond in JSON.\nUser: What is 2+2?\n\n{"', "target": '"answer": 4}'},
    {"prompt": "ReAct agent.\nQ: Capital of France?\nThought: Search.\nAction: ", "target": "search[capital of France]"},
    {"prompt": 'Output JSON: Name=Alice, Age=30\n\n{"name": "', "target": 'Alice", "age": 30}'},
    {"prompt": "SQL: Get users where age > 25\n\nSELECT ", "target": "* FROM users WHERE age > 25;"},
    {"prompt": "API call: Delete user 42\n\n", "target": "DELETE /api/users/42"},
    {"prompt": "```python\ndef fibonacci(n):\n    ", "target": "if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"},
    {"prompt": "Tools: web_search(q), read_file(path)\nTask: Weather in NYC\nThought: Search.\nAction: ", "target": 'web_search(q="weather NYC")'},
    {"prompt": 'Parse to JSON: "Meeting 3pm Room 204"\n\n{"', "target": '"event": "Meeting", "time": "3pm", "location": "Room 204"}'},
    {"prompt": "Function call: send_email(to=", "target": '"user@example.com", subject="Hello", body="Test")'},
    {"prompt": 'Product: Widget, Price: $9.99, Stock: 150\n\n{"product": "', "target": 'Widget", "price": 9.99, "stock": 150}'},
    {"prompt": "Bash: List .py files modified today\n\n```bash\n", "target": "find . -name '*.py' -mtime 0\n```"},
    {"prompt": 'Code review bot. Code: x = eval(input())\n\n{"', "target": '"verdict": "reject", "reason": "eval on user input"}'},
    {"prompt": "YAML config:\nserver:\n  host: 0.0.0.0\n  port: ", "target": "8080\n  workers: 4"},
    {"prompt": "GraphQL: Get user by ID with posts\n\n```graphql\n", "target": "query GetUser($id: ID!) {\n  user(id: $id) {\n    name\n    posts { title }\n  }\n}"},
    {"prompt": 'Router: GET /api/users/123\n\n{"', "target": '"handler": "getUser", "params": {"id": "123"}}'},
    {"prompt": "MongoDB: Orders over $100 last week\n\ndb.orders.find(", "target": '{"amount": {"$gt": 100}})'},
    {"prompt": "Dockerfile for Flask:\n\nFROM ", "target": "python:3.11-slim\nWORKDIR /app\nCOPY requirements.txt .\nRUN pip install -r requirements.txt"},
    {"prompt": 'CI decision: 142/142 unit pass, 38/40 integ pass\n\n{"', "target": '"action": "proceed", "deploy": true, "warnings": ["2 flaky tests"]}'},
    {"prompt": "Cron: Every Monday 9am\n\n", "target": "0 9 * * 1"},
]


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


def load_weight_index(model_dir):
    files = sorted(Path(model_dir).glob("*.safetensors"))
    index = {}
    for f in files:
        with safe_open(str(f), framework="pt", device="cpu") as sf:
            for key in sf.keys():
                index[key] = f
    return index


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


def get_module_for_weight(model, weight_name):
    """Get the nn.Module whose .weight corresponds to this weight_name."""
    # weight_name like "model.layers.5.self_attn.v_proj.weight"
    parts = weight_name.replace(".weight", "").split(".")
    module = model
    for part in parts:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


def run_causal_v2(base_dir, it_dir, device="cuda:0", model_name="unknown"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load attribution results
    safe_name = model_name.replace("/", "_").replace(" ", "_").lower()
    attr_path = RESULTS_DIR / f"step1_attribution_fp16_{safe_name}.json"
    if not attr_path.exists():
        print(f"ERROR: No attribution results at {attr_path}")
        return

    with open(attr_path) as f:
        attr_data = json.load(f)

    # Sort components by harm score
    all_components = sorted(attr_data["components"], key=lambda x: x["harm_score"], reverse=True)
    print(f"Loaded {len(all_components)} components from attribution")

    # Load model
    print(f"Loading IT model in float16 to {device}...")
    tokenizer = AutoTokenizer.from_pretrained(it_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        it_dir, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
    )
    model.eval()

    base_index = load_weight_index(base_dir)
    it_index = load_weight_index(it_dir)
    examples = AGENT_EXAMPLES[:20]

    # Baseline losses
    print("Measuring IT model loss...")
    it_loss = compute_loss(model, tokenizer, examples, device)
    print(f"  IT loss: {it_loss:.4f}")

    # ====================================================================
    # EXPERIMENT A: Cumulative rollback by component TYPE
    # Roll back ALL components of each type simultaneously
    # ====================================================================
    print("\n" + "="*80)
    print("EXPERIMENT A: Rollback by component type (all layers)")
    print("="*80)

    type_groups = {
        "W_Q": [], "W_K": [], "W_V": [], "W_O": [],
        "W_gate": [], "W_up": [], "W_down": [],
    }
    for comp in all_components:
        ct = comp["component_type"]
        if ct in type_groups:
            type_groups[ct].append(comp)

    type_rollback_results = {}

    for type_name in ["W_Q", "W_K", "W_V", "W_O", "W_gate", "W_up", "W_down"]:
        comps = type_groups[type_name]
        if not comps:
            continue

        # Install hooks for ALL components of this type
        hooks = []
        for comp in comps:
            weight_key = comp["name"]
            if weight_key not in base_index or weight_key not in it_index:
                continue

            with safe_open(str(base_index[weight_key]), framework="pt", device="cpu") as sf:
                base_w = sf.get_tensor(weight_key)
            with safe_open(str(it_index[weight_key]), framework="pt", device="cpu") as sf:
                it_w = sf.get_tensor(weight_key)

            delta_cpu = (it_w - base_w).float()
            del base_w, it_w

            module = get_module_for_weight(model, weight_key)

            def make_hook(delta):
                def hook_fn(mod, inp, out):
                    x = inp[0] if isinstance(inp, tuple) else inp
                    x_cpu = x.float().cpu()
                    correction = torch.nn.functional.linear(x_cpu, delta)
                    return out - correction.half().to(out.device)
                return hook_fn

            h = module.register_forward_hook(make_hook(delta_cpu))
            hooks.append(h)
            del delta_cpu

        loss = compute_loss(model, tokenizer, examples, device)
        for h in hooks:
            h.remove()
        torch.cuda.empty_cache()

        recovery = it_loss - loss  # positive = rolling back helped (reduced loss)
        type_rollback_results[type_name] = {
            "num_components": len(comps),
            "rollback_loss": float(loss),
            "recovery": float(recovery),
        }
        print(f"  {type_name:8s} ({len(comps):2d} comps): loss={loss:.4f}, recovery={recovery:+.4f}")

    # ====================================================================
    # EXPERIMENT B: Grouped rollback — V/O, Q/K, MLP, output-pathway
    # ====================================================================
    print("\n" + "="*80)
    print("EXPERIMENT B: Grouped rollback")
    print("="*80)

    groups = {
        "V/O": ["W_V", "W_O"],
        "Q/K": ["W_Q", "W_K"],
        "MLP": ["W_gate", "W_up", "W_down"],
        "output_pathway": ["W_V", "W_O", "W_down"],
        "all": ["W_Q", "W_K", "W_V", "W_O", "W_gate", "W_up", "W_down"],
    }

    group_results = {}

    for group_name, type_list in groups.items():
        comps = []
        for t in type_list:
            comps.extend(type_groups.get(t, []))

        if not comps:
            continue

        hooks = []
        for comp in comps:
            weight_key = comp["name"]
            if weight_key not in base_index or weight_key not in it_index:
                continue

            with safe_open(str(base_index[weight_key]), framework="pt", device="cpu") as sf:
                base_w = sf.get_tensor(weight_key)
            with safe_open(str(it_index[weight_key]), framework="pt", device="cpu") as sf:
                it_w = sf.get_tensor(weight_key)

            delta_cpu = (it_w - base_w).float()
            del base_w, it_w

            module = get_module_for_weight(model, weight_key)

            def make_hook(delta):
                def hook_fn(mod, inp, out):
                    x = inp[0] if isinstance(inp, tuple) else inp
                    x_cpu = x.float().cpu()
                    correction = torch.nn.functional.linear(x_cpu, delta)
                    return out - correction.half().to(out.device)
                return hook_fn

            h = module.register_forward_hook(make_hook(delta_cpu))
            hooks.append(h)
            del delta_cpu

        loss = compute_loss(model, tokenizer, examples, device)
        for h in hooks:
            h.remove()
        torch.cuda.empty_cache()

        recovery = it_loss - loss
        group_results[group_name] = {
            "types": type_list,
            "num_components": len(comps),
            "rollback_loss": float(loss),
            "recovery": float(recovery),
        }
        print(f"  {group_name:18s} ({len(comps):3d} comps): loss={loss:.4f}, recovery={recovery:+.4f}")

    # ====================================================================
    # EXPERIMENT C: Cumulative top-K rollback (by harm rank)
    # Roll back top-5, top-10, top-20, top-50, top-100 simultaneously
    # ====================================================================
    print("\n" + "="*80)
    print("EXPERIMENT C: Cumulative top-K rollback (by harm rank)")
    print("="*80)

    topk_results = {}
    for k in [5, 10, 20, 50]:
        top_comps = all_components[:k]

        hooks = []
        for comp in top_comps:
            weight_key = comp["name"]
            if weight_key not in base_index or weight_key not in it_index:
                continue

            with safe_open(str(base_index[weight_key]), framework="pt", device="cpu") as sf:
                base_w = sf.get_tensor(weight_key)
            with safe_open(str(it_index[weight_key]), framework="pt", device="cpu") as sf:
                it_w = sf.get_tensor(weight_key)

            delta_cpu = (it_w - base_w).float()
            del base_w, it_w

            module = get_module_for_weight(model, weight_key)

            def make_hook(delta):
                def hook_fn(mod, inp, out):
                    x = inp[0] if isinstance(inp, tuple) else inp
                    x_cpu = x.float().cpu()
                    correction = torch.nn.functional.linear(x_cpu, delta)
                    return out - correction.half().to(out.device)
                return hook_fn

            h = module.register_forward_hook(make_hook(delta_cpu))
            hooks.append(h)
            del delta_cpu

        loss = compute_loss(model, tokenizer, examples, device)
        for h in hooks:
            h.remove()
        torch.cuda.empty_cache()

        recovery = it_loss - loss

        # Composition of top-K
        type_counts = defaultdict(int)
        for comp in top_comps:
            type_counts[comp["component_type"]] += 1

        topk_results[k] = {
            "k": k,
            "rollback_loss": float(loss),
            "recovery": float(recovery),
            "type_composition": dict(type_counts),
        }
        print(f"  Top-{k:3d}: loss={loss:.4f}, recovery={recovery:+.4f}  composition={dict(type_counts)}")

    # ====================================================================
    # EXPERIMENT D: Mid-layer output pathway (mechanistic heuristic)
    # Roll back W_down + W_V + W_O at middle 1/3 layers only
    # ====================================================================
    print("\n" + "="*80)
    print("EXPERIMENT D: Mechanistic heuristic (output pathway, mid layers)")
    print("="*80)

    num_layers = model.config.num_hidden_layers
    mid_start = num_layers // 3
    mid_end = 2 * num_layers // 3

    heuristic_configs = {
        "output_mid": {
            "types": ["W_V", "W_O", "W_down"],
            "layer_range": (mid_start, mid_end),
        },
        "vo_mid": {
            "types": ["W_V", "W_O"],
            "layer_range": (mid_start, mid_end),
        },
        "mlp_mid": {
            "types": ["W_down"],
            "layer_range": (mid_start, mid_end),
        },
        "qk_mid": {
            "types": ["W_Q", "W_K"],
            "layer_range": (mid_start, mid_end),
        },
    }

    heuristic_results = {}

    for heur_name, config in heuristic_configs.items():
        comps = []
        for comp in all_components:
            if (comp["component_type"] in config["types"] and
                config["layer_range"][0] <= comp["layer"] < config["layer_range"][1]):
                comps.append(comp)

        if not comps:
            continue

        hooks = []
        for comp in comps:
            weight_key = comp["name"]
            if weight_key not in base_index or weight_key not in it_index:
                continue

            with safe_open(str(base_index[weight_key]), framework="pt", device="cpu") as sf:
                base_w = sf.get_tensor(weight_key)
            with safe_open(str(it_index[weight_key]), framework="pt", device="cpu") as sf:
                it_w = sf.get_tensor(weight_key)

            delta_cpu = (it_w - base_w).float()
            del base_w, it_w

            module = get_module_for_weight(model, weight_key)

            def make_hook(delta):
                def hook_fn(mod, inp, out):
                    x = inp[0] if isinstance(inp, tuple) else inp
                    x_cpu = x.float().cpu()
                    correction = torch.nn.functional.linear(x_cpu, delta)
                    return out - correction.half().to(out.device)
                return hook_fn

            h = module.register_forward_hook(make_hook(delta_cpu))
            hooks.append(h)
            del delta_cpu

        loss = compute_loss(model, tokenizer, examples, device)
        for h in hooks:
            h.remove()
        torch.cuda.empty_cache()

        recovery = it_loss - loss
        heuristic_results[heur_name] = {
            "config": config,
            "num_components": len(comps),
            "rollback_loss": float(loss),
            "recovery": float(recovery),
        }
        print(f"  {heur_name:18s} ({len(comps):2d} comps, L{mid_start}-{mid_end}): "
              f"loss={loss:.4f}, recovery={recovery:+.4f}")

    # ====================================================================
    # Save results
    # ====================================================================
    # Also need base model loss for reference
    print("\nLoading base model for reference loss...")
    del model
    torch.cuda.empty_cache()

    base_model = AutoModelForCausalLM.from_pretrained(
        base_dir, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
    )
    base_model.eval()
    base_loss = compute_loss(base_model, tokenizer, examples, device)
    print(f"  Base model loss: {base_loss:.4f}")
    del base_model
    torch.cuda.empty_cache()

    alignment_tax = it_loss - base_loss  # positive = IT is worse

    print(f"\n{'='*80}")
    print(f"SUMMARY: {model_name}")
    print(f"  Base loss:      {base_loss:.4f}")
    print(f"  IT loss:        {it_loss:.4f}")
    print(f"  Alignment tax:  {alignment_tax:+.4f}")
    print(f"{'='*80}")

    # Recovery fractions
    all_rollback_recovery = group_results.get("all", {}).get("recovery", 0)
    vo_recovery = group_results.get("V/O", {}).get("recovery", 0)
    qk_recovery = group_results.get("Q/K", {}).get("recovery", 0)
    mlp_recovery = group_results.get("MLP", {}).get("recovery", 0)
    output_recovery = group_results.get("output_pathway", {}).get("recovery", 0)

    print(f"  Full rollback recovery:    {all_rollback_recovery:+.4f} ({all_rollback_recovery/alignment_tax*100:.1f}% of tax)")
    print(f"  Output pathway (V/O+down): {output_recovery:+.4f} ({output_recovery/alignment_tax*100:.1f}% of tax)")
    print(f"  MLP only:                  {mlp_recovery:+.4f} ({mlp_recovery/alignment_tax*100:.1f}% of tax)")
    print(f"  V/O only:                  {vo_recovery:+.4f} ({vo_recovery/alignment_tax*100:.1f}% of tax)")
    print(f"  Q/K only:                  {qk_recovery:+.4f} ({qk_recovery/alignment_tax*100:.1f}% of tax)")

    output = {
        "analysis": "Causal intervention v2 - cumulative rollback",
        "model_pair": model_name,
        "base_loss": float(base_loss),
        "it_loss": float(it_loss),
        "alignment_tax": float(alignment_tax),
        "type_rollback": type_rollback_results,
        "group_rollback": group_results,
        "topk_rollback": topk_results,
        "heuristic_rollback": heuristic_results,
    }

    out_path = RESULTS_DIR / f"step2_causal_v2_{safe_name}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    base_dir = sys.argv[1]
    it_dir = sys.argv[2]
    device = sys.argv[3] if len(sys.argv) > 3 else "cuda:0"
    model_name = sys.argv[4] if len(sys.argv) > 4 else "unknown"

    print(f"Base: {base_dir}")
    print(f"IT:   {it_dir}")
    print(f"Device: {device}")
    print(f"Model: {model_name}")

    run_causal_v2(base_dir, it_dir, device=device, model_name=model_name)
