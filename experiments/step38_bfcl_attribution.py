"""
Step 38: Attribution on external BFCL data (patching on non-training data).

Runs the same activation-patching attribution as step1, but on BFCL examples
instead of the 209 agent examples. If the same output-pathway concentration
appears, it rules out overfitting to the custom evaluation set.

Uses a subset of BFCL (simple + multiple categories) formatted as
prompt-target pairs for CE loss computation.

Usage: python step38_bfcl_attribution.py <model_name> <device>
"""

import sys, json, torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from safetensors import safe_open

RESULTS_DIR = Path("./results")
BFCL_SNAPSHOT = Path.home() / ".cache/huggingface/hub/datasets--gorilla-llm--Berkeley-Function-Calling-Leaderboard/snapshots"

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


def load_bfcl_examples(max_examples=100):
    """Load BFCL examples as prompt-target pairs for CE evaluation."""
    snap_dir = None
    if BFCL_SNAPSHOT.exists():
        snapshots = list(BFCL_SNAPSHOT.iterdir())
        if snapshots:
            snap_dir = snapshots[0]

    if snap_dir is None:
        print("ERROR: BFCL snapshot not found")
        return []

    examples = []
    for category in ["simple", "multiple", "parallel"]:
        q_file = snap_dir / f"BFCL_v3_{category}.json"
        gt_file = snap_dir / "possible_answer" / f"BFCL_v3_{category}.json"

        if not q_file.exists() or not gt_file.exists():
            print(f"  Skipping {category}: files not found")
            continue

        with open(q_file) as f:
            questions = [json.loads(line) for line in f if line.strip()]
        with open(gt_file) as f:
            answers = [json.loads(line) for line in f if line.strip()]

        for q, a in zip(questions, answers):
            if len(examples) >= max_examples:
                break

            # Extract user message from question
            qdata = q.get("question", [])
            if isinstance(qdata, list) and len(qdata) > 0:
                # qdata is [[{role, content}, ...]]
                msgs = qdata[0] if isinstance(qdata[0], list) else qdata
                user_msg = ""
                for msg in msgs:
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        user_msg = msg.get("content", "")
                if not user_msg and isinstance(msgs[0], dict):
                    user_msg = msgs[0].get("content", "")
            else:
                user_msg = str(qdata)

            # Extract function schemas for context
            funcs = q.get("function", [])
            func_desc = ""
            for fn in funcs[:3]:  # Limit to 3 functions
                if isinstance(fn, dict):
                    func_desc += f"\nFunction: {fn.get('name', '')}"
                    func_desc += f"\nDescription: {fn.get('description', '')}"
                    params = fn.get('parameters', {}).get('properties', {})
                    if params:
                        func_desc += f"\nParameters: {', '.join(params.keys())}"
                    func_desc += "\n"

            # Build prompt
            prompt = f"You have access to the following functions:{func_desc}\nUser: {user_msg}\nCall the appropriate function:\n"

            # Build target from ground truth
            gt = a.get("ground_truth", [])
            if isinstance(gt, list) and len(gt) > 0:
                # Format as function call string
                target_parts = []
                for call in gt:
                    if isinstance(call, dict):
                        for fname, args in call.items():
                            arg_strs = []
                            for k, v in args.items():
                                val = v[0] if isinstance(v, list) and len(v) > 0 else v
                                arg_strs.append(f'{k}={json.dumps(val)}')
                            target_parts.append(f"{fname}({', '.join(arg_strs)})")
                target = "; ".join(target_parts)
            else:
                target = str(gt)

            if prompt and target and len(target) > 3:
                examples.append({"prompt": prompt[:1024], "target": target[:512]})

    print(f"Loaded {len(examples)} BFCL examples")
    return examples[:max_examples]


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


def run_analysis(model_name, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = MODEL_PAIRS[model_name]
    base_dir = config["base"]
    it_dir = config["it"]

    print(f"\n{'='*60}")
    print(f"BFCL attribution analysis: {model_name}")
    print(f"{'='*60}")

    # Load BFCL examples
    examples = load_bfcl_examples(max_examples=100)
    if len(examples) < 10:
        print(f"ERROR: Only {len(examples)} BFCL examples loaded, need at least 10")
        return None

    # Load model
    print(f"Loading IT model...")
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

    # Baseline loss
    print(f"Computing baseline loss on {len(examples)} BFCL examples...")
    baseline_loss = compute_loss(model, tokenizer, examples, device)
    print(f"  Baseline: {baseline_loss:.4f}")

    # Attribution via activation patching (same as step1)
    print(f"\nRunning attribution patching...")
    components = []
    type_scores = defaultdict(list)
    num_layers = model.config.num_hidden_layers

    proj_map = {
        "W_Q": "self_attn.q_proj", "W_K": "self_attn.k_proj",
        "W_V": "self_attn.v_proj", "W_O": "self_attn.o_proj",
        "W_gate": "mlp.gate_proj", "W_up": "mlp.up_proj",
        "W_down": "mlp.down_proj",
    }

    total_components = num_layers * 7
    done = 0

    for layer_idx in range(num_layers):
        for comp_type, proj_path in proj_map.items():
            weight_key = f"model.layers.{layer_idx}.{proj_path}.weight"
            if weight_key not in base_index:
                continue

            # Get module
            layer = model.model.layers[layer_idx]
            parts = proj_path.split(".")
            module = layer
            for p in parts:
                module = getattr(module, p)

            # Load base weight and compute delta
            with safe_open(str(base_index[weight_key]), framework="pt", device="cpu") as sf:
                base_w = sf.get_tensor(weight_key)
            it_w = module.weight.data.float().cpu()
            delta = (it_w - base_w.float())

            # Hook to subtract delta (revert to base)
            def make_hook(d):
                def hook_fn(mod, inp, out):
                    x = inp[0] if isinstance(inp, tuple) else inp
                    x_cpu = x.float().cpu()
                    correction = torch.nn.functional.linear(x_cpu, d)
                    return out - correction.half().to(out.device)
                return hook_fn

            hook = module.register_forward_hook(make_hook(delta))
            patched_loss = compute_loss(model, tokenizer, examples, device)
            hook.remove()

            harm_score = baseline_loss - patched_loss
            components.append({
                "component_type": comp_type,
                "layer": layer_idx,
                "harm_score": harm_score,
            })
            type_scores[comp_type].append(abs(harm_score))

            done += 1
            if done % 28 == 0:
                print(f"  {done}/{total_components} components done")

            del base_w, it_w, delta

    # Compute hierarchy
    total_abs = sum(sum(s) for s in type_scores.values())
    print(f"\nBFCL Attribution Hierarchy:")

    hierarchy = {}
    for comp in ["W_Q", "W_K", "W_V", "W_O", "W_gate", "W_up", "W_down"]:
        scores = type_scores.get(comp, [])
        if not scores:
            continue
        comp_sum = sum(scores)
        share = comp_sum / total_abs * 100 if total_abs > 0 else 0
        hierarchy[comp] = {
            "sum": comp_sum,
            "mean": float(np.mean(scores)),
            "share_pct": share,
            "n": len(scores),
        }
        print(f"  {comp:8s}: share={share:5.1f}%")

    # V/O vs Q/K
    vo_scores = type_scores.get("W_V", []) + type_scores.get("W_O", [])
    qk_scores = type_scores.get("W_Q", []) + type_scores.get("W_K", [])

    from scipy.stats import mannwhitneyu
    if vo_scores and qk_scores:
        stat, pval = mannwhitneyu(vo_scores, qk_scores, alternative='greater')
        ratio = np.mean(vo_scores) / np.mean(qk_scores) if np.mean(qk_scores) > 0 else float('inf')
    else:
        pval, ratio = 1.0, 0.0

    vo_share = hierarchy.get("W_V", {}).get("share_pct", 0) + hierarchy.get("W_O", {}).get("share_pct", 0)
    qk_share = hierarchy.get("W_Q", {}).get("share_pct", 0) + hierarchy.get("W_K", {}).get("share_pct", 0)
    mlp_share = sum(hierarchy.get(c, {}).get("share_pct", 0) for c in ["W_gate", "W_up", "W_down"])

    print(f"\n  MLP: {mlp_share:.1f}%  V/O: {vo_share:.1f}%  Q/K: {qk_share:.1f}%")
    print(f"  V/O:Q/K ratio: {ratio:.2f}x (p={pval:.2e})")

    results = {
        "analysis": "BFCL attribution (external validation)",
        "model": model_name,
        "num_examples": len(examples),
        "baseline_loss": baseline_loss,
        "hierarchy": hierarchy,
        "groups": {
            "MLP_share": mlp_share,
            "VO_share": vo_share,
            "QK_share": qk_share,
            "VO_QK_ratio": ratio,
            "VO_QK_pvalue": pval,
        },
        "components": components,
    }

    out_path = RESULTS_DIR / f"step38_bfcl_attribution_{model_name}.json"
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
