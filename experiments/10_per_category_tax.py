"""
Step 29: Per-category alignment tax breakdown.

Classify 209 agent examples into task categories and compute alignment tax
(IT loss - base loss) per category. Shows that even models with negative overall
tax have positive tax in specific task categories.

Runs one model per GPU. Usage:
    python 10_per_category_tax.py qwen2.5-7b cuda:0
    python 10_per_category_tax.py llama-3.1-8b cuda:1
    python 10_per_category_tax.py mistral-7b cuda:2
"""

import sys
import json
import re
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path("./results")

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


def classify_example(prompt, target):
    """Classify an agent example into a task category based on content."""
    p = prompt.lower()
    t = target.lower()
    combined = p + " " + t

    # Order matters: more specific patterns first
    if "```python" in combined or "def " in t or "import " in t or "class " in t:
        return "code"
    if "```bash" in combined or "#!/" in t or ("$(" in t and "echo" in t):
        return "bash"
    if re.search(r'\bselect\b.*\bfrom\b', combined) or "sql" in p:
        return "sql"
    if '{"' in t or '":{' in t or 'json' in p:
        return "json"
    if re.search(r'(get|post|put|delete|patch)\s+/api/', combined, re.I) or "api" in p.split():
        return "api"
    if "regex" in p or re.search(r'\\[dwsb]', t):
        return "regex"
    if "cron" in combined:
        return "cron"
    if "graphql" in combined or "query {" in t or "mutation {" in t:
        return "graphql"
    if "mongodb" in combined or "db." in t or "find({" in t:
        return "mongodb"
    if "docker" in combined or "dockerfile" in p:
        return "docker"
    if "terraform" in combined or 'resource "' in t:
        return "terraform"
    if "yaml" in p or "---\n" in t:
        return "yaml"
    if "xml" in p or "</" in t and ">" in t and "<" in t:
        return "xml"
    if "protobuf" in combined or "message " in t and "string " in t:
        return "protobuf"
    if "css" in p or "{" in t and ("color:" in t or "margin:" in t or "display:" in t):
        return "css"
    if "git " in t or "git" in p.split():
        return "git"
    if re.search(r'ci[/_ ]?cd|pipeline|github.actions|\.yml', combined):
        return "cicd"
    if "tools:" in p or "function" in p and "action:" in p:
        return "tool_call"
    if "thought:" in p and "action:" in p:
        return "react"
    if "translate" in p:
        return "translation"
    # Fallback
    return "other"


def compute_loss(model, tokenizer, prompt, target, device):
    """Compute cross-entropy loss on the target portion only."""
    full_text = prompt + target
    inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(device)
    prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
    if prompt_len >= input_ids.shape[1]:
        return None
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    logits = outputs.logits
    shift_logits = logits[0, prompt_len - 1:-1, :]
    shift_labels = input_ids[0, prompt_len:]
    loss = torch.nn.functional.cross_entropy(shift_logits, shift_labels, reduction='mean')
    return loss.item()


def run_per_category_tax(model_name, device):
    import gc
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = MODEL_CONFIGS[model_name]
    base_dir = config["base"]
    it_dir = config["it"]

    examples = AGENT_EXAMPLES_200
    print(f"Model: {model_name}, Device: {device}")
    print(f"Total examples: {len(examples)}")

    # Classify all examples
    categories = []
    for ex in examples:
        target = ex.get("target", ex.get("chosen", ""))
        cat = classify_example(ex["prompt"], target)
        categories.append(cat)

    cat_counts = defaultdict(int)
    for c in categories:
        cat_counts[c] += 1
    print(f"\nCategory distribution:")
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat:15s}: {count}")

    tokenizer = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model first
    print("\nLoading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_dir, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
    )
    model.eval()

    base_losses = []
    for i, ex in enumerate(examples):
        target = ex.get("target", ex.get("chosen", ""))
        loss = compute_loss(model, tokenizer, ex["prompt"], target, device)
        base_losses.append(loss)
        if (i + 1) % 50 == 0:
            print(f"  Base: {i+1}/{len(examples)}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Load IT model
    print("\nLoading IT model...")
    model = AutoModelForCausalLM.from_pretrained(
        it_dir, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
    )
    model.eval()

    it_losses = []
    for i, ex in enumerate(examples):
        target = ex.get("target", ex.get("chosen", ""))
        loss = compute_loss(model, tokenizer, ex["prompt"], target, device)
        it_losses.append(loss)
        if (i + 1) % 50 == 0:
            print(f"  IT: {i+1}/{len(examples)}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Per-category analysis
    cat_results = defaultdict(lambda: {"base": [], "it": [], "tax": []})
    for i, (bl, il, cat) in enumerate(zip(base_losses, it_losses, categories)):
        if bl is not None and il is not None:
            cat_results[cat]["base"].append(bl)
            cat_results[cat]["it"].append(il)
            cat_results[cat]["tax"].append(il - bl)

    # Overall
    valid = [(bl, il) for bl, il in zip(base_losses, it_losses) if bl is not None and il is not None]
    overall_base = np.mean([b for b, _ in valid])
    overall_it = np.mean([i for _, i in valid])
    overall_tax = overall_it - overall_base

    print(f"\n{'='*70}")
    print(f"PER-CATEGORY ALIGNMENT TAX: {model_name}")
    print(f"{'='*70}")
    print(f"Overall: base={overall_base:.4f}, IT={overall_it:.4f}, tax={overall_tax:+.4f}")
    print(f"\n{'Category':15s} {'N':>4s} {'Base':>8s} {'IT':>8s} {'Tax':>8s} {'Tax%':>7s} {'Dir':>5s}")
    print("-" * 55)

    n_positive = 0
    n_negative = 0
    output_cats = {}

    for cat in sorted(cat_results.keys(), key=lambda c: -np.mean(cat_results[c]["tax"])):
        cr = cat_results[cat]
        n = len(cr["base"])
        base_mean = np.mean(cr["base"])
        it_mean = np.mean(cr["it"])
        tax_mean = np.mean(cr["tax"])
        tax_pct = tax_mean / base_mean * 100 if base_mean > 0 else 0
        direction = "+" if tax_mean > 0 else "-"

        if tax_mean > 0:
            n_positive += n
        else:
            n_negative += n

        print(f"{cat:15s} {n:4d} {base_mean:8.4f} {it_mean:8.4f} {tax_mean:+8.4f} {tax_pct:+6.1f}% {direction:>5s}")

        output_cats[cat] = {
            "n": n,
            "base_mean": float(base_mean),
            "it_mean": float(it_mean),
            "tax_mean": float(tax_mean),
            "tax_pct": float(tax_pct),
            "base_losses": [float(x) for x in cr["base"]],
            "it_losses": [float(x) for x in cr["it"]],
        }

    print(f"\nExamples with positive tax: {n_positive}/{n_positive+n_negative}")
    print(f"Examples with negative tax: {n_negative}/{n_positive+n_negative}")

    # Per-example tax direction
    n_pos_ex = sum(1 for bl, il in valid if il > bl)
    n_neg_ex = sum(1 for bl, il in valid if il <= bl)
    print(f"\nPer-example: {n_pos_ex} positive, {n_neg_ex} negative")

    results = {
        "model": model_name,
        "n_examples": len(examples),
        "overall_base": float(overall_base),
        "overall_it": float(overall_it),
        "overall_tax": float(overall_tax),
        "n_positive_tax_examples": int(n_pos_ex),
        "n_negative_tax_examples": int(n_neg_ex),
        "categories": output_cats,
    }

    out_path = RESULTS_DIR / f"per_category_tax_{model_name}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    return results


if __name__ == "__main__":
    model_name = sys.argv[1]
    device = sys.argv[2] if len(sys.argv) > 2 else "cuda:0"
    run_per_category_tax(model_name, device)
