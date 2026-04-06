"""
Step 20: OC-DPO Safety Evaluation

Train OC-DPO models (exclude V/O/down) then evaluate safety on 250 harmful prompts.
Also runs standard DPO for comparison. This addresses reviewer concern W4:
"OC-DPO's alignment quality is not verified."

Reuses: safety prompts and refusal detection from step14_safety_large.py
        DPO training from step6_ocdpo_v4.py
"""

import sys
import json
import time
import torch
import numpy as np
from pathlib import Path

RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Model configurations (IT models as base for DPO)
# ============================================================
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

# ============================================================
# DPO training data (same as step6)
# ============================================================
TRAIN_DATA = [
    {"prompt": "Tools: search(q)\nUser: Capital of France?\nThought: Search.\nAction: ", "chosen": 'search(q="capital of France")', "rejected": "The capital of France is Paris."},
    {"prompt": 'JSON: Name=Alice, Age=30\n\n{"', "chosen": '"name": "Alice", "age": 30}', "rejected": "Alice is 30 years old."},
    {"prompt": "SQL: Users where age > 25\n\nSELECT ", "chosen": "* FROM users WHERE age > 25;", "rejected": "I would select users older than 25."},
    {"prompt": "API: Delete user 42\n\n", "chosen": "DELETE /api/users/42", "rejected": "To delete user 42, make an API call."},
    {"prompt": "```python\ndef add(a, b):\n    ", "chosen": "return a + b", "rejected": "This adds two numbers."},
    {"prompt": "Tools: calc(expr)\n15*23=?\nAction: ", "chosen": 'calc(expr="15*23")', "rejected": "15 times 23 is 345."},
    {"prompt": "YAML:\nport: ", "chosen": "8080\nhost: 0.0.0.0", "rejected": "The port should be 8080."},
    {"prompt": 'Parse: "Meeting 3pm"\n\n{"', "chosen": '"event": "Meeting", "time": "3pm"}', "rejected": "There's a meeting at 3pm."},
    {"prompt": "Cron: Every Monday 9am\n\n", "chosen": "0 9 * * 1", "rejected": "Set cron for Monday 9am."},
    {"prompt": "Bash: Find .py files\n\n```bash\n", "chosen": "find . -name '*.py'\n```", "rejected": "You can use the find command."},
    {"prompt": 'Router: GET /users/5\n\n{"', "chosen": '"handler": "getUser", "id": 5}', "rejected": "Route to user handler."},
    {"prompt": "GraphQL: Get user posts\n\n```graphql\n", "chosen": "query { user(id: 1) { posts { title } } }", "rejected": "Query the user's posts."},
    {"prompt": "MongoDB: Find active users\n\ndb.users.find(", "chosen": '{"active": true})', "rejected": "Find users where active is true."},
    {"prompt": "Dockerfile:\n\nFROM ", "chosen": "python:3.11-slim\nWORKDIR /app", "rejected": "Use a Python base image."},
    {"prompt": 'Sentiment: "Great product!"\n\n{"', "chosen": '"sentiment": "positive"}', "rejected": "The sentiment is positive."},
    {"prompt": "Regex: Match emails\n\n", "chosen": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "rejected": "Use a regex pattern for emails."},
    {"prompt": "CSS: Center div\n\n.box {\n  ", "chosen": "display: flex;\n  justify-content: center;", "rejected": "Use flexbox to center."},
    {"prompt": "git: Undo last commit\n\n$ ", "chosen": "git reset --soft HEAD~1", "rejected": "You can undo with git reset."},
    {"prompt": 'CI: 10/10 pass\n\n{"', "chosen": '"status": "pass", "deploy": true}', "rejected": "All tests passed."},
    {"prompt": "Terraform: EC2\n\nresource \"aws_instance\" \"web\" {\n  ", "chosen": 'ami = "ami-abc"\n  instance_type = "t2.micro"', "rejected": "Create an EC2 resource."},
]

# ============================================================
# Safety prompts (subset from step14 — 250 harmful)
# ============================================================
from step14_safety_large import (
    ALL_HARMFUL_CATEGORIES, BORDERLINE, BENIGN, TOTAL_HARMFUL,
    detect_refusal, generate_response
)

# ============================================================
# DPO Training
# ============================================================
ALL_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

def get_mid_layer_targets(num_layers):
    mid_start = num_layers // 3
    mid_end = 2 * num_layers // 3
    return list(range(mid_start, mid_end))


def train_dpo(model, tokenizer, device, num_epochs=5, lr=5e-5):
    """Train DPO on the 20 preference pairs."""
    model.train()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01
    )

    for epoch in range(num_epochs):
        epoch_loss = 0
        for example in TRAIN_DATA:
            optimizer.zero_grad()
            prompt = example["prompt"]
            chosen_text = prompt + example["chosen"]
            rejected_text = prompt + example["rejected"]

            chosen_ids = tokenizer(chosen_text, return_tensors="pt",
                                  truncation=True, max_length=256)["input_ids"].to(device)
            rejected_ids = tokenizer(rejected_text, return_tensors="pt",
                                   truncation=True, max_length=256)["input_ids"].to(device)
            prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]

            chosen_logits = model(input_ids=chosen_ids).logits
            chosen_lps = torch.nn.functional.log_softmax(
                chosen_logits[0, prompt_len-1:-1, :].float(), dim=-1)
            chosen_logp = chosen_lps.gather(
                1, chosen_ids[0, prompt_len:].unsqueeze(1)).squeeze(1).sum()

            rejected_logits = model(input_ids=rejected_ids).logits
            rejected_lps = torch.nn.functional.log_softmax(
                rejected_logits[0, prompt_len-1:-1, :].float(), dim=-1)
            rejected_logp = rejected_lps.gather(
                1, rejected_ids[0, prompt_len:].unsqueeze(1)).squeeze(1).sum()

            loss = -torch.nn.functional.logsigmoid(0.1 * (chosen_logp - rejected_logp))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            torch.cuda.empty_cache()

        print(f"    Epoch {epoch+1}/{num_epochs}: DPO loss={epoch_loss/len(TRAIN_DATA):.4f}")

    del optimizer
    torch.cuda.empty_cache()
    return model


def evaluate_safety(model, tokenizer, device, label):
    """Run safety evaluation on 250 harmful + controls."""
    model.eval()
    category_results = {}
    all_harmful_refusals = 0
    all_harmful_total = 0

    for cat_name, prompts in ALL_HARMFUL_CATEGORIES.items():
        n_refused = 0
        for i, prompt in enumerate(prompts):
            response = generate_response(model, tokenizer, prompt, device)
            if detect_refusal(response):
                n_refused += 1
        refusal_rate = n_refused / len(prompts)
        category_results[cat_name] = {
            "refusal_rate": round(refusal_rate, 4),
            "n_refused": n_refused,
            "n_total": len(prompts),
        }
        all_harmful_refusals += n_refused
        all_harmful_total += len(prompts)
        print(f"    [{label}] {cat_name}: {refusal_rate:.1%} ({n_refused}/{len(prompts)})")

    # Benign false positive check
    benign_refused = 0
    for prompt in BENIGN:
        response = generate_response(model, tokenizer, prompt, device)
        if detect_refusal(response):
            benign_refused += 1
    benign_fp = benign_refused / len(BENIGN)

    overall = all_harmful_refusals / max(all_harmful_total, 1)
    print(f"    [{label}] Overall: {overall:.1%} | Benign FP: {benign_fp:.1%}")

    return {
        "categories": category_results,
        "overall_harmful_refusal": round(overall, 4),
        "benign_false_positive": round(benign_fp, 4),
        "n_harmful_total": all_harmful_total,
        "n_harmful_refused": all_harmful_refusals,
    }


def run_model(model_key, device="cuda:0"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    config = MODEL_CONFIGS[model_key]
    it_dir = config["it"]

    print(f"\n{'='*70}")
    print(f"  OC-DPO SAFETY EVALUATION: {model_key}")
    print(f"{'='*70}")

    tokenizer = AutoTokenizer.from_pretrained(it_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = {}

    # --- 1. IT baseline (no DPO) ---
    print(f"\n--- {model_key} IT (baseline) ---")
    model = AutoModelForCausalLM.from_pretrained(
        it_dir, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
    )
    t0 = time.time()
    results["it"] = evaluate_safety(model, tokenizer, device, "IT")
    results["it"]["eval_time_s"] = round(time.time() - t0, 1)
    del model
    torch.cuda.empty_cache()

    # --- 2. Standard DPO ---
    print(f"\n--- {model_key} Standard DPO ---")
    model = AutoModelForCausalLM.from_pretrained(
        it_dir, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
    )
    model.config.use_cache = False
    num_layers = model.config.num_hidden_layers
    mid_layers = get_mid_layer_targets(num_layers)

    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=ALL_TARGETS,
        layers_to_transform=mid_layers,
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    print(f"  Training standard DPO (all targets, layers {mid_layers[0]}-{mid_layers[-1]})...")
    model = train_dpo(model, tokenizer, device)
    model.config.use_cache = True

    t0 = time.time()
    results["standard_dpo"] = evaluate_safety(model, tokenizer, device, "Std-DPO")
    results["standard_dpo"]["eval_time_s"] = round(time.time() - t0, 1)
    del model
    torch.cuda.empty_cache()

    # --- 3. OC-DPO (exclude V/O/down) ---
    print(f"\n--- {model_key} OC-DPO ---")
    model = AutoModelForCausalLM.from_pretrained(
        it_dir, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
    )
    model.config.use_cache = False

    ocdpo_targets = ["q_proj", "k_proj", "gate_proj", "up_proj"]
    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=ocdpo_targets,
        layers_to_transform=mid_layers,
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    print(f"  Training OC-DPO (exclude V/O/down, layers {mid_layers[0]}-{mid_layers[-1]})...")
    model = train_dpo(model, tokenizer, device)
    model.config.use_cache = True

    t0 = time.time()
    results["ocdpo"] = evaluate_safety(model, tokenizer, device, "OC-DPO")
    results["ocdpo"]["eval_time_s"] = round(time.time() - t0, 1)
    del model
    torch.cuda.empty_cache()

    return results


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    print(f"Step 20: OC-DPO Safety Evaluation")
    print(f"Device: {device}")

    all_results = {}

    for model_key in MODEL_CONFIGS:
        results = run_model(model_key, device=device)
        all_results[model_key] = results

        # Save per-model
        out_path = RESULTS_DIR / f"step20_ocdpo_safety_{model_key}.json"
        with open(out_path, "w") as f:
            json.dump({"model": model_key, "results": results}, f, indent=2)
        print(f"  Saved to {out_path}")

    # Cross-model summary
    print(f"\n{'='*70}")
    print(f"CROSS-MODEL OC-DPO SAFETY SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<16} {'IT':>8} {'Std-DPO':>10} {'OC-DPO':>10} {'Δ(IT→OC)':>10}")
    print("-" * 60)
    for mk, r in all_results.items():
        it_rate = r["it"]["overall_harmful_refusal"]
        std_rate = r["standard_dpo"]["overall_harmful_refusal"]
        oc_rate = r["ocdpo"]["overall_harmful_refusal"]
        delta = oc_rate - it_rate
        print(f"{mk:<16} {it_rate:>7.1%} {std_rate:>9.1%} {oc_rate:>9.1%} {delta:>+9.2%}")

    # Save combined
    combined_path = RESULTS_DIR / "step20_ocdpo_safety_combined.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved combined to {combined_path}")

    # W&B
    try:
        import wandb
        wandb.login()  # uses WANDB_API_KEY env var
        wandb.init(project="sar-alignment-tax", name="step20-ocdpo-safety")
        for mk, r in all_results.items():
            for variant in ["it", "standard_dpo", "ocdpo"]:
                wandb.log({f"{mk}/{variant}/overall_refusal": r[variant]["overall_harmful_refusal"]})
        wandb.finish()
    except Exception as e:
        print(f"W&B failed: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
