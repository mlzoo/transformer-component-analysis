"""
Step 21: MMLU 1000-Question Evaluation

Evaluate Base/IT/SAR-5% on 1000 MMLU questions using log-prob MC accuracy.
Addresses reviewer concern: "20 MMLU questions is completely insufficient."

Uses HuggingFace datasets to load MMLU, evaluates via last-token logprob for A/B/C/D.
"""

import sys
import json
import time
import torch
from pathlib import Path
from datasets import load_dataset

RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CONFIGS = {
    "qwen2.5-7b": {
        "base": "./models/Qwen2.5-7B",
        "it": "Qwen/Qwen2.5-7B-Instruct",
        "attribution": "attribution_qwen2.5-7b.json",
    },
    "llama-3.1-8b": {
        "base": "./models/Llama-3.1-8B",
        "it": "./models/Llama-3.1-8B-Instruct",
        "attribution": "attribution_llama-3.1-8b.json",
    },
    "mistral-7b": {
        "base": "./models/Mistral-7B-v0.3",
        "it": "./models/Mistral-7B-Instruct-v0.3",
        "attribution": "attribution_mistral-7b.json",
    },
    "yi-1.5-9b": {
        "base": "./models/Yi-1.5-9B",
        "it": "./models/Yi-1.5-9B-Chat",
        "attribution": "attribution_yi.json",
    },
}

# ============================================================
# Load MMLU questions
# ============================================================
def load_mmlu_questions(n=1000, seed=42):
    """Load n MMLU questions from HuggingFace datasets."""
    print(f"Loading MMLU dataset from HuggingFace...")
    # Load the test split of MMLU (all subjects)
    ds = load_dataset("cais/mmlu", "all", split="test", trust_remote_code=True)
    print(f"  Total MMLU test questions: {len(ds)}")

    # Shuffle and take n questions
    ds = ds.shuffle(seed=seed)
    if len(ds) > n:
        ds = ds.select(range(n))
    print(f"  Selected {len(ds)} questions")

    questions = []
    choice_labels = ["A", "B", "C", "D"]
    for item in ds:
        prompt = f"Q: {item['question']}\n"
        for i, choice in enumerate(item['choices']):
            prompt += f"{choice_labels[i]}) {choice}\n"
        prompt += "Answer:"
        target_idx = item['answer']
        target = f" {choice_labels[target_idx]}"
        questions.append({
            "prompt": prompt,
            "target": target,
            "subject": item.get("subject", "unknown"),
        })

    # Report subject distribution
    subjects = {}
    for q in questions:
        s = q["subject"]
        subjects[s] = subjects.get(s, 0) + 1
    print(f"  Subjects covered: {len(subjects)}")

    return questions


# ============================================================
# MC accuracy evaluation
# ============================================================
def compute_mc_accuracy(model, tokenizer, questions, device, label=""):
    """Multiple-choice accuracy using last-token logprobs."""
    correct = 0
    total = 0
    model.eval()
    t0 = time.time()

    with torch.no_grad():
        for i, q in enumerate(questions):
            inputs = tokenizer(q["prompt"], return_tensors="pt",
                             truncation=True, max_length=1024)
            input_ids = inputs["input_ids"].to(device)
            outputs = model(input_ids=input_ids)
            logits = outputs.logits[0, -1, :]  # Last token

            # Get logprobs for A, B, C, D
            choices = [" A", " B", " C", " D"]
            choice_ids = [tokenizer.encode(c, add_special_tokens=False)[-1] for c in choices]
            choice_logits = logits[choice_ids]
            predicted = choices[choice_logits.argmax().item()]

            if predicted.strip() == q["target"].strip():
                correct += 1
            total += 1

            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                acc = correct / total
                print(f"    [{label}] {i+1}/{len(questions)}  acc={acc:.1%}  ({elapsed:.0f}s)")

    elapsed = time.time() - t0
    acc = correct / total
    print(f"    [{label}] Final: {correct}/{total} = {acc:.1%}  ({elapsed:.0f}s)")
    return acc, correct, total, elapsed


# ============================================================
# SAR rollback
# ============================================================
def load_attribution_scores(model_key):
    attr_file = MODEL_CONFIGS[model_key]["attribution"]
    path = RESULTS_DIR / attr_file
    if not path.exists():
        print(f"  WARNING: attribution file not found: {path}")
        return {}
    with open(path) as f:
        data = json.load(f)
    scores = {}
    for c in data.get("components", []):
        name = c["name"]
        score = c.get("harm_score", c.get("mean_score", 0))
        scores[name] = score
    return scores


def apply_sar(model, base_dir, attribution_scores, k_pct=5):
    from safetensors import safe_open
    import glob

    sorted_comps = sorted(attribution_scores.items(), key=lambda x: x[1], reverse=True)
    n_rollback = max(1, int(len(sorted_comps) * k_pct / 100))
    rollback_names = {name for name, _ in sorted_comps[:n_rollback]}
    print(f"  SAR-{k_pct}%: rolling back {n_rollback}/{len(sorted_comps)} components")

    safetensor_files = sorted(glob.glob(str(Path(base_dir) / "*.safetensors")))
    rolled_back = 0
    model_state = dict(model.named_parameters())
    for sf_path in safetensor_files:
        with safe_open(sf_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in rollback_names and key in model_state:
                    base_tensor = f.get_tensor(key).to(
                        model_state[key].device, dtype=model_state[key].dtype
                    )
                    model_state[key].data.copy_(base_tensor)
                    rolled_back += 1
                    del base_tensor
    print(f"  Rolled back {rolled_back} tensors")
    return model


# ============================================================
# Main
# ============================================================
def run_model(model_key, questions, device="cuda:0"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = MODEL_CONFIGS[model_key]
    base_dir = config["base"]
    it_dir = config["it"]

    print(f"\n{'='*70}")
    print(f"  MMLU EVALUATION: {model_key} ({len(questions)} questions)")
    print(f"{'='*70}")

    results = {}

    # --- Base ---
    print(f"\n--- {model_key} Base ---")
    tokenizer = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_dir, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
    )
    acc, correct, total, elapsed = compute_mc_accuracy(
        model, tokenizer, questions, device, f"{model_key}/Base"
    )
    results["base"] = {"accuracy": round(acc, 4), "correct": correct, "total": total, "time_s": round(elapsed, 1)}
    del model
    torch.cuda.empty_cache()

    # --- IT ---
    print(f"\n--- {model_key} IT ---")
    it_tokenizer = AutoTokenizer.from_pretrained(it_dir, trust_remote_code=True)
    if it_tokenizer.pad_token is None:
        it_tokenizer.pad_token = it_tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        it_dir, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
    )
    acc, correct, total, elapsed = compute_mc_accuracy(
        model, tokenizer, questions, device, f"{model_key}/IT"
    )
    results["it"] = {"accuracy": round(acc, 4), "correct": correct, "total": total, "time_s": round(elapsed, 1)}
    del model
    torch.cuda.empty_cache()

    # --- SAR-5% ---
    print(f"\n--- {model_key} SAR-5% ---")
    attribution_scores = load_attribution_scores(model_key)
    if attribution_scores:
        model = AutoModelForCausalLM.from_pretrained(
            it_dir, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
        )
        model = apply_sar(model, base_dir, attribution_scores, k_pct=5)
        acc, correct, total, elapsed = compute_mc_accuracy(
            model, tokenizer, questions, device, f"{model_key}/SAR-5%"
        )
        results["sar_5pct"] = {"accuracy": round(acc, 4), "correct": correct, "total": total, "time_s": round(elapsed, 1)}
        del model
        torch.cuda.empty_cache()

    return results


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    n_questions = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    model_filter = sys.argv[3] if len(sys.argv) > 3 else None
    print(f"Step 21: MMLU {n_questions}-Question Evaluation")
    print(f"Device: {device}")

    questions = load_mmlu_questions(n=n_questions)

    models_to_run = [model_filter] if model_filter else list(MODEL_CONFIGS.keys())
    all_results = {}

    for model_key in models_to_run:
        results = run_model(model_key, questions, device=device)
        all_results[model_key] = results

        # Save per-model
        out_path = RESULTS_DIR / f"mmlu_{model_key}.json"
        with open(out_path, "w") as f:
            json.dump({"model": model_key, "n_questions": len(questions), "results": results}, f, indent=2)
        print(f"  Saved to {out_path}")

    # Cross-model summary
    print(f"\n{'='*70}")
    print(f"CROSS-MODEL MMLU SUMMARY ({n_questions} questions)")
    print(f"{'='*70}")
    print(f"{'Model':<16} {'Base':>8} {'IT':>8} {'SAR-5%':>8} {'Δ(IT→SAR)':>10}")
    print("-" * 55)
    for mk, r in all_results.items():
        base_acc = r["base"]["accuracy"]
        it_acc = r["it"]["accuracy"]
        sar_acc = r.get("sar_5pct", {}).get("accuracy", 0)
        delta = sar_acc - it_acc
        print(f"{mk:<16} {base_acc:>7.1%} {it_acc:>7.1%} {sar_acc:>7.1%} {delta:>+9.2%}")

    # Save combined
    combined = {
        "benchmark": "MMLU",
        "n_questions": n_questions,
        "results": all_results,
    }
    combined_path = RESULTS_DIR / "mmlu_combined.json"
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nSaved combined to {combined_path}")

    # W&B
    try:
        import wandb
        wandb.login()  # uses WANDB_API_KEY env var
        wandb.init(project="anonymous-submission", name=f"mmlu-{n_questions}")
        for mk, r in all_results.items():
            for variant in ["base", "it", "sar_5pct"]:
                if variant in r:
                    wandb.log({f"{mk}/{variant}/mmlu_acc": r[variant]["accuracy"]})
        wandb.finish()
    except Exception as e:
        print(f"W&B failed: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
