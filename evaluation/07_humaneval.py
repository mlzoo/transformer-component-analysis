"""
Step 18: HumanEval Code Generation Benchmark

Evaluates code generation capability using OpenAI's HumanEval benchmark (164 problems).
Measures pass@1 for Base, IT, and SAR-5% variants across 3 model families to quantify
the alignment tax on code generation.

Usage:
    python3 humaneval.py cuda:0
"""

import sys
import os
import json
import re
import math
import subprocess
import tempfile
import time
import gc
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb

# ============================================================
# Configuration
# ============================================================

RESULTS_DIR = Path("./results")

MODEL_CONFIGS = [
    {
        "name": "qwen2.5-7b",
        "base_dir": "./models/Qwen2.5-7B",
        "it_dir": "./models/Qwen2.5-7B-Instruct",
        "attribution_file": "./results/attribution_qwen2.5-7b.json",
    },
    {
        "name": "llama-3.1-8b",
        "base_dir": "./models/Llama-3.1-8B",
        "it_dir": "./models/Llama-3.1-8B-Instruct",
        "attribution_file": "./results/attribution_llama-3.1-8b.json",
    },
    {
        "name": "mistral-7b",
        "base_dir": "./models/Mistral-7B-v0.3",
        "it_dir": "./models/Mistral-7B-Instruct-v0.3",
        "attribution_file": "./results/attribution_mistral-7b.json",
    },
    {
        "name": "yi-1.5-9b",
        "base_dir": "./models/Yi-1.5-9B",
        "it_dir": "./models/Yi-1.5-9B-Chat",
        "attribution_file": "./results/attribution_yi.json",
    },
]

STOP_SEQUENCES = ["\ndef ", "\nclass ", "\n#", "\nif __name__"]

MAX_NEW_TOKENS = 512
EXEC_TIMEOUT = 5


# ============================================================
# Code extraction and execution
# ============================================================

def extract_code_from_response(response, prompt):
    """Extract function body from model response, handling markdown wrappers."""
    text = response

    # If response contains markdown code blocks, extract from them
    if "```python" in text:
        match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1)
    elif "```" in text:
        match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1)

    # If the extracted text contains the full function signature from the prompt,
    # we only want the body part (after the signature)
    # Get the last line of the prompt (typically the docstring closing or signature)
    prompt_lines = prompt.rstrip().split("\n")
    # Check if the response re-includes the function def
    if text.lstrip().startswith("def "):
        # The model regenerated the whole function; extract just the body
        # Find where the docstring ends (if any) and take from there
        # Actually, for HumanEval we want prompt + completion, so if the model
        # regenerated the whole function, we need to extract just the part after prompt
        # Try to find the prompt content in the response
        # Simplest: just use what's after the prompt's last meaningful line
        pass

    # Apply stop sequences to truncate
    for stop in STOP_SEQUENCES:
        idx = text.find(stop)
        if idx != -1:
            text = text[:idx]

    return text


def check_correctness(problem, completion, timeout=EXEC_TIMEOUT):
    """Execute the completed function against test cases in a subprocess."""
    code = problem["prompt"] + completion + "\n" + problem["test"] + f"\ncheck({problem['entry_point']})"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        f.flush()
        tmp_path = f.name
    try:
        result = subprocess.run(
            ["python3", tmp_path],
            capture_output=True,
            timeout=timeout,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ============================================================
# Generation
# ============================================================

def generate_completion(model, tokenizer, prompt_text, device):
    """Generate a code completion given a prompt."""
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Build stop token ids if possible (for early stopping)
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=1.0,  # temperature is ignored when do_sample=False
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    del outputs, input_ids, attention_mask
    torch.cuda.empty_cache()

    return response


def build_chat_prompt(tokenizer, prompt):
    """Build a chat-formatted prompt for IT/SAR models."""
    message = (
        "Complete the following Python function. Only output the function body, "
        "no explanation.\n\n" + prompt
    )
    messages = [{"role": "user", "content": message}]
    try:
        chat_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return chat_text
    except Exception:
        # Fallback if chat template not available
        return f"[INST] {message} [/INST]\n"


# ============================================================
# SAR implementation
# ============================================================

def load_attribution_scores(attribution_file):
    """Load attribution scores from JSON file."""
    path = Path(attribution_file)
    if not path.exists():
        print(f"  WARNING: Attribution file not found: {path}")
        return {}
    with open(path) as f:
        data = json.load(f)
    scores = {}
    for c in data.get("components", []):
        scores[c["name"]] = c.get("mean_score", 0)
    return scores


def apply_sar(model, base_dir, attribution_scores, k_pct=5):
    """Apply Surgical Alignment Reversal: replace top-k% most harmful IT weights with base weights."""
    from safetensors import safe_open
    import glob

    sorted_comps = sorted(attribution_scores.items(), key=lambda x: x[1], reverse=True)
    n_rollback = max(1, int(len(sorted_comps) * k_pct / 100))
    rollback_names = {name for name, _ in sorted_comps[:n_rollback]}
    print(f"  SAR-{k_pct}%: rolling back {n_rollback}/{len(sorted_comps)} components")

    safetensor_files = sorted(glob.glob(str(Path(base_dir) / "*.safetensors")))
    if not safetensor_files:
        # Try model subdirectory
        safetensor_files = sorted(glob.glob(str(Path(base_dir) / "**" / "*.safetensors"), recursive=True))

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
# Evaluation
# ============================================================

def evaluate_humaneval(model, tokenizer, device, problems, variant_label, is_base=False):
    """Evaluate pass@1 on HumanEval for a model variant."""
    model.eval()
    n_total = len(problems)
    n_pass = 0
    details = []

    for i, problem in enumerate(problems):
        task_id = problem["task_id"]
        prompt = problem["prompt"]
        entry_point = problem["entry_point"]

        # Build prompt
        if is_base:
            prompt_text = prompt
        else:
            prompt_text = build_chat_prompt(tokenizer, prompt)

        # Generate
        raw_response = generate_completion(model, tokenizer, prompt_text, device)

        # Extract code
        completion = extract_code_from_response(raw_response, prompt)

        # Check correctness
        passed = check_correctness(problem, completion)
        if passed:
            n_pass += 1

        details.append({
            "task_id": task_id,
            "entry_point": entry_point,
            "passed": passed,
            "completion_preview": completion[:200],
        })

        if (i + 1) % 20 == 0 or (i + 1) == n_total:
            print(f"    [{variant_label}] {i+1}/{n_total} -- running pass@1: {n_pass}/{i+1} ({n_pass/(i+1):.1%})")

    pass_at_1 = n_pass / max(n_total, 1)
    print(f"  [{variant_label}] Final pass@1: {n_pass}/{n_total} = {pass_at_1:.1%}")

    return {
        "pass_at_1": pass_at_1,
        "n_pass": n_pass,
        "n_total": n_total,
        "details": details,
    }


# ============================================================
# Main
# ============================================================

def run_model_family(config, device, problems):
    """Run HumanEval evaluation for one model family (Base, IT, SAR-5%)."""
    model_name = config["name"]
    base_dir = config["base_dir"]
    it_dir = config["it_dir"]
    attribution_file = config["attribution_file"]

    print(f"\n{'='*70}")
    print(f"  HUMANEVAL EVALUATION: {model_name}")
    print(f"{'='*70}")

    all_results = {}

    # --- Base model ---
    print(f"\n--- {model_name} BASE ---")
    tokenizer = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_dir, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
    )
    all_results["base"] = evaluate_humaneval(
        model, tokenizer, device, problems, f"{model_name}/Base", is_base=True
    )
    wandb.log({
        f"{model_name}/base_pass_at_1": all_results["base"]["pass_at_1"],
    })
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # --- IT model ---
    print(f"\n--- {model_name} IT ---")
    tokenizer = AutoTokenizer.from_pretrained(it_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        it_dir, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
    )
    all_results["it"] = evaluate_humaneval(
        model, tokenizer, device, problems, f"{model_name}/IT", is_base=False
    )
    wandb.log({
        f"{model_name}/it_pass_at_1": all_results["it"]["pass_at_1"],
    })
    del model
    # Keep tokenizer for SAR (uses IT tokenizer)
    gc.collect()
    torch.cuda.empty_cache()

    # --- SAR-5% ---
    attribution_scores = load_attribution_scores(attribution_file)
    if attribution_scores:
        print(f"\n--- {model_name} SAR-5% ---")
        model = AutoModelForCausalLM.from_pretrained(
            it_dir, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
        )
        model = apply_sar(model, base_dir, attribution_scores, k_pct=5)
        all_results["sar_5pct"] = evaluate_humaneval(
            model, tokenizer, device, problems, f"{model_name}/SAR-5%", is_base=False
        )
        wandb.log({
            f"{model_name}/sar5_pass_at_1": all_results["sar_5pct"]["pass_at_1"],
        })
        del model
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print(f"  Skipping SAR-5% for {model_name}: no attribution scores found")

    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # --- Summary for this model ---
    print(f"\n{'-'*50}")
    print(f"  HUMANEVAL SUMMARY: {model_name}")
    print(f"{'-'*50}")
    print(f"  {'Variant':<12} {'pass@1':>8} {'Passed':>8} {'Total':>6}")
    for vname, vres in all_results.items():
        print(f"  {vname:<12} {vres['pass_at_1']:>7.1%} {vres['n_pass']:>8} {vres['n_total']:>6}")

    if "base" in all_results and "it" in all_results:
        tax = all_results["it"]["pass_at_1"] - all_results["base"]["pass_at_1"]
        print(f"\n  Alignment tax (IT - Base): {tax:+.1%}")
        if "sar_5pct" in all_results:
            recovery = all_results["sar_5pct"]["pass_at_1"] - all_results["it"]["pass_at_1"]
            print(f"  SAR-5% recovery (SAR - IT): {recovery:+.1%}")

    # Save per-model results
    output = {
        "analysis": "HumanEval code generation benchmark",
        "model": model_name,
        "n_problems": len(problems),
        "results": {k: {kk: vv for kk, vv in v.items() if kk != "details"} for k, v in all_results.items()},
        "details": {k: v["details"] for k, v in all_results.items()},
    }
    out_path = RESULTS_DIR / f"humaneval_{model_name}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Saved to {out_path}")

    return all_results


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    model_filter = sys.argv[2] if len(sys.argv) > 2 else None
    print(f"Device: {device}")

    # W&B init
    # Set WANDB_API_KEY environment variable before running
    configs_to_run = [c for c in MODEL_CONFIGS if model_filter is None or c["name"] == model_filter]
    wandb.init(
        project="anonymous-submission",
        name=f"humaneval{'-' + model_filter if model_filter else ''}",
        config={
            "benchmark": "HumanEval",
            "n_problems": 164,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": 0,
            "do_sample": False,
            "exec_timeout": EXEC_TIMEOUT,
            "device": device,
            "models": [c["name"] for c in configs_to_run],
            "variants": ["base", "it", "sar_5pct"],
        },
    )

    # Load HumanEval dataset
    print("Loading HumanEval dataset...")
    dataset = load_dataset("openai/openai_humaneval", split="test")
    problems = list(dataset)
    print(f"Loaded {len(problems)} problems")

    # Run all model families
    combined_results = {}
    for config in configs_to_run:
        model_results = run_model_family(config, device, problems)
        combined_results[config["name"]] = {
            k: {kk: vv for kk, vv in v.items() if kk != "details"}
            for k, v in model_results.items()
        }

    # ============================================================
    # Combined summary
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  COMBINED HUMANEVAL RESULTS")
    print(f"{'='*70}")
    print(f"  {'Model':<16} {'Base':>8} {'IT':>8} {'SAR-5%':>8} {'Tax':>8} {'Recovery':>10}")
    print(f"  {'-'*60}")

    for mname, mres in combined_results.items():
        base_p1 = mres.get("base", {}).get("pass_at_1", float("nan"))
        it_p1 = mres.get("it", {}).get("pass_at_1", float("nan"))
        sar_p1 = mres.get("sar_5pct", {}).get("pass_at_1", float("nan"))
        tax = it_p1 - base_p1 if not (math.isnan(base_p1) or math.isnan(it_p1)) else float("nan")
        rec = sar_p1 - it_p1 if not (math.isnan(sar_p1) or math.isnan(it_p1)) else float("nan")

        base_s = f"{base_p1:.1%}" if not math.isnan(base_p1) else "N/A"
        it_s = f"{it_p1:.1%}" if not math.isnan(it_p1) else "N/A"
        sar_s = f"{sar_p1:.1%}" if not math.isnan(sar_p1) else "N/A"
        tax_s = f"{tax:+.1%}" if not math.isnan(tax) else "N/A"
        rec_s = f"{rec:+.1%}" if not math.isnan(rec) else "N/A"

        print(f"  {mname:<16} {base_s:>8} {it_s:>8} {sar_s:>8} {tax_s:>8} {rec_s:>10}")

    # Log combined to W&B
    for mname, mres in combined_results.items():
        for variant, vres in mres.items():
            wandb.summary[f"{mname}/{variant}/pass_at_1"] = vres.get("pass_at_1", None)

    # Save combined results
    combined_output = {
        "analysis": "HumanEval code generation benchmark (combined)",
        "n_problems": len(problems),
        "models": [c["name"] for c in MODEL_CONFIGS],
        "results": combined_results,
    }
    combined_path = RESULTS_DIR / "humaneval_combined.json"
    with open(combined_path, "w") as f:
        json.dump(combined_output, f, indent=2, default=str)
    print(f"\nSaved combined results to {combined_path}")

    wandb.finish()
    print("\nDone.")


if __name__ == "__main__":
    main()
