"""
Scaling Validation on Qwen2.5-14B

Validates output-pathway concentration (V/O > Q/K harm) and OC-DPO
effectiveness on a 14B model. This extends the 7B findings from steps 1/8/6.

Key questions:
  1. Does V/O > Q/K hold at 14B scale?
  2. Does OC-DPO still reduce alignment tax at 14B?

Usage:
  python scaling_14b.py                    # fp16 on 2+ GPUs (device_map=auto)
  python scaling_14b.py --device cuda:3    # 4-bit on single GPU
  python scaling_14b.py --num-examples 209 # use all examples (default: 209)
  python scaling_14b.py --attribution-only # skip OC-DPO, just run attribution

Requirements:
  - fp16 mode: ~2x 24GB GPUs (14B fp16 ~28GB, uses device_map="auto")
  - 4-bit mode: 1x 24GB GPU (~8GB)
"""

import sys
import json
import time
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from safetensors import safe_open
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from agent_examples_200 import AGENT_EXAMPLES_200 as ALL_EXAMPLES  # 209 examples

# OC-DPO targets (same as 05_ocdpo.py)
ALL_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
EVAL_EXAMPLES = ALL_EXAMPLES[:209]

# DPO training data (structured-generation preference pairs)
TRAIN_DATA = [
    {"prompt": "Tools: search(q)\nUser: Capital of France?\nThought: Search.\nAction: ", "chosen": 'search(q="capital of France")', "rejected": "The capital of France is Paris."},
    {"prompt": 'JSON: Name=Alice, Age=30\n\n{"', "chosen": '"name": "Alice", "age": 30}', "rejected": "Alice is 30 years old."},
    {"prompt": "SQL: Users where age > 25\n\nSELECT ", "chosen": "* FROM users WHERE age > 25;", "rejected": "I would select users older than 25."},
    {"prompt": "API: Delete user 42\n\n", "chosen": "DELETE /api/users/42", "rejected": "To delete user 42, make an API call."},
    {"prompt": "```python\\ndef add(a, b):\\n    ", "chosen": "return a + b", "rejected": "This adds two numbers."},
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
    {"prompt": "Regex: Match emails\n\n", "chosen": "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}", "rejected": "Use a regex pattern for emails."},
    {"prompt": "CSS: Center div\n\n.box {\n  ", "chosen": "display: flex;\n  justify-content: center;", "rejected": "Use flexbox to center."},
    {"prompt": "git: Undo last commit\n\n$ ", "chosen": "git reset --soft HEAD~1", "rejected": "You can undo with git reset."},
    {"prompt": 'CI: 10/10 pass\n\n{"', "chosen": '"status": "pass", "deploy": true}', "rejected": "All tests passed."},
    {"prompt": 'Terraform: EC2\n\nresource "aws_instance" "web" {\n  ', "chosen": 'ami = "ami-abc"\n  instance_type = "t2.micro"', "rejected": "Create an EC2 resource."},
]

RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "Qwen/Qwen2.5-14B"
IT_MODEL = "Qwen/Qwen2.5-14B-Instruct"


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def get_model_device(model):
    """Get the device of the first parameter (for input placement with device_map='auto')."""
    try:
        return model.device
    except AttributeError:
        return next(model.parameters()).device


def compute_loss(model, tokenizer, prompt, target, device):
    """Compute cross-entropy loss on target tokens only."""
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


def compute_agent_loss(model, tokenizer, examples, device):
    """Compute mean agent loss across examples (token-weighted)."""
    total_loss = 0
    total_tokens = 0
    model.eval()
    with torch.no_grad():
        for ex in examples:
            target = ex.get("target", ex.get("chosen", ""))
            full_text = ex["prompt"] + target
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


def classify_component(name):
    """Classify a parameter name into component type and layer index."""
    if "layers." not in name:
        return "other", -1
    parts = name.split(".")
    layer_idx = None
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                layer_idx = int(parts[i + 1])
            except ValueError:
                pass
    if layer_idx is None:
        return "other", -1
    if "q_proj" in name:
        return "W_Q", layer_idx
    elif "k_proj" in name:
        return "W_K", layer_idx
    elif "v_proj" in name:
        return "W_V", layer_idx
    elif "o_proj" in name:
        return "W_O", layer_idx
    elif "gate_proj" in name:
        return "W_gate", layer_idx
    elif "up_proj" in name:
        return "W_up", layer_idx
    elif "down_proj" in name:
        return "W_down", layer_idx
    return "other", layer_idx


# ---------------------------------------------------------------------------
# Phase 1: Attribution via activation patching
# ---------------------------------------------------------------------------

def run_attribution(wandb_run=None, num_examples=209, all_components=True):
    """Run attribution on agent examples using activation patching."""
    import glob
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\n" + "=" * 70)
    print("PHASE 1: Attribution via Activation Patching (14B)")
    print(f"  num_examples={num_examples}, all_components={all_components}")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load IT model with auto device map for multi-GPU (fp16, same as 7B experiments)
    print("Loading Qwen2.5-14B-Instruct in fp16 (device_map=auto)...")
    model = AutoModelForCausalLM.from_pretrained(
        IT_MODEL, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    device = get_model_device(model)
    print(f"  Model loaded. Input device: {device}")
    print(f"  Num layers: {model.config.num_hidden_layers}")

    # Index base model safetensors
    from huggingface_hub import snapshot_download
    base_dir = snapshot_download(BASE_MODEL)
    base_safetensors = sorted(glob.glob(str(Path(base_dir) / "*.safetensors")))
    print(f"  Base model safetensors: {len(base_safetensors)} files")

    it_params = dict(model.named_parameters())

    # Build component index
    components = []
    for sf_path in base_safetensors:
        with safe_open(sf_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in it_params and "weight" in key:
                    comp_type, layer = classify_component(key)
                    if comp_type != "other":
                        components.append({
                            "name": key,
                            "type": comp_type,
                            "layer": layer,
                            "sf_path": sf_path,
                        })

    print(f"  Found {len(components)} weight components")

    examples = ALL_EXAMPLES[:num_examples]
    if not all_components:
        # Attention only (legacy mode)
        components = [c for c in components if c["type"] in ("W_V", "W_O", "W_Q", "W_K")]
        print(f"  Filtered to {len(components)} attention components (V/O/Q/K only)")
    else:
        print(f"  Using all {len(components)} components (attention + MLP)")
    print(f"  Computing baseline losses on {len(examples)} examples...")
    baseline_losses = []
    for i, ex in enumerate(examples):
        loss = compute_loss(model, tokenizer, ex["prompt"],
                            ex.get("target", ex.get("chosen", "")), device)
        baseline_losses.append(loss)
        if (i + 1) % 50 == 0:
            print(f"    Baseline: {i + 1}/{len(examples)}")

    valid_examples = [(i, ex, bl) for i, (ex, bl) in enumerate(zip(examples, baseline_losses))
                      if bl is not None]
    print(f"  Valid examples: {len(valid_examples)}/{len(examples)}")

    if wandb_run:
        wandb_run.log({"attribution/num_components": len(components),
                       "attribution/num_valid_examples": len(valid_examples)})

    # For each component, measure harm via activation patching
    t0 = time.time()
    for ci, comp in enumerate(components):
        if (ci + 1) % 20 == 0 or ci == 0:
            elapsed = time.time() - t0
            eta = elapsed / (ci + 1) * (len(components) - ci - 1) if ci > 0 else 0
            print(f"  Component {ci + 1}/{len(components)}: {comp['name']} "
                  f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]")

        target_param = it_params[comp["name"]]
        param_device = target_param.device

        # Handle CPU-offloaded (meta) tensors
        if str(param_device) == "meta":
            comp["mean_score"] = 0.0
            comp["std_score"] = 0.0
            comp["delta_norm"] = 0.0
            comp["n_valid"] = 0
            continue

        original_data = target_param.data.clone()

        # Load base weight for this component
        with safe_open(comp["sf_path"], framework="pt", device="cpu") as f:
            base_w = f.get_tensor(comp["name"]).to(param_device, dtype=target_param.dtype)

        scores = []
        for vi, ex, bl in valid_examples:
            # Patch: revert to base weight (remove alignment delta)
            target_param.data.copy_(base_w)
            patched_loss = compute_loss(model, tokenizer, ex["prompt"],
                                        ex.get("target", ex.get("chosen", "")), device)
            # Restore IT weight
            target_param.data.copy_(original_data)

            if patched_loss is not None:
                # Positive = harmful (removing alignment change DECREASES loss)
                scores.append(bl - patched_loss)

        comp["mean_score"] = float(np.mean(scores)) if scores else 0.0
        comp["std_score"] = float(np.std(scores)) if scores else 0.0
        # Compute delta norm on CPU to avoid meta tensor issues with offloaded params
        try:
            comp["delta_norm"] = float((original_data.cpu().float() - base_w.cpu().float()).norm().item())
        except RuntimeError:
            comp["delta_norm"] = 0.0  # Skip if param is on meta device
        comp["n_valid"] = len(scores)

        del base_w, original_data
        torch.cuda.empty_cache()

        if wandb_run and (ci + 1) % 20 == 0:
            wandb_run.log({
                f"attribution/component_{ci}": comp["mean_score"],
                "attribution/progress": (ci + 1) / len(components),
            })

    elapsed_total = time.time() - t0
    print(f"  Attribution complete in {elapsed_total:.0f}s")

    # Compute V/O vs Q/K statistics
    vo_scores = []
    qk_scores = []
    type_scores = defaultdict(list)
    for comp in components:
        type_scores[comp["type"]].append(comp["mean_score"])
        if comp["type"] in ("W_V", "W_O"):
            vo_scores.append(comp["mean_score"])
        elif comp["type"] in ("W_Q", "W_K"):
            qk_scores.append(comp["mean_score"])

    vo_mean = float(np.mean(vo_scores))
    qk_mean = float(np.mean(qk_scores))
    stat, p_value = stats.mannwhitneyu(vo_scores, qk_scores, alternative='greater')

    total_harm = sum(max(0, np.mean(s)) for s in type_scores.values())
    type_shares = {}
    for t in ["W_Q", "W_K", "W_V", "W_O", "W_gate", "W_up", "W_down"]:
        if t in type_scores:
            mean = np.mean(type_scores[t])
            share = max(0, mean) / total_harm * 100 if total_harm > 0 else 0
            type_shares[t] = float(share)

    vo_share = sum(type_shares.get(t, 0) for t in ["W_V", "W_O"])
    qk_share = sum(type_shares.get(t, 0) for t in ["W_Q", "W_K"])
    vo_qk_ratio = vo_mean / qk_mean if qk_mean > 0 else float('inf')

    print(f"\n{'=' * 60}")
    print(f"V/O vs Q/K RESULTS (14B, {len(examples)} examples)")
    print(f"{'=' * 60}")
    print(f"V/O mean harm: {vo_mean:.6f} (n={len(vo_scores)})")
    print(f"Q/K mean harm: {qk_mean:.6f} (n={len(qk_scores)})")
    print(f"V/O / Q/K ratio: {vo_qk_ratio:.2f}x")
    print(f"Mann-Whitney p-value: {p_value:.6e}")
    print(f"V/O share: {vo_share:.1f}%, Q/K share: {qk_share:.1f}%")

    print(f"\nHarm hierarchy:")
    for t in ["W_down", "W_up", "W_gate", "W_O", "W_V", "W_Q", "W_K"]:
        if t in type_shares:
            print(f"  {t:8s}: mean={np.mean(type_scores[t]):.6f}, share={type_shares[t]:.1f}%")

    if wandb_run:
        wandb_run.log({
            "attribution/vo_mean": vo_mean,
            "attribution/qk_mean": qk_mean,
            "attribution/vo_qk_ratio": vo_qk_ratio,
            "attribution/mann_whitney_p": p_value,
            "attribution/vo_share": vo_share,
            "attribution/qk_share": qk_share,
            "attribution/total_components": len(components),
            "attribution/elapsed_seconds": elapsed_total,
        })
        for t, share in type_shares.items():
            wandb_run.log({f"attribution/share_{t}": share})

    # Clean up
    del model
    torch.cuda.empty_cache()

    return {
        "num_examples": len(examples),
        "num_valid": len(valid_examples),
        "num_components": len(components),
        "vo_mean": vo_mean,
        "qk_mean": qk_mean,
        "vo_qk_ratio": vo_qk_ratio,
        "mann_whitney_p": float(p_value),
        "mann_whitney_stat": float(stat),
        "vo_share": vo_share,
        "qk_share": qk_share,
        "type_shares": type_shares,
        "elapsed_seconds": elapsed_total,
        "components": [
            {"name": c["name"], "type": c["type"], "layer": c["layer"],
             "mean_score": c["mean_score"], "std_score": c["std_score"],
             "delta_norm": c["delta_norm"], "n_valid": c["n_valid"]}
            for c in components
        ],
    }


# ---------------------------------------------------------------------------
# Phase 2: OC-DPO with selective LoRA
# ---------------------------------------------------------------------------

CONDITION_TARGETS = {
    "standard": ALL_TARGETS,  # q,k,v,o,gate,up,down
    "ocdpo_exclude_output": ["q_proj", "k_proj", "gate_proj", "up_proj"],  # exclude v,o,down
    "exclude_qk": ["v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # exclude q,k (control)
}


def get_mid_layer_targets(num_layers):
    """Return list of layer indices for mid-third."""
    mid_start = num_layers // 3
    mid_end = 2 * num_layers // 3
    return list(range(mid_start, mid_end))


def run_ocdpo_condition(condition, tokenizer, wandb_run=None, num_epochs=5, lr=5e-5):
    """Train one DPO condition with selective LoRA on 14B."""
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    print(f"\n{'=' * 60}")
    print(f"OC-DPO Condition: {condition}")
    print(f"{'=' * 60}")

    # Load fresh base model each condition (fp16 on 2 GPUs)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True
    )
    model.config.use_cache = False
    device = get_model_device(model)

    num_layers = model.config.num_hidden_layers
    mid_layers = get_mid_layer_targets(num_layers)
    targets = CONDITION_TARGETS[condition]

    # Apply LoRA only to mid-third layers
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=targets,
        layers_to_transform=mid_layers,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  LoRA targets: {targets}")
    print(f"  LoRA on layers: {mid_layers[0]}-{mid_layers[-1]} ({len(mid_layers)} layers)")
    print(f"  Trainable: {trainable / 1e6:.1f}M / {total / 1e6:.1f}M ({100 * trainable / total:.3f}%)")

    # Pre-training eval on 49 eval examples
    pre_loss = compute_agent_loss(model, tokenizer, EVAL_EXAMPLES, device)
    print(f"  Pre-train agent loss: {pre_loss:.4f}")

    if wandb_run:
        wandb_run.log({f"ocdpo/{condition}/pre_loss": pre_loss})

    # DPO training
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

            # Forward chosen
            chosen_logits = model(input_ids=chosen_ids).logits
            chosen_lps = torch.nn.functional.log_softmax(
                chosen_logits[0, prompt_len - 1:-1, :].float(), dim=-1)
            chosen_logp = chosen_lps.gather(
                1, chosen_ids[0, prompt_len:].unsqueeze(1)).squeeze(1).sum()

            # Forward rejected
            rejected_logits = model(input_ids=rejected_ids).logits
            rejected_lps = torch.nn.functional.log_softmax(
                rejected_logits[0, prompt_len - 1:-1, :].float(), dim=-1)
            rejected_logp = rejected_lps.gather(
                1, rejected_ids[0, prompt_len:].unsqueeze(1)).squeeze(1).sum()

            # DPO loss (beta=0.1)
            loss = -torch.nn.functional.logsigmoid(0.1 * (chosen_logp - rejected_logp))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            torch.cuda.empty_cache()

        avg_epoch_loss = epoch_loss / len(TRAIN_DATA)
        print(f"  Epoch {epoch + 1}/{num_epochs}: DPO loss={avg_epoch_loss:.4f}")

        if wandb_run:
            wandb_run.log({f"ocdpo/{condition}/epoch_{epoch + 1}_dpo_loss": avg_epoch_loss})

    # Post-training eval
    post_loss = compute_agent_loss(model, tokenizer, EVAL_EXAMPLES, device)
    loss_change = post_loss - pre_loss
    print(f"  Post-train agent loss: {post_loss:.4f}")
    print(f"  Agent loss change: {loss_change:+.4f}")

    if wandb_run:
        wandb_run.log({
            f"ocdpo/{condition}/post_loss": post_loss,
            f"ocdpo/{condition}/loss_change": loss_change,
            f"ocdpo/{condition}/trainable_M": trainable / 1e6,
        })

    del model, optimizer
    torch.cuda.empty_cache()

    return {
        "condition": condition,
        "lora_targets": targets,
        "mid_layers": [mid_layers[0], mid_layers[-1]],
        "num_mid_layers": len(mid_layers),
        "pre_loss": float(pre_loss),
        "post_loss": float(post_loss),
        "loss_change": float(loss_change),
        "trainable_params_M": trainable / 1e6,
        "total_params_M": total / 1e6,
    }


def run_ocdpo(wandb_run=None):
    """Run all 3 OC-DPO conditions."""
    from transformers import AutoTokenizer

    print("\n" + "=" * 70)
    print("PHASE 2: OC-DPO Selective LoRA (14B)")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    conditions = ["standard", "ocdpo_exclude_output", "exclude_qk"]
    results = {}

    for condition in conditions:
        result = run_ocdpo_condition(condition, tokenizer, wandb_run=wandb_run)
        results[condition] = result

    # Summary
    print(f"\n{'=' * 60}")
    print(f"OC-DPO SUMMARY (14B)")
    print(f"{'=' * 60}")

    std_change = results["standard"]["loss_change"]
    print(f"Standard DPO (all targets) agent loss change: {std_change:+.4f}")
    print(f"(Positive = worse agent performance = alignment tax)\n")

    for name, r in results.items():
        if std_change != 0:
            tax_reduction = (std_change - r["loss_change"]) / abs(std_change) * 100
        else:
            tax_reduction = 0
        print(f"  {name:25s}: change={r['loss_change']:+.4f}, "
              f"tax reduction: {tax_reduction:.1f}%, targets={r['lora_targets']}")

    print(f"\n--- KEY HYPOTHESIS TEST ---")
    if std_change > 0:
        ocdpo_change = results["ocdpo_exclude_output"]["loss_change"]
        qk_change = results["exclude_qk"]["loss_change"]
        ocdpo_reduction = (std_change - ocdpo_change) / std_change * 100
        qk_reduction = (std_change - qk_change) / std_change * 100
        print(f"  Standard DPO tax:     {std_change:+.4f}")
        print(f"  OC-DPO tax:           {ocdpo_change:+.4f} (exclude V/O/down)")
        print(f"  Exclude Q/K tax:      {qk_change:+.4f} (control)")
        print(f"  OC-DPO reduces tax by {ocdpo_reduction:.1f}%")
        print(f"  Excl Q/K reduces by   {qk_reduction:.1f}%")
    else:
        print(f"  No alignment tax observed (loss change = {std_change:+.4f})")

    if wandb_run:
        wandb_run.log({
            "ocdpo/standard_loss_change": std_change,
            "ocdpo/ocdpo_loss_change": results["ocdpo_exclude_output"]["loss_change"],
            "ocdpo/exclude_qk_loss_change": results["exclude_qk"]["loss_change"],
        })
        if std_change > 0:
            wandb_run.log({
                "ocdpo/ocdpo_tax_reduction_pct": (std_change - results["ocdpo_exclude_output"]["loss_change"]) / std_change * 100,
                "ocdpo/exclude_qk_tax_reduction_pct": (std_change - results["exclude_qk"]["loss_change"]) / std_change * 100,
            })

    return {
        "conditions": results,
        "standard_loss_change": std_change,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--attribution-only", action="store_true",
                        help="Only run attribution, skip OC-DPO")
    parser.add_argument("--num-examples", type=int, default=209,
                        help="Number of agent examples for attribution (default: all 209)")
    parser.add_argument("--attention-only", action="store_true",
                        help="Only score attention components (V/O/Q/K), skip MLP")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable W&B logging")
    args = parser.parse_args()

    print("=" * 70)
    print("Scaling Validation - Qwen2.5-14B (FIXED: correct harm sign)")
    print("=" * 70)
    print(f"Base model:     {BASE_MODEL}")
    print(f"IT model:       {IT_MODEL}")
    print(f"Attribution:    {args.num_examples} agent examples, "
          f"{'attention only' if args.attention_only else 'all components'}")
    print(f"OC-DPO:         {'SKIP' if args.attribution_only else '3 conditions'}")
    print(f"Precision:      fp16")
    print(f"Device map:     auto (multi-GPU)")
    print()

    # W&B setup
    run = None
    if not args.no_wandb:
        import wandb
        run = wandb.init(
            project="anonymous-submission",
            name="scaling-14b",
            config={
                "base_model": BASE_MODEL,
                "it_model": IT_MODEL,
                "num_attribution_examples": args.num_examples,
                "all_components": not args.attention_only,
                "attribution_only": args.attribution_only,
                "precision": "fp16",
                "device_map": "auto",
                "fix": "corrected harm score sign (bl - patched, not patched - bl)",
            },
        )

    all_results = {
        "experiment": "scaling_14b",
        "base_model": BASE_MODEL,
        "it_model": IT_MODEL,
        "fix_note": "v2: corrected harm score sign (bl - patched_loss)",
    }

    # Phase 1: Attribution
    t_start = time.time()
    attribution_results = run_attribution(
        wandb_run=run,
        num_examples=args.num_examples,
        all_components=not args.attention_only,
    )
    all_results["attribution"] = attribution_results

    # Phase 2: OC-DPO (optional)
    if not args.attribution_only:
        ocdpo_results = run_ocdpo(wandb_run=run)
        all_results["ocdpo"] = ocdpo_results

    total_time = time.time() - t_start
    all_results["total_elapsed_seconds"] = total_time

    # Final summary
    print(f"\n{'=' * 70}")
    print(f"FINAL SUMMARY - Step 12 Scaling Validation (14B) [FIXED v2]")
    print(f"{'=' * 70}")
    print(f"Total time: {total_time / 60:.1f} minutes")
    print(f"\nAttribution ({args.num_examples} examples):")
    print(f"  V/O mean harm: {attribution_results['vo_mean']:.6f}")
    print(f"  Q/K mean harm: {attribution_results['qk_mean']:.6f}")
    print(f"  V/O / Q/K ratio: {attribution_results['vo_qk_ratio']:.2f}x")
    print(f"  Mann-Whitney p: {attribution_results['mann_whitney_p']:.2e}")
    print(f"  V/O share: {attribution_results['vo_share']:.1f}%")
    print(f"  Q/K share: {attribution_results['qk_share']:.1f}%")

    if not args.attribution_only:
        print(f"\nOC-DPO:")
        for name, r in all_results["ocdpo"]["conditions"].items():
            print(f"  {name:25s}: loss_change={r['loss_change']:+.4f}")

    if run:
        run.log({
            "total_elapsed_minutes": total_time / 60,
            "final/vo_qk_ratio": attribution_results["vo_qk_ratio"],
            "final/mann_whitney_p": attribution_results["mann_whitney_p"],
            "final/vo_share": attribution_results["vo_share"],
            "final/qk_share": attribution_results["qk_share"],
        })

    # Save results (v2 filename to not overwrite original)
    out_path = RESULTS_DIR / "scaling_14b.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    if run:
        run.finish()
        print("W&B run finished.")


if __name__ == "__main__":
    main()
