"""
OGPSA + Output-Pathway Hybrid

Reviewer suggestion: restrict OGPSA's gradient projection to only output-pathway
components (V/O/W_down), letting Q/K/gate/up train with standard LoRA.

This tests whether combining mechanistic insight (which components matter) with
OGPSA's soft constraint (gradient projection) outperforms standard OGPSA.

Conditions:
  1. ogpsa_standard: OGPSA on all 7 projections (baseline, same as 07_ogpsa_comparison)
  2. ogpsa_hybrid: OGPSA projection on V/O/W_down only, standard LoRA on Q/K/gate/up
  3. ocdpo: Hard exclusion of V/O/W_down (from previous OC-DPO experiment/19, for reference)

Usage: python 13_ogpsa_hybrid.py cuda:0
"""

import sys
import json
import time
import torch
import numpy as np
from pathlib import Path
from safetensors import safe_open

RESULTS_DIR = Path("./results")

sys.path.insert(0, "./experiments")
import importlib
_ocdpo = importlib.import_module("05_ocdpo")
EVAL_EXAMPLES = _ocdpo.EVAL_EXAMPLES
compute_agent_loss = _ocdpo.compute_agent_loss
ALL_TARGETS = _ocdpo.ALL_TARGETS
get_mid_layer_targets = _ocdpo.get_mid_layer_targets

MODEL_CONFIGS = {
    "qwen2.5-7b": {
        "base": "./models/Qwen2.5-7B",
        "it": "./models/Qwen2.5-7B-Instruct",
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

OUTPUT_PATHWAY = ["v_proj", "o_proj", "down_proj"]

NUM_PAIRS = 496
NUM_EPOCHS = 3
LR = 5e-5
LORA_R = 16
LORA_ALPHA = 32
DPO_BETA = 0.1
MAX_SEQ_LEN = 512
GRAD_ACCUM_STEPS = 4
SUBSPACE_RANK = 64

STRUCTURED_KEYWORDS = [
    "json", "api", "code", "function", "sql", "python", "javascript",
    "command", "script", "program", "database", "query", "format",
    "output", "parse", "generate", "template", "structure", "schema",
    "tool", "agent", "step", "instruction", "task", "workflow",
    "data", "extract", "convert", "transform", "yaml", "xml", "html",
    "csv", "table", "list", "array", "object", "class", "method",
    "algorithm", "implement", "write", "create", "build", "develop",
]


def load_ultrafeedback_pairs(num_pairs=496, seed=42):
    from datasets import load_dataset
    print(f"Loading UltraFeedback dataset...")
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
    print(f"  Total examples: {len(ds)}")

    scored = []
    for i, ex in enumerate(ds):
        prompt = ex["prompt"].lower()
        chosen_text = ex["chosen"][1]["content"] if len(ex["chosen"]) > 1 else ""
        combined = prompt + " " + chosen_text.lower()
        score = sum(1 for kw in STRUCTURED_KEYWORDS if kw in combined)
        score_gap = ex["score_chosen"] - ex["score_rejected"]
        if score_gap >= 2.0: score += 3
        elif score_gap >= 1.0: score += 1
        if len(chosen_text) > 2000 or len(chosen_text) < 20: score -= 5
        scored.append((score, i))

    scored.sort(key=lambda x: -x[0])
    rng = np.random.RandomState(seed)
    top_candidates = scored[:num_pairs * 2]
    rng.shuffle(top_candidates)
    selected_indices = [idx for _, idx in top_candidates[:num_pairs]]

    pairs = []
    for idx in selected_indices:
        ex = ds[idx]
        prompt = ex["prompt"]
        chosen_text = ex["chosen"][1]["content"] if len(ex["chosen"]) > 1 else ""
        rejected_text = ex["rejected"][1]["content"] if len(ex["rejected"]) > 1 else ""
        if not chosen_text or not rejected_text:
            continue
        pairs.append({"prompt": prompt, "chosen": chosen_text, "rejected": rejected_text,
                       "score_gap": ex["score_chosen"] - ex["score_rejected"]})

    pairs = pairs[:num_pairs]
    print(f"  Selected {len(pairs)} preference pairs")
    return pairs


def compute_capability_subspace(base_dir, it_dir, mid_layers, target_types, rank=SUBSPACE_RANK):
    """Compute capability subspace via SVD — only for specified target_types."""
    print(f"Computing capability subspace for targets: {target_types}")
    t0 = time.time()

    base_index = {}
    for f in sorted(Path(base_dir).glob("*.safetensors")):
        with safe_open(str(f), framework="pt", device="cpu") as sf:
            for key in sf.keys():
                base_index[key] = str(f)

    it_index = {}
    for f in sorted(Path(it_dir).glob("*.safetensors")):
        with safe_open(str(f), framework="pt", device="cpu") as sf:
            for key in sf.keys():
                it_index[key] = str(f)

    subspaces = {}
    common_keys = set(base_index.keys()) & set(it_index.keys())

    relevant_keys = []
    for k in common_keys:
        if not any(t in k for t in target_types):
            continue
        if "weight" not in k:
            continue
        for layer_idx in mid_layers:
            if f"layers.{layer_idx}." in k:
                relevant_keys.append(k)
                break

    relevant_keys.sort()
    print(f"  Computing SVD for {len(relevant_keys)} weight matrices...")

    for i, key in enumerate(relevant_keys):
        with safe_open(base_index[key], framework="pt", device="cpu") as f:
            base_w = f.get_tensor(key).float()
        with safe_open(it_index[key], framework="pt", device="cpu") as f:
            it_w = f.get_tensor(key).float()

        delta = it_w - base_w
        del base_w, it_w
        actual_rank = min(rank, min(delta.shape) - 1)
        if actual_rank <= 0:
            del delta
            continue

        try:
            U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
            subspaces[key] = {
                "U": U[:, :actual_rank].clone(),
                "V": Vh[:actual_rank, :].clone(),
                "S": S[:actual_rank].clone(),
            }
            del U, S, Vh
        except Exception as e:
            print(f"  SVD failed for {key}: {e}")
        del delta

        if (i + 1) % 20 == 0:
            print(f"    SVD progress: {i+1}/{len(relevant_keys)}")

    elapsed = time.time() - t0
    print(f"  Computed subspaces for {len(subspaces)} parameters in {elapsed:.0f}s")
    return subspaces


def project_grad_orthogonal_U(grad, U_k):
    projection = U_k @ (U_k.T @ grad.float())
    return (grad.float() - projection).to(grad.dtype)


def project_grad_orthogonal_V(grad, V_k):
    projection = grad.float() @ V_k.T @ V_k
    return (grad.float() - projection).to(grad.dtype)


def train_dpo_with_projection(model, tokenizer, train_data, device, subspaces,
                              lora_subspace_map, subspace_device_cache, condition_name,
                              wandb_run=None):
    """Core DPO training loop with optional OGPSA gradient projection."""
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01
    )

    pre_loss = compute_agent_loss(model, tokenizer, EVAL_EXAMPLES[:30], device)
    print(f"  Pre-train agent loss: {pre_loss:.4f}")

    model.train()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    global_step = 0
    n_projected = 0

    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        epoch_steps = 0
        t0 = time.time()
        optimizer.zero_grad()
        accum_loss = 0.0

        for i, example in enumerate(train_data):
            prompt = example["prompt"]
            chosen_text = prompt + "\n" + example["chosen"]
            rejected_text = prompt + "\n" + example["rejected"]

            chosen_ids = tokenizer(chosen_text, return_tensors="pt",
                                   truncation=True, max_length=MAX_SEQ_LEN)["input_ids"].to(device)
            rejected_ids = tokenizer(rejected_text, return_tensors="pt",
                                     truncation=True, max_length=MAX_SEQ_LEN)["input_ids"].to(device)
            prompt_ids = tokenizer(prompt + "\n", return_tensors="pt")["input_ids"]
            prompt_len = prompt_ids.shape[1]

            if prompt_len >= chosen_ids.shape[1] - 1 or prompt_len >= rejected_ids.shape[1] - 1:
                continue

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

            loss = -torch.nn.functional.logsigmoid(DPO_BETA * (chosen_logp - rejected_logp))
            scaled_loss = loss / GRAD_ACCUM_STEPS
            scaled_loss.backward()

            accum_loss += loss.item()
            epoch_steps += 1

            if (i + 1) % GRAD_ACCUM_STEPS == 0 or (i + 1) == len(train_data):
                # Project gradients for mapped parameters only
                if lora_subspace_map:
                    with torch.no_grad():
                        for name, param in model.named_parameters():
                            if param.requires_grad and param.grad is not None and name in lora_subspace_map:
                                info = lora_subspace_map[name]
                                cached = subspace_device_cache[info["orig_key"]]
                                if info["type"] == "B":
                                    U_k = cached["U"]
                                    if U_k.shape[0] == param.grad.shape[0]:
                                        param.grad = project_grad_orthogonal_U(param.grad, U_k)
                                        n_projected += 1
                                elif info["type"] == "A":
                                    V_k = cached["V"]
                                    if V_k.shape[1] == param.grad.shape[1]:
                                        param.grad = project_grad_orthogonal_V(param.grad, V_k)
                                        n_projected += 1

                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                optimizer.zero_grad()

                global_step += 1
                epoch_loss += accum_loss
                accum_loss = 0.0

            torch.cuda.empty_cache()

            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                print(f"    Epoch {epoch+1}, step {i+1}/{len(train_data)}: "
                      f"loss={epoch_loss/max(epoch_steps, 1):.4f}, {elapsed:.0f}s")

        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1}/{NUM_EPOCHS}: loss={epoch_loss/max(epoch_steps, 1):.4f} ({elapsed:.0f}s)")

    post_loss = compute_agent_loss(model, tokenizer, EVAL_EXAMPLES[:30], device)
    print(f"  Post-train agent loss: {post_loss:.4f}")
    print(f"  Loss change: {post_loss - pre_loss:+.4f}")
    print(f"  Gradient projections: {n_projected}")

    return {
        "condition": condition_name,
        "pre_loss": float(pre_loss),
        "post_loss": float(post_loss),
        "loss_change": float(post_loss - pre_loss),
        "n_gradient_projections": n_projected,
    }


def run_model(model_name, config, device, train_data):
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    from peft import LoraConfig, get_peft_model

    base_dir = config["base"]
    it_dir = config["it"]

    print(f"\n{'#'*70}")
    print(f"# Model: {model_name}")
    print(f"{'#'*70}")

    tokenizer = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_config = AutoConfig.from_pretrained(base_dir, trust_remote_code=True)
    num_layers = model_config.num_hidden_layers
    mid_layers = get_mid_layer_targets(num_layers)

    results = {}

    # ========== Condition 1: OGPSA Hybrid (project only output-pathway) ==========
    print(f"\n{'='*60}")
    print(f"OGPSA HYBRID: project V/O/W_down only, standard LoRA on Q/K/gate/up")
    print(f"{'='*60}")

    # Compute subspace only for output-pathway components
    subspaces_hybrid = compute_capability_subspace(
        base_dir, it_dir, mid_layers, target_types=OUTPUT_PATHWAY, rank=SUBSPACE_RANK)

    model = AutoModelForCausalLM.from_pretrained(
        base_dir, torch_dtype=torch.float16, device_map=device, trust_remote_code=True)
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=ALL_TARGETS,  # ALL projections get LoRA
        layers_to_transform=mid_layers,
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  LoRA on ALL targets, projection on output-pathway only")
    print(f"  Trainable: {trainable/1e6:.1f}M")

    # Map LoRA params to subspaces — only output-pathway components
    lora_subspace_map = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        orig_name = name.replace("base_model.model.", "").replace(".lora_A.default.weight", ".weight").replace(".lora_B.default.weight", ".weight")
        if orig_name in subspaces_hybrid:
            if "lora_B" in name:
                lora_subspace_map[name] = {"type": "B", "orig_key": orig_name}
            elif "lora_A" in name:
                lora_subspace_map[name] = {"type": "A", "orig_key": orig_name}

    subspace_cache = {}
    for info in lora_subspace_map.values():
        orig_key = info["orig_key"]
        if orig_key not in subspace_cache:
            subspace_cache[orig_key] = {
                "U": subspaces_hybrid[orig_key]["U"].to(device),
                "V": subspaces_hybrid[orig_key]["V"].to(device),
            }

    print(f"  Mapped {len(lora_subspace_map)} LoRA params to output-pathway subspaces")

    hybrid_result = train_dpo_with_projection(
        model, tokenizer, train_data, device,
        subspaces_hybrid, lora_subspace_map, subspace_cache,
        condition_name="ogpsa_hybrid")
    hybrid_result["trainable_params_M"] = trainable / 1e6
    hybrid_result["projected_targets"] = OUTPUT_PATHWAY
    hybrid_result["lora_targets"] = ALL_TARGETS
    results["ogpsa_hybrid"] = hybrid_result

    del model, subspaces_hybrid, subspace_cache
    for k in list(lora_subspace_map.keys()):
        del lora_subspace_map[k]
    torch.cuda.empty_cache()
    import gc; gc.collect()

    # ========== Condition 2: Standard OGPSA (project all components) ==========
    print(f"\n{'='*60}")
    print(f"OGPSA STANDARD: project all 7 component types")
    print(f"{'='*60}")

    subspaces_all = compute_capability_subspace(
        base_dir, it_dir, mid_layers, target_types=ALL_TARGETS, rank=SUBSPACE_RANK)

    model = AutoModelForCausalLM.from_pretrained(
        base_dir, torch_dtype=torch.float16, device_map=device, trust_remote_code=True)
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=ALL_TARGETS,
        layers_to_transform=mid_layers,
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable: {trainable/1e6:.1f}M")

    lora_subspace_map = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        orig_name = name.replace("base_model.model.", "").replace(".lora_A.default.weight", ".weight").replace(".lora_B.default.weight", ".weight")
        if orig_name in subspaces_all:
            if "lora_B" in name:
                lora_subspace_map[name] = {"type": "B", "orig_key": orig_name}
            elif "lora_A" in name:
                lora_subspace_map[name] = {"type": "A", "orig_key": orig_name}

    subspace_cache = {}
    for info in lora_subspace_map.values():
        orig_key = info["orig_key"]
        if orig_key not in subspace_cache:
            subspace_cache[orig_key] = {
                "U": subspaces_all[orig_key]["U"].to(device),
                "V": subspaces_all[orig_key]["V"].to(device),
            }

    print(f"  Mapped {len(lora_subspace_map)} LoRA params to all subspaces")

    standard_result = train_dpo_with_projection(
        model, tokenizer, train_data, device,
        subspaces_all, lora_subspace_map, subspace_cache,
        condition_name="ogpsa_standard")
    standard_result["trainable_params_M"] = trainable / 1e6
    standard_result["projected_targets"] = ALL_TARGETS
    standard_result["lora_targets"] = ALL_TARGETS
    results["ogpsa_standard"] = standard_result

    del model, subspaces_all, subspace_cache
    torch.cuda.empty_cache()
    gc.collect()

    # ========== Load reference results ==========
    # Pull standard DPO and OC-DPO from previous OC-DPO experiment/19
    for step_file in [f"ocdpo_large_{model_name}.json", f"ogpsa_comparison_{model_name}.json"]:
        path = RESULTS_DIR / step_file
        if path.exists():
            with open(path) as f:
                prev_data = json.load(f)
            for cond_name in ["standard", "ocdpo_exclude_output", "ogpsa"]:
                if cond_name in prev_data.get("conditions", {}) and cond_name not in results:
                    results[cond_name] = prev_data["conditions"][cond_name]
                    print(f"  Loaded {cond_name} from {step_file}")

    # ========== Summary ==========
    print(f"\n{'='*60}")
    print(f"SUMMARY: {model_name} ({NUM_PAIRS} UltraFeedback pairs)")
    print(f"{'='*60}")

    std_tax = results.get("standard", {}).get("loss_change", None)
    if std_tax:
        print(f"\n  Standard DPO tax: {std_tax:+.4f}")

    print(f"\n{'Method':<30} {'Loss Change':>12} {'Tax Red%':>10}")
    print("-" * 55)
    for cond_name in ["ogpsa_standard", "ogpsa_hybrid", "ocdpo_exclude_output"]:
        if cond_name in results:
            r = results[cond_name]
            change = r["loss_change"]
            if std_tax and std_tax != 0:
                red = (std_tax - change) / abs(std_tax) * 100
                print(f"  {cond_name:<28} {change:>+12.4f} {red:>9.1f}%")
            else:
                print(f"  {cond_name:<28} {change:>+12.4f}")

    # Save
    output = {
        "analysis": "OGPSA hybrid (output-pathway-restricted projection)",
        "model": model_name,
        "dataset": "HuggingFaceH4/ultrafeedback_binarized",
        "num_pairs": len(train_data),
        "num_epochs": NUM_EPOCHS,
        "subspace_rank": SUBSPACE_RANK,
        "conditions": results,
    }
    out_path = RESULTS_DIR / f"ogpsa_hybrid_{model_name}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")

    return results


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    single_model = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"OGPSA + Output-Pathway Hybrid")
    print(f"Device: {device}")

    train_data = load_ultrafeedback_pairs(num_pairs=NUM_PAIRS, seed=42)

    models_to_run = {single_model: MODEL_CONFIGS[single_model]} if single_model else MODEL_CONFIGS

    all_results = {}
    for model_name, config in models_to_run.items():
        all_results[model_name] = run_model(model_name, config, device, train_data)

    # Cross-model summary
    if len(all_results) > 1:
        print(f"\n{'#'*70}")
        print(f"CROSS-MODEL SUMMARY")
        print(f"{'#'*70}")
        for model_name, results in all_results.items():
            std_tax = results.get("standard", {}).get("loss_change", None)
            print(f"\n{model_name}:")
            for cond in ["ogpsa_standard", "ogpsa_hybrid", "ocdpo_exclude_output"]:
                if cond in results:
                    change = results[cond]["loss_change"]
                    if std_tax and std_tax != 0:
                        red = (std_tax - change) / abs(std_tax) * 100
                        print(f"  {cond}: {change:+.4f} ({red:.1f}% tax reduction)")
                    else:
                        print(f"  {cond}: {change:+.4f}")

    combined_path = RESULTS_DIR / "ogpsa_hybrid_combined.json"
    combined = {"analysis": "OGPSA hybrid cross-model", "models": {}}
    for model_name, results in all_results.items():
        combined["models"][model_name] = {
            cond: {"loss_change": results[cond]["loss_change"],
                   "pre_loss": results[cond].get("pre_loss"),
                   "post_loss": results[cond].get("post_loss")}
            for cond in results
        }
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2, default=str)
    print(f"\nSaved combined: {combined_path}")


if __name__ == "__main__":
    main()
