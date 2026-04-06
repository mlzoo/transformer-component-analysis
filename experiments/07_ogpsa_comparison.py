"""
Step 19: OGPSA vs OC-DPO Comparison at Scale (496 UltraFeedback Pairs)

Reviewer concern: Step 11 OGPSA comparison used only 20 preference pairs.
This repeats the OGPSA vs OC-DPO comparison on 496 UltraFeedback pairs
(same dataset as previous OC-DPO experiment) to validate whether OGPSA or OC-DPO is more
effective at reducing alignment tax at realistic data scale.

Conditions:
  - ogpsa: Standard LoRA targets (all 7) with OGPSA gradient projection
  - ocdpo_exclude_output: OC-DPO (q,k,gate,up only) — reused from previous OC-DPO experiment

OGPSA projects DPO gradients orthogonally to capability-relevant subspaces:
  1. Compute alignment delta: DW = W_IT - W_base for each LoRA target
  2. SVD of DW -> keep top-64 singular vectors as capability subspace
  3. After each backward: project LoRA B grads via U, LoRA A grads via V

Models: Qwen2.5-7B, Llama-3.1-8B, Mistral-7B-v0.3
Training: 496 pairs, 3 epochs, LoRA r=16 alpha=32 mid-third layers, lr=5e-5

Usage: python 07_ogpsa_comparison.py cuda:0
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
from 05_ocdpo import EVAL_EXAMPLES, compute_agent_loss, ALL_TARGETS, get_mid_layer_targets

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

OCDPO_TARGETS = ["q_proj", "k_proj", "gate_proj", "up_proj"]

NUM_PAIRS = 496
NUM_EPOCHS = 3
LR = 5e-5
LORA_R = 16
LORA_ALPHA = 32
DPO_BETA = 0.1
MAX_SEQ_LEN = 512
GRAD_ACCUM_STEPS = 4
SUBSPACE_RANK = 64

# Keywords for filtering structured/agent-relevant UltraFeedback examples
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
    """Load and filter UltraFeedback for structured/agent-relevant preference pairs.

    Same logic as previous OC-DPO experiment to ensure identical dataset.
    """
    from datasets import load_dataset

    print(f"Loading UltraFeedback dataset...")
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
    print(f"  Total examples: {len(ds)}")

    # Score each example for relevance to structured/agent tasks
    scored = []
    for i, ex in enumerate(ds):
        prompt = ex["prompt"].lower()
        chosen_text = ex["chosen"][1]["content"] if len(ex["chosen"]) > 1 else ""
        combined = prompt + " " + chosen_text.lower()

        score = sum(1 for kw in STRUCTURED_KEYWORDS if kw in combined)

        score_gap = ex["score_chosen"] - ex["score_rejected"]
        if score_gap >= 2.0:
            score += 3
        elif score_gap >= 1.0:
            score += 1

        if len(chosen_text) > 2000 or len(chosen_text) < 20:
            score -= 5

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

        pairs.append({
            "prompt": prompt,
            "chosen": chosen_text,
            "rejected": rejected_text,
            "score_gap": ex["score_chosen"] - ex["score_rejected"],
        })

    pairs = pairs[:num_pairs]
    print(f"  Selected {len(pairs)} preference pairs")
    avg_gap = np.mean([p["score_gap"] for p in pairs])
    print(f"  Avg score gap (chosen - rejected): {avg_gap:.2f}")
    return pairs


def compute_capability_subspace(base_dir, it_dir, mid_layers, rank=SUBSPACE_RANK):
    """Compute capability-relevant subspace via SVD of alignment deltas.

    For each LoRA-targetable projection in mid-layers, compute SVD of (IT - Base).
    The top-k singular vectors span the capability modification subspace.
    """
    print("Computing capability subspace from weight deltas...")
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
    target_types = ALL_TARGETS  # all 7 projections

    common_keys = set(base_index.keys()) & set(it_index.keys())

    # Only compute for mid-layer projections (matching LoRA scope)
    relevant_keys = []
    for k in common_keys:
        if not any(t in k for t in target_types):
            continue
        if "weight" not in k:
            continue
        # Check if this key belongs to a mid-layer
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
                "U": U[:, :actual_rank].clone(),   # (out_dim, k)
                "V": Vh[:actual_rank, :].clone(),   # (k, in_dim)
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
    """Project gradient orthogonally to capability subspace (output dimension).

    For LoRA B: grad' = grad - U_k @ U_k^T @ grad
    U_k: (out_dim, subspace_rank), grad: (out_dim, lora_rank)
    """
    projection = U_k @ (U_k.T @ grad.float())
    return (grad.float() - projection).to(grad.dtype)


def project_grad_orthogonal_V(grad, V_k):
    """Project gradient orthogonally to capability subspace (input dimension).

    For LoRA A: grad' = grad - grad @ V_k^T @ V_k
    V_k: (subspace_rank, in_dim), grad: (lora_rank, in_dim)
    """
    projection = grad.float() @ V_k.T @ V_k
    return (grad.float() - projection).to(grad.dtype)


def run_ogpsa_dpo(base_dir, it_dir, device, tokenizer, train_data, subspaces,
                  wandb_run=None):
    """Train DPO with OGPSA gradient projection on UltraFeedback data."""
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    print(f"\n{'='*60}")
    print(f"OGPSA DPO Training ({len(train_data)} pairs, {NUM_EPOCHS} epochs)")
    print(f"{'='*60}")

    # Load base model (OGPSA trains from base, not IT)
    model = AutoModelForCausalLM.from_pretrained(
        base_dir, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
    )
    model.config.use_cache = False

    num_layers = model.config.num_hidden_layers
    mid_layers = get_mid_layer_targets(num_layers)

    # Apply LoRA to ALL projections (OGPSA uses all 7, projects gradients instead)
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=ALL_TARGETS,
        layers_to_transform=mid_layers,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  LoRA targets: {ALL_TARGETS} (all projections)")
    print(f"  LoRA on layers: {mid_layers[0]}-{mid_layers[-1]}")
    print(f"  Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M ({100*trainable/total:.2f}%)")

    # Map LoRA parameter names to their original weight names for subspace lookup
    lora_subspace_map = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        orig_name = name.replace("base_model.model.", "")
        orig_name = orig_name.replace(".lora_A.default.weight", ".weight")
        orig_name = orig_name.replace(".lora_B.default.weight", ".weight")

        if orig_name in subspaces:
            if "lora_B" in name:
                lora_subspace_map[name] = {"type": "B", "orig_key": orig_name}
            elif "lora_A" in name:
                lora_subspace_map[name] = {"type": "A", "orig_key": orig_name}

    # Cache subspace matrices on device
    subspace_device_cache = {}
    for info in lora_subspace_map.values():
        orig_key = info["orig_key"]
        if orig_key not in subspace_device_cache:
            subspace_device_cache[orig_key] = {
                "U": subspaces[orig_key]["U"].to(device),
                "V": subspaces[orig_key]["V"].to(device),
            }

    print(f"  Mapped {len(lora_subspace_map)} LoRA params to subspaces "
          f"({len(subspace_device_cache)} unique projections cached)")

    # Pre-training agent loss
    pre_loss = compute_agent_loss(model, tokenizer, EVAL_EXAMPLES[:30], device)
    print(f"  Pre-train agent loss: {pre_loss:.4f}")

    if wandb_run is not None:
        wandb_run.log({
            "ogpsa/pre_agent_loss": pre_loss,
            "ogpsa/trainable_params_M": trainable / 1e6,
        })

    # Training with gradient projection
    model.train()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01
    )

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

            chosen_ids = tokenizer(
                chosen_text, return_tensors="pt",
                truncation=True, max_length=MAX_SEQ_LEN
            )["input_ids"].to(device)
            rejected_ids = tokenizer(
                rejected_text, return_tensors="pt",
                truncation=True, max_length=MAX_SEQ_LEN
            )["input_ids"].to(device)
            prompt_ids = tokenizer(prompt + "\n", return_tensors="pt")["input_ids"]
            prompt_len = prompt_ids.shape[1]

            # Skip if prompt is too long relative to response
            if prompt_len >= chosen_ids.shape[1] - 1 or prompt_len >= rejected_ids.shape[1] - 1:
                continue

            # Forward chosen
            chosen_logits = model(input_ids=chosen_ids).logits
            chosen_lps = torch.nn.functional.log_softmax(
                chosen_logits[0, prompt_len-1:-1, :].float(), dim=-1
            )
            chosen_logp = chosen_lps.gather(
                1, chosen_ids[0, prompt_len:].unsqueeze(1)
            ).squeeze(1).sum()

            # Forward rejected
            rejected_logits = model(input_ids=rejected_ids).logits
            rejected_lps = torch.nn.functional.log_softmax(
                rejected_logits[0, prompt_len-1:-1, :].float(), dim=-1
            )
            rejected_logp = rejected_lps.gather(
                1, rejected_ids[0, prompt_len:].unsqueeze(1)
            ).squeeze(1).sum()

            loss = -torch.nn.functional.logsigmoid(DPO_BETA * (chosen_logp - rejected_logp))
            # Scale loss for gradient accumulation
            scaled_loss = loss / GRAD_ACCUM_STEPS
            scaled_loss.backward()

            accum_loss += loss.item()
            epoch_steps += 1

            # Gradient accumulation step
            if (i + 1) % GRAD_ACCUM_STEPS == 0 or (i + 1) == len(train_data):
                # OGPSA: Project gradients orthogonally to capability subspace
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
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                optimizer.zero_grad()

                global_step += 1
                epoch_loss += accum_loss

                if wandb_run is not None and global_step % 25 == 0:
                    wandb_run.log({
                        "ogpsa/dpo_loss": accum_loss / GRAD_ACCUM_STEPS,
                        "ogpsa/step": global_step,
                    })

                accum_loss = 0.0

            torch.cuda.empty_cache()

            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                print(f"    Epoch {epoch+1}, step {i+1}/{len(train_data)}: "
                      f"loss={epoch_loss/max(epoch_steps, 1):.4f}, {elapsed:.0f}s elapsed")

        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1}/{NUM_EPOCHS}: DPO loss={avg_epoch_loss:.4f} "
              f"({epoch_steps} steps, {elapsed:.0f}s)")

        if wandb_run is not None:
            wandb_run.log({
                "ogpsa/epoch": epoch + 1,
                "ogpsa/epoch_dpo_loss": avg_epoch_loss,
            })

    print(f"  Total gradient projections applied: {n_projected}")

    # Post-training agent loss
    post_loss = compute_agent_loss(model, tokenizer, EVAL_EXAMPLES[:30], device)
    print(f"  Post-train agent loss: {post_loss:.4f}")
    print(f"  Agent loss change: {post_loss - pre_loss:+.4f}")

    if wandb_run is not None:
        wandb_run.log({
            "ogpsa/post_agent_loss": post_loss,
            "ogpsa/agent_loss_change": post_loss - pre_loss,
        })

    del model, optimizer
    for k in list(subspace_device_cache.keys()):
        del subspace_device_cache[k]
    torch.cuda.empty_cache()

    return {
        "condition": "ogpsa",
        "lora_targets": ALL_TARGETS,
        "num_train_pairs": len(train_data),
        "num_epochs": NUM_EPOCHS,
        "pre_loss": float(pre_loss),
        "post_loss": float(post_loss),
        "loss_change": float(post_loss - pre_loss),
        "trainable_params_M": trainable / 1e6,
        "subspace_rank": SUBSPACE_RANK,
        "n_gradient_projections": n_projected,
    }


def run_ocdpo_condition(base_dir, device, tokenizer, train_data, wandb_run=None):
    """Train OC-DPO (exclude V/O/down) on UltraFeedback data."""
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    print(f"\n{'='*60}")
    print(f"OC-DPO (exclude V/O/down) ({len(train_data)} pairs, {NUM_EPOCHS} epochs)")
    print(f"{'='*60}")

    model = AutoModelForCausalLM.from_pretrained(
        base_dir, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
    )
    model.config.use_cache = False

    num_layers = model.config.num_hidden_layers
    mid_layers = get_mid_layer_targets(num_layers)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=OCDPO_TARGETS,
        layers_to_transform=mid_layers,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  LoRA targets: {OCDPO_TARGETS}")
    print(f"  LoRA on layers: {mid_layers[0]}-{mid_layers[-1]}")
    print(f"  Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M ({100*trainable/total:.2f}%)")

    # Pre-training agent loss
    pre_loss = compute_agent_loss(model, tokenizer, EVAL_EXAMPLES[:30], device)
    print(f"  Pre-train agent loss: {pre_loss:.4f}")

    if wandb_run is not None:
        wandb_run.log({
            "ocdpo/pre_agent_loss": pre_loss,
            "ocdpo/trainable_params_M": trainable / 1e6,
        })

    # Training
    model.train()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01
    )

    global_step = 0
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

            chosen_ids = tokenizer(
                chosen_text, return_tensors="pt",
                truncation=True, max_length=MAX_SEQ_LEN
            )["input_ids"].to(device)
            rejected_ids = tokenizer(
                rejected_text, return_tensors="pt",
                truncation=True, max_length=MAX_SEQ_LEN
            )["input_ids"].to(device)
            prompt_ids = tokenizer(prompt + "\n", return_tensors="pt")["input_ids"]
            prompt_len = prompt_ids.shape[1]

            if prompt_len >= chosen_ids.shape[1] - 1 or prompt_len >= rejected_ids.shape[1] - 1:
                continue

            chosen_logits = model(input_ids=chosen_ids).logits
            chosen_lps = torch.nn.functional.log_softmax(
                chosen_logits[0, prompt_len-1:-1, :].float(), dim=-1
            )
            chosen_logp = chosen_lps.gather(
                1, chosen_ids[0, prompt_len:].unsqueeze(1)
            ).squeeze(1).sum()

            rejected_logits = model(input_ids=rejected_ids).logits
            rejected_lps = torch.nn.functional.log_softmax(
                rejected_logits[0, prompt_len-1:-1, :].float(), dim=-1
            )
            rejected_logp = rejected_lps.gather(
                1, rejected_ids[0, prompt_len:].unsqueeze(1)
            ).squeeze(1).sum()

            loss = -torch.nn.functional.logsigmoid(DPO_BETA * (chosen_logp - rejected_logp))
            scaled_loss = loss / GRAD_ACCUM_STEPS
            scaled_loss.backward()

            accum_loss += loss.item()
            epoch_steps += 1

            if (i + 1) % GRAD_ACCUM_STEPS == 0 or (i + 1) == len(train_data):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                optimizer.zero_grad()

                global_step += 1
                epoch_loss += accum_loss

                if wandb_run is not None and global_step % 25 == 0:
                    wandb_run.log({
                        "ocdpo/dpo_loss": accum_loss / GRAD_ACCUM_STEPS,
                        "ocdpo/step": global_step,
                    })

                accum_loss = 0.0

            torch.cuda.empty_cache()

            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                print(f"    Epoch {epoch+1}, step {i+1}/{len(train_data)}: "
                      f"loss={epoch_loss/max(epoch_steps, 1):.4f}, {elapsed:.0f}s elapsed")

        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1}/{NUM_EPOCHS}: DPO loss={avg_epoch_loss:.4f} "
              f"({epoch_steps} steps, {elapsed:.0f}s)")

        if wandb_run is not None:
            wandb_run.log({
                "ocdpo/epoch": epoch + 1,
                "ocdpo/epoch_dpo_loss": avg_epoch_loss,
            })

    # Post-training agent loss
    post_loss = compute_agent_loss(model, tokenizer, EVAL_EXAMPLES[:30], device)
    print(f"  Post-train agent loss: {post_loss:.4f}")
    print(f"  Agent loss change: {post_loss - pre_loss:+.4f}")

    if wandb_run is not None:
        wandb_run.log({
            "ocdpo/post_agent_loss": post_loss,
            "ocdpo/agent_loss_change": post_loss - pre_loss,
        })

    del model, optimizer
    torch.cuda.empty_cache()

    return {
        "condition": "ocdpo_exclude_output",
        "lora_targets": OCDPO_TARGETS,
        "num_train_pairs": len(train_data),
        "num_epochs": NUM_EPOCHS,
        "pre_loss": float(pre_loss),
        "post_loss": float(post_loss),
        "loss_change": float(post_loss - pre_loss),
        "trainable_params_M": trainable / 1e6,
    }


def load_previous OC-DPO experiment_results(model_name):
    """Try to load previous OC-DPO experiment results for reuse of OC-DPO condition."""
    path = RESULTS_DIR / f"ocdpo_large_{model_name}.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        conditions = data.get("conditions", {})
        if "ocdpo_exclude_output" in conditions:
            print(f"  Found previous OC-DPO experiment OC-DPO results at {path}")
            return conditions["ocdpo_exclude_output"]
    return None


def run_model(model_name, config, device, train_data):
    """Run OGPSA and OC-DPO comparison for one model."""
    import wandb
    from transformers import AutoTokenizer

    base_dir = config["base"]
    it_dir = config["it"]

    print(f"\n{'#'*70}")
    print(f"# Model: {model_name}")
    print(f"#   Base: {base_dir}")
    print(f"#   IT:   {it_dir}")
    print(f"{'#'*70}")

    wandb.login()  # uses WANDB_API_KEY env var
    run = wandb.init(
        project="anonymous-submission",
        name=f"ogpsa-comparison-{model_name}",
        config={
            "step": 19,
            "model": model_name,
            "base_dir": base_dir,
            "it_dir": it_dir,
            "num_pairs": len(train_data),
            "num_epochs": NUM_EPOCHS,
            "lr": LR,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "dpo_beta": DPO_BETA,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "subspace_rank": SUBSPACE_RANK,
            "dataset": "ultrafeedback_binarized",
            "conditions": ["ogpsa", "ocdpo_exclude_output"],
        },
        reinit=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = {}

    # Check for reusable previous OC-DPO experiment OC-DPO results
    previous OC-DPO experiment_ocdpo = load_previous OC-DPO experiment_results(model_name)

    # Compute capability subspace (needed for OGPSA)
    # Determine mid layers from a quick config load
    from transformers import AutoConfig
    model_config = AutoConfig.from_pretrained(base_dir, trust_remote_code=True)
    num_layers = model_config.num_hidden_layers
    mid_layers = get_mid_layer_targets(num_layers)
    print(f"  Model has {num_layers} layers, mid-third: {mid_layers[0]}-{mid_layers[-1]}")

    subspaces = compute_capability_subspace(base_dir, it_dir, mid_layers, rank=SUBSPACE_RANK)

    # Run OGPSA
    ogpsa_result = run_ogpsa_dpo(base_dir, it_dir, device, tokenizer, train_data,
                                  subspaces, wandb_run=run)
    results["ogpsa"] = ogpsa_result

    # Clean up subspaces from CPU memory
    del subspaces

    # Run or reuse OC-DPO
    if previous OC-DPO experiment_ocdpo is not None:
        print(f"\n  Reusing previous OC-DPO experiment OC-DPO results for {model_name}")
        results["ocdpo_exclude_output"] = previous OC-DPO experiment_ocdpo
        run.log({
            "ocdpo/pre_agent_loss_reused": previous OC-DPO experiment_ocdpo["pre_loss"],
            "ocdpo/post_agent_loss_reused": previous OC-DPO experiment_ocdpo["post_loss"],
            "ocdpo/agent_loss_change_reused": previous OC-DPO experiment_ocdpo["loss_change"],
        })
    else:
        print(f"\n  No previous OC-DPO experiment results found, running OC-DPO from scratch...")
        ocdpo_result = run_ocdpo_condition(base_dir, device, tokenizer, train_data,
                                            wandb_run=run)
        results["ocdpo_exclude_output"] = ocdpo_result

    # Also pull standard DPO from previous OC-DPO experiment if available
    previous OC-DPO experiment_path = RESULTS_DIR / f"ocdpo_large_{model_name}.json"
    standard_result = None
    if previous OC-DPO experiment_path.exists():
        with open(previous OC-DPO experiment_path) as f:
            previous OC-DPO experiment_data = json.load(f)
        if "standard" in previous OC-DPO experiment_data.get("conditions", {}):
            standard_result = previous OC-DPO experiment_data["conditions"]["standard"]
            results["standard"] = standard_result
            print(f"  Reused previous OC-DPO experiment standard DPO results")

    # Summary comparison
    print(f"\n{'='*60}")
    print(f"OGPSA vs OC-DPO COMPARISON: {model_name} (496 UltraFeedback pairs)")
    print(f"{'='*60}")

    ogpsa_change = results["ogpsa"]["loss_change"]
    ocdpo_change = results["ocdpo_exclude_output"]["loss_change"]

    # Use standard DPO as reference if available
    if standard_result is not None:
        std_change = standard_result["loss_change"]
        print(f"\n  Standard DPO alignment tax: {std_change:+.4f}")
    else:
        std_change = None
        print(f"\n  (No standard DPO baseline available)")

    print(f"\n{'Method':<25} {'Pre-loss':>10} {'Post-loss':>10} {'Change':>10}", end="")
    if std_change is not None:
        print(f" {'Tax Red.':>10}")
    else:
        print()
    print("-" * 70)

    for name, key in [("OGPSA", "ogpsa"), ("OC-DPO (excl V/O/down)", "ocdpo_exclude_output")]:
        r = results[key]
        line = f"{name:<25} {r['pre_loss']:>10.4f} {r['post_loss']:>10.4f} {r['loss_change']:>+10.4f}"
        if std_change is not None and std_change != 0:
            tax_red = (std_change - r["loss_change"]) / abs(std_change) * 100
            line += f" {tax_red:>9.1f}%"
        print(line)

    if standard_result is not None:
        r = standard_result
        line = f"{'Standard DPO':<25} {r['pre_loss']:>10.4f} {r['post_loss']:>10.4f} {r['loss_change']:>+10.4f}"
        if std_change is not None and std_change != 0:
            line += f" {0.0:>9.1f}%"
        print(line)

    # Key finding
    print(f"\n--- KEY FINDING ---")
    if std_change is not None and std_change != 0:
        ogpsa_red = (std_change - ogpsa_change) / abs(std_change) * 100
        ocdpo_red = (std_change - ocdpo_change) / abs(std_change) * 100
        print(f"  OGPSA tax reduction:  {ogpsa_red:.1f}%")
        print(f"  OC-DPO tax reduction: {ocdpo_red:.1f}%")
        if ocdpo_red > ogpsa_red:
            print(f"  OC-DPO outperforms OGPSA by {ocdpo_red - ogpsa_red:.1f} pp")
        else:
            print(f"  OGPSA outperforms OC-DPO by {ogpsa_red - ocdpo_red:.1f} pp")

        run.log({
            "summary/ogpsa_tax": ogpsa_change,
            "summary/ocdpo_tax": ocdpo_change,
            "summary/standard_tax": std_change,
            "summary/ogpsa_tax_reduction_pct": ogpsa_red,
            "summary/ocdpo_tax_reduction_pct": ocdpo_red,
        })
    else:
        print(f"  OGPSA loss change: {ogpsa_change:+.4f}")
        print(f"  OC-DPO loss change: {ocdpo_change:+.4f}")
        diff = ogpsa_change - ocdpo_change
        print(f"  Difference (OGPSA - OC-DPO): {diff:+.4f}")

        run.log({
            "summary/ogpsa_tax": ogpsa_change,
            "summary/ocdpo_tax": ocdpo_change,
        })

    wandb.finish()

    # Save per-model results
    output = {
        "analysis": "OGPSA vs OC-DPO at scale (496 UltraFeedback pairs)",
        "model": model_name,
        "base_dir": base_dir,
        "it_dir": it_dir,
        "dataset": "HuggingFaceH4/ultrafeedback_binarized",
        "num_pairs": len(train_data),
        "num_epochs": NUM_EPOCHS,
        "subspace_rank": SUBSPACE_RANK,
        "conditions": results,
    }
    out_path = RESULTS_DIR / f"ogpsa_comparison_{model_name}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")

    return results


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    print(f"Device: {device}")
    print(f"Step 19: OGPSA vs OC-DPO at Scale (496 UltraFeedback Pairs)")
    print(f"  Subspace rank: {SUBSPACE_RANK}")
    print(f"  Grad accumulation: {GRAD_ACCUM_STEPS}")
    print(f"  DPO beta: {DPO_BETA}")

    # Load training data once (shared across all models, same as previous OC-DPO experiment)
    train_data = load_ultrafeedback_pairs(num_pairs=NUM_PAIRS, seed=42)

    # Run each model sequentially
    all_results = {}
    for model_name, config in MODEL_CONFIGS.items():
        results = run_model(model_name, config, device, train_data)
        all_results[model_name] = results

    # Cross-model summary
    print(f"\n{'#'*70}")
    print(f"# CROSS-MODEL SUMMARY: OGPSA vs OC-DPO (496 UltraFeedback pairs)")
    print(f"{'#'*70}")

    print(f"\n{'Model':<20} {'OGPSA Tax':>12} {'OC-DPO Tax':>12} {'Std Tax':>12} "
          f"{'OGPSA Red%':>12} {'OC-DPO Red%':>12}")
    print("-" * 82)

    combined = {
        "analysis": "OGPSA vs OC-DPO at scale -- cross-model summary",
        "dataset": "HuggingFaceH4/ultrafeedback_binarized",
        "num_pairs": NUM_PAIRS,
        "num_epochs": NUM_EPOCHS,
        "subspace_rank": SUBSPACE_RANK,
        "models": {},
    }

    for model_name, results in all_results.items():
        ogpsa_tax = results["ogpsa"]["loss_change"]
        ocdpo_tax = results["ocdpo_exclude_output"]["loss_change"]
        std_tax = results.get("standard", {}).get("loss_change", None)

        if std_tax is not None and std_tax != 0:
            ogpsa_red = (std_tax - ogpsa_tax) / abs(std_tax) * 100
            ocdpo_red = (std_tax - ocdpo_tax) / abs(std_tax) * 100
            print(f"{model_name:<20} {ogpsa_tax:>+12.4f} {ocdpo_tax:>+12.4f} "
                  f"{std_tax:>+12.4f} {ogpsa_red:>11.1f}% {ocdpo_red:>11.1f}%")
        else:
            ogpsa_red = None
            ocdpo_red = None
            std_str = "N/A" if std_tax is None else f"{std_tax:+.4f}"
            print(f"{model_name:<20} {ogpsa_tax:>+12.4f} {ocdpo_tax:>+12.4f} "
                  f"{std_str:>12} {'N/A':>12} {'N/A':>12}")

        combined["models"][model_name] = {
            "ogpsa_tax": ogpsa_tax,
            "ocdpo_tax": ocdpo_tax,
            "standard_tax": std_tax,
            "ogpsa_tax_reduction_pct": round(ogpsa_red, 1) if ogpsa_red is not None else None,
            "ocdpo_tax_reduction_pct": round(ocdpo_red, 1) if ocdpo_red is not None else None,
            "ogpsa_pre_loss": results["ogpsa"]["pre_loss"],
            "ogpsa_post_loss": results["ogpsa"]["post_loss"],
            "ocdpo_pre_loss": results["ocdpo_exclude_output"]["pre_loss"],
            "ocdpo_post_loss": results["ocdpo_exclude_output"]["post_loss"],
        }

    # Save combined results
    combined_path = RESULTS_DIR / "ogpsa_comparison_combined.json"
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nSaved combined results to {combined_path}")


if __name__ == "__main__":
    main()
