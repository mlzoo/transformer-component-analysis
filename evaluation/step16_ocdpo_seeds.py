"""
Step 16: OC-DPO with Multiple Random Seeds for Confidence Intervals

Runs the 2 key conditions (standard DPO vs OC-DPO exclude-output) across
5 random seeds on all 3 model families. Reports mean +/- std for alignment
tax reduction, suitable for NeurIPS confidence intervals.

Usage:
    python step16_ocdpo_seeds.py cuda:3
"""

import sys
import json
import time
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, "./experiments")
from step6_ocdpo import (
    TRAIN_DATA,
    EVAL_EXAMPLES,
    compute_agent_loss,
    ALL_TARGETS,
    get_mid_layer_targets,
    CONDITION_TARGETS,
)

RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456, 789, 1337]
CONDITIONS = ["standard", "ocdpo_exclude_output"]

MODELS = [
    ("qwen2.5-7b", "./models/Qwen2.5-7B"),
    ("llama-3.1-8b", "./models/Llama-3.1-8B"),
    ("mistral-7b", "./models/Mistral-7B-v0.3"),
]


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_condition_seeded(base_dir, device, tokenizer, condition, seed,
                         num_epochs=5, lr=5e-5):
    """Train one DPO condition with a specific random seed."""
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    set_seed(seed)

    print(f"\n{'='*60}")
    print(f"Condition: {condition} | Seed: {seed}")
    print(f"{'='*60}")

    model = AutoModelForCausalLM.from_pretrained(
        base_dir, torch_dtype=torch.float16, device_map=device,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    num_layers = model.config.num_hidden_layers
    mid_layers = get_mid_layer_targets(num_layers)
    targets = CONDITION_TARGETS[condition]

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
    print(f"  LoRA on layers: {mid_layers[0]}-{mid_layers[-1]}")
    print(f"  Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M "
          f"({100*trainable/total:.2f}%)")

    # Pre-training eval
    pre_loss = compute_agent_loss(model, tokenizer, EVAL_EXAMPLES[:30], device)
    print(f"  Pre-train agent loss: {pre_loss:.4f}")

    # Training
    model.train()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01,
    )

    # Seed again right before training loop for consistent ordering
    set_seed(seed)

    for epoch in range(num_epochs):
        epoch_loss = 0
        # Shuffle training data with seed-controlled randomness
        indices = np.random.permutation(len(TRAIN_DATA))
        for idx in indices:
            example = TRAIN_DATA[idx]
            optimizer.zero_grad()

            prompt = example["prompt"]
            chosen_text = prompt + example["chosen"]
            rejected_text = prompt + example["rejected"]

            chosen_ids = tokenizer(
                chosen_text, return_tensors="pt",
                truncation=True, max_length=256,
            )["input_ids"].to(device)
            rejected_ids = tokenizer(
                rejected_text, return_tensors="pt",
                truncation=True, max_length=256,
            )["input_ids"].to(device)
            prompt_len = tokenizer(
                prompt, return_tensors="pt",
            )["input_ids"].shape[1]

            # Forward chosen
            chosen_logits = model(input_ids=chosen_ids).logits
            chosen_lps = torch.nn.functional.log_softmax(
                chosen_logits[0, prompt_len-1:-1, :].float(), dim=-1)
            chosen_logp = chosen_lps.gather(
                1, chosen_ids[0, prompt_len:].unsqueeze(1)).squeeze(1).sum()

            # Forward rejected
            rejected_logits = model(input_ids=rejected_ids).logits
            rejected_lps = torch.nn.functional.log_softmax(
                rejected_logits[0, prompt_len-1:-1, :].float(), dim=-1)
            rejected_logp = rejected_lps.gather(
                1, rejected_ids[0, prompt_len:].unsqueeze(1)).squeeze(1).sum()

            loss = -torch.nn.functional.logsigmoid(
                0.1 * (chosen_logp - rejected_logp))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            torch.cuda.empty_cache()

        print(f"  Epoch {epoch+1}/{num_epochs}: "
              f"DPO loss={epoch_loss/len(TRAIN_DATA):.4f}")

    # Post-training eval
    post_loss = compute_agent_loss(model, tokenizer, EVAL_EXAMPLES[:30], device)
    print(f"  Post-train agent loss: {post_loss:.4f}")
    print(f"  Agent loss change: {post_loss - pre_loss:+.4f}")

    del model, optimizer
    torch.cuda.empty_cache()

    return {
        "condition": condition,
        "seed": seed,
        "lora_targets": targets,
        "pre_loss": float(pre_loss),
        "post_loss": float(post_loss),
        "loss_change": float(post_loss - pre_loss),
        "trainable_params_M": trainable / 1e6,
    }


def run_model_seeds(model_name, base_dir, device, wandb_run=None):
    """Run all seeds and conditions for one model."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_results = []

    for seed in SEEDS:
        for condition in CONDITIONS:
            t0 = time.time()
            result = run_condition_seeded(
                base_dir, device, tokenizer, condition, seed)
            elapsed = time.time() - t0
            result["elapsed_sec"] = round(elapsed, 1)
            all_results.append(result)

            # Log to W&B
            if wandb_run is not None:
                wandb_run.log({
                    f"{condition}/seed_{seed}/pre_loss": result["pre_loss"],
                    f"{condition}/seed_{seed}/post_loss": result["post_loss"],
                    f"{condition}/seed_{seed}/loss_change": result["loss_change"],
                })

    # Compute statistics per condition
    condition_stats = {}
    for condition in CONDITIONS:
        cond_results = [r for r in all_results if r["condition"] == condition]
        changes = [r["loss_change"] for r in cond_results]
        pre_losses = [r["pre_loss"] for r in cond_results]
        post_losses = [r["post_loss"] for r in cond_results]

        condition_stats[condition] = {
            "loss_changes": changes,
            "pre_losses": pre_losses,
            "post_losses": post_losses,
            "mean": float(np.mean(changes)),
            "std": float(np.std(changes)),
            "min": float(np.min(changes)),
            "max": float(np.max(changes)),
            "per_seed": {
                r["seed"]: {
                    "pre_loss": r["pre_loss"],
                    "post_loss": r["post_loss"],
                    "loss_change": r["loss_change"],
                }
                for r in cond_results
            },
        }

    # Compute tax reduction statistics
    std_changes = condition_stats["standard"]["loss_changes"]
    ocdpo_changes = condition_stats["ocdpo_exclude_output"]["loss_changes"]

    # Per-seed tax reduction (paired comparison)
    tax_reductions = []
    for i, seed in enumerate(SEEDS):
        std_change = std_changes[i]
        ocdpo_change = ocdpo_changes[i]
        if abs(std_change) > 1e-6:
            reduction = (std_change - ocdpo_change) / abs(std_change) * 100
        else:
            reduction = 0.0
        tax_reductions.append(float(reduction))

    tax_reduction_stats = {
        "per_seed": {seed: pct for seed, pct in zip(SEEDS, tax_reductions)},
        "values": tax_reductions,
        "mean": float(np.mean(tax_reductions)),
        "std": float(np.std(tax_reductions)),
        "min": float(np.min(tax_reductions)),
        "max": float(np.max(tax_reductions)),
    }

    # Print summary
    print(f"\n{'='*70}")
    print(f"SEED EXPERIMENT SUMMARY: {model_name}")
    print(f"{'='*70}")

    for condition in CONDITIONS:
        stats = condition_stats[condition]
        print(f"\n  {condition}:")
        print(f"    Loss changes: {stats['loss_changes']}")
        print(f"    Mean +/- Std: {stats['mean']:+.4f} +/- {stats['std']:.4f}")

    print(f"\n  Tax reduction (OC-DPO vs standard):")
    print(f"    Per-seed: {tax_reductions}")
    print(f"    Mean +/- Std: {tax_reduction_stats['mean']:.1f}% "
          f"+/- {tax_reduction_stats['std']:.1f}%")
    print(f"    Range: [{tax_reduction_stats['min']:.1f}%, "
          f"{tax_reduction_stats['max']:.1f}%]")

    # Log summary to W&B
    if wandb_run is not None:
        wandb_run.log({
            "standard/mean_loss_change": condition_stats["standard"]["mean"],
            "standard/std_loss_change": condition_stats["standard"]["std"],
            "ocdpo/mean_loss_change": condition_stats["ocdpo_exclude_output"]["mean"],
            "ocdpo/std_loss_change": condition_stats["ocdpo_exclude_output"]["std"],
            "tax_reduction/mean_pct": tax_reduction_stats["mean"],
            "tax_reduction/std_pct": tax_reduction_stats["std"],
        })

    # Build output
    output = {
        "model": model_name,
        "seeds": SEEDS,
        "conditions": condition_stats,
        "tax_reduction_pct": tax_reduction_stats,
        "all_results": all_results,
    }

    # Save
    safe_name = model_name.replace("/", "_").replace(" ", "_").lower()
    out_path = RESULTS_DIR / f"step16_ocdpo_seeds_{safe_name}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")

    return output


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    print(f"Device: {device}")
    print(f"Seeds: {SEEDS}")
    print(f"Conditions: {CONDITIONS}")
    print(f"Models: {[m[0] for m in MODELS]}")

    import wandb
    wandb.login()  # uses WANDB_API_KEY env var

    all_model_results = {}

    for model_name, base_dir in MODELS:
        print(f"\n{'#'*70}")
        print(f"# MODEL: {model_name}")
        print(f"# Base: {base_dir}")
        print(f"{'#'*70}")

        run = wandb.init(
            project="anonymous-submission",
            name=f"step16-ocdpo-seeds-{model_name}",
            config={
                "step": 16,
                "experiment": "ocdpo_seeds",
                "model": model_name,
                "seeds": SEEDS,
                "conditions": CONDITIONS,
                "num_epochs": 5,
                "lr": 5e-5,
                "lora_rank": 16,
                "lora_alpha": 32,
                "num_train_pairs": len(TRAIN_DATA),
            },
            reinit=True,
        )

        output = run_model_seeds(model_name, base_dir, device, wandb_run=run)
        all_model_results[model_name] = output

        run.finish()

    # Final cross-model summary
    print(f"\n{'#'*70}")
    print(f"# CROSS-MODEL SUMMARY")
    print(f"{'#'*70}")

    all_tax_reductions = []
    for model_name, output in all_model_results.items():
        tr = output["tax_reduction_pct"]
        all_tax_reductions.extend(tr["values"])
        print(f"\n  {model_name}:")
        print(f"    Standard DPO loss change: "
              f"{output['conditions']['standard']['mean']:+.4f} "
              f"+/- {output['conditions']['standard']['std']:.4f}")
        print(f"    OC-DPO loss change:       "
              f"{output['conditions']['ocdpo_exclude_output']['mean']:+.4f} "
              f"+/- {output['conditions']['ocdpo_exclude_output']['std']:.4f}")
        print(f"    Tax reduction: {tr['mean']:.1f}% +/- {tr['std']:.1f}%")

    print(f"\n  Overall (all models, all seeds):")
    print(f"    Tax reduction: {np.mean(all_tax_reductions):.1f}% "
          f"+/- {np.std(all_tax_reductions):.1f}%")
    print(f"    N = {len(all_tax_reductions)} "
          f"(3 models x {len(SEEDS)} seeds)")

    # Save combined results
    combined_path = RESULTS_DIR / "step16_ocdpo_seeds_combined.json"
    combined = {
        "models": list(all_model_results.keys()),
        "seeds": SEEDS,
        "per_model": {
            name: {
                "standard_mean": out["conditions"]["standard"]["mean"],
                "standard_std": out["conditions"]["standard"]["std"],
                "ocdpo_mean": out["conditions"]["ocdpo_exclude_output"]["mean"],
                "ocdpo_std": out["conditions"]["ocdpo_exclude_output"]["std"],
                "tax_reduction_mean": out["tax_reduction_pct"]["mean"],
                "tax_reduction_std": out["tax_reduction_pct"]["std"],
            }
            for name, out in all_model_results.items()
        },
        "overall_tax_reduction": {
            "mean": float(np.mean(all_tax_reductions)),
            "std": float(np.std(all_tax_reductions)),
            "n": len(all_tax_reductions),
        },
    }
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nSaved combined to {combined_path}")


if __name__ == "__main__":
    main()
