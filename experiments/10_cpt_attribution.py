"""
Continual Pretraining Attribution Control
Test whether output-pathway concentration is alignment-SPECIFIC or
a general property of any fine-tuning (residual proximity hypothesis).

Method: Fine-tune base model on random text (continual pretraining, no alignment),
then run the same weight-patching attribution. If W_down > W_up > W_gate hierarchy
persists, the concentration is architectural rather than alignment-specific.
"""

import sys
import json
import torch
import random
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent))
from agent_examples_200 import AGENT_EXAMPLES_200
from weight_utils import classify_component, compute_ce_loss, get_model_params

MODEL_CONFIGS = {
    "qwen2.5-7b": {
        "base": "./models/Qwen2.5-7B",
        "it": "./models/Qwen2.5-7B-Instruct",
    },
    "llama-3.1-8b": {
        "base": "./models/Llama-3.1-8B",
        "it": "./models/Llama-3.1-8B-Instruct",
    },
}


def continual_pretrain(model, tokenizer, device, n_steps=500, lr=1e-5, max_length=256):
    """Run continual pretraining using LoRA to fit within 24GB GPU memory."""
    from peft import LoraConfig, get_peft_model

    print(f"    Loading text data for continual pretraining...")

    texts = []
    try:
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        for item in dataset:
            if len(item["text"]) > 200:
                texts.append(item["text"][:2000])
            if len(texts) >= n_steps * 2:
                break
    except Exception as e:
        print(f"    Wikitext unavailable ({e}), using synthetic text fallback...")
        random.seed(42)
        vocab = list(tokenizer.get_vocab().keys())[:5000]
        for _ in range(n_steps * 2):
            length = random.randint(100, 400)
            texts.append(" ".join(random.choices(vocab, k=length)))

    if not texts:
        raise RuntimeError("No training data available")

    print(f"    Loaded {len(texts)} texts, training for {n_steps} steps with LoRA...")

    # Apply LoRA to ALL linear layers so we can measure attribution on all components
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    losses = []
    for step in range(n_steps):
        text = texts[step % len(texts)]
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = enc["input_ids"].to(device)

        if input_ids.shape[1] < 10:
            continue

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())

        if (step + 1) % 100 == 0:
            recent = np.mean(losses[-100:])
            print(f"      Step {step+1}/{n_steps}, loss={recent:.4f}")

    # Merge LoRA weights into the base model for attribution
    model = model.merge_and_unload()
    model.gradient_checkpointing_disable()
    model.eval()
    return model, losses


def compute_attribution_scores(model, base_state, tokenizer, examples, device):
    """Compute per-component harm scores using correct named_parameters approach."""
    params = get_model_params(model)
    projections = [n for n in params if classify_component(n)[0] is not None]

    baseline_loss = compute_ce_loss(model, tokenizer, examples, device)
    scores = {}

    for i, name in enumerate(projections):
        original = params[name].data.clone()
        base_weight = base_state[name].to(device)

        with torch.no_grad():
            params[name].data.copy_(base_weight)

        patched_loss = compute_ce_loss(model, tokenizer, examples, device)
        scores[name] = baseline_loss - patched_loss

        with torch.no_grad():
            params[name].data.copy_(original)

        if (i + 1) % 50 == 0:
            print(f"      Attribution: {i+1}/{len(projections)}")

    return scores, baseline_loss


def analyze_hierarchy(scores):
    """Analyze component-type hierarchy from attribution scores."""
    type_scores = {}
    type_counts = {}
    for name, score in scores.items():
        comp_type, _ = classify_component(name)
        if comp_type:
            type_scores.setdefault(comp_type, []).append(score)
            type_counts[comp_type] = type_counts.get(comp_type, 0) + 1

    type_means = {k: float(np.mean(v)) for k, v in type_scores.items()}
    total_abs = sum(abs(np.sum(v)) for v in type_scores.values())
    type_shares = {k: abs(np.sum(v)) / total_abs * 100 for k, v in type_scores.items()} if total_abs > 0 else {}

    # V/O vs Q/K ratio
    vo_scores = type_scores.get("W_V", []) + type_scores.get("W_O", [])
    qk_scores = type_scores.get("W_Q", []) + type_scores.get("W_K", [])
    vo_mean = float(np.mean(vo_scores)) if vo_scores else 0
    qk_mean = float(np.mean(qk_scores)) if qk_scores else 0
    vo_qk_ratio = vo_mean / qk_mean if abs(qk_mean) > 1e-8 else 0.0

    # MLP hierarchy
    mlp_shares = {k: type_shares.get(k, 0) for k in ["W_down", "W_up", "W_gate"]}
    mlp_total = sum(mlp_shares.values())
    attn_shares = {k: type_shares.get(k, 0) for k in ["W_V", "W_O", "W_Q", "W_K"]}
    attn_total = sum(attn_shares.values())

    return {
        "type_shares": {k: float(v) for k, v in type_shares.items()},
        "type_means": type_means,
        "mlp_total": float(mlp_total),
        "attn_total": float(attn_total),
        "vo_qk_ratio": float(vo_qk_ratio),
        "vo_mean": float(vo_mean),
        "qk_mean": float(qk_mean),
        "mlp_hierarchy": {k: float(v) for k, v in mlp_shares.items()},
        "w_down_gt_w_up": bool(mlp_shares.get("W_down", 0) > mlp_shares.get("W_up", 0)),
        "w_up_gt_w_gate": bool(mlp_shares.get("W_up", 0) > mlp_shares.get("W_gate", 0)),
        "output_pathway_dominant": bool(
            type_shares.get("W_V", 0) + type_shares.get("W_O", 0) + type_shares.get("W_down", 0)
            > type_shares.get("W_Q", 0) + type_shares.get("W_K", 0) + type_shares.get("W_gate", 0) + type_shares.get("W_up", 0)
        ),
    }


def main():
    model_key = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5-7b"
    device = sys.argv[2] if len(sys.argv) > 2 else "cuda:0"
    n_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    lr = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-5

    config = MODEL_CONFIGS[model_key]
    print(f"={'='*60}")
    print(f"Continual Pretraining Attribution Control ({model_key})")
    print(f"  Device: {device}")
    print(f"  Training steps: {n_steps}, lr: {lr}")
    print(f"={'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(config["base"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    examples = AGENT_EXAMPLES_200[:209]
    print(f"  Loaded {len(examples)} evaluation examples")

    # Load base model and save state
    print(f"\n  Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        config["base"], torch_dtype=torch.float16, device_map=device, trust_remote_code=True
    )
    base_state = {k: v.cpu() for k, v in base_model.state_dict().items()}

    # Run continual pretraining on the base model
    print(f"\n  Running continual pretraining ({n_steps} steps, lr={lr})...")
    base_model, train_losses = continual_pretrain(base_model, tokenizer, device, n_steps=n_steps, lr=lr)

    # Now base_model has been modified by CPT — compute attribution
    print(f"\n  Computing CPT attribution (CPT model vs original base)...")
    cpt_scores, cpt_baseline = compute_attribution_scores(
        base_model, base_state, tokenizer, examples, device
    )
    cpt_hierarchy = analyze_hierarchy(cpt_scores)

    # Free CPT model, load IT model
    del base_model
    torch.cuda.empty_cache()

    print(f"\n  Loading IT model for comparison...")
    it_model = AutoModelForCausalLM.from_pretrained(
        config["it"], torch_dtype=torch.float16, device_map=device, trust_remote_code=True
    )

    # Move base_state back to compare
    print(f"  Computing IT attribution (IT model vs base)...")
    it_scores, it_baseline = compute_attribution_scores(
        it_model, base_state, tokenizer, examples, device
    )
    it_hierarchy = analyze_hierarchy(it_scores)

    del it_model
    torch.cuda.empty_cache()

    # Statistical comparison: correlation between CPT and IT score vectors
    common_params = sorted(set(cpt_scores.keys()) & set(it_scores.keys()))
    cpt_vec = np.array([cpt_scores[p] for p in common_params])
    it_vec = np.array([it_scores[p] for p in common_params])
    if len(common_params) > 1 and np.std(cpt_vec) > 0 and np.std(it_vec) > 0:
        correlation = float(np.corrcoef(cpt_vec, it_vec)[0, 1])
    else:
        correlation = 0.0

    # Determine conclusion
    cpt_output_dominant = cpt_hierarchy["output_pathway_dominant"]
    it_output_dominant = it_hierarchy["output_pathway_dominant"]

    if cpt_output_dominant and it_output_dominant:
        conclusion = (
            "Output-pathway concentration persists under non-alignment fine-tuning. "
            "This suggests the pattern is partially architectural/optimization-driven, "
            "not purely alignment-specific. The alignment tax mechanism overlaps with "
            "a general fine-tuning pattern."
        )
    elif it_output_dominant and not cpt_output_dominant:
        conclusion = (
            "Output-pathway concentration is specific to alignment (IT) fine-tuning. "
            "Continual pretraining does NOT produce the same hierarchy, supporting "
            "the hypothesis that instruction tuning specifically concentrates changes "
            "in output-pathway components."
        )
    else:
        conclusion = (
            "Mixed results: neither condition shows clear output-pathway dominance. "
            "Further investigation needed with more training steps or different data."
        )

    results = {
        "experiment": "Continual Pretraining Attribution Control",
        "model": model_key,
        "n_steps": n_steps,
        "n_examples": len(examples),
        "training": {
            "final_loss": float(np.mean(train_losses[-20:])) if train_losses else 0.0,
            "initial_loss": float(np.mean(train_losses[:5])) if train_losses else 0.0,
            "loss_reduction": float(np.mean(train_losses[:5]) - np.mean(train_losses[-20:])) if len(train_losses) > 20 else 0.0,
        },
        "continual_pretraining": {
            "hierarchy": cpt_hierarchy,
            "baseline_loss": float(cpt_baseline),
        },
        "instruction_tuning": {
            "hierarchy": it_hierarchy,
            "baseline_loss": float(it_baseline),
        },
        "comparison": {
            "score_correlation": correlation,
            "both_output_pathway_dominant": cpt_output_dominant and it_output_dominant,
            "cpt_output_dominant": cpt_output_dominant,
            "it_output_dominant": it_output_dominant,
            "cpt_mlp_pct": float(cpt_hierarchy["mlp_total"]),
            "it_mlp_pct": float(it_hierarchy["mlp_total"]),
            "cpt_vo_qk_ratio": float(cpt_hierarchy["vo_qk_ratio"]),
            "it_vo_qk_ratio": float(it_hierarchy["vo_qk_ratio"]),
            "cpt_wdown_gt_wup": cpt_hierarchy["w_down_gt_w_up"],
            "it_wdown_gt_wup": it_hierarchy["w_down_gt_w_up"],
            "conclusion": conclusion,
        },
    }

    print(f"\n  {'='*50}")
    print(f"  RESULTS:")
    print(f"  {'='*50}")
    print(f"  Continual Pretraining:")
    print(f"    MLP total: {cpt_hierarchy['mlp_total']:.1f}%")
    print(f"    W_down > W_up: {cpt_hierarchy['w_down_gt_w_up']}")
    print(f"    V/O:Q/K ratio: {cpt_hierarchy['vo_qk_ratio']:.2f}")
    print(f"    Output-pathway dominant: {cpt_output_dominant}")
    print(f"  Instruction Tuning:")
    print(f"    MLP total: {it_hierarchy['mlp_total']:.1f}%")
    print(f"    W_down > W_up: {it_hierarchy['w_down_gt_w_up']}")
    print(f"    V/O:Q/K ratio: {it_hierarchy['vo_qk_ratio']:.2f}")
    print(f"    Output-pathway dominant: {it_output_dominant}")
    print(f"  Score correlation: {correlation:.3f}")
    print(f"\n  Conclusion: {conclusion}")

    out_path = f"./results/cpt_attribution_{model_key}.json"

    class _NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_, np.integer)):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, cls=_NumpyEncoder)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
