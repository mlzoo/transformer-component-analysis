"""
Random Baseline for SAR (Multi-Seed)

Computes random component rollback at k=5% using N seeds (default 10).
Reports mean, std, and bootstrap 95% CI for recovery percentage and
SAR/random ratio. Used to support Table 4 claims.

Usage:
    python experiments/06_random_baseline.py <base_dir> <it_dir> <device> <model_name> [n_seeds]

Example:
    python experiments/06_random_baseline.py ./models/Qwen2.5-7B ./models/Qwen2.5-7B-Instruct cuda:0 qwen 10
"""

import sys
import json
import torch
import gc
import numpy as np
from pathlib import Path
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).parent))

RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(exist_ok=True)

from agent_examples_200 import BASE_EXAMPLES, EXTRA_EXAMPLES
AGENT_EXAMPLES = BASE_EXAMPLES + EXTRA_EXAMPLES

PROJ_MAP = {
    "W_Q": "self_attn.q_proj", "W_K": "self_attn.k_proj",
    "W_V": "self_attn.v_proj", "W_O": "self_attn.o_proj",
    "W_gate": "mlp.gate_proj", "W_up": "mlp.up_proj",
    "W_down": "mlp.down_proj",
}


def comp_to_weight_key(comp_type, layer):
    proj_path = PROJ_MAP[comp_type]
    return f"model.layers.{layer}.{proj_path}.weight"


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


def apply_random_sar(model, base_index, all_components, k_percent, seed):
    rng = np.random.RandomState(seed)
    n_select = max(1, int(len(all_components) * k_percent / 100))
    selected_idx = rng.choice(len(all_components), size=n_select, replace=False)
    selected = [all_components[i] for i in selected_idx]

    param_dict = dict(model.named_parameters())
    for comp in selected:
        weight_key = comp_to_weight_key(comp["component_type"], comp["layer"])
        if weight_key not in base_index or weight_key not in param_dict:
            continue
        with safe_open(str(base_index[weight_key]), framework="pt", device="cpu") as sf:
            base_w = sf.get_tensor(weight_key)
        param_dict[weight_key].data.copy_(base_w.to(param_dict[weight_key].dtype).to(param_dict[weight_key].device))


def bootstrap_ci(values, n_boot=10000, alpha=0.05, seed=42):
    rng = np.random.RandomState(seed)
    n = len(values)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot_means[b] = np.mean([values[i] for i in idx])
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def run_random_baseline(base_dir, it_dir, device, model_name, n_seeds=10):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    attr_path = RESULTS_DIR / f"attribution_{model_name}.json"
    if not attr_path.exists():
        print(f"ERROR: Attribution file not found at {attr_path}")
        print("Run 01_attribution.py first.")
        sys.exit(1)

    with open(attr_path) as f:
        attr_data = json.load(f)
    all_components = attr_data["components"]
    print(f"Loaded {len(all_components)} components from attribution")

    # Build base model weight index
    base_index = {}
    for f_path in sorted(Path(base_dir).glob("*.safetensors")):
        with safe_open(str(f_path), framework="pt", device="cpu") as sf:
            for key in sf.keys():
                base_index[key] = f_path
    print(f"Base model index: {len(base_index)} tensors")

    # Load IT model
    print(f"Loading IT model from {it_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(it_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Get IT baseline loss
    model = AutoModelForCausalLM.from_pretrained(
        it_dir, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
    )
    model.eval()
    it_loss = compute_loss(model, tokenizer, AGENT_EXAMPLES, device)
    print(f"IT baseline loss: {it_loss:.4f}")

    # Get base model loss
    base_model = AutoModelForCausalLM.from_pretrained(
        base_dir, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
    )
    base_model.eval()
    base_loss = compute_loss(base_model, tokenizer, AGENT_EXAMPLES, device)
    print(f"Base model loss: {base_loss:.4f}")
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    alignment_tax = it_loss - base_loss

    # Run random baseline at k=5% for n_seeds
    k_percent = 5.0
    random_recoveries = []
    random_losses = []

    for seed in range(n_seeds):
        # Reload fresh IT model each time
        if seed > 0:
            del model
            gc.collect()
            torch.cuda.empty_cache()
            model = AutoModelForCausalLM.from_pretrained(
                it_dir, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
            )
            model.eval()

        apply_random_sar(model, base_index, all_components, k_percent, seed)
        loss = compute_loss(model, tokenizer, AGENT_EXAMPLES, device)
        recovery = (it_loss - loss) / abs(alignment_tax) * 100 if abs(alignment_tax) > 1e-9 else 0
        random_recoveries.append(recovery)
        random_losses.append(loss)
        print(f"  Seed {seed}: loss={loss:.4f}, recovery={recovery:.1f}%")

    # Also get SAR-5% (top-k attribution) for ratio computation
    del model
    gc.collect()
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(
        it_dir, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
    )
    model.eval()

    sorted_comps = sorted(all_components, key=lambda x: x["harm_score"], reverse=True)
    n_select = max(1, int(len(sorted_comps) * k_percent / 100))
    top_k_comps = sorted_comps[:n_select]

    param_dict = dict(model.named_parameters())
    for comp in top_k_comps:
        weight_key = comp_to_weight_key(comp["component_type"], comp["layer"])
        if weight_key not in base_index or weight_key not in param_dict:
            continue
        with safe_open(str(base_index[weight_key]), framework="pt", device="cpu") as sf:
            base_w = sf.get_tensor(weight_key)
        param_dict[weight_key].data.copy_(base_w.to(param_dict[weight_key].dtype).to(param_dict[weight_key].device))

    sar_loss = compute_loss(model, tokenizer, AGENT_EXAMPLES, device)
    sar_recovery = (it_loss - sar_loss) / abs(alignment_tax) * 100 if abs(alignment_tax) > 1e-9 else 0
    print(f"\n  SAR-5% loss={sar_loss:.4f}, recovery={sar_recovery:.1f}%")

    # Compute SAR/random ratio with bootstrap CI
    mean_random = np.mean(random_recoveries)
    std_random = np.std(random_recoveries)
    ratio = sar_recovery / mean_random if abs(mean_random) > 1e-9 else float('inf')

    # Bootstrap CI for the ratio
    ratio_boots = []
    rng = np.random.RandomState(42)
    for _ in range(10000):
        boot_random = rng.choice(random_recoveries, size=n_seeds, replace=True)
        boot_mean = np.mean(boot_random)
        if abs(boot_mean) > 1e-9:
            ratio_boots.append(sar_recovery / boot_mean)
    ratio_ci = (float(np.percentile(ratio_boots, 2.5)), float(np.percentile(ratio_boots, 97.5)))

    random_ci = bootstrap_ci(random_recoveries)

    results = {
        "analysis": f"Random baseline ({n_seeds} seeds) vs SAR-5%",
        "model": model_name,
        "n_seeds": n_seeds,
        "k_percent": k_percent,
        "alignment_tax": float(alignment_tax),
        "it_loss": float(it_loss),
        "base_loss": float(base_loss),
        "sar_5pct": {
            "loss": float(sar_loss),
            "recovery_pct": float(sar_recovery),
            "n_components": n_select,
        },
        "random_baseline": {
            "mean_recovery_pct": float(mean_random),
            "std_recovery_pct": float(std_random),
            "ci_95": random_ci,
            "per_seed_recovery": [float(r) for r in random_recoveries],
            "per_seed_loss": [float(l) for l in random_losses],
        },
        "sar_over_random_ratio": {
            "ratio": float(ratio),
            "bootstrap_95_ci": ratio_ci,
        },
    }

    out_path = RESULTS_DIR / f"random_baseline_{model_name}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")
    print(f"Random recovery: {mean_random:.1f} ± {std_random:.1f}%")
    print(f"SAR/random ratio: {ratio:.1f}x (95% CI: [{ratio_ci[0]:.1f}, {ratio_ci[1]:.1f}])")

    del model
    gc.collect()


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python 06_random_baseline.py <base_dir> <it_dir> <device> <model_name> [n_seeds]")
        sys.exit(1)

    base_dir = sys.argv[1]
    it_dir = sys.argv[2]
    device = sys.argv[3]
    model_name = sys.argv[4] if len(sys.argv) > 4 else "unknown"
    n_seeds = int(sys.argv[5]) if len(sys.argv) > 5 else 10

    run_random_baseline(base_dir, it_dir, device, model_name, n_seeds)
