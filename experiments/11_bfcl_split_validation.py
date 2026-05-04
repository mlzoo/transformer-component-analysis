"""
Within-BFCL Split Validation

Addresses the cross-dataset transfer criticism: heuristic baseline (46.8%) nearly
matches BFCL→probe transfer (47%), suggesting attribution at k=5% adds no value
beyond structural priors.

Design:
  - Take all 318 BFCL examples (simple x100 + multiple x100 + parallel x100 + relevance x18)
  - For each of 10 random splits: shuffle, take first 159 as attribution half,
    remaining 159 as evaluation half
  - Run activation-patching attribution on the attribution half → select top-k% components
  - Evaluate SAR recovery on the evaluation half via CE loss
  - Compare against (a) uniform random baseline (10 seeds), (b) structural heuristic
    (W_V, W_O, W_down in middle third of layers), both selected to same k count
  - Report: attribution recovery %, random mean±std, heuristic recovery %
  - Robustness check: repeat across all 10 outer splits

Attribution method: activation patching — for each component (layer, type), we
register a forward hook that subtracts the IT-vs-base weight delta from the
output (equivalent to reverting that component to base weights), then measure
the change in CE loss. Components with the largest positive harm_score (i.e.
loss drops most when reverted) are the most alignment-responsible.

SAR application: permanently replace IT weights with base weights for selected
components, then measure CE loss on the held-out evaluation half.

Recovery metric: fraction of alignment tax recovered = (it_loss - sar_loss) / (it_loss - base_loss)
    Positive = recovery toward base, Negative = made worse, 0% = no effect

Usage:
    python bfcl_split_validation.py <cuda_device> [n_splits] [k_pct]
    e.g. python bfcl_split_validation.py cuda:0
         python bfcl_split_validation.py cuda:1 10 5
"""

import sys
import json
import time
import gc
import re
import ast as ast_mod
import torch
import numpy as np
from pathlib import Path
from safetensors import safe_open

RESULTS_DIR = Path("./results")
BFCL_DIR = Path("./data/bfcl")
BASE_MODEL = "./models/Qwen2.5-7B"
IT_MODEL = "./models/Qwen2.5-7B-Instruct"

PROJ_MAP = {
    "W_Q": "self_attn.q_proj",
    "W_K": "self_attn.k_proj",
    "W_V": "self_attn.v_proj",
    "W_O": "self_attn.o_proj",
    "W_gate": "mlp.gate_proj",
    "W_up": "mlp.up_proj",
    "W_down": "mlp.down_proj",
}


# ---------------------------------------------------------------------------
# BFCL data loading  (reuses BFCL loading patterns)
# ---------------------------------------------------------------------------

def load_bfcl_examples():
    """Load up to 100 examples per category from local BFCL snapshot.

    Returns a flat list of dicts with keys: id, prompt, target, category.
    The prompt/target pair is formatted for CE loss evaluation:
      prompt = function definitions + user request header
      target = ground-truth function call string(s)
    """
    category_limits = {"simple": 100, "multiple": 100, "parallel": 100, "relevance": 18}
    file_map = {
        "simple":    "BFCL_v3_simple.json",
        "multiple":  "BFCL_v3_multiple.json",
        "parallel":  "BFCL_v3_parallel.json",
        "relevance": "BFCL_v3_live_relevance.json",
    }

    all_examples = []

    for cat, limit in category_limits.items():
        q_path = BFCL_DIR / file_map[cat]
        gt_path = BFCL_DIR / "possible_answer" / file_map[cat]

        if not q_path.exists():
            print(f"  WARNING: {q_path} not found — skipping {cat}")
            continue

        with open(q_path) as f:
            questions = [json.loads(l) for l in f if l.strip()]

        gt_by_id = {}
        if gt_path.exists():
            with open(gt_path) as f:
                for l in f:
                    if l.strip():
                        obj = json.loads(l)
                        gt_by_id[obj["id"]] = obj.get("ground_truth", [])

        for q in questions[:limit]:
            qid = q["id"]
            # Extract user message from [[{role,content}]] structure
            question_text = ""
            msgs = q.get("question", [])
            if isinstance(msgs, list) and msgs:
                turn = msgs[0]
                if isinstance(turn, list):
                    for msg in turn:
                        if isinstance(msg, dict) and msg.get("role") == "user":
                            question_text = msg["content"]
                elif isinstance(turn, dict):
                    question_text = turn.get("content", "")

            funcs = q.get("function", [])
            gt = gt_by_id.get(qid, [])

            prompt = _build_prompt(question_text, funcs, cat)
            target = _build_target(gt, cat)

            if prompt and target and len(target) > 2:
                all_examples.append({
                    "id": qid,
                    "prompt": prompt,
                    "target": target,
                    "category": cat,
                })

    print(f"  Loaded {len(all_examples)} BFCL examples "
          f"({dict((c, sum(1 for e in all_examples if e['category']==c)) for c in category_limits)})")
    return all_examples


def _format_functions(functions):
    lines = []
    for fn in functions:
        name = fn.get("name", "?")
        desc = fn.get("description", "")
        props = fn.get("parameters", {}).get("properties", {})
        required = fn.get("parameters", {}).get("required", [])
        lines.append(f"Function: {name}")
        lines.append(f"  Description: {desc}")
        for pname, pinfo in props.items():
            req = "(required)" if pname in required else "(optional)"
            lines.append(f"    - {pname} ({pinfo.get('type','any')}, {req}): {pinfo.get('description','')}")
        lines.append("")
    return "\n".join(lines)


def _build_prompt(question_text, functions, category):
    func_text = _format_functions(functions)
    if category == "relevance":
        return (
            f"You have access to the following functions:\n\n{func_text}\n"
            f"User request: {question_text}\n\n"
            "If no function is suitable, respond with exactly: NO_FUNCTION_AVAILABLE\n\nResponse:"
        )
    return (
        f"You have access to the following functions:\n\n{func_text}\n"
        f"User request: {question_text}\n\n"
        "Call the appropriate function(s). Output ONLY the function call(s):\n\nFunction call:"
    )


def _build_target(ground_truth, category):
    """Convert ground truth to a string the model should produce."""
    if category == "relevance":
        return "NO_FUNCTION_AVAILABLE"
    parts = []
    for entry in ground_truth:
        if isinstance(entry, dict):
            for fname, fargs in entry.items():
                arg_strs = []
                for k, v in fargs.items():
                    val = v[0] if isinstance(v, list) and v else v
                    arg_strs.append(f"{k}={json.dumps(val)}")
                parts.append(f"{fname}({', '.join(arg_strs)})")
    return "\n".join(parts) if parts else ""


# ---------------------------------------------------------------------------
# CE loss computation
# ---------------------------------------------------------------------------

def compute_ce_loss(model, tokenizer, examples, device, max_length=512):
    """Token-level CE loss: sum(losses) / total_tokens. Consistent with other evaluation scripts."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for ex in examples:
            full_text = ex["prompt"] + ex["target"]
            enc = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = enc["input_ids"].to(device)
            prompt_len = tokenizer(ex["prompt"], return_tensors="pt",
                                   truncation=True, max_length=max_length)["input_ids"].shape[1]
            if prompt_len >= input_ids.shape[1]:
                continue
            out = model(input_ids=input_ids)
            shift_logits = out.logits[0, prompt_len - 1:-1, :]
            shift_labels = input_ids[0, prompt_len:]
            loss = torch.nn.functional.cross_entropy(shift_logits, shift_labels, reduction="sum")
            total_loss += loss.item()
            total_tokens += shift_labels.shape[0]
            del enc, input_ids, out, shift_logits, shift_labels, loss
    torch.cuda.empty_cache()
    return total_loss / max(total_tokens, 1)


# ---------------------------------------------------------------------------
# Attribution (activation-patching per component)
# ---------------------------------------------------------------------------

def run_attribution(model, tokenizer, examples, base_index, device):
    """Run per-component activation-patching attribution.

    For each (layer, component_type) we:
      1. Compute base_weight - it_weight delta
      2. Register a forward hook that subtracts F(x, delta) from the output
         (effectively reverting that component to base weights)
      3. Measure change in CE loss vs IT baseline
      4. harm_score = baseline_loss - patched_loss
         (positive → removing this component's alignment reduces loss → harmful to structured gen)

    Returns:
      components: list of {weight_key, component_type, layer, harm_score}
      baseline_loss: float
    """
    num_layers = model.config.num_hidden_layers

    baseline_loss = compute_ce_loss(model, tokenizer, examples, device)

    components = []
    total = num_layers * len(PROJ_MAP)
    done = 0

    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        for comp_type, proj_path in PROJ_MAP.items():
            weight_key = f"model.layers.{layer_idx}.{proj_path}.weight"
            if weight_key not in base_index:
                done += 1
                continue

            # Get module
            module = layer
            for part in proj_path.split("."):
                module = getattr(module, part)

            # Compute delta on CPU
            with safe_open(str(base_index[weight_key]), framework="pt", device="cpu") as sf:
                base_w = sf.get_tensor(weight_key).float()
            it_w = module.weight.data.float().cpu()
            delta = it_w - base_w  # shape [out, in]

            # Hook subtracts the delta contribution from the layer output.
            # d is moved to GPU in the model's dtype to keep the correction
            # in the same precision/device as `out`, avoiding CPU round-trips
            # and dtype mismatches that could perturb harm_score rankings.
            def make_hook(d):
                def hook_fn(mod, inp, out):
                    x = inp[0] if isinstance(inp, tuple) else inp
                    correction = torch.nn.functional.linear(
                        x, d.to(device=x.device, dtype=x.dtype)
                    )
                    return out - correction
                return hook_fn

            hook = module.register_forward_hook(make_hook(delta))
            patched_loss = compute_ce_loss(model, tokenizer, examples, device)
            hook.remove()

            harm_score = float(baseline_loss - patched_loss)
            components.append({
                "weight_key": weight_key,
                "component_type": comp_type,
                "layer": layer_idx,
                "harm_score": harm_score,
            })

            done += 1
            if done % 28 == 0:
                print(f"    Attribution: {done}/{total} components done", flush=True)

            del base_w, it_w, delta
            gc.collect()
            torch.cuda.empty_cache()

    return components, baseline_loss


# ---------------------------------------------------------------------------
# SAR application (weight replacement)
# ---------------------------------------------------------------------------

def select_top_k(components, k_count):
    """Return top k_count components by harm_score."""
    return sorted(components, key=lambda c: c["harm_score"], reverse=True)[:k_count]


def select_heuristic(components, n_layers, n_select):
    """Structural heuristic: W_V, W_O, W_down in middle third of layers.
    If that pool is smaller than n_select, pad with remaining layers (same types).
    """
    mid_start = n_layers // 3
    mid_end = 2 * n_layers // 3
    pool = [c for c in components
            if c["component_type"] in ("W_V", "W_O", "W_down")
            and mid_start <= c["layer"] < mid_end]
    if len(pool) >= n_select:
        return pool[:n_select]
    # Pad: add remaining W_V/W_O/W_down outside middle, then any remaining types
    extra = [c for c in components
             if c["component_type"] in ("W_V", "W_O", "W_down") and c not in pool]
    pool = pool + extra
    if len(pool) >= n_select:
        return pool[:n_select]
    # Still short — pad with remaining components by layer index (no attribution leakage)
    rest = [c for c in sorted(components, key=lambda x: x["layer"])
            if c not in pool]
    return (pool + rest)[:n_select]


def select_random(components, n_select, rng):
    """Uniform random selection of n_select components (no stratification)."""
    idx = rng.choice(len(components), size=n_select, replace=False)
    return [components[i] for i in idx]


def apply_sar_inplace(model_state, base_weights, selected):
    """Replace IT weights with base weights for selected components in-place."""
    for comp in selected:
        wk = comp["weight_key"]
        if wk in base_weights and wk in model_state:
            p = model_state[wk]
            base_t = base_weights[wk].to(device=p.device, dtype=p.dtype)
            p.data.copy_(base_t)
            del base_t


def restore_it_inplace(model_state, it_weights, selected):
    """Restore IT weights for selected components in-place."""
    for comp in selected:
        wk = comp["weight_key"]
        if wk in it_weights and wk in model_state:
            p = model_state[wk]
            it_t = it_weights[wk].to(device=p.device, dtype=p.dtype)
            p.data.copy_(it_t)
            del it_t


def compute_recovery(it_loss, base_loss, sar_loss):
    """Fraction of alignment tax recovered (can be > 1.0 or negative)."""
    tax = it_loss - base_loss
    if abs(tax) < 1e-8:
        return 0.0
    return (it_loss - sar_loss) / tax


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    n_splits = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    k_pct = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
    max_attr_ex = int(sys.argv[4]) if len(sys.argv) > 4 else 60
    n_random_seeds = 10

    print("=" * 70)
    print("Within-BFCL Split Validation")
    print(f"  Device:   {device}")
    print(f"  Splits:   {n_splits}")
    print(f"  k_pct:    {k_pct}%")
    print(f"  Max attribution examples: {max_attr_ex}")
    print(f"  Random seeds per split: {n_random_seeds}")
    print("=" * 70)

    t_start = time.time()

    # --- Load BFCL examples ---
    print("\nLoading BFCL examples...")
    all_examples = load_bfcl_examples()
    n_all = len(all_examples)
    print(f"  Total examples: {n_all}")
    assert n_all >= 50, f"Too few BFCL examples: {n_all}"

    # --- Load models ---
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\nLoading tokenizer from {IT_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(IT_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading IT model from {IT_MODEL}...")
    model = AutoModelForCausalLM.from_pretrained(
        IT_MODEL, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    total_components = n_layers * len(PROJ_MAP)
    k_count = max(1, int(total_components * k_pct / 100))
    print(f"  Layers: {n_layers}, Total components: {total_components}, k={k_count}")

    # --- Index base model weights ---
    print(f"\nIndexing base weights from {BASE_MODEL}...")
    base_index = {}
    for sf in sorted(Path(BASE_MODEL).glob("*.safetensors")):
        with safe_open(str(sf), framework="pt", device="cpu") as f:
            for key in f.keys():
                base_index[key] = sf
    print(f"  Indexed {len(base_index)} base weight tensors")

    # --- Pre-load all base + IT weights into CPU RAM for fast rollback ---
    # (28 layers × 7 projections × ~14M params each ≈ 7.7GB in fp16)
    # We only load the projection weight keys to save memory.
    print("\nLoading all base projection weights into CPU RAM...")
    base_weights = {}
    it_weights_cpu = {}
    model_state = dict(model.named_parameters())
    for key in base_index:
        if key in model_state:
            with safe_open(str(base_index[key]), framework="pt", device="cpu") as sf:
                base_weights[key] = sf.get_tensor(key)
    for key in base_weights:
        it_weights_cpu[key] = model_state[key].data.clone().cpu()
    print(f"  Loaded {len(base_weights)} base tensors, {len(it_weights_cpu)} IT tensors")

    # --- Compute full-data base loss once for reference ---
    print("\nComputing full-data IT baseline loss (for reference)...")
    it_loss_full = compute_ce_loss(model, tokenizer, all_examples, device)
    print(f"  IT loss (all {n_all} examples): {it_loss_full:.5f}")

    # Temporarily apply all base weights to get base loss
    for k in base_weights:
        if k in model_state:
            p = model_state[k]
            p.data.copy_(base_weights[k].to(device=p.device, dtype=p.dtype))
    torch.cuda.empty_cache()
    base_loss_full = compute_ce_loss(model, tokenizer, all_examples, device)
    print(f"  Base loss (all {n_all} examples): {base_loss_full:.5f}")
    print(f"  Alignment tax: {it_loss_full - base_loss_full:+.5f}")

    # Restore IT weights
    for k in it_weights_cpu:
        if k in model_state:
            p = model_state[k]
            p.data.copy_(it_weights_cpu[k].to(device=p.device, dtype=p.dtype))
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------------------
    # 10-split experiment
    # ---------------------------------------------------------------------------
    split_results = []

    for split_id in range(n_splits):
        print(f"\n{'='*60}")
        print(f"SPLIT {split_id+1}/{n_splits}")
        print(f"{'='*60}")
        t_split = time.time()

        rng = np.random.RandomState(split_id * 31 + 7)
        idx = rng.permutation(n_all)
        half = n_all // 2
        attr_idx = idx[:half]
        eval_idx = idx[half:]

        attr_examples_full = [all_examples[i] for i in attr_idx]
        eval_examples = [all_examples[i] for i in eval_idx]
        # Subsample attribution half for speed (196 components × n_examples forward passes)
        attr_examples = attr_examples_full[:max_attr_ex]
        print(f"  Attribution: {len(attr_examples)}/{len(attr_examples_full)}, Eval: {len(eval_examples)}")

        # Eval-half IT and base losses
        print("  Computing eval-half IT loss...")
        it_loss = compute_ce_loss(model, tokenizer, eval_examples, device)

        for k in base_weights:
            if k in model_state:
                p = model_state[k]
                p.data.copy_(base_weights[k].to(device=p.device, dtype=p.dtype))
        torch.cuda.empty_cache()
        print("  Computing eval-half base loss...")
        base_loss = compute_ce_loss(model, tokenizer, eval_examples, device)

        # Restore IT
        for k in it_weights_cpu:
            if k in model_state:
                p = model_state[k]
                p.data.copy_(it_weights_cpu[k].to(device=p.device, dtype=p.dtype))
        torch.cuda.empty_cache()

        tax = it_loss - base_loss
        print(f"  Eval-half: IT={it_loss:.5f}, Base={base_loss:.5f}, Tax={tax:+.5f}")

        # --- Attribution on attr_half ---
        print(f"  Running attribution on {len(attr_examples)} examples...")
        components, attr_baseline = run_attribution(
            model, tokenizer, attr_examples, base_index, device
        )
        print(f"  Attribution done. Attr-half baseline={attr_baseline:.5f}, "
              f"{len(components)} components scored")

        # Select top-k by attribution
        selected_attr = select_top_k(components, k_count)
        print(f"  Top-{k_pct}% = {len(selected_attr)} components: "
              f"{_count_types(selected_attr)}")

        # SAR with attribution-selected components
        apply_sar_inplace(model_state, base_weights, selected_attr)
        torch.cuda.empty_cache()
        sar_attr_loss = compute_ce_loss(model, tokenizer, eval_examples, device)
        restore_it_inplace(model_state, it_weights_cpu, selected_attr)
        torch.cuda.empty_cache()

        attr_recovery = compute_recovery(it_loss, base_loss, sar_attr_loss)
        print(f"  SAR(attribution) loss={sar_attr_loss:.5f}, recovery={attr_recovery:.1%}")

        # --- Heuristic baseline ---
        selected_heuristic = select_heuristic(components, n_layers, k_count)
        print(f"  Heuristic: {len(selected_heuristic)} components: "
              f"{_count_types(selected_heuristic)}")

        apply_sar_inplace(model_state, base_weights, selected_heuristic)
        torch.cuda.empty_cache()
        sar_heur_loss = compute_ce_loss(model, tokenizer, eval_examples, device)
        restore_it_inplace(model_state, it_weights_cpu, selected_heuristic)
        torch.cuda.empty_cache()

        heur_recovery = compute_recovery(it_loss, base_loss, sar_heur_loss)
        print(f"  SAR(heuristic) loss={sar_heur_loss:.5f}, recovery={heur_recovery:.1%}")

        # --- Random baseline (10 seeds) ---
        random_recoveries = []
        random_losses = []
        print(f"  Running {n_random_seeds} random baselines...")
        for rs in range(n_random_seeds):
            rs_rng = np.random.RandomState(split_id * 1000 + rs + 500)
            selected_rand = select_random(components, k_count, rs_rng)
            apply_sar_inplace(model_state, base_weights, selected_rand)
            torch.cuda.empty_cache()
            rand_loss = compute_ce_loss(model, tokenizer, eval_examples, device)
            restore_it_inplace(model_state, it_weights_cpu, selected_rand)
            torch.cuda.empty_cache()
            rand_rec = compute_recovery(it_loss, base_loss, rand_loss)
            random_losses.append(rand_loss)
            random_recoveries.append(rand_rec)

        rand_mean = float(np.mean(random_recoveries))
        rand_std = float(np.std(random_recoveries))
        print(f"  Random baseline: mean={rand_mean:.1%} ± {rand_std:.1%}  "
              f"(range {min(random_recoveries):.1%}–{max(random_recoveries):.1%})")

        elapsed = time.time() - t_split
        split_result = {
            "split_id": split_id,
            "n_attr": len(attr_examples),
            "n_eval": len(eval_examples),
            "it_loss": float(it_loss),
            "base_loss": float(base_loss),
            "alignment_tax": float(tax),
            "attr_sar": {
                "loss": float(sar_attr_loss),
                "recovery": float(attr_recovery),
                "selected_types": _count_types(selected_attr),
            },
            "heuristic": {
                "loss": float(sar_heur_loss),
                "recovery": float(heur_recovery),
                "selected_types": _count_types(selected_heuristic),
            },
            "random": {
                "losses": [float(x) for x in random_losses],
                "recoveries": [float(x) for x in random_recoveries],
                "mean_recovery": rand_mean,
                "std_recovery": rand_std,
                "min_recovery": float(min(random_recoveries)),
                "max_recovery": float(max(random_recoveries)),
            },
            "elapsed_s": elapsed,
        }
        split_results.append(split_result)
        print(f"  Split {split_id+1} done in {elapsed/60:.1f} min")

        # Intermediate save
        _save_results(split_results, it_loss_full, base_loss_full, k_pct, k_count)

    # ---------------------------------------------------------------------------
    # Aggregate summary
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY ACROSS ALL SPLITS")
    print("=" * 70)

    attr_recoveries = [s["attr_sar"]["recovery"] for s in split_results]
    heur_recoveries = [s["heuristic"]["recovery"] for s in split_results]
    rand_means = [s["random"]["mean_recovery"] for s in split_results]
    rand_stds = [s["random"]["std_recovery"] for s in split_results]

    print(f"  Attribution SAR recovery: {np.mean(attr_recoveries):.1%} ± {np.std(attr_recoveries):.1%}")
    print(f"  Heuristic recovery:       {np.mean(heur_recoveries):.1%} ± {np.std(heur_recoveries):.1%}")
    print(f"  Random recovery (mean):   {np.mean(rand_means):.1%} ± {np.mean(rand_stds):.1%}")
    print(f"  Attribution > Heuristic in {sum(a > h for a,h in zip(attr_recoveries, heur_recoveries))}/{len(split_results)} splits")
    print(f"  Attribution > Random(mean) in {sum(a > r for a,r in zip(attr_recoveries, rand_means))}/{len(split_results)} splits")

    # Paired statistical tests
    if len(split_results) >= 5:
        try:
            from scipy.stats import wilcoxon, ttest_rel
            diff_ah = np.array(attr_recoveries) - np.array(heur_recoveries)
            diff_ar = np.array(attr_recoveries) - np.array(rand_means)
            w_ah, p_ah = wilcoxon(diff_ah)
            t_ar, p_ar = ttest_rel(attr_recoveries, rand_means)
            print(f"  Wilcoxon attr vs heuristic: W={w_ah:.1f}, p={p_ah:.4f}")
            print(f"  Paired t-test attr vs random: t={t_ar:.2f}, p={p_ar:.4f}")
        except Exception as e:
            print(f"  Statistical test failed: {e}")

    total_elapsed = time.time() - t_start
    print(f"\n  Total elapsed: {total_elapsed/60:.1f} min")

    final = _save_results(split_results, it_loss_full, base_loss_full, k_pct, k_count)
    print(f"\nFull results saved to {RESULTS_DIR / 'bfcl_split_validation.json'}")
    return final


def _count_types(components):
    counts = {}
    for c in components:
        t = c["component_type"]
        counts[t] = counts.get(t, 0) + 1
    return counts


def _save_results(split_results, it_loss_full, base_loss_full, k_pct, k_count):
    attr_recoveries = [s["attr_sar"]["recovery"] for s in split_results]
    heur_recoveries = [s["heuristic"]["recovery"] for s in split_results]
    rand_means = [s["random"]["mean_recovery"] for s in split_results]
    rand_stds = [s["random"]["std_recovery"] for s in split_results]

    out = {
        "experiment": "within-BFCL split validation",
        "description": (
            "Attribution on held-out half of BFCL, evaluate SAR on other half. "
            "Compares attribution-selected vs random vs structural heuristic."
        ),
        "model": "qwen2.5-7b",
        "base_model": BASE_MODEL,
        "it_model": IT_MODEL,
        "k_pct": k_pct,
        "k_count": k_count,
        "n_splits": len(split_results),
        "full_data": {
            "it_loss": float(it_loss_full),
            "base_loss": float(base_loss_full),
            "alignment_tax": float(it_loss_full - base_loss_full),
        },
        "aggregate": {
            "attr_sar_mean_recovery": float(np.mean(attr_recoveries)),
            "attr_sar_std_recovery": float(np.std(attr_recoveries)),
            "heuristic_mean_recovery": float(np.mean(heur_recoveries)),
            "heuristic_std_recovery": float(np.std(heur_recoveries)),
            "random_mean_recovery": float(np.mean(rand_means)),
            "random_pooled_std": float(np.mean(rand_stds)),
            "attr_beats_heuristic_n_splits": int(
                sum(a > h for a, h in zip(attr_recoveries, heur_recoveries))
            ),
            "attr_beats_random_n_splits": int(
                sum(a > r for a, r in zip(attr_recoveries, rand_means))
            ),
        },
        "splits": split_results,
    }

    # Add paired tests if enough splits
    if len(split_results) >= 5:
        try:
            from scipy.stats import wilcoxon, ttest_rel
            diff_ah = np.array(attr_recoveries) - np.array(heur_recoveries)
            diff_ar = np.array(attr_recoveries) - np.array(rand_means)
            w_ah, p_ah = wilcoxon(diff_ah)
            t_ar, p_ar = ttest_rel(attr_recoveries, rand_means)
            out["statistical_tests"] = {
                "wilcoxon_attr_vs_heuristic": {"W": float(w_ah), "p": float(p_ah)},
                "paired_ttest_attr_vs_random": {"t": float(t_ar), "p": float(p_ar)},
            }
        except Exception:
            pass

    out_path = RESULTS_DIR / "bfcl_split_validation.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    main()
