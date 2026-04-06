"""
Step 35: Bootstrap confidence intervals for SAR and attribution statistics.

Computes bootstrap CIs for:
  1. V/O vs Q/K attribution ratio (resampling components within each group)
  2. SAR top-k% component selection stability (how often each component appears)
  3. Output-pathway share of total alignment change

Uses pre-computed attribution data from attribution.
CPU-only computation.

Usage: python sar_bootstrap_ci.py [model_name]
"""

import sys, json
import numpy as np
from pathlib import Path
from collections import Counter

RESULTS_DIR = Path("./results")
N_BOOTSTRAP = 10000
SEED = 42

MODELS = ["qwen2.5-7b", "llama-3.1-8b", "mistral-7b", "yi-1.5-9b"]


def load_attribution(model_name):
    path = RESULTS_DIR / f"attribution_{model_name}.json"
    if not path.exists():
        # Fallback for Yi and other models with different file patterns
        path = RESULTS_DIR / f"pipeline_{model_name.replace('-', '_').replace('.', '_')}_full_pipeline.json"
    if not path.exists():
        path = RESULTS_DIR / f"attribution_yi.json"  # Yi specific
    with open(path) as f:
        data = json.load(f)
    return data


def bootstrap_ci(values, n_boot=N_BOOTSTRAP, ci=95, stat_fn=np.mean, seed=SEED):
    """Compute bootstrap confidence interval for a statistic."""
    rng = np.random.RandomState(seed)
    values = np.array(values)
    boot_stats = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boot_stats.append(stat_fn(sample))
    boot_stats = np.array(boot_stats)
    alpha = (100 - ci) / 2
    lo = np.percentile(boot_stats, alpha)
    hi = np.percentile(boot_stats, 100 - alpha)
    return float(np.mean(boot_stats)), float(lo), float(hi)


def run_model(model_name):
    print(f"\n{'='*60}")
    print(f"Bootstrap CI analysis: {model_name}")
    print(f"{'='*60}")

    data = load_attribution(model_name)
    components = data["components"]
    baseline_loss = data["baseline_loss"]

    # Extract harm scores by type
    vo_scores = []
    qk_scores = []
    all_scores = []
    type_scores = {}

    for c in components:
        ct = c["component_type"]
        hs = abs(c["harm_score"])
        all_scores.append(hs)
        if ct not in type_scores:
            type_scores[ct] = []
        type_scores[ct].append(hs)

        if ct in ("W_V", "W_O"):
            vo_scores.append(hs)
        elif ct in ("W_Q", "W_K"):
            qk_scores.append(hs)

    vo_scores = np.array(vo_scores)
    qk_scores = np.array(qk_scores)

    # 1. Bootstrap CI for V/O vs Q/K ratio
    print(f"\n1. V/O vs Q/K attribution ratio")
    print(f"   V/O scores: n={len(vo_scores)}, mean={np.mean(vo_scores):.6f}")
    print(f"   Q/K scores: n={len(qk_scores)}, mean={np.mean(qk_scores):.6f}")
    observed_ratio = np.mean(vo_scores) / np.mean(qk_scores) if np.mean(qk_scores) > 0 else float('inf')
    print(f"   Observed ratio: {observed_ratio:.3f}x")

    rng = np.random.RandomState(SEED)
    boot_ratios = []
    for _ in range(N_BOOTSTRAP):
        vo_boot = rng.choice(vo_scores, size=len(vo_scores), replace=True)
        qk_boot = rng.choice(qk_scores, size=len(qk_scores), replace=True)
        qk_mean = np.mean(qk_boot)
        if qk_mean > 0:
            boot_ratios.append(np.mean(vo_boot) / qk_mean)
    boot_ratios = np.array(boot_ratios)
    ratio_lo = np.percentile(boot_ratios, 2.5)
    ratio_hi = np.percentile(boot_ratios, 97.5)
    print(f"   95% CI: [{ratio_lo:.3f}, {ratio_hi:.3f}]")

    # 2. Bootstrap CI for output-pathway share
    print(f"\n2. Output-pathway share of total change")
    output_types = {"W_V", "W_O", "W_down"}

    # Per-component scores for resampling
    comp_data = [(abs(c["harm_score"]), c["component_type"]) for c in components]
    observed_output_share = sum(s for s, t in comp_data if t in output_types) / sum(s for s, _ in comp_data) * 100

    print(f"   Observed output-pathway share: {observed_output_share:.1f}%")

    rng = np.random.RandomState(SEED)
    boot_shares = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.choice(len(comp_data), size=len(comp_data), replace=True)
        sample = [comp_data[i] for i in idx]
        total = sum(s for s, _ in sample)
        if total > 0:
            output_sum = sum(s for s, t in sample if t in output_types)
            boot_shares.append(output_sum / total * 100)
    boot_shares = np.array(boot_shares)
    share_lo = np.percentile(boot_shares, 2.5)
    share_hi = np.percentile(boot_shares, 97.5)
    print(f"   95% CI: [{share_lo:.1f}%, {share_hi:.1f}%]")

    # 3. SAR top-k selection stability
    print(f"\n3. SAR top-5% selection stability")
    n_components = len(components)
    k = max(1, int(n_components * 0.05))
    print(f"   Total components: {n_components}, top-k: {k}")

    # Sort by harm score (positive = harmful)
    sorted_comps = sorted(components, key=lambda c: c["harm_score"], reverse=True)
    original_topk = set()
    for c in sorted_comps[:k]:
        original_topk.add(f"{c['component_type']}_L{c['layer']}")

    # For selection stability, resample components and check how often each appears in top-k
    # We resample the harm scores and re-rank
    harm_scores = np.array([c["harm_score"] for c in components])
    comp_labels = [f"{c['component_type']}_L{c['layer']}" for c in components]

    rng = np.random.RandomState(SEED)
    selection_counts = Counter()
    n_stable_boot = N_BOOTSTRAP

    for _ in range(n_stable_boot):
        # Add noise proportional to score std to simulate sampling variance
        noise = rng.normal(0, np.std(harm_scores) * 0.1, size=len(harm_scores))
        noisy_scores = harm_scores + noise
        top_idx = np.argsort(-noisy_scores)[:k]
        for idx in top_idx:
            selection_counts[comp_labels[idx]] += 1

    # How often does the original top-k set appear?
    stability_scores = []
    for label in original_topk:
        freq = selection_counts.get(label, 0) / n_stable_boot * 100
        stability_scores.append(freq)

    mean_stability = np.mean(stability_scores)
    min_stability = np.min(stability_scores)

    print(f"   Original top-{k} components:")
    for label in sorted(original_topk):
        freq = selection_counts.get(label, 0) / n_stable_boot * 100
        print(f"     {label}: {freq:.1f}% selection frequency")
    print(f"   Mean stability: {mean_stability:.1f}%")
    print(f"   Min stability: {min_stability:.1f}%")

    # Type composition of top-k
    topk_types = Counter()
    for c in sorted_comps[:k]:
        topk_types[c["component_type"]] += 1
    output_pathway_count = sum(topk_types.get(t, 0) for t in ["W_V", "W_O", "W_down"])
    output_pathway_pct = output_pathway_count / k * 100

    print(f"\n   Top-{k} type composition:")
    for t in ["W_down", "W_up", "W_gate", "W_O", "W_V", "W_Q", "W_K"]:
        if t in topk_types:
            print(f"     {t}: {topk_types[t]} ({topk_types[t]/k*100:.0f}%)")
    print(f"   Output-pathway in top-k: {output_pathway_count}/{k} ({output_pathway_pct:.0f}%)")

    # 4. Bootstrap CI for per-type mean harm scores
    print(f"\n4. Per-type mean |harm score| with 95% CI")
    type_cis = {}
    for ct in ["W_Q", "W_K", "W_V", "W_O", "W_gate", "W_up", "W_down"]:
        scores = type_scores.get(ct, [])
        if not scores:
            continue
        mean_val, lo, hi = bootstrap_ci(scores)
        type_cis[ct] = {"mean": mean_val, "ci_lo": lo, "ci_hi": hi, "n": len(scores)}
        print(f"   {ct:8s}: {mean_val:.6f} [{lo:.6f}, {hi:.6f}] (n={len(scores)})")

    results = {
        "analysis": "Bootstrap confidence intervals for SAR and attribution",
        "model": model_name,
        "n_bootstrap": N_BOOTSTRAP,
        "n_components": n_components,
        "baseline_loss": baseline_loss,
        "vo_qk_ratio": {
            "observed": observed_ratio,
            "ci_lo": float(ratio_lo),
            "ci_hi": float(ratio_hi),
            "n_vo": len(vo_scores),
            "n_qk": len(qk_scores),
        },
        "output_pathway_share": {
            "observed": observed_output_share,
            "ci_lo": float(share_lo),
            "ci_hi": float(share_hi),
        },
        "sar_selection_stability": {
            "k": k,
            "mean_stability_pct": mean_stability,
            "min_stability_pct": min_stability,
            "output_pathway_in_topk": output_pathway_count,
            "output_pathway_pct": output_pathway_pct,
            "topk_type_composition": dict(topk_types),
            "per_component_frequency": {
                label: selection_counts.get(label, 0) / n_stable_boot * 100
                for label in sorted(original_topk)
            },
        },
        "per_type_ci": type_cis,
    }

    out_path = RESULTS_DIR / f"sar_bootstrap_ci_{model_name}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")
    return results


if __name__ == "__main__":
    if len(sys.argv) > 1:
        models = [sys.argv[1]]
    else:
        models = MODELS

    all_results = {}
    for m in models:
        all_results[m] = run_model(m)

    # Cross-model summary
    if len(models) > 1:
        print(f"\n{'='*60}")
        print(f"CROSS-MODEL SUMMARY")
        print(f"{'='*60}")
        for m in models:
            r = all_results[m]
            vr = r["vo_qk_ratio"]
            ops = r["output_pathway_share"]
            ss = r["sar_selection_stability"]
            print(f"\n  {m}:")
            print(f"    V/O:Q/K ratio: {vr['observed']:.2f}x [{vr['ci_lo']:.2f}, {vr['ci_hi']:.2f}]")
            print(f"    Output share:  {ops['observed']:.1f}% [{ops['ci_lo']:.1f}, {ops['ci_hi']:.1f}]")
            print(f"    SAR stability: {ss['mean_stability_pct']:.1f}% (output in top-k: {ss['output_pathway_pct']:.0f}%)")
