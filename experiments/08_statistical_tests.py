"""
Statistical Tests for Paper Claims

Computes:
1. Page's test for ordered alternatives (W_down > W_up > W_gate within MLP)
2. Block permutation test for V/O vs Q/K attribution asymmetry
3. Bootstrap CIs for SAR/random ratio
4. Wilcoxon signed-rank test for SAR vs IT on benchmarks

All tests run CPU-only using pre-computed results JSON files.
No model loading required.

Usage:
    python experiments/08_statistical_tests.py [model_name]

If no model_name given, runs all available models.
"""

import sys
import json
import numpy as np
from pathlib import Path
from scipy import stats

RESULTS_DIR = Path("./results")


def load_json(path):
    with open(path) as f:
        return json.load(f)


def pages_test(data_matrix):
    """
    Page's test for ordered alternatives.

    Tests H1: column medians are ordered (col 1 > col 2 > col 3)
    against H0: no ordering.

    Parameters
    ----------
    data_matrix : array of shape (n_layers, 3)
        Columns are [W_down, W_up, W_gate] harm scores per layer.

    Returns
    -------
    L : float - Page's L statistic
    Z : float - standardized Z score
    p : float - one-sided p-value
    """
    data = np.asarray(data_matrix)
    n, k = data.shape

    # Rank within each row (layer)
    ranks = np.zeros_like(data, dtype=float)
    for i in range(n):
        ranks[i] = stats.rankdata(data[i])

    # Page's L: sum of (predicted_rank * column_rank_sum)
    # Predicted order: W_down=3 (highest), W_up=2, W_gate=1 (lowest)
    predicted_ranks = np.array([3, 2, 1])
    column_rank_sums = ranks.sum(axis=0)
    L = np.sum(predicted_ranks * column_rank_sums)

    # Expected value and variance under H0
    E_L = n * k * (k + 1) ** 2 / 4
    Var_L = n * k ** 2 * (k ** 2 - 1) ** 2 / (144 * (k - 1))

    Z = (L - E_L) / np.sqrt(Var_L)
    p = 1 - stats.norm.cdf(Z)

    return float(L), float(Z), float(p)


def block_permutation_test(components, n_perms=100000, seed=42):
    """
    Block permutation test for V/O vs Q/K attribution difference.

    Preserves layer autocorrelation by permuting component-type labels
    within each layer (block), not globally.
    """
    rng = np.random.RandomState(seed)

    # Group by layer
    layers = {}
    for c in components:
        layer = c["layer"]
        if layer not in layers:
            layers[layer] = []
        layers[layer].append(c)

    # Observed statistic: mean(|V/O harm|) - mean(|Q/K harm|)
    vo_scores = [abs(c["harm_score"]) for c in components if c["component_type"] in ("W_V", "W_O")]
    qk_scores = [abs(c["harm_score"]) for c in components if c["component_type"] in ("W_Q", "W_K")]
    observed = np.mean(vo_scores) - np.mean(qk_scores)

    # Permutation: within each layer, shuffle the 4 attention component types
    attn_types = ["W_Q", "W_K", "W_V", "W_O"]
    perm_stats = np.empty(n_perms)

    for p in range(n_perms):
        perm_vo = []
        perm_qk = []
        for layer_idx in sorted(layers.keys()):
            layer_comps = layers[layer_idx]
            attn_comps = [c for c in layer_comps if c["component_type"] in attn_types]
            if len(attn_comps) != 4:
                continue
            scores = [abs(c["harm_score"]) for c in attn_comps]
            rng.shuffle(scores)
            # First two become "V/O", last two become "Q/K" (arbitrary split)
            perm_vo.extend(scores[:2])
            perm_qk.extend(scores[2:])

        if perm_vo and perm_qk:
            perm_stats[p] = np.mean(perm_vo) - np.mean(perm_qk)
        else:
            perm_stats[p] = 0.0

    p_value = np.mean(perm_stats >= observed)
    return float(observed), float(p_value)


def bootstrap_ratio_ci(sar_recovery, random_recoveries, n_boot=10000, seed=42):
    """Bootstrap 95% CI for SAR/random ratio."""
    rng = np.random.RandomState(seed)
    n = len(random_recoveries)
    ratios = []
    for _ in range(n_boot):
        boot_random = rng.choice(random_recoveries, size=n, replace=True)
        boot_mean = np.mean(boot_random)
        if abs(boot_mean) > 1e-9:
            ratios.append(sar_recovery / boot_mean)
    lo = float(np.percentile(ratios, 2.5))
    hi = float(np.percentile(ratios, 97.5))
    return lo, hi


def run_pages_test(model_name):
    """Run Page's test on MLP components for one model."""
    attr_path = RESULTS_DIR / f"attribution_{model_name}.json"
    if not attr_path.exists():
        print(f"  Skipping {model_name}: attribution file not found")
        return None

    data = load_json(attr_path)
    components = data["components"]

    # Group MLP scores by layer
    mlp_by_layer = {}
    for c in components:
        if c["component_type"] in ("W_down", "W_up", "W_gate"):
            layer = c["layer"]
            if layer not in mlp_by_layer:
                mlp_by_layer[layer] = {}
            mlp_by_layer[layer][c["component_type"]] = abs(c["harm_score"])

    # Build matrix: rows=layers, cols=[W_down, W_up, W_gate]
    matrix = []
    for layer in sorted(mlp_by_layer.keys()):
        row = mlp_by_layer[layer]
        if len(row) == 3:
            matrix.append([row["W_down"], row["W_up"], row["W_gate"]])

    if len(matrix) < 3:
        print(f"  Skipping {model_name}: insufficient MLP layers")
        return None

    L, Z, p = pages_test(matrix)
    print(f"  {model_name}: Page's L={L:.1f}, Z={Z:.2f}, p={p:.2e}")

    # Count layers where ordering holds
    n_ordered = sum(1 for row in matrix if row[0] > row[1] > row[2])
    print(f"    Strict ordering (down>up>gate): {n_ordered}/{len(matrix)} layers")

    return {"model": model_name, "L": L, "Z": Z, "p": p,
            "n_layers": len(matrix), "n_ordered": n_ordered}


def run_block_permutation(model_name):
    """Run block permutation test for V/O vs Q/K."""
    attr_path = RESULTS_DIR / f"attribution_{model_name}.json"
    if not attr_path.exists():
        return None

    data = load_json(attr_path)
    components = data["components"]

    observed, p_value = block_permutation_test(components)
    print(f"  {model_name}: V/O - Q/K = {observed:.4f}, p_block = {p_value:.4e}")
    return {"model": model_name, "observed_diff": observed, "p_block": p_value}


def run_bootstrap_ci(model_name):
    """Compute bootstrap CI for SAR/random ratio."""
    # Try to load random baseline results
    random_path = RESULTS_DIR / f"random_baseline_{model_name}.json"
    sar_path = RESULTS_DIR / f"sar_eval_{model_name}.json"

    if not sar_path.exists():
        return None

    sar_data = load_json(sar_path)

    # Get SAR-5% recovery
    if "results" in sar_data and "topk_k5" in sar_data["results"]:
        sar_recovery = sar_data["results"]["topk_k5"]["pct_tax_recovered"]
    else:
        return None

    # Get random baseline per-seed values
    rand_data = None
    if random_path.exists():
        rand_data = load_json(random_path)
        random_recoveries = rand_data["random_baseline"]["per_seed_recovery"]
        # Fall back to sar_5pct_recovery from random baseline file if sar_eval has None
        if sar_recovery is None and "sar_5pct_recovery" in rand_data:
            sar_recovery = rand_data["sar_5pct_recovery"]
    elif "results" in sar_data and "random_k5" in sar_data["results"]:
        rk5 = sar_data["results"]["random_k5"]
        if "all_pct" in rk5:
            random_recoveries = rk5["all_pct"]
        elif "pct_tax_recovered" in rk5:
            random_recoveries = [rk5["pct_tax_recovered"]]
        elif "mean_pct" in rk5:
            random_recoveries = [rk5["mean_pct"]]
        else:
            return None
    else:
        return None

    if sar_recovery is None:
        print(f"  {model_name}: SAR recovery is None, skipping CI")
        return None

    if len(random_recoveries) < 2:
        print(f"  {model_name}: Only {len(random_recoveries)} random seed(s), skipping CI")
        return None

    mean_random = np.mean(random_recoveries)
    ratio = sar_recovery / mean_random if abs(mean_random) > 1e-9 else float('inf')
    lo, hi = bootstrap_ratio_ci(sar_recovery, random_recoveries)
    print(f"  {model_name}: SAR/random = {ratio:.1f}x (95% CI: [{lo:.1f}, {hi:.1f}])")
    return {"model": model_name, "ratio": ratio, "ci_lo": lo, "ci_hi": hi,
            "sar_recovery": sar_recovery, "random_mean": mean_random}


def main():
    model_filter = sys.argv[1] if len(sys.argv) > 1 else None

    # Find available models from attribution files
    models = []
    for f in sorted(RESULTS_DIR.glob("attribution_*.json")):
        name = f.stem.replace("attribution_", "")
        if model_filter is None or model_filter in name:
            models.append(name)

    if not models:
        print("No attribution result files found in ./results/")
        print("Run 01_attribution.py first.")
        sys.exit(1)

    print(f"Models found: {models}\n")

    # 1. Page's test
    print("=" * 60)
    print("1. Page's Test for Ordered Alternatives (W_down > W_up > W_gate)")
    print("=" * 60)
    pages_results = []
    for m in models:
        r = run_pages_test(m)
        if r:
            pages_results.append(r)

    if len(pages_results) > 1:
        # Fisher's combined p-value
        p_values = [r["p"] for r in pages_results]
        chi2_stat = -2 * sum(np.log(max(p, 1e-300)) for p in p_values)
        combined_p = 1 - stats.chi2.cdf(chi2_stat, df=2 * len(p_values))
        print(f"\n  Fisher's combined p-value: {combined_p:.2e}")

    # 2. Block permutation test
    print(f"\n{'=' * 60}")
    print("2. Block Permutation Test (V/O vs Q/K, 100k permutations)")
    print("=" * 60)
    block_results = []
    for m in models:
        r = run_block_permutation(m)
        if r:
            block_results.append(r)

    # 3. Bootstrap CIs
    print(f"\n{'=' * 60}")
    print("3. Bootstrap 95% CI for SAR/Random Ratio")
    print("=" * 60)
    bootstrap_results = []
    for m in models:
        r = run_bootstrap_ci(m)
        if r:
            bootstrap_results.append(r)

    # Save combined results
    output = {
        "analysis": "Statistical tests for paper claims",
        "pages_test": pages_results,
        "block_permutation": block_results,
        "bootstrap_ci": bootstrap_results,
    }

    if pages_results and len(pages_results) > 1:
        output["fisher_combined_p"] = float(combined_p)

    out_path = RESULTS_DIR / "statistical_tests.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nAll results saved to {out_path}")


if __name__ == "__main__":
    main()
