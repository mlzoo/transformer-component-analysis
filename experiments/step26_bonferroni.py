"""
Step 26: Compute Bonferroni-corrected p-values for V/O vs Q/K attribution.

Addresses reviewer concern about multiple comparisons across 4 models.
No GPU needed — reads existing attribution data.
"""
import json
import numpy as np
from scipy import stats
from pathlib import Path

RESULTS_DIR = Path("./results")

def load_vo_qk_scores(filepath):
    """Load attribution scores and split into V/O and Q/K groups."""
    with open(filepath) as f:
        data = json.load(f)

    # Handle two possible data formats
    if "attribution" in data and "components" in data["attribution"]:
        components = data["attribution"]["components"]
    elif "components" in data:
        components = data["components"]
    else:
        raise ValueError(f"Unknown format in {filepath}")

    vo_scores = []
    qk_scores = []
    for comp in components:
        ctype = comp.get("type", comp.get("component_type", ""))
        score = comp.get("mean_score", comp.get("score", 0))
        # Handle both naming conventions: W_V/W_O or v_proj/o_proj
        if ctype in ("W_V", "W_O", "v_proj", "o_proj"):
            vo_scores.append(score)
        elif ctype in ("W_Q", "W_K", "q_proj", "k_proj"):
            qk_scores.append(score)

    return np.array(vo_scores), np.array(qk_scores)


def main():
    files = {
        "Qwen2.5-7B": RESULTS_DIR / "step8_attribution_200_qwen2.5-7b.json",
        "Llama-3.1-8B": RESULTS_DIR / "step8_attribution_200_llama-3.1-8b.json",
        "Mistral-7B": RESULTS_DIR / "step8_attribution_200_mistral-7b.json",
        "Qwen2.5-14B": RESULTS_DIR / "step12_scaling_14b_v2.json",
    }

    n_comparisons = len(files)  # 4
    print(f"Bonferroni correction for {n_comparisons} comparisons")
    print(f"Significance threshold: α = 0.05 / {n_comparisons} = {0.05/n_comparisons:.4f}\n")

    results = {}
    for model, fpath in files.items():
        if not fpath.exists():
            print(f"  {model}: FILE NOT FOUND ({fpath})")
            continue

        vo, qk = load_vo_qk_scores(fpath)

        if len(vo) == 0 or len(qk) == 0:
            print(f"  {model}: ERROR — empty V/O ({len(vo)}) or Q/K ({len(qk)}) arrays")
            continue

        # One-sided Mann-Whitney U test (H1: V/O > Q/K)
        # scipy's mannwhitneyu with alternative='greater' gives one-sided p
        U, p_two_sided = stats.mannwhitneyu(vo, qk, alternative='two-sided')
        U_one, p_one_sided = stats.mannwhitneyu(vo, qk, alternative='greater')

        # Rank-biserial correlation (positive = V/O > Q/K)
        n1, n2 = len(vo), len(qk)
        r = 2 * U_one / (n1 * n2) - 1

        # Bonferroni-corrected p-values
        p_bonf_one = min(p_one_sided * n_comparisons, 1.0)
        p_bonf_two = min(p_two_sided * n_comparisons, 1.0)

        sig_bonf = "YES" if p_bonf_one < 0.05 else "NO"

        print(f"{model}:")
        print(f"  n_VO={n1}, n_QK={n2}")
        print(f"  V/O mean={vo.mean():.6f}, Q/K mean={qk.mean():.6f}")
        print(f"  U={U_one:.1f}")
        print(f"  p (one-sided):  {p_one_sided:.2e}")
        print(f"  p (two-sided):  {p_two_sided:.2e}")
        print(f"  p_bonf (one):   {p_bonf_one:.2e}  → significant: {sig_bonf}")
        print(f"  rank-biserial r: {r:.3f}")
        print()

        results[model] = {
            "n_vo": n1, "n_qk": n2,
            "U": float(U_one),
            "p_one_sided": float(p_one_sided),
            "p_two_sided": float(p_two_sided),
            "p_bonferroni_one": float(p_bonf_one),
            "p_bonferroni_two": float(p_bonf_two),
            "rank_biserial_r": float(r),
            "significant_bonferroni": sig_bonf == "YES",
            "vo_mean": float(vo.mean()),
            "qk_mean": float(qk.mean()),
        }

    out = RESULTS_DIR / "step26_bonferroni.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
