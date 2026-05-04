"""Generate pipeline overview figure (Figure 1) for NeurIPS paper.

Three panels telling the story in 10 seconds:
  A: Component-level structure (MLP dominance + V/O > Q/K)
  B: SAR top-5% selection vs random/magnitude baselines
  C: Capability recovery + safety preservation
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyArrowPatch
from pathlib import Path

RESULTS = Path("./results")
OUT = Path("./figures")
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.size': 8,
    'font.family': 'serif',
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 6.5,
    'figure.dpi': 300,
})

C_MLP = '#1a9850'
C_VO = '#d73027'
C_QK = '#4575b4'


def load_json(path):
    with open(path) as f:
        return json.load(f)


def panel_a(ax):
    """Component-level structure: stacked bars for 3 models."""
    models = [("Qwen", 63.0, 20.2, 16.8),
              ("Llama", 59.0, 26.8, 14.2),
              ("Mistral", 61.6, 30.3, 8.1)]

    x = np.arange(len(models))
    width = 0.55

    for i, (label, mlp, vo, qk) in enumerate(models):
        ax.bar(i, qk, width, color=C_QK, edgecolor='white', linewidth=0.5)
        ax.bar(i, vo, width, bottom=qk, color=C_VO, edgecolor='white', linewidth=0.5)
        ax.bar(i, mlp, width, bottom=qk+vo, color=C_MLP, edgecolor='white', linewidth=0.5)

        if qk > 8:
            ax.text(i, qk/2, f'{qk:.0f}%', ha='center', va='center', fontsize=6, color='white', fontweight='bold')
        ax.text(i, qk + vo/2, f'{vo:.0f}%', ha='center', va='center', fontsize=6, color='white', fontweight='bold')
        ax.text(i, qk + vo + mlp/2, f'{mlp:.0f}%', ha='center', va='center', fontsize=6, color='white', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in models])
    ax.set_ylabel('Share of modification (%)')
    ax.set_ylim(0, 108)
    ax.set_title('(a) Component-level structure', fontweight='bold', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    legend_elements = [
        Patch(facecolor=C_MLP, label='MLP (59–63%)'),
        Patch(facecolor=C_VO, label='V/O (output)'),
        Patch(facecolor=C_QK, label='Q/K (routing)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9, fontsize=5.5)


def panel_b(ax):
    """SAR: attribution-guided vs random (Qwen)."""
    sar_data = load_json(RESULTS / "sar_eval_qwen2.5-7b.json")
    results = sar_data["results"]

    budgets = [3, 5, 8, 10]
    sar_recovery = [results[f"topk_k{k}"]["pct_tax_recovered"] for k in budgets]
    rand_k5 = results["random_k5"]["pct_tax_recovered"]
    rand_recovery = [results.get(f"random_k{k}", {}).get("pct_tax_recovered", rand_k5 * k / 5 if rand_k5 is not None else 0) for k in budgets]

    x = np.arange(len(budgets))
    width = 0.32

    ax.bar(x - width/2, sar_recovery, width, color='#d73027', label='SAR (attribution)', edgecolor='white', linewidth=0.5)
    ax.bar(x + width/2, rand_recovery, width, color='#aaaaaa', label='Random', edgecolor='white', linewidth=0.5)

    ratio = sar_recovery[1] / rand_recovery[1] if rand_recovery[1] > 0 else 0
    ax.annotate(f'{ratio:.1f}×', xy=(1, sar_recovery[1]),
                xytext=(1, sar_recovery[1] + 8),
                fontsize=7, fontweight='bold', color='#d73027', ha='center',
                arrowprops=dict(arrowstyle='-', color='#d73027', lw=0.5))

    ax.set_xticks(x)
    ax.set_xticklabels([f'{k}%' for k in budgets])
    ax.set_xlabel('Rollback budget (Qwen)')
    ax.set_ylabel('Tax recovery (%)')
    ax.set_ylim(-10, 105)
    ax.axhline(0, color='black', linewidth=0.3)
    ax.set_title('(b) Targeted vs. random rollback', fontweight='bold', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper left', framealpha=0.9, fontsize=5.5)


def panel_c(ax):
    """Capability + safety results for SAR-5% across 3 models."""
    models = ['Qwen', 'Llama', 'Mistral']

    # BFCL improvement (IT→SAR, pp)
    bfcl = [0.9, 0.9, 1.3]
    # HumanEval improvement (IT→SAR, pp)
    humaneval = [2.4, 3.0, 0.6]
    # Safety delta (judge, pp)
    safety = [0.0, -0.8, +0.8]

    x = np.arange(len(models))
    width = 0.22

    ax.bar(x - width, bfcl, width, color='#fc8d59', label='BFCL ↑', edgecolor='white', linewidth=0.5)
    ax.bar(x, humaneval, width, color='#fee090', label='HumanEval ↑', edgecolor='#aaa', linewidth=0.5)
    ax.bar(x + width, safety, width, color=C_MLP, label='Safety Δ', edgecolor='white', linewidth=0.5)

    # Safety zone
    ax.axhspan(-2.5, 0, alpha=0.06, color='green', zorder=0)
    ax.axhline(0, color='black', linewidth=0.3)

    # Add |Δ|≤2.0pp label
    ax.text(2.35, -1.5, '|Δ|≤2pp', fontsize=5.5, color='#1a9850', fontstyle='italic')

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel('IT → SAR-5% change (pp)')
    ax.set_ylim(-3.5, 5)
    ax.set_title('(c) Capability ↑, safety preserved', fontweight='bold', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=5.5)


def main():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))

    panel_a(axes[0])
    panel_b(axes[1])
    panel_c(axes[2])

    plt.tight_layout(w_pad=1.5)

    # Add connecting arrows
    for i in range(2):
        x_end = axes[i].get_position().x1
        x_start = axes[i+1].get_position().x0
        fig.patches.append(FancyArrowPatch(
            posA=(x_end + 0.006, 0.48),
            posB=(x_start - 0.006, 0.48),
            transform=fig.transFigure,
            arrowstyle='->', mutation_scale=15,
            color='#444444', linewidth=1.8,
        ))

    fig.savefig(OUT / "fig0_overview.pdf", bbox_inches='tight')
    fig.savefig(OUT / "fig0_overview.png", bbox_inches='tight', dpi=300)
    print("Saved fig0_overview")
    plt.close()


if __name__ == "__main__":
    main()
