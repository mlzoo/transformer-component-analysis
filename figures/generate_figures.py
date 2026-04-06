"""Generate figures for the paper."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

RESULTS = Path("./results")
OUT = Path("./figures")
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.size': 9,
    'font.family': 'serif',
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
})

MODELS = {
    "qwen2.5-7b": {"label": "Qwen2.5-7B", "layers": 28},
    "llama-3.1-8b": {"label": "Llama-3.1-8B", "layers": 32},
    "mistral-7b": {"label": "Mistral-7B", "layers": 32},
}

COMP_TYPES = ["W_Q", "W_K", "W_V", "W_O", "W_gate", "W_up", "W_down"]
COMP_LABELS = [r"$W_Q$", r"$W_K$", r"$W_V$", r"$W_O$", r"$W_{gate}$", r"$W_{up}$", r"$W_{down}$"]

# Colors: Q/K in blue, V/O in red/orange, MLP in green
COMP_COLORS = {
    "W_Q": "#4575b4", "W_K": "#74add1",        # blue (routing)
    "W_V": "#d73027", "W_O": "#f46d43",         # red (output attn)
    "W_gate": "#66c2a5", "W_up": "#3288bd", "W_down": "#1a9850",  # green (MLP)
}
GROUP_COLORS = {"Q/K\n(routing)": "#4575b4", "V/O\n(output)": "#d73027", "MLP": "#1a9850"}


def load_attribution(model_key):
    path = RESULTS / f"attribution_{model_key}.json"
    with open(path) as f:
        return json.load(f)


def load_gradient(model_key):
    path = RESULTS / f"gradient_analysis_{model_key}.json"
    with open(path) as f:
        return json.load(f)


# ========== FIGURE 1: Attribution Heatmap ==========
def fig1_attribution_heatmap():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.8), gridspec_kw={'width_ratios': [28, 32, 32]})

    # Compute global data range across all models for truly unified scale
    all_scores = []
    for mk in MODELS:
        d = load_attribution(mk)
        all_scores.extend(c["harm_score"] for c in d["components"])
    vmin = min(all_scores) * 1.05
    vmax = max(all_scores) * 1.05

    for idx, (model_key, info) in enumerate(MODELS.items()):
        ax = axes[idx]
        data = load_attribution(model_key)
        n_layers = info["layers"]

        # Build matrix: rows=component types, cols=layers
        mat = np.zeros((7, n_layers))
        for comp in data["components"]:
            ct = comp["component_type"]
            layer = comp["layer"]
            row = COMP_TYPES.index(ct)
            mat[row, layer] = comp["harm_score"]

        # Diverging colormap
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        im = ax.imshow(mat, aspect='auto', cmap='RdBu_r', norm=norm, interpolation='nearest')

        ax.set_title(info["label"], fontweight='bold')
        ax.set_xlabel("Layer")
        if idx == 0:
            ax.set_yticks(range(7))
            ax.set_yticklabels(COMP_LABELS)
        else:
            ax.set_yticks([])

        # Mark mid-1/3 region
        mid_start = n_layers // 3
        mid_end = 2 * n_layers // 3 - 1
        ax.axvline(mid_start - 0.5, color='black', linewidth=0.5, linestyle='--', alpha=0.5)
        ax.axvline(mid_end + 0.5, color='black', linewidth=0.5, linestyle='--', alpha=0.5)

    # Colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label("Harm score", fontsize=8)

    plt.tight_layout()
    fig.savefig(OUT / "fig1_attribution_heatmap.pdf", bbox_inches='tight')
    fig.savefig(OUT / "fig1_attribution_heatmap.png", bbox_inches='tight', dpi=300)
    print("Saved fig1_attribution_heatmap")
    plt.close()


# ========== FIGURE 2: Gradient Norm Comparison ==========
def fig2_gradient_comparison():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), sharey=True)

    for idx, (model_key, info) in enumerate(MODELS.items()):
        ax = axes[idx]
        data = load_gradient(model_key)
        layer_means = data["type_layer_means"]
        n_layers = len(layer_means["W_Q"])

        # Compute V/O mean and Q/K mean per layer
        vo_per_layer = [(np.array(layer_means["W_V"][i]) + np.array(layer_means["W_O"][i])) / 2
                        for i in range(n_layers)]
        qk_per_layer = [(np.array(layer_means["W_Q"][i]) + np.array(layer_means["W_K"][i])) / 2
                        for i in range(n_layers)]
        mlp_per_layer = [(np.array(layer_means["W_gate"][i]) + np.array(layer_means["W_up"][i]) +
                          np.array(layer_means["W_down"][i])) / 3 for i in range(n_layers)]

        layers = np.arange(n_layers)
        ax.fill_between(layers, vo_per_layer, alpha=0.3, color='#d73027')
        ax.fill_between(layers, qk_per_layer, alpha=0.3, color='#4575b4')
        ax.plot(layers, vo_per_layer, color='#d73027', linewidth=1.5, label='V/O (output)')
        ax.plot(layers, qk_per_layer, color='#4575b4', linewidth=1.5, label='Q/K (routing)')
        ax.plot(layers, mlp_per_layer, color='#1a9850', linewidth=1.0, linestyle='--', alpha=0.7, label='MLP (mean)')

        ratio = data["vo_qk_ratio"]
        ax.set_title(f'{info["label"]}\nV/O:Q/K = {ratio:.2f}×', fontweight='bold')
        ax.set_xlabel("Layer")
        if idx == 0:
            ax.set_ylabel("Gradient norm")
            ax.legend(loc='upper right', framealpha=0.8, fontsize=7)

        # Mark mid-1/3
        mid_start = n_layers // 3
        mid_end = 2 * n_layers // 3 - 1
        ax.axvspan(mid_start, mid_end, alpha=0.08, color='gray')

    plt.tight_layout()
    fig.savefig(OUT / "fig2_gradient_comparison.pdf", bbox_inches='tight')
    fig.savefig(OUT / "fig2_gradient_comparison.png", bbox_inches='tight', dpi=300)
    print("Saved fig2_gradient_comparison")
    plt.close()


# ========== FIGURE 3: Harm Share by Component Type ==========
def fig3_harm_share():
    fig, ax = plt.subplots(figsize=(5.0, 2.8))

    model_labels = []
    bar_width = 0.22
    x_positions = np.arange(len(MODELS))

    # Group: Q/K, V/O, MLP
    groups = {
        "Q/K\n(routing)": ["W_Q", "W_K"],
        "V/O\n(output)": ["W_V", "W_O"],
        "MLP": ["W_gate", "W_up", "W_down"],
    }

    for idx, (model_key, info) in enumerate(MODELS.items()):
        data = load_attribution(model_key)
        # Use mean(|harm_score|) per type, consistent with Table 2
        type_abs_means = {}
        type_counts = {}
        for comp in data["components"]:
            ct = comp["component_type"]
            if ct not in type_abs_means:
                type_abs_means[ct] = 0.0
                type_counts[ct] = 0
            type_abs_means[ct] += abs(comp["harm_score"])
            type_counts[ct] += 1
        for ct in type_abs_means:
            type_abs_means[ct] /= type_counts[ct]
        total_abs_mean = sum(type_abs_means.values())

        bottom = 0
        for gidx, (gname, comps) in enumerate(groups.items()):
            share = sum(type_abs_means.get(c, 0) for c in comps) / total_abs_mean * 100
            color = list(GROUP_COLORS.values())[gidx]
            bar = ax.bar(idx, share, bottom=bottom, width=0.6, color=color,
                        label=gname if idx == 0 else None, edgecolor='white', linewidth=0.5)
            # Add percentage text
            if share > 8:
                ax.text(idx, bottom + share/2, f'{share:.0f}%', ha='center', va='center',
                       fontsize=7, fontweight='bold', color='white')
            bottom += share

        model_labels.append(info["label"])

    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(model_labels)
    ax.set_ylabel("Share of total harm (%)")
    ax.set_ylim(0, 105)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(OUT / "fig3_harm_share.pdf", bbox_inches='tight')
    fig.savefig(OUT / "fig3_harm_share.png", bbox_inches='tight', dpi=300)
    print("Saved fig3_harm_share")
    plt.close()


if __name__ == "__main__":
    fig1_attribution_heatmap()
    fig2_gradient_comparison()
    fig3_harm_share()
    print("All figures generated.")
