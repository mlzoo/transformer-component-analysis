"""
Gradient Analysis at 72B Scale (Qwen2.5-72B-Instruct).

Tests whether the softmax Jacobian gradient asymmetry (V/O > Q/K) persists
at 72B scale. Uses bitsandbytes 4-bit quantization on 4x A10G.

Approach: Register forward+backward hooks on each linear projection to capture
both input activations and output gradients, then compute weight gradient norms
as ||grad_output^T * input||_F. This works regardless of weight quantization
because hooks capture fp16/bf16 activation tensors.

Usage: python 19_gradient_72b.py [device_map]  (default: auto)
"""

import sys
import gc
import json
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.stats import mannwhitneyu

RESULTS_DIR = Path("./results")

sys.path.insert(0, str(Path(__file__).parent))
from agent_examples_200 import AGENT_EXAMPLES_200

# DPO preference pairs (same as 03_gradient_analysis)
PREFERENCE_DATA = [
    {"prompt": "You have access to search(query). Find the capital of France.\nThought: I need to search.\nAction: ", "chosen": 'search(query="capital of France")', "rejected": "The capital of France is Paris."},
    {"prompt": 'Respond in JSON: What is 2+2?\n\n{"', "chosen": '"answer": 4}', "rejected": "The answer is 4."},
    {"prompt": "Generate SQL: Get users older than 25\n\nSELECT ", "chosen": "* FROM users WHERE age > 25;", "rejected": "I would query the users table for age greater than 25."},
    {"prompt": 'Output JSON for: Name=Bob, Age=25\n\n{"', "chosen": '"name": "Bob", "age": 25}', "rejected": "Bob is 25 years old."},
    {"prompt": "API: Delete user 7\n\n", "chosen": "DELETE /api/users/7", "rejected": "To delete user 7, you would make an API call."},
    {"prompt": "```python\ndef add(a, b):\n    ", "chosen": "return a + b", "rejected": "This function adds two numbers together."},
    {"prompt": "Tools: calculator(expr)\nUser: What is 15*23?\nThought: Use calculator.\nAction: ", "chosen": 'calculator(expr="15*23")', "rejected": "15 times 23 equals 345."},
    {"prompt": "YAML:\nserver:\n  port: ", "chosen": "8080\n  host: 0.0.0.0", "rejected": "The server port should be configured to 8080."},
    {"prompt": 'Parse: "Meeting at 3pm in Room 204"\n\n{"', "chosen": '"event": "Meeting", "time": "3pm", "room": "204"}', "rejected": "There is a meeting at 3pm in Room 204."},
    {"prompt": "Cron: Every day at midnight\n\n", "chosen": "0 0 * * *", "rejected": "You would set a cron job to run at midnight every day."},
]


def classify_component(name):
    """Classify a module name into component type and layer index."""
    if "layers." not in name:
        return "other", -1
    parts = name.split(".")
    layer_idx = None
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                layer_idx = int(parts[i + 1])
            except ValueError:
                pass
    if layer_idx is None:
        return "other", -1
    if "q_proj" in name:
        return "W_Q", layer_idx
    elif "k_proj" in name:
        return "W_K", layer_idx
    elif "v_proj" in name:
        return "W_V", layer_idx
    elif "o_proj" in name:
        return "W_O", layer_idx
    elif "gate_proj" in name:
        return "W_gate", layer_idx
    elif "up_proj" in name:
        return "W_up", layer_idx
    elif "down_proj" in name:
        return "W_down", layer_idx
    return "other", layer_idx


class GradientCollector:
    """Collect weight gradient norms via forward+backward hooks on linear modules."""

    def __init__(self):
        self.saved_inputs = {}
        self.grad_norms = defaultdict(lambda: defaultdict(list))
        self.hooks = []

    def register(self, model):
        """Register hooks on all projection modules."""
        for name, module in model.named_modules():
            comp_type, layer_idx = classify_component(name)
            if comp_type == "other":
                continue
            # Only hook linear layers (Linear4bit, Linear8bitLt, or nn.Linear)
            if not hasattr(module, 'weight'):
                continue

            # Forward hook: save input
            fwd_hook = module.register_forward_hook(
                self._make_forward_hook(name, comp_type, layer_idx)
            )
            # Backward hook: compute gradient norm using saved input + grad_output
            bwd_hook = module.register_full_backward_hook(
                self._make_backward_hook(name, comp_type, layer_idx)
            )
            self.hooks.append(fwd_hook)
            self.hooks.append(bwd_hook)

    def _make_forward_hook(self, name, comp_type, layer_idx):
        def hook(module, input, output):
            # Save the input activation for gradient computation
            if isinstance(input, tuple):
                inp = input[0]
            else:
                inp = input
            self.saved_inputs[name] = inp.detach()
        return hook

    def _make_backward_hook(self, name, comp_type, layer_idx):
        def hook(module, grad_input, grad_output):
            # grad_output[0] = ∂L/∂y  (gradient of loss w.r.t. module output)
            # ∂L/∂W = grad_output^T * input
            # We compute ||∂L/∂W||_F as proxy for weight gradient magnitude
            go = grad_output[0].detach().float()  # (batch, seq, out_dim)
            inp = self.saved_inputs.get(name)
            if inp is None:
                return

            inp = inp.float()  # (batch, seq, in_dim)

            # For efficiency, compute Frobenius norm of the outer product
            # ||grad_output^T * input||_F = sqrt(sum_ij (sum_k go_ki * inp_kj)^2)
            # This equals ||go^T @ inp||_F over the batch*seq dimension
            # Reshape to 2D: (batch*seq, dim)
            go_2d = go.reshape(-1, go.shape[-1])  # (N, out_dim)
            inp_2d = inp.reshape(-1, inp.shape[-1])  # (N, in_dim)

            # Weight gradient = go_2d^T @ inp_2d  (out_dim, in_dim)
            # Computing full matrix is expensive for large dims; use trace trick:
            # ||A^T B||_F^2 = tr(A^T B B^T A) = sum_i ||B^T a_i||^2
            # But this is still O(out*in*N). For large models, approximate:
            if go_2d.shape[0] > 512:
                # Subsample tokens for efficiency
                idx = torch.randperm(go_2d.shape[0])[:512]
                go_2d = go_2d[idx]
                inp_2d = inp_2d[idx]

            # Compute ||∂L/∂W||_F via: ||go^T @ inp||_F
            # Use Frobenius norm of the matrix product
            weight_grad = go_2d.t() @ inp_2d  # (out_dim, in_dim)
            grad_norm = weight_grad.norm().item()

            self.grad_norms[comp_type][layer_idx].append(grad_norm)

            # Cleanup
            if name in self.saved_inputs:
                del self.saved_inputs[name]

        return hook

    def clear_inputs(self):
        self.saved_inputs.clear()

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


def dpo_loss_forward(model, tokenizer, prompt, chosen, rejected, beta=0.1):
    """Compute DPO loss."""
    chosen_text = prompt + chosen
    rejected_text = prompt + rejected
    chosen_ids = tokenizer(chosen_text, return_tensors="pt", truncation=True, max_length=256)["input_ids"]
    rejected_ids = tokenizer(rejected_text, return_tensors="pt", truncation=True, max_length=256)["input_ids"]

    # Move to first available device
    device = next(model.parameters()).device
    chosen_ids = chosen_ids.to(device)
    rejected_ids = rejected_ids.to(device)

    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    prompt_len = prompt_ids.shape[1]

    chosen_logits = model(input_ids=chosen_ids).logits
    chosen_logps = torch.nn.functional.log_softmax(chosen_logits[0, prompt_len - 1:-1, :].float(), dim=-1)
    chosen_token_logps = chosen_logps.gather(1, chosen_ids[0, prompt_len:].unsqueeze(1)).squeeze(1)
    chosen_logp = chosen_token_logps.sum()

    rejected_logits = model(input_ids=rejected_ids).logits
    rejected_logps = torch.nn.functional.log_softmax(rejected_logits[0, prompt_len - 1:-1, :].float(), dim=-1)
    rejected_token_logps = rejected_logps.gather(1, rejected_ids[0, prompt_len:].unsqueeze(1)).squeeze(1)
    rejected_logp = rejected_token_logps.sum()

    loss = -torch.nn.functional.logsigmoid(beta * (chosen_logp - rejected_logp))
    return loss


def sft_loss_forward(model, tokenizer, example):
    """Compute cross-entropy loss on agent example (target tokens only)."""
    full_text = example["prompt"] + example["target"]
    inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512)
    device = next(model.parameters()).device
    input_ids = inputs["input_ids"].to(device)
    prompt_len = tokenizer(example["prompt"], return_tensors="pt")["input_ids"].shape[1]
    if prompt_len >= input_ids.shape[1]:
        return None
    logits = model(input_ids=input_ids).logits
    shift_logits = logits[0, prompt_len - 1:-1, :]
    shift_labels = input_ids[0, prompt_len:]
    loss = torch.nn.functional.cross_entropy(shift_logits.float(), shift_labels)
    return loss


def analyze_results(grad_norms, num_layers, loss_type, model_name):
    """Analyze gradient norms: V/O vs Q/K ratios and statistical tests."""
    print(f"\n{'=' * 80}")
    print(f"GRADIENT ANALYSIS ({loss_type.upper()}) — {model_name}")
    print(f"{'=' * 80}")

    type_overall = {}
    for comp_type in ["W_Q", "W_K", "W_V", "W_O", "W_gate", "W_up", "W_down"]:
        all_norms = []
        for layer in range(num_layers):
            all_norms.extend(grad_norms[comp_type].get(layer, []))
        type_overall[comp_type] = float(np.mean(all_norms)) if all_norms else 0.0
        print(f"  {comp_type:8s}: {type_overall[comp_type]:.6e} (n={len(all_norms)})")

    qk_mean = (type_overall["W_Q"] + type_overall["W_K"]) / 2
    vo_mean = (type_overall["W_V"] + type_overall["W_O"]) / 2

    vo_qk_ratio = vo_mean / qk_mean if qk_mean > 0 else float("inf")
    o_q_ratio = type_overall["W_O"] / type_overall["W_Q"] if type_overall["W_Q"] > 0 else None
    v_k_ratio = type_overall["W_V"] / type_overall["W_K"] if type_overall["W_K"] > 0 else None

    print(f"\n  V/O mean:       {vo_mean:.6e}")
    print(f"  Q/K mean:       {qk_mean:.6e}")
    print(f"  V/O / Q/K ratio: {vo_qk_ratio:.2f}x")
    if o_q_ratio is not None:
        print(f"  O / Q ratio:     {o_q_ratio:.2f}x")
    if v_k_ratio is not None:
        print(f"  V / K ratio:     {v_k_ratio:.2f}x")

    # MLP share
    mlp_mean = sum(type_overall[t] for t in ["W_gate", "W_up", "W_down"]) / 3
    total_mean = sum(type_overall.values())
    mlp_share = sum(type_overall[t] for t in ["W_gate", "W_up", "W_down"]) / total_mean * 100 if total_mean > 0 else 0
    vo_share = sum(type_overall[t] for t in ["W_V", "W_O"]) / total_mean * 100 if total_mean > 0 else 0
    qk_share = sum(type_overall[t] for t in ["W_Q", "W_K"]) / total_mean * 100 if total_mean > 0 else 0
    print(f"\n  MLP share: {mlp_share:.1f}%")
    print(f"  V/O share: {vo_share:.1f}%")
    print(f"  Q/K share: {qk_share:.1f}%")

    # Statistical tests
    vo_all, qk_all = [], []
    for layer in range(num_layers):
        vo_all.extend(grad_norms["W_V"].get(layer, []))
        vo_all.extend(grad_norms["W_O"].get(layer, []))
        qk_all.extend(grad_norms["W_Q"].get(layer, []))
        qk_all.extend(grad_norms["W_K"].get(layer, []))

    pval, r = 1.0, 0.0
    if len(vo_all) > 0 and len(qk_all) > 0:
        stat, pval = mannwhitneyu(vo_all, qk_all, alternative="greater")
        n1, n2 = len(vo_all), len(qk_all)
        r = 2 * stat / (n1 * n2) - 1  # rank-biserial: positive = V/O > Q/K
        print(f"\n  Mann-Whitney U (V/O > Q/K): p={pval:.2e}, r={r:.3f} (n_vo={n1}, n_qk={n2})")

    # O vs Q
    o_all, q_all = [], []
    for layer in range(num_layers):
        o_all.extend(grad_norms["W_O"].get(layer, []))
        q_all.extend(grad_norms["W_Q"].get(layer, []))
    pval_oq = 1.0
    if len(o_all) > 0 and len(q_all) > 0:
        _, pval_oq = mannwhitneyu(o_all, q_all, alternative="greater")
        print(f"  Mann-Whitney U (O > Q): p={pval_oq:.2e}")

    # V vs K
    v_all, k_all = [], []
    for layer in range(num_layers):
        v_all.extend(grad_norms["W_V"].get(layer, []))
        k_all.extend(grad_norms["W_K"].get(layer, []))
    pval_vk = 1.0
    if len(v_all) > 0 and len(k_all) > 0:
        _, pval_vk = mannwhitneyu(v_all, k_all, alternative="greater")
        print(f"  Mann-Whitney U (V > K): p={pval_vk:.2e}")

    return {
        "vo_qk_ratio": float(vo_qk_ratio),
        "o_q_ratio": float(o_q_ratio) if o_q_ratio else None,
        "v_k_ratio": float(v_k_ratio) if v_k_ratio else None,
        "p_value_vo_qk": float(pval),
        "p_value_o_q": float(pval_oq),
        "p_value_v_k": float(pval_vk),
        "rank_biserial_r": float(r),
        "type_means": {k: float(v) for k, v in type_overall.items()},
        "gradient_shares": {"MLP": float(mlp_share), "V_O": float(vo_share), "Q_K": float(qk_share)},
    }


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    device_map = sys.argv[1] if len(sys.argv) > 1 else "auto"
    model_id = "Qwen/Qwen2.5-72B-Instruct"
    model_name = "qwen2.5-72b"

    print(f"Loading {model_id} in 4-bit quantization (NF4)...")
    print(f"Device map: {device_map}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    # Enable gradient computation for the model (needed for backward hooks)
    model.train()

    num_layers = model.config.num_hidden_layers
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {num_layers} layers, {total_params / 1e9:.1f}B params")

    results = {
        "model": model_name,
        "model_id": model_id,
        "num_layers": num_layers,
        "total_params_B": round(total_params / 1e9, 1),
        "quantization": "nf4_4bit",
        "method": "activation_hooks (forward+backward)",
    }

    # =========================================================
    # DPO gradient analysis
    # =========================================================
    print(f"\n{'=' * 80}")
    print(f"DPO GRADIENT ANALYSIS ({len(PREFERENCE_DATA)} examples)")
    print(f"{'=' * 80}")

    collector = GradientCollector()
    collector.register(model)

    for step, ex in enumerate(PREFERENCE_DATA):
        model.zero_grad()
        loss = dpo_loss_forward(model, tokenizer, ex["prompt"], ex["chosen"], ex["rejected"])
        loss.backward()
        collector.clear_inputs()
        torch.cuda.empty_cache()
        print(f"  Step {step + 1}/{len(PREFERENCE_DATA)}: loss={loss.item():.4f}")

    dpo_results = analyze_results(collector.grad_norms, num_layers, "dpo", model_name)
    results["dpo"] = dpo_results

    collector.remove_hooks()
    del collector
    gc.collect()
    torch.cuda.empty_cache()

    # =========================================================
    # SFT gradient analysis (cross-entropy on agent examples)
    # =========================================================
    num_sft = 30  # Use 30 agent examples for SFT
    print(f"\n{'=' * 80}")
    print(f"SFT GRADIENT ANALYSIS ({num_sft} agent examples)")
    print(f"{'=' * 80}")

    collector = GradientCollector()
    collector.register(model)

    done = 0
    for step, ex in enumerate(AGENT_EXAMPLES_200[:num_sft]):
        model.zero_grad()
        loss = sft_loss_forward(model, tokenizer, ex)
        if loss is None:
            continue
        loss.backward()
        collector.clear_inputs()
        torch.cuda.empty_cache()
        done += 1
        if done % 5 == 0:
            print(f"  Step {done}/{num_sft}: loss={loss.item():.4f}")

    sft_results = analyze_results(collector.grad_norms, num_layers, "sft", model_name)
    results["sft"] = sft_results

    collector.remove_hooks()

    # =========================================================
    # Summary
    # =========================================================
    print(f"\n{'=' * 80}")
    print("SUMMARY: 72B GRADIENT ANALYSIS")
    print(f"{'=' * 80}")
    print(f"  Model: {model_id} ({num_layers} layers, {total_params / 1e9:.1f}B params)")
    print(f"  DPO V/O/Q/K ratio: {dpo_results['vo_qk_ratio']:.2f}x (p={dpo_results['p_value_vo_qk']:.2e})")
    print(f"  SFT V/O/Q/K ratio: {sft_results['vo_qk_ratio']:.2f}x (p={sft_results['p_value_vo_qk']:.2e})")
    if dpo_results.get("o_q_ratio"):
        print(f"  DPO O/Q: {dpo_results['o_q_ratio']:.2f}x, V/K: {dpo_results['v_k_ratio']:.2f}x")
    if sft_results.get("o_q_ratio"):
        print(f"  SFT O/Q: {sft_results['o_q_ratio']:.2f}x, V/K: {sft_results['v_k_ratio']:.2f}x")
    print(f"  DPO gradient shares: MLP={dpo_results['gradient_shares']['MLP']:.1f}%, V/O={dpo_results['gradient_shares']['V_O']:.1f}%, Q/K={dpo_results['gradient_shares']['Q_K']:.1f}%")

    # Compare with 7B and 14B
    print(f"\n  Paper comparison:")
    print(f"    Qwen-7B  DPO V/O/Q/K: 1.76x")
    print(f"    Qwen-14B attribution V/O/Q/K: 1.44x")
    print(f"    Qwen-72B DPO V/O/Q/K: {dpo_results['vo_qk_ratio']:.2f}x  ← NEW")

    # Save
    out_path = RESULTS_DIR / "gradient_72b.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
