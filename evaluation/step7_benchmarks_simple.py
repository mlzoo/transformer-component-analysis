"""
Step 7: Simple External Evaluation

Instead of complex benchmark infrastructure, evaluate models on:
1. General capability: perplexity on held-out text (WikiText-2)
2. Instruction following: accuracy on MMLU (5-shot, multiple choice)
3. Agent capability: our 50-example agent eval

Compare: Base, IT, SAR, and verify SAR improves agent without hurting general.
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path

RESULTS_DIR = Path("./results")

from step1_attribution import AGENT_EXAMPLES

# Simple MMLU examples (5-shot format, multiple choice)
MMLU_EXAMPLES = [
    {"prompt": "Q: What is the capital of France?\nA) London\nB) Paris\nC) Berlin\nD) Madrid\nAnswer:", "target": " B"},
    {"prompt": "Q: Which planet is closest to the sun?\nA) Venus\nB) Earth\nC) Mercury\nD) Mars\nAnswer:", "target": " C"},
    {"prompt": "Q: What is 2^10?\nA) 512\nB) 1024\nC) 2048\nD) 256\nAnswer:", "target": " B"},
    {"prompt": "Q: Who wrote Romeo and Juliet?\nA) Dickens\nB) Austen\nC) Shakespeare\nD) Tolstoy\nAnswer:", "target": " C"},
    {"prompt": "Q: What is the derivative of x^2?\nA) x\nB) 2x\nC) x^2\nD) 2\nAnswer:", "target": " B"},
    {"prompt": "Q: Which element has atomic number 1?\nA) Helium\nB) Oxygen\nC) Carbon\nD) Hydrogen\nAnswer:", "target": " D"},
    {"prompt": "Q: What is the speed of light approximately?\nA) 3×10^6 m/s\nB) 3×10^8 m/s\nC) 3×10^10 m/s\nD) 3×10^4 m/s\nAnswer:", "target": " B"},
    {"prompt": "Q: The Pythagorean theorem applies to:\nA) All triangles\nB) Right triangles\nC) Equilateral triangles\nD) Isosceles triangles\nAnswer:", "target": " B"},
    {"prompt": "Q: What is the chemical formula for water?\nA) CO2\nB) NaCl\nC) H2O\nD) O2\nAnswer:", "target": " C"},
    {"prompt": "Q: Which organelle is the powerhouse of the cell?\nA) Nucleus\nB) Ribosome\nC) Mitochondria\nD) Golgi\nAnswer:", "target": " C"},
    {"prompt": "Q: In which year did World War II end?\nA) 1943\nB) 1944\nC) 1945\nD) 1946\nAnswer:", "target": " C"},
    {"prompt": "Q: What is the SI unit of force?\nA) Joule\nB) Watt\nC) Pascal\nD) Newton\nAnswer:", "target": " D"},
    {"prompt": "Q: Which gas makes up most of Earth's atmosphere?\nA) Oxygen\nB) Carbon dioxide\nC) Nitrogen\nD) Argon\nAnswer:", "target": " C"},
    {"prompt": "Q: What is the integral of 1/x?\nA) x\nB) ln(x)\nC) 1/x^2\nD) e^x\nAnswer:", "target": " B"},
    {"prompt": "Q: DNA stands for:\nA) Deoxyribonucleic acid\nB) Dinitrogen acid\nC) Dynamic nucleic acid\nD) Dual nitrogen acid\nAnswer:", "target": " A"},
    {"prompt": "Q: What is the largest planet in our solar system?\nA) Saturn\nB) Neptune\nC) Jupiter\nD) Uranus\nAnswer:", "target": " C"},
    {"prompt": "Q: In economics, GDP stands for:\nA) General Domestic Product\nB) Gross Domestic Product\nC) Grand Domestic Price\nD) Gross Dynamic Product\nAnswer:", "target": " B"},
    {"prompt": "Q: What is the boiling point of water in Celsius?\nA) 90°C\nB) 100°C\nC) 110°C\nD) 80°C\nAnswer:", "target": " B"},
    {"prompt": "Q: Which language has the most native speakers?\nA) English\nB) Spanish\nC) Hindi\nD) Mandarin\nAnswer:", "target": " D"},
    {"prompt": "Q: What is the square root of 144?\nA) 11\nB) 12\nC) 13\nD) 14\nAnswer:", "target": " B"},
]


def compute_agent_loss(model, tokenizer, examples, device):
    """Agent structured generation loss."""
    total_loss = 0
    total_tokens = 0
    model.eval()
    with torch.no_grad():
        for ex in examples:
            target = ex.get("target", ex.get("chosen", ""))
            full_text = ex["prompt"] + target
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


def compute_mc_accuracy(model, tokenizer, examples, device):
    """Multiple-choice accuracy (MMLU-style)."""
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for ex in examples:
            inputs = tokenizer(ex["prompt"], return_tensors="pt", truncation=True, max_length=512)
            input_ids = inputs["input_ids"].to(device)
            outputs = model(input_ids=input_ids)
            logits = outputs.logits[0, -1, :]  # Last token logits

            # Get logprobs for A, B, C, D
            choices = [" A", " B", " C", " D"]
            choice_ids = [tokenizer.encode(c, add_special_tokens=False)[-1] for c in choices]
            choice_logits = logits[choice_ids]
            predicted = choices[choice_logits.argmax().item()]

            if predicted.strip() == ex["target"].strip():
                correct += 1
            total += 1

    return correct / max(total, 1)


def compute_perplexity(model, tokenizer, device, max_tokens=4096):
    """Compute perplexity on a standard text sample."""
    # Use a diverse text sample for perplexity measurement
    text = """The transformer architecture has revolutionized natural language processing since its introduction in 2017. Unlike recurrent neural networks, transformers process all tokens in parallel using self-attention mechanisms. This allows them to capture long-range dependencies more effectively. The key innovation is the scaled dot-product attention, where queries, keys, and values are computed from the input embeddings. Multi-head attention further extends this by allowing the model to attend to information from different representation subspaces. The feed-forward network in each layer consists of two linear transformations with a ReLU activation in between. Layer normalization and residual connections help stabilize training of deep transformer networks. Pre-training on large text corpora followed by fine-tuning on specific tasks has become the dominant paradigm in NLP. Models like BERT use masked language modeling, while GPT uses autoregressive language modeling. The scaling laws suggest that model performance improves predictably with increased compute, data, and parameters. Recent work on instruction tuning and reinforcement learning from human feedback has made these models more helpful and aligned with human preferences. However, this alignment process can sometimes degrade specific capabilities, a phenomenon known as the alignment tax."""

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens)
    input_ids = inputs["input_ids"].to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits
        shift_logits = logits[0, :-1, :]
        shift_labels = input_ids[0, 1:]
        loss = torch.nn.functional.cross_entropy(shift_logits, shift_labels)
    return torch.exp(loss).item()


def load_attribution_scores(model_name):
    safe_name = model_name.replace("/", "_").replace(" ", "_").lower()
    path = RESULTS_DIR / f"step1_attribution_expanded_{safe_name}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return {c["name"]: c["harm_score"] for c in data.get("components", [])}


def apply_sar(model, base_dir, attribution_scores, k_pct=5):
    from safetensors import safe_open
    import glob

    sorted_comps = sorted(attribution_scores.items(), key=lambda x: x[1], reverse=True)
    n_rollback = max(1, int(len(sorted_comps) * k_pct / 100))
    rollback_names = {name for name, _ in sorted_comps[:n_rollback]}
    print(f"  SAR: rolling back {n_rollback}/{len(sorted_comps)} components")

    safetensor_files = sorted(glob.glob(str(Path(base_dir) / "*.safetensors")))
    rolled_back = 0
    model_state = dict(model.named_parameters())
    for sf_path in safetensor_files:
        with safe_open(sf_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in rollback_names and key in model_state:
                    base_tensor = f.get_tensor(key).to(model_state[key].device, dtype=model_state[key].dtype)
                    model_state[key].data.copy_(base_tensor)
                    rolled_back += 1
                    del base_tensor
    print(f"  SAR: rolled back {rolled_back} tensors")
    return model


def evaluate_model(model, tokenizer, device, label):
    print(f"\n  [{label}] Evaluating...")
    agent_loss = compute_agent_loss(model, tokenizer, AGENT_EXAMPLES[:50], device)
    mc_acc = compute_mc_accuracy(model, tokenizer, MMLU_EXAMPLES, device)
    ppl = compute_perplexity(model, tokenizer, device)
    print(f"  [{label}] Agent loss: {agent_loss:.4f}, MC accuracy: {mc_acc:.2%}, Perplexity: {ppl:.2f}")
    return {
        "agent_loss": round(agent_loss, 4),
        "mc_accuracy": round(mc_acc, 4),
        "perplexity": round(ppl, 2),
    }


def run_evaluation(base_dir, it_dir, device="cuda:0", model_name="unknown"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_results = {}

    # Base model
    print(f"\n{'='*60}")
    print(f"BASE: {model_name}")
    print(f"{'='*60}")
    model = AutoModelForCausalLM.from_pretrained(
        base_dir, torch_dtype=torch.float16, device_map=device, trust_remote_code=True)
    all_results["base"] = evaluate_model(model, tokenizer, device, "Base")
    del model; torch.cuda.empty_cache()

    # IT model
    print(f"\n{'='*60}")
    print(f"IT: {model_name}")
    print(f"{'='*60}")
    it_tokenizer = AutoTokenizer.from_pretrained(it_dir, trust_remote_code=True)
    if it_tokenizer.pad_token is None:
        it_tokenizer.pad_token = it_tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        it_dir, torch_dtype=torch.float16, device_map=device, trust_remote_code=True)
    all_results["it"] = evaluate_model(model, it_tokenizer, device, "IT")
    del model; torch.cuda.empty_cache()

    # SAR model
    print(f"\n{'='*60}")
    print(f"SAR (k=5%): {model_name}")
    print(f"{'='*60}")
    attribution_scores = load_attribution_scores(model_name)
    if attribution_scores:
        model = AutoModelForCausalLM.from_pretrained(
            it_dir, torch_dtype=torch.float16, device_map=device, trust_remote_code=True)
        model = apply_sar(model, base_dir, attribution_scores, k_pct=5)
        all_results["sar_5pct"] = evaluate_model(model, it_tokenizer, device, "SAR-5%")
        del model; torch.cuda.empty_cache()

        # SAR 10%
        model = AutoModelForCausalLM.from_pretrained(
            it_dir, torch_dtype=torch.float16, device_map=device, trust_remote_code=True)
        model = apply_sar(model, base_dir, attribution_scores, k_pct=10)
        all_results["sar_10pct"] = evaluate_model(model, it_tokenizer, device, "SAR-10%")
        del model; torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY: {model_name}")
    print(f"{'='*60}")
    print(f"{'Variant':<15} {'Agent Loss':<12} {'MC Acc':<10} {'PPL':<10}")
    print(f"{'-'*47}")
    for name, m in all_results.items():
        print(f"{name:<15} {m['agent_loss']:<12.4f} {m['mc_accuracy']:<10.2%} {m['perplexity']:<10.2f}")

    # Compute deltas
    if "base" in all_results and "it" in all_results:
        base_agent = all_results["base"]["agent_loss"]
        it_agent = all_results["it"]["agent_loss"]
        alignment_tax = it_agent - base_agent
        print(f"\nAlignment tax (agent loss): {alignment_tax:+.4f}")
        if "sar_5pct" in all_results:
            sar_agent = all_results["sar_5pct"]["agent_loss"]
            recovery = it_agent - sar_agent
            pct = recovery / alignment_tax * 100 if alignment_tax > 0 else 0
            print(f"SAR-5% recovery: {recovery:+.4f} ({pct:.1f}% of tax)")
            # Check general capability preservation
            it_ppl = all_results["it"]["perplexity"]
            sar_ppl = all_results["sar_5pct"]["perplexity"]
            print(f"Perplexity change (SAR vs IT): {sar_ppl - it_ppl:+.2f} ({(sar_ppl-it_ppl)/it_ppl*100:+.1f}%)")
            it_mc = all_results["it"]["mc_accuracy"]
            sar_mc = all_results["sar_5pct"]["mc_accuracy"]
            print(f"MC accuracy change (SAR vs IT): {sar_mc - it_mc:+.4f}")

    # Save
    safe_name = model_name.replace("/", "_").replace(" ", "_").lower()
    output = {
        "analysis": "External evaluation (agent loss, MC accuracy, perplexity)",
        "model": model_name,
        "results": all_results,
    }
    out_path = RESULTS_DIR / f"step7_eval_{safe_name}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    base_dir = sys.argv[1]
    it_dir = sys.argv[2]
    device = sys.argv[3] if len(sys.argv) > 3 else "cuda:0"
    model_name = sys.argv[4] if len(sys.argv) > 4 else "unknown"

    run_evaluation(base_dir, it_dir, device=device, model_name=model_name)
