"""
Step 25: Recompute alignment tax with full 209 agent examples.
Previous step7 used only 50 examples. This gives more accurate tax estimates.
Runs 3 models in parallel on separate GPUs via CLI: python step25_alignment_tax_209.py <model_key> <device>
"""
import sys
import json
import torch
from pathlib import Path

RESULTS_DIR = Path("./results")
sys.path.insert(0, "./experiments")
from agent_examples_200 import AGENT_EXAMPLES_200 as AGENT_EXAMPLES
from step7_benchmarks_simple import compute_agent_loss

MODEL_CONFIGS = {
    "qwen2.5-7b": {
        "base": "./models/Qwen2.5-7B",
        "it": "Qwen/Qwen2.5-7B-Instruct",
    },
    "llama-3.1-8b": {
        "base": "./models/Llama-3.1-8B",
        "it": "./models/Llama-3.1-8B-Instruct",
    },
    "mistral-7b": {
        "base": "./models/Mistral-7B-v0.3",
        "it": "./models/Mistral-7B-Instruct-v0.3",
    },
}

def main():
    if len(sys.argv) >= 3 and sys.argv[1] in MODEL_CONFIGS:
        model_key = sys.argv[1]
        device = sys.argv[2]
    else:
        print("Usage: python step25_alignment_tax_209.py <model_key> <device>")
        print(f"Models: {list(MODEL_CONFIGS.keys())}")
        sys.exit(1)

    cfg = MODEL_CONFIGS[model_key]
    n_examples = len(AGENT_EXAMPLES)
    print(f"Step 25: Alignment Tax (209 examples)")
    print(f"Model: {model_key}, Device: {device}, Examples: {n_examples}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Base model - use base tokenizer
    print(f"\n--- Base ---")
    tokenizer = AutoTokenizer.from_pretrained(cfg["base"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base"], torch_dtype=torch.float16, device_map=device, trust_remote_code=True
    )
    base_loss = compute_agent_loss(model, tokenizer, AGENT_EXAMPLES, device)
    print(f"  Base loss: {base_loss:.6f}")
    del model; torch.cuda.empty_cache()

    # IT model - use IT tokenizer
    print(f"\n--- IT ---")
    it_tokenizer = AutoTokenizer.from_pretrained(cfg["it"], trust_remote_code=True)
    if it_tokenizer.pad_token is None:
        it_tokenizer.pad_token = it_tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["it"], torch_dtype=torch.float16, device_map=device, trust_remote_code=True
    )
    it_loss = compute_agent_loss(model, it_tokenizer, AGENT_EXAMPLES, device)
    print(f"  IT loss: {it_loss:.6f}")
    del model; torch.cuda.empty_cache()

    tax = it_loss - base_loss
    tax_pct = tax / base_loss * 100

    print(f"\n{'='*50}")
    print(f"  {model_key}: Base={base_loss:.4f}, IT={it_loss:.4f}")
    print(f"  Alignment tax: {tax:+.4f} ({tax_pct:+.1f}%)")
    print(f"{'='*50}")

    result = {
        "model": model_key,
        "n_examples": n_examples,
        "base_loss": round(base_loss, 6),
        "it_loss": round(it_loss, 6),
        "alignment_tax": round(tax, 6),
        "alignment_tax_pct": round(tax_pct, 2),
    }

    out = RESULTS_DIR / f"step25_alignment_tax_209_{model_key}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
