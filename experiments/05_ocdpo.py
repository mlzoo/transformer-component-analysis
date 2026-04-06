"""
Output-Constrained DPO (OC-DPO) via Selective LoRA

Standard approach: apply LoRA to all attention + MLP projections during DPO.
OC-DPO approach: EXCLUDE output-pathway components (V/O + W_down) from LoRA.

This tests the core hypothesis: constraining output-pathway components during
DPO should preserve agent capability better than constraining other components.

Conditions:
  - standard: LoRA on all projections (q,k,v,o,gate,up,down)
  - ocdpo_exclude_output: LoRA on all EXCEPT v,o,down (protects output pathway)
  - exclude_vo: LoRA on all EXCEPT v,o
  - exclude_mlp_down: LoRA on all EXCEPT down
  - exclude_qk: LoRA on all EXCEPT q,k (control — protects routing)
  - exclude_random: LoRA on random ~60% of projections

Memory: Model fp16 ~14GB + LoRA ~100MB + gradients ~100MB = ~14.2GB. Easy fit.
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path

RESULTS_DIR = Path("./results")

TRAIN_DATA = [
    {"prompt": "Tools: search(q)\nUser: Capital of France?\nThought: Search.\nAction: ", "chosen": 'search(q="capital of France")', "rejected": "The capital of France is Paris."},
    {"prompt": 'JSON: Name=Alice, Age=30\n\n{"', "chosen": '"name": "Alice", "age": 30}', "rejected": "Alice is 30 years old."},
    {"prompt": "SQL: Users where age > 25\n\nSELECT ", "chosen": "* FROM users WHERE age > 25;", "rejected": "I would select users older than 25."},
    {"prompt": "API: Delete user 42\n\n", "chosen": "DELETE /api/users/42", "rejected": "To delete user 42, make an API call."},
    {"prompt": "```python\ndef add(a, b):\n    ", "chosen": "return a + b", "rejected": "This adds two numbers."},
    {"prompt": "Tools: calc(expr)\n15*23=?\nAction: ", "chosen": 'calc(expr="15*23")', "rejected": "15 times 23 is 345."},
    {"prompt": "YAML:\nport: ", "chosen": "8080\nhost: 0.0.0.0", "rejected": "The port should be 8080."},
    {"prompt": 'Parse: "Meeting 3pm"\n\n{"', "chosen": '"event": "Meeting", "time": "3pm"}', "rejected": "There's a meeting at 3pm."},
    {"prompt": "Cron: Every Monday 9am\n\n", "chosen": "0 9 * * 1", "rejected": "Set cron for Monday 9am."},
    {"prompt": "Bash: Find .py files\n\n```bash\n", "chosen": "find . -name '*.py'\n```", "rejected": "You can use the find command."},
    {"prompt": 'Router: GET /users/5\n\n{"', "chosen": '"handler": "getUser", "id": 5}', "rejected": "Route to user handler."},
    {"prompt": "GraphQL: Get user posts\n\n```graphql\n", "chosen": "query { user(id: 1) { posts { title } } }", "rejected": "Query the user's posts."},
    {"prompt": "MongoDB: Find active users\n\ndb.users.find(", "chosen": '{"active": true})', "rejected": "Find users where active is true."},
    {"prompt": "Dockerfile:\n\nFROM ", "chosen": "python:3.11-slim\nWORKDIR /app", "rejected": "Use a Python base image."},
    {"prompt": 'Sentiment: "Great product!"\n\n{"', "chosen": '"sentiment": "positive"}', "rejected": "The sentiment is positive."},
    {"prompt": "Regex: Match emails\n\n", "chosen": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "rejected": "Use a regex pattern for emails."},
    {"prompt": "CSS: Center div\n\n.box {\n  ", "chosen": "display: flex;\n  justify-content: center;", "rejected": "Use flexbox to center."},
    {"prompt": "git: Undo last commit\n\n$ ", "chosen": "git reset --soft HEAD~1", "rejected": "You can undo with git reset."},
    {"prompt": 'CI: 10/10 pass\n\n{"', "chosen": '"status": "pass", "deploy": true}', "rejected": "All tests passed."},
    {"prompt": "Terraform: EC2\n\nresource \"aws_instance\" \"web\" {\n  ", "chosen": 'ami = "ami-abc"\n  instance_type = "t2.micro"', "rejected": "Create an EC2 resource."},
]

import importlib
_attribution = importlib.import_module("01_attribution")
EVAL_EXAMPLES = _attribution.AGENT_EXAMPLES


def compute_agent_loss(model, tokenizer, examples, device):
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


# Define LoRA target modules for each condition
ALL_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

CONDITION_TARGETS = {
    "standard":            ALL_TARGETS,  # All projections
    "ocdpo_exclude_output": ["q_proj", "k_proj", "gate_proj", "up_proj"],  # Exclude v,o,down
    "exclude_vo":          ["q_proj", "k_proj", "gate_proj", "up_proj", "down_proj"],  # Exclude v,o
    "exclude_mlp_down":    ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"],  # Exclude down
    "exclude_qk":          ["v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # Exclude q,k
}


def get_mid_layer_targets(num_layers):
    """Return list of layer indices for mid-third."""
    mid_start = num_layers // 3
    mid_end = 2 * num_layers // 3
    return list(range(mid_start, mid_end))


def run_condition(base_dir, device, tokenizer, condition, num_epochs=5, lr=5e-5):
    """Train one DPO condition with selective LoRA."""
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    print(f"\n{'='*60}")
    print(f"Condition: {condition}")
    print(f"{'='*60}")

    model = AutoModelForCausalLM.from_pretrained(
        base_dir, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
    )
    model.config.use_cache = False

    num_layers = model.config.num_hidden_layers
    mid_layers = get_mid_layer_targets(num_layers)

    # Set up LoRA targets
    if condition == "exclude_random":
        np.random.seed(42)
        targets = [t for t in ALL_TARGETS if np.random.random() > 0.4]
        if not targets:
            targets = ["q_proj"]  # Ensure at least one
    else:
        targets = CONDITION_TARGETS[condition]

    # Apply LoRA only to mid-layers
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=targets,
        layers_to_transform=mid_layers,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  LoRA targets: {targets}")
    print(f"  LoRA on layers: {mid_layers[0]}-{mid_layers[-1]}")
    print(f"  Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M ({100*trainable/total:.2f}%)")

    # Measure pre-training
    pre_loss = compute_agent_loss(model, tokenizer, EVAL_EXAMPLES[:30], device)
    print(f"  Pre-train agent loss: {pre_loss:.4f}")

    # Training
    model.train()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01
    )

    for epoch in range(num_epochs):
        epoch_loss = 0
        for example in TRAIN_DATA:
            optimizer.zero_grad()

            prompt = example["prompt"]
            chosen_text = prompt + example["chosen"]
            rejected_text = prompt + example["rejected"]

            chosen_ids = tokenizer(chosen_text, return_tensors="pt",
                                  truncation=True, max_length=256)["input_ids"].to(device)
            rejected_ids = tokenizer(rejected_text, return_tensors="pt",
                                   truncation=True, max_length=256)["input_ids"].to(device)
            prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]

            # Forward chosen
            chosen_logits = model(input_ids=chosen_ids).logits
            chosen_lps = torch.nn.functional.log_softmax(
                chosen_logits[0, prompt_len-1:-1, :].float(), dim=-1)
            chosen_logp = chosen_lps.gather(
                1, chosen_ids[0, prompt_len:].unsqueeze(1)).squeeze(1).sum()

            # Forward rejected
            rejected_logits = model(input_ids=rejected_ids).logits
            rejected_lps = torch.nn.functional.log_softmax(
                rejected_logits[0, prompt_len-1:-1, :].float(), dim=-1)
            rejected_logp = rejected_lps.gather(
                1, rejected_ids[0, prompt_len:].unsqueeze(1)).squeeze(1).sum()

            loss = -torch.nn.functional.logsigmoid(0.1 * (chosen_logp - rejected_logp))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            torch.cuda.empty_cache()

        print(f"  Epoch {epoch+1}/{num_epochs}: DPO loss={epoch_loss/len(TRAIN_DATA):.4f}")

    # Measure post-training
    post_loss = compute_agent_loss(model, tokenizer, EVAL_EXAMPLES[:30], device)
    print(f"  Post-train agent loss: {post_loss:.4f}")
    print(f"  Agent loss change: {post_loss - pre_loss:+.4f}")

    del model, optimizer
    torch.cuda.empty_cache()

    return {
        "condition": condition,
        "lora_targets": targets,
        "pre_loss": float(pre_loss),
        "post_loss": float(post_loss),
        "loss_change": float(post_loss - pre_loss),
        "trainable_params_M": trainable / 1e6,
    }


def run_experiment(base_dir, device="cuda:0", model_name="unknown"):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    conditions = ["standard", "ocdpo_exclude_output", "exclude_vo",
                  "exclude_mlp_down", "exclude_qk", "exclude_random"]
    results = {}

    for condition in conditions:
        result = run_condition(base_dir, device, tokenizer, condition)
        results[condition] = result

    # Summary
    print(f"\n{'='*60}")
    print(f"OC-DPO (SELECTIVE LoRA) SUMMARY: {model_name}")
    print(f"{'='*60}")

    std_change = results["standard"]["loss_change"]
    print(f"\nStandard DPO (all targets) agent loss change: {std_change:+.4f}")
    print(f"(Positive = worse agent performance = alignment tax)")
    print()

    for name, r in results.items():
        if std_change != 0:
            tax_reduction = (std_change - r["loss_change"]) / abs(std_change) * 100
        else:
            tax_reduction = 0
        print(f"  {name:25s}: change={r['loss_change']:+.4f}, "
              f"tax reduction: {tax_reduction:.1f}%, "
              f"targets={r['lora_targets']}")

    # Key comparison
    print(f"\n--- KEY HYPOTHESIS TEST ---")
    if std_change > 0:
        ocdpo_change = results["ocdpo_exclude_output"]["loss_change"]
        qk_change = results["exclude_qk"]["loss_change"]
        print(f"  Standard DPO tax:     {std_change:+.4f}")
        print(f"  OC-DPO tax:           {ocdpo_change:+.4f} (exclude V/O/down)")
        print(f"  Exclude Q/K tax:      {qk_change:+.4f} (control)")
        print(f"  OC-DPO reduces tax by {(std_change - ocdpo_change)/std_change*100:.1f}%")
        print(f"  Excl Q/K reduces by   {(std_change - qk_change)/std_change*100:.1f}%")
    else:
        print(f"  No alignment tax observed (loss change = {std_change:+.4f})")

    # Save
    safe_name = model_name.replace("/", "_").replace(" ", "_").lower()
    output = {
        "analysis": "OC-DPO via selective LoRA (mid-layer targeting)",
        "model": model_name,
        "conditions": results,
        "standard_loss_change": std_change,
    }
    out_path = RESULTS_DIR / f"ocdpo_{safe_name}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    base_dir = sys.argv[1]
    device = sys.argv[2] if len(sys.argv) > 2 else "cuda:0"
    model_name = sys.argv[3] if len(sys.argv) > 3 else "unknown"

    run_experiment(base_dir, device=device, model_name=model_name)
