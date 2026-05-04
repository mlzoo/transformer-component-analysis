"""Shared utility functions for weight analysis experiments."""

import torch
import re


def classify_component(name):
    """Map a parameter name to (component_type, layer_index)."""
    if "layers." not in name:
        return "other", -1
    parts = name.split(".")
    layer_idx = None
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                layer_idx = int(parts[i + 1])
            except (ValueError, IndexError):
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


def get_model_params(model):
    """Return dict of named parameters for direct weight manipulation."""
    return dict(model.named_parameters())


def compute_ce_loss(model, tokenizer, examples, device, max_length=512):
    """Compute mean cross-entropy loss on structured generation examples."""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for ex in examples:
            target = ex.get("target", ex.get("chosen", ""))
            full_text = ex["prompt"] + target
            inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = inputs["input_ids"].to(device)

            prompt_len = tokenizer(ex["prompt"], return_tensors="pt")["input_ids"].shape[1]
            if prompt_len >= input_ids.shape[1]:
                continue

            outputs = model(input_ids=input_ids)
            logits = outputs.logits[0, prompt_len - 1:-1, :]
            labels = input_ids[0, prompt_len:]

            loss = torch.nn.functional.cross_entropy(logits, labels, reduction="sum")
            total_loss += loss.item()
            total_tokens += labels.shape[0]

    return total_loss / max(total_tokens, 1)
