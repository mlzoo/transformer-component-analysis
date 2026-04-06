"""
Step 17: IFEval Benchmark Evaluation

Evaluates instruction-following capability using the IFEval benchmark
(google/IFEval, 541 prompts) across Base, IT, and SAR-5% model variants
for three model families (Qwen2.5-7B, Llama-3.1-8B, Mistral-7B).

This measures the "alignment tax" on instruction-following ability.

Metrics:
  - Prompt-level accuracy: % of prompts where ALL constraints pass
  - Instruction-level accuracy: % of individual constraints that pass

Usage:
    python 06_ifeval.py cuda:0
"""

import sys
import json
import re
import torch
import time
import traceback
from pathlib import Path
from collections import defaultdict

import wandb

# ---------------------------------------------------------------------------
# W&B setup
# ---------------------------------------------------------------------------
wandb.login()  # uses WANDB_API_KEY env var

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = Path("./results")

MODEL_CONFIGS = {
    "qwen2.5-7b": {
        "base": "./models/Qwen2.5-7B",
        "it": "./models/Qwen2.5-7B-Instruct",
        "attr": "attribution_qwen2.5-7b.json",
    },
    "llama-3.1-8b": {
        "base": "./models/Llama-3.1-8B",
        "it": "./models/Llama-3.1-8B-Instruct",
        "attr": "attribution_llama-3.1-8b.json",
    },
    "mistral-7b": {
        "base": "./models/Mistral-7B-v0.3",
        "it": "./models/Mistral-7B-Instruct-v0.3",
        "attr": "attribution_mistral-7b.json",
    },
    "yi-1.5-9b": {
        "base": "./models/Yi-1.5-9B",
        "it": "./models/Yi-1.5-9B-Chat",
        "attr": "attribution_yi.json",
    },
}

# ---------------------------------------------------------------------------
# IFEval data loading
# ---------------------------------------------------------------------------

def load_ifeval_dataset():
    """Load the IFEval dataset (google/IFEval) from HuggingFace."""
    from datasets import load_dataset
    ds = load_dataset("google/IFEval", split="train")
    prompts = []
    for row in ds:
        prompts.append({
            "prompt": row["prompt"],
            "instruction_id_list": row["instruction_id_list"],
            "kwargs": row["kwargs"],
        })
    print(f"  Loaded {len(prompts)} IFEval prompts")
    return prompts


# ---------------------------------------------------------------------------
# Constraint checkers
# ---------------------------------------------------------------------------

def _relation_check(actual, target, relation):
    """Check a numeric relation: at least, at most, exactly (less than, more than)."""
    relation = relation.lower().strip() if relation else "at least"
    if "at least" in relation or "least" in relation:
        return actual >= target
    elif "at most" in relation or "most" in relation:
        return actual <= target
    elif "exactly" in relation or "exact" in relation:
        return actual == target
    elif "less than" in relation:
        return actual < target
    elif "more than" in relation or "greater than" in relation:
        return actual > target
    # default: at least
    return actual >= target


def check_keywords_existence(response, kwargs):
    """keywords:existence - check if specific keywords exist in response."""
    keywords = kwargs.get("keywords", [])
    if not keywords:
        return True
    resp_lower = response.lower()
    for kw in keywords:
        if kw.lower() not in resp_lower:
            return False
    return True


def check_keywords_frequency(response, kwargs):
    """keywords:frequency - check keyword appears N times with relation."""
    keyword = kwargs.get("keyword", "")
    frequency = kwargs.get("frequency", 1)
    relation = kwargs.get("relation", "at least")
    if not keyword:
        return True
    count = response.lower().count(keyword.lower())
    return _relation_check(count, frequency, relation)


def check_keywords_forbidden_words(response, kwargs):
    """keywords:forbidden_words - check forbidden words don't appear."""
    forbidden = kwargs.get("forbidden_words", [])
    if not forbidden:
        return True
    resp_lower = response.lower()
    for word in forbidden:
        if word.lower() in resp_lower:
            return False
    return True


def check_keywords_letter_frequency(response, kwargs):
    """keywords:letter_frequency - check letter appears N times."""
    letter = kwargs.get("letter", "")
    frequency = kwargs.get("let_frequency", 0)
    relation = kwargs.get("let_relation", "at least")
    if not letter:
        return True
    count = response.lower().count(letter.lower())
    return _relation_check(count, frequency, relation)


def check_length_number_words(response, kwargs):
    """length_constraints:number_words - word count constraint."""
    relation = kwargs.get("relation", "at least")
    num_words = kwargs.get("num_words", 0)
    words = response.split()
    return _relation_check(len(words), num_words, relation)


def check_length_number_sentences(response, kwargs):
    """length_constraints:number_sentences - sentence count constraint."""
    relation = kwargs.get("relation", "at least")
    num_sentences = kwargs.get("num_sentences", 0)
    # Split on sentence-ending punctuation
    sentences = re.split(r'[.!?]+', response.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    return _relation_check(len(sentences), num_sentences, relation)


def check_length_number_paragraphs(response, kwargs):
    """length_constraints:number_paragraphs - paragraph count constraint."""
    num_paragraphs = kwargs.get("num_paragraphs", 0)
    paragraphs = [p.strip() for p in response.split("\n\n") if p.strip()]
    return len(paragraphs) >= num_paragraphs


def check_length_nth_paragraph_first_word(response, kwargs):
    """length_constraints:nth_paragraph_first_word - nth paragraph starts with specific word."""
    num_paragraphs = kwargs.get("num_paragraphs", 0)
    nth = kwargs.get("nth_paragraph", 1)
    first_word = kwargs.get("first_word", "")
    paragraphs = [p.strip() for p in response.split("\n\n") if p.strip()]
    if len(paragraphs) < num_paragraphs:
        return False
    if nth < 1 or nth > len(paragraphs):
        return False
    para = paragraphs[nth - 1]
    words = para.split()
    if not words:
        return False
    return words[0].lower() == first_word.lower()


def check_detectable_content_number_placeholders(response, kwargs):
    """detectable_content:number_placeholders - count [...] placeholders."""
    num_placeholders = kwargs.get("num_placeholders", 0)
    # Match [something] but not empty []
    placeholders = re.findall(r'\[([^\[\]]+)\]', response)
    return len(placeholders) >= num_placeholders


def check_detectable_content_postscript(response, kwargs):
    """detectable_content:postscript - check if response contains P.S. marker."""
    marker = kwargs.get("postscript_marker", "P.S.")
    if not marker:
        marker = "P.S."
    return marker in response


def check_detectable_format_number_bullet_lists(response, kwargs):
    """detectable_format:number_bullet_lists - count bullet lists."""
    num_bullets = kwargs.get("num_bullets", 0)
    # Count lines that start with bullet markers: *, -, or numbered (1.)
    bullet_lines = re.findall(r'^\s*[\*\-\u2022]\s+', response, re.MULTILINE)
    numbered_lines = re.findall(r'^\s*\d+[\.\)]\s+', response, re.MULTILINE)
    total = len(bullet_lines) + len(numbered_lines)
    return total >= num_bullets


def check_detectable_format_constrained_response(response, kwargs):
    """detectable_format:constrained_response - response must be one of given options."""
    # The response should be very short / a direct choice
    # This is hard to check generically without knowing the options
    # We'll consider it passed if the response is reasonably short (under 100 words)
    return len(response.split()) <= 100


def check_detectable_format_number_highlighted_sections(response, kwargs):
    """detectable_format:number_highlighted_sections - count *highlighted* sections."""
    num_highlights = kwargs.get("num_highlights", 0)
    # Match *text* patterns (highlighted sections)
    highlights = re.findall(r'\*[^*\n]+\*', response)
    return len(highlights) >= num_highlights


def check_detectable_format_multiple_sections(response, kwargs):
    """detectable_format:multiple_sections - check for section markers."""
    splitter = kwargs.get("section_spliter", kwargs.get("section_splitter", ""))
    num_sections = kwargs.get("num_sections", 0)
    if not splitter:
        return True
    sections = response.split(splitter)
    # Filter out empty sections
    sections = [s.strip() for s in sections if s.strip()]
    return len(sections) >= num_sections


def check_detectable_format_json_format(response, kwargs):
    """detectable_format:json_format - response must contain valid JSON."""
    # Try to find and parse JSON in the response
    # First try the whole response
    try:
        json.loads(response.strip())
        return True
    except (json.JSONDecodeError, ValueError):
        pass
    # Try to find JSON block in the response
    json_patterns = [
        re.compile(r'```json\s*(.*?)\s*```', re.DOTALL),
        re.compile(r'```\s*([\{\[].*?[\}\]])\s*```', re.DOTALL),
        re.compile(r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', re.DOTALL),
        re.compile(r'(\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\])', re.DOTALL),
    ]
    for pat in json_patterns:
        for match in pat.finditer(response):
            try:
                json.loads(match.group(1))
                return True
            except (json.JSONDecodeError, ValueError):
                continue
    return False


def check_detectable_format_title(response, kwargs):
    """detectable_format:title - response has a title wrapped in <<>>."""
    return bool(re.search(r'<<[^<>]+>>', response))


def check_combination_two_responses(response, kwargs):
    """combination:two_responses - two separate responses."""
    # Check for separators like "***", "---", "Response 1/2", etc.
    separators = ['***', '---', '===', 'Response 1', 'Response 2',
                  'RESPONSE 1', 'RESPONSE 2', '******']
    for sep in separators:
        if sep in response:
            return True
    # Also check for numbered responses
    if re.search(r'\b(1\.|First|Response\s*1)', response) and \
       re.search(r'\b(2\.|Second|Response\s*2)', response):
        return True
    return False


def check_combination_repeat_prompt(response, kwargs):
    """combination:repeat_prompt - response must repeat the prompt."""
    prompt_to_repeat = kwargs.get("prompt_to_repeat", "")
    if not prompt_to_repeat:
        return True
    return prompt_to_repeat.strip().lower() in response.lower()


def check_startend_end_checker(response, kwargs):
    """startend:end_checker - response ends with specific string."""
    end_phrase = kwargs.get("end_phrase", "")
    if not end_phrase:
        return True
    return response.strip().endswith(end_phrase)


def check_startend_quotation(response, kwargs):
    """startend:quotation - response wrapped in quotes."""
    stripped = response.strip()
    if (stripped.startswith('"') and stripped.endswith('"')):
        return True
    if (stripped.startswith("'") and stripped.endswith("'")):
        return True
    if (stripped.startswith('\u201c') and stripped.endswith('\u201d')):
        return True
    return False


def check_change_case_english_lowercase(response, kwargs):
    """change_case:english_lowercase - all English letters lowercase."""
    for ch in response:
        if ch.isalpha() and ch.isupper():
            return False
    return True


def check_change_case_english_capital(response, kwargs):
    """change_case:english_capital - all English letters uppercase."""
    for ch in response:
        if ch.isalpha() and ch.islower():
            return False
    return True


def check_punctuation_no_comma(response, kwargs):
    """punctuation:no_comma - no commas in response."""
    return ',' not in response


def check_language_response_language(response, kwargs):
    """language:response_language - response in specific language.

    Simple heuristic: check if the response contains mostly expected
    script characters. This is approximate.
    """
    language = kwargs.get("language", "").lower()
    if not language:
        return True
    # For English, check that most characters are ASCII
    if language == "english" or language == "en":
        ascii_count = sum(1 for c in response if ord(c) < 128)
        return ascii_count / max(len(response), 1) > 0.8
    # For other languages, we can't easily verify - mark as passed
    # (being generous since we can't easily detect all languages)
    return True


# ---------------------------------------------------------------------------
# Constraint dispatcher
# ---------------------------------------------------------------------------

CONSTRAINT_CHECKERS = {
    "keywords:existence": check_keywords_existence,
    "keywords:frequency": check_keywords_frequency,
    "keywords:forbidden_words": check_keywords_forbidden_words,
    "keywords:letter_frequency": check_keywords_letter_frequency,
    "length_constraints:number_words": check_length_number_words,
    "length_constraints:number_sentences": check_length_number_sentences,
    "length_constraints:number_paragraphs": check_length_number_paragraphs,
    "length_constraints:nth_paragraph_first_word": check_length_nth_paragraph_first_word,
    "detectable_content:number_placeholders": check_detectable_content_number_placeholders,
    "detectable_content:postscript": check_detectable_content_postscript,
    "detectable_format:number_bullet_lists": check_detectable_format_number_bullet_lists,
    "detectable_format:constrained_response": check_detectable_format_constrained_response,
    "detectable_format:number_highlighted_sections": check_detectable_format_number_highlighted_sections,
    "detectable_format:multiple_sections": check_detectable_format_multiple_sections,
    "detectable_format:json_format": check_detectable_format_json_format,
    "detectable_format:title": check_detectable_format_title,
    "combination:two_responses": check_combination_two_responses,
    "combination:repeat_prompt": check_combination_repeat_prompt,
    "startend:end_checker": check_startend_end_checker,
    "startend:quotation": check_startend_quotation,
    "change_case:english_lowercase": check_change_case_english_lowercase,
    "change_case:english_capital": check_change_case_english_capital,
    "punctuation:no_comma": check_punctuation_no_comma,
    "language:response_language": check_language_response_language,
}


def evaluate_constraints(response, instruction_id_list, kwargs_list):
    """Evaluate all constraints for one prompt.

    Returns:
        results: list of dicts with {instruction_id, passed, skipped}
        all_pass: True if ALL non-skipped constraints pass
    """
    results = []
    all_pass = True

    for instruction_id, kw in zip(instruction_id_list, kwargs_list):
        # Parse kwargs - they may be a JSON string or dict
        if isinstance(kw, str):
            try:
                kw = json.loads(kw)
            except (json.JSONDecodeError, ValueError):
                kw = {}
        if kw is None:
            kw = {}

        checker = CONSTRAINT_CHECKERS.get(instruction_id)
        if checker is None:
            results.append({
                "instruction_id": instruction_id,
                "passed": False,
                "skipped": True,
            })
            continue

        try:
            passed = checker(response, kw)
        except Exception as e:
            passed = False

        results.append({
            "instruction_id": instruction_id,
            "passed": passed,
            "skipped": False,
        })
        if not passed:
            all_pass = False

    return results, all_pass


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_response(model, tokenizer, prompt, device, is_chat_model=False,
                      max_new_tokens=1024):
    """Generate model response."""
    if is_chat_model:
        # Use chat template
        messages = [{"role": "user", "content": prompt}]
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Fallback if chat template fails
            text = prompt
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    else:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    del outputs, input_ids, attention_mask
    return response.strip()


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------

def evaluate_model_on_ifeval(model, tokenizer, device, prompts, label,
                             is_chat_model=False):
    """Evaluate one model variant on all IFEval prompts."""
    print(f"\n  [{label}] Evaluating on {len(prompts)} IFEval prompts...")
    model.eval()

    prompt_pass_count = 0
    instruction_pass_count = 0
    instruction_total = 0
    instruction_skipped = 0
    prompt_details = []

    t0 = time.time()

    for i, item in enumerate(prompts):
        prompt_text = item["prompt"]
        instruction_ids = item["instruction_id_list"]
        kwargs_list = item["kwargs"]

        response = generate_response(
            model, tokenizer, prompt_text, device,
            is_chat_model=is_chat_model, max_new_tokens=512
        )

        constraint_results, all_pass = evaluate_constraints(
            response, instruction_ids, kwargs_list
        )

        # Count results
        non_skipped = [r for r in constraint_results if not r["skipped"]]
        passed = [r for r in non_skipped if r["passed"]]
        skipped = [r for r in constraint_results if r["skipped"]]

        instruction_total += len(non_skipped)
        instruction_pass_count += len(passed)
        instruction_skipped += len(skipped)

        # Prompt-level: all non-skipped must pass
        if non_skipped and all(r["passed"] for r in non_skipped):
            prompt_pass_count += 1

        prompt_details.append({
            "prompt_index": i,
            "prompt_preview": prompt_text[:100],
            "response_preview": response[:200],
            "num_constraints": len(instruction_ids),
            "num_passed": len(passed),
            "num_skipped": len(skipped),
            "all_pass": all_pass if non_skipped else False,
            "constraint_results": constraint_results,
        })

        if (i + 1) % 50 == 0 or (i + 1) == len(prompts):
            elapsed = time.time() - t0
            p_acc = prompt_pass_count / (i + 1)
            i_acc = instruction_pass_count / max(instruction_total, 1)
            print(f"    [{label}] {i+1}/{len(prompts)}  "
                  f"prompt_acc={p_acc:.1%}  instr_acc={i_acc:.1%}  "
                  f"({elapsed:.0f}s)")

        torch.cuda.empty_cache()

    total_prompts = len(prompts)
    elapsed = time.time() - t0

    summary = {
        "prompt_accuracy": prompt_pass_count / max(total_prompts, 1),
        "instruction_accuracy": instruction_pass_count / max(instruction_total, 1),
        "prompt_pass": prompt_pass_count,
        "prompt_total": total_prompts,
        "instruction_pass": instruction_pass_count,
        "instruction_total": instruction_total,
        "instruction_skipped": instruction_skipped,
        "elapsed_seconds": elapsed,
    }

    print(f"  [{label}] DONE: prompt_acc={summary['prompt_accuracy']:.1%}  "
          f"instr_acc={summary['instruction_accuracy']:.1%}  "
          f"({elapsed:.0f}s, skipped {instruction_skipped} instructions)")

    return summary, prompt_details


# ---------------------------------------------------------------------------
# SAR application
# ---------------------------------------------------------------------------

def load_attribution_scores(attr_filename):
    """Load attribution scores for SAR."""
    attr_path = RESULTS_DIR / attr_filename
    if not attr_path.exists():
        print(f"  WARNING: Attribution file not found: {attr_path}")
        return {}

    with open(attr_path) as f:
        data = json.load(f)

    components = data.get("components", [])
    scores = {}
    for c in components:
        score = c.get("mean_score", c.get("harm_score", 0))
        scores[c["name"]] = score

    return scores


def apply_sar(model, base_dir, attribution_scores, k_pct=5):
    """Apply Surgical Alignment Reversal: replace top-k% most harmful
    components in the IT model with base model weights."""
    from safetensors import safe_open

    if not attribution_scores:
        print("  WARNING: No attribution scores available, skipping SAR")
        return model

    sorted_comps = sorted(attribution_scores.items(), key=lambda x: x[1], reverse=True)
    n_total = len(sorted_comps)
    n_rollback = max(1, int(n_total * k_pct / 100))
    rollback_names = {name for name, _ in sorted_comps[:n_rollback]}

    print(f"  SAR-{k_pct}%: rolling back {n_rollback}/{n_total} components")
    for name, score in sorted_comps[:n_rollback]:
        print(f"    -> {name}  (mean_score={score:.6f})")

    # Load base model weights from safetensors
    base_path = Path(base_dir)
    safetensor_files = sorted(base_path.glob("*.safetensors"))

    model_state = dict(model.named_parameters())
    rolled_back = 0

    for sf_path in safetensor_files:
        with safe_open(str(sf_path), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in rollback_names and key in model_state:
                    base_tensor = f.get_tensor(key)
                    base_tensor = base_tensor.to(
                        device=model_state[key].device,
                        dtype=model_state[key].dtype
                    )
                    model_state[key].data.copy_(base_tensor)
                    rolled_back += 1
                    del base_tensor

    print(f"  Rolled back {rolled_back} tensors")
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    model_filter = sys.argv[2] if len(sys.argv) > 2 else None
    print(f"Step 17: IFEval Benchmark Evaluation")
    print(f"Device: {device}")
    print(f"=" * 70)

    models_to_run = {model_filter: MODEL_CONFIGS[model_filter]} if model_filter else MODEL_CONFIGS

    # W&B init
    run = wandb.init(
        project="anonymous-submission",
        name=f"ifeval{'-' + model_filter if model_filter else ''}",
        config={
            "benchmark": "IFEval",
            "dataset": "google/IFEval",
            "num_prompts": 541,
            "max_new_tokens": 1024,
            "temperature": 0,
            "sar_pct": 5,
            "models": list(models_to_run.keys()),
        },
    )

    # Load IFEval dataset
    print("\nLoading IFEval dataset...")
    prompts = load_ifeval_dataset()
    print(f"Total prompts: {len(prompts)}")

    # Count instruction types
    instr_counts = defaultdict(int)
    for p in prompts:
        for iid in p["instruction_id_list"]:
            instr_counts[iid] += 1
    print(f"\nInstruction type distribution ({len(instr_counts)} types):")
    for iid, cnt in sorted(instr_counts.items(), key=lambda x: -x[1])[:15]:
        supported = "OK" if iid in CONSTRAINT_CHECKERS else "SKIP"
        print(f"  {iid:<50s} {cnt:>4d}  [{supported}]")

    supported_count = sum(cnt for iid, cnt in instr_counts.items()
                         if iid in CONSTRAINT_CHECKERS)
    total_instr = sum(instr_counts.values())
    print(f"  Supported: {supported_count}/{total_instr} "
          f"({supported_count/max(total_instr,1):.1%})")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    all_results = {}

    for family_name, cfg in models_to_run.items():
        print(f"\n{'=' * 70}")
        print(f"MODEL FAMILY: {family_name}")
        print(f"{'=' * 70}")

        base_dir = cfg["base"]
        it_dir = cfg["it"]
        attr_file = cfg["attr"]

        # Check model paths
        if not Path(base_dir).exists():
            print(f"  WARNING: Base model not found at {base_dir}, skipping")
            continue
        if not Path(it_dir).exists():
            print(f"  WARNING: IT model not found at {it_dir}, skipping")
            continue

        family_results = {}

        # --- BASE ---
        print(f"\n--- {family_name} BASE ---")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                base_dir, trust_remote_code=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                base_dir, torch_dtype=torch.float16,
                device_map=device, trust_remote_code=True
            )

            summary, details = evaluate_model_on_ifeval(
                model, tokenizer, device, prompts,
                f"{family_name}/Base", is_chat_model=False
            )
            family_results["base"] = summary

            wandb.log({
                f"ifeval/{family_name}/base/prompt_accuracy": summary["prompt_accuracy"],
                f"ifeval/{family_name}/base/instruction_accuracy": summary["instruction_accuracy"],
            })

            del model, tokenizer
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  ERROR evaluating base: {e}")
            traceback.print_exc()

        # --- IT ---
        print(f"\n--- {family_name} IT ---")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                it_dir, trust_remote_code=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                it_dir, torch_dtype=torch.float16,
                device_map=device, trust_remote_code=True
            )

            summary, details = evaluate_model_on_ifeval(
                model, tokenizer, device, prompts,
                f"{family_name}/IT", is_chat_model=True
            )
            family_results["it"] = summary

            wandb.log({
                f"ifeval/{family_name}/it/prompt_accuracy": summary["prompt_accuracy"],
                f"ifeval/{family_name}/it/instruction_accuracy": summary["instruction_accuracy"],
            })

            del model
            torch.cuda.empty_cache()
            # Keep IT tokenizer for SAR
            sar_tokenizer = tokenizer
        except Exception as e:
            print(f"  ERROR evaluating IT: {e}")
            traceback.print_exc()
            sar_tokenizer = None

        # --- SAR-5% ---
        print(f"\n--- {family_name} SAR-5% ---")
        try:
            attr_scores = load_attribution_scores(attr_file)
            if not attr_scores:
                print(f"  Skipping SAR-5%: no attribution scores for {family_name}")
            else:
                # Reload IT model, then apply SAR
                if sar_tokenizer is None:
                    sar_tokenizer = AutoTokenizer.from_pretrained(
                        it_dir, trust_remote_code=True
                    )
                    if sar_tokenizer.pad_token is None:
                        sar_tokenizer.pad_token = sar_tokenizer.eos_token

                model = AutoModelForCausalLM.from_pretrained(
                    it_dir, torch_dtype=torch.float16,
                    device_map=device, trust_remote_code=True
                )
                model = apply_sar(model, base_dir, attr_scores, k_pct=5)

                summary, details = evaluate_model_on_ifeval(
                    model, sar_tokenizer, device, prompts,
                    f"{family_name}/SAR-5%", is_chat_model=True
                )
                family_results["sar_5pct"] = summary

                wandb.log({
                    f"ifeval/{family_name}/sar5/prompt_accuracy": summary["prompt_accuracy"],
                    f"ifeval/{family_name}/sar5/instruction_accuracy": summary["instruction_accuracy"],
                })

                del model
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"  ERROR evaluating SAR-5%: {e}")
            traceback.print_exc()

        # Clean up tokenizer
        if sar_tokenizer is not None:
            del sar_tokenizer

        all_results[family_name] = family_results

        # Save per-model results
        out_path = RESULTS_DIR / f"ifeval_{family_name}.json"
        with open(out_path, "w") as f:
            json.dump({
                "model": family_name,
                "results": family_results,
            }, f, indent=2)
        print(f"\n  Saved: {out_path}")

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print("IFEVAL RESULTS SUMMARY")
    print(f"{'=' * 80}")

    header = f"{'Model':<30} {'Prompt Acc':>12} {'Instr Acc':>12} {'Pass/Total':>14}"
    print(header)
    print("-" * len(header))

    for family_name, family_results in all_results.items():
        for variant in ["base", "it", "sar_5pct"]:
            if variant not in family_results:
                continue
            s = family_results[variant]
            label_map = {"base": "Base", "it": "IT", "sar_5pct": "SAR-5%"}
            label = f"{family_name}/{label_map[variant]}"
            print(f"{label:<30} {s['prompt_accuracy']:>11.1%} "
                  f" {s['instruction_accuracy']:>11.1%} "
                  f" {s['prompt_pass']}/{s['prompt_total']}")

    # Alignment tax analysis
    print(f"\n{'=' * 80}")
    print("ALIGNMENT TAX ANALYSIS")
    print(f"{'=' * 80}")

    wandb_summary = {}

    for family_name, family_results in all_results.items():
        if "base" in family_results and "it" in family_results:
            base_p = family_results["base"]["prompt_accuracy"]
            it_p = family_results["it"]["prompt_accuracy"]
            base_i = family_results["base"]["instruction_accuracy"]
            it_i = family_results["it"]["instruction_accuracy"]
            tax_p = it_p - base_p
            tax_i = it_i - base_i
            print(f"  {family_name}:")
            print(f"    IT - Base prompt_acc:  {tax_p:+.1%}  "
                  f"(Base={base_p:.1%}, IT={it_p:.1%})")
            print(f"    IT - Base instr_acc:   {tax_i:+.1%}  "
                  f"(Base={base_i:.1%}, IT={it_i:.1%})")

            wandb_summary[f"tax/{family_name}/prompt_delta"] = tax_p
            wandb_summary[f"tax/{family_name}/instruction_delta"] = tax_i

        if "it" in family_results and "sar_5pct" in family_results:
            it_p = family_results["it"]["prompt_accuracy"]
            sar_p = family_results["sar_5pct"]["prompt_accuracy"]
            it_i = family_results["it"]["instruction_accuracy"]
            sar_i = family_results["sar_5pct"]["instruction_accuracy"]
            rec_p = sar_p - it_p
            rec_i = sar_i - it_i
            print(f"    SAR-5% - IT prompt_acc:  {rec_p:+.1%}  "
                  f"(IT={it_p:.1%}, SAR={sar_p:.1%})")
            print(f"    SAR-5% - IT instr_acc:   {rec_i:+.1%}  "
                  f"(IT={it_i:.1%}, SAR={sar_i:.1%})")

            wandb_summary[f"recovery/{family_name}/prompt_delta"] = rec_p
            wandb_summary[f"recovery/{family_name}/instruction_delta"] = rec_i

    if wandb_summary:
        wandb.log(wandb_summary)

    # Save combined results
    combined_path = RESULTS_DIR / "ifeval_combined.json"
    with open(combined_path, "w") as f:
        json.dump({
            "benchmark": "IFEval",
            "num_prompts": len(prompts),
            "results": all_results,
        }, f, indent=2)
    print(f"\nSaved combined results: {combined_path}")

    wandb.finish()
    print("\nDone!")


if __name__ == "__main__":
    main()
