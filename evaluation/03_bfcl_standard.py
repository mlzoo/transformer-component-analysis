"""
Standard BFCL Evaluation

Evaluates Base, IT, and SAR-5% models on the Berkeley Function Calling
Leaderboard dataset (gorilla-llm/Berkeley-Function-Calling-Leaderboard).

Categories evaluated (max 100 per category, 318 total):
  - simple:    Single function call with correct args
  - multiple:  Select correct function from several options
  - parallel:  Call multiple functions simultaneously
  - relevance: Detect when no suitable function is available (18 examples)

Metric: AST accuracy (correct function name AND correct arguments).

Usage:
    python 03_bfcl_standard.py cuda:0
"""

import sys
import json
import re
import ast
import torch
import time
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(exist_ok=True)
BFCL_SNAPSHOT = Path.home() / ".cache/huggingface/hub/datasets--gorilla-llm--Berkeley-Function-Calling-Leaderboard/snapshots"

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
        "attr": "attribution_yi-1.5-9b.json",
    },
}

# Maximum examples per category (set None for all)
MAX_PER_CATEGORY = {
    "simple": 100,
    "multiple": 100,
    "parallel": 100,
    "relevance": None,  # only 18 examples, use all
}

# ---------------------------------------------------------------------------
# BFCL data loading
# ---------------------------------------------------------------------------

def find_snapshot_dir():
    """Find the latest snapshot directory for the BFCL dataset."""
    if not BFCL_SNAPSHOT.exists():
        return None
    snapshots = list(BFCL_SNAPSHOT.iterdir())
    if not snapshots:
        return None
    # Return the first (usually only) snapshot
    return snapshots[0]


def load_bfcl_category(snap_dir, category):
    """Load a BFCL category (questions + ground truth).

    Returns list of dicts with keys: id, question_text, functions, ground_truth
    """
    # Map our category names to BFCL filenames
    file_map = {
        "simple": "BFCL_v3_simple.json",
        "multiple": "BFCL_v3_multiple.json",
        "parallel": "BFCL_v3_parallel.json",
        "relevance": "BFCL_v3_live_relevance.json",
    }

    q_path = snap_dir / file_map[category]
    if not q_path.exists():
        print(f"  WARNING: {q_path} not found")
        return []

    # Load questions (JSONL format)
    questions = []
    with open(q_path) as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))

    # Load ground truth (if available)
    gt_path = snap_dir / "possible_answer" / file_map[category]
    gt_by_id = {}
    if gt_path.exists():
        with open(gt_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    gt_by_id[obj["id"]] = obj.get("ground_truth", [])

    # Merge into unified format
    examples = []
    for q in questions:
        qid = q["id"]
        # Extract question text from nested structure: [[{role, content}]]
        question_text = ""
        if q.get("question"):
            msgs = q["question"]
            if isinstance(msgs, list) and len(msgs) > 0:
                turn = msgs[0]  # first turn
                if isinstance(turn, list):
                    for msg in turn:
                        if isinstance(msg, dict) and msg.get("role") == "user":
                            question_text = msg["content"]
                elif isinstance(turn, dict):
                    question_text = turn.get("content", "")

        # Extract function definitions
        functions = q.get("function", [])

        examples.append({
            "id": qid,
            "question_text": question_text,
            "functions": functions,
            "ground_truth": gt_by_id.get(qid, []),
            "category": category,
        })

    return examples


def load_all_bfcl_data():
    """Load all BFCL categories. Falls back to synthetic data on failure."""
    snap_dir = find_snapshot_dir()
    if snap_dir is None:
        print("WARNING: BFCL dataset not found in cache. Using synthetic fallback.")
        return None, True

    data = {}
    for cat in ["simple", "multiple", "parallel", "relevance"]:
        examples = load_bfcl_category(snap_dir, cat)
        limit = MAX_PER_CATEGORY.get(cat)
        if limit and len(examples) > limit:
            examples = examples[:limit]
        data[cat] = examples
        print(f"  Loaded {len(examples)} {cat} examples")

    total = sum(len(v) for v in data.values())
    if total == 0:
        print("WARNING: No BFCL examples loaded. Using synthetic fallback.")
        return None, True

    return data, False


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def format_functions_for_prompt(functions):
    """Format BFCL function definitions into a prompt-friendly string."""
    lines = []
    for func in functions:
        name = func.get("name", "unknown")
        desc = func.get("description", "")
        params = func.get("parameters", {})
        props = params.get("properties", {})
        required = params.get("required", [])

        param_strs = []
        for pname, pinfo in props.items():
            ptype = pinfo.get("type", "any")
            pdesc = pinfo.get("description", "")
            req = "(required)" if pname in required else "(optional)"
            param_strs.append(f"    - {pname} ({ptype}, {req}): {pdesc}")

        lines.append(f"Function: {name}")
        lines.append(f"  Description: {desc}")
        if param_strs:
            lines.append("  Parameters:")
            lines.extend(param_strs)
        lines.append("")

    return "\n".join(lines)


def build_prompt(example, is_relevance=False):
    """Build the evaluation prompt for one example."""
    funcs_text = format_functions_for_prompt(example["functions"])
    question = example["question_text"]

    if is_relevance:
        prompt = f"""You have access to the following functions:

{funcs_text}
User request: {question}

If a suitable function exists, call it in the format: function_name(arg1=value1, arg2=value2)
If NO function is suitable for the request, respond with exactly: NO_FUNCTION_AVAILABLE

Response:
"""
    else:
        prompt = f"""You have access to the following functions:

{funcs_text}
User request: {question}

Call the appropriate function(s) with correct arguments. Output ONLY the function call(s) in the format:
function_name(arg1=value1, arg2=value2)

If multiple functions need to be called, put each on a separate line.

Function call:
"""
    return prompt


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_function_calls(text):
    """Parse function calls from model output.

    Returns list of (func_name, {arg: value}) tuples.
    """
    calls = []
    # Try to find function_name(args) patterns
    for match in re.finditer(r'([\w.]+)\s*\((.*?)\)', text, re.DOTALL):
        func_name = match.group(1)
        args_str = match.group(2).strip()

        # Skip common false positives
        if func_name.lower() in ('if', 'for', 'while', 'print', 'len', 'str',
                                   'int', 'float', 'list', 'dict', 'type',
                                   'the', 'no', 'none', 'not', 'is', 'function',
                                   'format', 'range', 'return', 'def'):
            continue

        args = parse_args_string(args_str)
        calls.append((func_name, args))

    # Also try to parse JSON-style function calls: {"name": "...", "arguments": {...}}
    for match in re.finditer(r'\{[^{}]*"name"\s*:\s*"([^"]+)"[^{}]*"arguments"\s*:\s*(\{[^{}]*\})', text):
        func_name = match.group(1)
        try:
            args = json.loads(match.group(2))
            calls.append((func_name, args))
        except json.JSONDecodeError:
            pass

    return calls


def parse_args_string(args_str):
    """Parse argument string like 'key1=val1, key2="val2"' into dict."""
    args = {}
    if not args_str.strip():
        return args

    # Try ast.literal_eval on the whole thing as a dict
    try:
        maybe_dict = ast.literal_eval("{" + args_str + "}")
        if isinstance(maybe_dict, dict):
            return maybe_dict
    except (ValueError, SyntaxError):
        pass

    # Try keyword argument parsing
    # Match patterns: key=value, key="value", key='value', key=123
    for m in re.finditer(
        r'(\w+)\s*=\s*(?:'
        r'"((?:[^"\\]|\\.)*)"|'    # double-quoted
        r"'((?:[^'\\]|\\.)*)'|"    # single-quoted
        r'(\[.*?\])|'              # list
        r'(\{.*?\})|'              # dict
        r'(True|False|None)|'      # Python literals
        r'(-?\d+\.?\d*)'           # numbers
        r')',
        args_str
    ):
        key = m.group(1)
        # Find the matched value group
        value = (m.group(2) if m.group(2) is not None else
                 m.group(3) if m.group(3) is not None else
                 m.group(4) if m.group(4) is not None else
                 m.group(5) if m.group(5) is not None else
                 m.group(6) if m.group(6) is not None else
                 m.group(7) if m.group(7) is not None else
                 None)

        if value is None:
            continue

        # Try to convert to native types
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass

        args[key] = value

    return args


def normalize_value(v):
    """Normalize a value for comparison."""
    if isinstance(v, str):
        return v.strip().lower()
    if isinstance(v, (int, float)):
        return v
    if isinstance(v, bool):
        return v
    return str(v).strip().lower()


# ---------------------------------------------------------------------------
# AST evaluation
# ---------------------------------------------------------------------------

def check_ast_match(predicted_calls, ground_truth, category):
    """Check if predicted function calls match ground truth.

    ground_truth format: [{func_name: {arg: [acceptable_values]}}]

    Returns dict with func_correct, args_correct, ast_correct booleans.
    """
    if category == "relevance":
        # For relevance, ground truth is empty; model should NOT call a function
        no_call = len(predicted_calls) == 0
        return {
            "func_correct": no_call,
            "args_correct": no_call,
            "ast_correct": no_call,
        }

    if not ground_truth:
        # No ground truth available; skip
        return {"func_correct": False, "args_correct": False, "ast_correct": False}

    # For simple/multiple: expect one function call matching one ground truth entry
    # For parallel: expect multiple calls matching multiple ground truth entries
    gt_calls = []  # list of (func_name, {arg: [acceptable_values]})
    for gt_entry in ground_truth:
        if isinstance(gt_entry, dict):
            for fname, fargs in gt_entry.items():
                gt_calls.append((fname, fargs))

    if not gt_calls:
        return {"func_correct": False, "args_correct": False, "ast_correct": False}

    if category in ("simple", "multiple"):
        # Expect exactly one call
        gt_name, gt_args = gt_calls[0]

        # Check if any predicted call matches the function name
        func_correct = False
        args_correct = False
        for pred_name, pred_args in predicted_calls:
            if pred_name == gt_name:
                func_correct = True
                # Check arguments
                args_ok = check_args_match(pred_args, gt_args)
                if args_ok:
                    args_correct = True
                break  # take the first match

        return {
            "func_correct": func_correct,
            "args_correct": args_correct,
            "ast_correct": func_correct and args_correct,
        }

    elif category == "parallel":
        # All ground truth calls must be matched
        matched = 0
        func_matched = 0
        used_pred = set()

        for gt_name, gt_args in gt_calls:
            for i, (pred_name, pred_args) in enumerate(predicted_calls):
                if i in used_pred:
                    continue
                if pred_name == gt_name:
                    func_matched += 1
                    if check_args_match(pred_args, gt_args):
                        matched += 1
                    used_pred.add(i)
                    break

        all_func = func_matched == len(gt_calls)
        all_ast = matched == len(gt_calls)
        return {
            "func_correct": all_func,
            "args_correct": all_ast,
            "ast_correct": all_ast,
        }

    return {"func_correct": False, "args_correct": False, "ast_correct": False}


def check_args_match(pred_args, gt_args):
    """Check if predicted args match ground truth args.

    gt_args format: {arg_name: [acceptable_value1, acceptable_value2, ...]}
    Empty string in acceptable values means the arg is optional.
    """
    for arg_name, acceptable in gt_args.items():
        if not isinstance(acceptable, list):
            acceptable = [acceptable]

        # If "" is in acceptable values, the arg is optional
        is_optional = "" in acceptable
        acceptable_real = [v for v in acceptable if v != ""]

        if arg_name not in pred_args:
            if is_optional:
                continue  # OK, it's optional
            else:
                return False  # required arg missing

        pred_val = pred_args[arg_name]
        pred_norm = normalize_value(pred_val)

        # Check if predicted value matches any acceptable value
        matched = False
        for acc_val in acceptable_real:
            acc_norm = normalize_value(acc_val)
            if pred_norm == acc_norm:
                matched = True
                break
            # Numeric comparison
            try:
                if float(pred_val) == float(acc_val):
                    matched = True
                    break
            except (ValueError, TypeError):
                pass

        if not matched and acceptable_real:
            return False

    return True


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------

def generate_response(model, tokenizer, prompt, device, max_new_tokens=256):
    """Generate model response for a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = inputs["input_ids"].to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            temperature=None,
            top_p=None,
        )

    response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    del outputs, input_ids
    return response.strip()


def evaluate_category(model, tokenizer, device, examples, category):
    """Evaluate a model on one BFCL category."""
    results = {
        "func_correct": 0,
        "args_correct": 0,
        "ast_correct": 0,
        "total": len(examples),
        "details": [],
    }

    is_relevance = (category == "relevance")

    for i, ex in enumerate(examples):
        prompt = build_prompt(ex, is_relevance=is_relevance)
        response = generate_response(model, tokenizer, prompt, device)

        if is_relevance:
            # Check if model correctly refrains from calling a function
            calls = parse_function_calls(response)
            no_call = (len(calls) == 0 or
                       "NO_FUNCTION" in response.upper() or
                       "no suitable" in response.lower() or
                       "cannot" in response.lower()[:50] or
                       "none of" in response.lower()[:80])
            r = {
                "func_correct": no_call,
                "args_correct": no_call,
                "ast_correct": no_call,
            }
        else:
            calls = parse_function_calls(response)
            r = check_ast_match(calls, ex["ground_truth"], category)

        results["func_correct"] += int(r["func_correct"])
        results["args_correct"] += int(r["args_correct"])
        results["ast_correct"] += int(r["ast_correct"])

        results["details"].append({
            "id": ex["id"],
            "response_preview": response[:120],
            "func_ok": r["func_correct"],
            "args_ok": r["args_correct"],
            "ast_ok": r["ast_correct"],
        })

        if (i + 1) % 25 == 0:
            acc = results["ast_correct"] / (i + 1)
            print(f"    [{category}] {i+1}/{len(examples)}  AST={acc:.1%}")

        torch.cuda.empty_cache()

    return results


def evaluate_model_variant(model, tokenizer, device, bfcl_data, label):
    """Evaluate one model variant across all BFCL categories."""
    print(f"\n  [{label}] Starting BFCL evaluation...")
    model.eval()

    cat_results = {}
    for cat_name, examples in bfcl_data.items():
        if not examples:
            continue
        print(f"    [{label}] Evaluating {cat_name} ({len(examples)} examples)...")
        t0 = time.time()
        cat_results[cat_name] = evaluate_category(model, tokenizer, device, examples, cat_name)
        elapsed = time.time() - t0
        n = cat_results[cat_name]["total"]
        ast_acc = cat_results[cat_name]["ast_correct"] / max(n, 1)
        print(f"    [{label}] {cat_name}: AST={ast_acc:.1%}  ({elapsed:.1f}s)")

    # Compute overall metrics
    total_ast = sum(r["ast_correct"] for r in cat_results.values())
    total_func = sum(r["func_correct"] for r in cat_results.values())
    total_n = sum(r["total"] for r in cat_results.values())

    summary = {}
    for cat_name, r in cat_results.items():
        n = max(r["total"], 1)
        summary[cat_name] = {
            "func_accuracy": r["func_correct"] / n,
            "args_accuracy": r["args_correct"] / n,
            "ast_accuracy": r["ast_correct"] / n,
            "n": r["total"],
        }

    summary["overall"] = {
        "func_accuracy": total_func / max(total_n, 1),
        "ast_accuracy": total_ast / max(total_n, 1),
        "n": total_n,
    }

    print(f"  [{label}] Overall AST accuracy: {summary['overall']['ast_accuracy']:.1%} "
          f"(n={total_n})")

    return summary, cat_results


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
        score = c.get("harm_score", c.get("mean_score", 0))
        scores[c["name"]] = score

    return scores


def apply_sar(model, base_dir, attribution_scores, k_pct=5):
    """Apply Surgical Alignment Reversal: replace top-k% harmful components
    in the IT model with base model weights."""
    from safetensors import safe_open

    if not attribution_scores:
        print("  WARNING: No attribution scores available, skipping SAR")
        return model

    sorted_comps = sorted(attribution_scores.items(), key=lambda x: x[1], reverse=True)
    n_total = len(sorted_comps)
    n_rollback = max(1, int(n_total * k_pct / 100))
    rollback_names = {name for name, _ in sorted_comps[:n_rollback]}

    print(f"  SAR-{k_pct}%: rolling back {n_rollback}/{n_total} components")

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
# Synthetic BFCL fallback (in case real dataset fails)
# ---------------------------------------------------------------------------

def build_synthetic_bfcl():
    """Build a synthetic BFCL-style dataset as fallback."""
    print("  Building synthetic BFCL-style evaluation data...")

    simple_examples = [
        {
            "id": "synth_simple_0",
            "question_text": "Find the area of a triangle with base 10 and height 5.",
            "functions": [{"name": "calculate_triangle_area", "description": "Calculate the area of a triangle.",
                           "parameters": {"type": "dict", "properties": {
                               "base": {"type": "integer", "description": "The base."},
                               "height": {"type": "integer", "description": "The height."}}, "required": ["base", "height"]}}],
            "ground_truth": [{"calculate_triangle_area": {"base": [10], "height": [5]}}],
            "category": "simple",
        },
        {
            "id": "synth_simple_1",
            "question_text": "Get the current weather in San Francisco.",
            "functions": [{"name": "get_weather", "description": "Get current weather for a city.",
                           "parameters": {"type": "dict", "properties": {
                               "city": {"type": "string", "description": "City name."}}, "required": ["city"]}}],
            "ground_truth": [{"get_weather": {"city": ["San Francisco"]}}],
            "category": "simple",
        },
        {
            "id": "synth_simple_2",
            "question_text": "Search the web for 'machine learning tutorials'.",
            "functions": [{"name": "web_search", "description": "Search the web.",
                           "parameters": {"type": "dict", "properties": {
                               "query": {"type": "string", "description": "Search query."}}, "required": ["query"]}}],
            "ground_truth": [{"web_search": {"query": ["machine learning tutorials"]}}],
            "category": "simple",
        },
    ]

    # Repeat pattern to get ~25 examples per category
    data = {
        "simple": simple_examples * 8,  # 24 examples
        "multiple": simple_examples * 5,  # 15 examples
        "parallel": [],
        "relevance": [
            {
                "id": f"synth_rel_{i}",
                "question_text": q,
                "functions": [{"name": "get_weather", "description": "Get weather.", "parameters": {"type": "dict", "properties": {"city": {"type": "string"}}, "required": ["city"]}}],
                "ground_truth": [],
                "category": "relevance",
            }
            for i, q in enumerate([
                "Play some jazz music.",
                "Book a flight to London.",
                "Order pizza from nearby.",
                "Turn off the lights.",
                "Take a screenshot.",
            ])
        ],
    }
    return data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    print(f"Standard BFCL Evaluation")
    print(f"Device: {device}")
    print(f"=" * 70)

    # W&B setup (set WANDB_MODE=disabled to skip logging)
    import os
    try:
        import wandb
        HAS_WANDB = True
    except ImportError:
        wandb = None
        HAS_WANDB = False
    if HAS_WANDB:
        wandb_mode = os.environ.get("WANDB_MODE", "disabled")
        if wandb_mode != "disabled":
            wandb.login()
        run = wandb.init(project="anonymous-submission", name="bfcl-standard",
                         mode=wandb_mode)

    # Load BFCL data
    print("\nLoading BFCL dataset...")
    bfcl_data, is_synthetic = load_all_bfcl_data()
    if bfcl_data is None:
        print("\nERROR: BFCL dataset not available. Please download it first:")
        print("  python -c \"from datasets import load_dataset; load_dataset('gorilla-llm/Berkeley-Function-Calling-Leaderboard')\"")
        print("\nFalling back to synthetic data for demonstration only.")
        print("NOTE: Results from synthetic data are NOT comparable to paper numbers.\n")
        bfcl_data = build_synthetic_bfcl()
        is_synthetic = True

    total_examples = sum(len(v) for v in bfcl_data.values())
    print(f"Total examples: {total_examples} ({'SYNTHETIC - not paper results' if is_synthetic else 'real BFCL'})")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    all_results = {}

    for family_name, cfg in MODEL_CONFIGS.items():
        print(f"\n{'=' * 70}")
        print(f"MODEL FAMILY: {family_name}")
        print(f"{'=' * 70}")

        base_dir = cfg["base"]
        it_dir = cfg["it"]
        attr_file = cfg["attr"]

        # Check model paths exist
        if not Path(base_dir).exists():
            print(f"  WARNING: Base model not found at {base_dir}, skipping {family_name}")
            continue

        family_results = {}

        # --- BASE ---
        print(f"\n--- {family_name} BASE ---")
        try:
            base_tokenizer = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=True)
            if base_tokenizer.pad_token is None:
                base_tokenizer.pad_token = base_tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                base_dir, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
            )
            summary, details = evaluate_model_variant(model, base_tokenizer, device, bfcl_data, f"{family_name}/Base")
            family_results["base"] = summary

            # Log to W&B
            if HAS_WANDB:
                for cat, metrics in summary.items():
                    if isinstance(metrics, dict) and "ast_accuracy" in metrics:
                        wandb.log({f"bfcl/{family_name}/base/{cat}_ast": metrics["ast_accuracy"]})

            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  ERROR evaluating base: {e}")
            traceback.print_exc()

        # --- IT ---
        print(f"\n--- {family_name} IT ---")
        try:
            it_tokenizer = AutoTokenizer.from_pretrained(it_dir, trust_remote_code=True)
            if it_tokenizer.pad_token is None:
                it_tokenizer.pad_token = it_tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                it_dir, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
            )
            summary, details = evaluate_model_variant(model, it_tokenizer, device, bfcl_data, f"{family_name}/IT")
            family_results["it"] = summary

            if HAS_WANDB:
                for cat, metrics in summary.items():
                    if isinstance(metrics, dict) and "ast_accuracy" in metrics:
                        wandb.log({f"bfcl/{family_name}/it/{cat}_ast": metrics["ast_accuracy"]})

            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  ERROR evaluating IT: {e}")
            traceback.print_exc()

        # --- SAR-5% ---
        print(f"\n--- {family_name} SAR-5% ---")
        try:
            attr_scores = load_attribution_scores(attr_file)
            if not attr_scores:
                print(f"  Skipping SAR-5%: no attribution scores for {family_name}")
            else:
                # Reload IT model, then apply SAR
                model = AutoModelForCausalLM.from_pretrained(
                    it_dir, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
                )
                model = apply_sar(model, base_dir, attr_scores, k_pct=5)

                # Use IT tokenizer for SAR (same chat template)
                sar_tokenizer = AutoTokenizer.from_pretrained(it_dir, trust_remote_code=True)
                if sar_tokenizer.pad_token is None:
                    sar_tokenizer.pad_token = sar_tokenizer.eos_token

                summary, details = evaluate_model_variant(
                    model, sar_tokenizer, device, bfcl_data, f"{family_name}/SAR-5%"
                )
                family_results["sar_5pct"] = summary

                if HAS_WANDB:
                    for cat, metrics in summary.items():
                        if isinstance(metrics, dict) and "ast_accuracy" in metrics:
                            wandb.log({f"bfcl/{family_name}/sar5/{cat}_ast": metrics["ast_accuracy"]})

                del model
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"  ERROR evaluating SAR-5%: {e}")
            traceback.print_exc()

        all_results[family_name] = family_results

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print(f"BFCL RESULTS SUMMARY {'(SYNTHETIC)' if is_synthetic else '(REAL BFCL v3)'}")
    print(f"{'=' * 80}")

    categories = ["simple", "multiple", "parallel", "relevance", "overall"]
    header = f"{'Model':<28}" + "".join(f"{c:>12}" for c in categories)
    print(header)
    print("-" * len(header))

    for family_name, family_results in all_results.items():
        for variant in ["base", "it", "sar_5pct"]:
            if variant not in family_results:
                continue
            label = f"{family_name}/{variant}"
            row = f"{label:<28}"
            for cat in categories:
                if cat in family_results[variant]:
                    acc = family_results[variant][cat]["ast_accuracy"]
                    row += f"{acc:>11.1%} "
                else:
                    row += f"{'N/A':>12}"
            print(row)

    # Alignment tax analysis
    print(f"\n{'=' * 80}")
    print("ALIGNMENT TAX ANALYSIS (IT vs Base)")
    print(f"{'=' * 80}")

    tax_summary = {}
    for family_name, family_results in all_results.items():
        if "base" in family_results and "it" in family_results:
            base_ast = family_results["base"]["overall"]["ast_accuracy"]
            it_ast = family_results["it"]["overall"]["ast_accuracy"]
            tax = it_ast - base_ast
            print(f"  {family_name}: Base={base_ast:.1%}  IT={it_ast:.1%}  Tax={tax:+.1%}")

            tax_entry = {"base_ast": base_ast, "it_ast": it_ast, "tax": tax}

            if "sar_5pct" in family_results:
                sar_ast = family_results["sar_5pct"]["overall"]["ast_accuracy"]
                recovery = sar_ast - it_ast
                tax_entry["sar5_ast"] = sar_ast
                tax_entry["sar5_recovery"] = recovery
                print(f"           SAR-5%={sar_ast:.1%}  Recovery={recovery:+.1%}")

            tax_summary[family_name] = tax_entry
            if HAS_WANDB:
                wandb.log({
                    f"bfcl_tax/{family_name}/base_ast": base_ast,
                    f"bfcl_tax/{family_name}/it_ast": it_ast,
                    f"bfcl_tax/{family_name}/tax": tax,
                })
                if "sar5_ast" in tax_entry:
                    wandb.log({
                        f"bfcl_tax/{family_name}/sar5_ast": tax_entry["sar5_ast"],
                        f"bfcl_tax/{family_name}/sar5_recovery": tax_entry["sar5_recovery"],
                    })

    # Log summary table to W&B
    table_data = []
    for family_name, family_results in all_results.items():
        for variant in ["base", "it", "sar_5pct"]:
            if variant not in family_results:
                continue
            row = {
                "model": family_name,
                "variant": variant,
            }
            for cat in categories:
                if cat in family_results[variant]:
                    row[f"{cat}_ast"] = family_results[variant][cat]["ast_accuracy"]
            table_data.append(row)

    if table_data and HAS_WANDB:
        wandb.log({"bfcl_summary": wandb.Table(
            columns=list(table_data[0].keys()),
            data=[list(r.values()) for r in table_data]
        )})

    # Save results
    output = {
        "analysis": "bfcl_standard",
        "dataset": "synthetic_bfcl_style" if is_synthetic else "gorilla-llm/Berkeley-Function-Calling-Leaderboard",
        "categories": {cat: len(exs) for cat, exs in bfcl_data.items()},
        "total_examples": total_examples,
        "results": all_results,
        "alignment_tax": tax_summary,
    }

    out_path = RESULTS_DIR / "bfcl_standard.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    if HAS_WANDB:
        wandb.finish()
    print("\nDone.")


if __name__ == "__main__":
    main()
