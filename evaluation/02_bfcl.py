"""
Function Calling Benchmark (BFCL-style)

Evaluates structured function calling capability across Base/IT/SAR models.
Inspired by Berkeley Function Calling Leaderboard (BFCL) categories:
  1. Simple Function Call (single function, correct args)
  2. Multiple Function Selection (choose from several functions)
  3. Parallel Function Calls (call multiple functions)
  4. Relevance Detection (no suitable function available)

Metrics:
  - Function name accuracy (correct function selected)
  - Argument accuracy (correct arguments provided)
  - AST accuracy (both name and args correct)
  - Format validity (output is parseable as a function call)

Evaluates: Base, IT, SAR-5%, SAR-10%
"""

import sys
import json
import re
import torch
from pathlib import Path

RESULTS_DIR = Path("./results")

# ============================================================
# Function definitions (tools available to the model)
# ============================================================

TOOL_DEFINITIONS = {
    "get_weather": {
        "description": "Get current weather for a city",
        "parameters": {"city": "string", "unit": "string (celsius/fahrenheit, default: celsius)"}
    },
    "search_web": {
        "description": "Search the web for information",
        "parameters": {"query": "string", "num_results": "integer (default: 5)"}
    },
    "send_email": {
        "description": "Send an email to a recipient",
        "parameters": {"to": "string", "subject": "string", "body": "string"}
    },
    "create_calendar_event": {
        "description": "Create a calendar event",
        "parameters": {"title": "string", "date": "string (YYYY-MM-DD)", "time": "string (HH:MM)", "duration_minutes": "integer"}
    },
    "calculate": {
        "description": "Evaluate a mathematical expression",
        "parameters": {"expression": "string"}
    },
    "translate_text": {
        "description": "Translate text between languages",
        "parameters": {"text": "string", "source_lang": "string", "target_lang": "string"}
    },
    "get_stock_price": {
        "description": "Get current stock price",
        "parameters": {"symbol": "string"}
    },
    "set_reminder": {
        "description": "Set a reminder",
        "parameters": {"message": "string", "time": "string (HH:MM)", "date": "string (YYYY-MM-DD, optional)"}
    },
    "file_search": {
        "description": "Search for files matching a pattern",
        "parameters": {"pattern": "string", "directory": "string (default: .)"}
    },
    "run_sql": {
        "description": "Execute a SQL query on the database",
        "parameters": {"query": "string", "database": "string (default: main)"}
    },
    "http_request": {
        "description": "Make an HTTP request",
        "parameters": {"method": "string (GET/POST/PUT/DELETE)", "url": "string", "body": "string (optional)", "headers": "dict (optional)"}
    },
    "create_file": {
        "description": "Create a new file with content",
        "parameters": {"path": "string", "content": "string"}
    },
}

# ============================================================
# Test cases organized by BFCL category
# ============================================================

# Category 1: Simple Function Call (one function, straightforward)
SIMPLE_CALLS = [
    {
        "tools": ["get_weather"],
        "query": "What's the weather in Tokyo?",
        "expected_func": "get_weather",
        "expected_args": {"city": "Tokyo"},
        "required_args": ["city"],
    },
    {
        "tools": ["calculate"],
        "query": "What is 15 * 23 + 7?",
        "expected_func": "calculate",
        "expected_args": {"expression": "15 * 23 + 7"},
        "required_args": ["expression"],
    },
    {
        "tools": ["search_web"],
        "query": "Search for recent news about AI regulation",
        "expected_func": "search_web",
        "expected_args": {"query": "recent news AI regulation"},
        "required_args": ["query"],
    },
    {
        "tools": ["get_stock_price"],
        "query": "What's Apple's stock price?",
        "expected_func": "get_stock_price",
        "expected_args": {"symbol": "AAPL"},
        "required_args": ["symbol"],
    },
    {
        "tools": ["translate_text"],
        "query": "Translate 'hello world' from English to Spanish",
        "expected_func": "translate_text",
        "expected_args": {"text": "hello world", "source_lang": "English", "target_lang": "Spanish"},
        "required_args": ["text", "target_lang"],
    },
    {
        "tools": ["set_reminder"],
        "query": "Remind me to call the dentist at 3pm",
        "expected_func": "set_reminder",
        "expected_args": {"message": "call the dentist", "time": "15:00"},
        "required_args": ["message", "time"],
    },
    {
        "tools": ["file_search"],
        "query": "Find all Python files in the src directory",
        "expected_func": "file_search",
        "expected_args": {"pattern": "*.py", "directory": "src"},
        "required_args": ["pattern"],
    },
    {
        "tools": ["run_sql"],
        "query": "Get all users from the database",
        "expected_func": "run_sql",
        "expected_args": {"query": "SELECT * FROM users"},
        "required_args": ["query"],
    },
    {
        "tools": ["create_file"],
        "query": "Create a file called hello.txt with content 'Hello World'",
        "expected_func": "create_file",
        "expected_args": {"path": "hello.txt", "content": "Hello World"},
        "required_args": ["path", "content"],
    },
    {
        "tools": ["http_request"],
        "query": "Make a GET request to https://api.example.com/users",
        "expected_func": "http_request",
        "expected_args": {"method": "GET", "url": "https://api.example.com/users"},
        "required_args": ["method", "url"],
    },
    {
        "tools": ["get_weather"],
        "query": "Temperature in Paris in fahrenheit?",
        "expected_func": "get_weather",
        "expected_args": {"city": "Paris", "unit": "fahrenheit"},
        "required_args": ["city"],
    },
    {
        "tools": ["send_email"],
        "query": "Send an email to bob@example.com about the meeting tomorrow",
        "expected_func": "send_email",
        "expected_args": {"to": "bob@example.com", "subject": "meeting tomorrow"},
        "required_args": ["to", "subject"],
    },
    {
        "tools": ["create_calendar_event"],
        "query": "Schedule a team standup on 2024-03-15 at 9:00 for 30 minutes",
        "expected_func": "create_calendar_event",
        "expected_args": {"title": "team standup", "date": "2024-03-15", "time": "9:00", "duration_minutes": 30},
        "required_args": ["title", "date", "time"],
    },
    {
        "tools": ["calculate"],
        "query": "What's the square root of 144?",
        "expected_func": "calculate",
        "expected_args": {"expression": "sqrt(144)"},
        "required_args": ["expression"],
    },
    {
        "tools": ["run_sql"],
        "query": "Count the number of orders placed in January",
        "expected_func": "run_sql",
        "expected_args": {"query": "SELECT COUNT(*) FROM orders WHERE month = 1"},
        "required_args": ["query"],
    },
]

# Category 2: Multiple Function Selection (choose correct one from many)
MULTI_SELECT = [
    {
        "tools": ["get_weather", "search_web", "calculate", "send_email"],
        "query": "What is 2^10?",
        "expected_func": "calculate",
        "required_args": ["expression"],
    },
    {
        "tools": ["get_weather", "get_stock_price", "translate_text", "search_web"],
        "query": "What's the current price of Tesla stock?",
        "expected_func": "get_stock_price",
        "required_args": ["symbol"],
    },
    {
        "tools": ["send_email", "create_calendar_event", "set_reminder", "search_web"],
        "query": "Email alice@work.com the quarterly report",
        "expected_func": "send_email",
        "required_args": ["to"],
    },
    {
        "tools": ["file_search", "run_sql", "http_request", "create_file"],
        "query": "Query the database for all products under $50",
        "expected_func": "run_sql",
        "required_args": ["query"],
    },
    {
        "tools": ["get_weather", "translate_text", "calculate", "set_reminder"],
        "query": "Translate 'good morning' to French",
        "expected_func": "translate_text",
        "required_args": ["text", "target_lang"],
    },
    {
        "tools": ["search_web", "file_search", "create_file", "http_request"],
        "query": "Find all .json files in the config directory",
        "expected_func": "file_search",
        "required_args": ["pattern"],
    },
    {
        "tools": ["get_weather", "search_web", "send_email", "create_calendar_event"],
        "query": "Check the weather in London",
        "expected_func": "get_weather",
        "required_args": ["city"],
    },
    {
        "tools": ["calculate", "get_stock_price", "run_sql", "translate_text"],
        "query": "What is 15% of 250?",
        "expected_func": "calculate",
        "required_args": ["expression"],
    },
    {
        "tools": ["send_email", "set_reminder", "create_calendar_event", "search_web"],
        "query": "Set up a meeting with the team on Friday at 2pm for 1 hour",
        "expected_func": "create_calendar_event",
        "required_args": ["title", "time"],
    },
    {
        "tools": ["http_request", "run_sql", "file_search", "search_web"],
        "query": "POST to https://api.example.com/data with body {\"key\": \"value\"}",
        "expected_func": "http_request",
        "required_args": ["method", "url"],
    },
    {
        "tools": ["get_weather", "get_stock_price", "calculate", "translate_text"],
        "query": "What's NVIDIA stock at?",
        "expected_func": "get_stock_price",
        "required_args": ["symbol"],
    },
    {
        "tools": ["create_file", "send_email", "file_search", "run_sql"],
        "query": "Write 'import os' to a file called utils.py",
        "expected_func": "create_file",
        "required_args": ["path", "content"],
    },
    {
        "tools": ["search_web", "translate_text", "get_weather", "calculate"],
        "query": "Search for Python tutorials for beginners",
        "expected_func": "search_web",
        "required_args": ["query"],
    },
    {
        "tools": ["set_reminder", "send_email", "create_calendar_event", "calculate"],
        "query": "Remind me to buy groceries at 5:30pm",
        "expected_func": "set_reminder",
        "required_args": ["message", "time"],
    },
    {
        "tools": ["run_sql", "http_request", "search_web", "file_search"],
        "query": "Delete all expired sessions from the database",
        "expected_func": "run_sql",
        "required_args": ["query"],
    },
]

# Category 3: Parallel Function Calls (need to call multiple functions)
PARALLEL_CALLS = [
    {
        "tools": ["get_weather", "get_stock_price"],
        "query": "What's the weather in NYC and what's AAPL stock price?",
        "expected_funcs": ["get_weather", "get_stock_price"],
        "min_calls": 2,
    },
    {
        "tools": ["send_email", "set_reminder"],
        "query": "Email john@work.com about the deadline and set a reminder for 5pm to follow up",
        "expected_funcs": ["send_email", "set_reminder"],
        "min_calls": 2,
    },
    {
        "tools": ["search_web", "calculate"],
        "query": "Search for the population of Japan and calculate 126 million divided by 377975 sq km",
        "expected_funcs": ["search_web", "calculate"],
        "min_calls": 2,
    },
    {
        "tools": ["get_weather", "translate_text"],
        "query": "Check weather in Berlin and translate 'raining' to German",
        "expected_funcs": ["get_weather", "translate_text"],
        "min_calls": 2,
    },
    {
        "tools": ["create_file", "run_sql"],
        "query": "Create a backup.sql file and run SELECT * FROM users to get the data",
        "expected_funcs": ["create_file", "run_sql"],
        "min_calls": 2,
    },
    {
        "tools": ["get_weather", "get_weather"],
        "query": "Compare weather in Tokyo and London",
        "expected_funcs": ["get_weather", "get_weather"],
        "min_calls": 2,
    },
    {
        "tools": ["file_search", "file_search"],
        "query": "Find all .py files in src/ and all .json files in config/",
        "expected_funcs": ["file_search", "file_search"],
        "min_calls": 2,
    },
    {
        "tools": ["calculate", "calculate"],
        "query": "What is 15*23 and what is 100/7?",
        "expected_funcs": ["calculate", "calculate"],
        "min_calls": 2,
    },
    {
        "tools": ["translate_text", "translate_text"],
        "query": "Translate 'hello' to both French and Spanish",
        "expected_funcs": ["translate_text", "translate_text"],
        "min_calls": 2,
    },
    {
        "tools": ["http_request", "run_sql"],
        "query": "Fetch data from https://api.example.com/sync and update the local database with INSERT INTO sync_log VALUES('done')",
        "expected_funcs": ["http_request", "run_sql"],
        "min_calls": 2,
    },
]

# Category 4: Relevance Detection (no suitable function)
RELEVANCE_DETECTION = [
    {
        "tools": ["get_weather", "calculate"],
        "query": "Send an email to my boss about the project update",
        "expected_func": None,  # No email function available
    },
    {
        "tools": ["search_web", "translate_text"],
        "query": "Set a timer for 10 minutes",
        "expected_func": None,  # No timer function
    },
    {
        "tools": ["get_stock_price", "calculate"],
        "query": "Book a flight to Paris for next Tuesday",
        "expected_func": None,  # No booking function
    },
    {
        "tools": ["file_search", "create_file"],
        "query": "Play some relaxing music",
        "expected_func": None,
    },
    {
        "tools": ["run_sql", "http_request"],
        "query": "What's the weather forecast for tomorrow?",
        "expected_func": None,  # No weather function in available tools
    },
    {
        "tools": ["get_weather", "get_stock_price"],
        "query": "Order pizza from the nearest restaurant",
        "expected_func": None,
    },
    {
        "tools": ["calculate", "translate_text"],
        "query": "Take a screenshot of my desktop",
        "expected_func": None,
    },
    {
        "tools": ["search_web", "file_search"],
        "query": "Turn off the living room lights",
        "expected_func": None,
    },
    {
        "tools": ["send_email", "create_calendar_event"],
        "query": "Download the latest version of Python",
        "expected_func": None,
    },
    {
        "tools": ["set_reminder", "calculate"],
        "query": "Edit the photo to make it brighter",
        "expected_func": None,
    },
]


def format_tools_prompt(tool_names):
    """Format available tools into a prompt section."""
    lines = ["You have access to the following functions:\n"]
    for name in tool_names:
        if name in TOOL_DEFINITIONS:
            t = TOOL_DEFINITIONS[name]
            params = ", ".join(f"{k}: {v}" for k, v in t["parameters"].items())
            lines.append(f"  {name}({params})")
            lines.append(f"    Description: {t['description']}\n")
    return "\n".join(lines)


def format_prompt(test_case):
    """Create the full prompt for a test case."""
    tools_text = format_tools_prompt(test_case["tools"])
    query = test_case["query"]

    prompt = f"""{tools_text}
User request: {query}

Call the appropriate function(s) with correct arguments. Output ONLY the function call(s) in the format:
function_name(arg1="value1", arg2="value2")

If no function is suitable, respond with: NO_FUNCTION_AVAILABLE

Function call:
"""
    return prompt


def parse_function_call(text):
    """Parse a function call string into (name, args) tuple."""
    text = text.strip()
    # Try to match function_name(args)
    match = re.match(r'(\w+)\s*\((.*)\)', text, re.DOTALL)
    if not match:
        return None, {}

    func_name = match.group(1)
    args_str = match.group(2).strip()

    if not args_str:
        return func_name, {}

    # Parse keyword arguments
    args = {}
    # Match key=value or key="value" patterns
    for m in re.finditer(r'(\w+)\s*=\s*(?:"([^"]*?)"|\'([^\']*?)\'|(\d+(?:\.\d+)?)|(\w+))', args_str):
        key = m.group(1)
        value = m.group(2) or m.group(3) or m.group(4) or m.group(5)
        if value and value.isdigit():
            value = int(value)
        args[key] = value

    return func_name, args


def parse_all_calls(text):
    """Parse multiple function calls from text."""
    calls = []
    # Find all function call patterns
    for match in re.finditer(r'(\w+)\s*\(([^)]*)\)', text):
        func_name = match.group(1)
        # Skip common false positives
        if func_name.lower() in ('if', 'for', 'while', 'print', 'len', 'str', 'int', 'float',
                                   'format', 'the', 'no', 'none', 'not', 'is', 'function'):
            continue
        if func_name in TOOL_DEFINITIONS:
            args_str = match.group(2)
            args = {}
            for am in re.finditer(r'(\w+)\s*=\s*(?:"([^"]*?)"|\'([^\']*?)\'|(\d+(?:\.\d+)?)|(\w+))', args_str):
                key = am.group(1)
                value = am.group(2) or am.group(3) or am.group(4) or am.group(5)
                if value and isinstance(value, str) and value.isdigit():
                    value = int(value)
                args[key] = value
            calls.append((func_name, args))

    return calls


def evaluate_simple(model, tokenizer, device, test_cases):
    """Evaluate simple and multi-select function calling."""
    results = {"correct_func": 0, "correct_args": 0, "ast_correct": 0,
               "valid_format": 0, "total": 0, "details": []}

    model.eval()
    with torch.no_grad():
        for tc in test_cases:
            prompt = format_prompt(tc)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            input_ids = inputs["input_ids"].to(device)

            outputs = model.generate(
                input_ids, max_new_tokens=150, do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True).strip()

            # Parse function call
            func_name, args = parse_function_call(response.split("\n")[0])

            expected_func = tc.get("expected_func", tc.get("expected_funcs", [None])[0] if "expected_funcs" in tc else None)
            func_correct = func_name == expected_func
            valid = func_name is not None

            # Check required args
            required_args = tc.get("required_args", [])
            args_correct = all(k in args for k in required_args) if func_correct else False

            ast_correct = func_correct and args_correct

            results["correct_func"] += int(func_correct)
            results["correct_args"] += int(args_correct)
            results["ast_correct"] += int(ast_correct)
            results["valid_format"] += int(valid)
            results["total"] += 1

            results["details"].append({
                "query": tc["query"][:60],
                "expected": expected_func,
                "got_func": func_name,
                "got_args": args,
                "func_ok": func_correct,
                "args_ok": args_correct,
                "ast_ok": ast_correct,
            })

            del outputs
            torch.cuda.empty_cache()

    return results


def evaluate_parallel(model, tokenizer, device, test_cases):
    """Evaluate parallel function calling."""
    results = {"detected_multi": 0, "all_funcs_correct": 0, "total": 0, "details": []}

    model.eval()
    with torch.no_grad():
        for tc in test_cases:
            prompt = format_prompt(tc)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            input_ids = inputs["input_ids"].to(device)

            outputs = model.generate(
                input_ids, max_new_tokens=300, do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True).strip()

            calls = parse_all_calls(response)
            detected_multi = len(calls) >= tc["min_calls"]

            # Check if all expected functions are present
            got_funcs = [c[0] for c in calls]
            expected_funcs = tc["expected_funcs"]
            all_present = all(f in got_funcs for f in set(expected_funcs))

            results["detected_multi"] += int(detected_multi)
            results["all_funcs_correct"] += int(all_present)
            results["total"] += 1

            results["details"].append({
                "query": tc["query"][:60],
                "expected_funcs": expected_funcs,
                "got_funcs": got_funcs,
                "multi_ok": detected_multi,
                "funcs_ok": all_present,
            })

            del outputs
            torch.cuda.empty_cache()

    return results


def evaluate_relevance(model, tokenizer, device, test_cases):
    """Evaluate relevance detection (should NOT call a function)."""
    results = {"correct_rejection": 0, "total": 0, "details": []}

    model.eval()
    with torch.no_grad():
        for tc in test_cases:
            prompt = format_prompt(tc)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            input_ids = inputs["input_ids"].to(device)

            outputs = model.generate(
                input_ids, max_new_tokens=150, do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True).strip()

            # Check if model correctly identifies no function is suitable
            calls = parse_all_calls(response)
            no_call = len(calls) == 0 or "NO_FUNCTION" in response.upper() or "no suitable" in response.lower() or "none" in response.lower()[:20]

            results["correct_rejection"] += int(no_call)
            results["total"] += 1

            results["details"].append({
                "query": tc["query"][:60],
                "response": response[:80],
                "correct": no_call,
            })

            del outputs
            torch.cuda.empty_cache()

    return results


def load_attribution_scores(model_name):
    """Load attribution scores for SAR."""
    safe_name = model_name.replace("/", "_").replace(" ", "_").lower()
    for prefix in ["attribution"]:
        path = RESULTS_DIR / f"{prefix}_{safe_name}.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            return {c["name"]: c.get("harm_score", c.get("mean_score", 0)) for c in data.get("components", [])}
    return {}


def apply_sar(model, base_dir, attribution_scores, k_pct=5):
    """Apply Surgical Alignment Reversal."""
    from safetensors import safe_open
    import glob

    sorted_comps = sorted(attribution_scores.items(), key=lambda x: x[1], reverse=True)
    n_rollback = max(1, int(len(sorted_comps) * k_pct / 100))
    rollback_names = {name for name, _ in sorted_comps[:n_rollback]}
    print(f"  SAR-{k_pct}%: rolling back {n_rollback}/{len(sorted_comps)} components")

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
    print(f"  Rolled back {rolled_back} tensors")
    return model


def evaluate_variant(model, tokenizer, device, label):
    """Run full BFCL evaluation on a model variant."""
    print(f"\n  [{label}] Evaluating function calling...")

    simple_res = evaluate_simple(model, tokenizer, device, SIMPLE_CALLS)
    multi_res = evaluate_simple(model, tokenizer, device, MULTI_SELECT)
    parallel_res = evaluate_parallel(model, tokenizer, device, PARALLEL_CALLS)
    relevance_res = evaluate_relevance(model, tokenizer, device, RELEVANCE_DETECTION)

    # Aggregate
    total_ast = simple_res["ast_correct"] + multi_res["ast_correct"]
    total_func = simple_res["correct_func"] + multi_res["correct_func"]
    total_single = simple_res["total"] + multi_res["total"]

    summary = {
        "simple": {
            "func_accuracy": simple_res["correct_func"] / max(simple_res["total"], 1),
            "arg_accuracy": simple_res["correct_args"] / max(simple_res["total"], 1),
            "ast_accuracy": simple_res["ast_correct"] / max(simple_res["total"], 1),
            "n": simple_res["total"],
        },
        "multi_select": {
            "func_accuracy": multi_res["correct_func"] / max(multi_res["total"], 1),
            "arg_accuracy": multi_res["correct_args"] / max(multi_res["total"], 1),
            "ast_accuracy": multi_res["ast_correct"] / max(multi_res["total"], 1),
            "n": multi_res["total"],
        },
        "parallel": {
            "multi_detect_rate": parallel_res["detected_multi"] / max(parallel_res["total"], 1),
            "all_funcs_correct": parallel_res["all_funcs_correct"] / max(parallel_res["total"], 1),
            "n": parallel_res["total"],
        },
        "relevance": {
            "correct_rejection": relevance_res["correct_rejection"] / max(relevance_res["total"], 1),
            "n": relevance_res["total"],
        },
        "overall": {
            "func_accuracy": total_func / max(total_single, 1),
            "ast_accuracy": total_ast / max(total_single, 1),
        },
    }

    print(f"  [{label}] Simple AST: {summary['simple']['ast_accuracy']:.1%} "
          f"| Multi-select AST: {summary['multi_select']['ast_accuracy']:.1%} "
          f"| Parallel: {summary['parallel']['multi_detect_rate']:.1%} "
          f"| Relevance: {summary['relevance']['correct_rejection']:.1%}")

    return summary


def run_bfcl(base_dir, it_dir, device="cuda:0", model_name="unknown"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(it_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_results = {}

    # Base model
    print(f"\n{'='*60}")
    print(f"BFCL EVALUATION: {model_name}")
    print(f"{'='*60}")

    print(f"\n--- BASE ---")
    base_tokenizer = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=True)
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_dir, torch_dtype=torch.float16, device_map=device, trust_remote_code=True)
    all_results["base"] = evaluate_variant(model, base_tokenizer, device, "Base")
    del model; torch.cuda.empty_cache()

    # IT model
    print(f"\n--- IT ---")
    model = AutoModelForCausalLM.from_pretrained(
        it_dir, torch_dtype=torch.float16, device_map=device, trust_remote_code=True)
    all_results["it"] = evaluate_variant(model, tokenizer, device, "IT")
    del model; torch.cuda.empty_cache()

    # SAR-5%
    attribution_scores = load_attribution_scores(model_name)
    if attribution_scores:
        print(f"\n--- SAR-5% ---")
        model = AutoModelForCausalLM.from_pretrained(
            it_dir, torch_dtype=torch.float16, device_map=device, trust_remote_code=True)
        model = apply_sar(model, base_dir, attribution_scores, k_pct=5)
        all_results["sar_5pct"] = evaluate_variant(model, tokenizer, device, "SAR-5%")
        del model; torch.cuda.empty_cache()

        # SAR-10%
        print(f"\n--- SAR-10% ---")
        model = AutoModelForCausalLM.from_pretrained(
            it_dir, torch_dtype=torch.float16, device_map=device, trust_remote_code=True)
        model = apply_sar(model, base_dir, attribution_scores, k_pct=10)
        all_results["sar_10pct"] = evaluate_variant(model, tokenizer, device, "SAR-10%")
        del model; torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print(f"BFCL SUMMARY: {model_name}")
    print(f"{'='*60}")
    print(f"{'Variant':<12} {'Simple AST':>12} {'MultiSel AST':>14} {'Parallel':>10} {'Relevance':>10} {'Overall AST':>12}")
    print(f"{'-'*70}")
    for name, r in all_results.items():
        print(f"{name:<12} {r['simple']['ast_accuracy']:>11.1%} {r['multi_select']['ast_accuracy']:>13.1%} "
              f"{r['parallel']['multi_detect_rate']:>9.1%} {r['relevance']['correct_rejection']:>9.1%} "
              f"{r['overall']['ast_accuracy']:>11.1%}")

    # Alignment tax on function calling
    if "base" in all_results and "it" in all_results:
        base_ast = all_results["base"]["overall"]["ast_accuracy"]
        it_ast = all_results["it"]["overall"]["ast_accuracy"]
        print(f"\nFunction calling alignment tax: {it_ast - base_ast:+.1%}")
        if "sar_5pct" in all_results:
            sar_ast = all_results["sar_5pct"]["overall"]["ast_accuracy"]
            print(f"SAR-5% recovery: {sar_ast - it_ast:+.1%}")

    # Save
    safe_name = model_name.replace("/", "_").replace(" ", "_").lower()
    output = {
        "analysis": "BFCL-style function calling benchmark",
        "model": model_name,
        "n_simple": len(SIMPLE_CALLS),
        "n_multi_select": len(MULTI_SELECT),
        "n_parallel": len(PARALLEL_CALLS),
        "n_relevance": len(RELEVANCE_DETECTION),
        "n_total": len(SIMPLE_CALLS) + len(MULTI_SELECT) + len(PARALLEL_CALLS) + len(RELEVANCE_DETECTION),
        "results": all_results,
    }
    out_path = RESULTS_DIR / f"bfcl_{safe_name}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    base_dir = sys.argv[1]
    it_dir = sys.argv[2]
    device = sys.argv[3] if len(sys.argv) > 3 else "cuda:0"
    model_name = sys.argv[4] if len(sys.argv) > 4 else "unknown"

    run_bfcl(base_dir, it_dir, device=device, model_name=model_name)
