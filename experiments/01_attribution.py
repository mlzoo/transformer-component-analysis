"""
Attribution with larger, more diverse agent example set.
50 examples instead of 20 for more statistical power on V/O vs Q/K.
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from safetensors import safe_open

RESULTS_DIR = Path("./results")

# Expanded agent examples — 50 diverse structured generation tasks
AGENT_EXAMPLES = [
    # Tool calls
    {"prompt": "You are a helpful assistant with tools.\nTools: search(query), calculator(expr)\nUser: Population of France times 2?\nThought: Search first.\nAction: search(query=\"population of France\")\nObservation: 68 million.\nThought: Multiply.\nAction: ", "target": 'calculator(expression="68000000 * 2")'},
    {"prompt": "Tools: web_search(q), read_file(path)\nTask: Weather in NYC\nThought: Search.\nAction: ", "target": 'web_search(q="weather NYC")'},
    {"prompt": "Function call: send_email(to=", "target": '"user@example.com", subject="Hello", body="Test")'},
    {"prompt": "Tools: calculator(expr)\nUser: What is 15*23?\nThought: Use calculator.\nAction: ", "target": 'calculator(expr="15*23")'},
    {"prompt": "Tools: translate(text, lang)\nUser: Translate hello to Spanish\nAction: ", "target": 'translate(text="hello", lang="es")'},
    # JSON
    {"prompt": 'Respond in JSON.\nUser: What is 2+2?\n\n{"', "target": '"answer": 4}'},
    {"prompt": 'Output JSON: Name=Alice, Age=30\n\n{"name": "', "target": 'Alice", "age": 30}'},
    {"prompt": 'Parse to JSON: "Meeting 3pm Room 204"\n\n{"', "target": '"event": "Meeting", "time": "3pm", "location": "Room 204"}'},
    {"prompt": 'Product: Widget, Price: $9.99, Stock: 150\n\n{"product": "', "target": 'Widget", "price": 9.99, "stock": 150}'},
    {"prompt": 'Code review bot. Code: x = eval(input())\n\n{"', "target": '"verdict": "reject", "reason": "eval on user input"}'},
    {"prompt": 'Router: GET /api/users/123\n\n{"', "target": '"handler": "getUser", "params": {"id": "123"}}'},
    {"prompt": 'CI decision: 142/142 unit pass, 38/40 integ pass\n\n{"', "target": '"action": "proceed", "deploy": true, "warnings": ["2 flaky tests"]}'},
    {"prompt": 'Error log: NullPointerException at UserService.java:42\n\n{"', "target": '"severity": "high", "file": "UserService.java", "line": 42}'},
    {"prompt": 'Sentiment: "This product is amazing!"\n\n{"', "target": '"sentiment": "positive", "confidence": 0.95}'},
    {"prompt": 'Extract entities: "John works at Google in NYC"\n\n{"', "target": '"entities": [{"text": "John", "type": "PERSON"}, {"text": "Google", "type": "ORG"}]}'},
    # ReAct
    {"prompt": "ReAct agent.\nQ: Capital of France?\nThought: Search.\nAction: ", "target": "search[capital of France]"},
    {"prompt": "ReAct agent.\nQ: Who wrote Hamlet?\nThought: I should look this up.\nAction: ", "target": "search[author of Hamlet]"},
    {"prompt": "ReAct agent.\nQ: Distance from Earth to Mars?\nThought: Need to search.\nAction: ", "target": "search[distance Earth Mars]"},
    # SQL
    {"prompt": "SQL: Get users where age > 25\n\nSELECT ", "target": "* FROM users WHERE age > 25;"},
    {"prompt": "SQL: Count orders per customer\n\nSELECT ", "target": "customer_id, COUNT(*) FROM orders GROUP BY customer_id;"},
    {"prompt": "SQL: Top 10 products by revenue\n\nSELECT ", "target": "product_name, SUM(price * quantity) as revenue FROM orders GROUP BY product_name ORDER BY revenue DESC LIMIT 10;"},
    # API
    {"prompt": "API call: Delete user 42\n\n", "target": "DELETE /api/users/42"},
    {"prompt": "API call: Update user 7 email\n\n", "target": 'PATCH /api/users/7 {"email": "new@example.com"}'},
    {"prompt": "API call: List all products\n\n", "target": "GET /api/products"},
    # Code
    {"prompt": "```python\ndef fibonacci(n):\n    ", "target": "if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"},
    {"prompt": "```python\ndef is_palindrome(s):\n    ", "target": "return s == s[::-1]"},
    {"prompt": "```python\ndef binary_search(arr, target):\n    ", "target": "left, right = 0, len(arr) - 1\n    while left <= right:"},
    {"prompt": "```javascript\nfunction debounce(fn, delay) {\n    ", "target": "let timer;\n    return function(...args) {"},
    # Bash
    {"prompt": "Bash: List .py files modified today\n\n```bash\n", "target": "find . -name '*.py' -mtime 0\n```"},
    {"prompt": "Bash: Find processes using port 8080\n\n```bash\n", "target": "lsof -i :8080\n```"},
    # Config
    {"prompt": "YAML config:\nserver:\n  host: 0.0.0.0\n  port: ", "target": "8080\n  workers: 4"},
    {"prompt": "Docker Compose:\nservices:\n  web:\n    image: ", "target": "nginx:latest\n    ports:\n      - '80:80'"},
    {"prompt": "Dockerfile for Flask:\n\nFROM ", "target": "python:3.11-slim\nWORKDIR /app\nCOPY requirements.txt .\nRUN pip install -r requirements.txt"},
    # GraphQL
    {"prompt": "GraphQL: Get user by ID with posts\n\n```graphql\n", "target": "query GetUser($id: ID!) {\n  user(id: $id) {\n    name\n    posts { title }\n  }\n}"},
    {"prompt": "GraphQL mutation: Create user\n\n```graphql\n", "target": "mutation CreateUser($input: UserInput!) {\n  createUser(input: $input) {\n    id\n    name\n  }\n}"},
    # MongoDB
    {"prompt": "MongoDB: Orders over $100 last week\n\ndb.orders.find(", "target": '{"amount": {"$gt": 100}})'},
    {"prompt": "MongoDB: Update user email\n\ndb.users.updateOne(", "target": '{"_id": ObjectId("abc")}, {"$set": {"email": "new@test.com"}})'},
    # Cron
    {"prompt": "Cron: Every Monday 9am\n\n", "target": "0 9 * * 1"},
    {"prompt": "Cron: Every 5 minutes\n\n", "target": "*/5 * * * *"},
    # Regex
    {"prompt": "Regex: Match email addresses\n\n", "target": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"},
    {"prompt": "Regex: Match ISO date (YYYY-MM-DD)\n\n", "target": r"\d{4}-\d{2}-\d{2}"},
    # Markdown
    {"prompt": "Create a markdown table:\n| Name | Age |\n|", "target": "------|-----|\n| Alice | 30 |\n| Bob | 25 |"},
    # TypeScript
    {"prompt": "```typescript\ninterface User {\n  ", "target": "id: number;\n  name: string;\n  email: string;\n}"},
    # CSS
    {"prompt": "CSS: Center div horizontally and vertically\n\n.container {\n  ", "target": "display: flex;\n  justify-content: center;\n  align-items: center;\n}"},
    # git
    {"prompt": "Git: Undo last commit but keep changes\n\n$ ", "target": "git reset --soft HEAD~1"},
    # terraform
    {"prompt": "Terraform: AWS EC2 instance\n\nresource \"aws_instance\" \"web\" {\n  ", "target": 'ami           = "ami-0c55b159cbfafe1f0"\n  instance_type = "t2.micro"'},
    # nginx
    {"prompt": "Nginx: Reverse proxy to port 3000\n\nlocation / {\n    ", "target": "proxy_pass http://localhost:3000;\n    proxy_set_header Host $host;"},
    # makefile
    {"prompt": "Makefile target: build and test\n\nall: build test\n\nbuild:\n\t", "target": "go build -o bin/app ./cmd/main.go\n\ntest:\n\tgo test ./..."},
    # GitHub Actions
    {"prompt": "GitHub Actions: Run tests on push\n\nname: CI\non: push\njobs:\n  test:\n    runs-on: ", "target": "ubuntu-latest\n    steps:\n      - uses: actions/checkout@v3"},
]


def classify_component(name):
    if "layers." not in name:
        return "other", -1
    parts = name.split(".")
    layer_idx = None
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try: layer_idx = int(parts[i + 1])
            except: pass
    if layer_idx is None:
        return "other", -1
    if "q_proj" in name: return "W_Q", layer_idx
    elif "k_proj" in name: return "W_K", layer_idx
    elif "v_proj" in name: return "W_V", layer_idx
    elif "o_proj" in name: return "W_O", layer_idx
    elif "gate_proj" in name: return "W_gate", layer_idx
    elif "up_proj" in name: return "W_up", layer_idx
    elif "down_proj" in name: return "W_down", layer_idx
    return "other", layer_idx


def compute_loss(model, tokenizer, examples, device):
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for ex in examples:
            full_text = ex["prompt"] + ex["target"]
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


def run_attribution(base_dir, it_dir, device="cuda:0", model_name="unknown"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading IT model in float16 to {device}...")
    tokenizer = AutoTokenizer.from_pretrained(it_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        it_dir, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
    )
    model.eval()

    base_index = {}
    for f in sorted(Path(base_dir).glob("*.safetensors")):
        with safe_open(str(f), framework="pt", device="cpu") as sf:
            for key in sf.keys():
                base_index[key] = f

    it_index = {}
    for f in sorted(Path(it_dir).glob("*.safetensors")):
        with safe_open(str(f), framework="pt", device="cpu") as sf:
            for key in sf.keys():
                it_index[key] = f

    examples = AGENT_EXAMPLES  # All 49 base examples

    print(f"Measuring baseline IT loss on {len(examples)} examples...")
    baseline_loss = compute_loss(model, tokenizer, examples, device)
    print(f"  Baseline IT loss: {baseline_loss:.4f}")

    results = []
    num_layers = model.config.num_hidden_layers

    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        components = [
            ("W_Q", layer.self_attn.q_proj, f"model.layers.{layer_idx}.self_attn.q_proj.weight"),
            ("W_K", layer.self_attn.k_proj, f"model.layers.{layer_idx}.self_attn.k_proj.weight"),
            ("W_V", layer.self_attn.v_proj, f"model.layers.{layer_idx}.self_attn.v_proj.weight"),
            ("W_O", layer.self_attn.o_proj, f"model.layers.{layer_idx}.self_attn.o_proj.weight"),
            ("W_gate", layer.mlp.gate_proj, f"model.layers.{layer_idx}.mlp.gate_proj.weight"),
            ("W_up", layer.mlp.up_proj, f"model.layers.{layer_idx}.mlp.up_proj.weight"),
            ("W_down", layer.mlp.down_proj, f"model.layers.{layer_idx}.mlp.down_proj.weight"),
        ]

        for comp_name, proj_module, weight_key in components:
            if weight_key not in base_index or weight_key not in it_index:
                continue

            with safe_open(str(base_index[weight_key]), framework="pt", device="cpu") as sf:
                base_w = sf.get_tensor(weight_key)
            with safe_open(str(it_index[weight_key]), framework="pt", device="cpu") as sf:
                it_w = sf.get_tensor(weight_key)

            delta_cpu = (it_w - base_w).float()
            delta_norm = torch.norm(delta_cpu).item()
            del base_w, it_w

            def make_hook(delta):
                def hook_fn(module, input, output):
                    x = input[0] if isinstance(input, tuple) else input
                    x_cpu = x.float().cpu()
                    correction = torch.nn.functional.linear(x_cpu, delta)
                    return output - correction.half().to(output.device)
                return hook_fn

            hook = proj_module.register_forward_hook(make_hook(delta_cpu))
            rollback_loss = compute_loss(model, tokenizer, examples, device)
            hook.remove()

            harm_score = baseline_loss - rollback_loss
            results.append({
                "name": weight_key, "component_type": comp_name, "layer": layer_idx,
                "baseline_loss": float(baseline_loss), "rollback_loss": float(rollback_loss),
                "harm_score": float(harm_score), "delta_norm": float(delta_norm),
            })

            marker = "***" if abs(harm_score) > 0.01 else ""
            print(f"  {comp_name:8s} L{layer_idx:2d}: harm={harm_score:+.4f} {marker}")
            del delta_cpu
            torch.cuda.empty_cache()

    # Analysis
    by_type = defaultdict(list)
    for r in results:
        by_type[r["component_type"]].append(r)

    print(f"\n{'='*80}")
    print(f"ATTRIBUTION RESULTS ({model_name}) — 50 examples")
    print(f"Baseline loss: {baseline_loss:.4f}")
    print(f"{'='*80}")

    type_stats = {}
    for comp in ["W_Q", "W_K", "W_V", "W_O", "W_gate", "W_up", "W_down"]:
        entries = by_type.get(comp, [])
        if not entries: continue
        harms = [e["harm_score"] for e in entries]
        pos_count = sum(1 for h in harms if h > 0)
        type_stats[comp] = {
            "count": len(entries), "mean_harm": float(np.mean(harms)),
            "sum_harm": float(np.sum(harms)), "positive_fraction": pos_count / len(entries),
        }
        print(f"  {comp:8s}: mean={np.mean(harms):+.4f}, sum={np.sum(harms):+.4f}, pos={pos_count}/{len(entries)}")

    vo = [e["harm_score"] for e in by_type.get("W_V", []) + by_type.get("W_O", [])]
    qk = [e["harm_score"] for e in by_type.get("W_Q", []) + by_type.get("W_K", [])]
    mlp = [e["harm_score"] for e in by_type.get("W_gate", []) + by_type.get("W_up", []) + by_type.get("W_down", [])]
    total = sum(max(0, h) for h in [r["harm_score"] for r in results])
    vo_pos = sum(max(0, h) for h in vo)
    qk_pos = sum(max(0, h) for h in qk)
    mlp_pos = sum(max(0, h) for h in mlp)

    print(f"\n  Total pos: {total:.4f}")
    print(f"  MLP: {mlp_pos:.4f} ({mlp_pos/total*100:.1f}%)")
    print(f"  V/O: {vo_pos:.4f} ({vo_pos/total*100:.1f}%)")
    print(f"  Q/K: {qk_pos:.4f} ({qk_pos/total*100:.1f}%)")

    from scipy.stats import mannwhitneyu
    stat, pval = mannwhitneyu(vo, qk, alternative='greater')
    print(f"  V/O > Q/K: p = {pval:.6f}")

    output = {
        "analysis": "Attribution via activation patching (float16, 50 examples)",
        "model_pair": model_name, "num_examples": len(examples),
        "baseline_loss": float(baseline_loss),
        "type_statistics": type_stats, "components": results,
    }
    safe_name = model_name.replace("/", "_").replace(" ", "_").lower()
    out_path = RESULTS_DIR / f"attribution_{safe_name}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    base_dir = sys.argv[1]
    it_dir = sys.argv[2]
    device = sys.argv[3] if len(sys.argv) > 3 else "cuda:0"
    model_name = sys.argv[4] if len(sys.argv) > 4 else "unknown"

    run_attribution(base_dir, it_dir, device=device, model_name=model_name)
