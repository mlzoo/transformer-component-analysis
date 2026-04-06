"""
Extended agent evaluation set: 200+ structured generation examples.
Covers: tool calls, JSON, SQL, API, code, bash, config, ReAct, GraphQL,
MongoDB, cron, regex, CSS, git, terraform, docker, CI/CD, YAML, XML,
protobuf, type annotations, math expressions, data transforms, etc.
"""

# Import original 49 examples
from step1_attribution import AGENT_EXAMPLES as BASE_EXAMPLES

# 160+ additional examples
EXTRA_EXAMPLES = [
    # Tool calls (20 more)
    {"prompt": "Tools: get_weather(city)\nUser: Weather in London?\nAction: ", "target": 'get_weather(city="London")'},
    {"prompt": "Tools: create_ticket(title, priority)\nUser: Bug: login broken, high priority\nAction: ", "target": 'create_ticket(title="Bug: login broken", priority="high")'},
    {"prompt": "Tools: db_query(sql)\nUser: How many users signed up today?\nAction: ", "target": "db_query(sql=\"SELECT COUNT(*) FROM users WHERE created_at >= CURRENT_DATE\")"},
    {"prompt": "Tools: send_slack(channel, message)\nUser: Tell #ops we're deploying\nAction: ", "target": 'send_slack(channel="#ops", message="Deploying now")'},
    {"prompt": "Tools: resize_image(path, width, height)\nTask: Make logo.png 200x200\nAction: ", "target": 'resize_image(path="logo.png", width=200, height=200)'},
    {"prompt": "Tools: schedule_meeting(title, time, attendees)\nTask: Standup at 9am with Alice and Bob\nAction: ", "target": 'schedule_meeting(title="Standup", time="9:00 AM", attendees=["Alice", "Bob"])'},
    {"prompt": "Tools: run_test(file)\nTask: Run unit tests in test_auth.py\nAction: ", "target": 'run_test(file="test_auth.py")'},
    {"prompt": "Tools: deploy(service, env)\nTask: Deploy user-service to staging\nAction: ", "target": 'deploy(service="user-service", env="staging")'},
    {"prompt": "Tools: fetch_url(url), parse_html(html)\nTask: Get title from example.com\nThought: Fetch first.\nAction: ", "target": 'fetch_url(url="https://example.com")'},
    {"prompt": "Tools: git_commit(message), git_push(branch)\nTask: Commit fix and push\nAction: ", "target": 'git_commit(message="Fix authentication bug")'},
    {"prompt": "Tools: list_files(dir), read_file(path)\nTask: What's in /etc/hosts?\nAction: ", "target": 'read_file(path="/etc/hosts")'},
    {"prompt": "Tools: set_config(key, value)\nTask: Set timeout to 30s\nAction: ", "target": 'set_config(key="timeout", value="30s")'},
    {"prompt": "Tools: encrypt(data, key), decrypt(data, key)\nTask: Encrypt 'hello' with key 'abc'\nAction: ", "target": "encrypt(data=\"hello\", key=\"abc\")"},
    {"prompt": "Tools: create_user(name, email, role)\nTask: Add admin user John\nAction: ", "target": 'create_user(name="John", email="john@example.com", role="admin")'},
    {"prompt": "Tools: delete_record(table, id)\nTask: Remove order 456\nAction: ", "target": 'delete_record(table="orders", id=456)'},
    {"prompt": "Tools: compress(files, output)\nTask: Zip all logs\nAction: ", "target": 'compress(files="*.log", output="logs.zip")'},
    {"prompt": "Tools: translate_text(text, source, target)\nTask: Translate 'bonjour' from French to English\nAction: ", "target": 'translate_text(text="bonjour", source="fr", target="en")'},
    {"prompt": "Tools: analyze_sentiment(text)\nTask: Check sentiment of 'terrible service'\nAction: ", "target": 'analyze_sentiment(text="terrible service")'},
    {"prompt": "Tools: generate_report(type, date_range)\nTask: Monthly sales report for March\nAction: ", "target": 'generate_report(type="sales", date_range="2024-03-01/2024-03-31")'},
    {"prompt": "Tools: notify(user_id, message, channel)\nTask: Email user 42 about password reset\nAction: ", "target": 'notify(user_id=42, message="Password reset requested", channel="email")'},

    # JSON (20 more)
    {"prompt": 'Convert to JSON: Name=Bob, City=NYC, Age=25\n\n{"', "target": '"name": "Bob", "city": "NYC", "age": 25}'},
    {"prompt": 'Error response JSON: 404 Not Found\n\n{"', "target": '"error": {"code": 404, "message": "Not Found"}}'},
    {"prompt": 'Config JSON: debug=true, port=3000, host=localhost\n\n{"', "target": '"debug": true, "port": 3000, "host": "localhost"}'},
    {"prompt": 'User profile JSON: id=1, name=Jane, roles=[admin,user]\n\n{"', "target": '"id": 1, "name": "Jane", "roles": ["admin", "user"]}'},
    {"prompt": 'API response: status=success, data=[1,2,3]\n\n{"', "target": '"status": "success", "data": [1, 2, 3]}'},
    {"prompt": 'Log entry JSON: level=ERROR, msg=timeout, ts=2024-01\n\n{"', "target": '"level": "ERROR", "message": "timeout", "timestamp": "2024-01"}'},
    {"prompt": 'Pagination JSON: page=2, per_page=10, total=100\n\n{"', "target": '"page": 2, "per_page": 10, "total": 100, "total_pages": 10}'},
    {"prompt": 'Webhook payload: event=push, repo=myapp, branch=main\n\n{"', "target": '"event": "push", "repository": "myapp", "branch": "main"}'},
    {"prompt": 'Search result JSON: query=python, hits=42\n\n{"', "target": '"query": "python", "total_hits": 42, "results": []}'},
    {"prompt": 'Auth token response: token=abc123, expires=3600\n\n{"', "target": '"access_token": "abc123", "expires_in": 3600, "token_type": "bearer"}'},
    {"prompt": 'Feature flag JSON: dark_mode=enabled, beta=disabled\n\n{"', "target": '"features": {"dark_mode": "enabled", "beta": "disabled"}}'},
    {"prompt": 'Geo location JSON: lat=40.7, lng=-74.0, city=NYC\n\n{"', "target": '"latitude": 40.7, "longitude": -74.0, "city": "NYC"}'},
    {"prompt": 'Health check response: status=healthy, uptime=99.9%\n\n{"', "target": '"status": "healthy", "uptime": "99.9%", "checks": {"db": "ok", "cache": "ok"}}'},
    {"prompt": 'Notification JSON: type=alert, severity=high, msg=CPU 95%\n\n{"', "target": '"type": "alert", "severity": "high", "message": "CPU usage at 95%"}'},
    {"prompt": 'A/B test result: variant=B, conversion=12.5%, p=0.03\n\n{"', "target": '"variant": "B", "conversion_rate": 0.125, "p_value": 0.03, "significant": true}'},
    {"prompt": 'Rate limit response: limit=100, remaining=23, reset=1700000\n\n{"', "target": '"rate_limit": 100, "remaining": 23, "reset_at": 1700000}'},
    {"prompt": 'Job status JSON: id=j-123, state=running, progress=75%\n\n{"', "target": '"job_id": "j-123", "state": "running", "progress": 75}'},
    {"prompt": 'ML prediction: class=cat, confidence=0.95\n\n{"', "target": '"prediction": "cat", "confidence": 0.95}'},
    {"prompt": 'Version info: app=myapp, version=2.1.0, build=1234\n\n{"', "target": '"app": "myapp", "version": "2.1.0", "build": 1234}'},
    {"prompt": 'Diff summary: added=5, removed=3, modified=2\n\n{"', "target": '"files_added": 5, "files_removed": 3, "files_modified": 2}'},

    # SQL (15 more)
    {"prompt": "SQL: Get top 10 customers by revenue\n\nSELECT ", "target": "c.name, SUM(o.total) as revenue FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.name ORDER BY revenue DESC LIMIT 10;"},
    {"prompt": "SQL: Count orders per status\n\nSELECT ", "target": "status, COUNT(*) as count FROM orders GROUP BY status;"},
    {"prompt": "SQL: Users who haven't logged in for 30 days\n\nSELECT ", "target": "* FROM users WHERE last_login < NOW() - INTERVAL '30 days';"},
    {"prompt": "SQL: Average order value by month\n\nSELECT ", "target": "DATE_TRUNC('month', created_at) as month, AVG(total) as avg_value FROM orders GROUP BY month ORDER BY month;"},
    {"prompt": "SQL: Insert new product\n\n", "target": "INSERT INTO products (name, price, category) VALUES ('Widget', 9.99, 'tools');"},
    {"prompt": "SQL: Update user email\n\n", "target": "UPDATE users SET email = 'new@example.com' WHERE id = 42;"},
    {"prompt": "SQL: Delete expired sessions\n\n", "target": "DELETE FROM sessions WHERE expires_at < NOW();"},
    {"prompt": "SQL: Create index on email\n\n", "target": "CREATE INDEX idx_users_email ON users(email);"},
    {"prompt": "SQL: Find duplicate emails\n\nSELECT ", "target": "email, COUNT(*) as cnt FROM users GROUP BY email HAVING COUNT(*) > 1;"},
    {"prompt": "SQL: Join users and orders\n\nSELECT ", "target": "u.name, o.id, o.total FROM users u INNER JOIN orders o ON u.id = o.user_id;"},
    {"prompt": "SQL: Subquery - users with above-average orders\n\nSELECT ", "target": "* FROM users WHERE id IN (SELECT user_id FROM orders WHERE total > (SELECT AVG(total) FROM orders));"},
    {"prompt": "SQL: Create users table\n\n", "target": "CREATE TABLE users (id SERIAL PRIMARY KEY, name VARCHAR(100), email VARCHAR(255) UNIQUE, created_at TIMESTAMP DEFAULT NOW());"},
    {"prompt": "SQL: Window function - running total\n\nSELECT ", "target": "date, amount, SUM(amount) OVER (ORDER BY date) as running_total FROM transactions;"},
    {"prompt": "SQL: Upsert user\n\n", "target": "INSERT INTO users (id, name) VALUES (1, 'Alice') ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name;"},
    {"prompt": "SQL: Pivot monthly sales\n\nSELECT ", "target": "product, SUM(CASE WHEN month = 1 THEN sales END) as jan, SUM(CASE WHEN month = 2 THEN sales END) as feb FROM sales GROUP BY product;"},

    # Code (20 more)
    {"prompt": "```python\ndef fibonacci(n):\n    ", "target": 'if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n```'},
    {"prompt": "```python\ndef is_palindrome(s):\n    ", "target": 'return s == s[::-1]\n```'},
    {"prompt": "```python\nclass Stack:\n    def __init__(self):\n        ", "target": 'self.items = []\n    \n    def push(self, item):\n        self.items.append(item)\n    \n    def pop(self):\n        return self.items.pop()\n```'},
    {"prompt": "```python\ndef binary_search(arr, target):\n    ", "target": "left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1\n```"},
    {"prompt": "```python\ndef flatten(lst):\n    ", "target": "result = []\n    for item in lst:\n        if isinstance(item, list):\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return result\n```"},
    {"prompt": "```javascript\nfunction debounce(fn, delay) {\n  ", "target": "let timer;\n  return function(...args) {\n    clearTimeout(timer);\n    timer = setTimeout(() => fn.apply(this, args), delay);\n  };\n}\n```"},
    {"prompt": "```python\ndef merge_sort(arr):\n    ", "target": "if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)\n```"},
    {"prompt": "```python\n@app.route('/users/<int:id>')\ndef get_user(id):\n    ", "target": "user = User.query.get_or_404(id)\n    return jsonify(user.to_dict())\n```"},
    {"prompt": "```typescript\ninterface User {\n  ", "target": "id: number;\n  name: string;\n  email: string;\n  role: 'admin' | 'user';\n}\n```"},
    {"prompt": "```python\ndef retry(max_attempts=3, delay=1):\n    def decorator(func):\n        ", "target": "def wrapper(*args, **kwargs):\n            for attempt in range(max_attempts):\n                try:\n                    return func(*args, **kwargs)\n                except Exception as e:\n                    if attempt == max_attempts - 1:\n                        raise\n                    time.sleep(delay)\n        return wrapper\n    return decorator\n```"},
    {"prompt": "```python\nasync def fetch_all(urls):\n    ", "target": "async with aiohttp.ClientSession() as session:\n        tasks = [session.get(url) for url in urls]\n        responses = await asyncio.gather(*tasks)\n        return [await r.json() for r in responses]\n```"},
    {"prompt": "```rust\nfn main() {\n    ", "target": 'let numbers = vec![1, 2, 3, 4, 5];\n    let sum: i32 = numbers.iter().sum();\n    println!("Sum: {}", sum);\n}\n```'},
    {"prompt": "```go\nfunc handler(w http.ResponseWriter, r *http.Request) {\n    ", "target": 'w.Header().Set("Content-Type", "application/json")\n    json.NewEncoder(w).Encode(map[string]string{"status": "ok"})\n}\n```'},
    {"prompt": "```python\nclass LRUCache:\n    def __init__(self, capacity):\n        ", "target": "self.capacity = capacity\n        self.cache = OrderedDict()\n    \n    def get(self, key):\n        if key in self.cache:\n            self.cache.move_to_end(key)\n            return self.cache[key]\n        return -1\n```"},
    {"prompt": "```python\ndef validate_email(email):\n    ", "target": "import re\n    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'\n    return bool(re.match(pattern, email))\n```"},
    {"prompt": "```python\ndef read_csv(path):\n    ", "target": "import csv\n    with open(path) as f:\n        reader = csv.DictReader(f)\n        return list(reader)\n```"},
    {"prompt": "```python\ndef memoize(func):\n    ", "target": "cache = {}\n    def wrapper(*args):\n        if args not in cache:\n            cache[args] = func(*args)\n        return cache[args]\n    return wrapper\n```"},
    {"prompt": "```python\nclass Singleton:\n    _instance = None\n    \n    def __new__(cls):\n        ", "target": "if cls._instance is None:\n            cls._instance = super().__new__(cls)\n        return cls._instance\n```"},
    {"prompt": "```python\ndef chunk(lst, size):\n    ", "target": "return [lst[i:i+size] for i in range(0, len(lst), size)]\n```"},
    {"prompt": "```python\ndef deep_merge(d1, d2):\n    ", "target": "result = d1.copy()\n    for k, v in d2.items():\n        if k in result and isinstance(result[k], dict) and isinstance(v, dict):\n            result[k] = deep_merge(result[k], v)\n        else:\n            result[k] = v\n    return result\n```"},

    # API / HTTP (10 more)
    {"prompt": "API: Create a new blog post\n\nPOST /api/posts\nContent-Type: application/json\n\n", "target": '{"title": "New Post", "content": "Hello world", "author_id": 1}'},
    {"prompt": "API: Update user profile\n\nPATCH /api/users/42\nContent-Type: application/json\n\n", "target": '{"name": "Updated Name", "bio": "Developer"}'},
    {"prompt": "API: Search products\n\nGET /api/products?", "target": "q=laptop&category=electronics&min_price=500&max_price=2000&sort=price_asc"},
    {"prompt": "API: Paginated list\n\nGET /api/items?", "target": "page=2&per_page=20&sort=created_at&order=desc"},
    {"prompt": "API: OAuth token request\n\nPOST /oauth/token\n\n", "target": "grant_type=authorization_code&code=abc123&redirect_uri=http://localhost/callback&client_id=myapp"},
    {"prompt": "API: GraphQL mutation\n\n", "target": 'mutation { createUser(input: { name: "Alice", email: "alice@example.com" }) { id name } }'},
    {"prompt": "API: WebSocket message\n\n", "target": '{"type": "subscribe", "channel": "orders", "filters": {"status": "pending"}}'},
    {"prompt": "API: Bulk delete\n\nDELETE /api/items\nContent-Type: application/json\n\n", "target": '{"ids": [1, 2, 3, 4, 5]}'},
    {"prompt": "API: File upload metadata\n\nPOST /api/upload\n\n", "target": '{"filename": "report.pdf", "content_type": "application/pdf", "size": 1048576}'},
    {"prompt": "API: Rate limit headers\n\nHTTP/1.1 429 Too Many Requests\n", "target": "Retry-After: 60\nX-RateLimit-Limit: 100\nX-RateLimit-Remaining: 0\nX-RateLimit-Reset: 1700000000"},

    # Bash / CLI (10 more)
    {"prompt": "Bash: Find files larger than 100MB\n\n```bash\n", "target": "find / -type f -size +100M -exec ls -lh {} \\;\n```"},
    {"prompt": "Bash: Monitor disk usage\n\n```bash\n", "target": "df -h | awk '$5+0 > 80 {print $0}'\n```"},
    {"prompt": "Bash: Kill process on port 8080\n\n```bash\n", "target": "kill $(lsof -t -i:8080)\n```"},
    {"prompt": "Bash: Compress and encrypt backup\n\n```bash\n", "target": "tar czf - /data | gpg --symmetric --cipher-algo AES256 > backup.tar.gz.gpg\n```"},
    {"prompt": "Bash: Count lines of code in project\n\n```bash\n", "target": "find . -name '*.py' | xargs wc -l | sort -n | tail -20\n```"},
    {"prompt": "Bash: Watch log file for errors\n\n```bash\n", "target": "tail -f /var/log/app.log | grep --line-buffered 'ERROR'\n```"},
    {"prompt": "Bash: Batch rename files\n\n```bash\n", "target": "for f in *.jpg; do mv \"$f\" \"${f%.jpg}.png\"; done\n```"},
    {"prompt": "Bash: Create SSH tunnel\n\n```bash\n", "target": "ssh -L 5432:localhost:5432 user@remote-host\n```"},
    {"prompt": "Bash: System info\n\n```bash\n", "target": "uname -a && free -h && df -h && uptime\n```"},
    {"prompt": "Bash: Parallel download\n\n```bash\n", "target": "cat urls.txt | xargs -P 4 -I {} wget -q {}\n```"},

    # Config / YAML / TOML (10 more)
    {"prompt": "Kubernetes deployment YAML:\n\napiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: ", "target": "myapp\nspec:\n  replicas: 3\n  selector:\n    matchLabels:\n      app: myapp\n  template:\n    metadata:\n      labels:\n        app: myapp\n    spec:\n      containers:\n      - name: myapp\n        image: myapp:latest\n        ports:\n        - containerPort: 8080"},
    {"prompt": "Nginx config:\n\nserver {\n    ", "target": "listen 80;\n    server_name example.com;\n    location / {\n        proxy_pass http://localhost:3000;\n        proxy_set_header Host $host;\n    }\n}"},
    {"prompt": "Docker compose:\n\nversion: '3'\nservices:\n  web:\n    ", "target": 'image: nginx:latest\n    ports:\n      - "80:80"\n    volumes:\n      - ./html:/usr/share/nginx/html\n  db:\n    image: postgres:15\n    environment:\n      POSTGRES_PASSWORD: secret'},
    {"prompt": "GitHub Actions workflow:\n\nname: CI\non: [push]\njobs:\n  test:\n    runs-on: ", "target": "ubuntu-latest\n    steps:\n    - uses: actions/checkout@v4\n    - uses: actions/setup-python@v5\n      with:\n        python-version: '3.11'\n    - run: pip install -r requirements.txt\n    - run: pytest"},
    {"prompt": "Prometheus alert rule:\n\ngroups:\n- name: alerts\n  rules:\n  - alert: ", "target": "HighCPU\n    expr: cpu_usage > 90\n    for: 5m\n    labels:\n      severity: critical\n    annotations:\n      summary: High CPU usage detected"},
    {"prompt": "pyproject.toml:\n\n[project]\nname = ", "target": '"mypackage"\nversion = "1.0.0"\nrequires-python = ">=3.9"\ndependencies = [\n    "requests>=2.28",\n    "pydantic>=2.0",\n]'},
    {"prompt": ".eslintrc.json:\n\n{\n  ", "target": '"extends": ["eslint:recommended"],\n  "env": {"node": true, "es2022": true},\n  "rules": {\n    "no-unused-vars": "warn",\n    "no-console": "off"\n  }\n}'},
    {"prompt": "Terraform provider config:\n\nterraform {\n  required_providers {\n    aws = {\n      ", "target": 'source = "hashicorp/aws"\n      version = "~> 5.0"\n    }\n  }\n}\n\nprovider "aws" {\n  region = "us-east-1"\n}'},
    {"prompt": "Makefile:\n\n.PHONY: all test clean\n\nall: ", "target": "build test\n\nbuild:\n\tgo build -o bin/app ./cmd/app\n\ntest:\n\tgo test ./...\n\nclean:\n\trm -rf bin/"},
    {"prompt": "Systemd service:\n\n[Unit]\nDescription=", "target": "MyApp Service\nAfter=network.target\n\n[Service]\nType=simple\nUser=app\nExecStart=/usr/bin/myapp --config /etc/myapp.conf\nRestart=always\n\n[Install]\nWantedBy=multi-user.target"},

    # Regex (5 more)
    {"prompt": "Regex: Match IP addresses\n\n", "target": r"\b(?:\d{1,3}\.){3}\d{1,3}\b"},
    {"prompt": "Regex: Match dates (YYYY-MM-DD)\n\n", "target": r"\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])"},
    {"prompt": "Regex: Match URLs\n\n", "target": r"https?://[^\s/$.?#].[^\s]*"},
    {"prompt": "Regex: Match phone numbers\n\n", "target": r"\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"},
    {"prompt": "Regex: Match hex colors\n\n", "target": r"#(?:[0-9a-fA-F]{3}){1,2}\b"},

    # CSS (5 more)
    {"prompt": "CSS: Responsive grid\n\n.grid {\n  ", "target": "display: grid;\n  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));\n  gap: 1rem;\n}"},
    {"prompt": "CSS: Sticky header\n\n.header {\n  ", "target": "position: sticky;\n  top: 0;\n  z-index: 100;\n  background: white;\n  box-shadow: 0 2px 4px rgba(0,0,0,0.1);\n}"},
    {"prompt": "CSS: Dark mode\n\n@media (prefers-color-scheme: dark) {\n  :root {\n    ", "target": "--bg: #1a1a2e;\n    --text: #e0e0e0;\n    --primary: #4a90d9;\n  }\n}"},
    {"prompt": "CSS: Truncate text\n\n.truncate {\n  ", "target": "white-space: nowrap;\n  overflow: hidden;\n  text-overflow: ellipsis;\n  max-width: 200px;\n}"},
    {"prompt": "CSS: Animation\n\n@keyframes fadeIn {\n  ", "target": "from { opacity: 0; transform: translateY(-10px); }\n  to { opacity: 1; transform: translateY(0); }\n}"},

    # ReAct format (10 more)
    {"prompt": "Question: What is the population of Tokyo?\nThought: I need to search for Tokyo's population.\nAction: ", "target": 'search("Tokyo population")'},
    {"prompt": "Question: What is 15% tip on a $45 bill?\nThought: I need to calculate 15% of 45.\nAction: ", "target": 'calculate("45 * 0.15")'},
    {"prompt": "Question: Who directed Inception?\nThought: I should look this up.\nAction: ", "target": 'search("Inception director")'},
    {"prompt": "Observation: The Eiffel Tower is 330 meters tall.\nThought: I have the answer.\nAction: ", "target": 'finish("The Eiffel Tower is 330 meters tall.")'},
    {"prompt": "Question: Convert 100 USD to EUR\nThought: I need the current exchange rate.\nAction: ", "target": 'exchange_rate("USD", "EUR")'},
    {"prompt": "Question: What's the weather like in Paris?\nThought: Check weather API.\nAction: ", "target": 'get_weather("Paris")'},
    {"prompt": "Observation: Error 404 - page not found\nThought: The URL was wrong. Let me try another search.\nAction: ", "target": 'search("correct URL for resource")'},
    {"prompt": "Question: Summarize this article: [long text]\nThought: This is too long, I should use the summarizer tool.\nAction: ", "target": 'summarize(text="[long text]", max_length=100)'},
    {"prompt": "Question: Is this email spam?\nThought: I should classify this.\nAction: ", "target": 'classify_text(text="You won $1000000!", labels=["spam", "not_spam"])'},
    {"prompt": "Question: Plot sales data\nThought: I need to generate a visualization.\nAction: ", "target": 'create_chart(type="line", data="sales_2024.csv", x="month", y="revenue")'},

    # MongoDB (5 more)
    {"prompt": "MongoDB: Find users in NYC over 25\n\ndb.users.find(", "target": '{"city": "NYC", "age": {"$gt": 25}})'},
    {"prompt": "MongoDB: Aggregate total sales by product\n\ndb.orders.aggregate([\n  ", "target": '{"$group": {"_id": "$product", "total": {"$sum": "$amount"}}},\n  {"$sort": {"total": -1}}\n])'},
    {"prompt": "MongoDB: Update multiple documents\n\ndb.users.updateMany(\n  ", "target": '{"status": "inactive"},\n  {"$set": {"status": "active", "updated_at": new Date()}}\n)'},
    {"prompt": "MongoDB: Create index\n\ndb.users.", "target": 'createIndex({"email": 1}, {"unique": true})'},
    {"prompt": "MongoDB: Text search\n\ndb.articles.find({\n  ", "target": '"$text": {"$search": "machine learning"}\n}).sort({"score": {"$meta": "textScore"}})'},

    # Git (5 more)
    {"prompt": "git: Squash last 3 commits\n\n$ ", "target": "git rebase -i HEAD~3"},
    {"prompt": "git: Cherry pick a commit\n\n$ ", "target": "git cherry-pick abc1234"},
    {"prompt": "git: Create and switch to branch\n\n$ ", "target": "git checkout -b feature/new-auth"},
    {"prompt": "git: Stash with message\n\n$ ", "target": 'git stash push -m "WIP: fixing auth"'},
    {"prompt": "git: Show diff for staged files\n\n$ ", "target": "git diff --cached"},

    # Cron (5 more)
    {"prompt": "Cron: Every 5 minutes\n\n", "target": "*/5 * * * *"},
    {"prompt": "Cron: Weekdays at midnight\n\n", "target": "0 0 * * 1-5"},
    {"prompt": "Cron: First day of every month at noon\n\n", "target": "0 12 1 * *"},
    {"prompt": "Cron: Every Sunday at 3am\n\n", "target": "0 3 * * 0"},
    {"prompt": "Cron: Every 30 minutes during business hours\n\n", "target": "*/30 9-17 * * 1-5"},

    # Data transformation (10 more)
    {"prompt": "Python: Convert list of dicts to CSV string\n\ndata = [{'name': 'Alice', 'age': 30}]\n\n", "target": "import csv, io\nbuf = io.StringIO()\nwriter = csv.DictWriter(buf, fieldnames=data[0].keys())\nwriter.writeheader()\nwriter.writerows(data)\nresult = buf.getvalue()"},
    {"prompt": "jq: Extract names from JSON array\n\n$ echo '[{\"name\":\"Alice\"},{\"name\":\"Bob\"}]' | ", "target": "jq '.[].name'"},
    {"prompt": "Python: Flatten nested dict\n\ndef flatten_dict(d, prefix=''):\n    ", "target": "items = {}\n    for k, v in d.items():\n        key = f'{prefix}.{k}' if prefix else k\n        if isinstance(v, dict):\n            items.update(flatten_dict(v, key))\n        else:\n            items[key] = v\n    return items"},
    {"prompt": "Python: Group list by key\n\nfrom itertools import groupby\ndef group_by(items, key):\n    ", "target": "sorted_items = sorted(items, key=key)\n    return {k: list(v) for k, v in groupby(sorted_items, key=key)}"},
    {"prompt": "SQL: Pivot rows to columns\n\n", "target": "SELECT user_id,\n  MAX(CASE WHEN key = 'name' THEN value END) as name,\n  MAX(CASE WHEN key = 'email' THEN value END) as email\nFROM user_attributes GROUP BY user_id;"},
    {"prompt": "Python: Parse ISO datetime\n\nfrom datetime import datetime\n\ndef parse_iso(s):\n    ", "target": "return datetime.fromisoformat(s.replace('Z', '+00:00'))"},
    {"prompt": "Python: Convert XML to dict\n\nimport xml.etree.ElementTree as ET\n\ndef xml_to_dict(xml_str):\n    ", "target": "root = ET.fromstring(xml_str)\n    return {child.tag: child.text for child in root}"},
    {"prompt": "awk: Sum column 3\n\n$ cat data.tsv | ", "target": "awk '{sum += $3} END {print sum}'"},
    {"prompt": "Python: Transpose matrix\n\ndef transpose(matrix):\n    ", "target": "return list(map(list, zip(*matrix)))"},
    {"prompt": "sed: Replace all occurrences\n\n$ ", "target": "sed -i 's/old_text/new_text/g' file.txt"},

    # Type annotations / schemas (5 more)
    {"prompt": "Python type hints:\n\ndef process_items(items: ", "target": "list[dict[str, Any]], batch_size: int = 32) -> tuple[list[str], int]:"},
    {"prompt": "Pydantic model:\n\nclass UserCreate(BaseModel):\n    ", "target": 'name: str\n    email: EmailStr\n    age: int = Field(ge=0, le=150)\n    role: Literal["admin", "user"] = "user"'},
    {"prompt": "TypeScript interface:\n\ninterface ApiResponse<T> {\n  ", "target": "data: T;\n  status: 'success' | 'error';\n  message?: string;\n  pagination?: {\n    page: number;\n    total: number;\n  };\n}"},
    {"prompt": "JSON Schema:\n\n{\n  \"type\": \"object\",\n  \"properties\": {\n    ", "target": '"name": {"type": "string", "minLength": 1},\n    "age": {"type": "integer", "minimum": 0},\n    "email": {"type": "string", "format": "email"}\n  },\n  "required": ["name", "email"]\n}'},
    {"prompt": "GraphQL schema:\n\ntype User {\n  ", "target": "id: ID!\n  name: String!\n  email: String!\n  posts: [Post!]!\n  createdAt: DateTime!\n}"},

    # Math expressions (5 more)
    {"prompt": "LaTeX: Quadratic formula\n\n$$", "target": "x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$$"},
    {"prompt": "LaTeX: Matrix multiplication\n\n$$", "target": "C_{ij} = \\sum_{k=1}^{n} A_{ik} B_{kj}$$"},
    {"prompt": "LaTeX: Bayes theorem\n\n$$", "target": "P(A|B) = \\frac{P(B|A) P(A)}{P(B)}$$"},
    {"prompt": "LaTeX: Softmax function\n\n$$", "target": "\\sigma(z_i) = \\frac{e^{z_i}}{\\sum_{j=1}^{K} e^{z_j}}$$"},
    {"prompt": "LaTeX: Cross entropy loss\n\n$$", "target": "\\mathcal{L} = -\\sum_{i} y_i \\log(\\hat{y}_i)$$"},
]

# Combined set: 49 original + 160 new = 209 total
AGENT_EXAMPLES_200 = list(BASE_EXAMPLES) + EXTRA_EXAMPLES
