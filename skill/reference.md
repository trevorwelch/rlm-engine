# RLM Reference Guide

## Example Session: Analyzing a Large Django Codebase

```
User: /rlm ~/projects/myapp/models.py "find all design patterns and suggest improvements"
```

### Phase 1 — Init + Recon

**Call 1:** `rlm_init(file_paths=["/Users/me/projects/myapp/models.py"])`
Returns: `session_id: "a1b2c3d4"`, total: 45,000 chars, 1,200 lines

**Call 2:** `rlm_exec(session_id="a1b2c3d4", code=...)`
```python
with open(context_path) as f:
    content = f.read()

import re
classes = re.findall(r'^class (\w+)\(.*\):', content, re.MULTILINE)
print(f"Total: {len(content)} chars, {content.count(chr(10))} lines")
print(f"Classes ({len(classes)}): {classes}")

# Find class boundaries for chunking
class_positions = [(m.start(), m.group(1)) for m in re.finditer(r'^class \w+', content, re.MULTILINE)]
print(f"Class positions: {class_positions[:10]}")
```

### Phase 2 — Chunk + Analyze

**Call 3:** `rlm_exec(session_id="a1b2c3d4", code=...)`
```python
# Split by class definition
chunks = re.split(r'(?=^class \w+)', content, flags=re.MULTILINE)
chunks = [c for c in chunks if c.strip()]  # Remove empty
print(f"Chunks: {len(chunks)}, sizes: {[len(c) for c in chunks]}")

# Build analysis prompts
prompts = []
for i, chunk in enumerate(chunks):
    prompts.append(f"""Analyze this Django model code for design patterns and anti-patterns.

Identify:
1. Design patterns used (Repository, Factory, Observer, etc.)
2. Django-specific patterns (fat models, managers, signals, etc.)
3. Anti-patterns or code smells
4. Specific improvement suggestions with code examples

Code section {i+1}/{len(chunks)}:
```python
{chunk}
```

Be specific — reference class/method names.""")

# Parallel analysis
results = llm_query_batch(prompts)
for i, r in enumerate(results):
    print(f"=== Chunk {i+1} ===")
    print(r[:300])
```

### Phase 3 — Aggregate

**Call 4:** `rlm_exec(session_id="a1b2c3d4", code=...)`
```python
combined = "\n\n---\n\n".join([f"Section {i+1}:\n{r}" for i, r in enumerate(results)])

final = llm_query(f"""Synthesize these per-section analyses of a Django models.py file.

Create a comprehensive report with:
1. Overall architecture assessment
2. Design patterns found (with examples from the code)
3. Top 5 improvement recommendations, ordered by impact
4. Anti-patterns to address

Section analyses:
{combined}""")

print(final)
```

**Call 5:** `rlm_cleanup(session_id="a1b2c3d4")`

---

## Chunking Strategies by File Type

### Python Source Code
```python
import re
# Split by top-level class or function
chunks = re.split(r'(?=^(?:class |def )\w+)', content, flags=re.MULTILINE)
```

### Log Files
```python
# Split by date boundaries
chunks = re.split(r'(?=^\d{4}-\d{2}-\d{2})', content, flags=re.MULTILINE)
# Or fixed-size line chunks
lines = content.split('\n')
chunk_size = 500
chunks = ['\n'.join(lines[i:i+chunk_size]) for i in range(0, len(lines), chunk_size)]
```

### Markdown Documents
```python
# Split by top-level headings
chunks = re.split(r'(?=^# )', content, flags=re.MULTILINE)
# Or second-level headings for finer granularity
chunks = re.split(r'(?=^## )', content, flags=re.MULTILINE)
```

### JSON / JSONL
```python
import json
# For JSON arrays
data = json.loads(content)
chunk_size = 100
chunks = [json.dumps(data[i:i+chunk_size], indent=2) for i in range(0, len(data), chunk_size)]

# For JSONL
lines = content.strip().split('\n')
chunk_size = 200
chunks = ['\n'.join(lines[i:i+chunk_size]) for i in range(0, len(lines), chunk_size)]
```

### CSV / Tabular Data
```python
lines = content.split('\n')
header = lines[0]
data_lines = lines[1:]
chunk_size = 500
chunks = [header + '\n' + '\n'.join(data_lines[i:i+chunk_size])
          for i in range(0, len(data_lines), chunk_size)]
```

---

## Available Functions Reference

### `llm_query(prompt: str) -> str`
Single Gemini completion. Best for:
- Final synthesis of chunk results
- One-off analysis of a specific section
- Follow-up questions on earlier results

### `llm_query_batch(prompts: list[str]) -> list[str]`
Parallel Gemini completions (up to 8 concurrent). Best for:
- Analyzing multiple chunks simultaneously
- Processing multiple questions against the same data
- Any scenario with 2+ independent LLM calls

Returns results in the same order as input prompts. Failed calls return error strings prefixed with `[LLM_ERROR]`.

### Built-in Variables
- `context_path` — Absolute path to the concatenated context file
- `scratch_dir` — Temp directory for writing intermediate files

### Allowed Imports
Standard library modules: `re`, `json`, `os`, `collections`, `itertools`, `math`, `statistics`, `pathlib`, `textwrap`, `csv`, `datetime`, `functools`, etc.

**Blocked modules** (security): `subprocess`, `shutil`, `socket`, `http`, `urllib`, `ftplib`, `smtplib`

---

## When to Increase Iterations

- **First pass results are shallow** — Drill into specific chunks with more targeted prompts
- **File has nested structure** — Two-pass: first identify top-level structure, then analyze sub-sections
- **Multiple questions** — Separate analysis passes for each question, then synthesize
- **Results are contradictory** — Re-analyze conflicting chunks with more context

## Cost Estimates

Gemini 2.5 Flash pricing: $0.15/1M input, $0.60/1M output

| Scenario | Est. Cost |
|----------|-----------|
| 50K char file, 5 chunks | ~$0.01 |
| 500K char file, 10 chunks + synthesis | ~$0.05 |
| 2M char file, 20 chunks + 2 synthesis | ~$0.15 |
| 10M char multi-file, 50 chunks | ~$0.50 |

## Limitations (v1)

- **Depth 1 only** — Sub-LLM calls are plain completions (no nested REPLs)
- **No network access** from REPL (subprocess, socket blocked)
- **Text files only** — Binary, image, PDF files not supported in context
- **Output truncated** at 12K chars per exec call — use variables to store full results
- **Session timeout** — 30 minutes of inactivity
- **No streaming** — Results returned after full execution
