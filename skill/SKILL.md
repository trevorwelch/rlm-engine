---
name: rlm
description: >
  Analyze large files beyond context window limits using the Recursive Language
  Model pattern. Use when the user asks to "analyze a large file", "process a huge
  document", "find patterns in a big codebase", "review a large log file", or needs
  to work with content too large for direct context. Loads files into a sandboxed
  Python REPL and uses Gemini sub-LLM calls for parallel chunk analysis.
argument-hint: "<file_path> [query]"
user-invocable: true
allowed-tools: mcp__rlm-engine__*, Read, Glob, Grep, Write, Bash, WebSearch, WebFetch
---

# RLM — Recursive Language Model Analysis

You are using the RLM pattern to analyze large files. **You write Python code that runs in a sandboxed REPL; Gemini Flash handles semantic analysis of chunks.**

**Arguments received:** $ARGUMENTS

## Core Principle

**Code filters, sub-LLMs reason.** Use Python to slice, filter, regex, and structure the data. Use `llm_query`/`llm_query_batch` for semantic understanding of chunks. Never try to load the full file into an LLM prompt — that's the whole point of RLM.

## Phase 1 — Init + Recon

### 1a. Resolve sources
Parse the arguments to extract file path(s), URL(s), and/or query.

- **Local files**: If the path contains globs or wildcards, use Glob to resolve to absolute paths.
- **URLs/web research**: If the user provides a URL or asks about a web topic, use WebFetch/WebSearch to gather content, then write it to temp files with `Write` (e.g., `/tmp/rlm_web_<topic>.md`). Pass those temp files to `rlm_init` alongside any local files.
- **Mixed**: You can combine local files and web-fetched content in the same session.

### 1b. Initialize session
```
rlm_init(file_paths=["/absolute/path/to/file"], model="gemini-2.5-flash")
```
This returns a `session_id` and `context_info` with file sizes, line counts, and total chars.

### 1c. Recon — understand the file structure
Call `rlm_exec` with code to read and assess the file:
```python
with open(context_path) as f:
    content = f.read()
print(f"Total length: {len(content)} chars, {content.count(chr(10))} lines")
# Show first 500 chars to understand structure
print(content[:500])
# Show last 200 chars
print("---END---")
print(content[-200:])
```

For code files, also look for structural markers:
```python
import re
classes = re.findall(r'^class (\w+)', content, re.MULTILINE)
functions = re.findall(r'^def (\w+)', content, re.MULTILINE)
print(f"Classes: {classes}")
print(f"Top-level functions: {functions}")
```

## Phase 2 — Filter + Analyze (iterate as needed)

### 2a. Design chunking strategy
Based on recon, choose how to split. **Always chunk by document structure**, not arbitrary byte offsets:

- **Python/code**: Split by class or function definitions
- **Logs**: Split by timestamp boundaries or fixed line counts (500-1000 lines)
- **Markdown/docs**: Split by heading levels (`## ` or `# `)
- **JSON**: Split by top-level array elements or object keys
- **Generic text**: Split by paragraph boundaries (double newlines)

### 2b. Filter first, then analyze
Use Python to eliminate irrelevant sections before sending to the sub-LLM:
```python
import re

# Example: find relevant sections with regex
sections = re.split(r'\nclass ', content)
relevant = [s for s in sections if 'pattern_keyword' in s.lower()]
print(f"Found {len(relevant)} relevant sections out of {len(sections)}")
```

### 2c. Parallel sub-LLM analysis
**Always use `llm_query_batch` over sequential `llm_query` when processing 2+ chunks.**

```python
# Build prompts for each chunk
prompts = []
for i, chunk in enumerate(chunks):
    prompts.append(f"""Analyze this code section and answer: {user_query}

Section {i+1}/{len(chunks)}:
{chunk}

Provide a concise, structured answer.""")

# Parallel execution — all chunks analyzed simultaneously
results = llm_query_batch(prompts)

# Store for aggregation
chunk_results = list(zip(range(len(chunks)), results))
for i, result in chunk_results:
    print(f"--- Chunk {i+1} ---")
    print(result[:500])  # Preview
```

### Chunk sizing guidelines
- Aim for **10K-50K+ chars per chunk** (Gemini Flash handles 1M tokens)
- Target **5-15 chunks** per analysis pass
- Too many tiny chunks = noisy results; too few huge chunks = less focus
- If a single chunk exceeds 200K chars, consider sub-splitting it

### Iteration
If results are incomplete or need refinement:
- Drill deeper into specific chunks
- Filter based on first-pass results
- Adjust prompts for more specificity
- Variables persist — never re-read files or redo work!

## Phase 3 — Aggregate + Answer

### 3a. Combine results
```python
# Aggregate all chunk results
combined = "\n\n".join(results)

# Optional: final synthesis via sub-LLM
final = llm_query(f"""Based on these analysis results from multiple sections of a file,
provide a comprehensive answer to: {user_query}

Individual section analyses:
{combined}

Synthesize into a clear, organized answer.""")

print(final)
```

### 3b. Present to user
Read the final output and present it to the user in a well-organized format. Add your own interpretation and context where helpful.

### 3c. Cleanup
```
rlm_cleanup(session_id)
```

## Key Rules

1. **Variables persist** — Data stored in one `rlm_exec` call is available in the next. Never re-read files.
2. **Batch over sequential** — Always prefer `llm_query_batch([...])` over multiple `llm_query()` calls.
3. **Filter before analyzing** — Use Python regex/keywords to narrow down before sending to Gemini.
4. **Chunk by structure** — Split on class/function/heading boundaries, not byte counts.
5. **Don't over-chunk** — 5-15 focused chunks beats 100 tiny ones.
6. **Truncation is OK** — REPL output is capped at 12K chars. Use variables to store full results and access them across calls.
7. **Always cleanup** — Call `rlm_cleanup` when done.

## Available REPL Functions

| Function | Description |
|----------|-------------|
| `llm_query(prompt)` | Single Gemini completion — use for synthesis or one-off analysis |
| `llm_query_batch(prompts)` | Parallel Gemini calls — use for analyzing multiple chunks |

## Available REPL Variables

| Variable | Description |
|----------|-------------|
| `context_path` | Path to the concatenated context file |
| `scratch_dir` | Temp directory for scratch files |

Standard Python imports are available (re, json, os, collections, itertools, etc.). Network modules (subprocess, socket, http) are blocked.

See the reference guide for detailed examples and tuning.
