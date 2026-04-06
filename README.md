# RLM Engine

Analyze files beyond Claude's context window using the Recursive Language Model pattern.

RLM loads files into a sandboxed Python REPL and uses parallel Gemini Flash calls to reason about chunks. Claude writes the Python code that filters, slices, and structures the data; Gemini handles the semantic analysis. The results flow back to Claude for synthesis.

**Code filters, sub-LLMs reason.**

## Why

Claude's context window has limits. A 500K-line log, a 10MB codebase, or a massive JSON dump won't fit. RLM keeps the full content in a sandbox and only brings summarized results back into context. Gemini Flash (1M token context, ~$0.01-0.50 per analysis) does the heavy lifting on individual chunks in parallel.

## How It Works

```
┌─────────────┐     ┌───────────────────────────────────────────┐
│  Claude      │     │  RLM Engine (MCP Server)                  │
│              │     │                                           │
│  1. Init     │────>│  Load files into sandboxed Python REPL    │
│  2. Write    │────>│  Execute Python: filter, chunk, regex     │
│     Python   │     │  ┌─────────────────────────────────────┐  │
│  3. Get      │<────│  │ llm_query_batch() ──> Gemini Flash  │  │
│     results  │     │  │  chunk 1 ─────────>  analysis 1     │  │
│  4. Synthesize│    │  │  chunk 2 ─────────>  analysis 2     │  │
│  5. Cleanup  │────>│  │  chunk N ─────────>  analysis N     │  │
│              │     │  └─────────────────────────────────────┘  │
└─────────────┘     └───────────────────────────────────────────┘
```

Three tools, three phases:

| Tool | Purpose |
|------|---------|
| `rlm_init(file_paths)` | Load files, create REPL session |
| `rlm_exec(session_id, code)` | Run Python in sandbox (variables persist across calls) |
| `rlm_cleanup(session_id)` | Tear down session, free resources |

The REPL injects two functions for sub-LLM calls:
- `llm_query(prompt)` — single Gemini completion
- `llm_query_batch(prompts)` — parallel Gemini completions (up to 8 concurrent)

## Install

RLM Engine has two parts: an **MCP server** (the execution engine) and a **skill** (teaches Claude how to use it). Both install from this repo.

### Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python package manager)
- A [Gemini API key](https://aistudio.google.com/apikey)

### 1. Clone

```bash
git clone https://github.com/trevorwelch/rlm-engine.git
cd rlm-engine
```

### 2. Register the MCP server

```bash
claude mcp add -s user rlm-engine \
  -e GEMINI_API_KEY=your-key-here \
  -- uv run --directory /path/to/rlm-engine python server.py
```

Replace `/path/to/rlm-engine` with the absolute path where you cloned the repo.

### 3. Install the skill

```bash
cp -r skill/SKILL.md ~/.claude/skills/rlm/SKILL.md
cp -r skill/reference.md ~/.claude/skills/rlm/reference.md
```

Or symlink if you want updates from git pulls:

```bash
mkdir -p ~/.claude/skills/rlm
ln -sf /path/to/rlm-engine/skill/SKILL.md ~/.claude/skills/rlm/SKILL.md
ln -sf /path/to/rlm-engine/skill/reference.md ~/.claude/skills/rlm/reference.md
```

### 4. Verify

In Claude Code, run:

```
/rlm /path/to/some/large/file.py "summarize this"
```

## Usage

Invoke with `/rlm` followed by a file path and optional query:

```
/rlm ~/projects/app/models.py "find design patterns"
/rlm ~/logs/production.log "find all errors related to auth"
/rlm ~/data/export.json "summarize the top trends"
```

Claude will automatically:
1. Load the file(s) into a sandboxed REPL
2. Analyze the structure (classes, functions, headings, etc.)
3. Chunk by document structure (not arbitrary byte offsets)
4. Send chunks to Gemini Flash in parallel
5. Aggregate and synthesize results
6. Clean up the session

### Multiple files

```
/rlm ~/project/src/*.py "find all API endpoints and their auth requirements"
```

### Web content

```
/rlm https://example.com/docs "summarize the API documentation"
```

The skill will fetch the content, save to a temp file, and process it through the same pipeline.

## Architecture

```
rlm-engine/
├── server.py          # FastMCP server — 3 tools (init, exec, cleanup)
├── repl_env.py        # Sandboxed Python REPL with persistent variables
├── gemini_client.py   # Thread-safe Gemini API client with batch support
├── pyproject.toml     # Dependencies: google-genai, fastmcp
└── skill/
    ├── SKILL.md       # Orchestration guide (teaches Claude the 3-phase workflow)
    └── reference.md   # Examples, chunking strategies, cost estimates
```

### Security

The REPL sandbox blocks dangerous modules: `subprocess`, `os`, `pathlib`, `socket`, `http`, `urllib`, `shutil`, `ctypes`, `multiprocessing`. Built-in functions `eval`, `exec`, `compile`, and `input` are also blocked. Output is capped at 12K chars per execution to prevent context blowout.

### Sessions

Sessions persist variables across `rlm_exec` calls and auto-expire after 30 minutes of inactivity. Always call `rlm_cleanup` when done.

## Cost

Gemini 2.5 Flash: $0.15/1M input tokens, $0.60/1M output tokens.

| File Size | Chunks | Estimated Cost |
|-----------|--------|---------------|
| 50K chars | 5 | ~$0.01 |
| 500K chars | 10 | ~$0.05 |
| 2M chars | 20 | ~$0.15 |
| 10M chars | 50 | ~$0.50 |

## License

MIT
