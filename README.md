# RLM Engine

Analyze files beyond Claude's context window using the Recursive Language Model pattern.

RLM loads files into a sandboxed Python REPL and uses parallel local LLM calls to reason about chunks. Claude writes the Python code that filters, slices, and structures the data; a local model (via mlx-lm) handles the semantic analysis. The results flow back to Claude for synthesis.

**Code filters, sub-LLMs reason.**

## How It Works

```
┌─────────────┐     ┌───────────────────────────────────────────┐
│  Claude      │     │  RLM Engine (MCP Server)                  │
│              │     │                                           │
│  1. Init     │────>│  Load files into sandboxed Python REPL    │
│  2. Write    │────>│  Execute Python: filter, chunk, regex     │
│     Python   │     │  ┌─────────────────────────────────────┐  │
│  3. Get      │<────│  │ llm_query_batch() ──> local model   │  │
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
- `llm_query(prompt)` — single completion
- `llm_query_batch(prompts)` — parallel completions (up to 8 concurrent)

## Install

### Prerequisites

- [uv](https://docs.astral.sh/uv/)
- [mlx-lm](https://github.com/ml-explore/mlx-lm) (`pip install mlx-lm`)

### 1. Clone

```bash
git clone https://github.com/trevorwelch/rlm-engine.git ~/coding/rlm-engine
```

### 2. Start mlx-lm server

```bash
mlx_lm.server --model mlx-community/Qwen2.5-Coder-7B-Instruct-8bit --port 8080
```

Or any model you prefer. The engine talks to it via the OpenAI-compatible API.

### 3. Register the MCP server

```bash
claude mcp add -s user rlm-engine \
  -- uv run --directory /path/to/rlm-engine python server.py
```

To use a different port or remote server, set `RLM_BASE_URL`:

```bash
claude mcp add -s user rlm-engine \
  -e RLM_BASE_URL=http://localhost:1234/v1 \
  -- uv run --directory /path/to/rlm-engine python server.py
```

### 4. Install the skill

```bash
mkdir -p ~/.claude/skills/rlm
ln -sf /path/to/rlm-engine/skill/SKILL.md ~/.claude/skills/rlm/SKILL.md
ln -sf /path/to/rlm-engine/skill/reference.md ~/.claude/skills/rlm/reference.md
```

### 5. Verify

```
/rlm ~/path/to/some/large/file.py "summarize this"
```

## Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `RLM_BASE_URL` | `http://localhost:8080/v1` | OpenAI-compatible API endpoint |
| `RLM_API_KEY` | `local` | API key (not needed for mlx-lm) |

Works with any OpenAI-compatible server: mlx-lm, Ollama, vLLM, llama.cpp, LM Studio, etc.

## License

MIT
