---
active: true
phase: publish
cycle: 2
max_cycles: 5
max_retries: 3
builder_model: opus
reviewer_model: sonnet
spec_name: cleanup-and-robustness
repo_path: /Users/tw/coding/rlm-engine
worktree_path: /Users/tw/coding/rlm-engine/.claude/worktrees/theodore-cleanup-and-robustness-20260409-125012
branch_name: theodore/cleanup-and-robustness-20260409-125012
pr_number: null
pr_url: null
test_command: uv run pytest
started_at: "2026-04-09T19:51:00Z"
---

## Spec

# RLM Engine: Cleanup & Robustness

## Overview

The rlm-engine MCP server works but has several code quality and robustness issues that should be fixed. This is a cleanup pass — no new features, just making the existing code correct and resilient.

## Changes

### 1. Rename gemini_client.py → llm_client.py

The file provides a generic OpenAI-compatible client for local mlx-lm servers. The "gemini" name is a leftover from an earlier iteration. Rename the file and update all imports.

### 2. Fix model resolution

`_resolve_model()` in the client module caches the model name in a module-level global (`_model_name`) and never invalidates it. If the mlx-lm server is restarted with a different model, the stale name is used for all subsequent calls.

Fix: query `/v1/models` on each call (it's a cheap local HTTP call) rather than caching forever. Remove the global cache.

Additionally, if the server query fails, raise a `RuntimeError` instead of silently falling back to a bogus model name. The caller needs to know the server is unreachable.

### 3. Remove duplicate imports in server.py

`shutil` is imported at the top of server.py and again inside `_cleanup_expired()` and `rlm_cleanup()`. Remove the redundant inline imports.

### 4. Add thread safety to session management

`_sessions` dict in server.py has no synchronization. Concurrent MCP tool calls could race on session creation, lookup, and cleanup. Add a `threading.Lock` to protect all `_sessions` access.

### 5. Improve batch timeout handling

`llm_completion_batch()` uses `as_completed(..., timeout=CALL_TIMEOUT)` where `CALL_TIMEOUT=120`. This caps the *entire batch* at 120s regardless of batch size. For 18+ prompts on a local model, this is too tight.

Fix: scale the timeout based on batch size. Use `CALL_TIMEOUT * max(1, len(prompts) // 4)` as the batch timeout, with a per-prompt timeout remaining at `CALL_TIMEOUT`. Remove the redundant per-future `timeout` in `future.result()` since `as_completed` already enforces the outer timeout.

### 6. Non-blocking server startup feedback

`_ensure_mlx_server()` polls in a blocking loop for up to 60s with no indication of progress. Add logging via `print()` to stderr so the MCP client can see startup progress (e.g., "Waiting for mlx-lm server to start..." and "mlx-lm server ready on port XXXX").

## Out of Scope

- New features or tools
- Changes to the REPL sandbox security model
- Changes to the MCP protocol or tool signatures (except the model parameter default which was already fixed)

## Builder Study

### Project Overview
- **Language**: Python 3.10+
- **Frameworks**: `fastmcp >= 2.0.0, < 3` (MCP server), `openai >= 1.0.0` (local LLM client)

### Directory Structure
```
/
├── server.py          # MCP server entry point; exposes rlm_init, rlm_exec, rlm_cleanup tools
├── repl_env.py        # Sandboxed Python REPL with persistent state across exec calls
├── gemini_client.py   # LLM client (OpenAI-compatible); llm_completion + llm_completion_batch
├── pyproject.toml     # Project metadata and dependencies
├── README.md
├── LICENSE
└── skill/
    ├── SKILL.md       # Claude Code skill definition
    └── reference.md   # Reference documentation
```

Flat module structure — three `.py` files form the entire runtime.

### Architecture
- **MCP Tool Server**: `server.py` uses `FastMCP` to expose three tools (`rlm_init`, `rlm_exec`, `rlm_cleanup`)
- **Session-based stateful REPL**: Sessions stored in module-level dict with 30-min idle timeout
- **Sandboxing**: `REPLEnv` restricts builtins and blocks dangerous imports
- **Concurrent sub-LLM calls**: `llm_completion_batch` uses `ThreadPoolExecutor` (max 8 workers)
- **Lazy singleton client**: Thread-safe OpenAI client initialization
- **Environment-driven config**: `RLM_MLX_MODEL`, `RLM_MLX_PORT`, `RLM_BASE_URL`, `RLM_API_KEY`

### Test Framework
- No existing test suite — tests must be created from scratch
- No test framework configured yet — pytest should be added as dev dependency
- Test command: `uv run pytest`

### Configuration
- `pyproject.toml`: project metadata, Python constraint, two runtime dependencies
- No linting/formatting config present

## Reviewer Study

### Code Style
- 4-space indentation, `snake_case` functions/variables, `UPPER_CASE` constants
- Private helpers prefixed with `_`
- Module-level and function-level docstrings with `Args:`/`Returns:` sections
- Modern Python type syntax (`str | None`, `list[str]`)

### Module Boundaries
Clean layering: `server.py` -> `repl_env.py` -> `gemini_client.py` with no upward imports.

### Error Handling
- MCP tools return `{"error": "..."}` dicts on failure (not exceptions)
- `gemini_client` returns `"[LLM_ERROR] ..."` strings on failure
- Cleanup uses `ignore_errors=True` and bare `except Exception: pass`
- Output truncated at `MAX_OUTPUT_CHARS = 12_000`

### Security
- REPL sandbox blocks dangerous builtins and network/system modules
- No auth on MCP tools — delegated to transport layer
- File reads use `errors="replace"` for encoding safety

### API Design
- Session-oriented MCP pattern: init → exec → cleanup
- All tools return plain dicts with inline error fields
- Sessions expire after 30 minutes of inactivity

### Dependencies
- Two runtime deps in pyproject.toml; `mlx_vlm` is implicit (subprocess)
- No dev dependencies declared

## Acceptance Criteria

### [FUNC-1] File renamed and importable as llm_client
When the worktree is inspected, `gemini_client.py` does not exist and `llm_client.py` exists in its place. Importing `llm_client` in the worktree's Python environment succeeds without error.
- Test approach: `assert not os.path.exists("gemini_client.py")` and `import llm_client` in a pytest fixture; grep confirms zero occurrences of `gemini_client` in all `.py` files.

### [FUNC-2] All imports reference llm_client
When any source file (server.py, repl_env.py, or any test) imports the client module, it uses `llm_client`, not `gemini_client`.
- Test approach: `grep -r "gemini_client" *.py` returns no matches in the worktree root.

### [FUNC-3] Model name resolved fresh on each call
When `_resolve_model()` is called twice and the mlx-lm server returns a different model name on the second call, the second call returns the new model name (not the cached one).
- Test approach: Unit test with a mock HTTP server that returns `model-a` then `model-b`; assert both return values differ and match the mock responses.

### [FUNC-4] Batch timeout scales with prompt count
When `llm_completion_batch()` is called with N prompts, the timeout passed to `as_completed` equals `CALL_TIMEOUT * max(1, N // 4)`, not a fixed 120s.
- Test approach: Unit test patching `concurrent.futures.as_completed`; assert the `timeout` kwarg equals `120 * max(1, N // 4)` for N=1, N=4, N=18.

### [FUNC-5] No duplicate shutil imports in server.py
When server.py is parsed, `import shutil` appears exactly once at module level and not inside any function body.
- Test approach: `ast.parse` server.py and assert no `Import` node for `shutil` exists inside a `FunctionDef` subtree.

### [FUNC-6] Sessions dict protected by a lock
When multiple threads call session-mutating MCP tools concurrently, no `RuntimeError` or data corruption occurs and all sessions are created correctly.
- Test approach: Integration test spawning 20 threads each calling `rlm_init`; assert all 20 session IDs are distinct and present in the final sessions dict.

### [FUNC-7] Startup progress printed to stderr
When `_ensure_mlx_server()` is called and the server is not yet ready, at least one progress message is printed to `sys.stderr` before the server becomes ready, and a "ready" message is printed once it responds.
- Test approach: Unit test capturing `sys.stderr` with a mock that simulates a slow server (fails first 2 polls, succeeds on 3rd); assert stderr contains both a waiting message and a ready message including the port number.

### [EDGE-1] Server unreachable raises RuntimeError
When `_resolve_model()` is called and the mlx-lm server is not running (connection refused), a `RuntimeError` is raised rather than returning a fallback or empty string.
- Test approach: Unit test pointing the client at a closed port; assert `pytest.raises(RuntimeError)`.

### [EDGE-2] Batch of 1 prompt uses minimum timeout
When `llm_completion_batch()` is called with a single prompt, the batch timeout is `CALL_TIMEOUT * 1` (not zero or fractional).
- Test approach: Covered by FUNC-4 N=1 case; explicitly assert `timeout == 120`.

### [QUAL-1] Test suite passes
All tests in the worktree pass with zero failures or errors.
- Check: `uv run pytest` exits with code 0.

### [QUAL-2] pytest added as dev dependency
The project's `pyproject.toml` lists pytest under `[dependency-groups]` or `[tool.uv.dev-dependencies]`.
- Check: `uv run pytest --version` succeeds without installing anything extra.

## Findings

None yet.

## Mutation Findings (Cycle 1)

[M1] testing/major server.py:90 -- Mutation survived: Removed `with _sessions_lock:` from `_get_session()` (lock bypass) -- The thread safety test (FUNC-6) only checks concurrent `rlm_init` calls but does not verify that `_get_session()` actually acquires the lock. Add a test that either inspects the function body via AST for lock usage or that races `_get_session` calls with session mutations to detect the missing lock.
