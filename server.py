"""
RLM Engine — MCP server for Recursive Language Model analysis.

Provides a sandboxed Python REPL with local LLM sub-calls (via mlx-lm)
for analyzing files that exceed context window limits.
"""

import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

from fastmcp import FastMCP

from repl_env import REPLEnv

MLX_MODEL = os.getenv("RLM_MLX_MODEL", "mlx-community/gemma-4-e2b-it-4bit")
MLX_PORT = int(os.getenv("RLM_MLX_PORT", "8080"))

mcp = FastMCP("rlm-engine")
_mlx_process: subprocess.Popen | None = None


def _mlx_server_running() -> bool:
    """Check if something is listening on the mlx-lm port."""
    try:
        with socket.create_connection(("127.0.0.1", MLX_PORT), timeout=1):
            return True
    except OSError:
        return False


def _ensure_mlx_server():
    """Start mlx-lm server if it's not already running."""
    global _mlx_process
    if _mlx_server_running():
        return

    print(f"Waiting for mlx-lm server to start...", file=sys.stderr)

    _mlx_process = subprocess.Popen(
        ["mlx_vlm.server", "--model", MLX_MODEL, "--port", str(MLX_PORT)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait for server to become ready
    for _ in range(60):
        time.sleep(1)
        if _mlx_server_running():
            print(f"mlx-lm server ready on port {MLX_PORT}", file=sys.stderr)
            return
    raise RuntimeError(f"mlx-lm server failed to start on port {MLX_PORT}")

# Session management
_sessions: dict[str, dict] = {}  # session_id -> {"env": REPLEnv, "last_used": float}
_sessions_lock = threading.Lock()
SESSION_TIMEOUT = 30 * 60  # 30 minutes


def _generate_session_id() -> str:
    return uuid.uuid4().hex[:8]


def _cleanup_expired():
    """Remove sessions that haven't been used in SESSION_TIMEOUT seconds."""
    now = time.time()
    expired = [
        sid for sid, s in _sessions.items()
        if now - s["last_used"] > SESSION_TIMEOUT
    ]
    for sid in expired:
        try:
            session = _sessions[sid]
            session["env"].cleanup()
            shutil.rmtree(session.get("temp_dir", ""), ignore_errors=True)
        except Exception:
            pass
        del _sessions[sid]


def _get_session(session_id: str) -> dict:
    """Get a session by ID, raising if not found."""
    with _sessions_lock:
        _cleanup_expired()
        if session_id not in _sessions:
            raise ValueError(f"Session '{session_id}' not found (expired or invalid)")
        session = _sessions[session_id]
        session["last_used"] = time.time()
        return session


@mcp.tool()
def rlm_init(
    file_paths: list[str],
    model: str | None = None,
) -> dict:
    """
    Initialize an RLM analysis session. Loads files into a Python REPL
    environment with llm_query() and llm_query_batch() available for
    sub-LLM calls on chunks.

    Args:
        file_paths: Absolute paths to files to analyze.
        model: Model name for sub-LLM calls (default: whatever mlx-lm is serving).

    Returns:
        session_id, context_info (file sizes, line counts, total chars).
    """
    with _sessions_lock:
        _cleanup_expired()

    _ensure_mlx_server()

    # Validate files exist
    files_info = []
    for fp in file_paths:
        p = Path(fp)
        if not p.exists():
            return {"error": f"File not found: {fp}"}
        if not p.is_file():
            return {"error": f"Not a file: {fp}"}

    # Concatenate all files into a single context file
    temp_dir = tempfile.mkdtemp(prefix="rlm_ctx_")
    context_path = os.path.join(temp_dir, "context.txt")
    total_chars = 0
    total_lines = 0

    with open(context_path, "w", encoding="utf-8", errors="replace") as out:
        for fp in file_paths:
            p = Path(fp)
            try:
                content = p.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                return {"error": f"Failed to read {fp}: {e}"}

            chars = len(content)
            lines = content.count("\n") + 1
            files_info.append({
                "path": fp,
                "name": p.name,
                "chars": chars,
                "lines": lines,
            })
            total_chars += chars
            total_lines += lines

            # Write with file header
            out.write(f"{'='*60}\n")
            out.write(f"FILE: {fp}\n")
            out.write(f"{'='*60}\n")
            out.write(content)
            if not content.endswith("\n"):
                out.write("\n")
            out.write("\n")

    # Create REPL environment
    env = REPLEnv(context_path=context_path, model=model)

    session_id = _generate_session_id()
    with _sessions_lock:
        _sessions[session_id] = {
            "env": env,
            "last_used": time.time(),
            "context_path": context_path,
            "temp_dir": temp_dir,
        }

    return {
        "session_id": session_id,
        "context_info": {
            "files": files_info,
            "total_chars": total_chars,
            "total_lines": total_lines,
            "context_path": context_path,
        },
    }


@mcp.tool()
def rlm_exec(
    session_id: str,
    code: str,
) -> dict:
    """
    Execute Python code in an RLM REPL session. Variables persist across
    calls. The code has access to:
    - context_path: path to the loaded file(s)
    - scratch_dir: temp directory for scratch files
    - llm_query(prompt): call local LLM on a text prompt (for analyzing chunks)
    - llm_query_batch(prompts): parallel local LLM calls on multiple prompts

    Args:
        session_id: Session from rlm_init.
        code: Python code to execute.

    Returns:
        stdout, stderr, execution_time.
    """
    try:
        session = _get_session(session_id)
    except ValueError as e:
        return {"error": str(e)}

    return session["env"].execute(code)


@mcp.tool()
def rlm_cleanup(session_id: str) -> dict:
    """Clean up an RLM session and free resources."""
    with _sessions_lock:
        if session_id not in _sessions:
            return {"status": "not_found", "message": f"Session '{session_id}' not found"}

        session = _sessions.pop(session_id)

    try:
        session["env"].cleanup()
        shutil.rmtree(session.get("temp_dir", ""), ignore_errors=True)
    except Exception:
        pass

    return {"status": "cleaned_up", "session_id": session_id}


if __name__ == "__main__":
    mcp.run()
