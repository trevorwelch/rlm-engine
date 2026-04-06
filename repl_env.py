"""
Python REPL environment for RLM sessions.

Provides a sandboxed REPL with persistent variables and injected
llm_query/llm_query_batch functions for sub-LLM calls.
"""

import ast
import io
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

from gemini_client import llm_completion, llm_completion_batch

MAX_OUTPUT_CHARS = 12_000


class REPLEnv:
    def __init__(self, context_path: str, model: str = "default_model"):
        self.context_path = context_path
        self.model = model
        self.temp_dir = tempfile.mkdtemp(prefix="rlm_")
        self._lock = threading.Lock()

        # Restricted builtins — block dangerous functions
        safe_builtins = {
            k: v
            for k, v in __builtins__.items()
            if k not in ("eval", "exec", "input", "compile", "breakpoint", "__import__")
        } if isinstance(__builtins__, dict) else {
            k: getattr(__builtins__, k)
            for k in dir(__builtins__)
            if k not in ("eval", "exec", "input", "compile", "breakpoint", "__import__")
            and not k.startswith("_")
        }

        # Allow __import__ with restrictions
        def _safe_import(name, *args, **kwargs):
            blocked = {
                "subprocess", "shutil", "socket", "http", "urllib",
                "ftplib", "smtplib", "os", "pathlib", "signal",
                "ctypes", "multiprocessing", "webbrowser",
            }
            if name in blocked or name.startswith(("http.", "urllib.")):
                raise ImportError(f"Module '{name}' is blocked in the RLM REPL for security")
            return __import__(name, *args, **kwargs)

        safe_builtins["__import__"] = _safe_import
        safe_builtins["print"] = print  # Ensure print is available
        safe_builtins["__name__"] = "__repl__"
        safe_builtins["__build_class__"] = __build_class__

        # Build llm_query wrappers that bind the model
        def llm_query(prompt: str) -> str:
            """Call local LLM on a text prompt. Use for semantic analysis of chunks."""
            return llm_completion(prompt, model=self.model)

        def llm_query_batch(prompts: list[str]) -> list[str]:
            """Parallel local LLM calls on multiple prompts. Use for batch chunk analysis."""
            return llm_completion_batch(prompts, model=self.model)

        self.globals = {
            "__builtins__": safe_builtins,
            "context_path": self.context_path,
            "scratch_dir": self.temp_dir,
            "llm_query": llm_query,
            "llm_query_batch": llm_query_batch,
        }
        self.locals = {}

    def execute(self, code: str) -> dict:
        """
        Execute Python code in the REPL. Variables persist across calls.
        Last expression is auto-printed (notebook-style).

        Returns {"stdout": str, "stderr": str, "execution_time": float}
        """
        with self._lock:
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            start = time.time()

            # Merge locals into globals for persistence
            combined = {**self.globals, **self.locals}

            try:
                tree = ast.parse(code)
            except SyntaxError as e:
                return {
                    "stdout": "",
                    "stderr": f"SyntaxError: {e}",
                    "execution_time": time.time() - start,
                }

            # Split into statements and possibly a final expression
            last_expr = None
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last_expr = tree.body.pop()

            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture

            try:
                # Execute all statements
                if tree.body:
                    compiled = compile(tree, "<rlm_repl>", "exec")
                    exec(compiled, combined)

                # Evaluate and print last expression
                if last_expr is not None:
                    expr_code = compile(
                        ast.Expression(body=last_expr.value),
                        "<rlm_repl>",
                        "eval",
                    )
                    result = eval(expr_code, combined)
                    if result is not None:
                        print(repr(result))

            except Exception as e:
                print(f"{type(e).__name__}: {e}", file=stderr_capture)
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

            # Update persistent locals (exclude builtins and injected functions)
            injected = {"__builtins__", "context_path", "scratch_dir", "llm_query", "llm_query_batch"}
            self.locals = {
                k: v for k, v in combined.items()
                if k not in injected and k not in self.globals
            }

            elapsed = time.time() - start
            stdout_text = stdout_capture.getvalue()
            stderr_text = stderr_capture.getvalue()

            # Truncate to prevent context blowout
            if len(stdout_text) > MAX_OUTPUT_CHARS:
                stdout_text = stdout_text[:MAX_OUTPUT_CHARS] + f"\n... [truncated, {len(stdout_capture.getvalue())} total chars]"
            if len(stderr_text) > MAX_OUTPUT_CHARS:
                stderr_text = stderr_text[:MAX_OUTPUT_CHARS] + f"\n... [truncated]"

            return {
                "stdout": stdout_text,
                "stderr": stderr_text,
                "execution_time": round(elapsed, 2),
            }

    def get_variable(self, name: str) -> str:
        """Retrieve a variable's string value from the REPL namespace."""
        combined = {**self.globals, **self.locals}
        if name not in combined:
            return f"[ERROR] Variable '{name}' not found"
        val = combined[name]
        text = str(val)
        if len(text) > MAX_OUTPUT_CHARS:
            text = text[:MAX_OUTPUT_CHARS] + f"\n... [truncated, {len(str(val))} total chars]"
        return text

    def cleanup(self):
        """Remove temp directory and reset state."""
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass
        self.locals = {}
