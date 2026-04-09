"""Tests for the cleanup-and-robustness spec."""

import ast
import os
import sys
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

WORKTREE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# [FUNC-1] File renamed and importable as llm_client
# ---------------------------------------------------------------------------


class TestFileRenamed:
    def test_gemini_client_does_not_exist(self):
        assert not os.path.exists(os.path.join(WORKTREE, "gemini_client.py"))

    def test_llm_client_exists(self):
        assert os.path.exists(os.path.join(WORKTREE, "llm_client.py"))

    def test_llm_client_importable(self):
        import llm_client
        assert hasattr(llm_client, "llm_completion")
        assert hasattr(llm_client, "llm_completion_batch")


# ---------------------------------------------------------------------------
# [FUNC-2] All imports reference llm_client
# ---------------------------------------------------------------------------


class TestNoGeminiReferences:
    def test_no_gemini_client_in_source_files(self):
        source_files = ["server.py", "repl_env.py", "llm_client.py"]
        for name in source_files:
            path = os.path.join(WORKTREE, name)
            if os.path.exists(path):
                content = open(path).read()
                assert "gemini_client" not in content, (
                    f"{name} still references gemini_client"
                )


# ---------------------------------------------------------------------------
# [FUNC-3] Model name resolved fresh on each call
# ---------------------------------------------------------------------------


class TestResolveModel:
    def test_returns_fresh_model_each_call(self):
        from llm_client import _resolve_model

        call_count = 0

        def fake_list():
            nonlocal call_count
            call_count += 1
            mock_model = MagicMock()
            mock_model.id = f"model-{call_count}"
            result = MagicMock()
            result.data = [mock_model]
            return result

        mock_client = MagicMock()
        mock_client.models.list = fake_list

        with patch("llm_client.get_client", return_value=mock_client):
            first = _resolve_model()
            second = _resolve_model()

        assert first == "model-1"
        assert second == "model-2"
        assert first != second


# ---------------------------------------------------------------------------
# [EDGE-1] Server unreachable raises RuntimeError
# ---------------------------------------------------------------------------


class TestResolveModelError:
    def test_connection_refused_raises_runtime_error(self):
        from llm_client import _resolve_model

        mock_client = MagicMock()
        mock_client.models.list.side_effect = ConnectionError("Connection refused")

        with patch("llm_client.get_client", return_value=mock_client):
            with pytest.raises(RuntimeError, match="Failed to query model"):
                _resolve_model()

    def test_empty_models_raises_runtime_error(self):
        from llm_client import _resolve_model

        mock_client = MagicMock()
        result = MagicMock()
        result.data = []
        mock_client.models.list.return_value = result

        with patch("llm_client.get_client", return_value=mock_client):
            with pytest.raises(RuntimeError, match="no models"):
                _resolve_model()


# ---------------------------------------------------------------------------
# [FUNC-4] Batch timeout scales with prompt count
# [EDGE-2] Batch of 1 prompt uses minimum timeout
# ---------------------------------------------------------------------------


class TestBatchTimeout:
    @pytest.mark.parametrize("n,expected_factor", [
        (1, 1),
        (4, 1),
        (8, 2),
        (18, 4),
    ])
    def test_timeout_scales_with_batch_size(self, n, expected_factor):
        from llm_client import CALL_TIMEOUT

        prompts = [f"prompt {i}" for i in range(n)]
        expected_timeout = CALL_TIMEOUT * max(1, n // 4)
        assert expected_timeout == CALL_TIMEOUT * expected_factor

        captured_timeout = {}

        original_as_completed = __import__(
            "concurrent.futures", fromlist=["as_completed"]
        ).as_completed

        def mock_as_completed(fs, timeout=None):
            captured_timeout["value"] = timeout
            return iter([])  # no futures to iterate

        with patch("llm_client.as_completed", side_effect=mock_as_completed):
            with patch("llm_client.llm_completion", return_value="ok"):
                from llm_client import llm_completion_batch
                llm_completion_batch(prompts, model="test-model")

        assert captured_timeout["value"] == expected_timeout

    def test_single_prompt_minimum_timeout(self):
        """EDGE-2: single prompt uses CALL_TIMEOUT * 1, not zero."""
        from llm_client import CALL_TIMEOUT

        captured_timeout = {}

        def mock_as_completed(fs, timeout=None):
            captured_timeout["value"] = timeout
            return iter([])

        with patch("llm_client.as_completed", side_effect=mock_as_completed):
            with patch("llm_client.llm_completion", return_value="ok"):
                from llm_client import llm_completion_batch
                llm_completion_batch(["single"], model="test-model")

        assert captured_timeout["value"] == CALL_TIMEOUT


# ---------------------------------------------------------------------------
# [FUNC-5] No duplicate shutil imports in server.py
# ---------------------------------------------------------------------------


class TestNoDuplicateShutil:
    def test_shutil_imported_once_at_module_level(self):
        server_path = os.path.join(WORKTREE, "server.py")
        with open(server_path) as f:
            tree = ast.parse(f.read())

        # Check no import shutil inside any function
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for child in ast.walk(node):
                    if isinstance(child, ast.Import):
                        for alias in child.names:
                            assert alias.name != "shutil", (
                                f"Duplicate 'import shutil' inside {node.name}()"
                            )

    def test_shutil_at_module_level(self):
        server_path = os.path.join(WORKTREE, "server.py")
        with open(server_path) as f:
            tree = ast.parse(f.read())

        # Check shutil is imported at module level
        module_imports = [
            alias.name
            for node in tree.body
            if isinstance(node, ast.Import)
            for alias in node.names
        ]
        assert "shutil" in module_imports


# ---------------------------------------------------------------------------
# [FUNC-6] Sessions dict protected by a lock
# ---------------------------------------------------------------------------


class TestSessionThreadSafety:
    def test_concurrent_session_creation(self):
        sys.path.insert(0, WORKTREE)

        # We need to test that _sessions_lock exists and is used
        import server

        assert hasattr(server, "_sessions_lock")
        assert isinstance(server._sessions_lock, type(threading.Lock()))

        # Simulate concurrent rlm_init calls using the session management
        # directly (without actually starting mlx server)
        errors = []
        session_ids = []
        lock = threading.Lock()

        def create_session(i):
            try:
                sid = server._generate_session_id()
                env = MagicMock()
                with server._sessions_lock:
                    server._sessions[sid] = {
                        "env": env,
                        "last_used": time.time(),
                        "temp_dir": f"/tmp/fake_{i}",
                    }
                with lock:
                    session_ids.append(sid)
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=create_session, args=(i,))
                   for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during concurrent creation: {errors}"
        assert len(session_ids) == 20
        assert len(set(session_ids)) == 20  # all distinct

        # Verify all sessions exist
        with server._sessions_lock:
            for sid in session_ids:
                assert sid in server._sessions

            # Clean up
            for sid in session_ids:
                del server._sessions[sid]

    def test_get_session_acquires_lock(self):
        """[M1] Verify _get_session() uses _sessions_lock via AST inspection."""
        server_path = os.path.join(WORKTREE, "server.py")
        with open(server_path) as f:
            tree = ast.parse(f.read())

        # Find _get_session function
        get_session_func = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_get_session":
                get_session_func = node
                break

        assert get_session_func is not None, "_get_session() not found in server.py"

        # Check that the function body contains a `with _sessions_lock:` statement
        found_lock = False
        for node in ast.walk(get_session_func):
            if isinstance(node, ast.With):
                for item in node.items:
                    ctx = item.context_expr
                    if isinstance(ctx, ast.Name) and ctx.id == "_sessions_lock":
                        found_lock = True
                    elif isinstance(ctx, ast.Attribute) and ctx.attr == "_sessions_lock":
                        found_lock = True

        assert found_lock, (
            "_get_session() does not use 'with _sessions_lock:' -- "
            "lock protection is missing"
        )


# ---------------------------------------------------------------------------
# [FUNC-7] Startup progress printed to stderr
# ---------------------------------------------------------------------------


class TestStartupProgress:
    def test_progress_messages_on_slow_start(self):
        import server

        poll_count = 0

        def mock_running():
            nonlocal poll_count
            poll_count += 1
            # Fail first 2 polls, succeed on 3rd
            return poll_count >= 3

        mock_popen = MagicMock()

        with patch.object(server, "_mlx_server_running", side_effect=mock_running):
            with patch.object(server, "_mlx_process", None):
                with patch("server.subprocess.Popen", return_value=mock_popen):
                    with patch("server.time.sleep"):
                        # Capture stderr
                        old_stderr = sys.stderr
                        from io import StringIO
                        captured = StringIO()
                        sys.stderr = captured
                        try:
                            # Reset so _ensure_mlx_server tries to start
                            server._mlx_process = None
                            server._ensure_mlx_server()
                        finally:
                            sys.stderr = old_stderr

        output = captured.getvalue()
        assert "Waiting for mlx-lm server to start" in output
        assert f"ready on port {server.MLX_PORT}" in output


# ---------------------------------------------------------------------------
# [QUAL-2] pytest added as dev dependency
# ---------------------------------------------------------------------------


class TestDevDependency:
    def test_pytest_in_pyproject(self):
        pyproject_path = os.path.join(WORKTREE, "pyproject.toml")
        with open(pyproject_path) as f:
            content = f.read()
        assert "pytest" in content
