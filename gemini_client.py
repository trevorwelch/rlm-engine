"""
Local LLM client for RLM sub-LLM calls via mlx-lm (OpenAI-compatible API).

Provides llm_completion() and llm_completion_batch() for the REPL environment.
"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai

_client: openai.OpenAI | None = None
_client_lock = threading.Lock()
_model_name: str | None = None

CALL_TIMEOUT = 120


def get_client() -> openai.OpenAI:
    """Thread-safe lazy client initialization."""
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is None:
            base_url = os.getenv("RLM_BASE_URL", "http://localhost:8080/v1")
            api_key = os.getenv("RLM_API_KEY", "local")
            _client = openai.OpenAI(base_url=base_url, api_key=api_key)
    return _client


def _resolve_model() -> str:
    """Auto-detect the model name from the server's /v1/models endpoint."""
    global _model_name
    if _model_name is not None:
        return _model_name
    try:
        client = get_client()
        models = client.models.list()
        if models.data:
            _model_name = models.data[0].id
            return _model_name
    except Exception:
        pass
    _model_name = "default_model"
    return _model_name


def llm_completion(prompt: str, model: str | None = None) -> str:
    """Single LLM completion. Returns the response text."""
    client = get_client()
    model = model or _resolve_model()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        return f"[LLM_ERROR] {e}"
    choice = response.choices[0] if response.choices else None
    if choice is None or choice.message.content is None:
        return "[LLM_ERROR] Empty response from local model"
    return choice.message.content


def llm_completion_batch(
    prompts: list[str],
    model: str | None = None,
    max_workers: int = 8,
) -> list[str]:
    """
    Parallel LLM completions via ThreadPoolExecutor.

    Returns results in the same order as the input prompts.
    Failed calls return the error message as a string.
    """
    results: list[str | None] = [None] * len(prompts)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(llm_completion, prompt, model): i
            for i, prompt in enumerate(prompts)
        }
        for future in as_completed(future_to_idx, timeout=CALL_TIMEOUT):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result(timeout=CALL_TIMEOUT)
            except TimeoutError:
                results[idx] = "[LLM_ERROR] LLM call timed out"
            except Exception as e:
                results[idx] = f"[LLM_ERROR] {e}"

    return [r if r is not None else "[LLM_ERROR] No result" for r in results]
